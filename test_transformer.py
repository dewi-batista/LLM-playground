from itertools import pairwise
from pathlib import Path
from tqdm import tqdm

import json
import pickle
import re
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"


def iter_pre_tokens(sequence: str):
    sequence = sequence.strip()
    first = True
    for match in re.finditer(r"\S+", sequence):
        token = match.group(0)
        if first:
            yield token
            first = False
        else:
            yield " " + token


def make_bpe_encoder(encodings):
    merges = {tuple(pair): new_token for pair, new_token in encodings}
    ranks = {tuple(pair): i for i, (pair, _) in enumerate(encodings)}
    cache = {}

    def encode(text: str):
        if text in cache:
            return cache[text]
        ids = list(text.encode("utf-8"))
        while True:
            best_pair = None
            best_rank = None
            for p in pairwise(ids):
                r = ranks.get(p)
                if r is None:
                    continue
                if best_rank is None or r < best_rank:
                    best_rank = r
                    best_pair = p
            if best_pair is None:
                break
            new_token = merges[best_pair]
            merged = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
                    merged.append(new_token)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged
        cache[text] = ids
        return ids

    return encode


def positional_encoding(seq_len, d_model, device):
    positions = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)  # (n, 1)

    i = torch.arange(d_model, device=device, dtype=torch.float32)  # (d,)
    div_term = torch.pow(10_000.0, (2 * (i // 2)) / d_model)

    angles = positions / div_term  # (n, d)

    pe = torch.zeros_like(angles)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])

    return pe


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_QKV = nn.Linear(d_model, 3 * d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, X):
        B, T, _ = X.shape
        Q, K, V = self.W_QKV(X).chunk(3, dim=-1)
        Q = Q.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = K.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = V.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)

        M = torch.triu(X.new_full((T, T), float("-inf")), diagonal=1)
        A = torch.softmax((Q @ K.transpose(-2, -1)) / (self.d_head**0.5) + M, dim=-1)
        O = A @ V
        O = O.transpose(1, 2).reshape(B, T, self.num_heads * self.d_head)
        O = self.dropout_attn(self.W_O(O))
        H_1 = self.ln1(X + O)

        H_2 = self.dropout_ffn(self.W_2(self.act(self.W_1(H_1))))
        H_3 = self.ln2(H_1 + H_2)
        return H_3


def token_to_cli(token: str) -> str:
    if token.startswith(" "):
        n = len(token) - len(token.lstrip(" "))
        return ("_" * n) + token[n:]
    return token


def encode_text_to_indices(text, bpe_encode, token_id_to_index):
    token_ids = []
    for pre_tok in iter_pre_tokens(text):
        token_ids.extend(bpe_encode(pre_tok))
    token_ids = torch.as_tensor(token_ids, dtype=torch.long)
    idx = token_id_to_index[token_ids]
    idx = idx[idx >= 0]
    return idx.tolist()


@torch.no_grad()
def topk_next_tokens(prompt_indeces, E, model, final_lay_norm, U, pe, index_to_token, topk: int):
    x = torch.as_tensor(prompt_indeces, dtype=torch.long, device=E.weight.device).unsqueeze(0)  # (1, T)
    X = E(x) + pe[: x.shape[1]]
    logits = U(final_lay_norm(model(X)))[0, -1]  # (V,)
    probs = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, k=topk)
    return [(token_to_cli(index_to_token[int(i)]), float(v)) for v, i in zip(values, indices)]


def main():
    if len(sys.argv) >= 2 and sys.argv[1] in {"-h", "--help"}:
        print(f"usage: python {Path(__file__).name} <language> <vocab_timestamp> <model_number> [prompt...]")
        print(f"   or: python {Path(__file__).name} <language> <vocab_timestamp> <model_number> --gen <n> [prompt...]")
        raise SystemExit(0)

    if len(sys.argv) < 4:
        print(f"usage: python {Path(__file__).name} <language> <vocab_timestamp> <model_number> [prompt...]")
        raise SystemExit(1)

    language = sys.argv[1]
    timestamp = sys.argv[2]
    model_number = sys.argv[3]
    if not model_number.isdigit():
        print(f"usage: python {Path(__file__).name} <language> <vocab_timestamp> <model_number> [prompt...]")
        raise SystemExit(1)
    model_number = int(model_number)

    gen_n = 0
    prompt_args = sys.argv[4:]
    if len(prompt_args) >= 2 and prompt_args[0] == "--gen":
        gen_n = int(prompt_args[1])
        prompt_args = prompt_args[2:]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    run_dir = MODELS_DIR / language / timestamp
    checkpoint_path = run_dir / f"{language}_transformer_{timestamp}_{model_number}.ckpt"
    if not checkpoint_path.exists():
        print(f"checkpoint not found: {checkpoint_path}")
        raise SystemExit(1)

    tqdm.write(f"device: {device} (cuda={torch.cuda.is_available()}, cuda_devices={torch.cuda.device_count()})")
    if torch.cuda.is_available():
        tqdm.write(f"cuda[0]: {torch.cuda.get_device_name(0)}")
    tqdm.write(f"checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    vocab_path = Path(ckpt.get("bpe_vocab_path", run_dir / f"{language}_{timestamp}.json"))
    if not vocab_path.is_absolute():
        vocab_path = HERE / vocab_path
    encodings_path = Path(ckpt.get("bpe_encodings_path", run_dir / f"{language}_{timestamp}.pkl"))
    if not encodings_path.is_absolute():
        encodings_path = HERE / encodings_path

    with open(vocab_path) as f:
        vocab = json.load(f)
    with open(encodings_path, "rb") as f:
        encodings = pickle.load(f)
    bpe_encode = make_bpe_encoder(encodings)

    index_to_token = ckpt["index_to_token"]
    token_str_to_index = {t: i for i, t in enumerate(index_to_token)}
    token_id_to_index = torch.full((len(vocab),), -1, dtype=torch.long)
    for token_id in range(len(vocab)):
        idx = token_str_to_index.get(vocab[str(token_id)]["string"])
        if idx is not None:
            token_id_to_index[token_id] = idx

    token_to_index_cli = {token_to_cli(t): i for i, t in enumerate(index_to_token)}
    V = len(index_to_token)
    d_model = int(ckpt["d_model"])
    num_heads = int(ckpt["num_heads"])
    num_blocks = int(ckpt["num_blocks"])
    d_ff = int(ckpt["d_ff"])
    dropout = float(ckpt["dropout"])
    seq_len = int(ckpt["seq_len"])

    mapped = int((token_id_to_index >= 0).sum())
    if mapped != V:
        tqdm.write(f"WARNING: vocab mapping mismatch (mapped={mapped}, V={V})")

    tqdm.write(f"arch: V={V}, d_model={d_model}, blocks={num_blocks}, heads={num_heads}, seq_len={seq_len}")

    E = nn.Embedding(V, d_model).to(device)
    final_lay_norm = nn.LayerNorm(d_model).to(device)
    model = nn.Sequential(*[TransformerBlock(d_model, d_ff, num_heads, dropout) for _ in range(num_blocks)]).to(
        device
    )
    U = nn.Linear(d_model, V, bias=False).to(device)
    U.weight = E.weight

    E.load_state_dict(ckpt["E_state_dict"])
    model.load_state_dict(ckpt["model_state_dict"])
    final_lay_norm.load_state_dict(ckpt["final_lay_norm_state_dict"])

    E.eval()
    model.eval()
    final_lay_norm.eval()
    U.eval()

    pe = positional_encoding(seq_len, d_model, device=device)

    def run_once(prompt: str):
        prompt_indeces = encode_text_to_indices(prompt, bpe_encode, token_id_to_index)
        if not prompt_indeces:
            print("<no usable tokens in prompt after pruning>")
            return

        if len(prompt_indeces) > seq_len:
            prompt_indeces = prompt_indeces[-seq_len:]

        topk = topk_next_tokens(
            prompt_indeces,
            E,
            model,
            final_lay_norm,
            U,
            pe,
            index_to_token,
            topk=10,
        )
        print("top10:", topk)

        if gen_n > 0:
            indeces = list(prompt_indeces)
            for _ in range(gen_n):
                top1 = topk_next_tokens(
                    indeces[-seq_len:],
                    E,
                    model,
                    final_lay_norm,
                    U,
                    pe,
                    index_to_token,
                    topk=1,
                )
                next_token_cli, _ = top1[0]
                next_idx = token_to_index_cli.get(next_token_cli)
                if next_idx is None:
                    break
                indeces.append(next_idx)
            text_out = "".join(index_to_token[i] for i in indeces)
            print("\n---\n" + text_out + "\n---")

    if prompt_args:
        run_once(" ".join(prompt_args))
        raise SystemExit(0)

    while True:
        try:
            prompt = input("> ").strip()
        except EOFError:
            break
        if not prompt:
            break
        run_once(prompt)


if __name__ == "__main__":
    main()

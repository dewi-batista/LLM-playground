from itertools import pairwise
from pathlib import Path
from tqdm import tqdm

import json
import math
import pickle
import re
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"

BENCH_SENTENCES = [
    "The capital of France is Paris",
    "Paris is the capital of France",
    "Rome is the capital of Italy",
    "The capital of Italy is Rome",
    "The quick brown fox jumps over the lazy dog",
    "The largest planet in the solar system is Jupiter",
    "The author of Hamlet is William Shakespeare",
    "Machine learning is a field of artificial intelligence",
    "New York is a city in the United States",
    "London is a city in the United Kingdom",
    "London is a city in England",
    "My favourite basketballer is Michael Jordan",
    "I like to drink coffee in the morning",
]
BENCH_NEXT_TOKENS = 5


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
        H = self.ln1(X)
        Q, K, V = self.W_QKV(H).chunk(3, dim=-1)
        Q = Q.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = K.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = V.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)

        O = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        O = O.transpose(1, 2).reshape(B, T, self.num_heads * self.d_head)
        X = X + self.dropout_attn(self.W_O(O))

        H = self.ln2(X)
        X = X + self.dropout_ffn(self.W_2(self.act(self.W_1(H))))
        return X


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

def encode_pre_tokens_to_indices(pre_tokens, bpe_encode, token_id_to_index):
    token_ids = []
    for tok in pre_tokens:
        token_ids.extend(bpe_encode(tok))
    if not token_ids:
        return []
    token_ids = torch.as_tensor(token_ids, dtype=torch.long)
    idx = token_id_to_index[token_ids]
    idx = idx[idx >= 0]
    return idx.tolist()

PUNCT_SUFFIXES = [",", ".", ":", ";", "!", "?"]
PUNCT_PREFIXES = ["-"]

def target_variants(token: str) -> list[str]:
    base = token.rstrip("".join(PUNCT_SUFFIXES))
    variants = [token]
    if base and base != token:
        variants.append(base)
    for p in PUNCT_SUFFIXES:
        if base:
            variants.append(base + p)
    stripped = base.lstrip(" ")
    leading = base[: len(base) - len(stripped)]
    for p in PUNCT_PREFIXES:
        if stripped:
            variants.append(p + stripped)
            if leading:
                variants.append(leading + p + stripped)
            for s in PUNCT_SUFFIXES:
                variants.append(p + stripped + s)
                if leading:
                    variants.append(leading + p + stripped + s)
    seen = set()
    out = []
    for v in variants:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def choose_next_token(
    logits: torch.Tensor,
    prev_tokens: list[int],
    *,
    sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram: int,
) -> int:
    logits = logits.clone()

    if repetition_penalty and repetition_penalty != 1.0 and prev_tokens:
        t = torch.as_tensor(list(set(prev_tokens)), device=logits.device)
        v = logits[t]
        logits[t] = torch.where(v > 0, v / repetition_penalty, v * repetition_penalty)

    if no_repeat_ngram and no_repeat_ngram > 0 and len(prev_tokens) >= (no_repeat_ngram - 1):
        n = int(no_repeat_ngram)
        seen = {}
        for i in range(len(prev_tokens) - n + 1):
            prefix = tuple(prev_tokens[i : i + n - 1])
            nxt = prev_tokens[i + n - 1]
            if prefix in seen:
                seen[prefix].add(nxt)
            else:
                seen[prefix] = {nxt}
        banned = seen.get(tuple(prev_tokens[-(n - 1) :]), set())
        if banned:
            logits[list(banned)] = -float("inf")

    if not sample:
        return int(torch.argmax(logits).item())

    if temperature and temperature != 1.0:
        logits = logits / float(temperature)

    if top_k and top_k > 0:
        k = min(int(top_k), logits.numel())
        cutoff = torch.topk(logits, k=k).values[-1]
        logits = torch.where(logits < cutoff, torch.full_like(logits, -float("inf")), logits)

    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        remove = cum_probs > float(top_p)
        remove[0] = False  # keep at least 1
        sorted_logits[remove] = -float("inf")

        logits = torch.full_like(logits, -float("inf"))
        logits.scatter_(0, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def next_token_logits(prompt_indeces, E, model, final_lay_norm, U, pe):
    x = torch.as_tensor(prompt_indeces, dtype=torch.long, device=E.weight.device).unsqueeze(0)  # (1, T)
    X = E(x) + pe[: x.shape[1]]
    return U(final_lay_norm(model(X)))[0, -1]  # (V,)


@torch.no_grad()
def topk_next_tokens(prompt_indeces, E, model, final_lay_norm, U, pe, index_to_token, topk: int):
    logits = next_token_logits(prompt_indeces, E, model, final_lay_norm, U, pe)
    probs = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, k=topk)
    return [(token_to_cli(index_to_token[int(i)]), round(float(v), 2)) for v, i in zip(values, indices)]


def main():
    if len(sys.argv) >= 2 and sys.argv[1] in {"-h", "--help"}:
        print(f"usage: python {Path(__file__).name} <language> <vocab_timestamp> <model_number> [prompt...]")
        print(f"   or: python {Path(__file__).name} <language> <vocab_timestamp> <model_number> --gen <n> [prompt...]")
        print(f"   or: python {Path(__file__).name} <language> <vocab_timestamp> <model_number> --bench")
        print("\ndecoding flags (optional): --sample --temp 1.0 --top_k 0 --top_p 1.0 --rep_penalty 1.1 --no_repeat_ngram 3")
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

    # NOTE: no argparse on purpose (trying to keep it minimal + readable)
    gen_n = 0
    bench = False
    sample = False
    temperature = 0.5
    top_k = 0
    top_p = 1.0
    repetition_penalty = 1.1
    no_repeat_ngram = 3
    prompt_args = []
    args = sys.argv[4:]
    i = 0
    while i < len(args):
        if args[i] == "--gen":
            gen_n = int(args[i + 1])
            i += 2
            continue
        if args[i] == "--bench":
            bench = True
            i += 1
            continue
        if args[i] == "--sample":
            sample = True
            i += 1
            continue
        if args[i] == "--temp":
            temperature = float(args[i + 1])
            i += 2
            continue
        if args[i] == "--top_k":
            top_k = int(args[i + 1])
            i += 2
            continue
        if args[i] == "--top_p":
            top_p = float(args[i + 1])
            i += 2
            continue
        if args[i] == "--rep_penalty":
            repetition_penalty = float(args[i + 1])
            i += 2
            continue
        if args[i] == "--no_repeat_ngram":
            no_repeat_ngram = int(args[i + 1])
            i += 2
            continue
        prompt_args.append(args[i])
        i += 1

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    run_dir = MODELS_DIR / language / timestamp
    checkpoint_path = run_dir / f"training_run_{model_number}" / "weights.ckpt"
    if not checkpoint_path.exists():
        print(f"checkpoint not found: {checkpoint_path}")
        raise SystemExit(1)

    tqdm.write(f"device: {device} (cuda={torch.cuda.is_available()}, cuda_devices={torch.cuda.device_count()})")
    if torch.cuda.is_available():
        tqdm.write(f"cuda[0]: {torch.cuda.get_device_name(0)}")
    tqdm.write(f"checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    global_step = ckpt.get("global_step")
    total_steps = ckpt.get("total_steps")
    tokens_per_step = ckpt.get("tokens_per_step")
    train_tokens = ckpt.get("train_tokens")
    val_ppl = ckpt.get("val_ppl")
    best_val_ppl = ckpt.get("best_val_ppl")

    if global_step is not None:
        if total_steps is not None:
            tqdm.write(f"trained: global_step={int(global_step):_}/{int(total_steps):_}")
        else:
            tqdm.write(f"trained: global_step={int(global_step):_}")
    if global_step is not None and tokens_per_step is not None:
        seen_tokens = int(float(global_step) * float(tokens_per_step))
        if train_tokens is not None:
            tqdm.write(f"tokens : seen~{seen_tokens:_} / budget={int(float(train_tokens)):_}")
        else:
            tqdm.write(f"tokens : seen~{seen_tokens:_}")
    if val_ppl is not None:
        tqdm.write(f"val   : ppl={float(val_ppl):.2f} (nll={math.log(float(val_ppl)):.4f})")
    if best_val_ppl is not None:
        tqdm.write(f"best  : ppl={float(best_val_ppl):.2f} (nll={math.log(float(best_val_ppl)):.4f})")

    vocab_path = Path(ckpt.get("bpe_vocab_path", run_dir / "vocabulary.json"))
    if not vocab_path.is_absolute():
        vocab_path = HERE / vocab_path
    encodings_path = Path(ckpt.get("bpe_encodings_path", run_dir / "merges.pkl"))
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
                logits = next_token_logits(indeces[-seq_len:], E, model, final_lay_norm, U, pe)
                next_idx = choose_next_token(
                    logits,
                    indeces[-seq_len:],
                    sample=sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram=no_repeat_ngram,
                )
                indeces.append(next_idx)
            text_out = "".join(index_to_token[i] for i in indeces)
            print("\n---\n" + text_out + "\n---")

    def eval_holdout(prompt: str):
        pre_tokens = list(iter_pre_tokens(prompt))
        if len(pre_tokens) < 2:
            run_once(prompt)
            return

        context_tokens = pre_tokens[:-1]
        target_token = pre_tokens[-1]
        context_text = "".join(context_tokens)

        context_indeces = encode_pre_tokens_to_indices(context_tokens, bpe_encode, token_id_to_index)
        if not context_indeces:
            print("<no usable context tokens after pruning>")
            return
        if len(context_indeces) > seq_len:
            context_indeces = context_indeces[-seq_len:]

        logits = next_token_logits(context_indeces, E, model, final_lay_norm, U, pe)
        probs = torch.softmax(logits, dim=-1)
        values, indices = torch.topk(probs, k=10)
        top10 = [token_to_cli(index_to_token[int(i)]) for v, i in zip(values, indices)]

        target_idx = token_str_to_index.get(target_token)
        target_pieces = None
        target_note = None
        if target_idx is None:
            target_pieces = encode_pre_tokens_to_indices([target_token], bpe_encode, token_id_to_index)
            if not target_pieces:
                print(f"{context_text}")
                print(f"Target : {token_to_cli(target_token)} (<not in vocab after pruning>)")
                print("top10:", top10)
                return
            target_idx = int(target_pieces[0])
            if len(target_pieces) > 1:
                target_note = " (target is multi-token; rank shown for first token)"

        candidates = []
        for tok in target_variants(target_token):
            idx = token_str_to_index.get(tok)
            if idx is None:
                pieces = encode_pre_tokens_to_indices([tok], bpe_encode, token_id_to_index)
                if not pieces:
                    continue
                idx = int(pieces[0])
            candidates.append((tok, int(idx)))

        best_tok, best_idx = (target_token, int(target_idx))
        best_rank = None
        best_logit = None
        for tok, idx in candidates or [(target_token, int(target_idx))]:
            t_logit = logits[idx]
            r = int((logits > t_logit).sum().item()) + 1
            if best_rank is None or r < best_rank:
                best_rank = r
                best_tok, best_idx = tok, idx
                best_logit = t_logit

        rank = int(best_rank)
        target_prob = float(torch.exp(best_logit - torch.logsumexp(logits, dim=-1)).item())
        # target_ppl = float(torch.exp(torch.logsumexp(logits, dim=-1) - target_logit).item())

        print(f"\n{context_text} [{token_to_cli(target_token)}, {rank}]")
        if target_pieces is not None and len(target_pieces) > 1:
            pieces_cli = [token_to_cli(index_to_token[int(i)]) for i in target_pieces]
            print(f"Target tokens: {pieces_cli}")
        print(top10)

    def eval_bench(prompt: str, next_tokens: int = BENCH_NEXT_TOKENS):
        pre_tokens = list(iter_pre_tokens(prompt))
        if len(pre_tokens) < 2:
            run_once(prompt)
            return

        context_tokens = pre_tokens[:-1]
        target_token = pre_tokens[-1]
        context_text = "".join(context_tokens)

        context_indeces = encode_pre_tokens_to_indices(context_tokens, bpe_encode, token_id_to_index)
        if not context_indeces:
            print("<no usable context tokens after pruning>")
            return
        if len(context_indeces) > seq_len:
            context_indeces = context_indeces[-seq_len:]

        logits0 = next_token_logits(context_indeces, E, model, final_lay_norm, U, pe)
        probs0 = torch.softmax(logits0, dim=-1)
        values0, indices0 = torch.topk(probs0, k=10)
        top10 = [(token_to_cli(index_to_token[int(i)]), round(float(v), 2)) for v, i in zip(values0, indices0)]

        candidates = []
        for tok in target_variants(target_token):
            idx = token_str_to_index.get(tok)
            if idx is None:
                pieces = encode_pre_tokens_to_indices([tok], bpe_encode, token_id_to_index)
                if not pieces:
                    continue
                idx = int(pieces[0])
            candidates.append((tok, idx))

        best_rank = None
        best_tok = target_token
        for tok, idx in candidates:
            t_logit = logits0[idx]
            r = int((logits0 > t_logit).sum().item()) + 1
            if best_rank is None or r < best_rank:
                best_rank = r
                best_tok = tok

        indeces = list(context_indeces)
        generated = []
        for _ in range(next_tokens):
            logits = next_token_logits(indeces[-seq_len:], E, model, final_lay_norm, U, pe)
            next_idx = choose_next_token(
                logits,
                indeces[-seq_len:],
                sample=sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram=no_repeat_ngram,
            )
            indeces.append(next_idx)
            generated.append(token_to_cli(index_to_token[next_idx]))

        if best_rank is None:
            rank_part = "<not in vocab after pruning>"
        else:
            rank_part = str(int(best_rank))
        header = f"\n{context_text} [{token_to_cli(target_token)}, {rank_part}]"
        print(header)
        print(top10)
        print(generated)

    if bench:
        if gen_n > 0:
            print("ERROR: --bench and --gen are mutually exclusive")
            raise SystemExit(1)
        for i, s in enumerate(BENCH_SENTENCES):
            # print(f"\n=== {i + 1}/{len(BENCH_SENTENCES)} ===")
            eval_bench(s)
        print()
        raise SystemExit(0)

    if prompt_args:
        prompt = " ".join(prompt_args)
        if gen_n > 0:
            run_once(prompt)
            raise SystemExit(0)

        eval_holdout(prompt)
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

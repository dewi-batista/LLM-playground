from pathlib import Path

import json
import math
import pickle
import sys

import torch
import torch.nn as nn

from tfs_utils.inference_scripts import (
    TransformerBlock,
    build_token_id_to_index,
    encode_pre_tokens_to_indices,
    iter_pre_tokens,
    make_bpe_encoder,
    next_token_logits,
    positional_encoding,
    sample_next_token,
    target_variants,
    token_to_cli,
)

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
    "The patron saint of Wales was Dewi",
    "The patron saint of Wales was Dewi Sant",
]
NEXT_TOKENS = 5

# decoding knobs
SAMPLE = True
TEMPERATURE = 0.7
REPETITION_PENALTY = 1.1
NO_REPEAT_NGRAM = 3

def main():
    language = sys.argv[1]
    timestamp = sys.argv[2]
    model_number = int(sys.argv[3])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    run_dir = MODELS_DIR / language / timestamp
    checkpoint_path = run_dir / f"training_run_{model_number}" / "weights.ckpt"

    print(f"device: {device} (cuda={torch.cuda.is_available()}, cuda_devices={torch.cuda.device_count()})")
    if torch.cuda.is_available():
        print(f"cuda[0]: {torch.cuda.get_device_name(0)}")
    print(f"checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    global_step = ckpt["global_step"]
    total_steps = ckpt["total_steps"]
    tokens_per_step = ckpt["tokens_per_step"]
    train_tokens = ckpt["train_tokens"]
    val_ppl = ckpt["val_ppl"]
    best_val_ppl = ckpt["best_val_ppl"]

    print(f"trained: global_step={int(global_step):_}/{int(total_steps):_}")
    seen_tokens = int(float(global_step) * float(tokens_per_step))
    print(f"tokens : seen~{seen_tokens:_} / budget={int(float(train_tokens)):_}")
    print(f"val   : ppl={float(val_ppl):.2f} (nll={math.log(float(val_ppl)):.4f})")
    print(f"best  : ppl={float(best_val_ppl):.2f} (nll={math.log(float(best_val_ppl)):.4f})")

    vocab_path = Path(ckpt["bpe_vocab_path"])
    encodings_path = Path(ckpt["bpe_encodings_path"])

    with open(vocab_path) as f:
        vocab = json.load(f)
    with open(encodings_path, "rb") as f:
        encodings = pickle.load(f)
    bpe_encode = make_bpe_encoder(encodings)

    index_to_token = ckpt["index_to_token"]
    token_id_to_index, token_str_to_index = build_token_id_to_index(vocab, index_to_token)

    V = len(index_to_token)
    d_model = int(ckpt["d_model"])
    num_heads = int(ckpt["num_heads"])
    num_blocks = int(ckpt["num_blocks"])
    d_ff = int(ckpt["d_ff"])
    dropout = float(ckpt["dropout"])
    seq_len = int(ckpt["seq_len"])
    print(f"arch: V={V}, d_model={d_model}, blocks={num_blocks}, heads={num_heads}, seq_len={seq_len}")

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

    for sentence in BENCH_SENTENCES:
        pre_tokens = list(iter_pre_tokens(sentence))
        context_tokens = pre_tokens[:-1]
        target_token = pre_tokens[-1]
        context_text = "".join(context_tokens)

        context_indeces = encode_pre_tokens_to_indices(context_tokens, bpe_encode, token_id_to_index)
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
            candidates.append(int(idx))

        best_rank = None
        for idx in candidates:
            t_logit = logits0[idx]
            r = int((logits0 > t_logit).sum().item()) + 1
            if best_rank is None or r < best_rank:
                best_rank = r

        indeces = list(context_indeces)
        generated = []
        for _ in range(NEXT_TOKENS):
            logits = next_token_logits(indeces[-seq_len:], E, model, final_lay_norm, U, pe)
            next_idx = sample_next_token(
                logits,
                indeces[-seq_len:],
                sample=SAMPLE,
                temperature=TEMPERATURE,
                repetition_penalty=REPETITION_PENALTY,
                no_repeat_ngram=NO_REPEAT_NGRAM,
            )
            indeces.append(next_idx)
            generated.append(token_to_cli(index_to_token[next_idx]))

        rank_part = str(int(best_rank)) if best_rank is not None else "<not in vocab after pruning>"
        print(f"\n{context_text} [{token_to_cli(target_token)}, {rank_part}]")
        print(top10)
        print(generated)


if __name__ == "__main__":
    main()

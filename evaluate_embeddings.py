from pathlib import Path

import json
import sys
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent

VOCAB_PATH = HERE / "./data/vocabulary.json"
MODELS_DIR = HERE / "models"  # produced by train.py
WHICH = "E"  # "E", "U", or "sum"
TOPK = 10

def latest_checkpoint(models_dir: Path) -> Path:
    candidates = [p for p in models_dir.glob("*.ckpt") if p.is_file() and len(p.stem) == 14 and p.stem.isdigit()]
    if not candidates:
        raise FileNotFoundError(f"no timestamped checkpoints found in {models_dir} (expected YYYYMMDDHHMMSS.ckpt)")
    return max(candidates, key=lambda p: p.stem)

def resolve_checkpoint_arg(arg: str) -> Path:
    path = Path(arg)
    if path.is_absolute() or path.exists():
        return path
    alt = HERE / path
    if alt.exists():
        return alt
    return path

def load_vocab(vocab_path: Path):
    vocab = json.loads(vocab_path.read_text())
    V = len(vocab)

    token_to_index = {}
    index_to_token = [None] * V
    for token, info in vocab.items():
        idx = int(info["index"])
        token_to_index[token] = idx
        index_to_token[idx] = token

    return token_to_index, index_to_token

def load_embeddings(checkpoint_path: Path, which: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    E = ckpt["E_state_dict"]["weight"]
    U = ckpt["U_state_dict"]["weight"]

    if which == "E":
        W = E
    elif which == "U":
        W = U
    elif which == "sum":
        W = E + U
    else:
        raise ValueError(f"which must be 'E', 'U', or 'sum' (got {which!r})")

    return F.normalize(W, dim=1)

def most_similar(word: str, W, token_to_index, index_to_token, topk: int):
    idx = token_to_index[word]
    sims = torch.mv(W, W[idx])
    sims[idx] = float("-inf")
    values, indices = torch.topk(sims, k=topk)
    return [(index_to_token[int(i)], float(v)) for v, i in zip(values, indices)]

def analogy(a: str, b: str, c: str, W, token_to_index, index_to_token, topk: int):
    vec = W[token_to_index[b]] - W[token_to_index[a]] + W[token_to_index[c]]
    vec = F.normalize(vec, dim=0)

    sims = torch.mv(W, vec)
    for w in (a, b, c):
        sims[token_to_index[w]] = float("-inf")
    values, indices = torch.topk(sims, k=topk)
    return [(index_to_token[int(i)], float(v)) for v, i in zip(values, indices)]

token_to_index, index_to_token = load_vocab(VOCAB_PATH)
if len(sys.argv) >= 2 and sys.argv[1] in {"-h", "--help"}:
    print(f"usage: python {Path(__file__).name} [checkpoint_path]")
    raise SystemExit(0)

checkpoint_path = (
    resolve_checkpoint_arg(sys.argv[1]) if len(sys.argv) >= 2 else latest_checkpoint(MODELS_DIR)
)
print(f"Using checkpoint: {checkpoint_path}")
W = load_embeddings(checkpoint_path, which=WHICH)

words = [
    "king",
    "queen",
    "man",
    "woman",
    "paris",
    "france",
    "rome",
    "italy",
    "london",
    "england",
]

print(f"Neighbors (which={WHICH}, topk={TOPK})")
for word in words:
    if word not in token_to_index:
        print(f"- {word}: <not in vocab>")
        continue
    nn = most_similar(word, W, token_to_index, index_to_token, topk=TOPK)
    nn_str = ", ".join(f"{w} ({s:.3f})" for w, s in nn)
    print(f"- {word}: {nn_str}")

print("\nAnalogies (b - a + c)")
analogies = [
    ("man", "king", "woman", "queen"),
    ("france", "paris", "italy", "rome"),
]
for a, b, c, expected in analogies:
    missing = [w for w in (a, b, c) if w not in token_to_index]
    if missing:
        print(f"- {b} - {a} + {c} = {expected}: <missing {missing}>")
        continue
    preds = analogy(a, b, c, W, token_to_index, index_to_token, topk=TOPK)
    pred_str = ", ".join(f"{w} ({s:.3f})" for w, s in preds)
    print(f"- {b} - {a} + {c} = {expected}: {pred_str}")

from pathlib import Path

import json
import sys
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent

VOCAB_PATH = HERE / "./data/vocabulary.json"
MODELS_DIR = HERE / "models"  # produced by train.py
WHICH = "E"  # "E", "U", or "sum"
TOPK = 5

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

def load_vocab_from_ckpt(ckpt, vocab_path: Path):
    if "index_to_token" in ckpt:
        index_to_token = ckpt["index_to_token"]
        token_to_index = {t: i for i, t in enumerate(index_to_token)}
        return token_to_index, index_to_token
    return load_vocab(vocab_path)

def load_embeddings(ckpt, which: str):
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


def most_similar_vec(vec, W, index_to_token, topk: int, exclude_indeces=()):
    sims = torch.mv(W, vec)
    for idx in exclude_indeces:
        sims[int(idx)] = float("-inf")
    values, indices = torch.topk(sims, k=topk)
    return [(index_to_token[int(i)], float(v)) for v, i in zip(values, indices)]


def parse_expression(tokens, token_to_index, W):
    # Example tokens: ["king", "-", "man", "+", "woman"]
    if len(tokens) == 1 and " " in tokens[0]:
        tokens = tokens[0].split()

    vec = torch.zeros_like(W[0])
    used = []
    sign = 1.0
    for t in tokens:
        if t == "+":
            sign = 1.0
            continue
        if t == "-":
            sign = -1.0
            continue
        if t not in token_to_index:
            raise KeyError(t)
        idx = token_to_index[t]
        vec = vec + (sign * W[idx])
        used.append(idx)
        sign = 1.0

    return F.normalize(vec, dim=0), used


if len(sys.argv) >= 2 and sys.argv[1] in {"-h", "--help"}:
    print(f"usage: python {Path(__file__).name} [checkpoint_path] word")
    print(f"   or: python {Path(__file__).name} [checkpoint_path] word1 word2")
    print(f"   or: python {Path(__file__).name} [checkpoint_path] word - word + word")
    raise SystemExit(0)

args = sys.argv[1:]
query_words = []

if args and (args[0].endswith(".ckpt") or Path(args[0]).is_file()):
    checkpoint_path = resolve_checkpoint_arg(args[0])
    query_words = args[1:]
else:
    checkpoint_path = latest_checkpoint(MODELS_DIR)
    query_words = args

print(f"Using checkpoint: {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
token_to_index, index_to_token = load_vocab_from_ckpt(ckpt, VOCAB_PATH)
W = load_embeddings(ckpt, which=WHICH)

if query_words:
    if any(t in {"+", "-"} for t in query_words) or (len(query_words) == 1 and any(ch in query_words[0] for ch in "+-")):
        vec, used = parse_expression(query_words, token_to_index, W)
        nn = most_similar_vec(vec, W, index_to_token, topk=TOPK, exclude_indeces=used)
        nn_str = ", ".join(f"{w} ({s:.3f})" for w, s in nn)
        print(f"Expr: {' '.join(query_words)}")
        print(nn_str)
        raise SystemExit(0)

    if len(query_words) == 1:
        word = query_words[0]
        if word not in token_to_index:
            print(f"{word}: <not in vocab>")
            raise SystemExit(0)
        nn = most_similar(word, W, token_to_index, index_to_token, topk=TOPK)
        nn_str = ", ".join(f"{w} ({s:.3f})" for w, s in nn)
        print(f"Neighbors (which={WHICH}, topk={TOPK})")
        print(f"{word}: {nn_str}")
        raise SystemExit(0)

    if len(query_words) == 2:
        w1, w2 = query_words
        if w1 not in token_to_index or w2 not in token_to_index:
            missing = [w for w in (w1, w2) if w not in token_to_index]
            print(f"<not in vocab>: {missing}")
            raise SystemExit(0)
        sim = float(torch.dot(W[token_to_index[w1]], W[token_to_index[w2]]).item())
        print(f"cosine({w1}, {w2}) = {sim:.4f}")
        raise SystemExit(0)

    print("Provide 1 word for neighbors, 2 words for cosine similarity, or use +/- for arithmetic.")
    raise SystemExit(0)

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

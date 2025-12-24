import json
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent

VOCAB_PATH = HERE / "vocabulary.json"
CHECKPOINT_PATH = HERE / "w2v.pt"  # produced by train.py
WHICH = "E"  # "E", "U", or "sum"
TOPK = 10


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
W = load_embeddings(CHECKPOINT_PATH, which=WHICH)

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


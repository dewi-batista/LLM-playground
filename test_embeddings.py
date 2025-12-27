from itertools import pairwise
from pathlib import Path

import pickle
import sys
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent

WHICH = "E"  # "E", "U", or "sum"
TOPK = 5

def resolve_checkpoint_arg(arg: str) -> Path:
    path = Path(arg)
    if path.is_absolute() or path.exists():
        return path
    alt = HERE / path
    if alt.exists():
        return alt
    return path

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

bpe_encode = None
token_bytes = None

def cli_to_token(text: str) -> str:
    if text.startswith("_"):
        n = len(text) - len(text.lstrip("_"))
        return (" " * n) + text[n:]
    return text

def token_to_cli(text: str) -> str:
    if text.startswith(" "):
        n = len(text) - len(text.lstrip(" "))
        return ("_" * n) + text[n:]
    return text

def load_bpe_encodings_from_ckpt(ckpt):
    enc_path = ckpt.get("bpe_encodings_path")
    if not enc_path:
        return None
    p = Path(enc_path)
    if not p.is_absolute():
        p = HERE / p
    with open(p, "rb") as f:
        return pickle.load(f)

def build_bpe_token_bytes(encodings):
    token_bytes = [bytes([i]) for i in range(256)] + [b""] * len(encodings)
    for pair, new_token in encodings:
        a, b = pair
        token_bytes[new_token] = token_bytes[a] + token_bytes[b]
    return token_bytes

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

def vector_for_text(text: str, W, token_to_index):
    if text in token_to_index:
        idx = token_to_index[text]
        return W[idx], [idx]

    if bpe_encode is None or token_bytes is None:
        raise KeyError(text)

    candidates = [text]
    if not text.startswith(" "):
        candidates.append(" " + text)

    best_vec = None
    best_used = None
    best_covered = -1
    for cand in candidates:
        ids = bpe_encode(cand)
        used = []
        vec = torch.zeros_like(W[0])
        covered = 0
        for token_id in ids:
            s = token_bytes[token_id].decode("utf-8", errors="backslashreplace")
            idx = token_to_index.get(s)
            if idx is None:
                continue
            vec = vec + W[idx]
            used.append(idx)
            covered += 1
        if covered > best_covered:
            best_covered = covered
            best_vec = vec
            best_used = used

    if best_covered <= 0:
        raise KeyError(text)

    return F.normalize(best_vec, dim=0), best_used


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
        subvec, subused = vector_for_text(cli_to_token(t), W, token_to_index)
        vec = vec + (sign * subvec)
        used.extend(subused)
        sign = 1.0

    return F.normalize(vec, dim=0), used


if len(sys.argv) >= 2 and sys.argv[1] in {"-h", "--help"}:
    print(f"usage: python {Path(__file__).name} <checkpoint_path> [word]")
    print(f"   or: python {Path(__file__).name} <checkpoint_path> word1 word2")
    print(f"   or: python {Path(__file__).name} <checkpoint_path> word - word + word")
    raise SystemExit(0)

if len(sys.argv) < 2:
    print(f"usage: python {Path(__file__).name} <checkpoint_path> [word]")
    raise SystemExit(1)

checkpoint_path = resolve_checkpoint_arg(sys.argv[1])
query_words = sys.argv[2:]

query_words = [cli_to_token(w) if w not in {"+", "-"} else w for w in query_words]

print(f"Using checkpoint: {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
index_to_token = ckpt["index_to_token"]
token_to_index = {t: i for i, t in enumerate(index_to_token)}
W = load_embeddings(ckpt, which=WHICH)

encodings = load_bpe_encodings_from_ckpt(ckpt)
if encodings is not None:
    token_bytes = build_bpe_token_bytes(encodings)
    bpe_encode = make_bpe_encoder(encodings)

use_leading_space = encodings is not None or any(t.startswith(" ") for t in token_to_index)

if query_words:
    if any(t in {"+", "-"} for t in query_words) or (len(query_words) == 1 and any(ch in query_words[0] for ch in "+-")):
        vec, used = parse_expression(query_words, token_to_index, W)
        nn = most_similar_vec(vec, W, index_to_token, topk=TOPK, exclude_indeces=used)
        nn_str = ", ".join(f"{token_to_cli(w)} ({s:.3f})" for w, s in nn)
        expr_str = " ".join(t if t in {"+", "-"} else token_to_cli(t) for t in query_words)
        print(f"Expr: {expr_str}")
        print(nn_str)
        raise SystemExit(0)

    if len(query_words) == 1:
        word = query_words[0]
        try:
            vec, used = vector_for_text(word, W, token_to_index)
        except KeyError:
            print(f"{token_to_cli(word)}: <not in vocab>")
            raise SystemExit(0)
        nn = most_similar_vec(vec, W, index_to_token, topk=TOPK, exclude_indeces=used)
        nn_str = ", ".join(f"{token_to_cli(w)} ({s:.3f})" for w, s in nn)
        print(f"Neighbors (which={WHICH}, topk={TOPK})")
        print(f"{token_to_cli(word)}: {nn_str}")
        raise SystemExit(0)

    if len(query_words) == 2:
        w1, w2 = query_words
        try:
            v1, _ = vector_for_text(w1, W, token_to_index)
            v2, _ = vector_for_text(w2, W, token_to_index)
        except KeyError as e:
            missing = e.args[0] if e.args else str(e)
            print(f"<not in vocab>: {[token_to_cli(missing)]}")
            raise SystemExit(0)
        sim = float(torch.dot(v1, v2).item())
        print(f"cosine({token_to_cli(w1)}, {token_to_cli(w2)}) = {sim:.4f}")
        raise SystemExit(0)

    print("Provide 1 word for neighbors, 2 words for cosine similarity, or use +/- for arithmetic.")
    raise SystemExit(0)

base_words = [
    "king", "queen", "man", "woman", "paris", "france", "rome", "italy",
    "london","england",
]
words = [" " + w for w in base_words] if use_leading_space else base_words

print(f"Neighbors (which={WHICH}, topk={TOPK})")
for word in words:
    try:
        vec, used = vector_for_text(word, W, token_to_index)
    except KeyError:
        print(f"- {token_to_cli(word)}: <not in vocab>")
        continue
    nn = most_similar_vec(vec, W, index_to_token, topk=TOPK, exclude_indeces=used)
    nn_str = ", ".join(f"{token_to_cli(w)} ({s:.3f})" for w, s in nn)
    print(f"- {token_to_cli(word)}: {nn_str}")

print("\nAnalogies (b - a + c)")
base_analogies = [
    ("man", "king", "woman", "queen"),
    ("france", "paris", "italy", "rome"),
]
analogies = [tuple((" " + w) for w in t) for t in base_analogies] if use_leading_space else base_analogies
for a, b, c, expected in analogies:
    try:
        vec, used = parse_expression([b, "-", a, "+", c], token_to_index, W)
    except KeyError:
        print(
            f"- {token_to_cli(b)} - {token_to_cli(a)} + {token_to_cli(c)} = {token_to_cli(expected)}: <missing>"
        )
        continue
    preds = most_similar_vec(vec, W, index_to_token, topk=TOPK, exclude_indeces=used)
    pred_str = ", ".join(f"{token_to_cli(w)} ({s:.3f})" for w, s in preds)
    print(f"- {token_to_cli(b)} - {token_to_cli(a)} + {token_to_cli(c)} = {token_to_cli(expected)}: {pred_str}")

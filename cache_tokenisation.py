from itertools import pairwise
from pathlib import Path
from tqdm import tqdm

import json
import numpy as np
import pickle
import re
import sys

HERE = Path(__file__).resolve().parent

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

def load_or_create_token_ids(language, timestamp, corpus_path=None, token_ids_path=None, encodings=None, vocab=None):
    run_dir = HERE / "models" / language / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = corpus_path or (HERE / "data" / f"{language}.txt")
    token_ids_path = token_ids_path or (run_dir / "token_ids.npy")

    if token_ids_path.exists():
        return np.load(token_ids_path, mmap_mode="r")

    if encodings is None:
        encodings_path = run_dir / "merges.pkl"
        with open(encodings_path, "rb") as f:
            encodings = pickle.load(f)

    if vocab is None:
        vocab_path = run_dir / "vocabulary.json"
        with open(vocab_path) as f:
            vocab = json.load(f)

    merges = {tuple(pair): new_token for pair, new_token in encodings}
    ranks = {tuple(pair): i for i, (pair, _) in enumerate(encodings)}
    cache = {}

    def bpe_encode(token: str):
        if token in cache:
            return cache[token]
        ids = list(token.encode("utf-8"))
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
        cache[token] = ids
        return ids

    with open(corpus_path) as f:
        corpus = f.read()

    vocab_size = len(vocab)
    total_token_ids = sum(int(info["count"]) for info in vocab.values())
    token_ids = np.empty(total_token_ids, dtype=np.uint16 if vocab_size < 65_536 else np.int32)
    pos = 0
    for token in tqdm(iter_pre_tokens(corpus), desc=f"tokenising {language}", unit="token"):
        ids = bpe_encode(token)
        token_ids[pos : pos + len(ids)] = ids
        pos += len(ids)
    token_ids = token_ids[:pos]
    np.save(token_ids_path, token_ids)
    return np.load(token_ids_path, mmap_mode="r")

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] in {"-h", "--help"}:
        print(f"usage: python {Path(__file__).name} <language> <vocab_timestamp>")
        raise SystemExit(1)

    language = sys.argv[1]
    timestamp = sys.argv[2]
    token_ids = load_or_create_token_ids(language, timestamp)
    print(f"saved: models/{language}/{timestamp}/token_ids.npy (len={len(token_ids)})")

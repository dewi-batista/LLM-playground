from pathlib import Path
from tqdm import tqdm

import json
import numpy as np
import pickle
import sys

from tfs_utils.core import iter_pre_tokens, make_bpe_encoder

HERE = Path(__file__).resolve().parent

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

    bpe_encode = make_bpe_encoder(encodings)

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

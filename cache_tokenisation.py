from pathlib import Path
from tqdm import tqdm

import json
import numpy as np
import pickle
import re
import sys

from tfs_utils.core import iter_pre_tokens, make_bpe_encoder

HERE = Path(__file__).resolve().parent
stream_threshold_bytes = 512 * 1024**2 # 512 MB
save_every_lines = 5_000_000

def load_or_create_token_ids(language, timestamp, corpus_path=None, token_ids_path=None, encodings=None, vocab=None):
    run_dir = HERE / "models" / language / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = corpus_path or (HERE / "data" / f"{language}.txt")
    token_ids_path = token_ids_path or (run_dir / "token_ids.npy")
    state_path = run_dir / "token_ids_state.json"

    if token_ids_path.exists() and not state_path.exists():
        return np.load(token_ids_path, mmap_mode="r")

    streaming = corpus_path.stat().st_size >= stream_threshold_bytes

    if encodings is None:
        encodings_path = run_dir / "merges.pkl"
        with open(encodings_path, "rb") as f:
            encodings = pickle.load(f)

    if vocab is None:
        vocab_path = run_dir / "vocabulary.json"
        with open(vocab_path) as f:
            vocab = json.load(f)

    bpe_encode = make_bpe_encoder(encodings, cache=not streaming)

    vocab_size = len(vocab)
    total_token_ids = sum(int(info["count"]) for info in vocab.values())
    dtype = np.uint16 if vocab_size < 65_536 else np.int32
    pos = 0
    lines_done = 0
    file_pos = 0
    first = True

    if token_ids_path.exists() and state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        pos = int(state["pos"])
        lines_done = int(state["lines_done"])
        file_pos = int(state["file_pos"])
        first = bool(state["first"])
        token_ids = np.lib.format.open_memmap(token_ids_path, mode="r+")
    else:
        token_ids = np.lib.format.open_memmap(token_ids_path, mode="w+", dtype=dtype, shape=(total_token_ids,))
        with open(state_path, "w") as f:
            json.dump({"pos": pos, "lines_done": lines_done, "file_pos": file_pos, "first": first}, f)
            f.write("\n")

    if not streaming:
        with open(corpus_path) as f:
            corpus = f.read()
        for token in tqdm(iter_pre_tokens(corpus), desc=f"tokenising {language}", unit="token"):
            ids = bpe_encode(token)
            token_ids[pos : pos + len(ids)] = ids
            pos += len(ids)
    else:
        with open(corpus_path) as f:
            f.seek(file_pos)
            pbar = tqdm(desc=f"tokenising {language}", unit="line", initial=lines_done)
            while True:
                line = f.readline()
                if not line:
                    break
                for match in re.finditer(r"\S+", line):
                    tok = match.group(0)
                    if first:
                        pre = tok
                        first = False
                    else:
                        pre = " " + tok
                    ids = bpe_encode(pre)
                    token_ids[pos : pos + len(ids)] = ids
                    pos += len(ids)
                lines_done += 1
                pbar.update(1)
                if lines_done % save_every_lines == 0:
                    token_ids.flush()
                    file_pos = f.tell()
                    with open(state_path, "w") as out:
                        json.dump({"pos": pos, "lines_done": lines_done, "file_pos": file_pos, "first": first}, out)
                        out.write("\n")
            pbar.close()

    token_ids.flush()
    del token_ids
    state_path.unlink(missing_ok=True)
    return np.load(token_ids_path, mmap_mode="r")

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] in {"-h", "--help"}:
        print(f"usage: python {Path(__file__).name} <language> <vocab_timestamp>")
        raise SystemExit(1)

    language = sys.argv[1]
    timestamp = sys.argv[2]
    token_ids = load_or_create_token_ids(language, timestamp)
    print(f"saved: models/{language}/{timestamp}/token_ids.npy (len={len(token_ids)})")

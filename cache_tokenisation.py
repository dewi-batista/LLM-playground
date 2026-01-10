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
num_chunks = 10


def _chunk_start(f, start):
    if start <= 0:
        f.seek(0)
        return
    f.seek(start - 1)
    prev = f.read(1)
    f.seek(start)
    if prev != b"\n":
        f.readline()


def _aligned_start_offset(path: Path, start: int) -> int:
    with open(path, "rb") as f:
        _chunk_start(f, start)
        return f.tell()


def _count_newlines(path: Path, start: int, end: int) -> int:
    if end <= start:
        return 0
    total = 0
    block = 8 * 1024 * 1024
    with open(path, "rb") as f:
        f.seek(start)
        remaining = end - start
        while remaining > 0:
            buf = f.read(min(block, remaining))
            if not buf:
                break
            total += buf.count(b"\n")
            remaining -= len(buf)
    return total


def cache_token_ids_chunk(language, timestamp, chunk_idx, *, corpus_path=None, encodings=None, vocab=None):
    run_dir = HERE / "models" / language / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = corpus_path or (HERE / "data" / f"{language}.txt")

    if encodings is None:
        encodings_path = run_dir / "merges.pkl"
        with open(encodings_path, "rb") as f:
            encodings = pickle.load(f)

    if vocab is None:
        vocab_path = run_dir / "vocabulary.json"
        with open(vocab_path) as f:
            vocab = json.load(f)

    streaming = corpus_path.stat().st_size >= stream_threshold_bytes
    bpe_encode = make_bpe_encoder(encodings, cache=not streaming)

    dtype = np.uint16
    file_size = corpus_path.stat().st_size
    start = (file_size * int(chunk_idx)) // num_chunks
    end = (file_size * (int(chunk_idx) + 1)) // num_chunks

    chunk_path = run_dir / f"token_ids_chunk_{int(chunk_idx):02d}.bin"
    chunk_meta_path = run_dir / f"token_ids_chunk_{int(chunk_idx):02d}.json"
    chunk_state_path = run_dir / f"token_ids_chunk_{int(chunk_idx):02d}_state.json"

    if chunk_meta_path.exists():
        return

    file_pos = start
    lines_done = 0
    first = (int(chunk_idx) == 0)
    total_lines = None
    if chunk_state_path.exists():
        with open(chunk_state_path) as f:
            state = json.load(f)
        file_pos = int(state["file_pos"])
        lines_done = int(state["lines_done"])
        first = bool(state["first"])
        total_lines = state.get("total_lines", None)

    if total_lines is None:
        aligned_start = 0 if int(chunk_idx) == 0 else _aligned_start_offset(corpus_path, start)
        end_for_count = file_size if int(chunk_idx) == (num_chunks - 1) else end
        total_lines = _count_newlines(corpus_path, aligned_start, end_for_count)
        if aligned_start < end_for_count and file_size > 0:
            if int(chunk_idx) == (num_chunks - 1):
                with open(corpus_path, "rb") as f:
                    f.seek(file_size - 1)
                    if f.read(1) != b"\n":
                        total_lines += 1
            else:
                with open(corpus_path, "rb") as f:
                    f.seek(end - 1)
                    if f.read(1) != b"\n":
                        total_lines += 1

    chunk_path.parent.mkdir(parents=True, exist_ok=True)

    with open(corpus_path, "rb") as f, open(chunk_path, "ab") as out:
        f.seek(file_pos)
        if not chunk_state_path.exists():
            _chunk_start(f, start)
        pbar = tqdm(
            desc=f"tokenising {language} chunk {int(chunk_idx)+1}/{num_chunks}",
            unit="line",
            initial=lines_done,
            total=total_lines,
        )
        while True:
            pos_before = f.tell()
            if int(chunk_idx) != (num_chunks - 1) and pos_before >= end:
                break
            line = f.readline()
            if not line:
                break
            line = line.decode("utf-8", errors="ignore")
            for match in re.finditer(r"\S+", line):
                tok = match.group(0)
                pre = tok if first else (" " + tok)
                first = False
                ids = bpe_encode(pre)
                np.asarray(ids, dtype=dtype).tofile(out)
            lines_done += 1
            pbar.update(1)
            if lines_done % save_every_lines == 0:
                out.flush()
                file_pos = f.tell()
                with open(chunk_state_path, "w") as s:
                    json.dump(
                        {
                            "file_pos": file_pos,
                            "lines_done": lines_done,
                            "first": first,
                            "total_lines": total_lines,
                        },
                        s,
                    )
                    s.write("\n")
        pbar.close()

    chunk_state_path.unlink(missing_ok=True)
    token_count = chunk_path.stat().st_size // np.dtype(dtype).itemsize
    with open(chunk_meta_path, "w") as f:
        json.dump({"token_count": token_count}, f)
        f.write("\n")


def stitch_token_ids(language, timestamp, *, token_ids_path=None):
    run_dir = HERE / "models" / language / timestamp
    token_ids_path = token_ids_path or (run_dir / "token_ids.npy")

    dtype = np.uint16
    chunk_paths = [run_dir / f"token_ids_chunk_{i:02d}.bin" for i in range(num_chunks)]
    meta_paths = [run_dir / f"token_ids_chunk_{i:02d}.json" for i in range(num_chunks)]
    if not all(p.exists() for p in meta_paths):
        return

    chunk_lens = []
    for p in chunk_paths:
        chunk_lens.append(p.stat().st_size // np.dtype(dtype).itemsize)
    total = sum(chunk_lens)

    token_ids = np.lib.format.open_memmap(token_ids_path, mode="w+", dtype=dtype, shape=(total,))
    pos = 0
    for p, n in zip(chunk_paths, chunk_lens):
        chunk = np.memmap(p, dtype=dtype, mode="r", shape=(n,))
        token_ids[pos : pos + n] = chunk
        pos += n
        del chunk
    token_ids.flush()
    del token_ids

    for p in chunk_paths + meta_paths:
        p.unlink(missing_ok=True)

def load_or_create_token_ids(language, timestamp, corpus_path=None, token_ids_path=None, encodings=None, vocab=None):
    run_dir = HERE / "models" / language / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = corpus_path or (HERE / "data" / f"{language}.txt")
    token_ids_path = token_ids_path or (run_dir / "token_ids.npy")

    if token_ids_path.exists():
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
    dtype = np.uint16

    if not streaming:
        total_token_ids = sum(int(info["count"]) for info in vocab.values())
        token_ids = np.lib.format.open_memmap(token_ids_path, mode="w+", dtype=dtype, shape=(total_token_ids,))
        pos = 0
        with open(corpus_path) as f:
            corpus = f.read()
        for token in tqdm(iter_pre_tokens(corpus), desc=f"tokenising {language}", unit="token"):
            ids = bpe_encode(token)
            token_ids[pos : pos + len(ids)] = ids
            pos += len(ids)
        token_ids.flush()
        del token_ids
    else:
        for i in range(num_chunks):
            cache_token_ids_chunk(
                language,
                timestamp,
                i,
                corpus_path=corpus_path,
                encodings=encodings,
                vocab=vocab,
            )
        stitch_token_ids(language, timestamp, token_ids_path=token_ids_path)
    return np.load(token_ids_path, mmap_mode="r")

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] in {"-h", "--help"}:
        print(f"usage: python {Path(__file__).name} <language> <vocab_timestamp> [chunk_idx]")
        raise SystemExit(1)

    language = sys.argv[1]
    timestamp = sys.argv[2]
    chunk_idx = int(sys.argv[3]) if len(sys.argv) > 3 else None
    if chunk_idx is None:
        token_ids = load_or_create_token_ids(language, timestamp)
        print(f"saved: models/{language}/{timestamp}/token_ids.npy (len={len(token_ids)})")
    else:
        cache_token_ids_chunk(language, timestamp, chunk_idx)
        stitch_token_ids(language, timestamp)

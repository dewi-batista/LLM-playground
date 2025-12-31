from pathlib import Path

import mmap
import sys


HERE = Path(__file__).resolve().parent
DEFAULT_PATH = HERE / "data" / "wiki103.txt"


def count_and_positions_in_file(
    path: Path, needle: bytes, show: int, chunk_size: int = 64 * 1024 * 1024
) -> tuple[int, list[int]]:
    tail_len = max(0, len(needle) - 1)
    count = 0
    positions = []
    tail = b""
    file_pos = 0

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            data = tail + chunk
            search = data[:-tail_len] if tail_len else data
            count += search.count(needle)

            if show and len(positions) < show:
                start = 0
                while len(positions) < show:
                    i = search.find(needle, start)
                    if i < 0:
                        break
                    positions.append(file_pos - len(tail) + i)
                    start = i + len(needle)

            tail = data[-tail_len:] if tail_len else b""
            file_pos += len(chunk)

    return count, positions


def snippets_from_positions(path: Path, positions: list[int], sequence: str, context: int) -> list[str]:
    needle_len = len(sequence.encode("utf-8"))
    out = []
    with open(path, "rb") as f:
        for pos in positions:
            start = max(0, pos - context)
            end = pos + needle_len + context
            f.seek(start)
            chunk = f.read(end - start)
            s = chunk.decode("utf-8", errors="replace").replace("\n", " ")
            out.append(s.replace(sequence, f"[{sequence}]"))
    return out


def count_and_snippets_tokens_in_file(path: Path, tokens: list[str], show: int, context: int) -> tuple[int, list[str]]:
    if not tokens:
        return 0, []
    tokens_b = [t.encode("utf-8") for t in tokens]
    anchor = max(tokens, key=len)
    anchor_b = anchor.encode("utf-8")

    count = 0
    snippets = []

    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            start = 0
            while True:
                pos = mm.find(anchor_b, start)
                if pos < 0:
                    break

                left = max(0, pos - context)
                right = min(len(mm), pos + len(anchor_b) + context)
                window = mm[left:right]
                if all(t in window for t in tokens_b):
                    count += 1
                    if show and len(snippets) < show:
                        s = window.decode("utf-8", errors="replace").replace("\n", " ")
                        if left > 0:
                            s = "..." + s
                        if right < len(mm):
                            s = s + "..."
                        for t in sorted(tokens, key=len, reverse=True):
                            s = s.replace(t, f"[{t}]")
                        snippets.append(s)

                start = pos + len(anchor_b)
        finally:
            mm.close()

    return count, snippets


def count_and_snippets_in_dataset(sequence: str, split: str, show: int, context: int) -> tuple[int, list[str]]:
    from datasets import load_dataset

    data = load_dataset("wikitext", "wikitext-103-v1", split=split)
    count = 0
    snippets = []
    n = len(sequence)

    for line in data["text"]:
        if not line.strip():
            continue
        c = line.count(sequence)
        if c:
            count += c
            if show and len(snippets) < show:
                i = line.find(sequence)
                left = max(0, i - context)
                right = min(len(line), i + n + context)
                s = line[left:right].replace("\n", " ")
                if left > 0:
                    s = "..." + s
                if right < len(line):
                    s = s + "..."
                snippets.append(s.replace(sequence, f"[{sequence}]"))

    return count, snippets


def count_and_snippets_tokens_in_dataset(tokens: list[str], split: str, show: int, context: int) -> tuple[int, list[str]]:
    from datasets import load_dataset

    data = load_dataset("wikitext", "wikitext-103-v1", split=split)
    count = 0
    snippets = []

    for line in data["text"]:
        if not line.strip():
            continue
        if not all(t in line for t in tokens):
            continue

        count += 1
        if show and len(snippets) < show:
            positions = [(line.find(t), len(t)) for t in tokens]
            first = min(i for i, _ in positions)
            last = max(i + n for i, n in positions)
            left = max(0, first - context)
            right = min(len(line), last + context)
            s = line[left:right].replace("\n", " ")
            if left > 0:
                s = "..." + s
            if right < len(line):
                s = s + "..."
            for t in sorted(tokens, key=len, reverse=True):
                s = s.replace(t, f"[{t}]")
            snippets.append(s)

    return count, snippets


if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
    print(
        f"usage: python {Path(__file__).name} <sequence...> [--path file] [--dataset] [--split train] [--show 5] [--context 80] [--tokens]"
    )
    raise SystemExit(1)

args = sys.argv[1:]
path = DEFAULT_PATH
use_dataset = False
split = "train"
show = 5
context = 80
tokens_mode = False

seq_parts = []
i = 0
while i < len(args):
    if args[i] == "--path":
        path = Path(args[i + 1])
        i += 2
        continue
    if args[i] == "--dataset":
        use_dataset = True
        i += 1
        continue
    if args[i] == "--split":
        split = args[i + 1]
        i += 2
        continue
    if args[i] == "--show":
        show = int(args[i + 1])
        i += 2
        continue
    if args[i] == "--context":
        context = int(args[i + 1])
        i += 2
        continue
    if args[i] == "--tokens":
        tokens_mode = True
        i += 1
        continue
    seq_parts.append(args[i])
    i += 1

if not seq_parts:
    print(
        f"usage: python {Path(__file__).name} <sequence...> [--path file] [--dataset] [--split train] [--show 5] [--context 80] [--tokens]"
    )
    raise SystemExit(1)

if tokens_mode:
    if len(seq_parts) > 5:
        print("ERROR: --tokens supports max 5 tokens")
        raise SystemExit(1)

    if use_dataset:
        count, snippets = count_and_snippets_tokens_in_dataset(seq_parts, split=split, show=show, context=context)
    else:
        if not path.is_absolute() and not path.exists():
            path = HERE / path
        count, snippets = count_and_snippets_tokens_in_file(path, seq_parts, show=show, context=context)
else:
    sequence = " ".join(seq_parts)
    if use_dataset:
        count, snippets = count_and_snippets_in_dataset(sequence, split=split, show=show, context=context)
    else:
        if not path.is_absolute() and not path.exists():
            path = HERE / path
        needle = sequence.encode("utf-8")
        count, positions = count_and_positions_in_file(path, needle, show=show)
        snippets = snippets_from_positions(path, positions, sequence, context=context)

print(f"count: {count}")
for i, s in enumerate(snippets):
    print(f"{i + 1}: {s}")

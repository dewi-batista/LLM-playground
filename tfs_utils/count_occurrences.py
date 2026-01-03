from pathlib import Path

import mmap
import sys

def count_and_positions_in_file(path: Path, needle: bytes, show: int, chunk_size: int = 64 * 1024 * 1024):
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

def snippets_from_positions(path: Path, positions: list[int], sequence: str, context: int):
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

def count_and_snippets_tokens_in_file(path: Path, tokens: list[str], show: int, context: int):
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

if __name__ == "__main__":

    HERE = Path(__file__).resolve().parents[1]
    DEFAULT_PATH = HERE / "data" / "wiki103.txt"

    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(
            f"usage: python {Path(__file__).name} <words...> [--path file] [--show 5] [--context 80] [--cooccur]\n"
            "\nexamples:\n"
            f"\nphrase search: python {Path(__file__).name} New York --path {DEFAULT_PATH.relative_to(HERE)} --show 5\n"
            f"\nco-occurrence search: python {Path(__file__).name} New York United States --cooccur --path {DEFAULT_PATH.relative_to(HERE)} --context 80 --show 5\n"
        )
        raise SystemExit(1)

    args = sys.argv[1:]
    path = DEFAULT_PATH
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
        if args[i] == "--show":
            show = int(args[i + 1])
            i += 2
            continue
        if args[i] == "--context":
            context = int(args[i + 1])
            i += 2
            continue
        if args[i] == "--cooccur":
            tokens_mode = True
            i += 1
            continue
        seq_parts.append(args[i]) # append tokens to search for
        i += 1

    if not seq_parts:
        print(f"usage: python {Path(__file__).name} <sequence...> [--path file] [--show 5] [--context 80] [--cooccur]")
        raise SystemExit(1)

    if tokens_mode:
        if not path.is_absolute() and not path.exists():
            path = HERE / path
        count, snippets = count_and_snippets_tokens_in_file(path, seq_parts, show=show, context=context)
    else:
        sequence = " ".join(seq_parts)
        if not path.is_absolute() and not path.exists():
            path = HERE / path
        needle = sequence.encode("utf-8")
        count, positions = count_and_positions_in_file(path, needle, show=show)
        snippets = snippets_from_positions(path, positions, sequence, context=context)

    print(f"count: {count}")
    for i, s in enumerate(snippets):
        print(f"{i + 1}: {s}")

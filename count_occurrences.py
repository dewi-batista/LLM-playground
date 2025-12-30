from pathlib import Path

import sys


HERE = Path(__file__).resolve().parent
DEFAULT_PATH = HERE / "data" / "wiki103.txt"


def count_occurrences(path: Path, needle: bytes, chunk_size: int = 64 * 1024 * 1024) -> int:
    tail_len = max(0, len(needle) - 1)
    count = 0
    tail = b""

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            data = tail + chunk
            if tail_len:
                count += data[:-tail_len].count(needle)
                tail = data[-tail_len:]
            else:
                count += data.count(needle)
                tail = b""

    return count


if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
    print(f"usage: python {Path(__file__).name} <sequence...> [--path file]")
    raise SystemExit(1)

args = sys.argv[1:]
path = DEFAULT_PATH
if "--path" in args:
    i = args.index("--path")
    path = Path(args[i + 1])
    args = args[:i] + args[i + 2 :]

sequence = " ".join(args)
if not sequence:
    print(f"usage: python {Path(__file__).name} <sequence...> [--path file]")
    raise SystemExit(1)

if not path.is_absolute() and not path.exists():
    path = HERE / path

needle = sequence.encode("utf-8")
print(count_occurrences(path, needle))

from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"


def main():
    if len(sys.argv) in {1, 2} and (len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}):
        print(f"usage: python {Path(__file__).name} <ckpt_path>")
        print(f"   or: python {Path(__file__).name} <language> <timestamp> <run>")
        raise SystemExit(1)

    if len(sys.argv) == 2:
        ckpt_path = Path(sys.argv[1])
        if not ckpt_path.is_absolute():
            ckpt_path = (HERE / ckpt_path).resolve()
    elif len(sys.argv) == 4:
        language = sys.argv[1]
        timestamp = sys.argv[2]
        run_arg = sys.argv[3]
        run_name = f"training_run_{int(run_arg)}" if run_arg.isdigit() else run_arg
        ckpt_path = (MODELS_DIR / language / timestamp / run_name / "weights.ckpt").resolve()
    else:
        print(f"usage: python {Path(__file__).name} <ckpt_path>")
        print(f"   or: python {Path(__file__).name} <language> <timestamp> <run>")
        raise SystemExit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    run_dir = ckpt_path.parent.parent
    try:
        run_dir_rel = run_dir.relative_to(HERE)
    except ValueError:
        run_dir_rel = run_dir

    updates = {
        "bpe_vocab_path": str(run_dir_rel / "vocabulary.json"),
        "bpe_encodings_path": str(run_dir_rel / "merges.pkl"),
        "token_ids_path": str(run_dir_rel / "token_ids.npy"),
        "vocabulary_path": str(run_dir_rel / "vocabulary.json"),
        "merges_path": str(run_dir_rel / "merges.pkl"),
    }

    to_change = {k: v for k, v in updates.items() if k in ckpt}
    if not to_change:
        print("no known *_path keys found in ckpt")
        raise SystemExit(1)

    for k, v in to_change.items():
        old = ckpt.get(k)
        if old != v:
            print(f"{k}: {old} -> {v}")
        ckpt[k] = v

    torch.save(ckpt, ckpt_path)
    print(f"saved: {ckpt_path}")


if __name__ == "__main__":
    main()

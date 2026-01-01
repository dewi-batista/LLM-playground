from pathlib import Path

import pprint
import sys
import torch

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE.parent / "models"

if len(sys.argv) < 3 or sys.argv[1] in {"-h", "--help"}:
    print(f"usage: python {Path(__file__).name} <language> <timestamp> <model_number>")
    raise SystemExit(1)

language = sys.argv[1]
timestamp = sys.argv[2]
model_number = sys.argv[3]

run_dir = MODELS_DIR / language / timestamp
ckpt_path = run_dir / "transformer" / f"training_run_{model_number}" / "weights.ckpt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

DROP = {
    "E_state_dict", "U_state_dict", "optimizer_state_dict",
    "rng_state_py", "rng_state_np", "rng_state_torch", "rng_state_cuda",
    "index_to_token",
}

print("index_to_token len:", len(ckpt.get("index_to_token", [])))
pprint.pp({k: v for k, v in ckpt.items() if k not in DROP})

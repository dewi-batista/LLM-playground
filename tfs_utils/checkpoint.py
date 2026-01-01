import json
import os
from pathlib import Path

import torch
from tqdm import tqdm


def atomic_torch_save(obj, path: Path) -> bool:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
        return True
    except Exception as e:
        tqdm.write(f"WARNING: checkpoint save failed: {e}")
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def atomic_json_save(obj: dict, path: Path) -> bool:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w") as f:
            json.dump(obj, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, path)
        return True
    except Exception as e:
        tqdm.write(f"WARNING: meta save failed: {e}")
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False


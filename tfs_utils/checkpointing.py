from pathlib import Path
from tqdm import tqdm

import os
import torch


def atomic_torch_save(obj, path: Path) -> bool:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.parent.mkdir(parents=True, exist_ok=True)

        with open(tmp_path, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, path)

        try:
            dir_fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            pass

        return True
    except Exception as e:
        tqdm.write(f"WARNING: ckpt save failed: {e}")
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False


import json
import os
import shutil
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


def atomic_torch_save_last_and_best(
    obj,
    *,
    last_path: Path,
    best_path: Path,
    save_best: bool,
) -> tuple[bool, bool]:
    save_tmp = last_path.with_suffix(last_path.suffix + ".save_tmp")
    link_tmp = last_path.with_suffix(last_path.suffix + ".link_tmp")
    try:
        torch.save(obj, save_tmp)
    except Exception as e:
        tqdm.write(f"WARNING: checkpoint save failed: {e}")
        try:
            save_tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False, False

    saved_last = False
    saved_best = False
    try:
        if save_best:
            os.replace(save_tmp, best_path)
            saved_best = True

            try:
                link_tmp.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                os.link(best_path, link_tmp)
            except Exception:
                try:
                    shutil.copyfile(best_path, link_tmp)
                except Exception as e:
                    tqdm.write(f"WARNING: last checkpoint link/copy failed: {e}")
                    return False, True
            os.replace(link_tmp, last_path)
            saved_last = True
        else:
            os.replace(save_tmp, last_path)
            saved_last = True
    except Exception as e:
        tqdm.write(f"WARNING: checkpoint finalize failed: {e}")
        try:
            save_tmp.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            link_tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False, saved_best

    return saved_last, saved_best

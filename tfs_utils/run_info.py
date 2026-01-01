from __future__ import annotations

from pathlib import Path
import socket
import subprocess
import sys

import torch


def _git_commit(repo_root: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    commit = (r.stdout or "").strip()
    return commit or None


def run_header(repo_root: Path) -> dict:
    return {
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "git_commit": _git_commit(repo_root),
    }


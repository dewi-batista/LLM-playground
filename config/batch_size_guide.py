"""
Batch size "that fits" on GPU — practical notes + helper functions.

Why there isn't a perfect formula
--------------------------------
- PyTorch uses a caching allocator: memory can be "reserved" even if not actively "allocated".
- Fragmentation + different kernels can change peaks slightly run-to-run.
- Peak VRAM happens during forward/backward/optimizer.step (not at a single obvious line).

Still, for a fixed model + fixed seq_len, peak VRAM is usually ~linear in batch_size,
so you can estimate the max batch size from 1–2 quick measurements.

Key idea
--------
Measure peak VRAM for a single *training step* (forward + backward + step) at two batch sizes,
then extrapolate.

Also useful: an OOM-based batch finder that runs one step and backs off.
(That method uses try/except; you don't need it in your main train script unless you want it.)
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import time
from typing import Callable

import torch


@dataclass(frozen=True)
class CudaMem:
    free: int
    total: int


def bytes_fmt(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def cuda_mem_info(device: torch.device | None = None) -> CudaMem:
    if device is None:
        device = torch.device("cuda")
    if device.type != "cuda":
        raise ValueError("cuda_mem_info() requires a CUDA device")
    free, total = torch.cuda.mem_get_info(device)
    return CudaMem(free=int(free), total=int(total))


@dataclass(frozen=True)
class PeakMem:
    allocated: int
    reserved: int


def measure_step_peak_mem(step_fn: Callable[[], None], device: torch.device | None = None) -> PeakMem:
    """
    Measures peak CUDA memory (allocated + reserved) while running step_fn().

    step_fn should run ONE training step end-to-end:
      - forward
      - loss.backward()
      - optimizer.step()

    Notes:
    - synchronize() makes timing + peak tracking more reliable (CUDA is async).
    - reserved >= allocated; reserved includes allocator cache. reserved is safer for "will it OOM?"
    """
    if device is None:
        device = torch.device("cuda")
    if device.type != "cuda":
        raise ValueError("measure_step_peak_mem() requires a CUDA device")

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    step_fn()

    torch.cuda.synchronize(device)
    return PeakMem(
        allocated=int(torch.cuda.max_memory_allocated(device)),
        reserved=int(torch.cuda.max_memory_reserved(device)),
    )


def estimate_bytes_per_batch(b1: int, m1: int, b2: int, m2: int) -> float:
    """
    Simple linear estimate of memory cost per +1 batch.
    Use m = peak reserved bytes (recommended) or peak allocated bytes (less conservative).
    """
    if b1 == b2:
        raise ValueError("need two different batch sizes")
    return (m2 - m1) / (b2 - b1)


def estimate_max_batch_size_linear(
    *,
    b2: int,
    m2: int,
    free_bytes: int,
    bytes_per_batch: float,
    margin_bytes: int = 2 * 1024**3,
) -> int:
    """
    Extrapolates max batch size that should fit given a measured point (b2, m2).

    - free_bytes: torch.cuda.mem_get_info()[0]
    - margin_bytes: headroom for fragmentation/spikes (1–2GB is typical)
    """
    usable = max(0, free_bytes - margin_bytes)
    extra_batches = int(usable / max(1.0, bytes_per_batch))
    return max(1, b2 + extra_batches)


def find_max_batch_size_by_oom(
    step_fn_for_batch: Callable[[int], None],
    *,
    start: int = 32,
    max_batch: int = 4096,
    device: torch.device | None = None,
) -> int:
    """
    Convenience OOM-based search:
    - grow batch size (doubling) until OOM
    - then binary search between last_ok and first_oom

    This is robust but slower than the linear-extrapolation method.
    """
    if device is None:
        device = torch.device("cuda")
    if device.type != "cuda":
        raise ValueError("find_max_batch_size_by_oom() requires a CUDA device")

    def ok(b: int) -> bool:
        try:
            step_fn_for_batch(b)
            torch.cuda.synchronize(device)
            return True
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            msg = str(e).lower()
            if "out of memory" not in msg:
                raise
            gc.collect()
            torch.cuda.empty_cache()
            return False

    b = start
    if not ok(b):
        return max(1, b // 2)

    last_ok = b
    while True:
        b = min(max_batch, b * 2)
        if b == last_ok:
            return last_ok
        if ok(b):
            last_ok = b
            if b == max_batch:
                return last_ok
        else:
            first_oom = b
            break

    lo, hi = last_ok, first_oom
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


"""
How you'd use the linear estimate in a training script (template)
---------------------------------------------------------------

Assume you can run a single training step given a batch_size:

    def one_step(batch_size: int):
        # sample data of shape (batch_size, seq_len)
        # forward/backward/step
        ...

Then:

    dev = torch.device("cuda")
    torch.cuda.empty_cache()
    gc.collect()

    # pick two batch sizes that definitely fit
    b1, b2 = 32, 64
    m1 = measure_step_peak_mem(lambda: one_step(b1), dev).reserved
    m2 = measure_step_peak_mem(lambda: one_step(b2), dev).reserved

    per_batch = estimate_bytes_per_batch(b1, m1, b2, m2)
    free = cuda_mem_info(dev).free

    b_max = estimate_max_batch_size_linear(
        b2=b2, m2=m2, free_bytes=free, bytes_per_batch=per_batch, margin_bytes=2 * 1024**3
    )

    print("b1 peak:", bytes_fmt(m1), "b2 peak:", bytes_fmt(m2))
    print("bytes/batch:", bytes_fmt(int(per_batch)))
    print("free now:", bytes_fmt(free), "estimated max batch:", b_max)

Reality check:
- Try b_max and b_max-1 once; if it OOMs, reduce margin or just step down a bit.
- Re-measure after changing seq_len, num_blocks, d_model, etc.
"""


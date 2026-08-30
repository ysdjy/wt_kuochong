"""Single-flight GPU gating, ported from the pattern already validated in the
old project's `final_statistical_evidence/scripts/run_transfer_tasks.py::gpu_free_enough`
-- polls `nvidia-smi`, waits (rather than launching concurrently) if another
process already has meaningful GPU memory in use. This is exactly why
`run_phm2010.py` defaults to `--workers 1`: real background training on this
project has previously triggered CUDA OOM under multi-way concurrency (see
`05_统计检验/seed_statistics/B9_PHM2010_D1_seed_landscape/`'s own README for
that incident).
"""
from __future__ import annotations

import subprocess
import time


def gpu_memory_used_mib() -> int | None:
    """Returns MiB currently used on GPU 0, or None if nvidia-smi is unavailable
    (e.g. no NVIDIA GPU present, or CPU-only machine) -- callers should treat
    None as "can't tell, proceed" rather than blocking forever."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode != 0:
            return None
        first_line = out.stdout.strip().splitlines()[0]
        return int(first_line.strip())
    except Exception:
        return None


def gpu_free_enough(wait_seconds: int = 600, threshold_mib: int = 2000, poll_interval: float = 5.0) -> bool:
    """Blocks (polling every `poll_interval` seconds) until GPU memory used is
    below `threshold_mib`, or until `wait_seconds` have elapsed. Returns True if
    it returned because the GPU was free enough, False if it gave up after
    timing out (caller may still proceed -- this is advisory, not a hard
    guarantee against OOM, since another process could start between the check
    and the caller's own allocation). No-op (returns True immediately) if
    nvidia-smi isn't available, so this never blocks CPU-only runs."""
    deadline = time.time() + wait_seconds
    while True:
        used = gpu_memory_used_mib()
        if used is None or used < threshold_mib:
            return True
        if time.time() >= deadline:
            print(
                f"[gpu_gate] WARNING: GPU still shows {used}MiB used after waiting "
                f"{wait_seconds}s (threshold {threshold_mib}MiB) -- proceeding anyway.",
            )
            return False
        time.sleep(poll_interval)

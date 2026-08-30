"""Builds the unified `run_meta.json` provenance record every formal run must
write, per RESULTS_POLICY.md. Every adapter should build its run_meta via
`build_run_meta()` rather than hand-rolling the dict, so the required key set
never silently drifts between methods.
"""
from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any


def _git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _torch_cuda_gpu_info() -> dict[str, Any]:
    info = {"torch_version": None, "cuda_version": None, "gpu_name": None}
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
        # Guard on device_count() too, not just is_available(): a smoke test run
        # under CUDA_VISIBLE_DEVICES="" (forcing CPU-only) can still have
        # is_available() return True on some driver/torch combinations while
        # device_count() is 0, in which case get_device_name(0) raises.
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dict_hash(d: dict) -> str:
    """Stable sha256 of a JSON-serializable dict, for config_hash/label_hash/etc."""
    blob = json.dumps(d, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_run_meta(
    *,
    method: str,
    dataset: str,
    task: str,
    train_cutters: list[str],
    test_cutter: str,
    seed: int,
    preprocess_seed: int,
    repo_root: Path,
    start_time: str,
    end_time: str,
    runtime_sec: float,
    feature_hash: str | None = None,
    split_hash: str | None = None,
    label_hash: str | None = None,
    evaluation_universe_hash: str | None = None,
    config_hash: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Assemble the full required run_meta.json key set (task spec section 25)."""
    meta = {
        "method": method,
        "dataset": dataset,
        "task": task,
        "train_cutters": train_cutters,
        "test_cutter": test_cutter,
        "seed": seed,
        "preprocess_seed": preprocess_seed,
        "git_commit": _git_commit(repo_root),
        "machine_alias": socket.gethostname(),
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "feature_hash": feature_hash,
        "split_hash": split_hash,
        "label_hash": label_hash,
        "evaluation_universe_hash": evaluation_universe_hash,
        "config_hash": config_hash,
        "start_time": start_time,
        "end_time": end_time,
        "runtime_sec": runtime_sec,
    }
    meta.update(_torch_cuda_gpu_info())
    if extra:
        meta.update(extra)
    return meta

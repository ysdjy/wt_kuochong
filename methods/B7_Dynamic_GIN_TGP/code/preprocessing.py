# -*- coding: utf-8 -*-
r"""
Dynamic GIN + TGP native preprocessing pipeline. Vendored from the old
project's `baselines/dynamic_gin_tgp/preprocessing.py` (sha256
29f08d3d2ff1a6883a029f45b2f2a36a1d6f8ea9a398a014501513e553ae19b4) with ONLY
path-routing changes (task spec section 35 permits this): raw-data root and
window-cache root are now resolved from `PHM2010_ROOT` env var / repo-relative
paths instead of hardcoded old-project-relative paths. Model logic, window
extraction, and Protocol-A manifest logic are unchanged. See
../source_manifest.json for the exact diff.

Pipeline (see old `baselines/dynamic_gin_tgp/PAPER_SPEC.md`, Cao et al.,
Measurement 2026, DOI: 10.1016/j.measurement.2025.119007):

    raw Fx,Fy,Fz,Vx,Vy,Vz  (PHM2010, {PHM2010_ROOT}/c{cond}/c{cond}/c_{cond}_{run:03d}.csv,
                             AE column dropped per paper Sec 3.2)
      -> stable-cut: drop first/last 40,000 points
      -> divide stable region into 10 equal portions
      -> extract 288-length window per portion (centered)
      -> Protocol B (this repo's only formal protocol): all 10 portions/run,
         condition-relative E/M/L labels via stage_labels.py, common run universe.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parents[1]

PHM2010_ROOT = Path(os.environ.get("PHM2010_ROOT", str(REPO_ROOT / "data" / "PHM2010" / "raw")))
CACHE_DIR = Path(os.environ.get(
    "B7_WINDOW_CACHE_DIR", str(REPO_ROOT / "data" / "PHM2010" / "cache" / "dynamic_gin_tgp" / "windows")
))

SAMPLE_LEN = 288          # Table 1, "one rotational cycle"
TRIM_POINTS = 40_000      # Sec 3.2: drop first/last 40,000 raw points
N_PORTIONS = 10           # Sec 3.2: divide stable region into 10 equal portions
N_CHANNELS = 6            # Fx,Fy,Fz,Vx,Vy,Vz (AE dropped)


def raw_csv_path(condition: str, run_id: int) -> Path:
    cond_lower = condition.lower()
    num = cond_lower[1:]  # "c1" -> "1"
    return PHM2010_ROOT / cond_lower / cond_lower / f"c_{num}_{run_id:03d}.csv"


def load_raw_pass(condition: str, run_id: int) -> np.ndarray:
    """Returns [6, N] array: Fx,Fy,Fz,Vx,Vy,Vz (AE column dropped)."""
    path = raw_csv_path(condition, run_id)
    df = pd.read_csv(path, header=None)
    arr = df.values[:, :6].astype(np.float32).T  # [6, N]
    return arr


def stable_cut_region(signal: np.ndarray, trim: int = TRIM_POINTS) -> np.ndarray:
    """Drop the first/last `trim` points per pass (Sec 3.2)."""
    n = signal.shape[1]
    if n <= 2 * trim:
        raise ValueError(f"Signal too short ({n} pts) to trim {trim} from each end")
    return signal[:, trim:n - trim]


def portion_bounds(stable_len: int, n_portions: int = N_PORTIONS):
    """10 equal-length portions (last portion absorbs the remainder)."""
    step = stable_len // n_portions
    bounds = [(i * step, (i + 1) * step) for i in range(n_portions - 1)]
    bounds.append(((n_portions - 1) * step, stable_len))
    return bounds


def extract_window(portion: np.ndarray, length: int = SAMPLE_LEN) -> np.ndarray:
    """Centered length-288 window inside a portion."""
    n = portion.shape[1]
    if n < length:
        raise ValueError(f"Portion too short ({n} pts) for a {length}-length window")
    start = (n - length) // 2
    return portion[:, start:start + length]


def all_portion_windows(condition: str, run_id: int) -> np.ndarray:
    """Returns [10, 6, 288]: one centered window per of the 10 stable-region
    portions for this run."""
    raw = load_raw_pass(condition, run_id)
    stable = stable_cut_region(raw)
    bounds = portion_bounds(stable.shape[1])
    windows = np.stack([
        extract_window(stable[:, a:b]) for a, b in bounds
    ], axis=0)  # [10,6,288]
    return windows.astype(np.float32)


def build_windows_cache(conditions=("C1", "C4", "C6"), max_runs: int | None = None,
                         verbose: bool = True) -> pd.DataFrame:
    """Extracts and caches all 10 portion-windows for every run of every
    condition to CACHE_DIR/{cond}_{run:03d}.npy (shape [10,6,288]).

    `max_runs`: cap on run_id per condition (for fast smoke tests only —
    production runs pass max_runs=None to cover the full 315-run universe,
    since the common evaluation universe run_id 12-315 needs runs up to 315).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for cond in conditions:
        max_run = max_runs or 315
        for run_id in range(1, max_run + 1):
            csv_path = raw_csv_path(cond, run_id)
            if not csv_path.exists():
                if verbose:
                    print(f"[build_windows_cache] missing {csv_path}, stopping {cond} at run_id={run_id}")
                break
            out_path = CACHE_DIR / f"{cond}_{run_id:03d}.npy"
            if not out_path.exists():
                windows = all_portion_windows(cond, run_id)  # [10,6,288]
                np.save(out_path, windows)
            rows.append({"condition": cond, "run_id": run_id, "npy_path": str(out_path)})
    return pd.DataFrame(rows)


def load_windows(condition: str, run_id: int) -> np.ndarray:
    path = CACHE_DIR / f"{condition}_{run_id:03d}.npy"
    return np.load(path)

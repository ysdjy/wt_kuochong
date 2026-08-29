# -*- coding: utf-8 -*-
r"""
Dynamic GIN + TGP native preprocessing pipeline, reimplemented from
`baselines/dynamic_gin_tgp/PAPER_SPEC.md` (Cao et al., Measurement 2026,
DOI: 10.1016/j.measurement.2025.119007).

Pipeline (see PAPER_SPEC.md sec 1-2 for the full annotated table):

    raw Fx,Fy,Fz,Vx,Vy,Vz  (PHM2010, archive/c{cond}/c{cond}/c_{cond}_{run:03d}.csv,
                             AE column dropped per paper Sec 3.2)
      -> stable-cut: drop first/last 40,000 points                 [Explicit]
      -> divide stable region into 10 equal portions                [Explicit]
      -> extract 288-length window per portion (centered)           [Explicit
                                                                       length;
                                                                       Missing
                                                                       exact
                                                                       start
                                                                       offset
                                                                       -> centered,
                                                                       see
                                                                       PAPER_SPEC]
      -> Protocol A: stratified 5/2/3 samples per run for
         Initial/Normal/Severe stages (first 300 passes only)       [Explicit]
      -> Protocol B: all 10 portions/run, condition-relative E/M/L
         labels, common run universe                                [Unified]

This module has NO project-shared preprocessing dependency (does not
import from 代码/ or from other baselines/); it only uses
`baselines/dynamic_gin_tgp/data/label_utils.py` for Protocol B stage
labels (Unified DC-PSR labels), per the task's directory-isolation
requirement. GASF encoding is NOT done here -- it is part of the network
forward pass (Table 1's "Spatial feature extraction" block), implemented
in model.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
ARCHIVE_DIR = PROJECT_ROOT / "archive"
DATA_DIR = THIS_DIR / "data"
WINDOW_DIR = DATA_DIR / "windows"

SAMPLE_LEN = 288          # Table 1, "one rotational cycle"
TRIM_POINTS = 40_000      # Sec 3.2: drop first/last 40,000 raw points
N_PORTIONS = 10           # Sec 3.2: divide stable region into 10 equal portions
N_CHANNELS = 6            # Fx,Fy,Fz,Vx,Vy,Vz (AE dropped)

# Protocol A stage boundaries (paper's own, Sec 3.1/3.2), 1-indexed pass numbers.
PROTO_A_STAGE_BOUNDS = {
    "initial": (1, 50),
    "normal": (51, 210),
    "severe": (211, 315),
}
PROTO_A_PASS_LIMIT = 300  # only first 300 passes used, Sec 3.2
PROTO_A_SAMPLES_PER_RUN = {"initial": 5, "normal": 2, "severe": 3}  # Sec 3.2


def raw_csv_path(condition: str, run_id: int) -> Path:
    cond_lower = condition.lower().replace("c", "c")
    num = cond_lower[1:]  # "c1" -> "1"
    return ARCHIVE_DIR / cond_lower / cond_lower / f"c_{num}_{run_id:03d}.csv"


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
    """Centered length-288 window inside a portion (PAPER_SPEC.md sec 2:
    exact start offset within a portion is Missing in the paper; centered
    window is the documented implementation choice)."""
    n = portion.shape[1]
    if n < length:
        raise ValueError(f"Portion too short ({n} pts) for a {length}-length window")
    start = (n - length) // 2
    return portion[:, start:start + length]


def all_portion_windows(condition: str, run_id: int) -> np.ndarray:
    """Returns [10, 6, 288]: one centered window per of the 10 stable-region
    portions for this run. Used directly by Protocol B (all portions kept)
    and subsetted by Protocol A (first-N-portions-per-stage rule)."""
    raw = load_raw_pass(condition, run_id)
    stable = stable_cut_region(raw)
    bounds = portion_bounds(stable.shape[1])
    windows = np.stack([
        extract_window(stable[:, a:b]) for a, b in bounds
    ], axis=0)  # [10,6,288]
    return windows.astype(np.float32)


def stage_for_pass_protocol_a(run_id: int) -> str | None:
    for stage, (lo, hi) in PROTO_A_STAGE_BOUNDS.items():
        if lo <= run_id <= hi:
            return stage
    return None


def build_protocol_a_manifest(condition: str) -> pd.DataFrame:
    """Stratified sampling per Sec 3.2: Initial 5/run, Normal 2/run, Severe
    3/run, drawn from the first N portions (N = samples-per-run) of each
    run's 10 stable-region portions -- reproducing the paper's own worked
    example ("collect 288-length sample points from each of the first
    five segments" for a 50-pass initial stage)."""
    rows = []
    for run_id in range(1, PROTO_A_PASS_LIMIT + 1):
        stage = stage_for_pass_protocol_a(run_id)
        if stage is None:
            continue
        n_samp = PROTO_A_SAMPLES_PER_RUN[stage]
        for portion_idx in range(n_samp):
            rows.append({
                "condition": condition, "run_id": run_id,
                "stage_original": stage, "portion_idx": portion_idx,
            })
    return pd.DataFrame(rows)


STAGE_A_TO_ID = {"initial": 0, "normal": 1, "severe": 2}


def build_all_windows_cache(conditions=("C1", "C4", "C6"), pass_limit: int | None = None,
                             verbose: bool = True) -> pd.DataFrame:
    """Extracts and caches all 10 portion-windows for every run of every
    condition to data/windows/{cond}_{run:03d}.npy (shape [10,6,288]).
    Returns a metadata DataFrame with condition, run_id, npy_path,
    stage_original (Protocol A label, None if run_id > 315 or condition
    lacks that pass), n_portions=10.

    pass_limit=None -> caches all 315 passes (needed for Protocol B, whose
    common test universe may need runs 301-315 too, per task instruction
    #39). Protocol A itself is restricted to 1-300 downstream regardless
    of what's cached here.
    """
    WINDOW_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for cond in conditions:
        max_run = pass_limit or 315
        for run_id in range(1, max_run + 1):
            csv_path = raw_csv_path(cond, run_id)
            if not csv_path.exists():
                if verbose:
                    print(f"[build_all_windows_cache] missing {csv_path}, stopping {cond} at run_id={run_id}")
                break
            out_path = WINDOW_DIR / f"{cond}_{run_id:03d}.npy"
            if not out_path.exists():
                windows = all_portion_windows(cond, run_id)  # [10,6,288]
                np.save(out_path, windows)
                if verbose and run_id % 50 == 0:
                    print(f"[build_all_windows_cache] {cond} run {run_id}/{max_run} cached")
            stage_orig = stage_for_pass_protocol_a(run_id)
            rows.append({
                "condition": cond, "run_id": run_id,
                "npy_path": str(out_path.relative_to(THIS_DIR)),
                "stage_original": stage_orig,
                "stage_original_id": STAGE_A_TO_ID.get(stage_orig, -1),
            })
    meta = pd.DataFrame(rows)
    meta.to_csv(DATA_DIR / "metadata.csv", index=False, encoding="utf-8-sig")
    return meta


def load_windows(condition: str, run_id: int) -> np.ndarray:
    path = WINDOW_DIR / f"{condition}_{run_id:03d}.npy"
    return np.load(path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass-limit", type=int, default=None)
    args = ap.parse_args()
    meta = build_all_windows_cache(pass_limit=args.pass_limit)
    print(meta.groupby("condition").size())
    print(f"Total cached runs: {len(meta)}")
    a_manifest = pd.concat([build_protocol_a_manifest(c) for c in ("C1", "C4", "C6")], ignore_index=True)
    print("Protocol A manifest sample counts (expect 2520 total, 840/tool):")
    print(a_manifest.groupby(["condition", "stage_original"]).size())
    print(f"Total Protocol A samples: {len(a_manifest)}")

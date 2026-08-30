# -*- coding: utf-8 -*-
r"""
DP2Net native preprocessing pipeline. Vendored from the old project's
`baselines/dp2net/preprocessing.py` (sha256
949c64aca1b9e23e387c20adb25f199a03750bd041c96c4645f78e0e5c961fbe) with ONLY
path-routing changes (task spec section 35): raw-data root and window-cache
root now resolve from `PHM2010_ROOT` env var / repo-relative paths. This
adapter only uses the "Protocol B-D1" (pooled-source, C1+C4->C6 style,
DP2Net-adapted) regime — Protocol A (paper-native sanity check) and
Protocol B-S (native single-source) code paths are dropped; see
../source_manifest.json.

Physical constants (S/G's kernel size k, Vst's period L, rise fraction P) are
unchanged from the paper's own PHM2010 process parameters (Table 1).

    raw Fx only  ({PHM2010_ROOT}/c{cond}/c{cond}/c_{cond}_{run:03d}.csv, column 0)
      -> Butterworth low-pass, cutoff=1733Hz, native 50kHz
      -> 8 windows/run, 4608-length, evenly spread across the run's
         low-pass-filtered signal (data/PHM2010/cache/dp2net/windows_unified/)
      -> DC-PSR unified E/M/L labels via stage_labels.py, task-parameterized split
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parents[1]

PHM2010_ROOT = Path(os.environ.get("PHM2010_ROOT", str(REPO_ROOT / "data" / "PHM2010" / "raw")))
CACHE_DIR = Path(os.environ.get(
    "B8_WINDOW_CACHE_DIR", str(REPO_ROOT / "data" / "PHM2010" / "cache" / "dp2net" / "windows_unified")
))

# ---------------------------------------------------------------------------
# Physical constants (PAPER_SPEC.md sec 1, Table 1 of the paper) -- unchanged
# ---------------------------------------------------------------------------
FS = 50_000          # Hz, native PHM2010 sampling rate
N_SPEED = 10_400       # rpm
N_TEETH = 3
KPOOL = 4                # Sec 4.3
K_RECEPTIVE = 25          # Sec 4.3, paper's own reported PHM2010 k
AP = 0.2                    # mm, axial cutting depth (Table 1)
D_TOOL_MM = 6.0               # mm, ball-nose cutter diameter
BETA_HELIX_DEG = 30.0            # deg
LOWPASS_CUTOFF_HZ = 1733            # Sec 4.2
SAMPLE_LEN = 4608                    # Sec 4.2, explicit (16 cycles)

VST_PERIOD_L = FS * (60.0 / N_SPEED) / N_TEETH   # ~= 96.15, Eq.(6)
VST_RISE_FRACTION_P = (AP / math.tan(math.radians(BETA_HELIX_DEG))) / (
    (D_TOOL_MM * math.pi) / N_TEETH
)  # Eq.(5)


def raw_csv_path(condition: str, run_id: int) -> Path:
    cond_lower = condition.lower()
    num = cond_lower[1:]
    return PHM2010_ROOT / cond_lower / cond_lower / f"c_{num}_{run_id:03d}.csv"


def load_raw_fx(condition: str, run_id: int) -> np.ndarray:
    """Returns [N] float32 array: Fx only (column 0)."""
    path = raw_csv_path(condition, run_id)
    df = pd.read_csv(path, header=None, usecols=[0])
    return df.values[:, 0].astype(np.float32)


def lowpass_filter(x: np.ndarray, cutoff_hz: float = LOWPASS_CUTOFF_HZ, fs: float = FS,
                    order: int = 4) -> np.ndarray:
    """4th-order zero-phase Butterworth low-pass."""
    nyq = fs / 2.0
    wn = cutoff_hz / nyq
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x).astype(np.float32)


def build_vst(length: int = SAMPLE_LEN, period: float = VST_PERIOD_L,
              rise_fraction: float = VST_RISE_FRACTION_P) -> np.ndarray:
    """Eq.(5)-(6): periodic standard trend vector."""
    period_i = max(2, int(round(period)))
    rise_len = max(1, int(round(period_i * rise_fraction)))
    rise_len = min(rise_len, period_i)
    one_period = np.zeros(period_i, dtype=np.float32)
    one_period[:rise_len] = np.linspace(-1.0, 1.0, rise_len, dtype=np.float32)
    n_periods = int(math.ceil(length / period_i))
    tiled = np.tile(one_period, n_periods)[:length]
    return tiled


def build_unified_windows_cache(conditions=("C1", "C4", "C6"), windows_per_run: int = 8,
                                 seed: int = 42, max_runs: int | None = None,
                                 verbose: bool = True) -> pd.DataFrame:
    """`windows_per_run` 4608-length windows per run, spread across the run's
    low-pass-filtered Fx signal (valid start range split into
    `windows_per_run` equal segments, one random window per segment).

    `max_runs`: cap on run_id per condition, for fast smoke tests only —
    production runs pass max_runs=None (full 315-run universe).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for cond in conditions:
        max_run = max_runs or 315
        for run_id in range(1, max_run + 1):
            csv_path = raw_csv_path(cond, run_id)
            if not csv_path.exists():
                if verbose:
                    print(f"[build_unified_windows_cache] missing {csv_path}, stopping {cond} at run_id={run_id}")
                break
            sig = None
            for w in range(windows_per_run):
                out_path = CACHE_DIR / f"{cond}_{run_id:03d}_{w}.npy"
                if not out_path.exists():
                    if sig is None:
                        raw = load_raw_fx(cond, run_id)
                        sig = lowpass_filter(raw)
                    seg_len = (len(sig) - SAMPLE_LEN) // windows_per_run
                    if seg_len <= 0:
                        raise ValueError(f"{cond} run {run_id} too short for {windows_per_run} windows")
                    lo = w * seg_len
                    hi = lo + seg_len
                    start = int(rng.integers(lo, hi))
                    end = start + SAMPLE_LEN
                    np.save(out_path, sig[start:end])
                rows.append({"condition": cond, "run_id": run_id, "window_idx": w, "npy_path": str(out_path)})
    return pd.DataFrame(rows)


def load_window(npy_path: str) -> np.ndarray:
    return np.load(npy_path).astype(np.float32)

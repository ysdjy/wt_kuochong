# -*- coding: utf-8 -*-
r"""
DP2Net native preprocessing pipeline, reimplemented from
`baselines/dp2net/PAPER_SPEC.md` (Lai et al., MSSP 2024, DOI:
10.1016/j.ymssp.2024.111421).

Pipeline (see PAPER_SPEC.md for the full annotated table):

    raw Fx only  (PHM2010, archive/c{cond}/c{cond}/c_{cond}_{run:03d}.csv,
                  column 0; Fy,Fz,Vx,Vy,Vz,AE unused)         [Explicit]
      -> Fourier/Butterworth low-pass, cutoff=1733Hz, native 50kHz
         (paper text says "5kHz" -- Conflict #1, resolved to 50kHz)  [Explicit
                                                                       cutoff;
                                                                       Conflict
                                                                       on fs]
      -> uniform-random 4608-length window sampling, 2000/class,
         fixed seed, recorded in sample_manifest.csv               [Explicit
                                                                       length;
                                                                       Missing
                                                                       exact
                                                                       stride/
                                                                       position]
      -> Protocol A: paper-native 4-stage labels (I/II/III via a
         documented wear-rate-change-point proxy -- ref [41]'s exact
         rule is not recoverable, see PAPER_SPEC.md; IV via explicit
         mean-VB>0.3mm rule), C1=source (70/30 train/val), C4/C6=target
         test only                                                  [Mixed]
      -> Protocol B: DC-PSR unified E/M/L labels via data/label_utils.py
         (3-class, no stage IV)                                     [Unified]

S/G's physical-property kernel size k, Vst's period L, and P (rise
fraction) are computed here from PHM2010 process parameters (Table 1 of
the paper) and are consumed by model.py.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
ARCHIVE_DIR = PROJECT_ROOT / "archive"
DATA_DIR = THIS_DIR / "data"
WINDOW_DIR = DATA_DIR / "windows"

# ---------------------------------------------------------------------------
# Physical constants (PAPER_SPEC.md sec 1, Table 1 of the paper)
# ---------------------------------------------------------------------------
FS = 50_000          # Hz, native PHM2010 sampling rate (paper's "5kHz" is a
                        # documented text error, see PAPER_SPEC.md Conflict #1)
N_SPEED = 10_400       # rpm
N_TEETH = 3
KPOOL = 4                # Sec 4.3
K_RECEPTIVE = 25          # Sec 4.3, paper's own reported PHM2010 k
AP = 0.2                    # mm, axial cutting depth (Table 1)
D_TOOL_MM = 6.0               # mm, PHM2010-documented ball-nose cutter diameter
                                 # [Missing in this paper's own Table 1 for
                                 # PHM2010 -- assumed from PHM2010 challenge
                                 # documentation, see PAPER_SPEC.md]
BETA_HELIX_DEG = 30.0            # deg [Missing for PHM2010 -- placeholder
                                    # assumption, see PAPER_SPEC.md]
LOWPASS_CUTOFF_HZ = 1733            # Sec 4.2, explicit
SAMPLE_LEN = 4608                    # Sec 4.2, explicit (16 cycles)
SAMPLES_PER_CLASS = 2000              # Sec 4.2, explicit

# Eq.(6): tooth-contact period, in samples
VST_PERIOD_L = FS * (60.0 / N_SPEED) / N_TEETH   # ~= 96.15
# Eq.(5): fraction of each period that is the monotonic -1->1 rise
VST_RISE_FRACTION_P = (AP / math.tan(math.radians(BETA_HELIX_DEG))) / (
    (D_TOOL_MM * math.pi) / N_TEETH
)

STAGE4_NAMES = ["I", "II", "III", "IV"]
STAGE4_TO_ID = {s: i for i, s in enumerate(STAGE4_NAMES)}


def raw_csv_path(condition: str, run_id: int) -> Path:
    cond_lower = condition.lower()
    num = cond_lower[1:]
    return ARCHIVE_DIR / cond_lower / cond_lower / f"c_{num}_{run_id:03d}.csv"


def load_raw_fx(condition: str, run_id: int) -> np.ndarray:
    """Returns [N] float32 array: Fx only (column 0)."""
    path = raw_csv_path(condition, run_id)
    df = pd.read_csv(path, header=None, usecols=[0])
    return df.values[:, 0].astype(np.float32)


def lowpass_filter(x: np.ndarray, cutoff_hz: float = LOWPASS_CUTOFF_HZ, fs: float = FS,
                    order: int = 4) -> np.ndarray:
    """4th-order zero-phase Butterworth low-pass (order/type Missing in
    paper -- standard default, PAPER_SPEC.md sec 2)."""
    nyq = fs / 2.0
    wn = cutoff_hz / nyq
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x).astype(np.float32)


def build_vst(length: int = SAMPLE_LEN, period: float = VST_PERIOD_L,
              rise_fraction: float = VST_RISE_FRACTION_P) -> np.ndarray:
    """Eq.(5)-(6): periodic standard trend vector -- monotonic -1->1 rise
    for `rise_fraction` of each period, constant 0 for the rest."""
    period_i = max(2, int(round(period)))
    rise_len = max(1, int(round(period_i * rise_fraction)))
    rise_len = min(rise_len, period_i)
    one_period = np.zeros(period_i, dtype=np.float32)
    one_period[:rise_len] = np.linspace(-1.0, 1.0, rise_len, dtype=np.float32)
    n_periods = int(math.ceil(length / period_i))
    tiled = np.tile(one_period, n_periods)[:length]
    return tiled


def raw_wear_table(condition: str) -> pd.DataFrame:
    cond_lower = condition.lower()
    wdf = pd.read_csv(ARCHIVE_DIR / cond_lower / f"{cond_lower}_wear.csv")
    wdf.columns = [str(c).strip() for c in wdf.columns]
    return wdf


def assign_paper_native_4stage(condition: str) -> pd.DataFrame:
    """Protocol A stage labels: I/II/III/IV, paper-native VB=mean(...)
    convention. Stage IV is the paper's own explicit rule (mean VB>0.3mm).
    Stages I/II/III use a documented wear-rate-change-point PROXY --
    NOT a verified reproduction of ref [41]'s exact criterion (see
    PAPER_SPEC.md "Protocol A stage-boundary handling").
    """
    wdf = raw_wear_table(condition)
    # NOTE: archive/*_wear.csv flute_1/2/3 values are in MICROMETERS
    # (um) -- this project's PHM2010 archive convention, confirmed by
    # value range (~30-220), which is the standard PHM2010 flank-wear
    # unit used throughout the tool-wear-monitoring literature. The
    # paper's own "mean VB > 0.3 mm" rule is therefore applied here as
    # 300 (um), not 0.3.
    vb = wdf[["flute_1", "flute_2", "flute_3"]].mean(axis=1).values
    run_id = wdf["cut"].astype(int).values
    order = np.argsort(run_id)
    run_id, vb = run_id[order], vb[order]

    vb_smooth = pd.Series(vb).rolling(window=7, min_periods=1, center=True).mean().values
    rate = np.diff(vb_smooth, prepend=vb_smooth[0])
    rate_smooth = pd.Series(rate).rolling(window=5, min_periods=1, center=True).mean().values

    is_iv = vb > 300.0  # paper-explicit, GB/T 16460-2016, in um (see note above)
    non_iv_idx = np.where(~is_iv)[0]

    stage = np.full(len(vb), "IV", dtype=object)
    fallback_used = False
    if len(non_iv_idx) >= 6:
        steady_median = float(np.median(rate_smooth[non_iv_idx]))
        # Stage I ends at first index (after a short initial-transient
        # skip) where the rate has dropped to <= steady_median.
        i_end = non_iv_idx[-1]
        for idx in non_iv_idx[3:]:
            if rate_smooth[idx] <= steady_median:
                i_end = idx
                break
        # Stage III begins at first index after i_end where rate
        # exceeds 1.5x the steady median (accelerated-wear onset).
        iii_start = None
        for idx in non_iv_idx:
            if idx > i_end and rate_smooth[idx] >= 1.5 * steady_median:
                iii_start = idx
                break
        for idx in non_iv_idx:
            if idx <= i_end:
                stage[idx] = "I"
            elif iii_start is not None and idx >= iii_start:
                stage[idx] = "III"
            else:
                stage[idx] = "II"
    else:
        # Degenerate case (too few non-IV passes for change-point
        # detection): documented tertile fallback.
        fallback_used = True
        n = len(non_iv_idx)
        thirds = np.array_split(non_iv_idx, 3) if n > 0 else []
        for name, idxs in zip(["I", "II", "III"], thirds):
            for idx in idxs:
                stage[idx] = name

    return pd.DataFrame({
        "condition": condition, "run_id": run_id, "VB_mean": vb,
        "VB_mean_smooth": vb_smooth, "rate_smooth": rate_smooth,
        "stage4": stage, "stage4_id": [STAGE4_TO_ID[s] for s in stage],
        "boundary_proxy_fallback_used": fallback_used,
    })


def build_sample_manifest(conditions=("C1", "C4", "C6"), samples_per_class: int = SAMPLES_PER_CLASS,
                           seed: int = 42) -> pd.DataFrame:
    """Uniform-random 4608-length window sampling, `samples_per_class`
    windows per stage (I/II/III/IV), drawn from that condition's passes
    belonging to each class. PAPER_SPEC.md sec 1: exact stride/position
    rule is Missing in the paper; this fixed-seed uniform-random choice
    is documented and fully reproducible via this manifest.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cond in conditions:
        stages = assign_paper_native_4stage(cond)
        for stage_name in STAGE4_NAMES:
            runs_this_stage = stages[stages["stage4"] == stage_name]["run_id"].tolist()
            if not runs_this_stage:
                continue
            for i in range(samples_per_class):
                run_id = int(rng.choice(runs_this_stage))
                rows.append({
                    "condition": cond, "run_id": run_id, "class": stage_name,
                    "class_id": STAGE4_TO_ID[stage_name], "sample_idx": i,
                })
    manifest = pd.DataFrame(rows)
    return manifest


def build_windows_cache(conditions=("C1", "C4", "C6"), samples_per_class: int = SAMPLES_PER_CLASS,
                         seed: int = 42, verbose: bool = True) -> pd.DataFrame:
    """Reads each run's Fx once, low-pass filters it, then for every
    manifest row assigned to that run draws a fresh random 4608-length
    window (start offset uniform in [0, N-4608)) and caches it to
    data/windows/{cond}_{run:03d}_{k}.npy. Writes data/sample_manifest.csv
    with exact (condition, run_id, class, start_idx, end_idx) per
    task instruction #49.
    """
    WINDOW_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest = build_sample_manifest(conditions, samples_per_class, seed)

    filtered_cache: dict[tuple, np.ndarray] = {}
    out_rows = []
    per_run_counter: dict[tuple, int] = {}
    for cond in conditions:
        n_rows = (manifest["condition"] == cond).sum()
        if verbose:
            print(f"[build_windows_cache] {cond}: {n_rows} windows to extract")
    for row_i, row in manifest.iterrows():
        cond, run_id = row["condition"], int(row["run_id"])
        key = (cond, run_id)
        if key not in filtered_cache:
            raw = load_raw_fx(cond, run_id)
            filtered_cache[key] = lowpass_filter(raw)
            if len(filtered_cache) % 50 == 0 and verbose:
                print(f"[build_windows_cache] filtered {len(filtered_cache)} unique runs so far")
        sig = filtered_cache[key]
        max_start = len(sig) - SAMPLE_LEN
        if max_start <= 0:
            raise ValueError(f"{cond} run {run_id} too short ({len(sig)} pts) for a {SAMPLE_LEN}-length window")
        start = int(rng.integers(0, max_start))
        end = start + SAMPLE_LEN
        window = sig[start:end]
        idx_in_run = per_run_counter.get(key, 0)
        per_run_counter[key] = idx_in_run + 1
        out_path = WINDOW_DIR / f"{cond}_{run_id:03d}_{idx_in_run}.npy"
        np.save(out_path, window)
        out_rows.append({
            "condition": cond, "run_id": run_id, "class": row["class"],
            "class_id": row["class_id"], "start_idx": start, "end_idx": end,
            "npy_path": str(out_path.relative_to(THIS_DIR)),
        })
    result = pd.DataFrame(out_rows)
    result.to_csv(DATA_DIR / "sample_manifest.csv", index=False, encoding="utf-8-sig")
    return result


WINDOW_DIR_UNIFIED = DATA_DIR / "windows_unified"


def build_unified_windows_cache(conditions=("C1", "C4", "C6"), windows_per_run: int = 8,
                                 seed: int = 42, verbose: bool = True) -> pd.DataFrame:
    """Protocol B (Unified DC-PSR E/M/L, both B-S and B-D1 variants):
    extracts `windows_per_run` 4608-length windows per run, spread across
    the run's low-pass-filtered Fx signal (the valid start range is split
    into `windows_per_run` equal segments, one random window drawn from
    each -- same rationale as Dynamic GIN's 10-portion scheme: even
    temporal coverage per run without a heavy overlap-avoidance
    algorithm). Labels are NOT baked in here (this cache is shared by
    both B-S and B-D1 splits); train.py attaches condition+run_id ->
    stage_id via data/label_utils.py at train time.
    Writes data/unified_manifest.csv (condition, run_id, window_idx,
    start_idx, end_idx, npy_path).
    """
    WINDOW_DIR_UNIFIED.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for cond in conditions:
        max_run = 315
        for run_id in range(1, max_run + 1):
            csv_path = raw_csv_path(cond, run_id)
            if not csv_path.exists():
                break
            sig = None
            for w in range(windows_per_run):
                out_path = WINDOW_DIR_UNIFIED / f"{cond}_{run_id:03d}_{w}.npy"
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
                else:
                    start, end = -1, -1  # already cached, exact offsets not recomputed
                rows.append({"condition": cond, "run_id": run_id, "window_idx": w,
                             "start_idx": start, "end_idx": end,
                             "npy_path": str(out_path.relative_to(THIS_DIR))})
            if verbose and run_id % 100 == 0:
                print(f"[build_unified_windows_cache] {cond} run {run_id}/{max_run}")
    manifest = pd.DataFrame(rows)
    manifest.to_csv(DATA_DIR / "unified_manifest.csv", index=False, encoding="utf-8-sig")
    return manifest


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-per-class", type=int, default=SAMPLES_PER_CLASS)
    ap.add_argument("--quick", action="store_true", help="tiny run for smoke testing")
    args = ap.parse_args()
    spc = 5 if args.quick else args.samples_per_class
    print(f"VST_PERIOD_L (Eq.6) = {VST_PERIOD_L:.3f} samples/tooth-period")
    print(f"VST_RISE_FRACTION_P (Eq.5) = {VST_RISE_FRACTION_P:.4f}")
    print(f"k (Eq.4, receptive field) = {K_RECEPTIVE} (physics estimate: "
          f"{(FS*60/N_SPEED)/N_TEETH/KPOOL:.2f})")
    manifest = build_windows_cache(samples_per_class=spc)
    print(manifest.groupby(["condition", "class"]).size())
    print(f"Total windows: {len(manifest)}")

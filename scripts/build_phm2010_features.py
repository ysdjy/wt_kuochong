#!/usr/bin/env python
"""Fallback run-level feature extractor: reconstructs a
`run_level_features_all.csv`-compatible feature table from raw PHM2010
signals under data/PHM2010/raw/.

Vendored (with only path-routing changes -- no algorithm change) from the old
project's `baselines/htt_net/data/build_run_level_features.py`
(sha256 d63cfa7e4927fbbcdd5221bb102b2540a6937a70501a8a6e22032ff7e338c68a,
legacy commit 811da096ee47bea4f65db193aa49e793dba6f47d).

*** IMPORTANT: this is NOT a bit-identical reproduction of the actual
data/PHM2010/features/run_level_features_all.csv already committed in this
repo. *** That file's original extraction script was lost (see
data/README.md) -- this script computes a standard, defensible but DIFFERENT
pool of per-cut statistical features. B1-B4/B9 in THIS repo are already
wired to the authoritative committed CSV and do not need this script at all.

Use this script only as a fallback if:
  - the committed feature CSV is ever lost/corrupted and you need SOME
    feature table to keep working with while investigating, or
  - you are extending this framework to a new raw-signal dataset that has no
    pre-existing feature table, and want a starting-point extractor.

Any numbers computed against this script's output are only comparable to
other numbers computed against the SAME output -- never mix them with numbers
computed against the committed run_level_features_all.csv.

Usage:
    python scripts/build_phm2010_features.py
    python scripts/build_phm2010_features.py --raw-root data/PHM2010/raw --out data/PHM2010/features/reconstructed_v1.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "PHM2010" / "raw"
DEFAULT_OUT_PATH = REPO_ROOT / "data" / "PHM2010" / "features" / "reconstructed_v1_NOT_bit_identical.csv"

CONDITIONS = ["c1", "c4", "c6"]
COND_NAME_MAP = {"c1": "C1", "c4": "C4", "c6": "C6"}
CHANNEL_NAMES = ["Fx", "Fy", "Fz", "Vx", "Vy", "Vz", "AE"]  # standard PHM2010 7-channel order
N_RUNS = 315


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    p.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    return p.parse_args()


def channel_features(x: np.ndarray, prefix: str) -> dict:
    x = x.astype(np.float64)
    rms = float(np.sqrt(np.mean(x ** 2)))
    mx, mn = float(x.max()), float(x.min())
    crest = float(np.max(np.abs(x)) / (rms + 1e-12))
    return {
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_std": float(x.std()),
        f"{prefix}_rms": rms,
        f"{prefix}_max": mx,
        f"{prefix}_min": mn,
        f"{prefix}_ptp": mx - mn,
        f"{prefix}_skew": float(skew(x)),
        f"{prefix}_kurt": float(kurtosis(x)),
        f"{prefix}_crest": crest,
    }


def build_condition(raw_root: Path, cond: str) -> pd.DataFrame:
    cond_dir = raw_root / cond / cond
    wear_path = raw_root / cond / f"{cond}_wear.csv"
    wear = pd.read_csv(wear_path)
    wear["VB"] = wear[["flute_1", "flute_2", "flute_3"]].mean(axis=1)
    vb_by_cut = dict(zip(wear["cut"].astype(int), wear["VB"].astype(float)))

    rows = []
    t0 = time.time()
    for run_id in range(1, N_RUNS + 1):
        fpath = cond_dir / f"c_{cond[1]}_{run_id:03d}.csv"
        if not fpath.exists():
            continue
        sig = pd.read_csv(fpath, header=None, dtype="float32", engine="c").values
        if sig.shape[1] != 7:
            raise ValueError(f"{fpath} has {sig.shape[1]} columns, expected 7")
        row = {"condition": COND_NAME_MAP[cond], "run_id": run_id, "VB": vb_by_cut.get(run_id, np.nan)}
        for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
            row.update(channel_features(sig[:, ch_idx], ch_name))
        rows.append(row)
        if run_id % 50 == 0:
            print(f"  {cond}: {run_id}/{N_RUNS} cuts processed ({time.time()-t0:.1f}s elapsed)")
    print(f"{cond}: done, {len(rows)} runs, {time.time()-t0:.1f}s")
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    raw_root = Path(args.raw_root)
    out_path = Path(args.out)

    if not raw_root.exists():
        print(f"ERROR: raw PHM2010 data not found at {raw_root}. Run scripts/download_phm2010.py first.")
        return 1

    parts = [build_condition(raw_root, cond) for cond in CONDITIONS]
    out = pd.concat(parts, axis=0).sort_values(["condition", "run_id"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out)} rows, {out.shape[1]} columns to {out_path}")
    print(out.groupby("condition").size())
    print(
        "\nREMINDER: this is NOT bit-identical to the committed "
        "data/PHM2010/features/run_level_features_all.csv -- see this script's own docstring."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

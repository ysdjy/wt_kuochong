# -*- coding: utf-8 -*-
"""
Builds PHM2010_9methods_10seeds_comparison.csv from each method's already-
generated summary_PHM2010/<method>/dataset_mean_std.csv (dataset-level =
per-seed equal-weight average across D1/D2/D3, then mean+-std over the 10
seeds, ddof=1 -- see aggregate_dataset.py). Pure pivot/reshape, no
recomputation of the underlying statistics.

Usage:
    python build_phm2010_9methods_comparison.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SEED_STATS_DIR = Path(__file__).resolve().parent.parent
SUMMARY_DIR = SEED_STATS_DIR / "summary_PHM2010"

METHOD_FULL_NAMES = {
    "B1": "B1_RF", "B2": "B2_TCN_GRU", "B3": "B3_Multitask_TCN_GRU",
    "B4": "B4_HTT_Net", "B5": "B5_MultiSource_Attention", "B6": "B6_MTF_AViTK",
    "B7": "B7_Dynamic_GIN_TGP", "B8": "B8_DP2Net", "B9": "B9_DC_PHSR",
}

# (output column prefix, Metric value in dataset_mean_std.csv)
FIELDS = [("Acc", "Acc"), ("MacroF1", "Macro-F1"), ("MF1", "M-F1"),
          ("MRecall", "M-Recall"), ("Smooth", "Smooth")]


def main():
    rows = []
    for m, full_name in METHOD_FULL_NAMES.items():
        ms_path = SUMMARY_DIR / m / "dataset_mean_std.csv"
        if not ms_path.exists():
            print(f"WARNING: {ms_path} missing, skipping {m}")
            continue
        ms = pd.read_csv(ms_path).set_index("Metric")
        row = {"Method": full_name}
        for out_prefix, metric_name in FIELDS:
            if metric_name in ms.index:
                row[f"{out_prefix}_mean"] = ms.loc[metric_name, "Mean"]
                row[f"{out_prefix}_std"] = ms.loc[metric_name, "Std"]
            else:
                row[f"{out_prefix}_mean"] = float("nan")
                row[f"{out_prefix}_std"] = float("nan")
        # N (how many of the 10 seeds actually contributed) -- carried through
        # for transparency, not one of the required columns but cheap to add.
        row["N_seeds"] = int(ms.loc["Acc", "N"]) if "Acc" in ms.index else 0
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = SUMMARY_DIR / "PHM2010_9methods_10seeds_comparison.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

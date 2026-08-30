# -*- coding: utf-8 -*-
"""
Generic aggregation for a dataset's task landscape studies (B9 or B3),
producing per-task seed_level_results.csv + mean_std.csv, an
all_tasks_seed_level.csv, a dataset_seed_level.csv (per-seed equal-weight
average across the dataset's tasks) + dataset_mean_std.csv (mean/std over
seed0-100 of those dataset-level values, ddof=1), plus run_status.csv and
failed_runs.csv. Pure aggregation of already-completed per-seed results --
no retraining, no seed selection.

Usage:
    python aggregate_dataset.py --dataset PHM2010 --method B9 --tasks D1,D2,D3 \
        --landscape_prefix B9_PHM2010 --out_dir ../summary_PHM2010/B9
    python aggregate_dataset.py --dataset PHM2010 --method B3 --tasks D1,D2,D3 \
        --landscape_prefix B3_PHM2010 --out_dir ../summary_PHM2010/B3
    python aggregate_dataset.py --dataset NASA_Milling --method B9 --tasks N1,N2,N3,N4 \
        --landscape_prefix B9_NASA --out_dir ../summary_NASA/B9
    python aggregate_dataset.py --dataset NASA_Milling --method B3 --tasks N1,N2,N3,N4 \
        --landscape_prefix B3_NASA --out_dir ../summary_NASA/B3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED_STATS_DIR = Path(__file__).resolve().parent.parent

CORE_COLS = ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Precision", "M-Recall",
             "M_to_E", "M_to_L", "Rev", "Jump", "Smooth", "q-MAE", "q-RMSE", "q-R2", "Spearman"]
ID_COLS = ["Method", "Dataset", "Task", "Seed"]
ALL_COLS = ID_COLS + CORE_COLS

N_SEEDS_EXPECTED = 101  # 0..100 inclusive


def load_task(landscape_prefix: str, task: str, expect_method: str, expect_dataset: str):
    task_dir = SEED_STATS_DIR / f"{landscape_prefix}_{task}_seed_landscape"
    results_dir = task_dir / "results"
    rows = []
    status_rows = []
    failed_rows = []
    seeds_seen = set()
    for seed in range(N_SEEDS_EXPECTED):
        sd = results_dir / f"seed{seed}"
        done = (sd / "DONE.flag").exists()
        failed = (sd / "FAILED.flag").exists()
        metrics_path = sd / "metrics.csv"
        if done and metrics_path.exists():
            df = pd.read_csv(metrics_path)
            if len(df) != 1:
                status_rows.append({"Task": task, "Seed": seed, "status": "DUPLICATE_ROWS", "n_rows": len(df)})
                continue
            row = df.iloc[0].to_dict()
            out = {"Method": expect_method, "Dataset": expect_dataset, "Task": task, "Seed": seed}
            for c in CORE_COLS:
                out[c] = row.get(c, np.nan)
            rows.append(out)
            seeds_seen.add(seed)
            status_rows.append({"Task": task, "Seed": seed, "status": "DONE"})
        elif failed:
            status_rows.append({"Task": task, "Seed": seed, "status": "FAILED"})
            failed_rows.append({"Task": task, "Seed": seed, "reason": "FAILED.flag present"})
        else:
            status_rows.append({"Task": task, "Seed": seed, "status": "MISSING"})
            failed_rows.append({"Task": task, "Seed": seed, "reason": "no DONE.flag, no FAILED.flag (never ran)"})

    missing = set(range(N_SEEDS_EXPECTED)) - seeds_seen
    dup = len(seeds_seen) != len(rows)
    return pd.DataFrame(rows), pd.DataFrame(status_rows), pd.DataFrame(failed_rows), sorted(missing), dup


def mean_std_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in CORE_COLS:
        vals = pd.to_numeric(df[m], errors="coerce").values
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            rows.append({"Metric": m, "Mean": np.nan, "Std": np.nan, "Min": np.nan, "Max": np.nan, "N": 0})
            continue
        mean = float(np.mean(valid))
        std = float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan
        rows.append({"Metric": m, "Mean": mean, "Std": std, "Min": float(np.min(valid)), "Max": float(np.max(valid)), "N": len(valid)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", required=True, choices=["B9", "B3"])
    ap.add_argument("--tasks", required=True, help="comma list, e.g. D1,D2,D3")
    ap.add_argument("--landscape_prefix", required=True, help="e.g. B9_PHM2010 or B3_NASA")
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()

    tasks = args.tasks.split(",")
    method_full = "B9_DC_PHSR" if args.method == "B9" else "B3_Multitask_TCN_GRU"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_task_dfs = []
    all_status = []
    all_failed = []
    hash_report = {}
    for task in tasks:
        df, status_df, failed_df, missing, dup = load_task(args.landscape_prefix, task, method_full, args.dataset)
        df.to_csv(args.out_dir / f"{task}_seed_level_results.csv", index=False, encoding="utf-8-sig")
        mean_std_table(df).to_csv(args.out_dir / f"{task}_mean_std.csv", index=False, encoding="utf-8-sig")
        all_task_dfs.append(df)
        all_status.append(status_df.assign(Task=task))
        all_failed.append(failed_df)
        print(f"[{args.method}] {task}: {len(df)}/{N_SEEDS_EXPECTED} done, missing={missing}, duplicate_seed_rows={dup}")

        # hash consistency check across this task's seeds (feature/split/gmm/window hashes must be identical)
        rd = SEED_STATS_DIR / f"{args.landscape_prefix}_{task}_seed_landscape" / "results"
        hashes_seen = set()
        for seed_dir in sorted(rd.glob("seed*")):
            rm = seed_dir / "run_meta.json"
            if rm.exists():
                with open(rm, "r", encoding="utf-8") as f:
                    m = json.load(f)
                key = (m.get("feature_hash"), m.get("split_hash"), m.get("gmm_hash"), m.get("window_hash"))
                hashes_seen.add(key)
        hash_report[task] = {"n_distinct_hash_tuples": len(hashes_seen), "consistent": len(hashes_seen) <= 1}

    all_tasks_df = pd.concat(all_task_dfs, ignore_index=True) if all_task_dfs else pd.DataFrame(columns=ALL_COLS)
    all_tasks_df.to_csv(args.out_dir / "all_tasks_seed_level.csv", index=False, encoding="utf-8-sig")

    status_df = pd.concat(all_status, ignore_index=True) if all_status else pd.DataFrame()
    status_df.to_csv(args.out_dir / "run_status.csv", index=False, encoding="utf-8-sig")
    failed_df = pd.concat(all_failed, ignore_index=True) if all_failed else pd.DataFrame()
    failed_df.to_csv(args.out_dir / "failed_runs.csv", index=False, encoding="utf-8-sig")

    # dataset-level: for each seed present in ALL tasks, equal-weight average across tasks, then mean+-std over seeds
    pivot = {}
    for task, df in zip(tasks, all_task_dfs):
        pivot[task] = df.set_index("Seed")[CORE_COLS]
    common_seeds = set.intersection(*[set(p.index) for p in pivot.values()]) if pivot else set()
    common_seeds = sorted(common_seeds)
    dataset_rows = []
    for seed in common_seeds:
        row = {"Dataset": args.dataset, "Method": method_full, "Seed": seed}
        for m in CORE_COLS:
            vals = [pivot[task].loc[seed, m] for task in tasks]
            vals = [v for v in vals if pd.notna(v)]
            row[m] = float(np.mean(vals)) if vals else np.nan
        dataset_rows.append(row)
    dataset_seed_level = pd.DataFrame(dataset_rows)
    dataset_seed_level.to_csv(args.out_dir / "dataset_seed_level.csv", index=False, encoding="utf-8-sig")
    mean_std_table(dataset_seed_level).to_csv(args.out_dir / "dataset_mean_std.csv", index=False, encoding="utf-8-sig")

    with open(args.out_dir / "hash_consistency_report.json", "w", encoding="utf-8") as f:
        json.dump(hash_report, f, indent=2)

    print(f"\n[{args.method}] dataset-level ({args.dataset}): {len(common_seeds)}/{N_SEEDS_EXPECTED} seeds common to all tasks")
    print(f"hash consistency: {hash_report}")
    print(f"Wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()

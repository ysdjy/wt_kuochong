#!/usr/bin/env python
"""Builds the `results/PHM2010/Bx_xxx/summary/` files described in
RESULTS_POLICY.md from whatever DONE.flag-complete cells currently exist
under `results/PHM2010/Bx_xxx/{D1,D2,D3}/seedNNN/`. Safe to run at any point
during a sweep (partial results are fine -- summary reflects "as of now").

Dataset-level mean+-std (RESULTS_POLICY.md "Dataset-level aggregation"):
for each seed N that completed ALL THREE tasks, dataset_metric_seedN =
mean(D1_metric_seedN, D2_metric_seedN, D3_metric_seedN); then across every
such seed, mean and numpy.std(ddof=1) (uniform sample-std convention across
this repo).

Usage:
    python scripts/aggregate_results.py --method B9
    python scripts/aggregate_results.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "shared"))
from phm2010.tasks import TASK_NAMES  # noqa: E402
from runners.registry import discover_method_dirs  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--method", help="Method id, e.g. B9")
    g.add_argument("--all", action="store_true", help="Aggregate every discoverable method")
    p.add_argument("--results-root", default=str(REPO_ROOT / "results"))
    return p.parse_args()


def load_cell(method_dirname: str, task: str, results_root: Path) -> pd.DataFrame:
    rows = []
    task_dir = results_root / "PHM2010" / method_dirname / task
    if not task_dir.exists():
        return pd.DataFrame(rows)
    for seed_dir in sorted(task_dir.iterdir()):
        if not seed_dir.is_dir() or not (seed_dir / "DONE.flag").exists():
            continue
        metrics_path = seed_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        seed = int(seed_dir.name.replace("seed", ""))
        rows.append({"task": task, "seed": seed, **metrics})
    return pd.DataFrame(rows)


def run_status_table(method_dirname: str, results_root: Path) -> pd.DataFrame:
    rows = []
    for task in TASK_NAMES:
        task_dir = results_root / "PHM2010" / method_dirname / task
        if not task_dir.exists():
            continue
        for seed_dir in sorted(task_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            if (seed_dir / "DONE.flag").exists():
                status = "DONE"
            elif (seed_dir / "FAILED.flag").exists():
                status = "FAILED"
            else:
                status = "MISSING"
            seed = int(seed_dir.name.replace("seed", "")) if seed_dir.name.startswith("seed") else None
            rows.append({"task": task, "seed": seed, "status": status})
    return pd.DataFrame(rows)


def aggregate_method(method_id: str, results_root: Path) -> None:
    method_dirname = discover_method_dirs()[method_id].name
    summary_dir = results_root / "PHM2010" / method_dirname / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_rows = pd.concat([load_cell(method_dirname, t, results_root) for t in TASK_NAMES], ignore_index=True)
    if all_rows.empty:
        print(f"[{method_id}] no completed cells found -- nothing to aggregate.")
        return
    all_rows.to_csv(summary_dir / "all_tasks_seed_level.csv", index=False)

    metric_cols = [c for c in all_rows.columns if c not in ("task", "seed")]

    for task in TASK_NAMES:
        sub = all_rows[all_rows["task"] == task]
        if sub.empty:
            continue
        stats = {
            m: {"mean": float(np.mean(sub[m])), "std": float(np.std(sub[m], ddof=1)) if len(sub) > 1 else float("nan"), "n": len(sub)}
            for m in metric_cols
        }
        pd.DataFrame(stats).T.reset_index(names="metric").to_csv(summary_dir / f"{task}_mean_std.csv", index=False)

    # dataset-level: seeds that completed all 3 tasks (i.e. have a row for
    # D1/D2/D3 each -- NOT "every metric is non-NaN": q_MAE/q_RMSE/q_R2 are
    # legitimately NaN for methods with no continuous output, per policy, and
    # must not disqualify an otherwise-complete seed from dataset-level
    # aggregation).
    tasks_per_seed = all_rows.groupby("seed")["task"].apply(set)
    complete_seeds = sorted(s for s, tset in tasks_per_seed.items() if tset == set(TASK_NAMES))

    # dropna=False: pandas' pivot_table silently DROPS a (metric, task) column
    # that is all-NaN by default -- a metric that's legitimately NaN for every
    # task (e.g. q_MAE on a classification-only method) must still produce a
    # (present-but-NaN) column, not a missing one, or looking it up below raises.
    pivot = all_rows.pivot_table(index="seed", columns="task", values=metric_cols, aggfunc="first", dropna=False)
    dataset_rows = []
    for s in complete_seeds:
        row = {"seed": s}
        for m in metric_cols:
            # nanmean: a metric that's NaN for every task (e.g. q_MAE on a
            # classification-only method) stays NaN; a metric NaN on some but
            # not all tasks (shouldn't normally happen) still gives a value.
            row[m] = float(np.nanmean([pivot.loc[s, (m, t)] for t in TASK_NAMES]))
        dataset_rows.append(row)
    dataset_df = pd.DataFrame(dataset_rows)
    dataset_df.to_csv(summary_dir / "dataset_seed_level.csv", index=False)

    if not dataset_df.empty:
        stats = {
            m: {"mean": float(np.mean(dataset_df[m])),
                "std": float(np.std(dataset_df[m], ddof=1)) if len(dataset_df) > 1 else float("nan"),
                "n": len(dataset_df)}
            for m in metric_cols
        }
        pd.DataFrame(stats).T.reset_index(names="metric").to_csv(summary_dir / "dataset_mean_std.csv", index=False)

    status_df = run_status_table(method_dirname, results_root)
    status_df.to_csv(summary_dir / "run_status.csv", index=False)
    status_df[status_df["status"] == "FAILED"].to_csv(summary_dir / "failed_runs.csv", index=False)

    print(f"[{method_id}] aggregated: {len(all_rows)} task-seed rows, "
          f"{len(complete_seeds)} seeds with all 3 tasks complete -> {summary_dir}")


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    method_ids = [args.method] if args.method else sorted(discover_method_dirs().keys(), key=lambda x: int(x[1:]))
    for mid in method_ids:
        aggregate_method(mid, results_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fill in the missing B1-B10 baseline rows for (task, seed) units that were
already run for B11/B12+A1-A6 only (from the 'seeds'/'dual' phases). Uses the
merge-on-persist behaviour in ExperimentRunner.run_unit so the existing
A1-A6/B11/B12 rows are preserved, not retrained.

    python scripts/01b_complete_overall_baselines.py --out-root experiments_mendeley
"""
from __future__ import annotations
import argparse, time, traceback
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from dcpsr import config as C
from dcpsr.datasets.mendeley import MendeleyMachineToolWear
from dcpsr.online_features import build_online_features, online_feature_columns
from dcpsr.runner import ExperimentRunner
from dcpsr.stages import build_stage_labels

DUAL = ["D1-M", "D2-M", "D3-M"]
MISSING_METHODS = [f"B{i}" for i in range(1, 11)]  # B1..B10 (B11/B12/A* preserved by merge)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="experiments_mendeley")
    ap.add_argument("--channel-set", default="primary", choices=["primary", "all"])
    args = ap.parse_args()

    root = Path(args.out_root)
    ad = MendeleyMachineToolWear(".", root / "01_features", channel_set=args.channel_set)
    table = ad.load_run_level_table()
    labelled, _ = build_stage_labels(table)
    raw_cols = ad.feature_columns(table)
    online = build_online_features(labelled, raw_cols)
    cand = online_feature_columns(online)
    tasks = {t.name: t for t in ad.task_definitions()}

    runner = ExperimentRunner(labelled, online, cand, root / "04_overall_comparison",
                              args.channel_set, "mendeley")
    units = [(t, s) for t in DUAL for s in C.FINAL_SEEDS]
    failures = []
    for i, (tn, sd) in enumerate(units, 1):
        d = runner.unit_dir(tasks[tn], sd)
        have = set()
        if (d / "metrics.csv").exists():
            have = set(pd.read_csv(d / "metrics.csv")["Method"].unique())
        need = [m for m in MISSING_METHODS if m not in have]
        if not need:
            print(f"[{i}/{len(units)}] {tn} seed{sd}: already has {MISSING_METHODS} -- skip", flush=True)
            continue
        print(f"\n[{i}/{len(units)}] {tn} seed{sd}: adding {need}", flush=True)
        t0 = time.time()
        try:
            r = runner.run_unit(tasks[tn], sd, need, fusion_search=False, save_embeddings=False)
            print(f"    done in {time.time()-t0:.1f}s -- methods now on disk: "
                  f"{sorted(r['metrics'].Method.unique())}", flush=True)
        except Exception as exc:                                     # noqa: BLE001
            print(f"    [FAIL] {tn} seed{sd}: {exc}", flush=True)
            traceback.print_exc()
            failures.append(dict(task=tn, seed=sd, methods=",".join(need), error=str(exc)))
    if failures:
        pd.DataFrame(failures).to_csv(root / "10_logs" / "failed_runs_baselines.csv",
                                      index=False, encoding="utf-8-sig")
    print(f"\ncompleted, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

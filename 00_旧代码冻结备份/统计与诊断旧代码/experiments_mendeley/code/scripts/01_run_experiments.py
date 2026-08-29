#!/usr/bin/env python3
"""Phases 2-8: run every (task, method, seed) unit. Resumable.

    python scripts/01_run_experiments.py --out-root experiments_mendeley --phase sanity
    python scripts/01_run_experiments.py --out-root experiments_mendeley --phase all --resume

Phases
  seeds    D1-M only, B11+B12, 5 seeds        Stage 2 -- checks mean/std machinery
  dual     D1-M/D2-M/D3-M, B11+B12, 5 seeds    Stage 3
  overall  D1-M/D2-M/D3-M, B1..B12, 5 seeds    Stage 4a -- Experiment 1 (+A1-A6 free)
  general  9 cross-machine tasks, B9..B12      Stage 4b -- Experiment 2
  loto     9 leave-one-tool-out tasks          optional, not one of the four
  all      overall + general

Run scripts/03_sanity_check.py FIRST. Do not start any phase until it reports 10/10.
"""
from __future__ import annotations
import argparse, json, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from dcpsr import config as C
from dcpsr.datasets.mendeley import MendeleyMachineToolWear
from dcpsr.online_features import build_online_features, online_feature_columns
from dcpsr.runner import ExperimentRunner
from dcpsr.stages import build_stage_labels

ALL_B = [f"B{i}" for i in range(1, 13)]
DUAL = ["D1-M", "D2-M", "D3-M"]
SINGLE = [f"MS{i}" for i in range(1, 7)]
# Stage-gated phases. Each phase writes into its own sub-root so the sanity
# unit never mixes with the formal results.
PHASES = {
    # Stage 2: one task, 5 seeds -- confirms the mean/std machinery
    "seeds":   (DUAL[:1], ["B11", "B12"], None, "04_overall_comparison"),
    # Stage 3: three dual-source tasks, 5 seeds, B11/B12 (+A1-A6 free)
    "dual":    (DUAL, ["B11", "B12"], None, "04_overall_comparison"),
    # Stage 4a: Experiment 1 -- full B1..B12 on the three dual-source tasks
    "overall": (DUAL, ALL_B, None, "04_overall_comparison"),
    # Stage 4b: Experiment 2 -- cross-machine generalization, B9..B12 only
    "general": (DUAL + SINGLE, ["B9", "B10", "B11", "B12"], None, "05_generalization"),
    # optional: leave-one-tool-out (not part of the four formal experiments)
    "loto":    ([f"LOTO_T{i}" for i in range(1, 10)], ["B9", "B10", "B11", "B12"],
                None, "05_generalization"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="experiments_mendeley")
    ap.add_argument("--channel-set", default="primary", choices=["primary", "all"])
    ap.add_argument("--phase", default="seeds", choices=list(PHASES) + ["all"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seeds", default="", help="override, comma list")
    ap.add_argument("--tasks", default="", help="override, comma list")
    args = ap.parse_args()

    root = Path(args.out_root)
    ad = MendeleyMachineToolWear(".", root / "01_features", channel_set=args.channel_set)
    table = ad.load_run_level_table()
    labelled, _ = build_stage_labels(table)
    raw_cols = [c for c in ad.feature_columns(table)]
    print(f"{len(table)} runs, {len(raw_cols)} raw features -> building causal online features")
    online = build_online_features(labelled, raw_cols)
    cand = online_feature_columns(online)
    print(f"{len(cand)} online relative features")

    tasks = {t.name: t for t in ad.task_definitions()}
    (root / "02_protocols").mkdir(parents=True, exist_ok=True)
    (root / "02_protocols" / "task_definitions.json").write_text(
        json.dumps([t.to_dict() for t in tasks.values()], indent=2, ensure_ascii=False))

    phases = ["overall", "general"] if args.phase == "all" else [args.phase]
    failures, done = [], 0
    for ph in phases:
        tnames, methods, seeds, subroot = PHASES[ph]
        if args.tasks:
            tnames = args.tasks.split(",")
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else (seeds or C.FINAL_SEEDS)
        runner = ExperimentRunner(labelled, online, cand, root / subroot,
                                  args.channel_set, "mendeley")
        total = len(tnames) * len(seeds)
        print(f"\n{'='*70}\nphase {ph}  ->  {root/subroot}\n"
              f"{len(tnames)} tasks x {len(seeds)} seeds = {total} units\n{'='*70}")
        k = 0
        for ti, tn in enumerate(tnames, 1):
            for si, sd in enumerate(seeds, 1):
                k += 1
                t = tasks[tn]
                if args.resume and runner.is_done(t, sd):
                    print(f"[{k}/{total}] [Task {ti}/{len(tnames)}] {tn}  "
                          f"[Seed {si}/{len(seeds)}] {sd}  -- already done, skipping",
                          flush=True)
                    continue
                print(f"\n[{k}/{total}] [Task {ti}/{len(tnames)}] {tn}  "
                      f"[Seed {si}/{len(seeds)}] {sd}", flush=True)
                t0 = time.time()
                try:
                    r = runner.run_unit(t, sd, methods)
                    done += 1
                    mm = r["metrics"]
                    show = mm[mm.Method.isin(["B11", "B12"])]
                    if len(show):
                        for _, row in show.iterrows():
                            print(f"    {row['Method']}: Acc={row.get('Acc', float('nan')):.4f} "
                                  f"MacroF1={row.get('Macro_F1', float('nan')):.4f} "
                                  f"M-F1={row.get('M_F1', float('nan')):.4f} "
                                  f"Smooth={row.get('Smooth', float('nan')):.4f}", flush=True)
                    print(f"    done in {time.time()-t0:.1f}s", flush=True)
                except Exception as exc:                          # noqa: BLE001
                    print(f"    [FAIL] {tn} seed{sd}: {exc}", flush=True)
                    traceback.print_exc()
                    failures.append(dict(phase=ph, task=tn, seed=sd, method=",".join(methods),
                                         stage="run_unit", error=f"{type(exc).__name__}: {exc}",
                                         timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")))
    (root / "10_logs").mkdir(parents=True, exist_ok=True)
    if failures:
        pd.DataFrame(failures).to_csv(root / "10_logs" / "failed_runs.csv",
                                      index=False, encoding="utf-8-sig")
    print(f"\ncompleted {done} units, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

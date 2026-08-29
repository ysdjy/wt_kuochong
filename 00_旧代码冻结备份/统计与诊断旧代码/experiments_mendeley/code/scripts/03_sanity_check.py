#!/usr/bin/env python3
"""End-to-end sanity check: D1-M (M1+M2 -> M3), seed 42, B11/B12 + A1-A6.

    conda run -n dcpsr python scripts/03_sanity_check.py --out-root experiments_mendeley

Runs ONE unit, prints the full diagnostic block, then applies 10 pass criteria.
Exit code 0 only if all 10 pass. Nothing else is started from here.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from dcpsr import config as C
from dcpsr.datasets.mendeley import MendeleyMachineToolWear
from dcpsr.online_features import build_online_features, online_feature_columns
from dcpsr.runner import ExperimentRunner
from dcpsr.stages import build_stage_labels

BAR = "=" * 78


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="experiments_mendeley")
    ap.add_argument("--channel-set", default="primary", choices=["primary", "all"])
    ap.add_argument("--task", default="D1-M")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.out_root)
    sdir = root / "03_sanity_check"
    sdir.mkdir(parents=True, exist_ok=True)

    import torch
    print(BAR); print("SANITY CHECK"); print(BAR)
    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if not torch.cuda.is_available():
        print("WARNING: running on CPU. Fine for the sanity check; the full "
              "5-seed protocol should not be started on CPU.")

    ad = MendeleyMachineToolWear(".", root / "01_features", channel_set=args.channel_set)
    table = ad.load_run_level_table()
    labelled, summ = build_stage_labels(table)
    raw_cols = ad.feature_columns(table)
    print(f"\nrun-level table: {len(table)} runs, {table.sequence_id.nunique()} tools, "
          f"{table.domain_id.nunique()} machines, {len(raw_cols)} run-level features")
    print(summ[["sequence_id", "domain_id", "n", "VB_min", "VB_max",
                "early", "middle", "late"]].to_string(index=False))

    online = build_online_features(labelled, raw_cols)
    cand = online_feature_columns(online)
    print(f"\nonline relative features: {len(cand)}")

    tasks = {t.name: t for t in ad.task_definitions()}
    runner = ExperimentRunner(labelled, online, cand, sdir, args.channel_set, "mendeley")
    res = runner.run_unit(tasks[args.task], args.seed, ["B11", "B12"])
    d = runner.unit_dir(tasks[args.task], args.seed)

    m = res["metrics"].set_index("Method")
    pred = pd.read_csv(d / "predictions_test_B11B12.csv")
    diag = pd.read_csv(d / "split_diagnostics.csv")
    hist = pd.read_csv(d / "training_history.csv")
    cfg = json.loads((d / "config.json").read_text())

    # ---------------------------------------------------------------- report
    print("\n" + BAR); print("DIAGNOSTICS"); print(BAR)
    print(diag.to_string(index=False))
    print(f"\nselected features: {cfg['n_features_selected']} of {cfg['n_features_candidate']}")
    print(f"fusion params (tuned on validation only): {cfg['fusion_parameters']}")
    print(f"\nq_true  [{pred.q_true.min():.4f}, {pred.q_true.max():.4f}]")
    print(f"q_hat   [{pred.q_hat.min():.4f}, {pred.q_hat.max():.4f}]  std={pred.q_hat.std():.4f}")

    cols = ["Acc", "Macro_F1", "M_F1", "M_Rec", "M_to_E", "M_to_L", "Rev", "Jump",
            "Smooth", "q_MAE", "q_R2"]
    print("\n--- B11 / B12 ---")
    print(m.loc[[i for i in ("B11", "B12") if i in m.index],
                [c for c in cols if c in m.columns]].round(4).to_string())
    print("\n--- Ablation A1-A6 ---")
    ab = [i for i in ("A1", "A2", "A3", "A4", "A5", "A6") if i in m.index]
    print(m.loc[ab, [c for c in ("Acc", "Macro_F1", "M_F1", "Smooth") if c in m.columns]]
          .round(4).to_string())

    # ------------------------------------------------------------- criteria
    print("\n" + BAR); print("PASS CRITERIA"); print(BAR)
    checks: list[tuple[str, bool, str]] = []

    num = pred.select_dtypes("number")
    checks.append(("1  no NaN in predictions", not bool(num.isna().any().any()),
                   f"{int(num.isna().sum().sum())} NaN cells"))

    tl = hist["train_loss"].values
    fell = len(tl) > 2 and tl[-1] < tl[0]
    checks.append(("2  training loss decreased", bool(fell),
                   f"{tl[0]:.4f} -> {tl[-1]:.4f} over {len(tl)} epochs"))

    qs = float(pred.q_hat.std())
    checks.append(("3  q_hat has not collapsed", qs > 0.05, f"std(q_hat)={qs:.4f}"))

    npc = pred["stage_pred_final"].nunique()
    checks.append(("4  all three stages predicted", npc == 3,
                   f"{npc} distinct predicted stages: "
                   f"{sorted(pred.stage_pred_final_name.unique())}"))

    fp = pred[[f"final_prob_{s}" for s in C.STAGE_NAMES]].values
    checks.append(("5  p_final rows sum to 1", bool(np.allclose(fp.sum(1), 1.0, atol=1e-8)),
                   f"max |sum-1| = {np.abs(fp.sum(1)-1).max():.2e}"))

    ch = []
    for _, sub in pred.groupby("sequence_id"):
        v = sub.sort_values("order_key")[[f"final_prob_{s}" for s in C.STAGE_NAMES]].values
        ch.append(float(np.abs(np.diff(v, axis=0)).sum(1).mean()) if len(v) > 1 else 0.0)
    checks.append(("6  stage probability varies along the run", min(ch) > 1e-6,
                   f"min per-tool mean L1 change = {min(ch):.2e}"))

    frac_e = float((pred.stage_pred_final == 0).mean())
    checks.append(("7  ordered filtering did not lock everything into Early",
                   frac_e < 0.9, f"{frac_e:.1%} of test windows predicted Early"))

    leak_cols = {"VB", "VB_smooth", "q_true", "stage_id", "rate_norm", "fine_state_true"}
    used = set(cfg["selected_features"])
    bad = used & leak_cols
    also = {c for c in used if c.split("__")[0] in leak_cols}
    checks.append(("8  test VB/q never enters the network input",
                   not bad and not also,
                   f"offending features: {sorted(bad | also) or 'none'}"))

    tst = set(cfg["task"]["test_sequences"])
    fit_on = set(cfg["split"]["train_sequences"]) | set(cfg["split"]["validation_sequences"])
    checks.append(("9  feature selection / scaler / GMM / fusion never saw test",
                   not (tst & fit_on) and cfg["fusion_tuned_on"] == "internal validation split",
                   f"test tools {sorted(tst)} vs fitted-on {sorted(fit_on)}"))

    want = ["config.json", "metrics.json", "metrics.csv", "DONE.flag",
            "feature_selection.csv", "fusion_params.json", "gmm_fine_state_mapping.csv",
            "predictions_test_B11B12.csv", "split_diagnostics.csv", "training_history.csv"]
    miss = [f for f in want if not (d / f).exists()]
    chain = [f"{v}_prob_{s}" for v in ("raw", "fine", "prior", "mix", "ordered", "final")
             for s in C.STAGE_NAMES]
    miss_chain = [c for c in chain if c not in pred.columns]
    checks.append(("10 all outputs saved (incl. full probability chain)",
                   not miss and not miss_chain,
                   f"missing files {miss or 'none'}; missing prob columns {miss_chain or 'none'}"))

    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")
    n_ok = sum(1 for _, o, _ in checks if o)
    print(f"\n{n_ok}/10 criteria passed")

    rep = dict(task=args.task, seed=args.seed, channel_set=args.channel_set,
               torch=torch.__version__, cuda=torch.cuda.is_available(),
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               n_runs=int(len(table)), n_tools=int(table.sequence_id.nunique()),
               n_run_level_features=len(raw_cols), n_online_features=len(cand),
               n_selected=cfg["n_features_selected"],
               metrics={k: {c: float(v) for c, v in r.items() if isinstance(v, (int, float))}
                        for k, r in m[[c for c in cols if c in m.columns]].iterrows()},
               criteria={n: bool(o) for n, o, _ in checks}, passed=n_ok)
    (sdir / "sanity_report.json").write_text(json.dumps(rep, indent=2, default=str))
    res["metrics"].to_csv(sdir / "sanity_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"\nwrote {sdir/'sanity_report.json'}")
    if n_ok < 10:
        print("\nSANITY CHECK FAILED -- do not start the full protocol.")
        return 1
    print("\nSANITY CHECK PASSED. Report the numbers above before starting the 5-seed run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Collect every finished unit into the paper tables and the Fig.5-9 plot data.

    conda run -n dcpsr python scripts/02_aggregate_tables.py --out-root experiments_mendeley

Every '+/-' in the paper is generated here. Nothing is typed by hand.
Two standard deviations, never merged:
    *_std_seed  5 fixed seeds within one task
    *_std_task  across per-task means
"""
from __future__ import annotations
import argparse, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from dcpsr import config as C
from dcpsr.aggregate import (ablation_complete_table, by_seed, cross_task_summary,
                             mean_std_by_task, publication_table)
from dcpsr.metrics import Q_METRIC_COLS, confusion_rows

DUAL = ["D1-M", "D2-M", "D3-M"]
SINGLE = [f"MS{i}" for i in range(1, 7)]
SRC_DIRS = ["04_overall_comparison", "05_generalization", "03_sanity_check"]


def collect(root: Path, include_sanity=False) -> tuple[pd.DataFrame, list[Path]]:
    dirs = SRC_DIRS if include_sanity else SRC_DIRS[:2]
    paths = [p for d in dirs for p in sorted((root / d / "runs").rglob("metrics.csv"))]
    if not paths:
        raise SystemExit(f"no metrics.csv under {root} -- run 01_run_experiments.py first")
    return by_seed([pd.read_csv(p) for p in paths]), paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="experiments_mendeley")
    ap.add_argument("--include-sanity", action="store_true")
    args = ap.parse_args()
    root = Path(args.out_root)
    bs, _ = collect(root, args.include_sanity)
    ms = mean_std_by_task(bs)

    d4, d5, d6, d7, d8, dF = (root / "04_overall_comparison", root / "05_generalization",
                              root / "06_ablation", root / "07_semantic_consistency",
                              root / "08_plot_data", root / "FINAL_REPORT")
    for d in (d4 / "by_seed", d4 / "summary", d4 / "predictions",
              d5 / "dual_source", d5 / "single_source", d5 / "summary",
              d6 / "by_seed", d6 / "summary",
              d7 / "q_metrics", d7 / "stage_semantics", d7 / "embeddings",
              d7 / "probability_evolution", dF):
        d.mkdir(parents=True, exist_ok=True)

    # ---------------- Experiment 1: overall comparison ---------------------
    ov_bs = bs[bs.task.isin(DUAL) & bs.Method.str.fullmatch(r"B\d+")]
    ov_ms = ms[ms.task.isin(DUAL) & ms.Method.str.fullmatch(r"B\d+")]
    ov_bs.to_csv(d4 / "by_seed" / "overall_comparison_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    ov_ms.to_csv(d4 / "summary" / "overall_comparison_mean_std_by_task.csv", index=False, encoding="utf-8-sig")
    ov_ct = cross_task_summary(ov_ms)
    ov_ct.to_csv(d4 / "summary" / "overall_comparison_cross_task_summary.csv", index=False, encoding="utf-8-sig")

    # ---------------- Experiment 2: cross-machine generalization -----------
    gen_bs = bs[bs.task.isin(DUAL + SINGLE) & bs.Method.isin(["B9", "B10", "B11", "B12"])]
    gen_ms = ms[ms.task.isin(DUAL + SINGLE) & ms.Method.isin(["B9", "B10", "B11", "B12"])]
    gen_bs.to_csv(d5 / "generalization_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    gen_ms.to_csv(d5 / "generalization_mean_std_by_task.csv", index=False, encoding="utf-8-sig")
    gen_ms[gen_ms.task.isin(DUAL)].to_csv(d5 / "dual_source" / "dual_source_mean_std.csv",
                                          index=False, encoding="utf-8-sig")
    gen_ms[gen_ms.task.isin(SINGLE)].to_csv(d5 / "single_source" / "single_source_mean_std.csv",
                                            index=False, encoding="utf-8-sig")
    parts = []
    for label, sub in (("dual_source", gen_ms[gen_ms.task.isin(DUAL)]),
                       ("single_source", gen_ms[gen_ms.task.isin(SINGLE)])):
        if len(sub):
            s = cross_task_summary(sub); s["scenario"] = label; parts.append(s)
    gen_sum = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    gen_sum.to_csv(d5 / "summary" / "generalization_cross_task_summary.csv",
                   index=False, encoding="utf-8-sig")

    # ---------------- Experiment 3: ablation --------------------------------
    ab_bs = bs[bs.Method.str.fullmatch(r"A\d") & bs.task.isin(DUAL)]
    ab_ms = ms[ms.Method.str.fullmatch(r"A\d") & ms.task.isin(DUAL)]
    ab_bs.to_csv(d6 / "by_seed" / "ablation_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    ab_ms.to_csv(d6 / "summary" / "ablation_mean_std.csv", index=False, encoding="utf-8-sig")
    if len(ab_ms):
        ablation_complete_table(ab_ms).to_csv(d6 / "summary" / "ablation_complete_table.csv",
                                              index=False, encoding="utf-8-sig")

    # ---------------- Experiment 4: semantic consistency -------------------
    qc = [c for c in Q_METRIC_COLS if c in bs.columns]
    sem_bs = bs[bs.Method.isin(["B11", "B12"])][["dataset", "task", "group", "Method", "seed"] + qc]
    sem_bs.to_csv(d7 / "q_metrics" / "q_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    sem_ms = ms[ms.Method.isin(["B11", "B12"])]
    sem_ms.to_csv(d7 / "q_metrics" / "q_metrics_summary.csv", index=False, encoding="utf-8-sig")
    _stage_semantics(root, d7 / "stage_semantics")
    _gather(root, "predictions_test_B11B12.csv", d7 / "probability_evolution")
    _merge_embeddings(root, d7 / "embeddings")
    for d in (d4 / "predictions",):
        _gather(root, "predictions_test_*.csv", d)

    # ---------------- plot data --------------------------------------------
    _fig5(root, d8 / "new_fig5")
    _fig6(root, d8 / "new_fig6", ov_ms, ov_ct)
    _fig7(root, d8 / "new_fig7", gen_ms, gen_sum, d7)
    _fig8(root, d8 / "new_fig8", ab_ms, ab_bs)
    _fig9(root, d8 / "new_fig9", sem_ms, d7)

    # ---------------- markdown preview -------------------------------------
    L = ["# FINAL TABLES (preview -- the CSVs are the source of truth)", "",
         "`± ` after a per-task number is **std over the 5 seeds** (`std_seed`).",
         "`± ` in a cross-task row is **std across per-task means** (`std_task`).",
         "They are different quantities and are never pooled.", ""]
    if len(ov_ms):
        L += ["## Table B  Overall comparison, per task (± = std_seed)", "",
              publication_table(ov_ms).to_markdown(index=False), "",
              "## Table B'  Collapsed across the three dual-source tasks (± = std_task)", "",
              publication_table(ov_ct, std_kind="task").to_markdown(index=False), ""]
    if len(gen_sum):
        L += ["## Table C  Cross-machine generalization (± = std_task)", "",
              publication_table(gen_sum, std_kind="task").to_markdown(index=False), ""]
    if len(ab_ms):
        L += ["## Table D  Ablation A1-A6 (± = std_seed)", "",
              ablation_complete_table(ab_ms).filter(
                  regex="^(Method|configuration|task|.*_display)$").to_markdown(index=False), ""]
    if len(sem_ms):
        L += ["## Table E  Degradation semantic consistency (± = std_seed)", "",
              publication_table(sem_ms, metrics=tuple(qc)).to_markdown(index=False), ""]
    (dF / "FINAL_TABLES.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote tables under", root)
    if len(ov_ct):
        print(publication_table(ov_ct, std_kind="task").to_string(index=False))
    return 0


# ------------------------------------------------------------------ helpers
def _runs(root: Path):
    for d in SRC_DIRS:
        yield from sorted((root / d / "runs").rglob("*")) if (root / d / "runs").exists() else []


def _find(root: Path, pattern: str):
    out = []
    for d in SRC_DIRS:
        rd = root / d / "runs"
        if rd.exists():
            out += sorted(rd.rglob(pattern))
    return out


def _tag(p: Path) -> str:
    return f"{p.parents[1].name}_{p.parent.name}"


def _gather(root: Path, pattern: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for p in _find(root, pattern):
        shutil.copy2(p, dest / f"{_tag(p)}_{p.name}")


def _merge_embeddings(root: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    frames = []
    for p in _find(root, "representation_embeddings.parquet") + _find(root, "representation_embeddings.csv"):
        frames.append(pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p))
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    try:
        df.to_parquet(dest / "representation_embeddings.parquet", index=False)
    except Exception:                                             # noqa: BLE001
        df.to_csv(dest / "representation_embeddings.csv", index=False, encoding="utf-8-sig")


def _stage_semantics(root: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in _find(root, "predictions_test_B11B12.csv"):
        df = pd.read_csv(p)
        task, seed = p.parents[1].name, int(p.parent.name.replace("seed", ""))
        for st, sub in df.groupby("stage_pred_final_name"):
            rows.append(dict(task=task, seed=seed, predicted_stage=st, n_samples=len(sub),
                             mean_q_true=sub.q_true.mean(), std_q_true=sub.q_true.std(ddof=1),
                             mean_q_hat=sub.q_hat.mean(), std_q_hat=sub.q_hat.std(ddof=1),
                             mean_VB=sub.VB.mean(), std_VB=sub.VB.std(ddof=1)))
    if not rows:
        return
    d = pd.DataFrame(rows)
    d.to_csv(dest / "stage_semantics_by_seed.csv", index=False, encoding="utf-8-sig")
    order = {"early": 0, "middle": 1, "late": 2}
    g = (d.groupby("predicted_stage")[["mean_q_true", "mean_q_hat", "mean_VB", "n_samples"]]
           .agg(["mean", "std"]).reset_index())
    g["_o"] = g["predicted_stage"].map(order)
    g = g.sort_values("_o").drop(columns="_o")
    g.to_csv(dest / "stage_semantics_summary.csv", index=False, encoding="utf-8-sig")
    # explicit Early < Middle < Late check on the pooled means
    mean_vb = d.groupby("predicted_stage")["mean_VB"].mean()
    ok = all(mean_vb.get(a, np.nan) < mean_vb.get(b, np.nan)
             for a, b in (("early", "middle"), ("middle", "late")))
    (dest / "monotonic_wear_semantics.txt").write_text(
        f"mean VB by predicted stage:\n{mean_vb.to_string()}\n"
        f"Early < Middle < Late : {ok}\n", encoding="utf-8")


def _fig5(root: Path, d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    a = root / "00_dataset_audit"
    for f, pref in (("dataset_quality_report.csv", "panelA_"),
                    ("stage_coverage_by_tool.csv", "panelA_"),
                    ("channel_summary.csv", "panelA_"),
                    ("per_run_stage_preview.csv", "panelB_")):
        if (a / f).exists():
            shutil.copy2(a / f, d / (pref + f))
    p = root / "02_protocols" / "task_definitions.json"
    if p.exists():
        shutil.copy2(p, d / "panelC_task_definitions.json")


def _fig6(root: Path, d: Path, ms, ct) -> None:
    d.mkdir(parents=True, exist_ok=True)
    ms.to_csv(d / "panelA_overall_mean_std.csv", index=False, encoding="utf-8-sig")
    ct.to_csv(d / "panelA_cross_task.csv", index=False, encoding="utf-8-sig")
    cm = []
    for p in _find(root, "predictions_test_*.csv"):
        mid = p.stem.replace("predictions_test_", "")
        df = pd.read_csv(p)
        col = "stage_pred_final" if mid == "B11B12" else f"stage_pred_{mid.lower()}"
        if col not in df.columns:
            continue
        r = confusion_rows(df, col)
        r["Method"] = "B12" if mid == "B11B12" else mid
        r["task"], r["seed"] = p.parents[1].name, p.parent.name
        cm.append(r)
        if mid == "B11B12":
            r2 = confusion_rows(df, "stage_pred_raw"); r2["Method"] = "B11"
            r2["task"], r2["seed"] = p.parents[1].name, p.parent.name
            cm.append(r2)
    if cm:
        pd.concat(cm, ignore_index=True).to_csv(d / "panelB_confusion_matrices.csv",
                                                index=False, encoding="utf-8-sig")
    k = [c for c in ms.columns if c.startswith(("M_F1", "M_Rec", "M_Pre", "M_to_E", "M_to_L"))]
    ms[["task", "Method"] + k].to_csv(d / "panelC_middle_stage.csv", index=False, encoding="utf-8-sig")
    k = [c for c in ms.columns if c.startswith(("Rev", "Jump", "Smooth"))]
    ms[["task", "Method"] + k].to_csv(d / "panelD_consistency.csv", index=False, encoding="utf-8-sig")


def _fig7(root: Path, d: Path, gen_ms, gen_sum, d7: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    gen_ms.to_csv(d / "panelA_machine_generalization_source.csv", index=False, encoding="utf-8-sig")
    gen_sum.to_csv(d / "panelC_cross_task_variability.csv", index=False, encoding="utf-8-sig")
    gen_ms.to_csv(d / "panelD_radar_source.csv", index=False, encoding="utf-8-sig")
    src = d7 / "probability_evolution"
    if src.exists():
        shutil.copytree(src, d / "panelB_probability_evolution", dirs_exist_ok=True)


def _fig8(root: Path, d: Path, ab_ms, ab_bs) -> None:
    d.mkdir(parents=True, exist_ok=True)
    ab_ms.to_csv(d / "ablation_task_level.csv", index=False, encoding="utf-8-sig")
    ab_bs.to_csv(d / "ablation_seed_level.csv", index=False, encoding="utf-8-sig")
    cols = [c for c in ("task", "Method", "Smooth_mean", "Smooth_std_seed",
                        "Acc_mean", "Acc_std_seed") if c in ab_ms.columns]
    if len(ab_ms) and len(cols) > 2:
        ab_ms[cols].to_csv(d / "tradeoff_smooth_vs_acc.csv", index=False, encoding="utf-8-sig")


def _fig9(root: Path, d: Path, sem_ms, d7: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    sem_ms.to_csv(d / "panelD_semantic_statistics.csv", index=False, encoding="utf-8-sig")
    for name in ("embeddings", "probability_evolution", "stage_semantics"):
        src = d7 / name
        if src.exists():
            shutil.copytree(src, d / name, dirs_exist_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

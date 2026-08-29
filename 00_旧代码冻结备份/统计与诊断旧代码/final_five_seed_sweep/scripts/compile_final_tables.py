# -*- coding: utf-8 -*-
"""
Compiles the final 9-method 5-seed sweep tables:
  - final_five_seed_sweep/results/FINAL_9_METHODS_SEED_LEVEL.csv
  - final_five_seed_sweep/results/FINAL_9_METHODS_5SEED.csv (mean+-std, ddof=1)

Reads directly from the real per-seed output files (no numbers hand-typed).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [42, 52, 62, 72, 82]

METRIC_COLS = ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Pre", "M-Rec",
               "M→E", "M→L", "Rev", "Jump", "Smooth"]

rows = []  # seed-level rows


def add_row(method, category, seed, metrics: dict, source_file: str):
    row = {"Category": category, "Method": method, "seed": seed, "source_file": source_file}
    for c in METRIC_COLS:
        row[c] = metrics.get(c, np.nan)
    for c in ["q-MAE", "q-RMSE", "q-R2"]:
        row[c] = metrics.get(c, np.nan)
    rows.append(row)


def read_generic_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="gbk")
    canonical = ["Method", "Method_name", "Stage_definition", "Model_type",
                 "Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Pre", "M-Rec",
                 "M→E", "M→L", "Rev", "Jump", "Smooth"]
    assert len(df.columns) == len(canonical), (path, list(df.columns))
    df.columns = canonical
    return df


# --- 1. Generic baselines: RF (B5), TCN-GRU (B10), Multi-task TCN-GRU (B11), DC-PSR (B12/FGDS-PSI) ---
GENERIC_SEED42_CSV = ROOT / "补充材料" / "小论文" / "4_comparison_experiment_recheck" / "1_results" / "FINAL_comparison_results.csv"
GENERIC_OTHER_CSV = ROOT / "final_five_seed_sweep" / "results" / "generic_baselines" / "seed{seed}" / "1_results" / "FINAL_comparison_results.csv"

GENERIC_METHODS = {
    "B5": ("RF", "generic_internal_baseline"),
    "B10": ("TCN-GRU", "generic_internal_baseline"),
    "B11": ("Multi-task TCN-GRU", "generic_internal_baseline"),
    "B12": ("DC-PSR", "proposed"),
}

for seed in SEEDS:
    path = GENERIC_SEED42_CSV if seed == 42 else Path(str(GENERIC_OTHER_CSV).format(seed=seed))
    df = read_generic_csv(path)
    for bid, (name, cat) in GENERIC_METHODS.items():
        r = df[df["Method"] == bid].iloc[0]
        metrics = {c: r[c] for c in METRIC_COLS}
        add_row(name, cat, seed, metrics, str(path.relative_to(ROOT)))

# --- 2. HTT-Net (adapted) ---
for seed in SEEDS:
    if seed == 42:
        path = ROOT / "outputs" / "htt_net" / "D1_C1C4_to_C6_SOURCE_ONLY_TUNED" / "metrics.json"
    else:
        path = ROOT / "outputs" / "htt_net" / f"D1_C1C4_to_C6_SOURCE_ONLY_TUNED_seed{seed}" / "metrics.json"
    d = json.load(open(path, encoding="utf-8"))
    add_row("HTT-Net (adapted)", "published_method", seed, d, str(path.relative_to(ROOT)))

# --- 3. Multi-source Attention ---
for seed in SEEDS:
    if seed == 42:
        path = ROOT / "outputs" / "multi_source_attention" / "unified_protocol" / "metrics.json"
    else:
        path = ROOT / "outputs" / "multi_source_attention" / "seed_sweep" / f"seed{seed}" / "unified_protocol" / "metrics.json"
    d = json.load(open(path, encoding="utf-8"))
    add_row("Multi-source Attention", "published_method", seed, d, str(path.relative_to(ROOT)))

# --- 4. MTF-AViTK (all 5 seeds freshly rerun under seed_sweep/) ---
for seed in SEEDS:
    path = ROOT / "outputs" / "mtf_avitk" / "seed_sweep" / f"seed{seed}" / "unified_protocol" / "metrics.json"
    d = json.load(open(path, encoding="utf-8"))
    add_row("MTF-AViTK", "published_method", seed, d, str(path.relative_to(ROOT)))

# --- 5. Dynamic GIN + TGP ---
for seed in SEEDS:
    path = ROOT / "outputs" / "dynamic_gin_tgp" / "unified_protocol" / f"seed{seed}" / "metrics.json"
    d = json.load(open(path, encoding="utf-8"))
    add_row("Dynamic GIN + TGP", "published_method", seed, d, str(path.relative_to(ROOT)))

# --- 6. DP2Net-adapted (Protocol B-D1 only) ---
for seed in SEEDS:
    path = ROOT / "outputs" / "dp2net" / "unified_protocol_B-D1" / f"seed{seed}" / "metrics.json"
    d = json.load(open(path, encoding="utf-8"))
    add_row("DP2Net-adapted", "published_method", seed, d, str(path.relative_to(ROOT)))

seed_df = pd.DataFrame(rows)
seed_df = seed_df.rename(columns={"M→E": "M_to_E", "M→L": "M_to_L", "M-Pre": "M-Precision", "M-Rec": "M-Recall"})

OUT_DIR = ROOT / "final_five_seed_sweep" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
seed_level_path = OUT_DIR / "FINAL_9_METHODS_SEED_LEVEL.csv"
seed_df.to_csv(seed_level_path, index=False, encoding="utf-8-sig")
print(f"Wrote {seed_level_path} ({len(seed_df)} rows)")

# --- Aggregate mean +- std (ddof=1) ---
METRIC_COLS_OUT = ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Precision", "M-Recall",
                    "M_to_E", "M_to_L", "Rev", "Jump", "Smooth"]

PARAMS = {
    "RF": "N/A (tree ensemble, n_estimators=400)",
    "TCN-GRU": "83,619",
    "Multi-task TCN-GRU": "84,009",
    "DC-PSR": "84,009 (shares Multi-task TCN-GRU network; deterministic post-processing adds 0 learned params)",
    "HTT-Net (adapted)": "3,502,723",
    "Multi-source Attention": "12,939,050",
    "MTF-AViTK": "309,371,072",
    "Dynamic GIN + TGP": "321,950",
    "DP2Net-adapted": "60,956 total (S+G+F); 60,031 at inference (G is training-only)",
}

PROTOCOL_NOTE = {
    "RF": "C1+C4->C6 (D1); windowed L=12 universe, 304 runs",
    "TCN-GRU": "C1+C4->C6 (D1); windowed L=12 universe, 304 runs",
    "Multi-task TCN-GRU": "C1+C4->C6 (D1); windowed L=12 universe, 304 runs",
    "DC-PSR": "C1+C4->C6 (D1); windowed L=12 universe, 304 runs; frozen B12_PARAMS post-processing on Multi-task TCN-GRU output",
    "HTT-Net (adapted)": "C1+C4->C6 (D1); SOURCE_ONLY_TUNED frozen config; windowed L=12 universe, 304 runs",
    "Multi-source Attention": "C1+C4->C6 (D1), unified Protocol B; native run-level universe, 315 runs",
    "MTF-AViTK": "C1+C4->C6 (D1), unified Protocol B; native run-level universe, 315 runs; ViT-L/32",
    "Dynamic GIN + TGP": "C1+C4->C6 (D1), unified Protocol B; native run-level universe, 315 runs; post-leakage-fix",
    "DP2Net-adapted": "C1+C4->C6 (D1), Protocol B-D1 pooled-source; native run-level universe, 315 runs",
}

REPRO_NOTE = {
    "RF": "internal/generic baseline, authoritative feature file",
    "TCN-GRU": "internal/generic baseline, authoritative feature file",
    "Multi-task TCN-GRU": "internal/generic baseline, DC-PSR's direct predecessor",
    "DC-PSR": "proposed method (main.tex headline result)",
    "HTT-Net (adapted)": "adapted from original long-sequence L~=2000 regime to unified run-level protocol",
    "Multi-source Attention": "paper omitted key CWT parameters; adapted reimplementation",
    "MTF-AViTK": "close original-protocol reproduction; unified protocol adapted to authoritative labels",
    "Dynamic GIN + TGP": "high-fidelity reproduction; original D1 closely matched paper",
    "DP2Net-adapted": "pooled-source adaptation for D1; original paper is single-source DG",
}

CATEGORY = {
    "RF": "Generic/internal baseline", "TCN-GRU": "Generic/internal baseline",
    "Multi-task TCN-GRU": "Generic/internal baseline", "DC-PSR": "Proposed",
    "HTT-Net (adapted)": "Published method", "Multi-source Attention": "Published method",
    "MTF-AViTK": "Published method", "Dynamic GIN + TGP": "Published method",
    "DP2Net-adapted": "Published method",
}

METHOD_ORDER = ["RF", "TCN-GRU", "Multi-task TCN-GRU", "HTT-Net (adapted)",
                "Multi-source Attention", "MTF-AViTK", "Dynamic GIN + TGP",
                "DP2Net-adapted", "DC-PSR"]

agg_rows = []
for method in METHOD_ORDER:
    sub = seed_df[seed_df["Method"] == method]
    assert len(sub) == 5, f"{method} has {len(sub)} seeds, expected 5"
    row = {"Category": CATEGORY[method], "Method": method}
    for c in METRIC_COLS_OUT:
        vals = sub[c].astype(float).values
        row[f"{c.replace('-', '_')}_mean"] = np.mean(vals)
        row[f"{c.replace('-', '_')}_std"] = np.std(vals, ddof=1)
    row["Params"] = PARAMS[method]
    row["Protocol_note"] = PROTOCOL_NOTE[method]
    row["Reproduction_note"] = REPRO_NOTE[method]
    agg_rows.append(row)

agg_df = pd.DataFrame(agg_rows)
agg_path = OUT_DIR / "FINAL_9_METHODS_5SEED.csv"
agg_df.to_csv(agg_path, index=False, encoding="utf-8-sig")
print(f"Wrote {agg_path} ({len(agg_df)} rows)")

# Print a quick human-readable summary
pd.set_option("display.width", 200)
print(agg_df[["Method", "Acc_mean", "Acc_std", "Macro-F1_mean".replace("-", "_"), "Macro_F1_std"]])

# -*- coding: utf-8 -*-
r"""
Condition-relative Early/Middle/Late stage labeling + D1/D2/D3 train/val/test
split, vendored so B7 no longer needs a live import of the old project's
`代码/main_experiment_3_fgds_psi_optimized.py` (self-containment requirement,
task spec section 38).

Vendored from two old-project sources, both read-only, at commit
811da096ee47bea4f65db193aa49e793dba6f47d:
  - `define_condition_relative_stages` / STAGE_NAMES / STAGE_TO_ID / ID_TO_STAGE /
    VAL_RATIO_STAGE / MIN_STAGE_VAL_LEN: `代码/main_experiment_3_fgds_psi_optimized.py`
    (verbatim formula/thresholds; only change: the diagnostic CSV side-effect
    write to a hardcoded DIR_RESULT is removed — this function is now pure).
  - Generalized (task-parameterized) split, replacing the original's
    hardcoded train={C1,C4}/test={C6}: ported from
    `final_statistical_evidence/scripts/methods/condition_split.py::make_split_fn`,
    itself already a validated generalization of the same original's
    `split_grouped_lifecycle` used for this project's already-frozen D2/D3
    results (see `final_statistical_evidence/results/TRANSFER_TASKS_MEAN_STD.csv`).

This exact labeling/split logic is intentionally duplicated verbatim (not
imported cross-method) in every internal method's `code/stage_labels.py` —
per task spec section 20, every one of the 9 methods MUST use identical stage
ground truth, so byte-identical vendored copies (checked by sha256 in each
method's source_manifest.json) are safer here than a shared import that could
accidentally diverge silently. See also this fork's parent-facing report for
a suggestion to eventually promote this into `shared/phm2010/` once all 9
methods' copies are confirmed identical.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parents[1]
PHM2010_ROOT = Path(os.environ.get("PHM2010_ROOT", str(REPO_ROOT / "data" / "PHM2010" / "raw")))

STAGE_NAMES = ["early", "middle", "late"]
STAGE_TO_ID = {"early": 0, "middle": 1, "late": 2}
ID_TO_STAGE = {0: "early", 1: "middle", 2: "late"}

Q_EARLY = 0.30
Q_LATE = 0.72
RATE_LATE_Q = 0.78

VAL_RATIO_STAGE = 0.20
MIN_STAGE_VAL_LEN = 8


def load_vb_table(conditions=("C1", "C4", "C6")) -> pd.DataFrame:
    """Build a (condition, run_id, VB) table from the raw PHM2010 wear CSVs,
    using VB = max(flute_1, flute_2, flute_3) (this project's confirmed
    real-data convention)."""
    rows = []
    for cond in conditions:
        cond_lower = cond.lower()
        wear_path = PHM2010_ROOT / cond_lower / f"{cond_lower}_wear.csv"
        wdf = pd.read_csv(wear_path)
        wdf.columns = [str(c).strip() for c in wdf.columns]
        vb = wdf[["flute_1", "flute_2", "flute_3"]].max(axis=1)
        for cut, v in zip(wdf["cut"].astype(int), vb):
            rows.append({"condition": cond, "run_id": int(cut), "VB": float(v)})
    return pd.DataFrame(rows)


def define_condition_relative_stages(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verbatim port of 代码/main_experiment_3_fgds_psi_optimized.py's function
    of the same name — only change: no longer writes a diagnostic CSV to a
    hardcoded results directory (pure function now); returns the same
    (labeled_df, thresholds_df) tuple."""
    parts, rows = [], []
    for cond, sub in df.groupby("condition"):
        sub = sub.sort_values("run_id").reset_index(drop=True).copy()
        vb_smooth = sub["VB"].rolling(window=7, min_periods=1, center=True).mean()
        q = (vb_smooth - vb_smooth.min()) / (vb_smooth.max() - vb_smooth.min() + 1e-12)
        rate = q.diff().fillna(0.0).rolling(window=5, min_periods=1, center=True).mean()
        rate_norm = (rate - rate.min()) / (rate.max() - rate.min() + 1e-12)
        th_e = float(q.quantile(Q_EARLY))
        th_l = float(q.quantile(Q_LATE))
        th_v = float(rate_norm.quantile(RATE_LATE_Q))
        stage = np.where(q <= th_e, "early", np.where((q >= th_l) | (rate_norm >= th_v), "late", "middle"))
        sub["VB_smooth"] = vb_smooth.values
        sub["q_true"] = q.values
        sub["rate_norm"] = rate_norm.values
        sub["stage"] = stage
        sub["stage_id"] = sub["stage"].map(STAGE_TO_ID).astype(int)
        cnt = sub["stage"].value_counts().reindex(STAGE_NAMES, fill_value=0)
        rows.append({
            "condition": cond, "theta_E": th_e, "theta_L": th_l, "theta_v": th_v,
            "VB_min": float(sub["VB"].min()), "VB_max": float(sub["VB"].max()),
            "early": int(cnt["early"]), "middle": int(cnt["middle"]), "late": int(cnt["late"]),
        })
        parts.append(sub)
    out = pd.concat(parts, axis=0).sort_values(["condition", "run_id"]).reset_index(drop=True)
    th_df = pd.DataFrame(rows)
    return out, th_df


def get_condition_relative_labels(conditions=("C1", "C4", "C6")) -> pd.DataFrame:
    df = load_vb_table(conditions)
    labeled, _thresholds = define_condition_relative_stages(df)
    return labeled


def make_split_fn(train_conditions: list[str], test_condition: str) -> Callable[[pd.DataFrame], tuple]:
    """Stage-stratified centered-slice val carve out of train_conditions;
    test_condition held out entirely. Generalizes the original's hardcoded
    train={C1,C4}/test={C6} to any (train_conditions, test_condition) pair —
    ported from condition_split.py::make_split_fn, itself already validated
    for this project's D2/D3 results."""
    def split_fn(df: pd.DataFrame):
        if set(train_conditions) & {test_condition}:
            raise ValueError(f"Leakage: train {train_conditions} overlaps test {test_condition}.")
        train_parts, val_parts = [], []
        for cond in train_conditions:
            sub = df[df["condition"] == cond].sort_values("run_id").reset_index(drop=True).copy()
            val_idx = []
            for st in STAGE_NAMES:
                gs = sub[sub["stage"] == st].sort_values("run_id")
                if len(gs) == 0:
                    continue
                n = max(MIN_STAGE_VAL_LEN, int(round(len(gs) * VAL_RATIO_STAGE)))
                n = min(n, max(len(gs) - 2, 1))
                start = max(0, (len(gs) - n) // 2)
                val_idx.extend(gs.iloc[start:start + n].index.tolist())
            val_idx = sorted(set(val_idx))
            val_parts.append(sub.loc[val_idx].copy())
            train_parts.append(sub.drop(index=val_idx).copy())
        final_train = pd.concat(train_parts).sort_values(["condition", "run_id"]).reset_index(drop=True)
        final_val = pd.concat(val_parts).sort_values(["condition", "run_id"]).reset_index(drop=True)
        test_df = df[df["condition"] == test_condition].sort_values("run_id").reset_index(drop=True).copy()
        return final_train, final_val, test_df
    return split_fn


def get_task_split(train_conditions: list[str], test_condition: str, conditions: tuple[str, ...] | None = None):
    """Full pipeline: load VB -> label -> task-specific split. Returns
    (train_df, val_df, test_df), each with columns including condition,
    run_id, VB, stage, stage_id."""
    all_conditions = conditions or tuple(sorted(set(train_conditions) | {test_condition}))
    labeled = get_condition_relative_labels(all_conditions)
    split_fn = make_split_fn(train_conditions, test_condition)
    return split_fn(labeled)

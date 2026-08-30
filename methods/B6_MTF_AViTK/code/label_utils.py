# -*- coding: utf-8 -*-
r"""
Condition-relative Early/Middle/Late stage-label utility, vendored from the
old project. Combines two old sources so this method needs no live import
from the old parent project (self-containment, task spec section 38):

  - `baselines/mtf_avitk/data/label_utils.py` (load_vb_table /
    get_condition_relative_labels / get_unified_split wrapper functions).
  - `代码/main_experiment_3_fgds_psi_optimized.py` (the actual
    STAGE_NAMES/STAGE_TO_ID/ID_TO_STAGE constants, Q_EARLY/Q_LATE/RATE_LATE_Q
    thresholds, and `define_condition_relative_stages` body) -- the old
    label_utils.py only *imported* this module as `dcpsr_base`; this file
    vendors the specific pieces actually used by MTF-AViTK instead.

VB convention: VB = max(flute_1, flute_2, flute_3) per run -- this project's
confirmed real-data convention (see the old project's
`baselines/htt_net/README.md`, "One correction this recovery surfaced").

ADAPTATION vs. the original `split_grouped_lifecycle` (task spec section 35,
allowed: task/seed routing): the original was hardcoded to
train={C1,C4}, test={C6}. `split_grouped_lifecycle_generic()` below is the
same stage-stratified centered-slice val carve, generalized to accept
arbitrary `train_cutters`/`test_cutter` -- ported from (not reinventing)
`final_statistical_evidence/scripts/methods/condition_split.py::make_split_fn`,
which already validated this exact generalization for the other 3 raw-signal
published baselines. Every threshold/formula below (Q_EARLY, Q_LATE,
RATE_LATE_Q, VAL_RATIO_STAGE, MIN_STAGE_VAL_LEN, rolling-window sizes) is
copied verbatim -- only the split's train/test cutter arguments are new.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = Path(os.environ.get("PHM2010_ROOT", str(REPO_ROOT / "data" / "PHM2010" / "raw")))

STAGE_NAMES = ["early", "middle", "late"]
STAGE_TO_ID = {"early": 0, "middle": 1, "late": 2}
ID_TO_STAGE = {0: "early", 1: "middle", 2: "late"}

# Verbatim from 代码/main_experiment_3_fgds_psi_optimized.py
Q_EARLY = 0.30
Q_LATE = 0.72
RATE_LATE_Q = 0.78
VAL_RATIO_STAGE = 0.20
MIN_STAGE_VAL_LEN = 8


def load_vb_table(cutters: tuple[str, ...]) -> pd.DataFrame:
    """Build a (condition, run_id, VB) table from the raw PHM2010 wear CSVs
    under PHM2010_ROOT, using VB = max(flute_1, flute_2, flute_3)."""
    rows = []
    for cond in cutters:
        cond_lower = cond.lower()
        wear_path = RAW_DIR / cond_lower / f"{cond_lower}_wear.csv"
        wdf = pd.read_csv(wear_path)
        wdf.columns = [str(c).strip() for c in wdf.columns]
        vb = wdf[["flute_1", "flute_2", "flute_3"]].max(axis=1)
        for cut, v in zip(wdf["cut"].astype(int), vb):
            rows.append({"condition": cond, "run_id": int(cut), "VB": float(v)})
    return pd.DataFrame(rows)


def define_condition_relative_stages(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verbatim port of 代码/main_experiment_3_fgds_psi_optimized.py::define_condition_relative_stages
    (minus the fine_state_true / N_FINE_STATES column, which only B3/B9's GMM
    stage-refinement step needs -- not used by MTF-AViTK)."""
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
    return out, pd.DataFrame(rows)


def split_grouped_lifecycle_generic(
    df: pd.DataFrame, train_cutters: list[str], test_cutter: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stage-stratified centered-slice train/val carve over `train_cutters`,
    `test_cutter` held out entirely. Generalization of the original
    hardcoded-C1+C4-train/C6-test split -- see module docstring."""
    if set(train_cutters) & {test_cutter}:
        raise ValueError(f"Leakage: train {train_cutters} overlaps test {test_cutter}.")
    train_parts, val_parts = [], []
    for cond in train_cutters:
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
    test_df = df[df["condition"] == test_cutter].sort_values("run_id").reset_index(drop=True).copy()
    return final_train, final_val, test_df


def get_condition_relative_labels(cutters: tuple[str, ...]) -> pd.DataFrame:
    df = load_vb_table(cutters)
    labeled, _thresholds = define_condition_relative_stages(df)
    return labeled


def get_unified_split(train_cutters: list[str], test_cutter: str):
    """Train/val/test split for the given task's cutters. Returns
    (train_df, val_df, test_df) with label columns from
    get_condition_relative_labels()."""
    all_cutters = tuple(train_cutters) + (test_cutter,)
    labeled = get_condition_relative_labels(all_cutters)
    return split_grouped_lifecycle_generic(labeled, train_cutters, test_cutter)

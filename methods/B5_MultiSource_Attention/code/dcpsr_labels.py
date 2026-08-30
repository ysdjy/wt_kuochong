# -*- coding: utf-8 -*-
"""Self-contained condition-relative Early/Middle/Late stage-label + split
logic for B5 (Multi-source Channel-Spatial Attention).

Vendored (not live-imported) from the old project's
`代码/main_experiment_3_fgds_psi_optimized.py` (functions
`define_condition_relative_stages`, `split_grouped_lifecycle`, and the
STAGE_NAMES/STAGE_TO_ID/ID_TO_STAGE/Q_EARLY/Q_LATE/RATE_LATE_Q/
VAL_RATIO_STAGE/MIN_STAGE_VAL_LEN/N_FINE_STATES constants), at
legacy_git_commit 811da096ee47bea4f65db193aa49e793dba6f47d — see
../source_manifest.json for exact provenance/hashes.

ADAPTATION vs. the original (allowed: task/seed routing, per task spec
section 35 -- architecture/thresholds/logic are byte-for-byte unchanged):
`split_grouped_lifecycle(df)` was hardcoded train={C1,C4}, test={C6}. Here
it takes explicit `train_conditions`/`test_condition` parameters instead,
so it works for D1/D2/D3 (any of the 3 task definitions in
shared/phm2010/tasks.py), not just D1. This is the same generalization the
old project's `final_statistical_evidence/scripts/methods/condition_split.py::make_split_fn`
already validated (D2/D3 numbers in that project's
`final_statistical_evidence/results/TRANSFER_TASKS_MEAN_STD.csv` were
produced by this exact logic) -- ported here as a real function instead of
a runtime monkeypatch, since this repo controls its own vendored copy of
the code being called.

The old `data/label_utils.py` also imported `代码/main_experiment_3_fgds_psi_optimized.py`
live from outside the repo purely to get these functions -- that violated
this repo's self-containment requirement (task spec section 38), which
this vendored module fixes.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

STAGE_NAMES = ["early", "middle", "late"]
STAGE_TO_ID = {"early": 0, "middle": 1, "late": 2}
ID_TO_STAGE = {0: "early", 1: "middle", 2: "late"}

Q_EARLY = 0.30
Q_LATE = 0.72
RATE_LATE_Q = 0.78

VAL_RATIO_STAGE = 0.20
MIN_STAGE_VAL_LEN = 8

N_FINE_STATES = 5

REPO_ROOT = Path(__file__).resolve().parents[3]
# Production location: scripts/download_phm2010.py's output. Accepts either
# env var name -- PHM2010_ROOT is what B6/B7/B8 and download_phm2010.py's own
# docs use; PHM2010_RAW_ROOT is kept as an alias so an existing override still
# works.
PHM2010_RAW_ROOT = Path(
    os.environ.get("PHM2010_ROOT") or os.environ.get("PHM2010_RAW_ROOT") or str(REPO_ROOT / "data" / "PHM2010" / "raw")
)


def load_vb_table(conditions=("C1", "C4", "C6")) -> pd.DataFrame:
    """(condition, run_id, VB) straight from the raw PHM2010 wear CSVs.
    VB = max(flute_1, flute_2, flute_3) -- this project's confirmed
    real-data convention (see data/README.md / htt_net README history),
    not the mean-of-three convention some published papers' own text uses.
    """
    rows = []
    for cond in conditions:
        cond_lower = cond.lower()
        wear_path = PHM2010_RAW_ROOT / cond_lower / f"{cond_lower}_wear.csv"
        wdf = pd.read_csv(wear_path)
        wdf.columns = [str(c).strip() for c in wdf.columns]
        vb = wdf[["flute_1", "flute_2", "flute_3"]].max(axis=1)
        for cut, v in zip(wdf["cut"].astype(int), vb):
            rows.append({"condition": cond, "run_id": int(cut), "VB": float(v)})
    return pd.DataFrame(rows)


def define_condition_relative_stages(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verbatim port of 代码/main_experiment_3_fgds_psi_optimized.py::define_condition_relative_stages
    (minus the CSV side-effect write, which was a diagnostic artifact of the
    old multi-model driver script, not needed here)."""
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
        sub["fine_state_true"] = pd.qcut(
            sub["q_true"].rank(method="first"), q=N_FINE_STATES, labels=False, duplicates="drop"
        ).astype(int)
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


def split_grouped_lifecycle(
    df: pd.DataFrame, train_conditions: list[str], test_condition: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generalized (task-parameterized) port of split_grouped_lifecycle.
    Stage-stratified centered-slice internal validation carved out of the
    train conditions ONLY; test_condition held out entirely (never touched
    for feature selection/scaling/model-selection). Identical slice logic
    to the original, just no longer hardcoded to C1+C4 -> C6.
    """
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


def get_condition_relative_labels(conditions=("C1", "C4", "C6")) -> pd.DataFrame:
    df = load_vb_table(conditions)
    labeled, _thresholds = define_condition_relative_stages(df)
    return labeled


def get_task_split(train_conditions: list[str], test_condition: str, conditions=("C1", "C4", "C6")):
    labeled = get_condition_relative_labels(conditions)
    return split_grouped_lifecycle(labeled, train_conditions, test_condition)

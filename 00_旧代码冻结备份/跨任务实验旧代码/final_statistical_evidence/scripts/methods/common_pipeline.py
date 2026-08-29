# -*- coding: utf-8 -*-
"""
Shared, condition-parameterized data pipeline for the internal window-based
methods (RF, TCN-GRU, Multi-task TCN-GRU, DC-PSR, HTT-Net) used by the
D2/D3 transfer-task runner.

This module imports `代码/main_experiment_3_fgds_psi_optimized.py`
UNMODIFIED (read-only) and reimplements only the one hardcoded choke point
that needs generalizing: `split_grouped_lifecycle(df)` hardcodes
train={C1,C4}, test={C6}. The replacement `split_by_conditions()` below is
copied verbatim in logic from `代码/7.7跨工况实验.py`'s
`split_train_val_by_conditions`/`prepare_task_data` (already-validated
no-leakage reference implementation: same VAL_RATIO_STAGE/MIN_STAGE_VAL_LEN
stage-stratified middle-slice carve, same train-only fit of feature
selection/GMM/scaler), just parameterized instead of hardcoded, and pointed
at the CURRENT machine's authoritative feature file. No hyperparameter,
architecture, or feature-engineering logic is changed anywhere.
"""
from __future__ import annotations

import os
import sys
import tempfile
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = REPO_ROOT / "代码"

_base = None


def import_base():
    global _base
    if _base is not None:
        return _base
    os.environ.setdefault("FGDS_RUN_DIR", str(Path(tempfile.gettempdir()) / "final_statistical_evidence_base_side_outputs"))
    sys.path.insert(0, str(CODE_DIR))
    import main_experiment_3_fgds_psi_optimized as base  # noqa
    base.FEATURE_FILE = REPO_ROOT / "baselines" / "htt_net" / "data" / "run_level_features_all.csv"
    _base = base
    return base


def set_train_seed(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"[set_train_seed] WARNING: use_deterministic_algorithms unavailable: {e}")


def load_label_df():
    base = import_base()
    df = base.load_feature_table()
    label_df, _ = base.define_condition_relative_stages(df)
    return label_df


def split_by_conditions(base, label_df, train_conditions, test_condition):
    """Verbatim logic from 代码/7.7跨工况实验.py::split_train_val_by_conditions,
    parameterized (never hardcoded to C1/C4/C6)."""
    if set(train_conditions) & {test_condition}:
        raise ValueError(f"Leakage: train conditions {train_conditions} overlap test condition {test_condition}.")
    train_parts, val_parts = [], []
    for cond in train_conditions:
        sub = label_df[label_df["condition"] == cond].sort_values("run_id").reset_index(drop=True).copy()
        val_idx = []
        for st in base.STAGE_NAMES:
            gs = sub[sub["stage"] == st].sort_values("run_id")
            if len(gs) == 0:
                continue
            n = max(base.MIN_STAGE_VAL_LEN, int(round(len(gs) * base.VAL_RATIO_STAGE)))
            n = min(n, max(len(gs) - 2, 1))
            start = max(0, (len(gs) - n) // 2)
            val_idx.extend(gs.iloc[start:start + n].index.tolist())
        val_idx = sorted(set(val_idx))
        val_parts.append(sub.loc[val_idx].copy())
        train_parts.append(sub.drop(index=val_idx).copy())
    train_raw = pd.concat(train_parts).sort_values(["condition", "run_id"]).reset_index(drop=True)
    val_raw = pd.concat(val_parts).sort_values(["condition", "run_id"]).reset_index(drop=True)
    test_raw = label_df[label_df["condition"] == test_condition].sort_values("run_id").reset_index(drop=True).copy()
    return train_raw, val_raw, test_raw


def prepare_task_data(train_conditions, test_condition):
    """Verbatim logic from 代码/7.7跨工况实验.py::prepare_task_data: builds
    online features per split (train-only-fit selector/GMM/scaler), returns
    (tr_pack, va_pack, te_pack, selected_features)."""
    base = import_base()
    label_df = load_label_df()
    train_raw, val_raw, test_raw = split_by_conditions(base, label_df, train_conditions, test_condition)

    raw_cols = base.get_raw_numeric_sensor_cols(train_raw)
    split_feat = base.build_online_features_by_split(
        {"train": train_raw, "val": val_raw, "test": test_raw}, raw_cols
    )
    feat_train = split_feat[split_feat["split_name_for_feature_build"] == "train"].copy()
    feat_val = split_feat[split_feat["split_name_for_feature_build"] == "val"].copy()
    feat_test = split_feat[split_feat["split_name_for_feature_build"] == "test"].copy()

    all_cols = base.feature_cols_from(feat_train)
    feat_train, feat_val = base.fill_by_train_median(feat_train, feat_val, all_cols)
    _, feat_test = base.fill_by_train_median(feat_train, feat_test, all_cols)

    selected, _ = base.select_features_train_only(feat_train)
    feat_train, feat_val = base.fill_by_train_median(feat_train, feat_val, selected)
    _, feat_test = base.fill_by_train_median(feat_train, feat_test, selected)

    gmm, raw_to_order = base.fit_train_gmm(feat_train)
    feat_train = base.assign_fine_states(feat_train, gmm, raw_to_order)
    feat_val = base.assign_fine_states(feat_val, gmm, raw_to_order)
    feat_test = base.assign_fine_states(feat_test, gmm, raw_to_order)

    scaler = StandardScaler().fit(feat_train[selected].values)
    for df in [feat_train, feat_val, feat_test]:
        df[selected] = np.nan_to_num(scaler.transform(df[selected].values), nan=0.0, posinf=0.0, neginf=0.0)

    L = base.BEST_ARCH["L"]
    tr_pack = base.make_pack(feat_train, selected, L, "train")
    va_pack = base.make_pack(feat_val, selected, L, "val")
    te_pack = base.make_pack(feat_test, selected, L, "test")
    return tr_pack, va_pack, te_pack, selected, feat_train, feat_test


# ---------------------------------------------------------------------------
# Metrics -- verbatim formulas from 代码/7.4对比实验.py::metrics_row/consistency
# ---------------------------------------------------------------------------
def pointwise_metrics(y_true, y_pred):
    p_each, r_each, f1_each, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cmn = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    return {
        "Acc": accuracy_score(y_true, y_pred),
        "MacroF1": macro_f1,
        "E_F1": f1_each[0],
        "M_F1": f1_each[1],
        "L_F1": f1_each[2],
        "M_Precision": p_each[1],
        "M_Recall": r_each[1],
        "M_to_E": cmn[1, 0],
        "M_to_L": cmn[1, 2],
    }


def sequence_metrics(y_pred, prob):
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_pred) <= 1:
        return {"Rev": 0, "Jump": 0, "Smooth": float("nan")}
    dy = np.diff(y_pred)
    return {
        "Rev": int(np.sum(dy < 0)),
        "Jump": int(np.sum(np.abs(dy) >= 2)),
        "Smooth": float(np.mean(np.sum(np.abs(np.diff(prob, axis=0)), axis=1))),
    }


def full_metrics(y_true, y_pred, prob):
    m = pointwise_metrics(y_true, y_pred)
    m.update(sequence_metrics(y_pred, prob))
    return m

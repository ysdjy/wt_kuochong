"""Unified stage-classification / sequence-diagnostic metrics for PHM2010 (and future
NASA / MTW-CM) tool-wear stage experiments.

This is the ONE authoritative implementation for the new `wt_kuochong` repo. Every
method's `evaluate()` must call into this module for its formal comparison-table
numbers; a method may additionally keep its own native/original-paper metric code,
but that never substitutes for these values in `results/**/metrics.json`.

Formulas are ported verbatim (not reinvented) from the paper's own authoritative
source, `paper_data/99_scripts/build_paper_data.py::recompute_transfer_metrics`,
which is itself the source of the frozen Acc/MacroF1/M_F1/... numbers already used
in the manuscript. See also `paper_data/New_figure/_shared/data_utils.py` for the
order-independent/order-dependent split this module also follows (safe subset for
bootstrap resampling: classification_metrics; NOT safe under resampling:
sequence_diagnostics, which depends on consecutive-sample order and must only be
evaluated on the true, unresampled per-run sequence).
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

STAGE_ORDER = ["early", "middle", "late"]
STAGE_ID = {"early": 0, "middle": 1, "late": 2}


def stage_id(value) -> int:
    """Map a stage label (string, already-int, or numpy scalar) to its canonical 0/1/2 id."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    return STAGE_ID[str(value).strip().lower()]


def _to_ids(values: Iterable) -> np.ndarray:
    return np.array([stage_id(v) for v in values], dtype=int)


def confusion_matrix(truth_ids: Sequence[int], pred_ids: Sequence[int], n_classes: int = 3) -> np.ndarray:
    """cm[i, j] = count of true label i predicted as label j."""
    truth_ids = np.asarray(truth_ids)
    pred_ids = np.asarray(pred_ids)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(truth_ids, pred_ids):
        cm[t, p] += 1
    return cm


def precision_recall_f1_per_class(cm: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    n = cm.shape[0]
    precision, recall, f1 = [], [], []
    for label in range(n):
        tp = cm[label, label]
        fp = cm[:, label].sum() - tp
        fn = cm[label, :].sum() - tp
        p_ = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) > 0 else 0.0
        precision.append(float(p_))
        recall.append(float(r_))
        f1.append(float(f_))
    return precision, recall, f1


def classification_metrics(true_stage: Iterable, pred_stage: Iterable) -> dict:
    """Order-independent metrics. Safe to evaluate on a block-resampled (reordered /
    duplicated) sequence, e.g. inside a moving-block bootstrap.

    Accepts stage labels as strings ("early"/"middle"/"late") or already-integer ids.
    """
    truth_ids = _to_ids(true_stage)
    pred_ids = _to_ids(pred_stage)
    cm = confusion_matrix(truth_ids, pred_ids)
    precision, recall, f1 = precision_recall_f1_per_class(cm)
    middle_total = cm[1, :].sum()
    return {
        "Acc": float(np.mean(truth_ids == pred_ids)),
        "MacroF1": float(np.mean(f1)),
        "E_F1": f1[0],
        "M_F1": f1[1],
        "L_F1": f1[2],
        "M_Precision": precision[1],
        "M_Recall": recall[1],
        "M_to_E": float(cm[1, 0] / (middle_total + 1e-12)),
        "M_to_L": float(cm[1, 2] / (middle_total + 1e-12)),
    }


def sequence_diagnostics(pred_stage: Iterable, probs: np.ndarray) -> dict:
    """Order-DEPENDENT diagnostics: Rev, Jump, Smooth.

    `pred_stage` and `probs` MUST already be sorted by the run's natural per-cutter
    run_id order (the true, unresampled lifecycle sequence) before calling this —
    do not call on a bootstrap-resampled or otherwise reordered index.

    probs: array of shape (N, 3), columns [p_early, p_middle, p_late].
    """
    pred_ids = _to_ids(pred_stage)
    probs = np.asarray(probs, dtype=float)
    diffs = np.diff(pred_ids)
    variation = np.abs(np.diff(probs, axis=0)).sum(axis=1)
    return {
        "Rev": int(np.sum(diffs < 0)),
        "Jump": int(np.sum(np.abs(diffs) >= 2)),
        "Smooth": float(np.mean(variation)) if len(variation) > 0 else float("nan"),
    }


def q_regression_metrics(q_true: Iterable | None, q_pred: Iterable | None) -> dict:
    """Continuous degradation-index regression metrics (q-MAE / q-RMSE / q-R2).

    Only applicable to methods that output a continuous degradation index. If a
    method does not produce one, pass q_true=None or q_pred=None (or an empty
    sequence) and this returns NaN for all three — per project policy, methods
    without a q output must NEVER have this faked, only left as NaN.
    """
    if q_true is None or q_pred is None:
        return {"q_MAE": float("nan"), "q_RMSE": float("nan"), "q_R2": float("nan")}
    q_true = np.asarray(list(q_true), dtype=float)
    q_pred = np.asarray(list(q_pred), dtype=float)
    if q_true.size == 0 or q_pred.size == 0:
        return {"q_MAE": float("nan"), "q_RMSE": float("nan"), "q_R2": float("nan")}
    mae = float(np.mean(np.abs(q_true - q_pred)))
    rmse = float(np.sqrt(np.mean((q_true - q_pred) ** 2)))
    ss_res = np.sum((q_true - q_pred) ** 2)
    ss_tot = np.sum((q_true - q_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"q_MAE": mae, "q_RMSE": rmse, "q_R2": r2}


def compute_all_metrics(
    true_stage: Iterable,
    pred_stage: Iterable,
    probs: np.ndarray,
    q_true: Iterable | None = None,
    q_pred: Iterable | None = None,
) -> dict:
    """Full formal metric set for one run's ordered (per run_id) predictions.

    Caller is responsible for sorting true_stage/pred_stage/probs (and q_true/q_pred,
    if provided) by run_id ascending BEFORE calling this — sequence_diagnostics
    depends on that order.
    """
    out = {}
    out.update(classification_metrics(true_stage, pred_stage))
    out.update(sequence_diagnostics(pred_stage, probs))
    out.update(q_regression_metrics(q_true, q_pred))
    return out

"""Evaluation metrics. Identical schema across all three datasets."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from . import config as C

METRIC_COLS = ["Acc", "Macro_F1", "E_F1", "M_F1", "L_F1", "M_Pre", "M_Rec",
               "M_to_E", "M_to_L", "Rev", "Jump", "Smooth"]
Q_METRIC_COLS = ["q_MAE", "q_RMSE", "q_R2", "Spearman", "Pearson", "q_Smooth"]


def classification_metrics(y_true, y_pred, stages_present=None) -> dict:
    y_true = np.asarray(y_true, int); y_pred = np.asarray(y_pred, int)
    labels = sorted(stages_present) if stages_present is not None else [0, 1, 2]
    f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per = dict(zip(labels, f1s))
    m = dict(
        Acc=float(accuracy_score(y_true, y_pred)),
        Macro_F1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        E_F1=float(per.get(0, np.nan)), M_F1=float(per.get(1, np.nan)), L_F1=float(per.get(2, np.nan)),
        M_Pre=float(precision_score(y_true, y_pred, labels=[1], average="micro", zero_division=0)),
        M_Rec=float(recall_score(y_true, y_pred, labels=[1], average="micro", zero_division=0)),
        stages_present=",".join(C.ID_TO_STAGE[l] for l in labels),
    )
    mid = y_true == 1
    n_mid = int(mid.sum())
    m["M_to_E"] = float((y_pred[mid] == 0).sum() / n_mid) if n_mid else np.nan
    m["M_to_L"] = float((y_pred[mid] == 2).sum() / n_mid) if n_mid else np.nan
    m["n_middle_true"] = n_mid
    return m


def consistency_metrics(df: pd.DataFrame, pred_col: str, prob_cols: list[str]) -> dict:
    """Rev  = backward stage transitions (late->middle, middle->early, late->early)
       Jump = transitions skipping a stage (early->late or late->early)
       Smooth = mean L1 change of the probability vector between adjacent runs
    All computed per sequence and summed / averaged across sequences."""
    rev = jump = 0
    sm, w = [], []
    for _, sub in df.groupby("sequence_id", sort=False):
        sub = sub.sort_values("order_key")
        s = sub[pred_col].values.astype(int)
        d = np.diff(s)
        rev += int((d < 0).sum())
        jump += int((np.abs(d) >= 2).sum())
        P = sub[prob_cols].values.astype(float)
        if len(P) > 1:
            sm.append(float(np.abs(np.diff(P, axis=0)).sum(axis=1).mean()))
            w.append(len(P) - 1)
    smooth = float(np.average(sm, weights=w)) if sm else np.nan
    return dict(Rev=int(rev), Jump=int(jump), Smooth=smooth)


def q_metrics(q_true, q_pred, df=None) -> dict:
    qt = np.asarray(q_true, float); qp = np.asarray(q_pred, float)
    ok = np.isfinite(qt) & np.isfinite(qp)
    qt, qp = qt[ok], qp[ok]
    if len(qt) < 3:
        return {k: np.nan for k in Q_METRIC_COLS}
    err = qp - qt
    ss = float(((qt - qt.mean()) ** 2).sum())
    out = dict(q_MAE=float(np.abs(err).mean()),
               q_RMSE=float(np.sqrt((err ** 2).mean())),
               q_R2=float(1 - (err ** 2).sum() / ss) if ss > 0 else np.nan,
               Spearman=float(spearmanr(qt, qp).statistic),
               Pearson=float(pearsonr(qt, qp)[0]))
    if df is not None:
        vals, wts = [], []
        for _, sub in df.groupby("sequence_id", sort=False):
            v = sub.sort_values("order_key")["q_hat"].values.astype(float)
            if len(v) > 1:
                vals.append(float(np.abs(np.diff(v)).mean())); wts.append(len(v) - 1)
        out["q_Smooth"] = float(np.average(vals, weights=wts)) if vals else np.nan
    else:
        out["q_Smooth"] = np.nan
    return out


def evaluate(df: pd.DataFrame, variant: str, with_q: bool = True) -> dict:
    pred_col = f"stage_pred_{variant}"
    prob_cols = [f"{variant}_prob_{s}" for s in C.STAGE_NAMES]
    present = sorted(df["stage_true_id"].unique().tolist())
    m = classification_metrics(df["stage_true_id"], df[pred_col], present)
    m.update(consistency_metrics(df, pred_col, prob_cols))
    if with_q and "q_hat" in df:
        m.update(q_metrics(df["q_true"], df["q_hat"], df))
    return m


def confusion_rows(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for ti, tn in C.ID_TO_STAGE.items():
        sub = df[df.stage_true_id == ti]
        for pi, pn in C.ID_TO_STAGE.items():
            n = int((sub[pred_col] == pi).sum())
            rows.append(dict(true_stage=tn, pred_stage=pn, count=n,
                             normalized=(n / len(sub)) if len(sub) else np.nan))
    return pd.DataFrame(rows)

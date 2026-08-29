# -*- coding: utf-8 -*-
r"""
Main experiment 3: FGDS-PSI optimized main experiment.

Purpose:
    Run the proposed strict no-leak main experiment before comparison and
    ablation experiments, with concise outputs and journal-style figures.

Input:
    C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\1run_run_level_features\02_features\run_level_features_all.csv

Output:
    C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文\3_main_experiment_fgds_psi

Main outputs:
    1_results:
        main_experiment_summary.csv
        selected_features.csv
        split_and_stage_summary.csv
        best_probability_params.csv
        test_C6_metrics.csv
    2_models:
        fgds_psi_best_model.pth
    3_figures:
        journal-style figures for stage definition, feature ranking,
        3D fine-state space, training curves, q prediction,
        probability evolution, confusion matrix and probability simplex.
    4_predictions:
        validation_predictions.csv
        test_C6_predictions.csv

Notes:
    This version focuses on the proposed main experiment only.
    It fixes the architecture to the best setting reported in the manuscript
    to avoid repeatedly running expensive proxy architecture search.
"""

from pathlib import Path
import copy
import os
import random
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from scipy.special import expit
from scipy.stats import spearmanr

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# =========================================================
# 0. Configuration
# =========================================================
ROOT = Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM")
FEATURE_FILE = ROOT / "PHM实验" / "1run_run_level_features" / "02_features" / "run_level_features_all.csv"

RUN_NAME = "3_main_experiment_fgds_psi"
RUN_DIR = Path(os.environ.get("FGDS_RUN_DIR", str(ROOT / "PHM实验" / "小论文" / RUN_NAME)))

DIR_RESULT = RUN_DIR / "1_results"
DIR_MODEL = RUN_DIR / "2_models"
DIR_FIG = RUN_DIR / "3_figures"
DIR_PRED = RUN_DIR / "4_predictions"
for d in [DIR_RESULT, DIR_MODEL, DIR_FIG, DIR_PRED]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STAGE_NAMES = ["early", "middle", "late"]
STAGE_TO_ID = {"early": 0, "middle": 1, "late": 2}
ID_TO_STAGE = {0: "early", 1: "middle", 2: "late"}
STAGE_COLORS = {"early": "#2E8B57", "middle": "#D97706", "late": "#C53030"}
FINE_COLORS = ["#2E8B57", "#5DAE61", "#D9A441", "#E07A2F", "#C53030"]

N_FINE_STATES = 5

Q_EARLY = 0.30
Q_LATE = 0.72
RATE_LATE_Q = 0.78

VAL_RATIO_STAGE = 0.20
MIN_STAGE_VAL_LEN = 8

VAR_THRESHOLD = 1e-10
MAX_FEATURE_POOL = 260
N_SELECTED_FEATURES = 45
REDUNDANCY_THRESHOLD = 0.92

BATCH_SIZE = 32
EPOCHS = 120
PATIENCE = 18
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

LAMBDA_STAGE = 1.00
LAMBDA_FINE = 0.25
LAMBDA_Q = 0.30
LAMBDA_MONO = 0.03

# Best architecture used for the proposed main experiment.
BEST_ARCH = {
    "L": 12,
    "dropout": 0.20,
    "lr": 5e-4,
    "channels": (32, 64, 64),
    "gru_hidden": 64,
}

# Probability parameter search on internal validation only.
ETA_LIST = [0.55, 0.65, 0.75]
FINE_WEIGHT_LIST = [0.10, 0.20, 0.30]
TEMP_LIST = [1.0, 1.2]
MID_FLOOR_LIST = [0.04, 0.08, 0.12]
LATE_TAU_LIST = [0.60, 0.66, 0.72]
EARLY_TAU_LIST = [0.34, 0.38, 0.42]
ORDER_BLEND_LIST = [0.25, 0.50]

LATE_SUPPRESS_K = 18.0
EARLY_SUPPRESS_K = 18.0

DPI = 600
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.sans-serif"] = ["Times New Roman", "SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["legend.frameon"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# =========================================================
# 1. Utilities
# =========================================================
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def savefig(path):
    path = Path(path)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()


def savefig_multi(stem):
    """Save one journal-ready bitmap only. No duplicate PDF/TIF copies."""
    stem = Path(stem)
    plt.tight_layout()
    plt.savefig(stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    plt.close()


def normalize_condition_name(x):
    s = str(x).strip().upper()
    if s in ["1", "C1"]:
        return "C1"
    if s in ["4", "C4"]:
        return "C4"
    if s in ["6", "C6"]:
        return "C6"
    return s


def infer_vb_column(df):
    for c in ["VB", "VB_max", "vb", "vb_max"]:
        if c in df.columns:
            return c
    lower = {str(c).lower(): c for c in df.columns}
    for k in ["vb", "vb_max", "vbmax"]:
        if k in lower:
            return lower[k]
    raise ValueError("Cannot find VB or VB_max column.")


def is_meta_or_label_col(col):
    c = str(col).strip().lower()
    c_clean = re.sub(r"[^a-z0-9_]+", "_", c)
    exact = {
        "condition", "run_id", "run", "run_index", "cut", "cut_index", "cutid", "cut_id",
        "time", "timestamp", "cycle", "sample_id", "sample_index", "order", "sequence", "seq",
        "file_name", "filename", "signal_len", "n_channels",
        "vb", "vb_max", "vbmean", "vb_mean", "vbmax", "flute_1", "flute_2", "flute_3",
        "stage", "stage_id", "phase", "phase_id", "fine_state", "fine_state_true",
        "q_true", "q_deg", "q_hat", "q_norm", "vb_smooth", "vb_norm",
        "rate", "rate_norm", "rate_smooth", "dvb", "delta_vb",
        "run_progress", "progress", "life_ratio", "life_percent", "rul", "label", "target",
        "dominant_flute", "final_dominant_flute",
    }
    if c_clean in exact:
        return True
    forbidden = [
        "vb", "flute", "stage", "phase", "target", "label",
        "q_true", "q_deg", "q_hat", "q_norm",
        "rate_smooth", "rate_norm", "vb_smooth", "vb_norm",
        "progress", "life_ratio", "rul", "dvb", "delta_vb", "dominant",
    ]
    if any(k in c_clean for k in forbidden):
        return True
    progress_patterns = [
        r"(^|_)run(_|$)", r"(^|_)run_id(_|$)", r"(^|_)run_index(_|$)",
        r"(^|_)cut(_|$)", r"(^|_)cut_index(_|$)",
        r"(^|_)cycle(_|$)", r"(^|_)order(_|$)", r"(^|_)sequence(_|$)", r"(^|_)seq(_|$)",
        r"(^|_)timestamp(_|$)",
    ]
    return any(re.search(p, c_clean) for p in progress_patterns)


def calc_q_metrics(q_true, q_pred):
    q_true = np.asarray(q_true, dtype=float).reshape(-1)
    q_pred = np.asarray(q_pred, dtype=float).reshape(-1)
    mask = np.isfinite(q_true) & np.isfinite(q_pred)
    if mask.sum() < 3:
        return {"q_MAE": np.nan, "q_RMSE": np.nan, "q_R2": np.nan}
    q_true = q_true[mask]
    q_pred = np.clip(q_pred[mask], 0.0, 1.0)
    return {
        "q_MAE": mean_absolute_error(q_true, q_pred),
        "q_RMSE": np.sqrt(mean_squared_error(q_true, q_pred)),
        "q_R2": r2_score(q_true, q_pred),
    }


def clf_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
    )
    p_each, r_each, f1_each, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    true_ratio = np.bincount(y_true, minlength=3) / max(len(y_true), 1)
    pred_ratio = np.bincount(y_pred, minlength=3) / max(len(y_pred), 1)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "precision": p,
        "recall": r,
        "f1": f1,
        "early_precision": p_each[0],
        "early_recall": r_each[0],
        "early_f1": f1_each[0],
        "middle_precision": p_each[1],
        "middle_recall": r_each[1],
        "middle_f1": f1_each[1],
        "late_precision": p_each[2],
        "late_recall": r_each[2],
        "late_f1": f1_each[2],
        "middle_to_early_rate": cm_norm[1, 0],
        "middle_to_late_rate": cm_norm[1, 2],
        "ratio_penalty": float(np.sum(np.abs(true_ratio - pred_ratio))),
    }


def stage_consistency_metrics(pred_df, method):
    """
    Manuscript metrics:
    Rev: reverse stage transitions, e.g., M->E or L->M.
    Jump: non-adjacent E<->L jumps.
    Smooth: mean L1 change of adjacent stage-probability vectors.
    """
    pred_col = "stage_pred_raw" if method == "raw" else f"stage_pred_{method}"
    prob_prefix = "raw" if method == "raw" else method
    prob_cols = [f"{prob_prefix}_prob_{s}" for s in STAGE_NAMES]
    rows = []
    for _, g in pred_df.sort_values(["condition", "cut_index"]).groupby("condition"):
        y = g[pred_col].values.astype(int)
        if len(y) <= 1:
            continue
        dy = np.diff(y)
        rev = int(np.sum(dy < 0))
        jump = int(np.sum(np.abs(dy) >= 2))
        if all(c in g.columns for c in prob_cols):
            P = g[prob_cols].values.astype(float)
            smooth = float(np.mean(np.sum(np.abs(np.diff(P, axis=0)), axis=1)))
        else:
            smooth = np.nan
        rows.append({"Rev": rev, "Jump": jump, "Smooth": smooth})
    if not rows:
        return {"Rev": np.nan, "Jump": np.nan, "Smooth": np.nan}
    df = pd.DataFrame(rows)
    return {
        "Rev": int(df["Rev"].sum()),
        "Jump": int(df["Jump"].sum()),
        "Smooth": float(df["Smooth"].mean()),
    }


def manuscript_metric_row(pred_df, method, method_id=None, split="test_C6"):
    """Return a result row using the exact metric names in the manuscript."""
    m = evaluate(pred_df, method)
    c = stage_consistency_metrics(pred_df, method)
    row = {
        "Split": split,
        "Method": method_id if method_id is not None else method,
        "Output": method,
        "Acc": m["acc"],
        "Macro-F1": m["f1"],
        "E-F1": m["early_f1"],
        "M-F1": m["middle_f1"],
        "L-F1": m["late_f1"],
        "M-Pre": m["middle_precision"],
        "M-Rec": m["middle_recall"],
        "M→E": m["middle_to_early_rate"],
        "M→L": m["middle_to_late_rate"],
        "Rev": c["Rev"],
        "Jump": c["Jump"],
        "Smooth": c["Smooth"],
        "q-MAE": m["q_MAE"],
        "q-RMSE": m["q_RMSE"],
        "q-R2": m["q_R2"],
    }
    return row


def compact_prediction_table(pred_test):
    out = pd.DataFrame({
        "condition": pred_test["condition"],
        "cut_index": pred_test["cut_index"],
        "true_stage": pred_test["stage_true"],
        "pred_stage_A1": pred_test["stage_pred_raw_name"],
        "pred_stage_A2": pred_test["stage_pred_raw_fine_name"],
        "pred_stage_A3": pred_test["stage_pred_raw_prior_name"],
        "pred_stage_A4": pred_test["stage_pred_mix_name"],
        "pred_stage_A5": pred_test["stage_pred_ordered_name"],
        "pred_stage_A6": pred_test["stage_pred_final_name"],
        "q_true": pred_test["q_true_model"],
        "q_hat": pred_test["q_hat"],
    })
    prefix_map = {
        "raw": "raw",
        "fine": "fine",
        "prior": "prior",
        "mix": "mix",
        "ordered": "ordered",
        "final": "final",
    }
    stage_letter = {"early": "E", "middle": "M", "late": "L"}
    for prefix in prefix_map:
        for st in STAGE_NAMES:
            out[f"{prefix}_prob_{stage_letter[st]}"] = pred_test[f"{prefix}_prob_{st}"]
    return out


def save_a6_detail_tables(pred_test):
    """Save compact tables matching manuscript Table 14 and Table 15 style."""
    y_true = pred_test["stage_true_id"].values.astype(int)
    y_pred = pred_test["stage_pred_final"].values.astype(int)
    p_each, r_each, f1_each, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )
    rows = []
    for i, st in enumerate(["early", "middle", "late"]):
        rows.append({
            "Stage": st,
            "Precision": p_each[i],
            "Recall": r_each[i],
            "F1-score": f1_each[i],
            "Support": int(support[i]),
        })
    rows.append({
        "Stage": "Accuracy",
        "Precision": np.nan,
        "Recall": np.nan,
        "F1-score": accuracy_score(y_true, y_pred),
        "Support": int(len(y_true)),
    })
    rows.append({
        "Stage": "Macro avg",
        "Precision": float(np.mean(p_each)),
        "Recall": float(np.mean(r_each)),
        "F1-score": float(np.mean(f1_each)),
        "Support": int(len(y_true)),
    })
    pd.DataFrame(rows).to_csv(DIR_RESULT / "FINAL_classification_report_A6.csv", index=False, encoding="utf-8-sig")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cmn = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    cm_df = pd.DataFrame(cmn, columns=["Pred early", "Pred middle", "Pred late"])
    cm_df.insert(0, "True stage", ["early", "middle", "late"])
    cm_df.to_csv(DIR_RESULT / "FINAL_confusion_matrix_A6.csv", index=False, encoding="utf-8-sig")


# =========================================================
# 2. Data, labels and split
# =========================================================
def load_feature_table():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURE_FILE}")
    df = pd.read_csv(FEATURE_FILE)
    df.columns = [str(c).strip() for c in df.columns]
    df["condition"] = df["condition"].apply(normalize_condition_name)
    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce").astype(int)
    vb_col = infer_vb_column(df)
    df["VB"] = pd.to_numeric(df[vb_col], errors="coerce")
    df = df[df["condition"].isin(["C1", "C4", "C6"])].copy()
    return df.dropna(subset=["VB"]).sort_values(["condition", "run_id"]).reset_index(drop=True)


def define_condition_relative_stages(df):
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
    th_df.to_csv(DIR_RESULT / "condition_relative_stage_thresholds.csv", index=False, encoding="utf-8-sig")
    return out, th_df


def split_grouped_lifecycle(df):
    train_parts, val_parts = [], []
    for cond in ["C1", "C4"]:
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
    test_c6 = df[df["condition"] == "C6"].sort_values("run_id").reset_index(drop=True).copy()
    rows = []
    for name, sub in [("final_train", final_train), ("final_internal_val", final_val), ("test_C6", test_c6)]:
        for cond, g in sub.groupby("condition"):
            cnt = g["stage"].value_counts().reindex(STAGE_NAMES, fill_value=0)
            rows.append({"split": name, "condition": cond, "n": len(g), **cnt.to_dict()})
    pd.DataFrame(rows).to_csv(DIR_RESULT / "split_and_stage_summary.csv", index=False, encoding="utf-8-sig")
    return final_train, final_val, test_c6


# =========================================================
# 3. Split-safe online features
# =========================================================
def get_raw_numeric_sensor_cols(df):
    cols = []
    for c in df.columns:
        if is_meta_or_label_col(c):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.95:
            cols.append(c)
    return cols


def build_online_features_for_subset(df_subset, raw_cols, split_name):
    parts = []
    for cond, sub in df_subset.groupby("condition"):
        sub = sub.sort_values("run_id").reset_index(drop=True).copy()
        feat = sub[["condition", "run_id"]].copy()
        for col in raw_cols:
            x = pd.to_numeric(sub[col], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
            med0 = np.nanmedian(x.values)
            med0 = 0.0 if not np.isfinite(med0) else float(med0)
            filled, hist = [], []
            for v in x.values:
                if np.isfinite(v):
                    hist.append(float(v))
                    filled.append(float(v))
                else:
                    filled.append(float(np.median(hist)) if hist else med0)
            x = pd.Series(filled, dtype=float)
            exp_mean = x.expanding(min_periods=3).mean()
            exp_std = x.expanding(min_periods=3).std().replace(0, np.nan)
            feat[f"{col}__rel"] = ((x - exp_mean) / (exp_std + 1e-8)).fillna(0.0).values
            feat[f"{col}__slope"] = x.diff().fillna(0.0).values
            ranks, hist = [], []
            for v in x.values:
                hist.append(v)
                arr = np.asarray(hist, dtype=float)
                ranks.append(float((arr <= v).mean()))
            feat[f"{col}__online_rank"] = ranks
        parts.append(feat)
    feat_only = pd.concat(parts).sort_values(["condition", "run_id"]).reset_index(drop=True)
    meta = ["condition", "run_id", "VB", "VB_smooth", "q_true", "rate_norm", "stage", "stage_id", "fine_state_true"]
    out = df_subset[meta].copy().merge(feat_only, on=["condition", "run_id"], how="left")
    out["split_name_for_feature_build"] = split_name
    return out.replace([np.inf, -np.inf], np.nan)


def build_online_features_by_split(split_dict, raw_cols):
    return pd.concat(
        [build_online_features_for_subset(sub, name) if False else build_online_features_for_subset(sub, raw_cols, name)
         for name, sub in split_dict.items()],
        axis=0,
    ).reset_index(drop=True)


def feature_cols_from(feat_df):
    meta = {"condition", "run_id", "VB", "VB_smooth", "q_true", "rate_norm",
            "stage", "stage_id", "fine_state_true", "split_name_for_feature_build"}
    return [c for c in feat_df.columns if c not in meta]


def fill_by_train_median(train, apply, cols):
    tr, ap = train.copy(), apply.copy()
    for c in cols:
        med = pd.to_numeric(tr[c], errors="coerce").replace([np.inf, -np.inf], np.nan).median()
        med = 0.0 if not np.isfinite(med) else med
        tr[c] = pd.to_numeric(tr[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med)
        ap[c] = pd.to_numeric(ap[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med)
    return tr, ap


def select_features_train_only(feat_train):
    cols = feature_cols_from(feat_train)
    ft = feat_train.copy()
    for c in cols:
        med = pd.to_numeric(ft[c], errors="coerce").replace([np.inf, -np.inf], np.nan).median()
        med = 0.0 if not np.isfinite(med) else med
        ft[c] = pd.to_numeric(ft[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med)
    vt = [c for c in cols if ft[c].nunique(dropna=False) > 1 and np.nanvar(ft[c].values) > VAR_THRESHOLD]
    X = ft[vt].values
    mi_s = mutual_info_classif(X, ft["stage_id"].values.astype(int), random_state=RANDOM_SEED)
    mi_q = mutual_info_regression(X, ft["q_true"].values.astype(float), random_state=RANDOM_SEED)
    rows = []
    conds = sorted(ft["condition"].unique())
    for i, c in enumerate(vt):
        rho, _ = spearmanr(ft[c].values.astype(float), ft["q_true"].values.astype(float))
        rho = 0.0 if not np.isfinite(rho) else abs(float(rho))
        instability = 0.0
        if len(conds) >= 2:
            g0, g1 = ft[ft["condition"] == conds[0]][c].values, ft[ft["condition"] == conds[1]][c].values
            instability = abs(np.nanmean(g0) - np.nanmean(g1)) / (0.5 * (np.nanstd(g0) + np.nanstd(g1)) + 1e-8)
            instability = 9.0 if not np.isfinite(instability) else float(instability)
        score = 1.10 * float(mi_s[i]) + 0.35 * float(mi_q[i]) + 0.55 * rho - 0.35 * instability
        rows.append({"feature": c, "mi_stage": mi_s[i], "mi_q": mi_q[i],
                     "spearman_abs_q": rho, "domain_instability": instability, "feature_score": score})
    score_df = pd.DataFrame(rows).sort_values("feature_score", ascending=False).head(MAX_FEATURE_POOL)
    selected, recs = [], []
    for _, row in score_df.iterrows():
        c = row["feature"]
        if selected:
            x = ft[c].values.astype(float)
            max_corr = max(abs(spearmanr(x, ft[s].values.astype(float))[0]) if np.isfinite(spearmanr(x, ft[s].values.astype(float))[0]) else 0.0 for s in selected)
        else:
            max_corr = np.nan
        if len(selected) == 0 or max_corr < REDUNDANCY_THRESHOLD:
            selected.append(c)
            item = row.to_dict()
            item["selected_rank"] = len(selected)
            item["max_abs_corr_with_selected"] = max_corr
            recs.append(item)
        if len(selected) >= N_SELECTED_FEATURES:
            break
    selected_df = pd.DataFrame(recs)
    selected_df.to_csv(DIR_RESULT / "selected_features.csv", index=False, encoding="utf-8-sig")
    return selected, selected_df


def fit_train_gmm(feat_train):
    X = np.nan_to_num(feat_train[["q_true", "rate_norm"]].values.astype(float), nan=0.0, posinf=1.0, neginf=0.0)
    gmm = GaussianMixture(n_components=N_FINE_STATES, covariance_type="full", random_state=RANDOM_SEED, reg_covar=1e-5, n_init=10)
    gmm.fit(X)
    comp = gmm.predict(X)
    mean_q = pd.DataFrame({"comp": comp, "q": feat_train["q_true"].values}).groupby("comp")["q"].mean().sort_values()
    raw_to_order = {raw: i for i, raw in enumerate(mean_q.index.tolist())}
    mapping = pd.DataFrame([
        {"raw_component": int(raw), "fine_state": int(order), "mean_q_train": float(mean_q.loc[raw])}
        for raw, order in raw_to_order.items()
    ]).sort_values("fine_state")
    mapping.to_csv(DIR_RESULT / "gmm_fine_state_mapping.csv", index=False, encoding="utf-8-sig")
    return gmm, raw_to_order


def assign_fine_states(feat_df, gmm, raw_to_order):
    out = feat_df.copy()
    X = np.nan_to_num(out[["q_true", "rate_norm"]].values.astype(float), nan=0.0, posinf=1.0, neginf=0.0)
    raw = gmm.predict(X)
    out["fine_state_true"] = np.array([raw_to_order[int(r)] for r in raw], dtype=int)
    return out


# =========================================================
# 4. Windowing and model
# =========================================================
def build_windows(df_sub, features, L, split_name):
    Xs, ys, yf, yq, meta = [], [], [], [], []
    for cond, sub in df_sub.sort_values(["condition", "run_id"]).groupby("condition"):
        sub = sub.sort_values("run_id").reset_index(drop=True)
        Xv = sub[features].values.astype(np.float32)
        runs = sub["run_id"].values.astype(int)
        for end in range(L - 1, len(sub)):
            start = end - L + 1
            if not np.all(np.diff(runs[start:end + 1]) == 1):
                continue
            Xs.append(Xv[start:end + 1])
            ys.append(int(sub["stage_id"].iloc[end]))
            yf.append(int(sub["fine_state_true"].iloc[end]))
            yq.append(float(sub["q_true"].iloc[end]))
            meta.append({
                "condition": cond, "cut_index": int(runs[end]), "VB_true": float(sub["VB"].iloc[end]),
                "q_true": float(sub["q_true"].iloc[end]), "stage_true_id": int(sub["stage_id"].iloc[end]),
                "stage_true": sub["stage"].iloc[end], "fine_state_true": int(sub["fine_state_true"].iloc[end]),
                "split": split_name,
            })
    return np.asarray(Xs, np.float32), np.asarray(ys), np.asarray(yf), np.asarray(yq, np.float32), pd.DataFrame(meta)


class StageDataset(Dataset):
    def __init__(self, X, ys, yf, yq):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.ys = torch.tensor(ys, dtype=torch.long)
        self.yf = torch.tensor(yf, dtype=torch.long)
        self.yq = torch.tensor(yq, dtype=torch.float32).view(-1, 1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.ys[i], self.yf[i], self.yq[i]


def make_pack(df_sub, features, L, split_name):
    X, ys, yf, yq, meta = build_windows(df_sub, features, L, split_name)
    loader = DataLoader(StageDataset(X, ys, yf, yq), batch_size=BATCH_SIZE, shuffle=(split_name == "final_train"))
    return {"X": X, "ys": ys, "yf": yf, "yq": yq, "meta": meta, "loader": loader}


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x if self.chomp_size == 0 else x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation), Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation), Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
    def forward(self, x):
        y = self.net(x)
        r = x if self.down is None else self.down(x)
        m = min(y.size(-1), r.size(-1))
        return F.relu(y[:, :, :m] + r[:, :, :m])


class TCNGRUMultiTask(nn.Module):
    def __init__(self, input_dim, channels=(32, 64, 64), gru_hidden=64, dropout=0.2):
        super().__init__()
        layers, ch = [], input_dim
        for i, out_ch in enumerate(channels):
            layers.append(TemporalBlock(ch, out_ch, 3, 2 ** i, dropout))
            ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.gru = nn.GRU(input_size=channels[-1], hidden_size=gru_hidden, batch_first=True)
        self.shared = nn.Sequential(nn.Linear(gru_hidden, 64), nn.ReLU(), nn.Dropout(dropout))
        self.stage_head = nn.Linear(64, 3)
        self.fine_head = nn.Linear(64, N_FINE_STATES)
        self.q_head = nn.Linear(64, 1)
    def forward(self, x):
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        z = self.shared(h[:, -1, :])
        stage_logits = self.stage_head(z)
        fine_logits = self.fine_head(z)
        q_hat = torch.sigmoid(self.q_head(z))
        return {
            "stage_logits": stage_logits,
            "fine_logits": fine_logits,
            "stage_prob": F.softmax(stage_logits, dim=1),
            "fine_prob": F.softmax(fine_logits, dim=1),
            "q_hat": q_hat,
        }


def class_weights(y, n):
    cnt = np.bincount(y, minlength=n).astype(float)
    w = cnt.sum() / (n * np.maximum(cnt, 1.0))
    return torch.tensor(w / w.mean(), dtype=torch.float32, device=DEVICE)


def monotonic_q_loss(q_hat):
    if q_hat.numel() <= 2:
        return torch.tensor(0.0, device=q_hat.device)
    return torch.relu(-(q_hat[1:] - q_hat[:-1])).mean()


def run_epoch(model, loader, optimizer=None, stage_w=None, fine_w=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    losses, SP, FP, QH, YS, YF, YQ = [], [], [], [], [], [], []
    for X, ys, yf, yq in loader:
        X, ys, yf, yq = X.to(DEVICE), ys.to(DEVICE), yf.to(DEVICE), yq.to(DEVICE)
        with torch.set_grad_enabled(train):
            out = model(X)
            loss = (
                LAMBDA_STAGE * F.cross_entropy(out["stage_logits"], ys, weight=stage_w)
                + LAMBDA_FINE * F.cross_entropy(out["fine_logits"], yf, weight=fine_w)
                + LAMBDA_Q * F.smooth_l1_loss(out["q_hat"], yq)
                + LAMBDA_MONO * monotonic_q_loss(out["q_hat"])
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        SP.append(out["stage_prob"].detach().cpu().numpy())
        FP.append(out["fine_prob"].detach().cpu().numpy())
        QH.append(out["q_hat"].detach().cpu().numpy().reshape(-1))
        YS.append(ys.detach().cpu().numpy())
        YF.append(yf.detach().cpu().numpy())
        YQ.append(yq.detach().cpu().numpy().reshape(-1))
    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "stage_prob": np.concatenate(SP),
        "fine_prob": np.concatenate(FP),
        "q_hat": np.clip(np.concatenate(QH), 0, 1),
        "ys": np.concatenate(YS),
        "yf": np.concatenate(YF),
        "yq": np.concatenate(YQ),
    }


def train_model(train_pack, val_pack, input_dim):
    set_seed(RANDOM_SEED)
    model = TCNGRUMultiTask(input_dim, BEST_ARCH["channels"], BEST_ARCH["gru_hidden"], BEST_ARCH["dropout"]).to(DEVICE)
    sw, fw = class_weights(train_pack["ys"], 3), class_weights(train_pack["yf"], N_FINE_STATES)
    opt = torch.optim.AdamW(model.parameters(), lr=BEST_ARCH["lr"], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=1e-5)
    best_state, best_score, wait, best_epoch = None, np.inf, 0, 0
    hist = []
    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_pack["loader"], opt, sw, fw)
        va = run_epoch(model, val_pack["loader"], None, sw, fw)
        pred = np.argmax(va["stage_prob"], axis=1)
        m = clf_metrics(va["ys"], pred)
        qm = calc_q_metrics(va["yq"], va["q_hat"])
        score = 1.0 * (1 - m["acc"]) + 1.0 * (1 - m["f1"]) + 1.4 * (1 - m["middle_recall"]) + 0.8 * m["middle_to_late_rate"] + 0.7 * m["middle_to_early_rate"] + 0.2 * qm["q_RMSE"]
        scheduler.step(score)
        hist.append({"epoch": epoch, "train_loss": tr["loss"], "val_loss": va["loss"], "val_acc": m["acc"], "val_macro_f1": m["f1"], "val_middle_recall": m["middle_recall"], "val_q_RMSE": qm["q_RMSE"], "score": score})
        if score < best_score:
            best_score, best_state, best_epoch, wait = score, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
        if wait >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist), best_score, best_epoch


def predict_model(model, pack):
    out = run_epoch(model, pack["loader"], None)
    df = pack["meta"].copy().reset_index(drop=True)
    df["q_hat"] = out["q_hat"]
    df["q_true_model"] = out["yq"]
    for i, st in enumerate(STAGE_NAMES):
        df[f"raw_prob_{st}"] = out["stage_prob"][:, i]
    for i in range(N_FINE_STATES):
        df[f"fine_prob_{i}"] = out["fine_prob"][:, i]
    df["stage_pred_raw"] = np.argmax(out["stage_prob"], axis=1)
    df["stage_pred_raw_name"] = df["stage_pred_raw"].map(ID_TO_STAGE)
    return df


# =========================================================
# 5. Probability inference
# =========================================================
def qhat_prior(q_hat, early_tau, late_tau, mid_floor):
    q = np.clip(np.asarray(q_hat, dtype=float), 0, 1)
    sigma = 0.17
    pe = np.exp(-0.5 * ((q - 0.18) / sigma) ** 2) * expit(EARLY_SUPPRESS_K * (early_tau - q))
    pm = np.maximum(np.exp(-0.5 * ((q - 0.50) / sigma) ** 2), mid_floor)
    pl = np.exp(-0.5 * ((q - 0.84) / sigma) ** 2) * expit(LATE_SUPPRESS_K * (q - late_tau))
    P = np.vstack([pe, pm, pl]).T
    return P / (P.sum(axis=1, keepdims=True) + 1e-12)


def fine_to_stage_prob(pred_df):
    Fp = pred_df[[f"fine_prob_{i}" for i in range(N_FINE_STATES)]].values.astype(float)
    Fp = Fp / (Fp.sum(axis=1, keepdims=True) + 1e-12)
    P = np.zeros((len(pred_df), 3))
    P[:, 0], P[:, 1], P[:, 2] = Fp[:, 0], Fp[:, 1] + Fp[:, 2] + Fp[:, 3], Fp[:, 4]
    return P / (P.sum(axis=1, keepdims=True) + 1e-12)


def temperature_scale(P, temp):
    logp = np.log(np.clip(P, 1e-12, 1.0)) / temp
    logp -= logp.max(axis=1, keepdims=True)
    out = np.exp(logp)
    return out / (out.sum(axis=1, keepdims=True) + 1e-12)


def causal_ordered_filter(P):
    trans = np.array([[0.9400, 0.0550, 0.0050], [0.0050, 0.9550, 0.0400], [0.0005, 0.0100, 0.9895]])
    alpha = np.array([0.90, 0.10, 1e-6])
    alpha /= alpha.sum()
    out = np.zeros_like(P)
    for i in range(len(P)):
        alpha = (alpha @ trans if i > 0 else alpha) * P[i]
        alpha = alpha / (alpha.sum() + 1e-12)
        out[i] = alpha
    return out


def apply_probability_inference(pred_df, params):
    df = pred_df.copy().sort_values(["condition", "cut_index"]).reset_index(drop=True)
    raw = temperature_scale(df[[f"raw_prob_{s}" for s in STAGE_NAMES]].values, params["temperature"])
    prior = qhat_prior(df["q_hat"].values, params["early_tau"], params["late_tau"], params["mid_floor"])
    fine_stage = fine_to_stage_prob(df)
    raw_fine = params["eta"] * raw + (1 - params["eta"]) * fine_stage
    raw_fine /= raw_fine.sum(axis=1, keepdims=True) + 1e-12
    raw_prior = params["eta"] * raw + (1 - params["eta"]) * prior
    raw_prior /= raw_prior.sum(axis=1, keepdims=True) + 1e-12
    aux = params["fine_weight"] * fine_stage + (1 - params["fine_weight"]) * prior
    mix = params["eta"] * raw + (1 - params["eta"]) * aux
    mix /= mix.sum(axis=1, keepdims=True) + 1e-12
    ordered = np.zeros_like(mix)
    for _, idx in df.groupby("condition").groups.items():
        idx = list(idx)
        ordered[idx] = causal_ordered_filter(mix[idx])
    beta = float(params["order_blend"])
    if beta <= 0:
        raise ValueError("Final probability requires order_blend beta > 0.")
    final = (1 - beta) * mix + beta * ordered
    final /= final.sum(axis=1, keepdims=True) + 1e-12
    for arr_name, arr in [
        ("fine", fine_stage),
        ("prior", prior),
        ("raw_fine", raw_fine),
        ("raw_prior", raw_prior),
        ("mix", mix),
        ("ordered", ordered),
        ("final", final),
    ]:
        for i, st in enumerate(STAGE_NAMES):
            df[f"{arr_name}_prob_{st}"] = arr[:, i]
        df[f"stage_pred_{arr_name}"] = np.argmax(arr, axis=1)
        df[f"stage_pred_{arr_name}_name"] = df[f"stage_pred_{arr_name}"].map(ID_TO_STAGE)
    return df


def evaluate(pred_df, method):
    col = "stage_pred_raw" if method == "raw" else f"stage_pred_{method}"
    y_true, y_pred = pred_df["stage_true_id"].values.astype(int), pred_df[col].values.astype(int)
    m = clf_metrics(y_true, y_pred)
    qm = calc_q_metrics(pred_df["q_true_model"], pred_df["q_hat"])
    return {**m, **qm}


def probability_param_search(pred_val_raw):
    rows = []
    for eta in ETA_LIST:
        for fw in FINE_WEIGHT_LIST:
            for temp in TEMP_LIST:
                for mf in MID_FLOOR_LIST:
                    for lt in LATE_TAU_LIST:
                        for et in EARLY_TAU_LIST:
                            for ob in ORDER_BLEND_LIST:
                                p = {"eta": eta, "fine_weight": fw, "temperature": temp, "mid_floor": mf, "late_tau": lt, "early_tau": et, "order_blend": ob}
                                pred = apply_probability_inference(pred_val_raw, p)
                                m = evaluate(pred, "final")
                                score = 1.0 * (1 - m["acc"]) + 1.0 * (1 - m["f1"]) + 1.4 * (1 - m["middle_recall"]) + 0.8 * m["middle_to_late_rate"] + 0.7 * m["middle_to_early_rate"] + 0.4 * m["ratio_penalty"] + 0.15 * m["q_RMSE"]
                                rows.append({**p, **{f"val_{k}": v for k, v in m.items()}, "val_score": score})
    rank = pd.DataFrame(rows).sort_values("val_score").reset_index(drop=True)
    if (rank["order_blend"] <= 0).any():
        raise RuntimeError("Probability ranking contains order_blend <= 0, which is not allowed for Final.")
    rank.to_csv(DIR_RESULT / "FINAL_probability_param_ranking.csv", index=False, encoding="utf-8-sig")
    b = rank.iloc[0]
    return {k: float(b[k]) for k in ["eta", "fine_weight", "temperature", "mid_floor", "late_tau", "early_tau", "order_blend"]}, rank


# =========================================================
# 6. Figures
# =========================================================
def plot_stage_definition(label_df):
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.0), sharex=False)
    for ax, cond in zip(axes, ["C1", "C4", "C6"]):
        sub = label_df[label_df["condition"] == cond].sort_values("run_id")
        for st in STAGE_NAMES:
            g = sub[sub["stage"] == st]
            ax.scatter(g["run_id"], g["VB"], s=12, color=STAGE_COLORS[st], label=st, alpha=0.85, linewidths=0)
        ax.plot(sub["run_id"], sub["VB_smooth"], color="#111111", lw=1.4, label="Smoothed VB")
        ax.set_ylabel("VB")
        ax.set_title(cond, loc="left", fontweight="bold")
        ax.grid(True, ls="--", lw=0.4, alpha=0.3)
    axes[-1].set_xlabel("Run index")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.01))
    savefig_multi(DIR_FIG / "Fig01_condition_relative_stage_definition")


def plot_feature_ranking(selected_df):
    df = selected_df.sort_values("feature_score").tail(25)
    plt.figure(figsize=(7.0, 6.0))
    plt.barh(df["feature"], df["feature_score"], color="#4677B8", alpha=0.88)
    plt.xlabel("Feature score")
    plt.ylabel("")
    plt.title("Top selected online features", fontweight="bold")
    plt.grid(axis="x", ls="--", lw=0.4, alpha=0.3)
    savefig_multi(DIR_FIG / "Fig02_selected_feature_ranking")


def plot_gmm_3d(feat_train, feat_test):
    fig = plt.figure(figsize=(10.5, 4.4))
    for i, (df, title) in enumerate([(feat_train, "Training space (C1+C4)"), (feat_test, "External test space (C6)")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        colors = [FINE_COLORS[int(s)] for s in df["fine_state_true"]]
        ax.scatter(df["q_true"], df["rate_norm"], df["VB_smooth"], c=colors, s=12, alpha=0.86, depthshade=False)
        ax.set_xlabel(r"$q_{c,t}$")
        ax.set_ylabel(r"$\nu_{c,t}^{norm}$")
        ax.set_zlabel("Smoothed VB")
        ax.set_title(title, fontweight="bold")
        ax.view_init(elev=22, azim=-58)
    savefig_multi(DIR_FIG / "Fig03_3D_fine_grained_degradation_state_space")


def plot_training_history(hist):
    fig, ax1 = plt.subplots(figsize=(7.2, 4.0))
    ax1.plot(hist["epoch"], hist["train_loss"], color="#4C78A8", lw=1.8, label="Training loss")
    ax1.plot(hist["epoch"], hist["val_loss"], color="#F58518", lw=1.8, label="Validation loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, ls="--", lw=0.4, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(hist["epoch"], hist["val_macro_f1"], color="#54A24B", lw=1.6, label="Validation Macro-F1")
    ax2.set_ylabel("Macro-F1")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right")
    savefig_multi(DIR_FIG / "Fig04_training_dynamics")


def plot_confusion(pred_df):
    y_true = pred_df["stage_true_id"].values.astype(int)
    y_pred = pred_df["stage_pred_final"].values.astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cmn = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    plt.figure(figsize=(4.6, 4.0))
    im = plt.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(3), ["E", "M", "L"])
    plt.yticks(range(3), ["E", "M", "L"])
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.2f}", ha="center", va="center", fontsize=10)
    plt.xlabel("Predicted stage")
    plt.ylabel("True stage")
    plt.title("A6 Final confusion matrix on C6", fontweight="bold")
    savefig_multi(DIR_FIG / "Fig05_C6_final_confusion_matrix")


def plot_q_and_prob(pred_df):
    g = pred_df[pred_df["split"] == "test_C6"].sort_values("cut_index")
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.2), sharex=True)
    axes[0].plot(g["cut_index"], g["q_true_model"], color="#111111", lw=2.0, label=r"$q_{c,t}$")
    axes[0].plot(g["cut_index"], g["q_hat"], color="#4C78A8", lw=1.8, label=r"$\hat{q}_{c,t}$")
    axes[0].set_ylabel("Relative degradation")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, ls="--", lw=0.4, alpha=0.3)
    prob_labels = {"early": r"$p^{final,E}_{c,t}$", "middle": r"$p^{final,M}_{c,t}$", "late": r"$p^{final,L}_{c,t}$"}
    for st in STAGE_NAMES:
        axes[1].plot(g["cut_index"], g[f"final_prob_{st}"], color=STAGE_COLORS[st], lw=1.8, label=prob_labels[st])
    axes[1].scatter(g["cut_index"], g["stage_true_id"] / 2.0, s=10, color="#222222", alpha=0.28, label="True stage / 2")
    axes[1].set_xlabel("Run index")
    axes[1].set_ylabel("Stage probability")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper left", ncol=4)
    axes[1].grid(True, ls="--", lw=0.4, alpha=0.3)
    savefig_multi(DIR_FIG / "Fig06_C6_q_prediction_and_stage_probability")


def plot_probability_simplex(pred_df):
    g = pred_df[pred_df["split"] == "test_C6"].sort_values("cut_index").copy()
    P = g[[f"final_prob_{s}" for s in STAGE_NAMES]].values
    x = P[:, 1] + 0.5 * P[:, 2]
    y = (np.sqrt(3) / 2) * P[:, 2]
    plt.figure(figsize=(5.4, 4.8))
    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, np.sqrt(3) / 2, 0]
    plt.plot(tri_x, tri_y, color="#333333", lw=1.0)
    colors = [STAGE_COLORS[ID_TO_STAGE[int(s)]] for s in g["stage_true_id"]]
    plt.scatter(x, y, c=colors, s=18, alpha=0.82, edgecolors="white", linewidths=0.25)
    plt.text(-0.03, -0.04, "E", ha="right", va="top")
    plt.text(1.03, -0.04, "M", ha="left", va="top")
    plt.text(0.5, np.sqrt(3) / 2 + 0.035, "L", ha="center", va="bottom")
    plt.axis("equal")
    plt.axis("off")
    plt.title(r"C6 final probability simplex $p^{final}_{c,t}$", fontweight="bold")
    savefig_multi(DIR_FIG / "Fig07_C6_final_probability_simplex")


def plot_feature_pca(feat_train, feat_test, selected):
    data = pd.concat([feat_train.assign(domain="train C1+C4"), feat_test.assign(domain="test C6")], axis=0)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    Z = pca.fit_transform(data[selected].values)
    data["PC1"], data["PC2"] = Z[:, 0], Z[:, 1]
    plt.figure(figsize=(6.2, 5.0))
    for domain, marker in [("train C1+C4", "o"), ("test C6", "^")]:
        sub = data[data["domain"] == domain]
        plt.scatter(sub["PC1"], sub["PC2"], c=[STAGE_COLORS[s] for s in sub["stage"]], marker=marker, s=18, alpha=0.72, label=domain, linewidths=0)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.title("Selected feature space projection", fontweight="bold")
    plt.legend(loc="best")
    plt.grid(True, ls="--", lw=0.4, alpha=0.3)
    savefig_multi(DIR_FIG / "Fig08_selected_feature_space_PCA")


def make_figures(label_df, selected_df, feat_train, feat_test, hist, pred_all, pred_test, selected):
    plot_stage_definition(label_df)
    plot_feature_ranking(selected_df)
    plot_gmm_3d(feat_train, feat_test)
    plot_training_history(hist)
    plot_confusion(pred_test)
    plot_q_and_prob(pred_all)
    plot_probability_simplex(pred_test)
    plot_feature_pca(feat_train, feat_test, selected)


# =========================================================
# 7. Main
# =========================================================
def main():
    set_seed(RANDOM_SEED)
    print("=" * 100)
    print("Main experiment 3: FGDS-PSI optimized no-leak main run")
    print("=" * 100)
    print(f"Device: {DEVICE}")
    print(f"Feature file: {FEATURE_FILE}")
    print(f"Output dir: {RUN_DIR}")

    raw_df = load_feature_table()
    label_df, stage_th = define_condition_relative_stages(raw_df)
    final_train_raw, final_val_raw, test_c6_raw = split_grouped_lifecycle(label_df)

    raw_cols = get_raw_numeric_sensor_cols(final_train_raw)
    print(f"Raw numeric sensor features: {len(raw_cols)}")

    split_feat = build_online_features_by_split({
        "final_train": final_train_raw,
        "final_internal_val": final_val_raw,
        "test_C6": test_c6_raw,
    }, raw_cols)
    feat_train = split_feat[split_feat["split_name_for_feature_build"] == "final_train"].copy()
    feat_val = split_feat[split_feat["split_name_for_feature_build"] == "final_internal_val"].copy()
    feat_test = split_feat[split_feat["split_name_for_feature_build"] == "test_C6"].copy()

    all_cols = feature_cols_from(feat_train)
    feat_train, feat_val = fill_by_train_median(feat_train, feat_val, all_cols)
    _, feat_test = fill_by_train_median(feat_train, feat_test, all_cols)

    selected, selected_df = select_features_train_only(feat_train)
    feat_train, feat_val = fill_by_train_median(feat_train, feat_val, selected)
    _, feat_test = fill_by_train_median(feat_train, feat_test, selected)

    gmm, raw_to_order = fit_train_gmm(feat_train)
    feat_train = assign_fine_states(feat_train, gmm, raw_to_order)
    feat_val = assign_fine_states(feat_val, gmm, raw_to_order)
    feat_test = assign_fine_states(feat_test, gmm, raw_to_order)

    scaler = StandardScaler().fit(feat_train[selected].values)
    for df in [feat_train, feat_val, feat_test]:
        df[selected] = np.nan_to_num(scaler.transform(df[selected].values), nan=0.0, posinf=0.0, neginf=0.0)

    L = BEST_ARCH["L"]
    tr_pack = make_pack(feat_train, selected, L, "final_train")
    va_pack = make_pack(feat_val, selected, L, "final_internal_val")
    te_pack = make_pack(feat_test, selected, L, "test_C6")

    model, hist, best_score, best_epoch = train_model(tr_pack, va_pack, len(selected))
    hist.to_csv(DIR_RESULT / "training_history.csv", index=False, encoding="utf-8-sig")
    torch.save(model.state_dict(), DIR_MODEL / "fgds_psi_best_model.pth")

    pred_val_raw = predict_model(model, va_pack)
    pred_test_raw = predict_model(model, te_pack)
    best_params, prob_rank = probability_param_search(pred_val_raw)

    pred_val = apply_probability_inference(pred_val_raw, best_params)
    pred_test = apply_probability_inference(pred_test_raw, best_params)
    pred_all = pd.concat([pred_val, pred_test], axis=0).reset_index(drop=True)
    pred_val.to_csv(DIR_PRED / "FINAL_best_val_predictions.csv", index=False, encoding="utf-8-sig")
    pred_test.to_csv(DIR_PRED / "test_C6_predictions_full_internal.csv", index=False, encoding="utf-8-sig")
    compact_prediction_table(pred_test).to_csv(
        DIR_PRED / "FINAL_best_test_C6_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    method_id_map = {
        "raw": "A1",
        "raw_fine": "A2",
        "raw_prior": "A3",
        "mix": "A4",
        "ordered": "A5",
        "final": "A6",
    }
    metric_rows = []
    for method in ["raw", "raw_fine", "raw_prior", "mix", "ordered", "final"]:
        metric_rows.append(manuscript_metric_row(pred_test, method, method_id_map[method], split="test_C6"))
    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(DIR_RESULT / "FINAL_ablation_outputs.csv", index=False, encoding="utf-8-sig")
    save_a6_detail_tables(pred_test)

    b12_cols = ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M→E", "M→L"]
    b12_row = metric_df[metric_df["Output"] == "final"][b12_cols].copy()
    b12_row.insert(0, "Method", "B12")
    b12_row.insert(1, "Method name", "FGDS-PSI")
    b12_row.to_csv(DIR_RESULT / "FINAL_main_result_B12_for_Table9.csv", index=False, encoding="utf-8-sig")

    mt = metric_df[metric_df["Output"] == "final"].iloc[0].to_dict()
    summary = pd.DataFrame([{
        "experiment": RUN_NAME,
        "feature_file": str(FEATURE_FILE),
        "train_conditions": "C1+C4 final_train blocks",
        "test_condition": "C6",
        "strict_no_leakage": "train/val/test online features built separately; C6 excluded from selection/scaling/GMM/model/probability tuning",
        "best_architecture": str(BEST_ARCH),
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "raw_numeric_sensor_features": len(raw_cols),
        "selected_features": len(selected),
        "eta": best_params["eta"],
        "fine_weight": best_params["fine_weight"],
        "temperature": best_params["temperature"],
        "mid_floor": best_params["mid_floor"],
        "late_tau": best_params["late_tau"],
        "early_tau": best_params["early_tau"],
        "order_blend": best_params["order_blend"],
        **{f"test_final_{k}": v for k, v in mt.items() if k not in ["Split", "Method", "Output"]},
    }])
    summary.to_csv(DIR_RESULT / "FINAL_experiment_summary.csv", index=False, encoding="utf-8-sig")

    make_figures(label_df, selected_df, feat_train, feat_test, hist, pred_all, pred_test, selected)

    a4 = metric_df[metric_df["Method"] == "A4"].iloc[0].to_dict()
    a5 = metric_df[metric_df["Method"] == "A5"].iloc[0].to_dict()
    a6 = metric_df[metric_df["Method"] == "A6"].iloc[0].to_dict()
    a6_differs_from_a4 = not np.allclose(
        pred_test[[f"mix_prob_{s}" for s in STAGE_NAMES]].values,
        pred_test[[f"final_prob_{s}" for s in STAGE_NAMES]].values,
        atol=1e-12,
    )
    print("\nDone.")
    print(f"Best order_blend  : {best_params['order_blend']:.4f}")
    print(f"A4 Mix metrics    : Acc={a4['Acc']:.4f}, Macro-F1={a4['Macro-F1']:.4f}, M-F1={a4['M-F1']:.4f}, M→E={a4['M→E']:.4f}, M→L={a4['M→L']:.4f}, Smooth={a4['Smooth']:.6f}")
    print(f"A5 Ordered metrics: Acc={a5['Acc']:.4f}, Macro-F1={a5['Macro-F1']:.4f}, M-F1={a5['M-F1']:.4f}, M→E={a5['M→E']:.4f}, M→L={a5['M→L']:.4f}, Smooth={a5['Smooth']:.6f}")
    print(f"A6 Final metrics  : Acc={a6['Acc']:.4f}, Macro-F1={a6['Macro-F1']:.4f}, M-F1={a6['M-F1']:.4f}, M→E={a6['M→E']:.4f}, M→L={a6['M→L']:.4f}, Smooth={a6['Smooth']:.6f}")
    print(f"A6 differs from A4: {a6_differs_from_a4}")
    if not a6_differs_from_a4:
        print("WARNING: A6 Final is identical to A4 Mix. Check order_blend beta and probability fusion.")
    print(f"Final C6 Acc      : {mt['Acc']:.4f}")
    print(f"Final C6 Macro-F1 : {mt['Macro-F1']:.4f}")
    print(f"Final C6 M-Rec    : {mt['M-Rec']:.4f}")
    print(f"Final C6 M-F1     : {mt['M-F1']:.4f}")
    print(f"Final C6 M→L      : {mt['M→L']:.4f}")
    print(f"q-RMSE            : {mt['q-RMSE']:.4f}")
    print(f"Figures           : {DIR_FIG}")


if __name__ == "__main__":
    main()

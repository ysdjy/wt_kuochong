# -*- coding: utf-8 -*-
r"""
44run：严格无泄露的细粒度退化状态辅助阶段概率识别

相对上一版 43run 的关键修正：
1. 先划分 final_train / final_internal_val / test_C6，再分别构造 online features，避免 train/val 之间传感器统计量串用；
2. proxy validation 采用嵌套流程：C1->C4 时只用 C1 做特征选择、标准化、GMM、训练；C4->C1 同理；
3. final 模型只用 final_train，即 C1/C4 训练段，做特征选择、标准化、GMM、模型训练；
4. C6 只用于最终测试评价，不参与特征选择、标准化、GMM、架构选择、概率参数选择；
5. 严格排除 VB、q_true、stage、run_id、cut_index、time/progress 等标签/进度类变量；
6. 保留 TCN-GRU 多任务结构、5 个 fine-state 辅助任务、q_hat 主任务、early/late 双侧抑制与 ordered filtering；
7. 输出候选特征检查表，便于确认没有进度/标签泄露。

输入：
C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\1run_run_level_features\02_features\run_level_features_all.csv

输出：
C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\44run_fine_state_tcn_gru_strict_no_leak
"""

from pathlib import Path
import copy
import random
import warnings
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import expit
from scipy.stats import spearmanr

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.mixture import GaussianMixture

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# =========================================================
# 0. 全局配置
# =========================================================
ROOT = Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM")
FEATURE_FILE = ROOT / "PHM实验" / "1run_run_level_features" / "02_features" / "run_level_features_all.csv"

RUN_NAME = "44run_fine_state_tcn_gru_strict_no_leak"
RUN_DIR = ROOT / "PHM实验" / RUN_NAME

DIR_FINAL = RUN_DIR / "00_final_results"
DIR_INTERIM = RUN_DIR / "01_intermediate"
DIR_MODEL = RUN_DIR / "02_models"
DIR_FIG = RUN_DIR / "03_figures"
DIR_PRED = RUN_DIR / "04_predictions"

for d in [DIR_FINAL, DIR_INTERIM, DIR_MODEL, DIR_FIG, DIR_PRED]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STAGE_NAMES = ["early", "middle", "late"]
STAGE_TO_ID = {"early": 0, "middle": 1, "late": 2}
ID_TO_STAGE = {0: "early", 1: "middle", 2: "late"}

N_FINE_STATES = 5

# 阶段定义参数，仅用于生成评价标签/训练标签，不作为输入特征
Q_EARLY = 0.30
Q_LATE = 0.72
RATE_LATE_Q = 0.78

# grouped lifecycle validation
VAL_RATIO_STAGE = 0.20
MIN_STAGE_VAL_LEN = 8

# 特征选择
VAR_THRESHOLD = 1e-10
MAX_FEATURE_POOL = 260
N_SELECTED_FEATURES = 45
REDUNDANCY_THRESHOLD = 0.92

# 训练
BATCH_SIZE = 32
EPOCHS = 120
PATIENCE = 18
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

# 损失权重
LAMBDA_STAGE = 1.00
LAMBDA_FINE = 0.25
LAMBDA_Q = 0.30
LAMBDA_MONO = 0.03

# 架构搜索空间：为了运行时间可控，保留上版表现较好的候选
ARCH_LIST = [
    {"L": 12, "dropout": 0.20, "lr": 5e-4, "channels": (32, 64, 64), "gru_hidden": 64},
    {"L": 15, "dropout": 0.20, "lr": 5e-4, "channels": (32, 64, 64), "gru_hidden": 64},
    {"L": 15, "dropout": 0.20, "lr": 3e-4, "channels": (32, 64, 64), "gru_hidden": 64},
    {"L": 10, "dropout": 0.15, "lr": 5e-4, "channels": (16, 32, 32), "gru_hidden": 48},
    {"L": 12, "dropout": 0.15, "lr": 5e-4, "channels": (16, 32, 32), "gru_hidden": 48},
    {"L": 15, "dropout": 0.15, "lr": 5e-4, "channels": (16, 32, 32), "gru_hidden": 48},
]

# 概率融合参数搜索：只用 final_internal_val 选，不使用 C6
ETA_LIST = [0.55, 0.65, 0.75]
FINE_WEIGHT_LIST = [0.10, 0.20, 0.30]
TEMP_LIST = [1.0, 1.2]
MID_FLOOR_LIST = [0.04, 0.08, 0.12]
LATE_TAU_LIST = [0.60, 0.66, 0.72]
EARLY_TAU_LIST = [0.34, 0.38, 0.42]
ORDER_BLEND_LIST = [0.0, 0.25, 0.50]

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


# =========================================================
# 1. 工具函数
# =========================================================
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
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
    cols = list(df.columns)
    for c in ["VB", "VB_max", "vb", "vb_max"]:
        if c in cols:
            return c
    lower = {str(c).lower(): c for c in cols}
    for k in ["vb", "vb_max", "vbmax"]:
        if k in lower:
            return lower[k]
    raise ValueError("找不到 VB 或 VB_max 标签列。")


def is_meta_or_label_col(col):
    """
    严格排除标签列、进度列、阶段列、磨损列。
    注意：不要粗暴排除包含 time 的所有列，避免误伤 time-domain 特征；
    这里主要排除 exact name 和典型进度字段。
    """
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

    # 常见标签/阶段/退化字段
    forbidden_substrings = [
        "vb", "flute", "stage", "phase", "target", "label",
        "q_true", "q_deg", "q_hat", "q_norm",
        "rate_smooth", "rate_norm", "vb_smooth", "vb_norm",
        "progress", "life_ratio", "rul", "dvb", "delta_vb", "dominant",
    ]
    if any(k in c_clean for k in forbidden_substrings):
        return True

    # 进度字段：只排除明确带 run/cut/index/cycle/order/seq 的列
    progress_patterns = [
        r"(^|_)run(_|$)", r"(^|_)run_id(_|$)", r"(^|_)run_index(_|$)",
        r"(^|_)cut(_|$)", r"(^|_)cut_index(_|$)",
        r"(^|_)cycle(_|$)", r"(^|_)order(_|$)", r"(^|_)sequence(_|$)", r"(^|_)seq(_|$)",
        r"(^|_)timestamp(_|$)",
    ]
    if any(re.search(p, c_clean) for p in progress_patterns):
        return True

    return False


def calc_q_metrics(q_true, q_pred):
    q_true = np.asarray(q_true, dtype=float).reshape(-1)
    q_pred = np.asarray(q_pred, dtype=float).reshape(-1)
    mask = np.isfinite(q_true) & np.isfinite(q_pred)
    if mask.sum() < 3:
        return {"q_MAE": 999.0, "q_RMSE": 999.0, "q_R2": -999.0}
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
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
    )
    p_each, r_each, f1_each, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    row_sum = cm.sum(axis=1, keepdims=True) + 1e-12
    cm_norm = cm / row_sum
    true_ratio = np.bincount(y_true, minlength=3) / max(len(y_true), 1)
    pred_ratio = np.bincount(y_pred, minlength=3) / max(len(y_pred), 1)
    return {
        "acc": acc,
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


def make_classification_report_long(split, method, y_true, y_pred):
    rep = classification_report(
        y_true, y_pred, labels=[0, 1, 2], target_names=STAGE_NAMES,
        output_dict=True, zero_division=0
    )
    rows = []
    for label, d in rep.items():
        if isinstance(d, dict):
            rows.append({
                "split": split,
                "method": method,
                "label": label,
                "precision": d.get("precision", np.nan),
                "recall": d.get("recall", np.nan),
                "f1-score": d.get("f1-score", np.nan),
                "support": d.get("support", np.nan),
                "value": np.nan,
            })
        else:
            rows.append({
                "split": split,
                "method": method,
                "label": label,
                "precision": np.nan,
                "recall": np.nan,
                "f1-score": np.nan,
                "support": np.nan,
                "value": d,
            })
    return rows


def make_confusion_long(split, method, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    row_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    rows = []
    for i, ti in enumerate(STAGE_NAMES):
        for j, pj in enumerate(STAGE_NAMES):
            rows.append({
                "split": split,
                "method": method,
                "true_stage": ti,
                "pred_stage": pj,
                "count": int(cm[i, j]),
                "row_norm": float(row_norm[i, j]),
            })
    return rows


def make_ratio_rows(split, method, y_true, y_pred):
    true_ratio = np.bincount(y_true, minlength=3) / max(len(y_true), 1)
    pred_ratio = np.bincount(y_pred, minlength=3) / max(len(y_pred), 1)
    rows = []
    for i, name in enumerate(STAGE_NAMES):
        rows.append({"split": split, "method": "TRUE", "stage": name, "ratio": true_ratio[i]})
        rows.append({"split": split, "method": method, "stage": name, "ratio": pred_ratio[i]})
    return rows


# =========================================================
# 2. 数据读取、阶段标签、先划分
# =========================================================
def load_feature_table():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"找不到特征文件：{FEATURE_FILE}")
    df = pd.read_csv(FEATURE_FILE)
    df.columns = [str(c).strip() for c in df.columns]
    if "condition" not in df.columns or "run_id" not in df.columns:
        raise ValueError("特征表必须包含 condition 和 run_id。")
    df["condition"] = df["condition"].apply(normalize_condition_name)
    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce").astype(int)
    vb_col = infer_vb_column(df)
    df["VB"] = pd.to_numeric(df[vb_col], errors="coerce")
    df = df[df["condition"].isin(["C1", "C4", "C6"])].copy()
    df = df.dropna(subset=["VB"]).sort_values(["condition", "run_id"]).reset_index(drop=True)
    return df


def define_condition_relative_stages(df):
    out_parts = []
    th_rows = []
    for cond, sub in df.groupby("condition"):
        sub = sub.sort_values("run_id").reset_index(drop=True).copy()
        vb_smooth = sub["VB"].rolling(window=7, min_periods=1, center=True).mean()
        vb_min, vb_max = float(vb_smooth.min()), float(vb_smooth.max())
        q = (vb_smooth - vb_min) / (vb_max - vb_min + 1e-12)
        rate = q.diff().fillna(0.0)
        rate = rate.rolling(window=5, min_periods=1, center=True).mean()
        rate_norm = (rate - rate.min()) / (rate.max() - rate.min() + 1e-12)

        q_early_th = float(q.quantile(Q_EARLY))
        q_late_th = float(q.quantile(Q_LATE))
        rate_late_th = float(rate_norm.quantile(RATE_LATE_Q))

        stage = []
        for qi, ri in zip(q.values, rate_norm.values):
            if qi <= q_early_th:
                stage.append("early")
            elif (qi >= q_late_th) or (ri >= rate_late_th):
                stage.append("late")
            else:
                stage.append("middle")

        sub["VB_smooth"] = vb_smooth.values
        sub["q_true"] = q.values
        sub["rate_norm"] = rate_norm.values
        sub["stage"] = stage
        sub["stage_id"] = sub["stage"].map(STAGE_TO_ID).astype(int)
        # 这里只是初始细粒度标签，后续 final/proxy 内会用 train-only GMM 重写 fine_state_true
        sub["fine_state_true"] = pd.qcut(
            sub["q_true"].rank(method="first"),
            q=N_FINE_STATES,
            labels=False,
            duplicates="drop",
        ).astype(int)

        th_rows.append({
            "condition": cond,
            "q_early_quantile": Q_EARLY,
            "q_late_quantile": Q_LATE,
            "rate_late_quantile": RATE_LATE_Q,
            "q_early_th": q_early_th,
            "q_late_th": q_late_th,
            "rate_late_th": rate_late_th,
            "VB_min": float(sub["VB"].min()),
            "VB_max": float(sub["VB"].max()),
            "VB_smooth_min": vb_min,
            "VB_smooth_max": vb_max,
            "stage_count": str(sub["stage"].value_counts().reindex(STAGE_NAMES, fill_value=0).to_dict()),
            "fine_state_count": str(sub["fine_state_true"].value_counts().sort_index().to_dict()),
        })
        out_parts.append(sub)

    out = pd.concat(out_parts, axis=0).sort_values(["condition", "run_id"]).reset_index(drop=True)
    th_df = pd.DataFrame(th_rows)
    th_df.to_csv(DIR_FINAL / "FINAL_condition_relative_stage_thresholds.csv", index=False, encoding="utf-8-sig")
    out.to_csv(DIR_INTERIM / "loaded_with_condition_relative_stage.csv", index=False, encoding="utf-8-sig")
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
            block_len = max(MIN_STAGE_VAL_LEN, int(round(len(gs) * VAL_RATIO_STAGE)))
            block_len = min(block_len, max(len(gs) - 2, 1))
            start = max(0, (len(gs) - block_len) // 2)
            chosen = gs.iloc[start:start + block_len].index.tolist()
            val_idx.extend(chosen)
        val_idx = sorted(set(val_idx))
        val_parts.append(sub.loc[val_idx].copy())
        train_parts.append(sub.drop(index=val_idx).copy())

    final_train = pd.concat(train_parts, axis=0).sort_values(["condition", "run_id"]).reset_index(drop=True)
    final_val = pd.concat(val_parts, axis=0).sort_values(["condition", "run_id"]).reset_index(drop=True)
    test_c6 = df[df["condition"] == "C6"].sort_values("run_id").reset_index(drop=True).copy()

    rows = []
    for name, sub in [("final_train", final_train), ("final_internal_val", final_val), ("test_C6", test_c6)]:
        for cond, g in sub.groupby("condition"):
            rows.append({
                "split": name,
                "condition": cond,
                "n_samples": len(g),
                "run_start": int(g["run_id"].min()),
                "run_end": int(g["run_id"].max()),
                "VB_min": float(g["VB"].min()),
                "VB_max": float(g["VB"].max()),
                "stage_count": str(g["stage"].value_counts().reindex(STAGE_NAMES, fill_value=0).to_dict()),
            })
    split_df = pd.DataFrame(rows)
    split_df.to_csv(DIR_FINAL / "FINAL_split_metadata.csv", index=False, encoding="utf-8-sig")
    return final_train, final_val, test_c6, split_df


# =========================================================
# 3. 严格 split-safe online features 与特征选择
# =========================================================
def get_raw_numeric_sensor_cols(df_for_columns):
    cols = []
    for col in df_for_columns.columns:
        if is_meta_or_label_col(col):
            continue
        s = pd.to_numeric(df_for_columns[col], errors="coerce")
        if s.notna().mean() > 0.95:
            cols.append(col)
    return cols


def build_online_features_for_subset(df_subset, raw_cols, split_name):
    """
    对单个 split 单独构造 online features。
    只使用该 split 内当前及历史传感器值，不使用其他 split 的传感器分布信息。
    对 test_C6 来说，相当于只使用 C6 在线历史，不用 C6 未来、不用 C6 标签。
    """
    out_parts = []
    for cond, sub in df_subset.groupby("condition"):
        sub = sub.sort_values("run_id").reset_index(drop=True).copy()
        feat_part = sub[["condition", "run_id"]].copy()
        for col in raw_cols:
            x = pd.to_numeric(sub[col], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
            global_med = np.nanmedian(x.values)
            if not np.isfinite(global_med):
                global_med = 0.0
            filled, hist = [], []
            for v in x.values:
                if np.isfinite(v):
                    hist.append(float(v))
                    filled.append(float(v))
                else:
                    filled.append(float(np.median(hist)) if hist else float(global_med))
            x = pd.Series(filled, dtype=float)

            exp_mean = x.expanding(min_periods=3).mean()
            exp_std = x.expanding(min_periods=3).std().replace(0, np.nan)
            rel = ((x - exp_mean) / (exp_std + 1e-8)).fillna(0.0)
            slope = x.diff().fillna(0.0)

            ranks, hist = [], []
            for v in x.values:
                hist.append(v)
                arr = np.asarray(hist, dtype=float)
                ranks.append(float((arr <= v).mean()))
            online_rank = pd.Series(ranks, dtype=float)

            feat_part[f"{col}__rel"] = rel.values
            feat_part[f"{col}__slope"] = slope.values
            feat_part[f"{col}__online_rank"] = online_rank.values
        out_parts.append(feat_part)

    feat_only = pd.concat(out_parts, axis=0).sort_values(["condition", "run_id"]).reset_index(drop=True)
    meta_cols = [
        "condition", "run_id", "VB", "VB_smooth", "q_true", "rate_norm",
        "stage", "stage_id", "fine_state_true"
    ]
    merged = df_subset[meta_cols].copy().merge(feat_only, on=["condition", "run_id"], how="left")
    merged["split_name_for_feature_build"] = split_name
    return merged.replace([np.inf, -np.inf], np.nan)


def build_online_features_by_split(split_dict, raw_cols):
    parts = []
    for split_name, sub in split_dict.items():
        parts.append(build_online_features_for_subset(sub, raw_cols, split_name))
    return pd.concat(parts, axis=0).sort_values(["condition", "run_id", "split_name_for_feature_build"]).reset_index(drop=True)


def fill_features_by_train_median(feat_train, feat_apply, feature_cols=None):
    if feature_cols is None:
        feature_cols = [
            c for c in feat_train.columns
            if c not in ["condition", "run_id", "VB", "VB_smooth", "q_true", "rate_norm", "stage", "stage_id", "fine_state_true", "split_name_for_feature_build"]
        ]
    medians = {}
    out_train = feat_train.copy()
    out_apply = feat_apply.copy()
    for c in feature_cols:
        s = pd.to_numeric(out_train[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = s.median()
        if not np.isfinite(med):
            med = 0.0
        medians[c] = med
        out_train[c] = s.fillna(med)
        out_apply[c] = pd.to_numeric(out_apply[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med)
    return out_train, out_apply, medians


def feature_selection_condition_invariant_from_train(feat_train, out_prefix="FINAL"):
    candidate_cols = [
        c for c in feat_train.columns
        if c not in ["condition", "run_id", "VB", "VB_smooth", "q_true", "rate_norm", "stage", "stage_id", "fine_state_true", "split_name_for_feature_build"]
    ]
    pd.DataFrame({"online_candidate_feature": candidate_cols}).to_csv(
        DIR_FINAL / f"{out_prefix}_CHECK_online_candidate_features.csv",
        index=False,
        encoding="utf-8-sig",
        )

    ft = feat_train.copy()
    for c in candidate_cols:
        ft[c] = pd.to_numeric(ft[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = ft[c].median()
        if not np.isfinite(med):
            med = 0.0
        ft[c] = ft[c].fillna(med)

    vt_cols = []
    for c in candidate_cols:
        if ft[c].nunique(dropna=False) <= 1:
            continue
        if np.nanvar(ft[c].values) <= VAR_THRESHOLD:
            continue
        vt_cols.append(c)

    X = ft[vt_cols].values
    y_stage = ft["stage_id"].values.astype(int)
    y_q = ft["q_true"].values.astype(float)

    mi_stage = mutual_info_classif(X, y_stage, random_state=RANDOM_SEED)
    mi_q = mutual_info_regression(X, y_q, random_state=RANDOM_SEED)

    rows = []
    for i, c in enumerate(vt_cols):
        x = ft[c].values.astype(float)
        rho, _ = spearmanr(x, y_q)
        rho = 0.0 if not np.isfinite(rho) else abs(float(rho))

        # 如果只有一个工况训练，例如 proxy C1->C4，domain_instability 置 0
        conds = sorted(ft["condition"].unique())
        if len(conds) >= 2:
            g0 = ft[ft["condition"] == conds[0]][c].values
            g1 = ft[ft["condition"] == conds[1]][c].values
            instability = abs(np.nanmean(g0) - np.nanmean(g1)) / (0.5 * (np.nanstd(g0) + np.nanstd(g1)) + 1e-8)
            if not np.isfinite(instability):
                instability = 9.0
        else:
            instability = 0.0

        score = 1.10 * float(mi_stage[i]) + 0.35 * float(mi_q[i]) + 0.55 * rho - 0.35 * float(instability)
        rows.append({
            "feature": c,
            "mi_stage": float(mi_stage[i]),
            "mi_q": float(mi_q[i]),
            "spearman_abs_q": rho,
            "domain_instability_train_conditions": float(instability),
            "feature_score": float(score),
        })

    score_df = pd.DataFrame(rows).sort_values("feature_score", ascending=False).reset_index(drop=True)
    score_df.to_csv(DIR_INTERIM / f"{out_prefix}_feature_score_all_train_only.csv", index=False, encoding="utf-8-sig")
    score_df = score_df.head(MAX_FEATURE_POOL).copy()

    selected, selected_records = [], []
    for _, row in score_df.iterrows():
        feat = row["feature"]
        if not selected:
            max_corr = np.nan
            ok = True
        else:
            corrs = []
            x = ft[feat].values.astype(float)
            for sf in selected:
                y = ft[sf].values.astype(float)
                rho, _ = spearmanr(x, y)
                corrs.append(abs(rho) if np.isfinite(rho) else 0.0)
            max_corr = max(corrs) if corrs else 0.0
            ok = max_corr < REDUNDANCY_THRESHOLD
        if ok:
            selected.append(feat)
            rec = row.to_dict()
            rec["selected_rank"] = len(selected)
            rec["max_abs_corr_with_selected"] = max_corr
            selected_records.append(rec)
        if len(selected) >= N_SELECTED_FEATURES:
            break

    selected_df = pd.DataFrame(selected_records)
    if not selected_df.empty:
        selected_df = selected_df[
            ["selected_rank", "feature", "mi_stage", "mi_q", "spearman_abs_q",
             "domain_instability_train_conditions", "feature_score", "max_abs_corr_with_selected"]
        ]
    if out_prefix == "FINAL":
        selected_df.to_csv(DIR_FINAL / "FINAL_selected_stage_invariant_features.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{
            "raw_candidate_features": len(candidate_cols),
            "after_variance_filter": len(vt_cols),
            "selected_stage_invariant_features": len(selected),
        }]).to_csv(DIR_FINAL / "FINAL_feature_count_summary.csv", index=False, encoding="utf-8-sig")
    return selected, selected_df


def fit_scaler_on_train(feat_train, selected_features):
    scaler = StandardScaler()
    scaler.fit(feat_train[selected_features].values)
    return scaler


def transform_with_scaler(feat_df, selected_features, scaler):
    out = feat_df.copy()
    out[selected_features] = scaler.transform(out[selected_features].values)
    out[selected_features] = np.nan_to_num(out[selected_features].values, nan=0.0, posinf=0.0, neginf=0.0)
    return out


# =========================================================
# 4. Train-only GMM 细粒度状态
# =========================================================
def fit_train_only_gmm(feat_train, out_prefix="FINAL"):
    X = feat_train[["q_true", "rate_norm"]].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    gmm = GaussianMixture(
        n_components=N_FINE_STATES,
        covariance_type="full",
        random_state=RANDOM_SEED,
        reg_covar=1e-5,
        n_init=10,
    )
    gmm.fit(X)
    comp = gmm.predict(X)
    tmp = pd.DataFrame({"raw_component": comp, "q": feat_train["q_true"].values})
    mean_q = tmp.groupby("raw_component")["q"].mean().sort_values()
    raw_to_order = {raw: i for i, raw in enumerate(mean_q.index.tolist())}
    rows = []
    for raw, order in raw_to_order.items():
        sid = 0 if order == 0 else (2 if order == N_FINE_STATES - 1 else 1)
        rows.append({
            "raw_component": int(raw),
            "ordered_fine_state": int(order),
            "mean_q_train": float(mean_q.loc[raw]),
            "mapped_stage_id": int(sid),
            "mapped_stage": ID_TO_STAGE[sid],
        })
    map_df = pd.DataFrame(rows).sort_values("ordered_fine_state")
    if out_prefix == "FINAL":
        map_df.to_csv(DIR_FINAL / "FINAL_gmm_fine_state_mapping.csv", index=False, encoding="utf-8-sig")
    return gmm, raw_to_order, map_df


def assign_gmm_fine_states(feat_df, gmm, raw_to_order):
    out = feat_df.copy()
    X = out[["q_true", "rate_norm"]].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    raw = gmm.predict(X)
    out["fine_state_true"] = np.array([raw_to_order[int(r)] for r in raw], dtype=int)
    return out


# =========================================================
# 5. 窗口构建
# =========================================================
def build_windows(df_sub, selected_features, window_length, split_name):
    df_sub = df_sub.sort_values(["condition", "run_id"]).reset_index(drop=True)
    X_list, y_stage, y_fine, y_q, meta = [], [], [], [], []
    for cond, sub in df_sub.groupby("condition"):
        sub = sub.sort_values("run_id").reset_index(drop=True)
        Xv = sub[selected_features].values.astype(np.float32)
        stage = sub["stage_id"].values.astype(int)
        fine = sub["fine_state_true"].values.astype(int)
        q = sub["q_true"].values.astype(np.float32)
        run_ids = sub["run_id"].values.astype(int)
        for end in range(window_length - 1, len(sub)):
            start = end - window_length + 1
            rw = run_ids[start:end + 1]
            if not np.all(np.diff(rw) == 1):
                continue
            X_list.append(Xv[start:end + 1])
            y_stage.append(stage[end])
            y_fine.append(fine[end])
            y_q.append(q[end])
            meta.append({
                "condition": cond,
                "run_id_start": int(rw[0]),
                "run_id_end": int(rw[-1]),
                "cut_index": int(rw[-1]),
                "VB_true": float(sub["VB"].iloc[end]),
                "q_true": float(q[end]),
                "stage_true_id": int(stage[end]),
                "stage_true": ID_TO_STAGE[int(stage[end])],
                "fine_state_true": int(fine[end]),
                "window_length": int(window_length),
                "split": split_name,
            })
    return (
        np.asarray(X_list, dtype=np.float32),
        np.asarray(y_stage, dtype=np.int64),
        np.asarray(y_fine, dtype=np.int64),
        np.asarray(y_q, dtype=np.float32),
        pd.DataFrame(meta),
    )


class StageDataset(Dataset):
    def __init__(self, X, y_stage, y_fine, y_q):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_stage = torch.tensor(y_stage, dtype=torch.long)
        self.y_fine = torch.tensor(y_fine, dtype=torch.long)
        self.y_q = torch.tensor(y_q, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_stage[idx], self.y_fine[idx], self.y_q[idx]


def make_pack(df_sub, selected_features, L, split_name):
    X, ys, yf, yq, meta = build_windows(df_sub, selected_features, L, split_name)
    ds = StageDataset(X, ys, yf, yq)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(split_name in ["train", "final_train"]))
    return {"X": X, "ys": ys, "yf": yf, "yq": yq, "meta": meta, "loader": loader}


# =========================================================
# 6. 模型
# =========================================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x if self.chomp_size == 0 else x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        if out.size(-1) != res.size(-1):
            m = min(out.size(-1), res.size(-1))
            out = out[:, :, :m]
            res = res[:, :, :m]
        return self.relu(out + res)


class TCNGRUMultiTask(nn.Module):
    def __init__(self, input_dim, channels=(32, 64, 64), gru_hidden=64, dropout=0.2):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(channels):
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size=3, dilation=2 ** i, dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.gru = nn.GRU(
            input_size=channels[-1],
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.shared = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.stage_head = nn.Linear(64, 3)
        self.fine_head = nn.Linear(64, N_FINE_STATES)
        self.q_head = nn.Linear(64, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.tcn(x)
        h = h.transpose(1, 2)
        gru_out, _ = self.gru(h)
        z = self.shared(gru_out[:, -1, :])
        stage_logits = self.stage_head(z)
        fine_logits = self.fine_head(z)
        q_hat = torch.sigmoid(self.q_head(z))
        stage_prob = F.softmax(stage_logits, dim=1)
        fine_prob = F.softmax(fine_logits, dim=1)
        stage_prob = torch.nan_to_num(stage_prob, nan=1 / 3, posinf=1.0, neginf=0.0)
        fine_prob = torch.nan_to_num(fine_prob, nan=1 / N_FINE_STATES, posinf=1.0, neginf=0.0)
        stage_prob = stage_prob / (stage_prob.sum(dim=1, keepdim=True) + 1e-12)
        fine_prob = fine_prob / (fine_prob.sum(dim=1, keepdim=True) + 1e-12)
        q_hat = torch.nan_to_num(q_hat, nan=0.5, posinf=1.0, neginf=0.0)
        return {
            "stage_logits": stage_logits,
            "fine_logits": fine_logits,
            "stage_prob": stage_prob,
            "fine_prob": fine_prob,
            "q_hat": q_hat,
        }


# =========================================================
# 7. 训练与预测
# =========================================================
def class_weights(y, n_classes):
    cnt = np.bincount(y, minlength=n_classes).astype(float)
    w = cnt.sum() / (n_classes * np.maximum(cnt, 1.0))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def monotonic_q_loss(q_hat):
    if q_hat.numel() <= 2:
        return torch.tensor(0.0, device=q_hat.device)
    dq = q_hat[1:] - q_hat[:-1]
    return torch.relu(-dq).mean()


def run_epoch(model, loader, optimizer=None, stage_w=None, fine_w=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    losses = []
    all_stage_prob, all_fine_prob, all_q_hat = [], [], []
    all_ys, all_yf, all_yq = [], [], []
    for X, ys, yf, yq in loader:
        X = X.to(DEVICE)
        ys = ys.to(DEVICE)
        yf = yf.to(DEVICE)
        yq = yq.to(DEVICE)
        with torch.set_grad_enabled(is_train):
            out = model(X)
            loss_stage = F.cross_entropy(out["stage_logits"], ys, weight=stage_w)
            loss_fine = F.cross_entropy(out["fine_logits"], yf, weight=fine_w)
            loss_q = F.smooth_l1_loss(out["q_hat"], yq)
            loss_mono = monotonic_q_loss(out["q_hat"])
            loss = LAMBDA_STAGE * loss_stage + LAMBDA_FINE * loss_fine + LAMBDA_Q * loss_q + LAMBDA_MONO * loss_mono
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                optimizer.step()
        losses.append(float(loss.detach().cpu()) if torch.isfinite(loss) else 999.0)
        sp = out["stage_prob"].detach().cpu().numpy()
        fp = out["fine_prob"].detach().cpu().numpy()
        qh = out["q_hat"].detach().cpu().numpy().reshape(-1)
        sp = np.nan_to_num(sp, nan=1 / 3, posinf=1.0, neginf=0.0)
        fp = np.nan_to_num(fp, nan=1 / N_FINE_STATES, posinf=1.0, neginf=0.0)
        qh = np.clip(np.nan_to_num(qh, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
        sp = sp / (sp.sum(axis=1, keepdims=True) + 1e-12)
        fp = fp / (fp.sum(axis=1, keepdims=True) + 1e-12)
        all_stage_prob.append(sp)
        all_fine_prob.append(fp)
        all_q_hat.append(qh)
        all_ys.append(ys.detach().cpu().numpy())
        all_yf.append(yf.detach().cpu().numpy())
        all_yq.append(yq.detach().cpu().numpy().reshape(-1))
    if len(all_ys) == 0:
        return {
            "loss": 999.0,
            "stage_prob": np.zeros((0, 3)),
            "fine_prob": np.zeros((0, N_FINE_STATES)),
            "q_hat": np.zeros(0),
            "ys": np.zeros(0, dtype=int),
            "yf": np.zeros(0, dtype=int),
            "yq": np.zeros(0),
        }
    return {
        "loss": float(np.mean(losses)) if losses else 999.0,
        "stage_prob": np.concatenate(all_stage_prob, axis=0),
        "fine_prob": np.concatenate(all_fine_prob, axis=0),
        "q_hat": np.concatenate(all_q_hat, axis=0),
        "ys": np.concatenate(all_ys, axis=0),
        "yf": np.concatenate(all_yf, axis=0),
        "yq": np.concatenate(all_yq, axis=0),
    }


def train_model(train_pack, val_pack, input_dim, arch):
    set_seed(RANDOM_SEED)
    model = TCNGRUMultiTask(
        input_dim=input_dim,
        channels=arch["channels"],
        gru_hidden=arch["gru_hidden"],
        dropout=arch["dropout"],
    ).to(DEVICE)
    stage_w = class_weights(train_pack["ys"], 3)
    fine_w = class_weights(train_pack["yf"], N_FINE_STATES)
    opt = torch.optim.AdamW(model.parameters(), lr=arch["lr"], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=1e-5)
    best_state, best_score, best_epoch, patience_count = None, np.inf, 0, 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_pack["loader"], optimizer=opt, stage_w=stage_w, fine_w=fine_w)
        va = run_epoch(model, val_pack["loader"], optimizer=None, stage_w=stage_w, fine_w=fine_w)
        pred = np.argmax(va["stage_prob"], axis=1)
        m = clf_metrics(va["ys"], pred)
        q_m = calc_q_metrics(va["yq"], va["q_hat"])
        score = (
                1.0 * (1 - m["acc"])
                + 1.0 * (1 - m["f1"])
                + 1.4 * (1 - m["middle_recall"])
                + 0.8 * m["middle_to_late_rate"]
                + 0.7 * m["middle_to_early_rate"]
                + 0.25 * m["ratio_penalty"]
                + 0.20 * q_m["q_RMSE"]
        )
        scheduler.step(score)
        history.append({
            "epoch": epoch,
            "train_loss": tr["loss"],
            "val_loss": va["loss"],
            "val_acc": m["acc"],
            "val_f1": m["f1"],
            "val_middle_recall": m["middle_recall"],
            "val_middle_to_early_rate": m["middle_to_early_rate"],
            "val_middle_to_late_rate": m["middle_to_late_rate"],
            "val_ratio_penalty": m["ratio_penalty"],
            "val_q_RMSE": q_m["q_RMSE"],
            "score": score,
        })
        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_score, best_epoch


def predict_model(model, pack):
    out = run_epoch(model, pack["loader"], optimizer=None)
    pred_df = pack["meta"].copy().reset_index(drop=True)
    pred_df["q_hat"] = np.clip(out["q_hat"], 0.0, 1.0)
    pred_df["q_true_model"] = out["yq"]
    for i, st in enumerate(STAGE_NAMES):
        pred_df[f"raw_prob_{st}"] = out["stage_prob"][:, i]
    for i in range(N_FINE_STATES):
        pred_df[f"fine_prob_{i}"] = out["fine_prob"][:, i]
    pred_df["stage_pred_raw"] = np.argmax(out["stage_prob"], axis=1)
    pred_df["stage_pred_raw_name"] = pred_df["stage_pred_raw"].map(ID_TO_STAGE)
    pred_df["fine_pred_raw"] = np.argmax(out["fine_prob"], axis=1)
    return pred_df


# =========================================================
# 8. 概率推理
# =========================================================
def qhat_prior(q_hat, early_tau, late_tau, mid_floor):
    q = np.asarray(q_hat, dtype=float).reshape(-1)
    q = np.clip(np.nan_to_num(q, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
    c_early, c_mid, c_late = 0.18, 0.50, 0.84
    sigma = 0.17
    pe = np.exp(-0.5 * ((q - c_early) / sigma) ** 2)
    pm = np.exp(-0.5 * ((q - c_mid) / sigma) ** 2)
    pl = np.exp(-0.5 * ((q - c_late) / sigma) ** 2)
    early_gate = expit(EARLY_SUPPRESS_K * (early_tau - q))
    late_gate = expit(LATE_SUPPRESS_K * (q - late_tau))
    pe = pe * early_gate
    pl = pl * late_gate
    pm = np.maximum(pm, mid_floor)
    P = np.vstack([pe, pm, pl]).T
    P = np.nan_to_num(P, nan=1 / 3, posinf=1.0, neginf=0.0)
    return P / (P.sum(axis=1, keepdims=True) + 1e-12)


def fine_to_stage_prob(pred_df):
    fine_cols = [f"fine_prob_{i}" for i in range(N_FINE_STATES)]
    Fp = pred_df[fine_cols].values.astype(float)
    Fp = np.nan_to_num(Fp, nan=1 / N_FINE_STATES, posinf=1.0, neginf=0.0)
    Fp = Fp / (Fp.sum(axis=1, keepdims=True) + 1e-12)
    P = np.zeros((len(pred_df), 3), dtype=float)
    P[:, 0] = Fp[:, 0]
    P[:, 1] = Fp[:, 1] + Fp[:, 2] + Fp[:, 3]
    P[:, 2] = Fp[:, 4]
    return P / (P.sum(axis=1, keepdims=True) + 1e-12)


def temperature_scale(P, temp):
    P = np.asarray(P, dtype=float)
    P = np.clip(P, 1e-12, 1.0)
    logp = np.log(P) / temp
    logp = logp - logp.max(axis=1, keepdims=True)
    out = np.exp(logp)
    return out / (out.sum(axis=1, keepdims=True) + 1e-12)


def causal_ordered_filter(P, trans=None):
    P = np.asarray(P, dtype=float)
    P = np.nan_to_num(P, nan=1 / 3, posinf=1.0, neginf=0.0)
    P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
    if trans is None:
        trans = np.array([
            [0.9400, 0.0550, 0.0050],
            [0.0050, 0.9550, 0.0400],
            [0.0005, 0.0100, 0.9895],
        ], dtype=float)
    prior = np.array([0.90, 0.10, 1e-6], dtype=float)
    prior = prior / prior.sum()
    out = np.zeros_like(P)
    alpha = prior * P[0]
    alpha = alpha / (alpha.sum() + 1e-12)
    out[0] = alpha
    for t in range(1, len(P)):
        alpha = (alpha @ trans) * P[t]
        alpha = alpha / (alpha.sum() + 1e-12)
        out[t] = alpha
    return out


def apply_probability_inference(pred_df, params):
    df = pred_df.copy().sort_values(["condition", "cut_index"]).reset_index(drop=True)
    raw = df[[f"raw_prob_{s}" for s in STAGE_NAMES]].values.astype(float)
    raw = temperature_scale(raw, params["temperature"])
    fine_stage = fine_to_stage_prob(df)
    prior = qhat_prior(
        df["q_hat"].values,
        early_tau=params["early_tau"],
        late_tau=params["late_tau"],
        mid_floor=params["mid_floor"],
    )
    aux = params["fine_weight"] * fine_stage + (1.0 - params["fine_weight"]) * prior
    aux = aux / (aux.sum(axis=1, keepdims=True) + 1e-12)
    mix = params["eta"] * raw + (1.0 - params["eta"]) * aux
    mix = mix / (mix.sum(axis=1, keepdims=True) + 1e-12)
    ordered = np.zeros_like(mix)
    for cond, idx in df.groupby("condition").groups.items():
        idx = list(idx)
        ordered[idx] = causal_ordered_filter(mix[idx])
    final = (1.0 - params["order_blend"]) * mix + params["order_blend"] * ordered
    final = final / (final.sum(axis=1, keepdims=True) + 1e-12)
    for i, st in enumerate(STAGE_NAMES):
        df[f"prior_prob_{st}"] = prior[:, i]
        df[f"mix_prob_{st}"] = mix[:, i]
        df[f"ordered_prob_{st}"] = ordered[:, i]
        df[f"final_prob_{st}"] = final[:, i]
    for method, arr in [("prior", prior), ("mix", mix), ("ordered", ordered), ("final", final)]:
        df[f"stage_pred_{method}"] = np.argmax(arr, axis=1)
        df[f"stage_pred_{method}_name"] = df[f"stage_pred_{method}"].map(ID_TO_STAGE)
    return df


def evaluate_pred_df(pred_df, split_name):
    rows_report, rows_cm, rows_ratio = [], [], []
    y_true = pred_df["stage_true_id"].values.astype(int)
    metrics = {}
    for method in ["raw", "prior", "mix", "ordered", "final"]:
        pred_col = "stage_pred_raw" if method == "raw" else f"stage_pred_{method}"
        if pred_col not in pred_df.columns:
            continue
        y_pred = pred_df[pred_col].values.astype(int)
        m = clf_metrics(y_true, y_pred)
        metrics[method] = m
        rows_report.extend(make_classification_report_long(split_name, method, y_true, y_pred))
        rows_cm.extend(make_confusion_long(split_name, method, y_true, y_pred))
        rows_ratio.extend(make_ratio_rows(split_name, method, y_true, y_pred))
    q_m = calc_q_metrics(pred_df["q_true_model"].values, pred_df["q_hat"].values)
    return metrics, q_m, rows_report, rows_cm, rows_ratio


def probability_param_search(pred_val_raw):
    rows = []
    for eta in ETA_LIST:
        for fine_w in FINE_WEIGHT_LIST:
            for temp in TEMP_LIST:
                for mid_floor in MID_FLOOR_LIST:
                    for late_tau in LATE_TAU_LIST:
                        for early_tau in EARLY_TAU_LIST:
                            for order_blend in ORDER_BLEND_LIST:
                                params = {
                                    "eta": eta,
                                    "fine_weight": fine_w,
                                    "temperature": temp,
                                    "mid_floor": mid_floor,
                                    "late_tau": late_tau,
                                    "early_tau": early_tau,
                                    "order_blend": order_blend,
                                }
                                pred_val = apply_probability_inference(pred_val_raw, params)
                                metrics, q_m, _, _, _ = evaluate_pred_df(pred_val, "final_internal_val")
                                m = metrics["final"]
                                mo = metrics["ordered"]
                                score = (
                                        1.0 * (1 - m["acc"])
                                        + 1.0 * (1 - m["f1"])
                                        + 1.4 * (1 - m["middle_recall"])
                                        + 0.8 * m["middle_to_late_rate"]
                                        + 0.7 * m["middle_to_early_rate"]
                                        + 0.4 * m["ratio_penalty"]
                                        + 0.25 * (1 - mo["middle_recall"])
                                        + 0.15 * q_m["q_RMSE"]
                                )
                                rows.append({
                                    **params,
                                    "val_score": score,
                                    "val_final_acc": m["acc"],
                                    "val_final_f1": m["f1"],
                                    "val_final_middle_precision": m["middle_precision"],
                                    "val_final_middle_recall": m["middle_recall"],
                                    "val_final_middle_f1": m["middle_f1"],
                                    "val_final_ratio_penalty": m["ratio_penalty"],
                                    "val_final_middle_to_early_rate": m["middle_to_early_rate"],
                                    "val_final_middle_to_late_rate": m["middle_to_late_rate"],
                                    "val_ordered_acc": mo["acc"],
                                    "val_ordered_f1": mo["f1"],
                                    "val_ordered_middle_recall": mo["middle_recall"],
                                    **q_m,
                                })
    res = pd.DataFrame(rows).sort_values("val_score", ascending=True).reset_index(drop=True)
    res.to_csv(DIR_FINAL / "FINAL_probability_param_ranking.csv", index=False, encoding="utf-8-sig")
    best = res.iloc[0].to_dict()
    params = {
        "eta": float(best["eta"]),
        "fine_weight": float(best["fine_weight"]),
        "temperature": float(best["temperature"]),
        "mid_floor": float(best["mid_floor"]),
        "late_tau": float(best["late_tau"]),
        "early_tau": float(best["early_tau"]),
        "order_blend": float(best["order_blend"]),
    }
    return params, res


# =========================================================
# 9. 嵌套代理验证
# =========================================================
def arch_name(arch):
    return (
        f"L{arch['L']}"
        f"_drop{str(arch['dropout']).replace('.', 'p')}"
        f"_lr{str(arch['lr']).replace('.', 'p')}"
        f"_ch{'-'.join(map(str, arch['channels']))}"
        f"_gru{arch['gru_hidden']}"
    )


def prepare_no_leak_train_val_test(train_raw, val_raw, raw_sensor_cols, out_prefix="TMP"):
    """
    对一个 train/val 设置执行完整无泄露预处理：
    - train、val 分别构造 online features；
    - 只用 train 选特征、填充、GMM、scaler；
    - val 只 transform 和评估。
    """
    split_feat = build_online_features_by_split({"train": train_raw, "val": val_raw}, raw_sensor_cols)
    feat_train = split_feat[split_feat["split_name_for_feature_build"] == "train"].copy()
    feat_val = split_feat[split_feat["split_name_for_feature_build"] == "val"].copy()

    # 初步填充所有候选，方便特征选择
    candidate_cols = [
        c for c in feat_train.columns
        if c not in ["condition", "run_id", "VB", "VB_smooth", "q_true", "rate_norm", "stage", "stage_id", "fine_state_true", "split_name_for_feature_build"]
    ]
    feat_train, feat_val, _ = fill_features_by_train_median(feat_train, feat_val, candidate_cols)

    selected, _ = feature_selection_condition_invariant_from_train(feat_train, out_prefix=out_prefix)
    feat_train, feat_val, _ = fill_features_by_train_median(feat_train, feat_val, selected)

    gmm, raw_to_order, _ = fit_train_only_gmm(feat_train, out_prefix=out_prefix)
    feat_train = assign_gmm_fine_states(feat_train, gmm, raw_to_order)
    feat_val = assign_gmm_fine_states(feat_val, gmm, raw_to_order)

    scaler = fit_scaler_on_train(feat_train, selected)
    feat_train = transform_with_scaler(feat_train, selected, scaler)
    feat_val = transform_with_scaler(feat_val, selected, scaler)

    return feat_train, feat_val, selected


def proxy_validate_arch(label_df, raw_sensor_cols, arch, train_cond, val_cond):
    train_raw = label_df[label_df["condition"] == train_cond].copy()
    val_raw = label_df[label_df["condition"] == val_cond].copy()
    feat_train, feat_val, selected = prepare_no_leak_train_val_test(
        train_raw, val_raw, raw_sensor_cols, out_prefix=f"PROXY_{train_cond}_to_{val_cond}"
    )
    tr_pack = make_pack(feat_train, selected, arch["L"], "train")
    va_pack = make_pack(feat_val, selected, arch["L"], "val")
    if len(tr_pack["X"]) < 20 or len(va_pack["X"]) < 20:
        return None
    model, hist, best_score, best_epoch = train_model(tr_pack, va_pack, len(selected), arch)
    pred_raw = predict_model(model, va_pack)
    default_params = {
        "eta": 0.65,
        "fine_weight": 0.20,
        "temperature": 1.0,
        "mid_floor": 0.08,
        "late_tau": 0.66,
        "early_tau": 0.38,
        "order_blend": 0.0,
    }
    pred = apply_probability_inference(pred_raw, default_params)
    metrics, q_m, _, _, _ = evaluate_pred_df(pred, "proxy")
    m = metrics["final"]
    score = (
            1.0 * (1 - m["acc"])
            + 1.0 * (1 - m["f1"])
            + 1.4 * (1 - m["middle_recall"])
            + 0.8 * m["middle_to_late_rate"]
            + 0.7 * m["middle_to_early_rate"]
            + 0.4 * m["ratio_penalty"]
            + 0.20 * q_m["q_RMSE"]
    )
    return {
        "score": score,
        "acc": m["acc"],
        "f1": m["f1"],
        "middle_recall": m["middle_recall"],
        "middle_to_early_rate": m["middle_to_early_rate"],
        "middle_to_late_rate": m["middle_to_late_rate"],
        "q_RMSE": q_m["q_RMSE"],
        "best_epoch": best_epoch,
        "n_selected": len(selected),
    }


def run_proxy_arch_search(label_df, raw_sensor_cols):
    rows = []
    for arch in ARCH_LIST:
        name = arch_name(arch)
        print(f"\n[Strict nested proxy architecture] {name}")
        r14 = proxy_validate_arch(label_df, raw_sensor_cols, arch, "C1", "C4")
        r41 = proxy_validate_arch(label_df, raw_sensor_cols, arch, "C4", "C1")
        if r14 is None or r41 is None:
            continue
        proxy_mean = 0.5 * (r14["score"] + r41["score"])
        proxy_worst = max(r14["score"], r41["score"])
        proxy_final = 0.45 * proxy_mean + 0.55 * proxy_worst
        print(
            f"  C1->C4: acc={r14['acc']:.4f}, f1={r14['f1']:.4f}, middle_recall={r14['middle_recall']:.4f}, "
            f"M->E={r14['middle_to_early_rate']:.4f}, M->L={r14['middle_to_late_rate']:.4f}, score={r14['score']:.4f}"
        )
        print(
            f"  C4->C1: acc={r41['acc']:.4f}, f1={r41['f1']:.4f}, middle_recall={r41['middle_recall']:.4f}, "
            f"M->E={r41['middle_to_early_rate']:.4f}, M->L={r41['middle_to_late_rate']:.4f}, score={r41['score']:.4f}"
        )
        rows.append({
            "arch_name": name,
            "window_length": arch["L"],
            "dropout": arch["dropout"],
            "learning_rate": arch["lr"],
            "channels": str(arch["channels"]),
            "gru_hidden": arch["gru_hidden"],
            "proxy_mean_score": proxy_mean,
            "proxy_worst_score": proxy_worst,
            "proxy_final_score": proxy_final,
            "proxy_C1_to_C4_score": r14["score"],
            "proxy_C1_to_C4_acc": r14["acc"],
            "proxy_C1_to_C4_f1": r14["f1"],
            "proxy_C1_to_C4_middle_recall": r14["middle_recall"],
            "proxy_C1_to_C4_middle_to_early_rate": r14["middle_to_early_rate"],
            "proxy_C1_to_C4_middle_to_late_rate": r14["middle_to_late_rate"],
            "proxy_C4_to_C1_score": r41["score"],
            "proxy_C4_to_C1_acc": r41["acc"],
            "proxy_C4_to_C1_f1": r41["f1"],
            "proxy_C4_to_C1_middle_recall": r41["middle_recall"],
            "proxy_C4_to_C1_middle_to_early_rate": r41["middle_to_early_rate"],
            "proxy_C4_to_C1_middle_to_late_rate": r41["middle_to_late_rate"],
        })
    ranking = pd.DataFrame(rows).sort_values("proxy_final_score", ascending=True).reset_index(drop=True)
    ranking.to_csv(DIR_FINAL / "FINAL_proxy_model_ranking.csv", index=False, encoding="utf-8-sig")
    if ranking.empty:
        raise RuntimeError("代理验证没有成功训练任何架构。")
    best_name = ranking.iloc[0]["arch_name"]
    best_arch = next(a for a in ARCH_LIST if arch_name(a) == best_name)
    print(f"\nBest architecture by strict nested proxy validation: {best_name}")
    return best_arch, ranking


# =========================================================
# 10. 可视化
# =========================================================
def plot_stage_definition(label_df):
    colors = {"early": "#5B8FF9", "middle": "#61DDAA", "late": "#F6BD16"}
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    for ax, cond in zip(axes, ["C1", "C4", "C6"]):
        sub = label_df[label_df["condition"] == cond].sort_values("run_id")
        ax.plot(sub["run_id"], sub["VB"], color="black", linewidth=2.0, label="VB")
        ax.plot(sub["run_id"], sub["VB_smooth"], color="gray", linewidth=1.6, linestyle="--", label="Smoothed VB")
        for st in STAGE_NAMES:
            g = sub[sub["stage"] == st]
            ax.scatter(g["run_id"], g["VB"], s=18, color=colors[st], label=st, alpha=0.85, edgecolors="none")
        ax.set_title(f"{cond}: condition-relative stage definition")
        ax.set_xlabel("Cut index")
        ax.set_ylabel("VB")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(ncol=4, fontsize=8, loc="upper left")
    savefig(DIR_FIG / "fig01_condition_stage_definition.png")


def plot_selected_features(selected_df):
    if selected_df is None or selected_df.empty:
        return
    df = selected_df.sort_values("feature_score", ascending=True)
    plt.figure(figsize=(9, max(5, 0.24 * len(df))))
    plt.barh(df["feature"], df["feature_score"], color="#5B8FF9", edgecolor="black", linewidth=0.3)
    plt.xlabel("Feature score")
    plt.ylabel("Selected feature")
    plt.title("Selected stage-sensitive and condition-invariant features")
    savefig(DIR_FIG / "fig02_selected_features.png")


def plot_confusion_heatmap(conf_df, split, method, fname):
    sub = conf_df[(conf_df["split"] == split) & (conf_df["method"] == method)].copy()
    if sub.empty:
        return
    mat = np.zeros((3, 3))
    cnt = np.zeros((3, 3), dtype=int)
    for _, r in sub.iterrows():
        i = STAGE_TO_ID[r["true_stage"]]
        j = STAGE_TO_ID[r["pred_stage"]]
        mat[i, j] = r["row_norm"]
        cnt[i, j] = int(r["count"])
    plt.figure(figsize=(5.6, 5.0))
    im = plt.imshow(mat, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, label="Row-normalized value")
    plt.xticks(range(3), STAGE_NAMES)
    plt.yticks(range(3), STAGE_NAMES)
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f"{cnt[i, j]}\n({mat[i, j]:.2f})", ha="center", va="center", fontsize=10)
    plt.xlabel("Predicted stage")
    plt.ylabel("True stage")
    plt.title(f"{split} - {method} confusion matrix")
    savefig(DIR_FIG / fname)


def plot_probability_evolution(pred_df, split, method):
    prob_cols = [f"{method}_prob_{s}" for s in STAGE_NAMES]
    if any(c not in pred_df.columns for c in prob_cols):
        return
    sub = pred_df[pred_df["split"] == split].sort_values(["condition", "cut_index"]).copy()
    for cond, g in sub.groupby("condition"):
        plt.figure(figsize=(13, 5))
        plt.plot(g["cut_index"], g[prob_cols[0]], linewidth=2, label="early probability")
        plt.plot(g["cut_index"], g[prob_cols[1]], linewidth=2, label="middle probability")
        plt.plot(g["cut_index"], g[prob_cols[2]], linewidth=2, label="late probability")
        plt.scatter(g["cut_index"], g["stage_true_id"] / 2.0, s=12, color="black", alpha=0.35, label="true stage / 2")
        plt.xlabel("Cut index")
        plt.ylabel("Probability")
        plt.ylim(-0.05, 1.05)
        plt.title(f"{cond} - {method} stage probability evolution")
        plt.legend(loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.3)
        savefig(DIR_FIG / f"fig_prob_{split}_{cond}_{method}.png")


def plot_qhat(pred_df, split):
    sub = pred_df[pred_df["split"] == split].sort_values(["condition", "cut_index"])
    for cond, g in sub.groupby("condition"):
        plt.figure(figsize=(13, 4.8))
        plt.plot(g["cut_index"], g["q_true_model"], color="black", linewidth=2.4, label="True q")
        plt.plot(g["cut_index"], g["q_hat"], color="#5B8FF9", linewidth=2.0, label="Predicted q_hat")
        plt.xlabel("Cut index")
        plt.ylabel("Normalized degradation position")
        plt.title(f"{cond} - q_hat prediction")
        plt.legend(loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.3)
        savefig(DIR_FIG / f"fig_qhat_{split}_{cond}.png")


def plot_stage_ratio(ratio_df):
    for split in ratio_df["split"].unique():
        sub = ratio_df[ratio_df["split"] == split]
        for method in [m for m in sub["method"].unique() if m != "TRUE"]:
            dfp = sub[sub["method"].isin(["TRUE", method])]
            pivot = dfp.pivot_table(index="stage", columns="method", values="ratio", aggfunc="mean").reindex(STAGE_NAMES)
            plt.figure(figsize=(6.5, 4.2))
            x = np.arange(len(STAGE_NAMES))
            width = 0.35
            plt.bar(x - width / 2, pivot["TRUE"].values, width, label="True")
            plt.bar(x + width / 2, pivot[method].values, width, label=method)
            plt.xticks(x, STAGE_NAMES)
            plt.ylabel("Stage ratio")
            plt.title(f"{split} - stage ratio comparison ({method})")
            plt.legend()
            plt.grid(True, axis="y", linestyle="--", alpha=0.3)
            savefig(DIR_FIG / f"fig_ratio_{split}_{method}.png")


def make_visualizations(label_df, selected_df, pred_all, conf_df, ratio_df):
    plot_stage_definition(label_df)
    plot_selected_features(selected_df)
    for method in ["raw", "mix", "ordered", "final"]:
        plot_confusion_heatmap(conf_df, "test_C6", method, f"fig_confusion_test_{method}.png")
        plot_probability_evolution(pred_all, "test_C6", method)
    plot_qhat(pred_all, "test_C6")
    plot_stage_ratio(ratio_df)


# =========================================================
# 11. 主流程
# =========================================================
def main():
    set_seed(RANDOM_SEED)
    print("=" * 120)
    print("44run：严格无泄露的细粒度退化状态辅助阶段概率识别")
    print("=" * 120)
    print(f"Device: {DEVICE}")
    print(f"Feature file: {FEATURE_FILE}")
    print(f"Run dir: {RUN_DIR}")

    # 1. 标签构造与 split。C6 标签只用于最终评价。
    raw_df = load_feature_table()
    label_df, threshold_df = define_condition_relative_stages(raw_df)
    final_train_raw, final_val_raw, test_c6_raw, split_df = split_grouped_lifecycle(label_df)

    # 2. 只基于 C1/C4 训练段确定可用传感器列名，避免把 C6 独有列纳入候选
    raw_sensor_cols = get_raw_numeric_sensor_cols(final_train_raw)
    pd.DataFrame({"raw_sensor_col": raw_sensor_cols}).to_csv(
        DIR_FINAL / "CHECK_raw_sensor_cols.csv", index=False, encoding="utf-8-sig"
    )
    print(f"候选原始传感器特征数：{len(raw_sensor_cols)}")

    # 3. 严格嵌套 proxy validation 选架构
    best_arch, proxy_ranking = run_proxy_arch_search(label_df[label_df["condition"].isin(["C1", "C4"])].copy(), raw_sensor_cols)

    # 4. final 无泄露预处理：train/val/test 分别构造 online features；只用 final_train 选特征、GMM、scaler
    split_feat = build_online_features_by_split(
        {
            "final_train": final_train_raw,
            "final_internal_val": final_val_raw,
            "test_C6": test_c6_raw,
        },
        raw_sensor_cols,
    )
    feat_train = split_feat[split_feat["split_name_for_feature_build"] == "final_train"].copy()
    feat_val = split_feat[split_feat["split_name_for_feature_build"] == "final_internal_val"].copy()
    feat_test = split_feat[split_feat["split_name_for_feature_build"] == "test_C6"].copy()

    candidate_cols = [
        c for c in feat_train.columns
        if c not in ["condition", "run_id", "VB", "VB_smooth", "q_true", "rate_norm", "stage", "stage_id", "fine_state_true", "split_name_for_feature_build"]
    ]
    feat_train, feat_val, _ = fill_features_by_train_median(feat_train, feat_val, candidate_cols)
    _, feat_test, _ = fill_features_by_train_median(feat_train, feat_test, candidate_cols)

    selected_features, selected_df = feature_selection_condition_invariant_from_train(feat_train, out_prefix="FINAL")
    feat_train, feat_val, _ = fill_features_by_train_median(feat_train, feat_val, selected_features)
    _, feat_test, _ = fill_features_by_train_median(feat_train, feat_test, selected_features)

    gmm, raw_to_order, gmm_map = fit_train_only_gmm(feat_train, out_prefix="FINAL")
    feat_train = assign_gmm_fine_states(feat_train, gmm, raw_to_order)
    feat_val = assign_gmm_fine_states(feat_val, gmm, raw_to_order)
    feat_test = assign_gmm_fine_states(feat_test, gmm, raw_to_order)

    scaler = fit_scaler_on_train(feat_train, selected_features)
    feat_train = transform_with_scaler(feat_train, selected_features, scaler)
    feat_val = transform_with_scaler(feat_val, selected_features, scaler)
    feat_test = transform_with_scaler(feat_test, selected_features, scaler)

    # 5. final 训练
    tr_pack = make_pack(feat_train, selected_features, best_arch["L"], "final_train")
    va_pack = make_pack(feat_val, selected_features, best_arch["L"], "final_internal_val")
    te_pack = make_pack(feat_test, selected_features, best_arch["L"], "test_C6")

    model, hist, best_score, best_epoch = train_model(tr_pack, va_pack, len(selected_features), best_arch)
    hist.to_csv(DIR_INTERIM / "final_model_training_history.csv", index=False, encoding="utf-8-sig")
    torch.save(model.state_dict(), DIR_MODEL / "final_best_model_state.pth")

    pred_val_raw = predict_model(model, va_pack)
    pred_test_raw = predict_model(model, te_pack)

    # 6. 只用 final_internal_val 选择概率参数
    best_params, prob_rank = probability_param_search(pred_val_raw)
    pred_val = apply_probability_inference(pred_val_raw, best_params)
    pred_test = apply_probability_inference(pred_test_raw, best_params)
    pred_all = pd.concat([pred_val, pred_test], axis=0).reset_index(drop=True)

    pred_val.to_csv(DIR_PRED / "FINAL_best_val_predictions.csv", index=False, encoding="utf-8-sig")
    pred_test.to_csv(DIR_FINAL / "FINAL_best_test_C6_predictions.csv", index=False, encoding="utf-8-sig")
    pred_all.to_csv(DIR_FINAL / "FINAL_all_predictions_val_and_test.csv", index=False, encoding="utf-8-sig")

    # 7. 评价
    report_rows, cm_rows, ratio_rows = [], [], []
    metrics_val, q_val, rep, cm, rr = evaluate_pred_df(pred_val, "final_internal_val")
    report_rows.extend(rep)
    cm_rows.extend(cm)
    ratio_rows.extend(rr)
    metrics_test, q_test, rep, cm, rr = evaluate_pred_df(pred_test, "test_C6")
    report_rows.extend(rep)
    cm_rows.extend(cm)
    ratio_rows.extend(rr)

    report_df = pd.DataFrame(report_rows)
    cm_df = pd.DataFrame(cm_rows)
    ratio_df = pd.DataFrame(ratio_rows)
    report_df.to_csv(DIR_FINAL / "FINAL_classification_reports_long.csv", index=False, encoding="utf-8-sig")
    cm_df.to_csv(DIR_FINAL / "FINAL_confusion_matrices_long.csv", index=False, encoding="utf-8-sig")
    ratio_df.to_csv(DIR_FINAL / "FINAL_stage_ratio_comparison.csv", index=False, encoding="utf-8-sig")

    mt_raw = metrics_test["raw"]
    mt_mix = metrics_test["mix"]
    mt_ordered = metrics_test["ordered"]
    mt_final = metrics_test["final"]

    summary = pd.DataFrame([{
        "experiment": RUN_NAME,
        "task": "Strict no-leak fine-grained degradation-state assisted stage probability classification",
        "feature_file": str(FEATURE_FILE),
        "train_conditions": "C1+C4 final_train blocks",
        "proxy_validation": "Strict nested C1->C4 and C4->C1; each fold has independent feature selection/scaling/GMM",
        "test_condition": "C6",
        "raw_numeric_sensor_feature_count": len(raw_sensor_cols),
        "online_candidate_features": len(candidate_cols),
        "selected_stage_invariant_features": len(selected_features),
        "stage_definition": "condition-relative q_deg + local rate; early/late explicitly defined, middle as stable degradation band",
        "feature_selection": "train-only MI(stage) + MI(q_deg) + Spearman(q_deg) - domain instability + redundancy filtering",
        "fine_state_strategy": "train-only GMM with 5 ordered fine states; S0->early, S1-S3->middle, S4->late",
        "model": "TCN-GRU multitask encoder with q_hat regression, 3-stage classification and 5-fine-state auxiliary classification",
        "probability_inference": "raw stage probability + fine-state mapped probability + q_hat prior + premature-early inhibition + late suppression + causal ordered filtering",
        "no_leakage_design": "C6 is only used for final evaluation; train/val/test online features are built separately; feature selection, scaling, GMM, model selection and probability selection use no C6",
        "best_arch_name": arch_name(best_arch),
        "best_window_length": best_arch["L"],
        "best_dropout": best_arch["dropout"],
        "best_learning_rate": best_arch["lr"],
        "best_channels": str(best_arch["channels"]),
        "best_gru_hidden": best_arch["gru_hidden"],
        "best_eta": best_params["eta"],
        "best_fine_weight": best_params["fine_weight"],
        "best_temperature": best_params["temperature"],
        "best_mid_floor": best_params["mid_floor"],
        "best_late_tau": best_params["late_tau"],
        "best_early_tau": best_params["early_tau"],
        "best_order_blend": best_params["order_blend"],
        "proxy_mean_score": float(proxy_ranking.iloc[0]["proxy_mean_score"]),
        "proxy_worst_score": float(proxy_ranking.iloc[0]["proxy_worst_score"]),
        "proxy_final_score": float(proxy_ranking.iloc[0]["proxy_final_score"]),
        "test_raw_acc": mt_raw["acc"],
        "test_raw_macro_f1": mt_raw["f1"],
        "test_raw_middle_precision": mt_raw["middle_precision"],
        "test_raw_middle_recall": mt_raw["middle_recall"],
        "test_raw_middle_f1": mt_raw["middle_f1"],
        "test_raw_middle_to_early_rate": mt_raw["middle_to_early_rate"],
        "test_raw_middle_to_late_rate": mt_raw["middle_to_late_rate"],
        "test_mix_acc": mt_mix["acc"],
        "test_mix_macro_f1": mt_mix["f1"],
        "test_mix_middle_precision": mt_mix["middle_precision"],
        "test_mix_middle_recall": mt_mix["middle_recall"],
        "test_mix_middle_f1": mt_mix["middle_f1"],
        "test_ordered_acc": mt_ordered["acc"],
        "test_ordered_macro_f1": mt_ordered["f1"],
        "test_ordered_middle_precision": mt_ordered["middle_precision"],
        "test_ordered_middle_recall": mt_ordered["middle_recall"],
        "test_ordered_middle_f1": mt_ordered["middle_f1"],
        "test_final_acc": mt_final["acc"],
        "test_final_macro_f1": mt_final["f1"],
        "test_final_middle_precision": mt_final["middle_precision"],
        "test_final_middle_recall": mt_final["middle_recall"],
        "test_final_middle_f1": mt_final["middle_f1"],
        "test_final_ratio_penalty": mt_final["ratio_penalty"],
        "test_final_middle_to_early_rate": mt_final["middle_to_early_rate"],
        "test_final_middle_to_late_rate": mt_final["middle_to_late_rate"],
        "test_q_MAE": q_test["q_MAE"],
        "test_q_RMSE": q_test["q_RMSE"],
        "test_q_R2": q_test["q_R2"],
        "FINAL_condition_relative_stage_thresholds": str(DIR_FINAL / "FINAL_condition_relative_stage_thresholds.csv"),
        "FINAL_selected_stage_invariant_features": str(DIR_FINAL / "FINAL_selected_stage_invariant_features.csv"),
        "FINAL_gmm_fine_state_mapping": str(DIR_FINAL / "FINAL_gmm_fine_state_mapping.csv"),
        "FINAL_proxy_model_ranking": str(DIR_FINAL / "FINAL_proxy_model_ranking.csv"),
        "FINAL_probability_param_ranking": str(DIR_FINAL / "FINAL_probability_param_ranking.csv"),
        "FINAL_best_test_C6_predictions": str(DIR_FINAL / "FINAL_best_test_C6_predictions.csv"),
        "FINAL_classification_reports_long": str(DIR_FINAL / "FINAL_classification_reports_long.csv"),
        "FINAL_confusion_matrices_long": str(DIR_FINAL / "FINAL_confusion_matrices_long.csv"),
        "FINAL_stage_ratio_comparison": str(DIR_FINAL / "FINAL_stage_ratio_comparison.csv"),
        "CHECK_raw_sensor_cols": str(DIR_FINAL / "CHECK_raw_sensor_cols.csv"),
        "CHECK_online_candidate_features": str(DIR_FINAL / "FINAL_CHECK_online_candidate_features.csv"),
        "figures_dir": str(DIR_FIG),
    }])
    summary.to_csv(DIR_FINAL / "FINAL_experiment_summary.csv", index=False, encoding="utf-8-sig")

    # 8. 可视化
    make_visualizations(label_df, selected_df, pred_all, cm_df, ratio_df)

    # 9. 控制台输出
    print("\n" + "=" * 120)
    print("实验完成。重点查看以下文件：")
    print(f"1. {DIR_FINAL / 'FINAL_experiment_summary.csv'}")
    print(f"2. {DIR_FINAL / 'FINAL_classification_reports_long.csv'}")
    print(f"3. {DIR_FINAL / 'FINAL_confusion_matrices_long.csv'}")
    print(f"4. {DIR_FINAL / 'FINAL_stage_ratio_comparison.csv'}")
    print(f"5. {DIR_FINAL / 'FINAL_probability_param_ranking.csv'}")
    print(f"6. {DIR_FINAL / 'FINAL_best_test_C6_predictions.csv'}")
    print(f"7. {DIR_FINAL / 'CHECK_raw_sensor_cols.csv'}")
    print(f"8. {DIR_FINAL / 'FINAL_CHECK_online_candidate_features.csv'}")
    print(f"9. 图像文件夹：{DIR_FIG}")
    print("=" * 120)
    print("\n【C6 test final 指标】")
    print(f"Acc = {mt_final['acc']:.4f}")
    print(f"Macro-F1 = {mt_final['f1']:.4f}")
    print(f"Middle Precision = {mt_final['middle_precision']:.4f}")
    print(f"Middle Recall = {mt_final['middle_recall']:.4f}")
    print(f"Middle F1 = {mt_final['middle_f1']:.4f}")
    print(f"Middle -> Early Rate = {mt_final['middle_to_early_rate']:.4f}")
    print(f"Middle -> Late Rate = {mt_final['middle_to_late_rate']:.4f}")
    print(f"q_RMSE = {q_test['q_RMSE']:.4f}")
    print("=" * 120)


if __name__ == "__main__":
    main()

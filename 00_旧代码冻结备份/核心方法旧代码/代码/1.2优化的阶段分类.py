# -*- coding: utf-8 -*-
"""
42run：细粒度退化状态辅助的阶段概率识别实验

核心思路：
1. 工况内相对退化尺度 q_deg + 局部退化速率定义 early/middle/late；
2. 仅用训练工况 C1+C4 的训练样本学习 GMM 细粒度隐状态；
3. 五细状态按训练集 q_deg 均值排序，并辅助三阶段分类；
4. 构造工况相对特征与在线相对特征，降低跨工况幅值差异；
5. 使用 TCN-GRU 多任务模型：
   - stage head：三阶段分类；
   - fine-state head：五细状态辅助分类；
   - q head：连续退化位置预测；
6. 概率融合：
   - raw stage probability；
   - fine-state mapped stage probability；
   - q_hat prior；
   - causal ordered filtering；
7. C6 只用于最终评价，不参与特征选择、标准化、GMM、参数选择或模型选择。

重点输出：
1. FINAL_experiment_summary.csv
2. FINAL_classification_reports_long.csv
3. FINAL_confusion_matrices_long.csv
4. FINAL_best_test_C6_predictions.csv
5. FINAL_stage_ratio_comparison.csv
6. FINAL_selected_stage_invariant_features.csv
7. 03_figures 文件夹
"""

from pathlib import Path
import copy
import random
import warnings
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
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
# 0. 全局配置
# =========================================================
ROOT = Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM")
FEATURE_FILE = ROOT / "PHM实验" / "1run_run_level_features" / "02_features" / "run_level_features_all.csv"

RUN_NAME = "42run_fine_state_tcn_gru_stage_probability_no_leak"
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
USE_AMP = DEVICE == "cuda"

BATCH_SIZE = 32
EPOCHS = 120
PATIENCE = 18
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0

# 阶段定义参数
Q_EARLY = 0.30
Q_LATE = 0.72
RATE_LATE_Q = 0.78
RATE_SMOOTH_WIN = 7
VB_SMOOTH_WIN = 7
LATE_CONFIRM_WIN = 3

STAGE_NAMES = ["early", "middle", "late"]
STAGE_TO_ID = {"early": 0, "middle": 1, "late": 2}
ID_TO_STAGE = {0: "early", 1: "middle", 2: "late"}

N_FINE_STATES = 5

# 特征构造与筛选
VAR_THRESHOLD = 1e-8
TOP_FEATURES = 45
MAX_RAW_FEATURES_FOR_REL = 180
REDUNDANCY_THRESHOLD = 0.92

# 多任务损失权重
LAMBDA_STAGE = 1.0
LAMBDA_FINE = 0.35
LAMBDA_Q = 0.55
LAMBDA_Q_MONO = 0.05

# 阶段损失中 middle 加权
MIDDLE_CLASS_BOOST = 1.55

# 候选模型结构
ARCH_SPACE = [
    {"window_length": 10, "dropout": 0.10, "lr": 1e-3, "channels": (16, 32, 32), "gru_hidden": 48},
    {"window_length": 12, "dropout": 0.10, "lr": 1e-3, "channels": (16, 32, 32), "gru_hidden": 48},
    {"window_length": 15, "dropout": 0.15, "lr": 1e-3, "channels": (16, 32, 32), "gru_hidden": 48},
    {"window_length": 10, "dropout": 0.15, "lr": 1e-3, "channels": (32, 64, 64), "gru_hidden": 64},
    {"window_length": 12, "dropout": 0.20, "lr": 1e-3, "channels": (32, 64, 64), "gru_hidden": 64},
    {"window_length": 15, "dropout": 0.20, "lr": 1e-3, "channels": (32, 64, 64), "gru_hidden": 64},
]

# 概率融合参数搜索，仅基于 final_internal_val，不看 C6
PROB_PARAM_SPACE = list(itertools.product(
    [0.55, 0.65, 0.75],      # eta: raw stage prob weight
    [0.10, 0.20, 0.30],      # fine_prob weight
    [1.0, 1.2],              # temperature
    [0.04, 0.08, 0.12],      # middle floor
    [0.60, 0.66, 0.72],      # late_tau
    [0.00, 0.25, 0.50],      # order_blend
))

DPI = 600
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.sans-serif"] = ["Times New Roman", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"


# =========================================================
# 1. 基础函数
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    candidates = ["VB", "VB_max", "vb", "vb_max"]
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    raise ValueError("找不到 VB 标签列，请确认存在 VB / VB_max / vb_max。")


def is_forbidden_feature(col):
    c = str(col).lower()

    forbidden_exact = {
        "condition", "run_id", "cut", "file_name", "signal_len", "n_channels",
        "vb", "vb_max", "vbmean", "vb_mean",
        "flute_1", "flute_2", "flute_3",
        "dominant_flute", "final_dominant_flute",
        "stage", "stage_id", "phase", "phase_id",
        "q_deg", "q_true", "q_hat", "q_norm",
        "vb_smooth", "rate", "rate_smooth", "rate_norm",
        "wear_rate", "dvb", "delta_vb", "delta",
        "progress", "run_progress",
        "fine_state", "fine_state_true",
        "label", "target",
    }

    if c in forbidden_exact:
        return True

    forbidden_patterns = [
        "vb",
        "flute",
        "stage",
        "phase",
        "q_deg",
        "q_true",
        "q_hat",
        "q_norm",
        "wear",
        "label",
        "target",
        "progress",
        "rate_smooth",
        "rate_norm",
        "dvb",
        "delta_vb",
        "fine_state",
    ]

    return any(p in c for p in forbidden_patterns)


def get_raw_numeric_sensor_cols(df):
    cols = []
    for col in df.columns:
        if is_forbidden_feature(col):
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().mean() > 0.95:
            cols.append(col)
    return cols


def softmax_temperature(prob, temperature=1.0):
    prob = np.clip(prob, 1e-12, 1.0)
    logits = np.log(prob)
    logits = logits / max(temperature, 1e-6)
    logits = logits - logits.max(axis=1, keepdims=True)
    expv = np.exp(logits)
    return expv / (expv.sum(axis=1, keepdims=True) + 1e-12)


def calc_q_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "q_MAE": mean_absolute_error(y_true, y_pred),
        "q_RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "q_R2": r2_score(y_true, y_pred),
        "q_mono_violation": float(np.mean(np.diff(y_pred) < -1e-4)) if len(y_pred) > 1 else 0.0,
    }


def stage_ratio_penalty(y_true, y_pred):
    penalty = 0.0
    for sid in [0, 1, 2]:
        true_ratio = np.mean(y_true == sid)
        pred_ratio = np.mean(y_pred == sid)
        penalty += abs(true_ratio - pred_ratio)
    return float(penalty)


def middle_to_late_rate(y_true, y_pred):
    mask = y_true == 1
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(y_pred[mask] == 2))


def classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
    )
    p_each, r_each, f1_each, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )
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
        "ratio_penalty": stage_ratio_penalty(y_true, y_pred),
        "middle_to_late_rate": middle_to_late_rate(y_true, y_pred),
    }


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()


# =========================================================
# 2. 数据加载与阶段标签
# =========================================================
def load_feature_table():
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


def define_condition_relative_stage(df):
    """
    评价标签允许使用各工况自身 VB，因为这是 ground truth 构造；
    但模型训练/特征选择/概率参数选择不能使用 C6。
    """
    parts = []
    th_rows = []

    for cond, sub in df.groupby("condition"):
        sub = sub.sort_values("run_id").reset_index(drop=True).copy()

        vb_smooth = sub["VB"].rolling(VB_SMOOTH_WIN, center=True, min_periods=1).mean()
        q = (vb_smooth - vb_smooth.min()) / (vb_smooth.max() - vb_smooth.min() + 1e-12)

        rate = q.diff().fillna(0.0)
        rate = rate.rolling(RATE_SMOOTH_WIN, center=True, min_periods=1).mean()
        rate_pos = rate.clip(lower=0)

        q_early_th = q.quantile(Q_EARLY)
        q_late_th = q.quantile(Q_LATE)
        rate_late_th = rate_pos.quantile(RATE_LATE_Q)

        early = q <= q_early_th

        late_raw = (q >= q_late_th) | ((q >= 0.55) & (rate_pos >= rate_late_th))

        late_confirm = late_raw.copy()
        late_arr = late_raw.astype(int).values
        for i in range(len(late_arr)):
            if late_arr[i] == 1:
                left = max(0, i - LATE_CONFIRM_WIN + 1)
                if late_arr[left:i + 1].sum() < min(LATE_CONFIRM_WIN, i + 1):
                    late_confirm.iloc[i] = False

        stage = np.array(["middle"] * len(sub), dtype=object)
        stage[early.values] = "early"
        stage[late_confirm.values] = "late"

        # 如果 early 和 late 有重叠，early 优先保护低退化区
        stage[early.values] = "early"

        sub["VB_smooth"] = vb_smooth
        sub["q_deg"] = q
        sub["rate_smooth"] = rate_pos
        sub["stage"] = stage
        sub["stage_id"] = sub["stage"].map(STAGE_TO_ID).astype(int)

        fine_state = pd.qcut(
            sub["q_deg"].rank(method="first"),
            q=N_FINE_STATES,
            labels=False
        ).astype(int)
        sub["fine_state_true"] = fine_state

        th_rows.append({
            "condition": cond,
            "q_early_quantile": Q_EARLY,
            "q_late_quantile": Q_LATE,
            "rate_late_quantile": RATE_LATE_Q,
            "q_early_th": float(q_early_th),
            "q_late_th": float(q_late_th),
            "rate_late_th": float(rate_late_th),
            "VB_min": float(sub["VB"].min()),
            "VB_max": float(sub["VB"].max()),
            "VB_smooth_min": float(vb_smooth.min()),
            "VB_smooth_max": float(vb_smooth.max()),
            "stage_count": str(sub["stage"].value_counts().reindex(STAGE_NAMES, fill_value=0).to_dict()),
            "fine_state_count": str(sub["fine_state_true"].value_counts().sort_index().to_dict())
        })

        parts.append(sub)

    out = pd.concat(parts, axis=0).sort_values(["condition", "run_id"]).reset_index(drop=True)
    th_df = pd.DataFrame(th_rows)
    th_df.to_csv(DIR_FINAL / "FINAL_condition_relative_stage_thresholds.csv", index=False, encoding="utf-8-sig")

    return out, th_df


def split_final_train_val_test(df):
    """
    final_train: C1+C4 中每阶段抽出中间连续块作为 internal val，其余训练；
    test: C6 全生命周期。
    """
    train_parts = []
    val_parts = []

    for cond in ["C1", "C4"]:
        sub = df[df["condition"] == cond].sort_values("run_id").reset_index(drop=True).copy()
        val_idx = []

        for sid in [0, 1, 2]:
            g = sub[sub["stage_id"] == sid].sort_values("run_id")
            n = len(g)
            if n == 0:
                continue
            block = max(6, int(round(n * 0.20)))
            block = min(block, max(1, n - 2))
            start = max(0, (n - block) // 2)
            selected = g.iloc[start:start + block].index.tolist()
            val_idx.extend(selected)

        val_idx = sorted(set(val_idx))
        val_sub = sub.loc[val_idx].copy()
        train_sub = sub.drop(index=val_idx).copy()

        train_parts.append(train_sub)
        val_parts.append(val_sub)

    final_train = pd.concat(train_parts, axis=0).sort_values(["condition", "run_id"]).reset_index(drop=True)
    final_val = pd.concat(val_parts, axis=0).sort_values(["condition", "run_id"]).reset_index(drop=True)
    test_c6 = df[df["condition"] == "C6"].sort_values("run_id").reset_index(drop=True).copy()

    split_rows = []
    for split, sub in [("final_train", final_train), ("final_internal_val", final_val), ("test_C6", test_c6)]:
        for cond, g in sub.groupby("condition"):
            split_rows.append({
                "split": split,
                "condition": cond,
                "n_samples": len(g),
                "run_start": int(g["run_id"].min()),
                "run_end": int(g["run_id"].max()),
                "VB_min": float(g["VB"].min()),
                "VB_max": float(g["VB"].max()),
                "stage_count": str(g["stage"].value_counts().reindex(STAGE_NAMES, fill_value=0).to_dict())
            })

    split_meta = pd.DataFrame(split_rows)
    split_meta.to_csv(DIR_FINAL / "FINAL_split_metadata.csv", index=False, encoding="utf-8-sig")
    return final_train, final_val, test_c6, split_meta


# =========================================================
# 3. 特征构造与筛选
# =========================================================
def build_online_relative_features(df, raw_cols):
    """
    只基于传感器特征本身构造在线相对特征，不用 VB/q/stage。
    对每个 condition 单独做 expanding median/std，保证只用当前及历史特征值。
    """
    out = df[["condition", "run_id", "VB", "q_deg", "rate_smooth", "stage", "stage_id", "fine_state_true"]].copy()

    raw_cols = raw_cols[:MAX_RAW_FEATURES_FOR_REL]

    for col in raw_cols:
        x_all = pd.to_numeric(df[col], errors="coerce")
        out[col] = x_all.values

    rel_parts = []

    for cond, sub in out.groupby("condition"):
        sub = sub.sort_values("run_id").reset_index(drop=True).copy()
        rel = sub[["condition", "run_id", "VB", "q_deg", "rate_smooth", "stage", "stage_id", "fine_state_true"]].copy()

        for col in raw_cols:
            x = pd.to_numeric(sub[col], errors="coerce").astype(float)
            med_global = x.median()
            x = x.fillna(med_global)

            exp_mean = x.expanding(min_periods=3).mean()
            exp_std = x.expanding(min_periods=3).std().replace(0, np.nan).fillna(x.std() + 1e-8)

            rel[f"{col}__rel"] = (x - exp_mean) / (exp_std + 1e-8)

            roll_mean = x.rolling(5, min_periods=2).mean()
            rel[f"{col}__slope"] = (x - roll_mean).fillna(0.0)

            rel[f"{col}__online_rank"] = x.expanding(min_periods=3).apply(
                lambda arr: pd.Series(arr).rank(pct=True).iloc[-1],
                raw=False
            ).fillna(0.5)

        rel_parts.append(rel)

    feat_df = pd.concat(rel_parts, axis=0).sort_values(["condition", "run_id"]).reset_index(drop=True)

    return feat_df


def fit_feature_selection(train_df, candidate_cols):
    """
    train-only 特征筛选：
    score = MI(stage) + 0.5 MI(q) + 0.35 |Spearman(q)| - 0.25 domain_instability
    """
    clean = train_df.copy()

    usable = []
    medians = {}

    for col in candidate_cols:
        s = pd.to_numeric(clean[col], errors="coerce")
        med = s.median()
        if pd.isna(med):
            continue
        clean[col] = s.fillna(med)
        medians[col] = med
        usable.append(col)

    vt = VarianceThreshold(VAR_THRESHOLD)
    X = clean[usable].values
    vt.fit(X)

    cols_vt = [usable[i] for i, keep in enumerate(vt.get_support()) if keep]

    X_vt = clean[cols_vt].values
    y_stage = clean["stage_id"].values
    y_q = clean["q_deg"].values

    mi_stage = mutual_info_classif(X_vt, y_stage, random_state=RANDOM_SEED)
    mi_q = mutual_info_regression(X_vt, y_q, random_state=RANDOM_SEED)

    rows = []
    for i, col in enumerate(cols_vt):
        rho, _ = spearmanr(clean[col].values, y_q)
        rho_abs = abs(rho) if pd.notna(rho) else 0.0

        c1 = clean[clean["condition"] == "C1"][col]
        c4 = clean[clean["condition"] == "C4"][col]
        domain_instability = abs(c1.mean() - c4.mean()) / (c1.std() + c4.std() + 1e-8)

        score = (
                1.00 * mi_stage[i]
                + 0.50 * mi_q[i]
                + 0.35 * rho_abs
                - 0.25 * domain_instability
        )

        rows.append({
            "feature": col,
            "mi_stage": mi_stage[i],
            "mi_q": mi_q[i],
            "spearman_abs_q": rho_abs,
            "domain_instability_C1_C4": domain_instability,
            "feature_score": score,
        })

    score_df = pd.DataFrame(rows).sort_values("feature_score", ascending=False).reset_index(drop=True)

    selected = []
    selected_rows = []

    for _, row in score_df.iterrows():
        col = row["feature"]
        if len(selected) == 0:
            selected.append(col)
            r = row.to_dict()
            r["selected_rank"] = len(selected)
            r["max_abs_corr_with_selected"] = np.nan
            selected_rows.append(r)
        else:
            corr_list = []
            for s_col in selected:
                rho, _ = spearmanr(clean[col], clean[s_col])
                corr_list.append(abs(rho) if pd.notna(rho) else 0.0)
            max_corr = max(corr_list)
            if max_corr < REDUNDANCY_THRESHOLD:
                selected.append(col)
                r = row.to_dict()
                r["selected_rank"] = len(selected)
                r["max_abs_corr_with_selected"] = max_corr
                selected_rows.append(r)

        if len(selected) >= TOP_FEATURES:
            break

    selected_df = pd.DataFrame(selected_rows)
    selected_df = selected_df[
        ["selected_rank", "feature", "mi_stage", "mi_q", "spearman_abs_q",
         "domain_instability_C1_C4", "feature_score", "max_abs_corr_with_selected"]
    ]

    selected_df.to_csv(DIR_FINAL / "FINAL_selected_stage_invariant_features.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([{
        "raw_candidate_features": len(candidate_cols),
        "after_variance_filter": len(cols_vt),
        "selected_stage_invariant_features": len(selected)
    }])
    summary.to_csv(DIR_FINAL / "FINAL_feature_selection_summary.csv", index=False, encoding="utf-8-sig")

    return selected, selected_df, medians


def apply_selected_features(df, selected_cols, medians):
    out = df.copy()
    for col in selected_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(medians.get(col, 0.0))
    return out


def fit_final_scaler(train_df, selected_cols):
    scaler = StandardScaler()
    scaler.fit(train_df[selected_cols].values)
    return scaler


def apply_scaler(df, selected_cols, scaler):
    out = df.copy()
    out[selected_cols] = scaler.transform(out[selected_cols].values)
    return out


# =========================================================
# 4. GMM 细状态
# =========================================================
def fit_gmm_fine_states(train_df, selected_cols):
    scaler_gmm = StandardScaler()
    X_train = scaler_gmm.fit_transform(train_df[selected_cols].values)

    gmm = GaussianMixture(
        n_components=N_FINE_STATES,
        covariance_type="diag",
        random_state=RANDOM_SEED,
        n_init=10,
        max_iter=500
    )
    raw_comp = gmm.fit_predict(X_train)

    tmp = train_df.copy()
    tmp["raw_component"] = raw_comp

    comp_q = tmp.groupby("raw_component")["q_deg"].mean().sort_values()
    comp_to_order = {comp: i for i, comp in enumerate(comp_q.index.tolist())}
    order_to_comp = {v: k for k, v in comp_to_order.items()}

    fine_ordered = np.array([comp_to_order[c] for c in raw_comp], dtype=int)

    # 默认 5 细状态映射：0 early, 1-3 middle, 4 late
    fine_to_stage = {
        0: 0,
        1: 1,
        2: 1,
        3: 1,
        4: 2,
    }

    mapping_rows = []
    for raw_c, ordered_s in comp_to_order.items():
        mean_q = float(tmp[tmp["raw_component"] == raw_c]["q_deg"].mean())
        mapping_rows.append({
            "raw_component": raw_c,
            "ordered_fine_state": ordered_s,
            "mean_q_train": mean_q,
            "mapped_stage_id": fine_to_stage[ordered_s],
            "mapped_stage": ID_TO_STAGE[fine_to_stage[ordered_s]],
        })

    pd.DataFrame(mapping_rows).sort_values("ordered_fine_state").to_csv(
        DIR_FINAL / "FINAL_gmm_fine_state_mapping.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return gmm, scaler_gmm, comp_to_order, fine_to_stage


def apply_gmm_fine_states(df, selected_cols, gmm, scaler_gmm, comp_to_order):
    X = scaler_gmm.transform(df[selected_cols].values)
    raw_comp = gmm.predict(X)
    proba_raw = gmm.predict_proba(X)

    proba_ordered = np.zeros_like(proba_raw)
    for raw_c, order_s in comp_to_order.items():
        proba_ordered[:, order_s] = proba_raw[:, raw_c]

    fine_state = np.array([comp_to_order[c] for c in raw_comp], dtype=int)

    out = df.copy()
    out["fine_state_gmm"] = fine_state

    return out, proba_ordered


# =========================================================
# 5. 窗口数据
# =========================================================
def build_windows(base_df, target_df, feature_cols, window_length, split_name):
    base = base_df.sort_values(["condition", "run_id"]).reset_index(drop=True).copy()
    target_keys = set(zip(target_df["condition"], target_df["run_id"]))

    X_list = []
    y_stage = []
    y_fine = []
    y_q = []
    meta_rows = []

    for cond, sub in base.groupby("condition"):
        sub = sub.sort_values("run_id").reset_index(drop=True)
        run_ids = sub["run_id"].values

        X = sub[feature_cols].values.astype(np.float32)
        s = sub["stage_id"].values.astype(np.int64)
        fine = sub["fine_state_gmm"].values.astype(np.int64)
        q = sub["q_deg"].values.astype(np.float32)
        vb = sub["VB"].values.astype(float)
        stage_name = sub["stage"].values

        for end_idx in range(window_length - 1, len(sub)):
            start_idx = end_idx - window_length + 1
            runs = run_ids[start_idx:end_idx + 1]

            if not np.all(np.diff(runs) == 1):
                continue

            key = (cond, int(run_ids[end_idx]))
            if key not in target_keys:
                continue

            X_list.append(X[start_idx:end_idx + 1])
            y_stage.append(s[end_idx])
            y_fine.append(fine[end_idx])
            y_q.append(q[end_idx])

            meta_rows.append({
                "condition": cond,
                "run_id_start": int(runs[0]),
                "run_id_end": int(runs[-1]),
                "cut_index": int(runs[-1]),
                "VB_true": float(vb[end_idx]),
                "q_true": float(q[end_idx]),
                "stage_true_id": int(s[end_idx]),
                "stage_true": str(stage_name[end_idx]),
                "fine_state_true": int(fine[end_idx]),
                "window_length": window_length,
                "split": split_name,
            })

    if len(X_list) == 0:
        return None

    return (
        np.asarray(X_list, dtype=np.float32),
        np.asarray(y_stage, dtype=np.int64),
        np.asarray(y_fine, dtype=np.int64),
        np.asarray(y_q, dtype=np.float32),
        pd.DataFrame(meta_rows)
    )


class StageWindowDataset(Dataset):
    def __init__(self, X, y_stage, y_fine, y_q):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_stage = torch.tensor(y_stage, dtype=torch.long)
        self.y_fine = torch.tensor(y_fine, dtype=torch.long)
        self.y_q = torch.tensor(y_q, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_stage[idx], self.y_fine[idx], self.y_q[idx]


# =========================================================
# 6. 模型：TCN-GRU 多任务
# =========================================================
class Chomp1d(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        if self.size == 0:
            return x
        return x[:, :, :-self.size]


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.net(x)
        res = x if self.down is None else self.down(x)
        if y.size(-1) != res.size(-1):
            m = min(y.size(-1), res.size(-1))
            y = y[:, :, :m]
            res = res[:, :, :m]
        return self.relu(y + res)


class TCNGRUMultiTask(nn.Module):
    def __init__(self, input_dim, channels=(16, 32, 32), gru_hidden=48, dropout=0.1):
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
            bidirectional=False
        )

        self.shared = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.stage_head = nn.Linear(64, 3)
        self.fine_head = nn.Linear(64, N_FINE_STATES)
        self.q_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: B, L, F
        z = x.transpose(1, 2)       # B, F, L
        z = self.tcn(z)             # B, C, L
        z = z.transpose(1, 2)       # B, L, C
        out, _ = self.gru(z)
        h = out[:, -1, :]
        h = self.shared(h)

        stage_logits = self.stage_head(h)
        fine_logits = self.fine_head(h)
        q_hat = self.q_head(h)

        return {
            "stage_logits": stage_logits,
            "fine_logits": fine_logits,
            "q_hat": q_hat,
            "stage_prob": F.softmax(stage_logits, dim=1),
            "fine_prob": F.softmax(fine_logits, dim=1),
        }


# =========================================================
# 7. 训练与推理
# =========================================================
def make_stage_class_weights(y_stage):
    counts = np.bincount(y_stage, minlength=3).astype(float)
    weights = counts.sum() / (counts + 1e-8)
    weights = weights / weights.mean()
    weights[1] *= MIDDLE_CLASS_BOOST
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def q_monotonic_loss(q_hat):
    if q_hat.shape[0] <= 1:
        return torch.tensor(0.0, device=q_hat.device)
    diff = q_hat[1:] - q_hat[:-1]
    return torch.relu(-diff).mean()


def run_epoch(model, loader, optimizer=None, stage_weights=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_stage_logits = []
    all_fine_logits = []
    all_q_hat = []
    all_stage_true = []
    all_fine_true = []
    all_q_true = []

    scaler_amp = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    for X, y_stage, y_fine, y_q in loader:
        X = X.to(DEVICE)
        y_stage = y_stage.to(DEVICE)
        y_fine = y_fine.to(DEVICE)
        y_q = y_q.to(DEVICE)

        with torch.set_grad_enabled(is_train):
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    out = model(X)
                    loss_stage = F.cross_entropy(out["stage_logits"], y_stage, weight=stage_weights)
                    loss_fine = F.cross_entropy(out["fine_logits"], y_fine)
                    loss_q = F.smooth_l1_loss(out["q_hat"], y_q)
                    loss_mono = q_monotonic_loss(out["q_hat"])
                    loss = (
                            LAMBDA_STAGE * loss_stage
                            + LAMBDA_FINE * loss_fine
                            + LAMBDA_Q * loss_q
                            + LAMBDA_Q_MONO * loss_mono
                    )
            else:
                out = model(X)
                loss_stage = F.cross_entropy(out["stage_logits"], y_stage, weight=stage_weights)
                loss_fine = F.cross_entropy(out["fine_logits"], y_fine)
                loss_q = F.smooth_l1_loss(out["q_hat"], y_q)
                loss_mono = q_monotonic_loss(out["q_hat"])
                loss = (
                        LAMBDA_STAGE * loss_stage
                        + LAMBDA_FINE * loss_fine
                        + LAMBDA_Q * loss_q
                        + LAMBDA_Q_MONO * loss_mono
                )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if USE_AMP:
                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()

        total_loss += loss.item() * X.size(0)

        all_stage_logits.append(out["stage_logits"].detach().cpu().numpy())
        all_fine_logits.append(out["fine_logits"].detach().cpu().numpy())
        all_q_hat.append(out["q_hat"].detach().cpu().numpy().reshape(-1))
        all_stage_true.append(y_stage.detach().cpu().numpy())
        all_fine_true.append(y_fine.detach().cpu().numpy())
        all_q_true.append(y_q.detach().cpu().numpy().reshape(-1))

    return {
        "loss": total_loss / len(loader.dataset),
        "stage_logits": np.concatenate(all_stage_logits, axis=0),
        "fine_logits": np.concatenate(all_fine_logits, axis=0),
        "q_hat": np.concatenate(all_q_hat, axis=0),
        "stage_true": np.concatenate(all_stage_true, axis=0),
        "fine_true": np.concatenate(all_fine_true, axis=0),
        "q_true": np.concatenate(all_q_true, axis=0),
    }


def train_model(train_pack, val_pack, feature_dim, arch, model_name):
    X_train, y_stage_train, y_fine_train, y_q_train, meta_train = train_pack
    X_val, y_stage_val, y_fine_val, y_q_val, meta_val = val_pack

    train_loader = DataLoader(
        StageWindowDataset(X_train, y_stage_train, y_fine_train, y_q_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        StageWindowDataset(X_val, y_stage_val, y_fine_val, y_q_val),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = TCNGRUMultiTask(
        input_dim=feature_dim,
        channels=arch["channels"],
        gru_hidden=arch["gru_hidden"],
        dropout=arch["dropout"]
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=arch["lr"], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)

    stage_weights = make_stage_class_weights(y_stage_train)

    best_state = None
    best_score = np.inf
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_out = run_epoch(model, train_loader, optimizer=optimizer, stage_weights=stage_weights)
        val_out = run_epoch(model, val_loader, optimizer=None, stage_weights=stage_weights)

        val_prob = F.softmax(torch.tensor(val_out["stage_logits"]), dim=1).numpy()
        val_pred = np.argmax(val_prob, axis=1)

        clf = classification_metrics(val_out["stage_true"], val_pred)
        q_m = calc_q_metrics(val_out["q_true"], val_out["q_hat"])

        score = (
                0.30 * (1 - clf["f1"])
                + 0.32 * (1 - clf["middle_recall"])
                + 0.18 * clf["middle_to_late_rate"]
                + 0.10 * clf["ratio_penalty"]
                + 0.10 * q_m["q_RMSE"]
        )

        scheduler.step(score)

        history.append({
            "epoch": epoch,
            "train_loss": train_out["loss"],
            "val_loss": val_out["loss"],
            "val_score": score,
            "val_acc": clf["acc"],
            "val_f1": clf["f1"],
            "val_middle_recall": clf["middle_recall"],
            "val_middle_to_late_rate": clf["middle_to_late_rate"],
            "val_q_RMSE": q_m["q_RMSE"],
            "val_q_R2": q_m["q_R2"],
        })

        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    pd.DataFrame(history).to_csv(DIR_INTERIM / f"{model_name}_history.csv", index=False, encoding="utf-8-sig")
    torch.save(model.state_dict(), DIR_MODEL / f"{model_name}_best.pth")

    return model, best_score, best_epoch


def predict_model(model, pack):
    X, y_stage, y_fine, y_q, meta = pack
    loader = DataLoader(StageWindowDataset(X, y_stage, y_fine, y_q), batch_size=BATCH_SIZE, shuffle=False)
    out = run_epoch(model, loader, optimizer=None, stage_weights=None)

    stage_prob = F.softmax(torch.tensor(out["stage_logits"]), dim=1).numpy()
    fine_prob = F.softmax(torch.tensor(out["fine_logits"]), dim=1).numpy()

    pred = meta.copy().reset_index(drop=True)
    pred["q_hat"] = out["q_hat"]
    pred["q_true_model"] = out["q_true"]

    for i, name in enumerate(STAGE_NAMES):
        pred[f"raw_prob_{name}"] = stage_prob[:, i]

    for i in range(N_FINE_STATES):
        pred[f"fine_prob_{i}"] = fine_prob[:, i]

    pred["stage_pred_raw"] = np.argmax(stage_prob, axis=1)
    pred["stage_pred_raw_name"] = pred["stage_pred_raw"].map(ID_TO_STAGE)

    return pred, stage_prob, fine_prob


# =========================================================
# 8. 概率融合与有序过滤
# =========================================================
def fine_prob_to_stage_prob(fine_prob, fine_to_stage):
    out = np.zeros((fine_prob.shape[0], 3), dtype=float)
    for f in range(fine_prob.shape[1]):
        sid = fine_to_stage.get(f, 1)
        out[:, sid] += fine_prob[:, f]
    out = out / (out.sum(axis=1, keepdims=True) + 1e-12)
    return out


def q_prior_from_qhat(q_hat, late_tau=0.66):
    q = np.clip(np.asarray(q_hat).reshape(-1), 0.0, 1.0)
    p = np.zeros((len(q), 3), dtype=float)

    # 三角/钟形先验。middle 在中间形成宽平台。
    p[:, 0] = np.clip((0.45 - q) / 0.45, 0, 1)
    p[:, 2] = np.clip((q - late_tau) / max(1e-6, 1.0 - late_tau), 0, 1)
    p[:, 1] = np.exp(-((q - 0.52) ** 2) / (2 * 0.18 ** 2))

    p = np.maximum(p, 1e-8)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def estimate_stage_transition(train_meta):
    counts = np.ones((3, 3), dtype=float) * 0.05

    for cond, sub in train_meta.groupby("condition"):
        s = sub.sort_values("cut_index")["stage_true_id"].values.astype(int)
        for a, b in zip(s[:-1], s[1:]):
            if b < a:
                counts[a, a] += 1.0
            else:
                counts[a, b] += 1.0

    # 左到右约束，允许停留、前进一格、小概率跳到后期
    trans = np.zeros((3, 3), dtype=float)
    trans[0] = [0.94, 0.055, 0.005]
    trans[1] = [0.005, 0.955, 0.040]
    trans[2] = [0.0005, 0.010, 0.9895]

    pd.DataFrame(trans, index=STAGE_NAMES, columns=STAGE_NAMES).to_csv(
        DIR_FINAL / "FINAL_stage_transition_matrix.csv",
        encoding="utf-8-sig"
    )

    return trans


def causal_ordered_filter(prob, trans):
    prob = np.clip(prob, 1e-12, 1.0)
    prob = prob / prob.sum(axis=1, keepdims=True)

    out = np.zeros_like(prob)
    alpha = prob[0] / prob[0].sum()
    out[0] = alpha

    for t in range(1, len(prob)):
        pred = alpha @ trans
        alpha = pred * prob[t]
        alpha = alpha / (alpha.sum() + 1e-12)
        out[t] = alpha

    return out


def apply_probability_inference(pred_df, stage_prob, fine_prob, fine_to_stage, trans, params):
    eta, fine_weight, temperature, mid_floor, late_tau, order_blend = params

    raw = softmax_temperature(stage_prob, temperature=temperature)
    fine_stage = fine_prob_to_stage_prob(fine_prob, fine_to_stage)
    prior = q_prior_from_qhat(pred_df["q_hat"].values, late_tau=late_tau)

    # late suppression：q_hat 未达到 late 区间时，late 概率不宜过早占主导
    q = np.clip(pred_df["q_hat"].values, 0, 1)
    suppress = 1 / (1 + np.exp(-16 * (q - late_tau)))
    prior[:, 2] *= suppress
    prior[:, 1] = np.maximum(prior[:, 1], mid_floor)
    prior = prior / (prior.sum(axis=1, keepdims=True) + 1e-12)

    rest = max(0.0, 1.0 - eta - fine_weight)
    mix = eta * raw + fine_weight * fine_stage + rest * prior
    mix[:, 1] = np.maximum(mix[:, 1], mid_floor)
    mix = mix / (mix.sum(axis=1, keepdims=True) + 1e-12)

    ordered = np.zeros_like(mix)
    for cond, idx in pred_df.groupby("condition").groups.items():
        idx = list(idx)
        sub_prob = mix[idx]
        ordered[idx] = causal_ordered_filter(sub_prob, trans)

    final = (1 - order_blend) * mix + order_blend * ordered
    final = final / (final.sum(axis=1, keepdims=True) + 1e-12)

    out = pred_df.copy()

    for i, name in enumerate(STAGE_NAMES):
        out[f"prior_prob_{name}"] = prior[:, i]
        out[f"mix_prob_{name}"] = mix[:, i]
        out[f"ordered_prob_{name}"] = ordered[:, i]
        out[f"final_prob_{name}"] = final[:, i]

    for method, prob in [
        ("prior", prior),
        ("mix", mix),
        ("ordered", ordered),
        ("final", final),
    ]:
        out[f"stage_pred_{method}"] = np.argmax(prob, axis=1)
        out[f"stage_pred_{method}_name"] = out[f"stage_pred_{method}"].map(ID_TO_STAGE)

    return out


def evaluate_prediction_df(pred_df, split_name):
    rows_report = []
    rows_cm = []
    rows_ratio = []

    y_true = pred_df["stage_true_id"].values.astype(int)

    methods = ["raw", "prior", "mix", "ordered", "final"]

    for method in methods:
        pred_col = f"stage_pred_{method}"
        if pred_col not in pred_df.columns:
            continue

        y_pred = pred_df[pred_col].values.astype(int)

        rep = classification_report(
            y_true,
            y_pred,
            target_names=STAGE_NAMES,
            output_dict=True,
            zero_division=0
        )

        for label, vals in rep.items():
            if isinstance(vals, dict):
                rows_report.append({
                    "split": split_name,
                    "method": method,
                    "label": label,
                    "precision": vals.get("precision", np.nan),
                    "recall": vals.get("recall", np.nan),
                    "f1-score": vals.get("f1-score", np.nan),
                    "support": vals.get("support", np.nan),
                    "value": np.nan,
                })
            else:
                rows_report.append({
                    "split": split_name,
                    "method": method,
                    "label": label,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1-score": np.nan,
                    "support": np.nan,
                    "value": vals,
                })

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)

        for i, true_name in enumerate(STAGE_NAMES):
            for j, pred_name in enumerate(STAGE_NAMES):
                rows_cm.append({
                    "split": split_name,
                    "method": method,
                    "true_stage": true_name,
                    "pred_stage": pred_name,
                    "count": int(cm[i, j]),
                    "row_norm": float(cm_norm[i, j]),
                })

        for sid, name in enumerate(STAGE_NAMES):
            rows_ratio.append({
                "split": split_name,
                "method": "TRUE",
                "stage": name,
                "ratio": float(np.mean(y_true == sid)),
            })
            rows_ratio.append({
                "split": split_name,
                "method": method,
                "stage": name,
                "ratio": float(np.mean(y_pred == sid)),
            })

    return pd.DataFrame(rows_report), pd.DataFrame(rows_cm), pd.DataFrame(rows_ratio)


def score_for_param_selection(pred_val):
    y_true = pred_val["stage_true_id"].values.astype(int)
    y_pred = pred_val["stage_pred_final"].values.astype(int)
    m = classification_metrics(y_true, y_pred)
    q_m = calc_q_metrics(pred_val["q_true"].values, pred_val["q_hat"].values)

    score = (
            0.25 * (1 - m["f1"])
            + 0.35 * (1 - m["middle_recall"])
            + 0.18 * m["middle_to_late_rate"]
            + 0.12 * m["ratio_penalty"]
            + 0.10 * q_m["q_RMSE"]
    )
    return score, m, q_m


# =========================================================
# 9. 代理验证流程
# =========================================================
def prepare_train_for_conditions(df_feat, train_conditions, target_conditions, selected_cols=None, prefit=None):
    train_df = df_feat[df_feat["condition"].isin(train_conditions)].copy()
    target_df = df_feat[df_feat["condition"].isin(target_conditions)].copy()

    if selected_cols is None:
        candidate_cols = [
            c for c in df_feat.columns
            if c not in ["condition", "run_id", "VB", "q_deg", "rate_smooth", "stage", "stage_id", "fine_state_true"]
               and not is_forbidden_feature(c)
        ]

        selected_cols, selected_df, medians = fit_feature_selection(train_df, candidate_cols)
    else:
        selected_df = None
        medians = prefit["medians"]

    train_df = apply_selected_features(train_df, selected_cols, medians)
    target_df = apply_selected_features(target_df, selected_cols, medians)

    scaler = StandardScaler()
    train_df[selected_cols] = scaler.fit_transform(train_df[selected_cols].values)
    target_df[selected_cols] = scaler.transform(target_df[selected_cols].values)

    gmm, scaler_gmm, comp_to_order, fine_to_stage = fit_gmm_fine_states(train_df, selected_cols)
    train_df, _ = apply_gmm_fine_states(train_df, selected_cols, gmm, scaler_gmm, comp_to_order)
    target_df, _ = apply_gmm_fine_states(target_df, selected_cols, gmm, scaler_gmm, comp_to_order)

    return train_df, target_df, selected_cols, {
        "selected_df": selected_df,
        "medians": medians,
        "scaler": scaler,
        "gmm": gmm,
        "scaler_gmm": scaler_gmm,
        "comp_to_order": comp_to_order,
        "fine_to_stage": fine_to_stage,
    }


def proxy_validate_architecture(df_feat, arch):
    scores = []
    rows = []

    for train_cond, test_cond in [(["C1"], ["C4"]), (["C4"], ["C1"])]:
        train_df, target_df, selected_cols, info = prepare_train_for_conditions(
            df_feat, train_cond, test_cond
        )

        L = arch["window_length"]
        train_pack = build_windows(train_df, train_df, selected_cols, L, "proxy_train")
        target_pack = build_windows(target_df, target_df, selected_cols, L, "proxy_target")

        if train_pack is None or target_pack is None:
            return None

        model_name = f"proxy_{train_cond[0]}_to_{test_cond[0]}_L{L}_drop{str(arch['dropout']).replace('.', 'p')}"
        model, _, _ = train_model(train_pack, target_pack, len(selected_cols), arch, model_name)

        pred_df, stage_prob, fine_prob = predict_model(model, target_pack)

        trans = estimate_stage_transition(train_pack[-1])
        params = (0.65, 0.20, 1.0, 0.06, 0.66, 0.25)
        pred_df = apply_probability_inference(pred_df, stage_prob, fine_prob, info["fine_to_stage"], trans, params)

        y_true = pred_df["stage_true_id"].values.astype(int)
        y_pred = pred_df["stage_pred_final"].values.astype(int)

        m = classification_metrics(y_true, y_pred)
        q_m = calc_q_metrics(pred_df["q_true"].values, pred_df["q_hat"].values)

        score = (
                0.25 * (1 - m["f1"])
                + 0.35 * (1 - m["middle_recall"])
                + 0.18 * m["middle_to_late_rate"]
                + 0.12 * m["ratio_penalty"]
                + 0.10 * q_m["q_RMSE"]
        )

        scores.append(score)

        rows.append({
            "proxy": f"{train_cond[0]}->{test_cond[0]}",
            "score": score,
            "acc": m["acc"],
            "f1": m["f1"],
            "middle_recall": m["middle_recall"],
            "middle_to_late_rate": m["middle_to_late_rate"],
            "ratio_penalty": m["ratio_penalty"],
            "q_RMSE": q_m["q_RMSE"],
            "q_R2": q_m["q_R2"],
        })

    proxy_mean = float(np.mean(scores))
    proxy_worst = float(np.max(scores))
    proxy_final = 0.45 * proxy_mean + 0.55 * proxy_worst

    return proxy_final, proxy_mean, proxy_worst, rows


# =========================================================
# 10. 可视化
# =========================================================
def plot_condition_stage_distribution(df):
    colors = {"early": "#4C78A8", "middle": "#59A14F", "late": "#E15759"}

    for cond, sub in df.groupby("condition"):
        sub = sub.sort_values("run_id")
        plt.figure(figsize=(12, 4.8))
        plt.plot(sub["run_id"], sub["VB"], color="black", linewidth=2.0, label="VB")

        for stage in STAGE_NAMES:
            g = sub[sub["stage"] == stage]
            plt.scatter(
                g["run_id"], g["VB"],
                s=24,
                color=colors[stage],
                label=stage,
                alpha=0.85
            )

        plt.xlabel("Cut index")
        plt.ylabel("VB")
        plt.title(f"Condition-relative stage definition on {cond}")
        plt.legend(loc="upper left")
        savefig(DIR_FIG / f"fig_stage_definition_{cond}.png")

    plt.figure(figsize=(13, 5))
    for cond, sub in df.groupby("condition"):
        sub = sub.sort_values("run_id")
        plt.plot(sub["run_id"], sub["q_deg"], linewidth=2.0, label=f"{cond} q_deg")
    plt.axhline(Q_EARLY, color="gray", linestyle="--", linewidth=1.2)
    plt.axhline(Q_LATE, color="gray", linestyle="--", linewidth=1.2)
    plt.xlabel("Cut index")
    plt.ylabel("Condition-relative degradation scale")
    plt.title("Condition-relative degradation scales")
    plt.legend()
    savefig(DIR_FIG / "fig_condition_relative_qdeg_all.png")


def plot_selected_features(selected_df):
    if selected_df is None or selected_df.empty:
        return
    dfp = selected_df.sort_values("feature_score", ascending=True)
    plt.figure(figsize=(9, max(6, 0.28 * len(dfp))))
    plt.barh(dfp["feature"], dfp["feature_score"])
    plt.xlabel("Feature score")
    plt.ylabel("Selected feature")
    plt.title("Selected stage-invariant features")
    savefig(DIR_FIG / "fig_selected_stage_features.png")


def plot_proxy_ranking(proxy_df):
    if proxy_df.empty:
        return

    dfp = proxy_df.sort_values("proxy_final_score", ascending=True).head(15)
    labels = dfp["arch_name"].values

    plt.figure(figsize=(12, 6))
    x = np.arange(len(dfp))
    plt.bar(x, dfp["proxy_final_score"].values)
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Proxy validation score")
    plt.title("Cross-condition proxy validation ranking")
    savefig(DIR_FIG / "fig_proxy_validation_ranking.png")


def plot_confusion_matrices(conf_df, split="test_C6"):
    methods = ["raw", "prior", "mix", "ordered", "final"]

    for method in methods:
        sub = conf_df[(conf_df["split"] == split) & (conf_df["method"] == method)]
        if len(sub) == 0:
            continue

        mat = np.zeros((3, 3))
        for _, r in sub.iterrows():
            i = STAGE_NAMES.index(r["true_stage"])
            j = STAGE_NAMES.index(r["pred_stage"])
            mat[i, j] = r["row_norm"]

        count_mat = np.zeros((3, 3))
        for _, r in sub.iterrows():
            i = STAGE_NAMES.index(r["true_stage"])
            j = STAGE_NAMES.index(r["pred_stage"])
            count_mat[i, j] = r["count"]

        plt.figure(figsize=(5.8, 5.2))
        im = plt.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, label="Row-normalized value")
        plt.xticks(range(3), STAGE_NAMES)
        plt.yticks(range(3), STAGE_NAMES)

        for i in range(3):
            for j in range(3):
                plt.text(
                    j, i,
                    f"{int(count_mat[i, j])}\n({mat[i, j]:.2f})",
                    ha="center",
                    va="center",
                    fontsize=10
                )

        plt.xlabel("Predicted stage")
        plt.ylabel("True stage")
        plt.title(f"{split} confusion matrix - {method}")
        savefig(DIR_FIG / f"fig_confusion_{split}_{method}.png")


def plot_probability_evolution(pred_df):
    sub = pred_df[pred_df["condition"] == "C6"].sort_values("cut_index")

    for method in ["raw", "prior", "mix", "ordered", "final"]:
        plt.figure(figsize=(13, 4.8))
        for stage in STAGE_NAMES:
            col = f"{method}_prob_{stage}"
            if col in sub.columns:
                plt.plot(sub["cut_index"], sub[col], linewidth=2, label=f"{stage}")

        plt.xlabel("Cut index")
        plt.ylabel("Stage probability")
        plt.title(f"C6 stage probability evolution - {method}")
        plt.legend(loc="upper left")
        savefig(DIR_FIG / f"fig_C6_probability_evolution_{method}.png")


def plot_qhat_curve(pred_df):
    sub = pred_df[pred_df["condition"] == "C6"].sort_values("cut_index")

    plt.figure(figsize=(13, 4.8))
    plt.plot(sub["cut_index"], sub["q_true"], color="black", linewidth=2.5, label="True q_deg")
    plt.plot(sub["cut_index"], sub["q_hat"], linewidth=2.2, label="Predicted q_hat")
    plt.xlabel("Cut index")
    plt.ylabel("Degradation position")
    plt.title("C6 q_hat prediction")
    plt.legend()
    savefig(DIR_FIG / "fig_C6_qhat_curve.png")


def plot_stage_prediction_timeline(pred_df):
    sub = pred_df[pred_df["condition"] == "C6"].sort_values("cut_index")
    y_true = sub["stage_true_id"].values
    y_final = sub["stage_pred_final"].values

    plt.figure(figsize=(13, 4.6))
    plt.step(sub["cut_index"], y_true, where="post", linewidth=2.8, label="True stage", color="black")
    plt.step(sub["cut_index"], y_final, where="post", linewidth=2.0, label="Predicted final stage")
    plt.yticks([0, 1, 2], STAGE_NAMES)
    plt.xlabel("Cut index")
    plt.ylabel("Stage")
    plt.title("C6 true and predicted stage timeline")
    plt.legend()
    savefig(DIR_FIG / "fig_C6_stage_timeline.png")


def plot_stage_ratio(ratio_df):
    sub = ratio_df[ratio_df["split"] == "test_C6"].copy()
    if sub.empty:
        return

    methods = ["TRUE", "raw", "prior", "mix", "ordered", "final"]
    x = np.arange(len(methods))
    width = 0.24

    plt.figure(figsize=(10, 5))
    for i, stage in enumerate(STAGE_NAMES):
        vals = []
        for m in methods:
            g = sub[(sub["method"] == m) & (sub["stage"] == stage)]
            vals.append(float(g["ratio"].iloc[0]) if len(g) > 0 else 0)
        plt.bar(x + (i - 1) * width, vals, width=width, label=stage)

    plt.xticks(x, methods)
    plt.ylabel("Stage ratio")
    plt.title("Stage ratio comparison on C6")
    plt.legend()
    savefig(DIR_FIG / "fig_C6_stage_ratio_comparison.png")


def plot_pca_feature_space(df_scaled, selected_cols, split_name):
    if len(selected_cols) < 2:
        return

    sub = df_scaled.copy()
    X = sub[selected_cols].values
    y = sub["stage_id"].values

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    z = pca.fit_transform(X)

    plt.figure(figsize=(7.2, 6))
    for sid, name in enumerate(STAGE_NAMES):
        idx = y == sid
        plt.scatter(z[idx, 0], z[idx, 1], s=18, alpha=0.75, label=name)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA of selected features - {split_name}")
    plt.legend()
    savefig(DIR_FIG / f"fig_pca_selected_features_{split_name}.png")


# =========================================================
# 11. 主流程
# =========================================================
def main():
    set_seed(RANDOM_SEED)

    print("=" * 120)
    print("42run：细粒度退化状态辅助的阶段概率识别实验")
    print("=" * 120)
    print(f"Device: {DEVICE}")
    print(f"Feature file: {FEATURE_FILE}")
    print(f"Run dir: {RUN_DIR}")

    # 1. 数据与标签
    df_raw = load_feature_table()
    df_label, th_df = define_condition_relative_stage(df_raw)

    plot_condition_stage_distribution(df_label)

    raw_sensor_cols = get_raw_numeric_sensor_cols(df_label)
    print(f"Raw numeric sensor features: {len(raw_sensor_cols)}")

    df_feat = build_online_relative_features(df_label, raw_sensor_cols)

    # 2. 代理验证选择结构：C1->C4, C4->C1
    proxy_rows_all = []
    arch_results = []

    for arch in ARCH_SPACE:
        arch_name = (
            f"L{arch['window_length']}"
            f"_drop{str(arch['dropout']).replace('.', 'p')}"
            f"_lr{str(arch['lr']).replace('.', 'p')}"
            f"_ch{'-'.join(map(str, arch['channels']))}"
        )

        print(f"\n[Proxy architecture] {arch_name}")

        res = proxy_validate_architecture(df_feat, arch)
        if res is None:
            continue

        proxy_final, proxy_mean, proxy_worst, proxy_rows = res

        row_main = {
            "arch_name": arch_name,
            "window_length": arch["window_length"],
            "dropout": arch["dropout"],
            "learning_rate": arch["lr"],
            "channels": str(arch["channels"]),
            "gru_hidden": arch["gru_hidden"],
            "proxy_mean_score": proxy_mean,
            "proxy_worst_score": proxy_worst,
            "proxy_final_score": proxy_final,
        }

        for r in proxy_rows:
            prefix = "proxy_" + r["proxy"].replace("->", "_to_")
            row_main[f"{prefix}_score"] = r["score"]
            row_main[f"{prefix}_acc"] = r["acc"]
            row_main[f"{prefix}_f1"] = r["f1"]
            row_main[f"{prefix}_middle_recall"] = r["middle_recall"]
            row_main[f"{prefix}_middle_to_late_rate"] = r["middle_to_late_rate"]

            rr = r.copy()
            rr["arch_name"] = arch_name
            proxy_rows_all.append(rr)

        arch_results.append(row_main)

        print(f"  proxy_mean={proxy_mean:.4f}, proxy_worst={proxy_worst:.4f}, proxy_final={proxy_final:.4f}")

    proxy_df = pd.DataFrame(arch_results).sort_values("proxy_final_score", ascending=True).reset_index(drop=True)
    proxy_df.to_csv(DIR_FINAL / "FINAL_proxy_model_ranking.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(proxy_rows_all).to_csv(
        DIR_INTERIM / "proxy_detail_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    if proxy_df.empty:
        raise RuntimeError("代理验证没有成功训练任何结构。")

    plot_proxy_ranking(proxy_df)

    best_arch_row = proxy_df.iloc[0]
    best_arch = None
    for arch in ARCH_SPACE:
        arch_name = (
            f"L{arch['window_length']}"
            f"_drop{str(arch['dropout']).replace('.', 'p')}"
            f"_lr{str(arch['lr']).replace('.', 'p')}"
            f"_ch{'-'.join(map(str, arch['channels']))}"
        )
        if arch_name == best_arch_row["arch_name"]:
            best_arch = arch
            break

    print(f"\nBest architecture: {best_arch_row['arch_name']}")

    # 3. final train/val/test 划分
    final_train_raw, final_val_raw, test_c6_raw, split_meta = split_final_train_val_test(df_label)

    # 用 df_feat 中对应行
    def pick_by_keys(source, target):
        keys = set(zip(target["condition"], target["run_id"]))
        mask = [(c, r) in keys for c, r in zip(source["condition"], source["run_id"])]
        return source.loc[mask].copy().sort_values(["condition", "run_id"]).reset_index(drop=True)

    final_train_feat = pick_by_keys(df_feat, final_train_raw)
    final_val_feat = pick_by_keys(df_feat, final_val_raw)
    test_c6_feat = pick_by_keys(df_feat, test_c6_raw)

    # 4. train-only 特征筛选、标准化、GMM
    candidate_cols = [
        c for c in df_feat.columns
        if c not in ["condition", "run_id", "VB", "q_deg", "rate_smooth", "stage", "stage_id", "fine_state_true"]
           and not is_forbidden_feature(c)
    ]

    selected_cols, selected_df, medians = fit_feature_selection(final_train_feat, candidate_cols)
    plot_selected_features(selected_df)

    final_train_feat = apply_selected_features(final_train_feat, selected_cols, medians)
    final_val_feat = apply_selected_features(final_val_feat, selected_cols, medians)
    test_c6_feat = apply_selected_features(test_c6_feat, selected_cols, medians)

    scaler = fit_final_scaler(final_train_feat, selected_cols)
    final_train_scaled = apply_scaler(final_train_feat, selected_cols, scaler)
    final_val_scaled = apply_scaler(final_val_feat, selected_cols, scaler)
    test_c6_scaled = apply_scaler(test_c6_feat, selected_cols, scaler)

    gmm, scaler_gmm, comp_to_order, fine_to_stage = fit_gmm_fine_states(final_train_scaled, selected_cols)

    final_train_scaled, _ = apply_gmm_fine_states(final_train_scaled, selected_cols, gmm, scaler_gmm, comp_to_order)
    final_val_scaled, _ = apply_gmm_fine_states(final_val_scaled, selected_cols, gmm, scaler_gmm, comp_to_order)
    test_c6_scaled, _ = apply_gmm_fine_states(test_c6_scaled, selected_cols, gmm, scaler_gmm, comp_to_order)

    plot_pca_feature_space(final_train_scaled, selected_cols, "final_train")
    plot_pca_feature_space(test_c6_scaled, selected_cols, "test_C6")

    # 5. 构造窗口
    L = best_arch["window_length"]

    # train 只用 train 自己窗口，避免内部 val 特征进入训练
    train_pack = build_windows(final_train_scaled, final_train_scaled, selected_cols, L, "final_train")

    # val 可以使用 C1+C4 的上下文特征，但目标只评价 val
    full_c1c4_scaled = pd.concat([final_train_scaled, final_val_scaled], axis=0).sort_values(
        ["condition", "run_id"]
    ).reset_index(drop=True)

    val_pack = build_windows(full_c1c4_scaled, final_val_scaled, selected_cols, L, "final_internal_val")

    # test 使用 C6 自身在线上下文
    test_pack = build_windows(test_c6_scaled, test_c6_scaled, selected_cols, L, "test_C6")

    if train_pack is None or val_pack is None or test_pack is None:
        raise RuntimeError("窗口构造失败，请检查窗口长度或数据连续性。")

    # 6. 训练 final 模型
    model_name = f"final_{best_arch_row['arch_name']}"
    model, best_score, best_epoch = train_model(
        train_pack,
        val_pack,
        len(selected_cols),
        best_arch,
        model_name
    )

    # 7. 预测 raw 概率
    pred_val_base, val_stage_prob, val_fine_prob = predict_model(model, val_pack)
    pred_test_base, test_stage_prob, test_fine_prob = predict_model(model, test_pack)

    trans = estimate_stage_transition(train_pack[-1])

    # 8. 概率参数选择：只用 final_internal_val
    param_rows = []

    for params in PROB_PARAM_SPACE:
        pred_val = apply_probability_inference(
            pred_val_base,
            val_stage_prob,
            val_fine_prob,
            fine_to_stage,
            trans,
            params
        )

        val_score, val_m, val_q = score_for_param_selection(pred_val)

        param_rows.append({
            "eta": params[0],
            "fine_weight": params[1],
            "temperature": params[2],
            "mid_floor": params[3],
            "late_tau": params[4],
            "order_blend": params[5],
            "val_score": val_score,
            "val_final_acc": val_m["acc"],
            "val_final_f1": val_m["f1"],
            "val_final_middle_precision": val_m["middle_precision"],
            "val_final_middle_recall": val_m["middle_recall"],
            "val_final_middle_f1": val_m["middle_f1"],
            "val_final_ratio_penalty": val_m["ratio_penalty"],
            "val_final_middle_to_late_rate": val_m["middle_to_late_rate"],
            "val_q_MAE": val_q["q_MAE"],
            "val_q_RMSE": val_q["q_RMSE"],
            "val_q_R2": val_q["q_R2"],
            "val_q_mono_violation": val_q["q_mono_violation"],
        })

    param_df = pd.DataFrame(param_rows).sort_values(
        ["val_score", "val_final_middle_to_late_rate", "val_final_ratio_penalty"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    param_df.to_csv(DIR_FINAL / "FINAL_probability_param_ranking.csv", index=False, encoding="utf-8-sig")

    best_params = tuple(param_df.iloc[0][["eta", "fine_weight", "temperature", "mid_floor", "late_tau", "order_blend"]].values)

    print("\nBest probability params:")
    print(best_params)

    # 9. 用最佳参数生成最终 val/test 结果
    pred_val_final = apply_probability_inference(
        pred_val_base,
        val_stage_prob,
        val_fine_prob,
        fine_to_stage,
        trans,
        best_params
    )

    pred_test_final = apply_probability_inference(
        pred_test_base,
        test_stage_prob,
        test_fine_prob,
        fine_to_stage,
        trans,
        best_params
    )

    pred_val_final.to_csv(DIR_PRED / "final_internal_val_predictions.csv", index=False, encoding="utf-8-sig")
    pred_test_final.to_csv(DIR_FINAL / "FINAL_best_test_C6_predictions.csv", index=False, encoding="utf-8-sig")

    # 10. 评价表
    rep_val, cm_val, ratio_val = evaluate_prediction_df(pred_val_final, "final_internal_val")
    rep_test, cm_test, ratio_test = evaluate_prediction_df(pred_test_final, "test_C6")

    report_long = pd.concat([rep_val, rep_test], axis=0).reset_index(drop=True)
    cm_long = pd.concat([cm_val, cm_test], axis=0).reset_index(drop=True)
    ratio_long = pd.concat([ratio_val, ratio_test], axis=0).reset_index(drop=True)

    report_long.to_csv(DIR_FINAL / "FINAL_classification_reports_long.csv", index=False, encoding="utf-8-sig")
    cm_long.to_csv(DIR_FINAL / "FINAL_confusion_matrices_long.csv", index=False, encoding="utf-8-sig")
    ratio_long.to_csv(DIR_FINAL / "FINAL_stage_ratio_comparison.csv", index=False, encoding="utf-8-sig")

    # 11. 可视化
    plot_confusion_matrices(cm_long, split="final_internal_val")
    plot_confusion_matrices(cm_long, split="test_C6")
    plot_probability_evolution(pred_test_final)
    plot_qhat_curve(pred_test_final)
    plot_stage_prediction_timeline(pred_test_final)
    plot_stage_ratio(ratio_long)

    # 12. 汇总
    y_test = pred_test_final["stage_true_id"].values.astype(int)
    y_raw = pred_test_final["stage_pred_raw"].values.astype(int)
    y_final = pred_test_final["stage_pred_final"].values.astype(int)
    y_mix = pred_test_final["stage_pred_mix"].values.astype(int)
    y_ordered = pred_test_final["stage_pred_ordered"].values.astype(int)

    raw_m = classification_metrics(y_test, y_raw)
    mix_m = classification_metrics(y_test, y_mix)
    ordered_m = classification_metrics(y_test, y_ordered)
    final_m = classification_metrics(y_test, y_final)
    q_m = calc_q_metrics(pred_test_final["q_true"].values, pred_test_final["q_hat"].values)

    summary = pd.DataFrame([{
        "experiment": RUN_NAME,
        "task": "Fine-grained degradation-state assisted stage probability classification with no-leak cross-condition validation",
        "feature_file": str(FEATURE_FILE),
        "train_conditions": "C1+C4",
        "proxy_validation": "C1->C4 and C4->C1",
        "test_condition": "C6",
        "raw_numeric_sensor_feature_count": len(raw_sensor_cols),
        "online_candidate_features": len(candidate_cols),
        "selected_stage_invariant_features": len(selected_cols),
        "stage_definition": "condition-relative q_deg + local rate; early/late explicitly defined, middle as stable degradation band",
        "feature_selection": "MI(stage) + MI(q_deg) + Spearman(q_deg) - C1/C4 domain instability + redundancy filtering",
        "fine_state_strategy": "train-only GMM with 5 ordered fine states; S0->early, S1-S3->middle, S4->late",
        "model": "TCN-GRU multitask encoder with q_hat regression, 3-stage classification and 5-fine-state auxiliary classification",
        "probability_inference": "raw stage probability + fine-state mapped probability + q_hat prior + late suppression + causal ordered filtering",
        "no_leakage_design": "C6 is only used for final evaluation; C6 labels are never used for feature selection, scaling, GMM, model selection or probability parameter selection",

        "best_arch_name": best_arch_row["arch_name"],
        "best_window_length": best_arch["window_length"],
        "best_dropout": best_arch["dropout"],
        "best_learning_rate": best_arch["lr"],
        "best_channels": str(best_arch["channels"]),
        "best_gru_hidden": best_arch["gru_hidden"],

        "best_eta": best_params[0],
        "best_fine_weight": best_params[1],
        "best_temperature": best_params[2],
        "best_mid_floor": best_params[3],
        "best_late_tau": best_params[4],
        "best_order_blend": best_params[5],

        "proxy_mean_score": best_arch_row["proxy_mean_score"],
        "proxy_worst_score": best_arch_row["proxy_worst_score"],
        "proxy_final_score": best_arch_row["proxy_final_score"],

        "test_raw_acc": raw_m["acc"],
        "test_raw_macro_f1": raw_m["f1"],
        "test_raw_middle_recall": raw_m["middle_recall"],
        "test_raw_middle_to_late_rate": raw_m["middle_to_late_rate"],

        "test_mix_acc": mix_m["acc"],
        "test_mix_macro_f1": mix_m["f1"],
        "test_mix_middle_recall": mix_m["middle_recall"],
        "test_mix_middle_to_late_rate": mix_m["middle_to_late_rate"],

        "test_ordered_acc": ordered_m["acc"],
        "test_ordered_macro_f1": ordered_m["f1"],
        "test_ordered_middle_recall": ordered_m["middle_recall"],
        "test_ordered_middle_to_late_rate": ordered_m["middle_to_late_rate"],

        "test_final_acc": final_m["acc"],
        "test_final_macro_f1": final_m["f1"],
        "test_final_middle_precision": final_m["middle_precision"],
        "test_final_middle_recall": final_m["middle_recall"],
        "test_final_middle_f1": final_m["middle_f1"],
        "test_final_ratio_penalty": final_m["ratio_penalty"],
        "test_final_middle_to_late_rate": final_m["middle_to_late_rate"],

        "test_q_MAE": q_m["q_MAE"],
        "test_q_RMSE": q_m["q_RMSE"],
        "test_q_R2": q_m["q_R2"],
        "test_q_mono_violation": q_m["q_mono_violation"],

        "FINAL_condition_relative_stage_thresholds": str(DIR_FINAL / "FINAL_condition_relative_stage_thresholds.csv"),
        "FINAL_selected_stage_invariant_features": str(DIR_FINAL / "FINAL_selected_stage_invariant_features.csv"),
        "FINAL_gmm_fine_state_mapping": str(DIR_FINAL / "FINAL_gmm_fine_state_mapping.csv"),
        "FINAL_proxy_model_ranking": str(DIR_FINAL / "FINAL_proxy_model_ranking.csv"),
        "FINAL_probability_param_ranking": str(DIR_FINAL / "FINAL_probability_param_ranking.csv"),
        "FINAL_best_test_C6_predictions": str(DIR_FINAL / "FINAL_best_test_C6_predictions.csv"),
        "FINAL_classification_reports_long": str(DIR_FINAL / "FINAL_classification_reports_long.csv"),
        "FINAL_confusion_matrices_long": str(DIR_FINAL / "FINAL_confusion_matrices_long.csv"),
        "FINAL_stage_ratio_comparison": str(DIR_FINAL / "FINAL_stage_ratio_comparison.csv"),
        "figures_dir": str(DIR_FIG),
    }])

    summary.to_csv(DIR_FINAL / "FINAL_experiment_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 120)
    print("实验完成。重点查看：")
    print(f"1. {DIR_FINAL / 'FINAL_experiment_summary.csv'}")
    print(f"2. {DIR_FINAL / 'FINAL_classification_reports_long.csv'}")
    print(f"3. {DIR_FINAL / 'FINAL_confusion_matrices_long.csv'}")
    print(f"4. {DIR_FINAL / 'FINAL_stage_ratio_comparison.csv'}")
    print(f"5. {DIR_FINAL / 'FINAL_best_test_C6_predictions.csv'}")
    print(f"6. {DIR_FINAL / 'FINAL_selected_stage_invariant_features.csv'}")
    print(f"7. {DIR_FIG}")
    print("=" * 120)


if __name__ == "__main__":
    main()
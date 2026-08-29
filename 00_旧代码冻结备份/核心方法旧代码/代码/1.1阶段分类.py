# -*- coding: utf-8 -*-
"""
38run：跨工况稳健阶段识别模块实验
Condition-invariant stage probability module with cross-condition proxy validation

核心目标：
1. 不再手工指定阶段敏感特征，而是用 MI + q_deg 相关性 + 跨工况稳定性惩罚 + 冗余过滤自动筛选；
2. 对输入特征做在线工况不变处理，减少 C1/C4/C6 的幅值分布差异；
3. 验证方式从 C1/C4 grouped validation 改为跨工况代理验证：
   - proxy-1: C1 train -> C4 validation
   - proxy-2: C4 train -> C1 validation
   最终选模看两个代理任务平均表现；
4. 模型采用多任务学习：
   - q_hat 连续退化位置回归；
   - early/middle/late 阶段分类；
5. 最终阶段概率：
   p_final = causal_ordered_filter( eta * p_net + (1 - eta) * p_prior(q_hat) )
   其中 p_prior 来自 q_hat，不使用测试集真实 VB，不使用 C6 全生命周期归一化标签；
6. 仅使用 C1/C4 标签选特征、调模型、选参数；C6 只作为最终测试。
"""

from pathlib import Path
import copy
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
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

RUN_NAME = "38run_stage_probability_condition_invariant_proxy_validation"
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

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# 训练参数
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 16
WEIGHT_DECAY = 1e-5

# 搜索空间：控制规模，不再暴力跑几百上千组
WINDOW_LENGTH_LIST = [10, 12, 15]
DROPOUT_LIST = [0.1, 0.2]
LEARNING_RATE_LIST = [1e-3, 5e-4]
CHANNELS_LIST = [
    (16, 32, 32),
    (32, 64, 64),
]

# 概率融合参数，训练一次网络后再搜索这些后处理参数
ETA_LIST = [0.45, 0.55, 0.65]
TEMPERATURE_LIST = [1.0, 1.2]
MID_FLOOR_LIST = [0.03, 0.06, 0.10]
LATE_TAU_LIST = [0.56, 0.60, 0.64]
ORDER_BLEND_LIST = [0.0, 0.25]  # 0 表示不用有序滤波；0.25 表示轻量因果有序修正

# 阶段定义参数
STAGE_NAMES = ["early", "middle", "late"]
STAGE_TO_ID = {"early": 0, "middle": 1, "late": 2}
ID_TO_STAGE = {0: "early", 1: "middle", 2: "late"}

Q_EARLY_QUANTILE = 0.30
Q_LATE_QUANTILE = 0.72
RATE_LATE_QUANTILE = 0.78

# 在线工况不变特征参数
BASELINE_N = 30
ROLLING_SLOPE_WINDOW = 5

# 特征选择参数
VAR_THRESHOLD = 1e-8
TOP_FEATURE_CANDIDATES = 180
MAX_SELECTED_FEATURES = 70
REDUNDANCY_THRESHOLD = 0.92
MIN_FEATURE_SCORE = 1e-4

# 多任务损失权重
LAMBDA_Q = 0.65
LAMBDA_STAGE = 1.00
LAMBDA_MONO_Q = 0.03

# 选模权重
SCORE_ACC_W = 1.00
SCORE_F1_W = 1.20
SCORE_MIDDLE_RECALL_W = 1.20
SCORE_RATIO_W = 0.35
SCORE_Q_RMSE_W = 0.50
SCORE_Q_R2_W = 0.20

# 可视化
DPI = 600
TOPK = 10

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.sans-serif"] = ["Times New Roman", "SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["legend.frameon"] = False
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


# =========================================================
# 1. 基础函数
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
    s = str(x).strip()
    if s.lower() in ["c1", "1"]:
        return "C1"
    if s.lower() in ["c4", "4"]:
        return "C4"
    if s.lower() in ["c6", "6"]:
        return "C6"
    return s.upper()


def infer_vb_column(df):
    cols = list(df.columns)
    if "VB" in cols:
        return "VB"
    if "VB_max" in cols:
        return "VB_max"
    if "vb_max" in cols:
        return "vb_max"

    lower_map = {str(c).lower(): c for c in cols}
    if "vb" in lower_map:
        return lower_map["vb"]
    if "vb_max" in lower_map:
        return lower_map["vb_max"]

    raise ValueError("找不到目标标签列。请确认特征表中存在 VB 或 VB_max / vb_max。")


def is_label_or_meta_column(col):
    c = str(col).lower()

    exact_exclude = {
        "condition", "run_id", "cut", "file_name", "signal_len", "n_channels",
        "vb", "vb_max", "vbmean", "vb_mean", "vb max", "vb mean",
        "flute_1", "flute_2", "flute_3",
        "final_dominant_flute", "dominant_flute",
        "initial_vb", "final_vb", "vb_range",
        "mean_vb", "std_vb", "mean_increment", "median_increment",
        "max_increment", "min_increment", "negative_increment_count",
        "first_cut", "last_cut", "n_cuts",
        "stage", "stage_id", "phase", "phase_id",
        "q_deg", "q_hat", "vb_smooth", "rate", "rate_smooth", "rate_norm",
        "run_progress", "progress", "vb_norm", "stage_score",
    }

    if c in exact_exclude:
        return True

    forbidden_patterns = [
        "flute", "vb_mean", "vbmax", "vb_max", "wear_label",
        "label", "target", "dominant", "stage", "phase",
        "q_deg", "qhat", "q_hat", "dvb", "delta_vb", "rate",
        "progress", "vb_norm", "stage_score"
    ]

    return any(p in c for p in forbidden_patterns)


def get_numeric_sensor_columns(df, target_col="VB"):
    candidate_cols = []
    excluded_cols = []

    for col in df.columns:
        if col == target_col or is_label_or_meta_column(col):
            excluded_cols.append(col)
            continue

        numeric_series = pd.to_numeric(df[col], errors="coerce")
        if numeric_series.notna().mean() > 0.95:
            candidate_cols.append(col)
        else:
            excluded_cols.append(col)

    return candidate_cols, excluded_cols


def calc_q_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "q_MAE": mean_absolute_error(y_true, y_pred),
        "q_RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "q_R2": r2_score(y_true, y_pred),
    }


def calc_clf_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    per_prec, per_rec, per_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average=None,
        zero_division=0
    )

    return {
        "Accuracy": acc,
        "Precision_macro": precision,
        "Recall_macro": recall,
        "F1_macro": f1,
        "early_precision": per_prec[0],
        "early_recall": per_rec[0],
        "early_f1": per_f1[0],
        "middle_precision": per_prec[1],
        "middle_recall": per_rec[1],
        "middle_f1": per_f1[1],
        "late_precision": per_prec[2],
        "late_recall": per_rec[2],
        "late_f1": per_f1[2],
    }


def stage_ratio_penalty(y_true, y_pred):
    true_ratio = np.bincount(y_true, minlength=3) / max(len(y_true), 1)
    pred_ratio = np.bincount(y_pred, minlength=3) / max(len(y_pred), 1)
    return float(np.sum(np.abs(true_ratio - pred_ratio)))


def monotonic_violation_ratio(q_seq):
    q_seq = np.asarray(q_seq, dtype=float)
    if len(q_seq) <= 1:
        return 0.0
    return float(np.mean(np.diff(q_seq) < -0.02))


# =========================================================
# 2. 数据读取与工况内相对阶段定义
# =========================================================
def build_condition_relative_stage_labels(df):
    """
    每个 condition 内部：
    1. 平滑 VB；
    2. 构造 q_deg = 工况内相对退化位置；
    3. 构造局部速率 rate_norm；
    4. early：q_deg 较低；
    5. late：q_deg 较高或速率较高；
    6. middle：其余样本。
    """
    out_parts = []
    threshold_rows = []

    for cond, sub in df.groupby("condition"):
        sub = sub.sort_values("run_id").reset_index(drop=True).copy()

        vb = sub["VB"].astype(float).values
        vb_smooth = pd.Series(vb).rolling(window=7, min_periods=1, center=True).mean().values

        vb_min = float(np.min(vb_smooth))
        vb_max = float(np.max(vb_smooth))
        q_deg = (vb_smooth - vb_min) / (vb_max - vb_min + 1e-8)
        q_deg = np.clip(q_deg, 0.0, 1.0)

        rate = pd.Series(q_deg).diff().fillna(0.0).rolling(
            window=5,
            min_periods=1,
            center=True
        ).mean().values
        rate_min = float(np.min(rate))
        rate_max = float(np.max(rate))
        rate_norm = (rate - rate_min) / (rate_max - rate_min + 1e-8)
        rate_norm = np.clip(rate_norm, 0.0, 1.0)

        q_early_th = float(np.quantile(q_deg, Q_EARLY_QUANTILE))
        q_late_th = float(np.quantile(q_deg, Q_LATE_QUANTILE))
        rate_late_th = float(np.quantile(rate_norm, RATE_LATE_QUANTILE))

        stage = []
        for qv, rv in zip(q_deg, rate_norm):
            if qv <= q_early_th:
                stage.append("early")
            elif (qv >= q_late_th) or (rv >= rate_late_th and qv > q_early_th):
                stage.append("late")
            else:
                stage.append("middle")

        sub["VB_smooth"] = vb_smooth
        sub["q_deg"] = q_deg
        sub["rate_norm"] = rate_norm
        sub["stage"] = stage
        sub["stage_id"] = sub["stage"].map(STAGE_TO_ID).astype(int)

        threshold_rows.append({
            "condition": cond,
            "q_early_quantile": Q_EARLY_QUANTILE,
            "q_late_quantile": Q_LATE_QUANTILE,
            "rate_late_quantile": RATE_LATE_QUANTILE,
            "q_early_th": q_early_th,
            "q_late_th": q_late_th,
            "rate_late_th": rate_late_th,
            "VB_min": float(sub["VB"].min()),
            "VB_max": float(sub["VB"].max()),
            "VB_smooth_min": vb_min,
            "VB_smooth_max": vb_max,
            "stage_count": str(sub["stage"].value_counts().reindex(STAGE_NAMES, fill_value=0).to_dict())
        })

        out_parts.append(sub)

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(
        DIR_FINAL / "FINAL_condition_relative_stage_thresholds.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return pd.concat(out_parts, axis=0).sort_values(["condition", "run_id"]).reset_index(drop=True)


def load_feature_table():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"找不到特征表：{FEATURE_FILE}")

    df = pd.read_csv(FEATURE_FILE)
    df.columns = [str(c).strip() for c in df.columns]

    if "condition" not in df.columns or "run_id" not in df.columns:
        raise ValueError("特征表缺少 condition 或 run_id 列。")

    df["condition"] = df["condition"].apply(normalize_condition_name)
    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce").astype(int)

    target_col = infer_vb_column(df)
    df["VB"] = pd.to_numeric(df[target_col], errors="coerce")

    df = df[df["condition"].isin(["C1", "C4", "C6"])].copy()
    df = df.dropna(subset=["VB"]).sort_values(["condition", "run_id"]).reset_index(drop=True)

    df = build_condition_relative_stage_labels(df)
    df.to_csv(DIR_INTERIM / "loaded_feature_table_with_condition_relative_stage.csv", index=False, encoding="utf-8-sig")

    return df


# =========================================================
# 3. 在线工况不变特征构造
# =========================================================
def build_online_condition_invariant_features(df, sensor_cols, baseline_n=30):
    """
    对每个 condition 独立构造在线相对特征：
    1. baseline median/IQR：只用每个 condition 前 baseline_n 个 run 的无标签特征；
    2. rel = (x - median_baseline) / IQR_baseline；
    3. delta = rel.diff()；
    4. slope = delta 的过去窗口滚动均值。

    注意：
    - 对 C6 也只使用 C6 前 baseline_n 个 run 的传感器特征作为在线基准；
    - 不使用 C6 的 VB、stage、q_deg；
    - 这是无标签在线校准，不属于标签泄露。
    """
    out_parts = []
    new_feature_cols = []
    baseline_rows = []

    for cond, sub in df.groupby("condition"):
        sub = sub.sort_values("run_id").reset_index(drop=True).copy()

        base = sub.iloc[:min(baseline_n, len(sub))].copy()

        for col in sensor_cols:
            x = pd.to_numeric(sub[col], errors="coerce").astype(float)
            base_x = pd.to_numeric(base[col], errors="coerce").astype(float)

            med = float(np.nanmedian(base_x.values))
            q75 = float(np.nanpercentile(base_x.values, 75))
            q25 = float(np.nanpercentile(base_x.values, 25))
            iqr = q75 - q25
            if not np.isfinite(iqr) or abs(iqr) < 1e-8:
                iqr = float(np.nanstd(base_x.values))
            if not np.isfinite(iqr) or abs(iqr) < 1e-8:
                iqr = 1.0

            rel_name = f"{col}__rel"
            delta_name = f"{col}__delta"
            slope_name = f"{col}__slope"

            rel = (x - med) / iqr
            rel = rel.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            delta = rel.diff().fillna(0.0)
            slope = delta.rolling(window=ROLLING_SLOPE_WINDOW, min_periods=1).mean()

            sub[rel_name] = rel.values
            sub[delta_name] = delta.values
            sub[slope_name] = slope.values

            for name in [rel_name, delta_name, slope_name]:
                if name not in new_feature_cols:
                    new_feature_cols.append(name)

            baseline_rows.append({
                "condition": cond,
                "feature": col,
                "baseline_n": len(base),
                "baseline_median": med,
                "baseline_iqr_or_std": iqr
            })

        out_parts.append(sub)

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(DIR_FINAL / "FINAL_online_baseline_feature_statistics.csv", index=False, encoding="utf-8-sig")

    out = pd.concat(out_parts, axis=0).sort_values(["condition", "run_id"]).reset_index(drop=True)
    return out, new_feature_cols


# =========================================================
# 4. 特征筛选：MI + q 相关性 + 跨工况稳定性惩罚 + 冗余过滤
# =========================================================
def fill_train_medians(df, feature_cols, medians=None):
    out = df.copy()
    if medians is None:
        medians = {}

    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if col not in medians:
            med = out[col].median()
            if pd.isna(med):
                med = 0.0
            medians[col] = med
        out[col] = out[col].fillna(medians[col])

    return out, medians


def select_stage_invariant_features(df_train_all, feature_cols):
    """
    只使用 C1+C4 标签进行阶段敏感特征选择。
    评分：
    score = 0.40 * MI(stage)
          + 0.35 * MI(q_deg)
          + 0.15 * |Spearman(feature, q_deg)|
          - 0.10 * domain_instability(C1, C4)

    domain_instability 越小越好，用 C1/C4 的均值差与方差差衡量。
    """
    train_df = df_train_all[df_train_all["condition"].isin(["C1", "C4"])].copy()
    train_df, medians = fill_train_medians(train_df, feature_cols, medians=None)

    # 方差过滤
    vt = VarianceThreshold(threshold=VAR_THRESHOLD)
    X_raw = train_df[feature_cols].values
    vt.fit(X_raw)
    kept_cols = [feature_cols[i] for i in range(len(feature_cols)) if vt.get_support()[i]]

    train_df = train_df[["condition", "stage_id", "q_deg"] + kept_cols].copy()
    X = train_df[kept_cols].values
    y_stage = train_df["stage_id"].values.astype(int)
    y_q = train_df["q_deg"].values.astype(float)

    mi_stage = mutual_info_classif(X, y_stage, random_state=RANDOM_SEED)
    mi_q = mutual_info_regression(X, y_q, random_state=RANDOM_SEED)

    def norm01(a):
        a = np.asarray(a, dtype=float)
        return (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a) + 1e-12)

    mi_stage_n = norm01(mi_stage)
    mi_q_n = norm01(mi_q)

    rows = []
    for i, col in enumerate(kept_cols):
        x = train_df[col].values
        rho, _ = spearmanr(x, y_q, nan_policy="omit")
        rho = abs(rho) if pd.notna(rho) else 0.0

        c1 = train_df[train_df["condition"] == "C1"][col].values
        c4 = train_df[train_df["condition"] == "C4"][col].values

        mean_gap = abs(np.nanmean(c1) - np.nanmean(c4))
        std_pool = np.nanstd(np.concatenate([c1, c4])) + 1e-8
        mean_instability = mean_gap / std_pool

        std_gap = abs(np.nanstd(c1) - np.nanstd(c4)) / (std_pool + 1e-8)
        instability = mean_instability + 0.5 * std_gap

        score = (
                0.40 * mi_stage_n[i]
                + 0.35 * mi_q_n[i]
                + 0.15 * rho
                - 0.10 * instability
        )

        rows.append({
            "feature": col,
            "mi_stage": float(mi_stage[i]),
            "mi_q": float(mi_q[i]),
            "spearman_abs_q": float(rho),
            "domain_instability_C1_C4": float(instability),
            "feature_score": float(score),
        })

    score_df = pd.DataFrame(rows).sort_values("feature_score", ascending=False).reset_index(drop=True)
    score_df.to_csv(DIR_INTERIM / "stage_invariant_feature_scores.csv", index=False, encoding="utf-8-sig")

    candidates = score_df[
        score_df["feature_score"] >= MIN_FEATURE_SCORE
        ].head(TOP_FEATURE_CANDIDATES)["feature"].tolist()

    selected = []
    records = []

    for feat in candidates:
        feat_score = float(score_df.loc[score_df["feature"] == feat, "feature_score"].iloc[0])

        if len(selected) == 0:
            selected.append(feat)
            records.append({
                "feature": feat,
                "selected": True,
                "reason": "first_feature",
                "feature_score": feat_score,
                "max_abs_spearman_with_selected": np.nan,
            })
            continue

        corrs = []
        x_feat = train_df[feat].values

        for sf in selected:
            rho, _ = spearmanr(x_feat, train_df[sf].values, nan_policy="omit")
            corrs.append(abs(rho) if pd.notna(rho) else 0.0)

        max_corr = max(corrs) if corrs else 0.0

        if max_corr < REDUNDANCY_THRESHOLD:
            selected.append(feat)
            records.append({
                "feature": feat,
                "selected": True,
                "reason": "kept",
                "feature_score": feat_score,
                "max_abs_spearman_with_selected": max_corr,
            })
        else:
            records.append({
                "feature": feat,
                "selected": False,
                "reason": "redundant",
                "feature_score": feat_score,
                "max_abs_spearman_with_selected": max_corr,
            })

        if len(selected) >= MAX_SELECTED_FEATURES:
            break

    selected_df = pd.DataFrame(records)
    selected_df.to_csv(DIR_INTERIM / "stage_invariant_feature_selection_process.csv", index=False, encoding="utf-8-sig")

    selected_final = score_df[score_df["feature"].isin(selected)].copy()
    selected_final.insert(0, "selected_rank", np.arange(1, len(selected_final) + 1))
    selected_final.to_csv(DIR_FINAL / "FINAL_selected_stage_invariant_features.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([{
        "raw_feature_count": len(feature_cols),
        "after_variance_filter": len(kept_cols),
        "selected_stage_invariant_features": len(selected),
    }])
    summary.to_csv(DIR_FINAL / "FINAL_feature_selection_summary.csv", index=False, encoding="utf-8-sig")

    return selected, medians, selected_final


# =========================================================
# 5. 数据划分与窗口构建
# =========================================================
def make_grouped_internal_val_split(df_train_conditions):
    """
    最终模型训练时使用 C1+C4 的 grouped lifecycle validation 做 early stopping。
    注意：最终选模主要来自 C1->C4 / C4->C1 代理验证，这里只用于训练停止。
    """
    train_parts = []
    val_parts = []

    for cond in ["C1", "C4"]:
        sub = df_train_conditions[df_train_conditions["condition"] == cond].sort_values("run_id").reset_index(drop=True)

        val_indices = []
        for stage in STAGE_NAMES:
            g = sub[sub["stage"] == stage].sort_values("run_id")
            if len(g) == 0:
                continue
            n = len(g)
            block_len = max(8, int(round(0.20 * n)))
            block_len = min(block_len, max(1, n - 2))
            start = max(0, (n - block_len) // 2)
            idx = g.iloc[start:start + block_len].index.tolist()
            val_indices.extend(idx)

        val_indices = sorted(set(val_indices))
        val_sub = sub.loc[val_indices].copy()
        train_sub = sub.drop(index=val_indices).copy()

        train_parts.append(train_sub)
        val_parts.append(val_sub)

    train_df = pd.concat(train_parts, axis=0).sort_values(["condition", "run_id"]).reset_index(drop=True)
    val_df = pd.concat(val_parts, axis=0).sort_values(["condition", "run_id"]).reset_index(drop=True)

    return train_df, val_df


def save_split_metadata(df):
    rows = []

    for split_name, conds in [
        ("proxy_train_C1", ["C1"]),
        ("proxy_val_C4", ["C4"]),
        ("proxy_train_C4", ["C4"]),
        ("proxy_val_C1", ["C1"]),
        ("final_test_C6", ["C6"]),
    ]:
        for cond in conds:
            g = df[df["condition"] == cond].sort_values("run_id")
            rows.append({
                "split": split_name,
                "condition": cond,
                "n_samples": len(g),
                "run_start": int(g["run_id"].min()),
                "run_end": int(g["run_id"].max()),
                "VB_min": float(g["VB"].min()),
                "VB_max": float(g["VB"].max()),
                "q_min": float(g["q_deg"].min()),
                "q_max": float(g["q_deg"].max()),
                "stage_count": str(g["stage"].value_counts().reindex(STAGE_NAMES, fill_value=0).to_dict())
            })

    split_df = pd.DataFrame(rows)
    split_df.to_csv(DIR_FINAL / "FINAL_split_metadata.csv", index=False, encoding="utf-8-sig")


def build_windows(df_sub, feature_cols, window_length):
    X_list, y_stage_list, y_q_list, meta_rows = [], [], [], []

    for cond, sub in df_sub.groupby("condition"):
        sub = sub.sort_values("run_id").reset_index(drop=True)

        run_ids = sub["run_id"].values
        X_feat = sub[feature_cols].values.astype(np.float32)
        y_stage = sub["stage_id"].values.astype(np.int64)
        y_q = sub["q_deg"].values.astype(np.float32)

        for end_idx in range(window_length - 1, len(sub)):
            start_idx = end_idx - window_length + 1
            run_window = run_ids[start_idx:end_idx + 1]

            if not np.all(np.diff(run_window) == 1):
                continue

            X_list.append(X_feat[start_idx:end_idx + 1])
            y_stage_list.append(y_stage[end_idx])
            y_q_list.append(y_q[end_idx])

            meta_rows.append({
                "condition": cond,
                "run_id_start": int(run_ids[start_idx]),
                "run_id_end": int(run_ids[end_idx]),
                "cut_index": int(run_ids[end_idx]),
                "window_length": int(window_length),
                "stage_true": ID_TO_STAGE[int(y_stage[end_idx])],
                "stage_true_id": int(y_stage[end_idx]),
                "q_true": float(y_q[end_idx]),
                "VB_true": float(sub["VB"].iloc[end_idx]),
                "VB_smooth": float(sub["VB_smooth"].iloc[end_idx]),
            })

    return (
        np.asarray(X_list, dtype=np.float32),
        np.asarray(y_stage_list, dtype=np.int64),
        np.asarray(y_q_list, dtype=np.float32),
        pd.DataFrame(meta_rows)
    )


class StageWindowDataset(Dataset):
    def __init__(self, X, y_stage, y_q):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_stage = torch.tensor(y_stage, dtype=torch.long)
        self.y_q = torch.tensor(y_q, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_stage[idx], self.y_q[idx]


# =========================================================
# 6. 模型：TCN 多任务阶段识别
# =========================================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


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


class MultiTaskStageTCN(nn.Module):
    def __init__(self, input_dim, channels=(32, 64, 64), kernel_size=3, dropout=0.1):
        super().__init__()

        layers = []
        in_ch = input_dim

        for i, out_ch in enumerate(channels):
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation=2 ** i, dropout=dropout))
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)
        hid = channels[-1]

        self.proj = nn.Sequential(
            nn.Linear(hid, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.stage_head = nn.Linear(64, 3)
        self.q_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.tcn(x)
        h = h[:, :, -1]
        z = self.proj(h)

        logits = self.stage_head(z)
        q_hat = self.q_head(z)

        return {
            "stage_logits": logits,
            "stage_prob": F.softmax(logits, dim=1),
            "q_hat": q_hat
        }


# =========================================================
# 7. 概率融合与因果有序滤波
# =========================================================
def q_prior_prob(q_hat, mid_floor=0.03, late_tau=0.60):
    """
    用 q_hat 构造阶段先验概率。
    这是无标签先验，只依赖网络预测的 q_hat。
    """
    q = np.asarray(q_hat, dtype=float).reshape(-1)
    sigma = 0.16

    centers = np.array([0.15, 0.50, 0.85], dtype=float)
    prob = np.exp(-0.5 * ((q[:, None] - centers[None, :]) / sigma) ** 2)

    # late 抑制：当 q_hat 还没到 late 区域时，不让 late 过早占主导
    late_gate = 1.0 / (1.0 + np.exp(-18.0 * (q - late_tau)))
    prob[:, 2] = prob[:, 2] * late_gate

    # middle floor：防止 middle 概率被完全压成 0
    prob[:, 1] = np.maximum(prob[:, 1], mid_floor)

    prob = prob / (prob.sum(axis=1, keepdims=True) + 1e-12)
    return prob


def estimate_transition_matrix_from_labels(df_train):
    """
    只从 C1/C4 训练标签估计 left-to-right transition。
    为避免过强转移导致塌缩，采用温和转移矩阵。
    """
    counts = np.ones((3, 3), dtype=float) * 0.5

    for cond, sub in df_train.groupby("condition"):
        s = sub.sort_values("run_id")["stage_id"].values.astype(int)
        for a, b in zip(s[:-1], s[1:]):
            if b >= a:
                counts[a, b] += 1.0
            else:
                # 少量允许回退，但权重很低
                counts[a, b] += 0.1

    trans = counts / (counts.sum(axis=1, keepdims=True) + 1e-12)

    # 温和化，避免转移矩阵过度支配观测概率
    base = np.array([
        [0.90, 0.09, 0.01],
        [0.02, 0.92, 0.06],
        [0.01, 0.03, 0.96],
    ])
    trans = 0.5 * trans + 0.5 * base
    trans = trans / (trans.sum(axis=1, keepdims=True) + 1e-12)

    pd.DataFrame(trans, index=STAGE_NAMES, columns=STAGE_NAMES).to_csv(
        DIR_FINAL / "FINAL_transition_matrix_train_only.csv",
        encoding="utf-8-sig"
    )

    return trans


def causal_forward_filter(prob_seq, trans_mat):
    """
    因果 forward filtering，只使用当前与过去，不使用未来。
    """
    prob_seq = np.asarray(prob_seq, dtype=float)
    T = len(prob_seq)
    out = np.zeros_like(prob_seq)

    if T == 0:
        return out

    alpha = prob_seq[0] / (prob_seq[0].sum() + 1e-12)
    out[0] = alpha

    for t in range(1, T):
        pred = alpha @ trans_mat
        alpha = pred * prob_seq[t]
        alpha = alpha / (alpha.sum() + 1e-12)
        out[t] = alpha

    return out


def apply_probability_methods(pred_df, raw_prob, q_hat, trans_mat,
                              eta=0.55, temperature=1.0,
                              mid_floor=0.03, late_tau=0.60,
                              order_blend=0.25):
    """
    raw: 网络 softmax
    mix: eta * raw + (1-eta) * q_hat prior
    ordered: 因果 forward filter
    final: (1-order_blend)*mix + order_blend*ordered
    """
    raw = np.asarray(raw_prob, dtype=float)
    raw = np.power(np.clip(raw, 1e-12, 1.0), 1.0 / temperature)
    raw = raw / (raw.sum(axis=1, keepdims=True) + 1e-12)

    prior = q_prior_prob(q_hat, mid_floor=mid_floor, late_tau=late_tau)

    mix = eta * raw + (1.0 - eta) * prior
    mix = mix / (mix.sum(axis=1, keepdims=True) + 1e-12)

    ordered = np.zeros_like(mix)

    for cond, idx in pred_df.groupby("condition").groups.items():
        idx = list(idx)
        sub = pred_df.loc[idx].sort_values("cut_index")
        order = sub.index.tolist()

        filt = causal_forward_filter(mix[order], trans_mat)
        ordered[order] = filt

    final = (1.0 - order_blend) * mix + order_blend * ordered
    final = final / (final.sum(axis=1, keepdims=True) + 1e-12)

    return raw, prior, mix, ordered, final


# =========================================================
# 8. 训练与评估
# =========================================================
def compute_loss(outputs, y_stage, y_q):
    loss_stage = F.cross_entropy(outputs["stage_logits"], y_stage)
    loss_q = F.smooth_l1_loss(outputs["q_hat"], y_q)

    # q_hat 单调性约束：一个 batch 内近似约束，不作为强约束
    q = outputs["q_hat"].view(-1)
    if len(q) > 1:
        mono_loss = torch.relu(q[:-1] - q[1:] - 0.02).mean()
    else:
        mono_loss = torch.tensor(0.0, device=q.device)

    total = LAMBDA_STAGE * loss_stage + LAMBDA_Q * loss_q + LAMBDA_MONO_Q * mono_loss

    return total, {
        "loss_stage": float(loss_stage.detach().cpu()),
        "loss_q": float(loss_q.detach().cpu()),
        "loss_mono_q": float(mono_loss.detach().cpu()),
    }


def run_epoch(model, loader, optimizer=None, scaler_amp=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    logs = []

    prob_all, logits_all, qhat_all = [], [], []
    y_stage_all, y_q_all = [], []

    for X_batch, y_stage, y_q in loader:
        X_batch = X_batch.to(DEVICE)
        y_stage = y_stage.to(DEVICE)
        y_q = y_q.to(DEVICE)

        with torch.set_grad_enabled(is_train):
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    outputs = model(X_batch)
                    loss, log = compute_loss(outputs, y_stage, y_q)
            else:
                outputs = model(X_batch)
                loss, log = compute_loss(outputs, y_stage, y_q)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if USE_AMP:
                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

        total_loss += float(loss.detach().cpu()) * X_batch.size(0)
        logs.append(log)

        prob_all.append(outputs["stage_prob"].detach().cpu().numpy())
        logits_all.append(outputs["stage_logits"].detach().cpu().numpy())
        qhat_all.append(outputs["q_hat"].detach().cpu().numpy())

        y_stage_all.append(y_stage.detach().cpu().numpy())
        y_q_all.append(y_q.detach().cpu().numpy())

    prob_all = np.concatenate(prob_all, axis=0)
    logits_all = np.concatenate(logits_all, axis=0)
    qhat_all = np.concatenate(qhat_all, axis=0).reshape(-1)
    y_stage_all = np.concatenate(y_stage_all, axis=0).reshape(-1)
    y_q_all = np.concatenate(y_q_all, axis=0).reshape(-1)

    return {
        "loss": total_loss / len(loader.dataset),
        "stage_prob": prob_all,
        "stage_logits": logits_all,
        "q_hat": qhat_all,
        "stage_true": y_stage_all,
        "q_true": y_q_all,
        "loss_log": pd.DataFrame(logs).mean().to_dict() if len(logs) else {},
    }


def train_model_for_arch(train_df, val_df, feature_cols, window_length, dropout, lr, channels, tag):
    set_seed(RANDOM_SEED)

    X_train, y_stage_train, y_q_train, meta_train = build_windows(train_df, feature_cols, window_length)
    X_val, y_stage_val, y_q_val, meta_val = build_windows(val_df, feature_cols, window_length)

    if len(X_train) == 0 or len(X_val) == 0:
        return None

    scaler = StandardScaler()
    n_train, L, d = X_train.shape
    X_train_2d = X_train.reshape(-1, d)
    X_val_2d = X_val.reshape(-1, d)

    X_train = scaler.fit_transform(X_train_2d).reshape(n_train, L, d).astype(np.float32)
    X_val = scaler.transform(X_val_2d).reshape(X_val.shape[0], L, d).astype(np.float32)

    train_loader = DataLoader(StageWindowDataset(X_train, y_stage_train, y_q_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(StageWindowDataset(X_val, y_stage_val, y_q_val), batch_size=BATCH_SIZE, shuffle=False)

    model = MultiTaskStageTCN(
        input_dim=len(feature_cols),
        channels=channels,
        kernel_size=3,
        dropout=dropout
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-5
    )
    scaler_amp = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_score = np.inf
    best_state = None
    best_epoch = 0
    patience_count = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_out = run_epoch(model, train_loader, optimizer=optimizer, scaler_amp=scaler_amp)
        val_out = run_epoch(model, val_loader, optimizer=None)

        val_raw_pred = np.argmax(val_out["stage_prob"], axis=1)
        val_clf = calc_clf_metrics(val_out["stage_true"], val_raw_pred)
        val_q = calc_q_metrics(val_out["q_true"], val_out["q_hat"])
        val_ratio = stage_ratio_penalty(val_out["stage_true"], val_raw_pred)

        val_score = (
                SCORE_ACC_W * (1.0 - val_clf["Accuracy"])
                + SCORE_F1_W * (1.0 - val_clf["F1_macro"])
                + SCORE_MIDDLE_RECALL_W * (1.0 - val_clf["middle_recall"])
                + SCORE_RATIO_W * val_ratio
                + SCORE_Q_RMSE_W * val_q["q_RMSE"]
                + SCORE_Q_R2_W * max(0.0, -val_q["q_R2"])
        )

        scheduler.step(val_score)

        history.append({
            "epoch": epoch,
            "train_loss": train_out["loss"],
            "val_loss": val_out["loss"],
            "val_score_raw": val_score,
            "val_acc_raw": val_clf["Accuracy"],
            "val_f1_raw": val_clf["F1_macro"],
            "val_middle_recall_raw": val_clf["middle_recall"],
            "val_ratio_penalty_raw": val_ratio,
            **val_q,
            **{f"train_{k}": v for k, v in train_out["loss_log"].items()},
        })

        if val_score < best_score:
            best_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(DIR_INTERIM / f"{tag}_history.csv", index=False, encoding="utf-8-sig")

    return {
        "model": model,
        "scaler": scaler,
        "best_epoch": best_epoch,
        "best_raw_val_score": best_score,
        "meta_train": meta_train,
        "meta_val": meta_val,
        "X_train": X_train,
        "y_stage_train": y_stage_train,
        "y_q_train": y_q_train,
        "X_val": X_val,
        "y_stage_val": y_stage_val,
        "y_q_val": y_q_val,
        "history": hist_df,
    }


def predict_with_model(model, scaler, df_sub, feature_cols, window_length):
    X, y_stage, y_q, meta = build_windows(df_sub, feature_cols, window_length)
    if len(X) == 0:
        raise RuntimeError("预测窗口为空，请检查 window_length 或数据连续性。")

    n, L, d = X.shape
    X_scaled = scaler.transform(X.reshape(-1, d)).reshape(n, L, d).astype(np.float32)

    loader = DataLoader(StageWindowDataset(X_scaled, y_stage, y_q), batch_size=BATCH_SIZE, shuffle=False)

    out = run_epoch(model, loader, optimizer=None)

    pred_df = meta.copy().reset_index(drop=True)
    pred_df["stage_true_id"] = out["stage_true"]
    pred_df["stage_true_name"] = pred_df["stage_true_id"].map(ID_TO_STAGE)
    pred_df["q_true"] = out["q_true"]
    pred_df["q_hat"] = out["q_hat"]

    for i, name in enumerate(STAGE_NAMES):
        pred_df[f"raw_prob_{name}"] = out["stage_prob"][:, i]

    return pred_df, out["stage_prob"], out["q_hat"]


def evaluate_probability_config(pred_df, raw_prob, q_hat, trans_mat,
                                eta, temperature, mid_floor, late_tau, order_blend,
                                split_name, method_prefix):
    raw, prior, mix, ordered, final = apply_probability_methods(
        pred_df=pred_df,
        raw_prob=raw_prob,
        q_hat=q_hat,
        trans_mat=trans_mat,
        eta=eta,
        temperature=temperature,
        mid_floor=mid_floor,
        late_tau=late_tau,
        order_blend=order_blend
    )

    y_true = pred_df["stage_true_id"].values.astype(int)

    methods = {
        "raw": raw,
        "prior": prior,
        "mix": mix,
        "ordered": ordered,
        "final": final,
    }

    metrics_rows = []
    pred_out = pred_df.copy()

    for method, prob in methods.items():
        pred = np.argmax(prob, axis=1)
        clf = calc_clf_metrics(y_true, pred)
        ratio_pen = stage_ratio_penalty(y_true, pred)

        metrics_rows.append({
            "split": split_name,
            "method": method,
            "Accuracy": clf["Accuracy"],
            "Precision_macro": clf["Precision_macro"],
            "Recall_macro": clf["Recall_macro"],
            "F1_macro": clf["F1_macro"],
            "middle_precision": clf["middle_precision"],
            "middle_recall": clf["middle_recall"],
            "middle_f1": clf["middle_f1"],
            "ratio_penalty": ratio_pen,
        })

        pred_out[f"stage_pred_{method}"] = pred
        pred_out[f"stage_pred_{method}_name"] = pd.Series(pred).map(ID_TO_STAGE).values

        for i, name in enumerate(STAGE_NAMES):
            pred_out[f"{method}_prob_{name}"] = prob[:, i]

    q_metrics = calc_q_metrics(pred_out["q_true"], pred_out["q_hat"])
    mono = monotonic_violation_ratio(pred_out.sort_values(["condition", "cut_index"])["q_hat"].values)

    for row in metrics_rows:
        row.update(q_metrics)
        row["mono_violation_qhat"] = mono
        row["eta"] = eta
        row["temperature"] = temperature
        row["mid_floor"] = mid_floor
        row["late_tau"] = late_tau
        row["order_blend"] = order_blend
        row["method_prefix"] = method_prefix

    return metrics_rows, pred_out


def proxy_score_from_rows(rows):
    df = pd.DataFrame(rows)
    final = df[df["method"] == "final"].iloc[0]

    score = (
            SCORE_ACC_W * (1.0 - final["Accuracy"])
            + SCORE_F1_W * (1.0 - final["F1_macro"])
            + SCORE_MIDDLE_RECALL_W * (1.0 - final["middle_recall"])
            + SCORE_RATIO_W * final["ratio_penalty"]
            + SCORE_Q_RMSE_W * final["q_RMSE"]
            + SCORE_Q_R2_W * max(0.0, -final["q_R2"])
    )

    return float(score)


# =========================================================
# 9. 代理验证搜索
# =========================================================
def run_proxy_search(df_all, selected_features):
    train_all = df_all[df_all["condition"].isin(["C1", "C4"])].copy()
    trans_mat = estimate_transition_matrix_from_labels(train_all)

    all_rank_rows = []

    arch_count = 0
    total_arch = (
            len(WINDOW_LENGTH_LIST)
            * len(DROPOUT_LIST)
            * len(LEARNING_RATE_LIST)
            * len(CHANNELS_LIST)
    )

    for L in WINDOW_LENGTH_LIST:
        for dropout in DROPOUT_LIST:
            for lr in LEARNING_RATE_LIST:
                for channels in CHANNELS_LIST:
                    arch_count += 1
                    arch_name = (
                        f"L{L}_drop{str(dropout).replace('.', 'p')}"
                        f"_lr{str(lr).replace('.', 'p')}"
                        f"_ch{'-'.join(map(str, channels))}"
                    )

                    print(f"\n[Proxy arch {arch_count}/{total_arch}] {arch_name}")

                    proxy_tasks = [
                        ("C1_to_C4", ["C1"], ["C4"]),
                        ("C4_to_C1", ["C4"], ["C1"]),
                    ]

                    proxy_pred_cache = {}
                    proxy_raw_score = []

                    valid_arch = True

                    for task_name, train_conds, val_conds in proxy_tasks:
                        train_df = df_all[df_all["condition"].isin(train_conds)].copy()
                        val_df = df_all[df_all["condition"].isin(val_conds)].copy()

                        tag = f"{arch_name}_{task_name}"

                        result = train_model_for_arch(
                            train_df=train_df,
                            val_df=val_df,
                            feature_cols=selected_features,
                            window_length=L,
                            dropout=dropout,
                            lr=lr,
                            channels=channels,
                            tag=tag
                        )

                        if result is None:
                            valid_arch = False
                            break

                        pred_df, raw_prob, q_hat = predict_with_model(
                            model=result["model"],
                            scaler=result["scaler"],
                            df_sub=val_df,
                            feature_cols=selected_features,
                            window_length=L
                        )

                        proxy_pred_cache[task_name] = {
                            "pred_df": pred_df,
                            "raw_prob": raw_prob,
                            "q_hat": q_hat,
                            "best_epoch": result["best_epoch"],
                            "raw_val_score": result["best_raw_val_score"],
                        }

                        raw_pred = np.argmax(raw_prob, axis=1)
                        raw_clf = calc_clf_metrics(pred_df["stage_true_id"].values, raw_pred)
                        raw_q = calc_q_metrics(pred_df["q_true"].values, q_hat)

                        proxy_raw_score.append({
                            "task": task_name,
                            "raw_acc": raw_clf["Accuracy"],
                            "raw_f1": raw_clf["F1_macro"],
                            "raw_middle_recall": raw_clf["middle_recall"],
                            **raw_q
                        })

                    if not valid_arch:
                        continue

                    for eta in ETA_LIST:
                        for temp in TEMPERATURE_LIST:
                            for mid_floor in MID_FLOOR_LIST:
                                for late_tau in LATE_TAU_LIST:
                                    for order_blend in ORDER_BLEND_LIST:

                                        task_scores = []
                                        task_metrics_flat = {}

                                        for task_name, cache in proxy_pred_cache.items():
                                            rows, pred_out = evaluate_probability_config(
                                                pred_df=cache["pred_df"],
                                                raw_prob=cache["raw_prob"],
                                                q_hat=cache["q_hat"],
                                                trans_mat=trans_mat,
                                                eta=eta,
                                                temperature=temp,
                                                mid_floor=mid_floor,
                                                late_tau=late_tau,
                                                order_blend=order_blend,
                                                split_name=task_name,
                                                method_prefix=arch_name
                                            )

                                            task_score = proxy_score_from_rows(rows)
                                            task_scores.append(task_score)

                                            df_rows = pd.DataFrame(rows)
                                            final = df_rows[df_rows["method"] == "final"].iloc[0]

                                            prefix = task_name
                                            task_metrics_flat[f"{prefix}_score"] = task_score
                                            task_metrics_flat[f"{prefix}_acc"] = final["Accuracy"]
                                            task_metrics_flat[f"{prefix}_f1"] = final["F1_macro"]
                                            task_metrics_flat[f"{prefix}_middle_recall"] = final["middle_recall"]
                                            task_metrics_flat[f"{prefix}_ratio_penalty"] = final["ratio_penalty"]
                                            task_metrics_flat[f"{prefix}_q_RMSE"] = final["q_RMSE"]
                                            task_metrics_flat[f"{prefix}_q_R2"] = final["q_R2"]

                                        mean_score = float(np.mean(task_scores))
                                        worst_score = float(np.max(task_scores))

                                        all_rank_rows.append({
                                            "arch_name": arch_name,
                                            "window_length": L,
                                            "dropout": dropout,
                                            "learning_rate": lr,
                                            "channels": str(channels),
                                            "eta": eta,
                                            "temperature": temp,
                                            "mid_floor": mid_floor,
                                            "late_tau": late_tau,
                                            "order_blend": order_blend,
                                            "proxy_mean_score": mean_score,
                                            "proxy_worst_score": worst_score,
                                            **task_metrics_flat
                                        })

                    print(f"  finished {arch_name}")

    ranking = pd.DataFrame(all_rank_rows)

    if ranking.empty:
        raise RuntimeError("代理验证搜索没有得到任何有效配置。")

    ranking = ranking.sort_values(
        ["proxy_mean_score", "proxy_worst_score", "C1_to_C4_middle_recall", "C4_to_C1_middle_recall"],
        ascending=[True, True, False, False]
    ).reset_index(drop=True)

    ranking.insert(0, "top_rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(DIR_FINAL / "FINAL_proxy_model_ranking.csv", index=False, encoding="utf-8-sig")

    return ranking, trans_mat


# =========================================================
# 10. 最终训练与 C6 测试
# =========================================================
def train_final_and_test(df_all, selected_features, best_cfg, trans_mat):
    train_all = df_all[df_all["condition"].isin(["C1", "C4"])].copy()
    test_c6 = df_all[df_all["condition"] == "C6"].copy()

    final_train, final_val = make_grouped_internal_val_split(train_all)

    L = int(best_cfg["window_length"])
    dropout = float(best_cfg["dropout"])
    lr = float(best_cfg["learning_rate"])
    channels = tuple(int(x.strip()) for x in str(best_cfg["channels"]).strip("()").split(","))

    eta = float(best_cfg["eta"])
    temp = float(best_cfg["temperature"])
    mid_floor = float(best_cfg["mid_floor"])
    late_tau = float(best_cfg["late_tau"])
    order_blend = float(best_cfg["order_blend"])

    tag = "FINAL_C1C4_train_internal_grouped_val"

    result = train_model_for_arch(
        train_df=final_train,
        val_df=final_val,
        feature_cols=selected_features,
        window_length=L,
        dropout=dropout,
        lr=lr,
        channels=channels,
        tag=tag
    )

    if result is None:
        raise RuntimeError("最终模型训练失败。")

    torch.save(result["model"].state_dict(), DIR_MODEL / "FINAL_best_stage_model.pth")

    # final validation
    val_pred_df, val_raw_prob, val_qhat = predict_with_model(
        model=result["model"],
        scaler=result["scaler"],
        df_sub=final_val,
        feature_cols=selected_features,
        window_length=L
    )

    val_rows, val_pred_out = evaluate_probability_config(
        pred_df=val_pred_df,
        raw_prob=val_raw_prob,
        q_hat=val_qhat,
        trans_mat=trans_mat,
        eta=eta,
        temperature=temp,
        mid_floor=mid_floor,
        late_tau=late_tau,
        order_blend=order_blend,
        split_name="final_internal_val",
        method_prefix="final_model"
    )

    # C6 test
    test_pred_df, test_raw_prob, test_qhat = predict_with_model(
        model=result["model"],
        scaler=result["scaler"],
        df_sub=test_c6,
        feature_cols=selected_features,
        window_length=L
    )

    test_rows, test_pred_out = evaluate_probability_config(
        pred_df=test_pred_df,
        raw_prob=test_raw_prob,
        q_hat=test_qhat,
        trans_mat=trans_mat,
        eta=eta,
        temperature=temp,
        mid_floor=mid_floor,
        late_tau=late_tau,
        order_blend=order_blend,
        split_name="test_C6",
        method_prefix="final_model"
    )

    val_pred_out.to_csv(DIR_FINAL / "FINAL_best_internal_val_predictions.csv", index=False, encoding="utf-8-sig")
    test_pred_out.to_csv(DIR_FINAL / "FINAL_best_test_C6_predictions.csv", index=False, encoding="utf-8-sig")

    report_rows = []
    confusion_rows = []
    ratio_rows = []

    for split_name, pred_out in [("final_internal_val", val_pred_out), ("test_C6", test_pred_out)]:
        y_true = pred_out["stage_true_id"].values.astype(int)

        for method in ["raw", "mix", "ordered", "final"]:
            y_pred = pred_out[f"stage_pred_{method}"].values.astype(int)

            report = classification_report(
                y_true,
                y_pred,
                labels=[0, 1, 2],
                target_names=STAGE_NAMES,
                output_dict=True,
                zero_division=0
            )

            for label, vals in report.items():
                if isinstance(vals, dict):
                    report_rows.append({
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
                    report_rows.append({
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
            row_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)

            for i, true_name in enumerate(STAGE_NAMES):
                for j, pred_name in enumerate(STAGE_NAMES):
                    confusion_rows.append({
                        "split": split_name,
                        "method": method,
                        "true_stage": true_name,
                        "pred_stage": pred_name,
                        "count": int(cm[i, j]),
                        "row_norm": float(row_norm[i, j]),
                    })

            true_ratio = np.bincount(y_true, minlength=3) / len(y_true)
            pred_ratio = np.bincount(y_pred, minlength=3) / len(y_pred)

            for i, s in enumerate(STAGE_NAMES):
                ratio_rows.append({
                    "split": split_name,
                    "method": "TRUE",
                    "stage": s,
                    "ratio": true_ratio[i]
                })
                ratio_rows.append({
                    "split": split_name,
                    "method": method,
                    "stage": s,
                    "ratio": pred_ratio[i]
                })

    report_df = pd.DataFrame(report_rows)
    confusion_df = pd.DataFrame(confusion_rows)
    ratio_df = pd.DataFrame(ratio_rows)

    report_df.to_csv(DIR_FINAL / "FINAL_classification_reports_long.csv", index=False, encoding="utf-8-sig")
    confusion_df.to_csv(DIR_FINAL / "FINAL_confusion_matrices_long.csv", index=False, encoding="utf-8-sig")
    ratio_df.to_csv(DIR_FINAL / "FINAL_stage_ratio_comparison.csv", index=False, encoding="utf-8-sig")

    test_final = pd.DataFrame(test_rows)
    test_final_metrics = test_final[test_final["method"] == "final"].iloc[0].to_dict()

    val_final = pd.DataFrame(val_rows)
    val_final_metrics = val_final[val_final["method"] == "final"].iloc[0].to_dict()

    return {
        "model": result["model"],
        "scaler": result["scaler"],
        "history": result["history"],
        "val_pred": val_pred_out,
        "test_pred": test_pred_out,
        "val_metrics": val_final_metrics,
        "test_metrics": test_final_metrics,
        "report_df": report_df,
        "confusion_df": confusion_df,
        "ratio_df": ratio_df,
    }


# =========================================================
# 11. 可视化
# =========================================================
def plot_condition_stage_curves(df_all):
    """
    三个工况的磨损曲线 + early/middle/late 阶段颜色标注。
    """
    colors = {"early": "#6BAED6", "middle": "#74C476", "late": "#FD8D3C"}

    for cond in ["C1", "C4", "C6"]:
        sub = df_all[df_all["condition"] == cond].sort_values("run_id").copy()

        plt.figure(figsize=(12.5, 4.8))
        plt.plot(sub["run_id"], sub["VB"], color="black", linewidth=2.2, label="True VB")
        plt.plot(sub["run_id"], sub["VB_smooth"], color="#4C78A8", linewidth=1.8, linestyle="--", label="Smoothed VB")

        for stage in STAGE_NAMES:
            g = sub[sub["stage"] == stage]
            plt.scatter(g["run_id"], g["VB"], s=22, color=colors[stage], label=stage, alpha=0.85)

        plt.xlabel("Cut index")
        plt.ylabel("Maximum flank wear VB")
        plt.title(f"{cond}: condition-relative stage division")
        plt.legend(loc="upper left", ncol=4)
        savefig(DIR_FIG / f"fig_stage_division_{cond}.png")

    # 合并图
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)
    for ax, cond in zip(axes, ["C1", "C4", "C6"]):
        sub = df_all[df_all["condition"] == cond].sort_values("run_id").copy()
        ax.plot(sub["run_id"], sub["VB"], color="black", linewidth=2.0, label="True VB")
        for stage in STAGE_NAMES:
            g = sub[sub["stage"] == stage]
            ax.scatter(g["run_id"], g["VB"], s=18, color=colors[stage], label=stage, alpha=0.85)
        ax.set_ylabel("VB")
        ax.set_title(f"{cond}")
        ax.legend(loc="upper left", ncol=4, fontsize=8)
    axes[-1].set_xlabel("Cut index")
    fig.suptitle("Condition-relative stage division of C1, C4 and C6", y=1.02)
    plt.tight_layout()
    plt.savefig(DIR_FIG / "fig_stage_division_all_conditions.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_q_and_rate(df_all):
    for cond in ["C1", "C4", "C6"]:
        sub = df_all[df_all["condition"] == cond].sort_values("run_id").copy()

        fig, axes = plt.subplots(2, 1, figsize=(12.5, 6.5), sharex=True)

        axes[0].plot(sub["run_id"], sub["q_deg"], linewidth=2.2)
        axes[0].set_ylabel("Relative degradation scale")
        axes[0].set_title(f"{cond}: q_deg and local degradation rate")

        axes[1].plot(sub["run_id"], sub["rate_norm"], linewidth=2.0)
        axes[1].set_xlabel("Cut index")
        axes[1].set_ylabel("Normalized local rate")

        plt.tight_layout()
        plt.savefig(DIR_FIG / f"fig_q_rate_{cond}.png", dpi=DPI, bbox_inches="tight")
        plt.close()


def plot_selected_features(selected_df):
    if selected_df.empty:
        return

    plot_df = selected_df.sort_values("feature_score", ascending=True).tail(40)

    plt.figure(figsize=(10, max(6, 0.28 * len(plot_df))))
    plt.barh(plot_df["feature"], plot_df["feature_score"], color=plt.cm.Blues(np.linspace(0.45, 0.9, len(plot_df))))
    plt.xlabel("Feature selection score")
    plt.ylabel("Selected stage-invariant feature")
    plt.title("Top selected stage-invariant features")
    savefig(DIR_FIG / "fig_selected_stage_invariant_features.png")


def plot_proxy_ranking(ranking):
    top = ranking.head(20).copy()

    plt.figure(figsize=(12, 5))
    labels = [f"Top-{i}" for i in top["top_rank"]]
    plt.bar(labels, top["proxy_mean_score"], edgecolor="black", linewidth=0.6)
    plt.xticks(rotation=45)
    plt.ylabel("Proxy mean score")
    plt.title("Top configurations by cross-condition proxy validation")
    savefig(DIR_FIG / "fig_proxy_ranking_top20.png")


def plot_test_probability_evolution(test_pred):
    pred = test_pred.sort_values("cut_index").copy()

    plt.figure(figsize=(13.5, 5.2))
    plt.plot(pred["cut_index"], pred["final_prob_early"], linewidth=2.0, label="Early probability")
    plt.plot(pred["cut_index"], pred["final_prob_middle"], linewidth=2.0, label="Middle probability")
    plt.plot(pred["cut_index"], pred["final_prob_late"], linewidth=2.0, label="Late probability")
    plt.xlabel("Cut index")
    plt.ylabel("Final stage probability")
    plt.title("C6: final stage probability evolution")
    plt.legend(loc="upper left")
    savefig(DIR_FIG / "fig_C6_final_stage_probability_evolution.png")


def plot_qhat_curve(test_pred):
    pred = test_pred.sort_values("cut_index").copy()

    plt.figure(figsize=(13.5, 5.2))
    plt.plot(pred["cut_index"], pred["q_true"], color="black", linewidth=2.4, label="True q_deg")
    plt.plot(pred["cut_index"], pred["q_hat"], color="#4C78A8", linewidth=2.2, label="Predicted q_hat")
    plt.xlabel("Cut index")
    plt.ylabel("Relative degradation scale")
    plt.title("C6: true q_deg and predicted q_hat")
    plt.legend(loc="upper left")
    savefig(DIR_FIG / "fig_C6_qhat_curve.png")


def plot_confusion_matrices(confusion_df):
    for split in confusion_df["split"].unique():
        sub_split = confusion_df[confusion_df["split"] == split]
        methods = ["raw", "mix", "ordered", "final"]

        fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))

        for ax, method in zip(axes, methods):
            sub = sub_split[sub_split["method"] == method]
            mat = np.zeros((3, 3))
            counts = np.zeros((3, 3), dtype=int)

            for _, row in sub.iterrows():
                i = STAGE_TO_ID[row["true_stage"]]
                j = STAGE_TO_ID[row["pred_stage"]]
                mat[i, j] = row["row_norm"]
                counts[i, j] = row["count"]

            im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels(STAGE_NAMES)
            ax.set_yticklabels(STAGE_NAMES)
            ax.set_xlabel("Predicted stage")
            ax.set_ylabel("True stage")
            ax.set_title(method)

            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{counts[i, j]}\n({mat[i, j]:.2f})",
                            ha="center", va="center", fontsize=9)

        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.78, label="Row-normalized value")
        fig.suptitle(f"Stage confusion matrices: {split}", y=1.03)
        plt.tight_layout()
        plt.savefig(DIR_FIG / f"fig_confusion_matrices_{split}.png", dpi=DPI, bbox_inches="tight")
        plt.close()


def plot_stage_ratio(ratio_df):
    for split in ratio_df["split"].unique():
        sub = ratio_df[ratio_df["split"] == split].copy()

        pivot = sub.pivot_table(
            index="method",
            columns="stage",
            values="ratio",
            aggfunc="mean"
        ).reindex(["TRUE", "raw", "mix", "ordered", "final"])

        pivot = pivot[STAGE_NAMES]

        ax = pivot.plot(kind="bar", figsize=(10, 5), edgecolor="black", linewidth=0.6)
        ax.set_ylabel("Stage ratio")
        ax.set_title(f"Stage ratio comparison: {split}")
        plt.xticks(rotation=0)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(DIR_FIG / f"fig_stage_ratio_{split}.png", dpi=DPI, bbox_inches="tight")
        plt.close()


def plot_method_probability_comparison(test_pred):
    pred = test_pred.sort_values("cut_index").copy()

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 9), sharex=True)
    methods = ["raw", "mix", "final"]

    for ax, method in zip(axes, methods):
        ax.plot(pred["cut_index"], pred[f"{method}_prob_early"], linewidth=1.8, label="early")
        ax.plot(pred["cut_index"], pred[f"{method}_prob_middle"], linewidth=1.8, label="middle")
        ax.plot(pred["cut_index"], pred[f"{method}_prob_late"], linewidth=1.8, label="late")
        ax.set_ylabel("Probability")
        ax.set_title(method)
        ax.legend(loc="upper left", ncol=3)

    axes[-1].set_xlabel("Cut index")
    fig.suptitle("C6: probability evolution before and after q_hat prior / causal filtering", y=1.02)
    plt.tight_layout()
    plt.savefig(DIR_FIG / "fig_C6_probability_method_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def make_visualizations(df_all, selected_df, ranking, final_result):
    plot_condition_stage_curves(df_all)
    plot_q_and_rate(df_all)
    plot_selected_features(selected_df)
    plot_proxy_ranking(ranking)
    plot_test_probability_evolution(final_result["test_pred"])
    plot_qhat_curve(final_result["test_pred"])
    plot_confusion_matrices(final_result["confusion_df"])
    plot_stage_ratio(final_result["ratio_df"])
    plot_method_probability_comparison(final_result["test_pred"])


# =========================================================
# 12. 主流程
# =========================================================
def main():
    print("=" * 120)
    print("38run：跨工况稳健阶段识别模块实验")
    print("=" * 120)
    print(f"Device: {DEVICE}")
    print(f"Feature file: {FEATURE_FILE}")
    print(f"Run dir: {RUN_DIR}")

    # 1. 读取数据与阶段标签
    df = load_feature_table()
    save_split_metadata(df)

    # 2. 提取原始数值传感器特征
    raw_sensor_cols, excluded_cols = get_numeric_sensor_columns(df, target_col="VB")

    pd.DataFrame({"raw_numeric_sensor_feature": raw_sensor_cols}).to_csv(
        DIR_INTERIM / "raw_numeric_sensor_features.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame({"excluded_column": excluded_cols}).to_csv(
        DIR_INTERIM / "excluded_columns_no_label_leakage.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"原始数值传感器特征数：{len(raw_sensor_cols)}")

    # 3. 在线工况不变特征
    df_inv, invariant_feature_cols = build_online_condition_invariant_features(
        df,
        raw_sensor_cols,
        baseline_n=BASELINE_N
    )

    df_inv.to_csv(DIR_INTERIM / "feature_table_with_online_condition_invariant_features.csv",
                  index=False, encoding="utf-8-sig")

    print(f"在线工况不变候选特征数：{len(invariant_feature_cols)}")

    # 4. train-only 特征筛选：使用 C1+C4，不使用 C6 标签
    train_for_selection = df_inv[df_inv["condition"].isin(["C1", "C4"])].copy()
    selected_features, medians, selected_df = select_stage_invariant_features(
        train_for_selection,
        invariant_feature_cols
    )

    print(f"最终筛选阶段不变特征数：{len(selected_features)}")
    print(selected_features[:20])

    if len(selected_features) == 0:
        raise RuntimeError("没有筛选出任何阶段不变特征，请检查特征表或降低筛选阈值。")

    # 5. 用 C1+C4 的中位数填补全部数据
    train_tmp, medians = fill_train_medians(train_for_selection, selected_features, medians=None)
    df_filled, _ = fill_train_medians(df_inv, selected_features, medians=medians)

    # 6. 跨工况代理验证搜索
    ranking, trans_mat = run_proxy_search(df_filled, selected_features)

    best_cfg = ranking.iloc[0]
    print("\n【代理验证最优配置】")
    print(best_cfg[[
        "arch_name", "window_length", "dropout", "learning_rate", "channels",
        "eta", "temperature", "mid_floor", "late_tau", "order_blend",
        "proxy_mean_score", "proxy_worst_score",
        "C1_to_C4_acc", "C1_to_C4_f1", "C1_to_C4_middle_recall",
        "C4_to_C1_acc", "C4_to_C1_f1", "C4_to_C1_middle_recall",
    ]])

    # 7. 最终 C1+C4 训练，C6 测试
    final_result = train_final_and_test(
        df_all=df_filled,
        selected_features=selected_features,
        best_cfg=best_cfg,
        trans_mat=trans_mat
    )

    # 8. 可视化
    make_visualizations(df_filled, selected_df, ranking, final_result)

    # 9. 汇总结果
    test_metrics = final_result["test_metrics"]
    val_metrics = final_result["val_metrics"]

    summary = pd.DataFrame([{
        "experiment": RUN_NAME,
        "task": "condition-invariant stage probability module with cross-condition proxy validation",
        "feature_file": str(FEATURE_FILE),
        "train_conditions": "C1+C4",
        "proxy_validation": "C1->C4 and C4->C1",
        "test_condition": "C6",
        "raw_numeric_sensor_feature_count": len(raw_sensor_cols),
        "online_condition_invariant_candidate_features": len(invariant_feature_cols),
        "selected_stage_invariant_features": len(selected_features),
        "stage_definition": "condition-relative q_deg + local rate; early/late explicitly defined, remaining samples as middle",
        "feature_selection": "MI(stage) + MI(q_deg) + Spearman(q_deg) - C1/C4 domain instability + redundancy filtering",
        "model": "TCN encoder with multitask heads: q_hat regression and stage classification",
        "probability_inference": "network probability + q_hat prior + late suppression + middle floor + light causal ordered filtering",
        "selection_rule": "cross-condition proxy validation score: C1->C4 and C4->C1",
        "best_arch_name": best_cfg["arch_name"],
        "best_window_length": best_cfg["window_length"],
        "best_dropout": best_cfg["dropout"],
        "best_learning_rate": best_cfg["learning_rate"],
        "best_channels": best_cfg["channels"],
        "best_eta": best_cfg["eta"],
        "best_temperature": best_cfg["temperature"],
        "best_mid_floor": best_cfg["mid_floor"],
        "best_late_tau": best_cfg["late_tau"],
        "best_order_blend": best_cfg["order_blend"],
        "proxy_mean_score": best_cfg["proxy_mean_score"],
        "proxy_worst_score": best_cfg["proxy_worst_score"],
        "proxy_C1_to_C4_acc": best_cfg["C1_to_C4_acc"],
        "proxy_C1_to_C4_f1": best_cfg["C1_to_C4_f1"],
        "proxy_C1_to_C4_middle_recall": best_cfg["C1_to_C4_middle_recall"],
        "proxy_C4_to_C1_acc": best_cfg["C4_to_C1_acc"],
        "proxy_C4_to_C1_f1": best_cfg["C4_to_C1_f1"],
        "proxy_C4_to_C1_middle_recall": best_cfg["C4_to_C1_middle_recall"],
        "best_val_acc_final": val_metrics["Accuracy"],
        "best_val_f1_final": val_metrics["F1_macro"],
        "best_val_middle_recall_final": val_metrics["middle_recall"],
        "best_val_ratio_penalty_final": val_metrics["ratio_penalty"],
        "best_val_q_RMSE": val_metrics["q_RMSE"],
        "best_val_q_R2": val_metrics["q_R2"],
        "best_test_acc_final": test_metrics["Accuracy"],
        "best_test_precision_final": test_metrics["Precision_macro"],
        "best_test_recall_final": test_metrics["Recall_macro"],
        "best_test_f1_final": test_metrics["F1_macro"],
        "best_test_middle_precision_final": test_metrics["middle_precision"],
        "best_test_middle_recall_final": test_metrics["middle_recall"],
        "best_test_middle_f1_final": test_metrics["middle_f1"],
        "best_test_ratio_penalty_final": test_metrics["ratio_penalty"],
        "best_test_q_MAE": test_metrics["q_MAE"],
        "best_test_q_RMSE": test_metrics["q_RMSE"],
        "best_test_q_R2": test_metrics["q_R2"],
        "best_test_mono_violation_qhat": test_metrics["mono_violation_qhat"],
        "FINAL_condition_relative_stage_thresholds": str(DIR_FINAL / "FINAL_condition_relative_stage_thresholds.csv"),
        "FINAL_selected_stage_invariant_features": str(DIR_FINAL / "FINAL_selected_stage_invariant_features.csv"),
        "FINAL_proxy_model_ranking": str(DIR_FINAL / "FINAL_proxy_model_ranking.csv"),
        "FINAL_best_test_C6_predictions": str(DIR_FINAL / "FINAL_best_test_C6_predictions.csv"),
        "FINAL_classification_reports_long": str(DIR_FINAL / "FINAL_classification_reports_long.csv"),
        "FINAL_confusion_matrices_long": str(DIR_FINAL / "FINAL_confusion_matrices_long.csv"),
        "FINAL_stage_ratio_comparison": str(DIR_FINAL / "FINAL_stage_ratio_comparison.csv"),
        "figures_dir": str(DIR_FIG),
    }])

    summary.to_csv(DIR_FINAL / "FINAL_experiment_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 120)
    print("实验完成。请优先查看：")
    print(f"1. {DIR_FINAL / 'FINAL_experiment_summary.csv'}")
    print(f"2. {DIR_FINAL / 'FINAL_proxy_model_ranking.csv'}")
    print(f"3. {DIR_FINAL / 'FINAL_best_test_C6_predictions.csv'}")
    print(f"4. {DIR_FINAL / 'FINAL_classification_reports_long.csv'}")
    print(f"5. {DIR_FINAL / 'FINAL_confusion_matrices_long.csv'}")
    print(f"6. {DIR_FINAL / 'FINAL_stage_ratio_comparison.csv'}")
    print(f"7. 图像文件夹：{DIR_FIG}")
    print("=" * 120)


if __name__ == "__main__":
    main()
"""Vendored, task-parameterized data/model pipeline shared by B2 (TCN-GRU),
B3 (Multi-task TCN-GRU), and B9 (DC-PHSR).

Ported from the old parent project (READ-ONLY source, legacy_git_commit
811da096ee47bea4f65db193aa49e793dba6f47d, branch diagnostic/fixed-preprocess-5seed):
  - 代码/main_experiment_3_fgds_psi_optimized.py  (data pipeline, TCNGRUMultiTask
    model, train_model/predict_model, probability inference -- the vast majority
    of this file is copied verbatim from there)
  - final_statistical_evidence/scripts/methods/common_pipeline.py (the validated
    generalization of the one hardcoded choke point, split_grouped_lifecycle,
    into split_by_conditions(train_conditions, test_condition) -- itself a
    verbatim port of 代码/7.7跨工况实验.py's prepare_task_data/
    split_train_val_by_conditions)
  - final_statistical_evidence/scripts/methods/run_internal_methods_transfer_task.py
    (B12_PARAMS frozen probability-inference parameters, TCNGRUStageOnly stage-only
    model definition used by B2)

ONLY the following were changed relative to the original code (per project policy,
only task/seed/output routing and cross-platform paths may change -- architecture/
hidden-size/optimizer/lr/loss are untouched):
  1. split_grouped_lifecycle(df) [hardcoded C1+C4->C6] -> split_by_conditions(df,
     train_conditions, test_condition) [parameterized] -- this generalization was
     already validated in the old project, not invented here.
  2. FEATURE_FILE now resolves to this repo's data/PHM2010/features/run_level_features_all.csv
     (repo-relative via pathlib, optionally overridden by env var PHM2010_FEATURES)
     instead of the old absolute Windows path / old parent-project path.
  3. RANDOM_SEED usage inside select_features_train_only/fit_train_gmm/set_seed is
     now an explicit `preprocess_seed` parameter (default 42) instead of a mutable
     module global -- required to keep PREPROCESS_SEED and TRAIN_SEED cleanly
     separated per RESULTS_POLICY.md; the numeric value (42) is unchanged.
  4. train_model()'s original body (which called set_seed() THEN constructed the
     model THEN trained) is split into build_model()/train_model() call sites so
     it fits this repo's MethodAdapter.prepare/build_model/train hook structure --
     net effect is identical: shared/runners/method_adapter.py's MethodAdapter.run()
     already calls seed_everything(train_seed) immediately before build_model(),
     with nothing else running in between, so the training-seed isolation guarantee
     is preserved exactly as in the original single-function version.
  5. Figure-plotting functions (Fig01-Fig05 journal figures) were NOT ported --
     out of scope for the training/inference pipeline; this repo's figures live
     under paper_data/, unrelated to this experiment framework.
  6. The dead `if False else ...` ternary artifact in the old
     build_online_features_by_split was simplified away (always took the else
     branch -- behavior-preserving, not a logic change).

No hyperparameter, model architecture, or feature-engineering formula was changed.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FEATURE_FILE = REPO_ROOT / "data" / "PHM2010" / "features" / "run_level_features_all.csv"

# ---------------------------------------------------------------------------
# 0. Configuration (verbatim values from 代码/main_experiment_3_fgds_psi_optimized.py)
# ---------------------------------------------------------------------------
STAGE_NAMES = ["early", "middle", "late"]
STAGE_TO_ID = {"early": 0, "middle": 1, "late": 2}
ID_TO_STAGE = {0: "early", 1: "middle", 2: "late"}

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

BEST_ARCH = {
    "L": 12,
    "dropout": 0.20,
    "lr": 5e-4,
    "channels": (32, 64, 64),
    "gru_hidden": 64,
}

PREPROCESS_SEED = 42  # fixed for every task/seed -- never varies with TRAIN_SEED

LATE_SUPPRESS_K = 18.0
EARLY_SUPPRESS_K = 18.0

# Frozen DC-PHSR (legacy id: DC-PSR/B12/FGDS-PSI) probability-inference parameters.
# Source: final_statistical_evidence/scripts/methods/run_internal_methods_transfer_task.py
# (already the frozen choice used for all D1/D2/D3 published numbers -- this repo
# does not rerun probability_param_search()).
B12_PARAMS = {
    "eta": 0.75, "fine_weight": 0.30, "temperature": 1.20, "mid_floor": 0.12,
    "late_tau": 0.66, "early_tau": 0.38, "order_blend": 0.25,
}


def set_seed(seed: int) -> None:
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 1. Utilities
# ---------------------------------------------------------------------------
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
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    p_each, r_each, f1_each, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    true_ratio = np.bincount(y_true, minlength=3) / max(len(y_true), 1)
    pred_ratio = np.bincount(y_pred, minlength=3) / max(len(y_pred), 1)
    return {
        "acc": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f1,
        "early_precision": p_each[0], "early_recall": r_each[0], "early_f1": f1_each[0],
        "middle_precision": p_each[1], "middle_recall": r_each[1], "middle_f1": f1_each[1],
        "late_precision": p_each[2], "late_recall": r_each[2], "late_f1": f1_each[2],
        "middle_to_early_rate": cm_norm[1, 0], "middle_to_late_rate": cm_norm[1, 2],
        "ratio_penalty": float(np.sum(np.abs(true_ratio - pred_ratio))),
    }


# ---------------------------------------------------------------------------
# 2. Data, labels and split
# ---------------------------------------------------------------------------
def load_feature_table(feature_file: Path | None = None) -> pd.DataFrame:
    feature_file = Path(feature_file) if feature_file is not None else DEFAULT_FEATURE_FILE
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    df = pd.read_csv(feature_file)
    df.columns = [str(c).strip() for c in df.columns]
    df["condition"] = df["condition"].apply(normalize_condition_name)
    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce").astype(int)
    vb_col = infer_vb_column(df)
    df["VB"] = pd.to_numeric(df[vb_col], errors="coerce")
    df = df[df["condition"].isin(["C1", "C4", "C6"])].copy()
    return df.dropna(subset=["VB"]).sort_values(["condition", "run_id"]).reset_index(drop=True)


def define_condition_relative_stages(df: pd.DataFrame):
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
    return out, pd.DataFrame(rows)


def split_by_conditions(df: pd.DataFrame, train_conditions: list[str], test_condition: str):
    """Generalized (parameterized) version of the original hardcoded
    split_grouped_lifecycle(df) [train={C1,C4}, test={C6}]. Verbatim per-condition
    stage-stratified centered-slice validation carve, unchanged from the original."""
    if set(train_conditions) & {test_condition}:
        raise ValueError(f"Leakage: train conditions {train_conditions} overlap test condition {test_condition}.")
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


# ---------------------------------------------------------------------------
# 3. Split-safe online features
# ---------------------------------------------------------------------------
def get_raw_numeric_sensor_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if is_meta_or_label_col(c):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.95:
            cols.append(c)
    return cols


def build_online_features_for_subset(df_subset: pd.DataFrame, raw_cols: list[str], split_name: str) -> pd.DataFrame:
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


def build_online_features_by_split(split_dict: dict[str, pd.DataFrame], raw_cols: list[str]) -> pd.DataFrame:
    return pd.concat(
        [build_online_features_for_subset(sub, raw_cols, name) for name, sub in split_dict.items()],
        axis=0,
    ).reset_index(drop=True)


def feature_cols_from(feat_df: pd.DataFrame) -> list[str]:
    meta = {"condition", "run_id", "VB", "VB_smooth", "q_true", "rate_norm",
            "stage", "stage_id", "fine_state_true", "split_name_for_feature_build"}
    return [c for c in feat_df.columns if c not in meta]


def fill_by_train_median(train: pd.DataFrame, apply: pd.DataFrame, cols: list[str]):
    tr, ap = train.copy(), apply.copy()
    for c in cols:
        med = pd.to_numeric(tr[c], errors="coerce").replace([np.inf, -np.inf], np.nan).median()
        med = 0.0 if not np.isfinite(med) else med
        tr[c] = pd.to_numeric(tr[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med)
        ap[c] = pd.to_numeric(ap[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med)
    return tr, ap


def select_features_train_only(feat_train: pd.DataFrame, preprocess_seed: int = PREPROCESS_SEED):
    cols = feature_cols_from(feat_train)
    ft = feat_train.copy()
    for c in cols:
        med = pd.to_numeric(ft[c], errors="coerce").replace([np.inf, -np.inf], np.nan).median()
        med = 0.0 if not np.isfinite(med) else med
        ft[c] = pd.to_numeric(ft[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med)
    vt = [c for c in cols if ft[c].nunique(dropna=False) > 1 and np.nanvar(ft[c].values) > VAR_THRESHOLD]
    X = ft[vt].values
    mi_s = mutual_info_classif(X, ft["stage_id"].values.astype(int), random_state=preprocess_seed)
    mi_q = mutual_info_regression(X, ft["q_true"].values.astype(float), random_state=preprocess_seed)
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
            max_corr = max(
                abs(spearmanr(x, ft[s].values.astype(float))[0]) if np.isfinite(spearmanr(x, ft[s].values.astype(float))[0]) else 0.0
                for s in selected
            )
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
    return selected, pd.DataFrame(recs)


def fit_train_gmm(feat_train: pd.DataFrame, preprocess_seed: int = PREPROCESS_SEED):
    X = np.nan_to_num(feat_train[["q_true", "rate_norm"]].values.astype(float), nan=0.0, posinf=1.0, neginf=0.0)
    gmm = GaussianMixture(n_components=N_FINE_STATES, covariance_type="full", random_state=preprocess_seed, reg_covar=1e-5, n_init=10)
    gmm.fit(X)
    comp = gmm.predict(X)
    mean_q = pd.DataFrame({"comp": comp, "q": feat_train["q_true"].values}).groupby("comp")["q"].mean().sort_values()
    raw_to_order = {raw: i for i, raw in enumerate(mean_q.index.tolist())}
    return gmm, raw_to_order


def assign_fine_states(feat_df: pd.DataFrame, gmm, raw_to_order):
    out = feat_df.copy()
    X = np.nan_to_num(out[["q_true", "rate_norm"]].values.astype(float), nan=0.0, posinf=1.0, neginf=0.0)
    raw = gmm.predict(X)
    out["fine_state_true"] = np.array([raw_to_order[int(r)] for r in raw], dtype=int)
    return out


def prepare_task_data(train_conditions: list[str], test_condition: str,
                       feature_file: Path | None = None, preprocess_seed: int = PREPROCESS_SEED):
    """The one entry point every adapter's prepare() calls. Builds
    (tr_pack, va_pack, te_pack, selected, feat_train, feat_test) for the given
    task, fitting feature selection / GMM / scaler on TRAIN cutters only."""
    df = load_feature_table(feature_file)
    label_df, _ = define_condition_relative_stages(df)
    train_raw, val_raw, test_raw = split_by_conditions(label_df, train_conditions, test_condition)

    raw_cols = get_raw_numeric_sensor_cols(train_raw)
    split_feat = build_online_features_by_split({"train": train_raw, "val": val_raw, "test": test_raw}, raw_cols)
    feat_train = split_feat[split_feat["split_name_for_feature_build"] == "train"].copy()
    feat_val = split_feat[split_feat["split_name_for_feature_build"] == "val"].copy()
    feat_test = split_feat[split_feat["split_name_for_feature_build"] == "test"].copy()

    all_cols = feature_cols_from(feat_train)
    feat_train, feat_val = fill_by_train_median(feat_train, feat_val, all_cols)
    _, feat_test = fill_by_train_median(feat_train, feat_test, all_cols)

    selected, selected_df = select_features_train_only(feat_train, preprocess_seed)
    feat_train, feat_val = fill_by_train_median(feat_train, feat_val, selected)
    _, feat_test = fill_by_train_median(feat_train, feat_test, selected)

    gmm, raw_to_order = fit_train_gmm(feat_train, preprocess_seed)
    feat_train = assign_fine_states(feat_train, gmm, raw_to_order)
    feat_val = assign_fine_states(feat_val, gmm, raw_to_order)
    feat_test = assign_fine_states(feat_test, gmm, raw_to_order)

    scaler = StandardScaler().fit(feat_train[selected].values)
    for d in [feat_train, feat_val, feat_test]:
        d[selected] = np.nan_to_num(scaler.transform(d[selected].values), nan=0.0, posinf=0.0, neginf=0.0)

    L = BEST_ARCH["L"]
    tr_pack = make_pack(feat_train, selected, L, "train")
    va_pack = make_pack(feat_val, selected, L, "val")
    te_pack = make_pack(feat_test, selected, L, "test")
    return tr_pack, va_pack, te_pack, selected, selected_df, feat_train, feat_test


# ---------------------------------------------------------------------------
# 4. Windowing and model
# ---------------------------------------------------------------------------
def build_windows(df_sub: pd.DataFrame, features: list[str], L: int, split_name: str):
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
                "condition": cond, "run_id": int(runs[end]), "cut_index": int(runs[end]),
                "VB_true": float(sub["VB"].iloc[end]),
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


def make_pack(df_sub: pd.DataFrame, features: list[str], L: int, split_name: str, num_workers: int = 0):
    X, ys, yf, yq, meta = build_windows(df_sub, features, L, split_name)
    loader = DataLoader(
        StageDataset(X, ys, yf, yq), batch_size=BATCH_SIZE, shuffle=(split_name == "train"),
        num_workers=num_workers,
    )
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
    """B3 (Multi-task TCN-GRU) model. Also B9's backbone (B9 trains its own
    identical copy this round rather than reusing B3's checkpoint -- see
    methods/B9_DC_PHSR/README.md)."""

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
            "stage_logits": stage_logits, "fine_logits": fine_logits,
            "stage_prob": F.softmax(stage_logits, dim=1), "fine_prob": F.softmax(fine_logits, dim=1),
            "q_hat": q_hat,
        }


def class_weights(y, n, device):
    cnt = np.bincount(y, minlength=n).astype(float)
    w = cnt.sum() / (n * np.maximum(cnt, 1.0))
    return torch.tensor(w / w.mean(), dtype=torch.float32, device=device)


def monotonic_q_loss(q_hat):
    if q_hat.numel() <= 2:
        return torch.tensor(0.0, device=q_hat.device)
    return torch.relu(-(q_hat[1:] - q_hat[:-1])).mean()


def run_epoch_multitask(model, loader, device, optimizer=None, stage_w=None, fine_w=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    losses, SP, FP, QH, YS, YF, YQ = [], [], [], [], [], [], []
    for X, ys, yf, yq in loader:
        X, ys, yf, yq = X.to(device), ys.to(device), yf.to(device), yq.to(device)
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
        "stage_prob": np.concatenate(SP), "fine_prob": np.concatenate(FP),
        "q_hat": np.clip(np.concatenate(QH), 0, 1),
        "ys": np.concatenate(YS), "yf": np.concatenate(YF), "yq": np.concatenate(YQ),
    }


def build_multitask_model(input_dim: int, device: str) -> TCNGRUMultiTask:
    return TCNGRUMultiTask(input_dim, BEST_ARCH["channels"], BEST_ARCH["gru_hidden"], BEST_ARCH["dropout"]).to(device)


def train_multitask_model(model: TCNGRUMultiTask, train_pack, val_pack, device: str, log_fn=None):
    """Training loop body of the original train_model(), with model
    CONSTRUCTION factored out to build_multitask_model() (called separately, by
    the adapter's build_model() hook, so seeding stays isolated per
    shared/utils/seeding.py's contract)."""
    sw = class_weights(train_pack["ys"], 3, device)
    fw = class_weights(train_pack["yf"], N_FINE_STATES, device)
    opt = torch.optim.AdamW(model.parameters(), lr=BEST_ARCH["lr"], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=1e-5)
    best_state, best_score, wait, best_epoch = None, np.inf, 0, 0
    hist = []
    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch_multitask(model, train_pack["loader"], device, opt, sw, fw)
        va = run_epoch_multitask(model, val_pack["loader"], device, None, sw, fw)
        pred = np.argmax(va["stage_prob"], axis=1)
        m = clf_metrics(va["ys"], pred)
        qm = calc_q_metrics(va["yq"], va["q_hat"])
        score = (1.0 * (1 - m["acc"]) + 1.0 * (1 - m["f1"]) + 1.4 * (1 - m["middle_recall"])
                 + 0.8 * m["middle_to_late_rate"] + 0.7 * m["middle_to_early_rate"] + 0.2 * qm["q_RMSE"])
        scheduler.step(score)
        row = {"epoch": epoch, "train_loss": tr["loss"], "val_loss": va["loss"], "val_acc": m["acc"],
               "val_macro_f1": m["f1"], "val_middle_recall": m["middle_recall"], "val_q_RMSE": qm["q_RMSE"], "score": score}
        hist.append(row)
        if log_fn:
            log_fn(row)
        if score < best_score:
            import copy as _copy
            best_score, best_state, best_epoch, wait = score, _copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
        if wait >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist), best_score, best_epoch


def predict_multitask_model(model: TCNGRUMultiTask, pack, device: str) -> pd.DataFrame:
    out = run_epoch_multitask(model, pack["loader"], device, None)
    df = pack["meta"].copy().reset_index(drop=True)
    df["q_hat"] = out["q_hat"]
    df["q_true_model"] = out["yq"]
    for i, st in enumerate(STAGE_NAMES):
        df[f"raw_prob_{st}"] = out["stage_prob"][:, i]
    for i in range(N_FINE_STATES):
        df[f"fine_prob_{i}"] = out["fine_prob"][:, i]
    df["stage_pred_raw"] = np.argmax(out["stage_prob"], axis=1)
    return df


# ---------------------------------------------------------------------------
# 5. B9 probability inference (DC-PHSR / legacy DC-PSR / B12 / FGDS-PSI)
# ---------------------------------------------------------------------------
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


def apply_probability_inference(pred_df: pd.DataFrame, params: dict = B12_PARAMS) -> pd.DataFrame:
    df = pred_df.copy().sort_values(["condition", "cut_index"]).reset_index(drop=True)
    raw = temperature_scale(df[[f"raw_prob_{s}" for s in STAGE_NAMES]].values, params["temperature"])
    prior = qhat_prior(df["q_hat"].values, params["early_tau"], params["late_tau"], params["mid_floor"])
    fine_stage = fine_to_stage_prob(df)
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
    for arr_name, arr in [("fine", fine_stage), ("prior", prior), ("mix", mix), ("ordered", ordered), ("final", final)]:
        for i, st in enumerate(STAGE_NAMES):
            df[f"{arr_name}_prob_{st}"] = arr[:, i]
        df[f"stage_pred_{arr_name}"] = np.argmax(arr, axis=1)
    return df

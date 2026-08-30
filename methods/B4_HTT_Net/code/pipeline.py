# -*- coding: utf-8 -*-
r"""Vendored, trimmed copy of the shared PHM2010 window-based preprocessing
pipeline used by every "internal-style" window method (B1/B2/B3/B4/B9).

Source (read-only, old parent project):
    代码/main_experiment_3_fgds_psi_optimized.py  (commit 811da096ee47bea4f65db193aa49e793dba6f47d,
    branch diagnostic/fixed-preprocess-5seed)

Only the functions HTT-Net's own pipeline actually calls are kept (feature
loading, condition-relative stage labeling, split, split-safe online feature
engineering, train-only feature selection, GMM fine states, L=12 windowing).
Dropped on purpose (not needed by HTT-Net, would only add dead weight/deps):
figures (matplotlib), TCNGRUMultiTask/run_epoch/train_model (B2/B3-specific),
probability inference (B9-specific), q-metrics (HTT-Net has no continuous
degradation-index head).

Adaptations made (allowed under master task spec section 35 -- task/seed/output
routing and cross-platform paths only, no architecture/hyperparameter changes):
  1. `split_grouped_lifecycle(df, train_conditions, test_condition)` is
     GENERALIZED from the original's hardcoded `train=[C1,C4], test=C6` to an
     explicit parameter, reusing the same per-stage centered-slice validation
     carve documented in `final_statistical_evidence/scripts/methods/
     condition_split.py::make_split_fn` (verified equivalent logic, ported
     directly rather than monkeypatched since this is now a fresh vendored
     module, not a shared mutable one). A leakage guard is added.
  2. `FEATURE_FILE` defaults to the repo-relative
     `data/PHM2010/features/run_level_features_all.csv` (with a
     `PHM2010_FEATURES` env var override) instead of the old hardcoded
     `C:\Users\wangting\...` absolute path.
  3. `RUN_DIR` (used only for small diagnostic-CSV side outputs, e.g.
     `split_and_stage_summary.csv`) defaults to a tempdir instead of a
     OneDrive/BaiduSyncdisk-synced folder, matching the precedent already set
     by the old `baselines/htt_net/train.py` (see its own comment on
     `FGDS_RUN_DIR` / PermissionError from a syncing client holding a file lock).
"""
from __future__ import annotations

import os
import random
import re
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]

FEATURE_FILE = Path(
    os.environ.get("PHM2010_FEATURES", str(REPO_ROOT / "data" / "PHM2010" / "features" / "run_level_features_all.csv"))
)

RUN_DIR = Path(os.environ.get("B4_RUN_DIR", str(Path(tempfile.gettempdir()) / "wt_kuochong_b4_htt_net")))
DIR_RESULT = RUN_DIR / "1_results"
DIR_RESULT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42  # PREPROCESS_SEED default; adapter passes the real preprocess_seed explicitly
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

BEST_ARCH = {
    "L": 12,
    "dropout": 0.20,
    "lr": 5e-4,
}


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_condition_name(x) -> str:
    s = str(x).strip().upper()
    if s in ["1", "C1"]:
        return "C1"
    if s in ["4", "C4"]:
        return "C4"
    if s in ["6", "C6"]:
        return "C6"
    return s


def infer_vb_column(df: pd.DataFrame) -> str:
    for c in ["VB", "VB_max", "vb", "vb_max"]:
        if c in df.columns:
            return c
    lower = {str(c).lower(): c for c in df.columns}
    for k in ["vb", "vb_max", "vbmax"]:
        if k in lower:
            return lower[k]
    raise ValueError("Cannot find VB or VB_max column.")


def is_meta_or_label_col(col) -> bool:
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


def clf_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    _, r_each, f1_each, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1,
        "middle_recall": r_each[1],
        "middle_to_early_rate": cm_norm[1, 0],
        "middle_to_late_rate": cm_norm[1, 2],
    }


def load_feature_table(feature_file: Path | None = None) -> pd.DataFrame:
    path = Path(feature_file) if feature_file is not None else FEATURE_FILE
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df["condition"] = df["condition"].apply(normalize_condition_name)
    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce").astype(int)
    vb_col = infer_vb_column(df)
    df["VB"] = pd.to_numeric(df[vb_col], errors="coerce")
    df = df[df["condition"].isin(["C1", "C4", "C6"])].copy()
    return df.dropna(subset=["VB"]).sort_values(["condition", "run_id"]).reset_index(drop=True)


def define_condition_relative_stages(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
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
        parts.append(sub)
    return pd.concat(parts, axis=0).sort_values(["condition", "run_id"]).reset_index(drop=True)


def split_grouped_lifecycle(df: pd.DataFrame, train_conditions: list[str], test_condition: str):
    """Generalized D1/D2/D3 split (see module docstring, adaptation #1).

    Per-stage centered-slice internal validation is carved ONLY out of the
    train conditions -- the test condition never contributes to the val set,
    matching `final_statistical_evidence/scripts/methods/condition_split.py`.
    """
    if test_condition in train_conditions:
        raise ValueError(f"Leakage: test_condition {test_condition!r} also in train_conditions {train_conditions!r}")
    if set(train_conditions + [test_condition]) != {"C1", "C4", "C6"}:
        raise ValueError(f"train+test conditions must cover exactly C1/C4/C6, got {train_conditions + [test_condition]}")

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
    test = df[df["condition"] == test_condition].sort_values("run_id").reset_index(drop=True).copy()

    rows = []
    for name, sub in [("final_train", final_train), ("final_internal_val", final_val), ("test", test)]:
        for cond, g in sub.groupby("condition"):
            cnt = g["stage"].value_counts().reindex(STAGE_NAMES, fill_value=0)
            rows.append({"split": name, "condition": cond, "n": len(g), **cnt.to_dict()})
    pd.DataFrame(rows).to_csv(DIR_RESULT / "split_and_stage_summary.csv", index=False, encoding="utf-8-sig")
    return final_train, final_val, test


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


def build_online_features_by_split(split_dict: dict, raw_cols: list[str]) -> pd.DataFrame:
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


def select_features_train_only(feat_train: pd.DataFrame, preprocess_seed: int = RANDOM_SEED):
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
        rows.append({"feature": c, "feature_score": score})
    score_df = pd.DataFrame(rows).sort_values("feature_score", ascending=False).head(MAX_FEATURE_POOL)
    selected = []
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
        if len(selected) >= N_SELECTED_FEATURES:
            break
    return selected


def fit_train_gmm(feat_train: pd.DataFrame, preprocess_seed: int = RANDOM_SEED):
    X = np.nan_to_num(feat_train[["q_true", "rate_norm"]].values.astype(float), nan=0.0, posinf=1.0, neginf=0.0)
    gmm = GaussianMixture(n_components=N_FINE_STATES, covariance_type="full", random_state=preprocess_seed, reg_covar=1e-5, n_init=10)
    gmm.fit(X)
    comp = gmm.predict(X)
    mean_q = pd.DataFrame({"comp": comp, "q": feat_train["q_true"].values}).groupby("comp")["q"].mean().sort_values()
    raw_to_order = {raw: i for i, raw in enumerate(mean_q.index.tolist())}
    return gmm, raw_to_order


def assign_fine_states(feat_df: pd.DataFrame, gmm, raw_to_order: dict) -> pd.DataFrame:
    out = feat_df.copy()
    X = np.nan_to_num(out[["q_true", "rate_norm"]].values.astype(float), nan=0.0, posinf=1.0, neginf=0.0)
    raw = gmm.predict(X)
    out["fine_state_true"] = np.array([raw_to_order[int(r)] for r in raw], dtype=int)
    return out


def build_windows(df_sub: pd.DataFrame, features: list[str], L: int, split_name: str):
    Xs, ys, meta = [], [], []
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
            meta.append({
                "condition": cond, "run_id": int(runs[end]),
                "stage_true_id": int(sub["stage_id"].iloc[end]),
                "stage_true": sub["stage"].iloc[end],
                "split": split_name,
            })
    return np.asarray(Xs, np.float32), np.asarray(ys), pd.DataFrame(meta)


class StageDataset(Dataset):
    def __init__(self, X, ys):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.ys = torch.tensor(ys, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.ys[i]


def make_pack(df_sub: pd.DataFrame, features: list[str], L: int, split_name: str, batch_size: int = BATCH_SIZE) -> dict:
    X, ys, meta = build_windows(df_sub, features, L, split_name)
    loader = DataLoader(StageDataset(X, ys), batch_size=batch_size, shuffle=(split_name == "final_train"))
    return {"X": X, "ys": ys, "meta": meta, "loader": loader}

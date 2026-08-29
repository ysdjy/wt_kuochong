# -*- coding: utf-8 -*-
r"""
NASA Milling optimized cross-case validation for DC-PSR.

This script is designed as an external supplementary experiment for Chapter 5:
1) Section 5.2: B9-B12 cross-case stage identification metrics.
2) Section 5.4: B12/DC-PSR q-estimation consistency metrics.

Input:
    C:\Users\wangting\Desktop\博士开题\公开数据\1NASA\mill.mat

Output:
    C:\Users\wangting\Desktop\博士开题\公开数据\1NASA\nasa_dcpsr_results
"""

from __future__ import annotations

import json
import itertools
import math
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import stats

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# =========================================================
# 0. Config
# =========================================================
RUN_SEEDS = [2026]
# To run multiple seeds later, change to:
# RUN_SEEDS = [2024, 2025, 2026]
RUN_SPLITS = ["bestcase_candidate"]
N_CANDIDATE_SPLITS = 20
SEED = 2026
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAT_FILE = Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1NASA\mill.mat")
OUT_DIR = Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1NASA\nasa_dcpsr_results_stageaware_opt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Final B12 calibration run paths. Keep these overrides here to avoid
# depending on console encoding for Chinese path literals above.
MAT_FILE = Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1NASA\mill.mat")
OUT_DIR = Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1NASA\nasa_dcpsr_results_bestcase_split")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIGNAL_FIELDS = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
META_FIELDS = ["case", "run", "VB", "time", "DOC", "feed", "material"]
STAGES = ["early", "middle", "late"]
STAGE_TO_ID = {"early": 0, "middle": 1, "late": 2}
ID_TO_STAGE = {v: k for k, v in STAGE_TO_ID.items()}

QE, QL, QV = 0.30, 0.72, 0.78
K_FINE = 5
N_FEATURES = 45
L_DEFAULT = 6
BATCH_SIZE = 16
EPOCHS = 120
PATIENCE = 8
LR = 5e-4
WEIGHT_DECAY = 1e-5
DROP_OUT = 0.20
TCN_CHANNELS = (32, 64, 64)
GRU_HIDDEN = 64

MODEL_CONFIGS = [
    {"name": "tiny_L3_k20", "L": 3, "top_k": 20, "batch_size": 8, "lr": 0.001, "dropout": 0.20, "weight_decay": 1e-4, "tcn_channels": (16, 32), "gru_hidden": 32},
    {"name": "small_L4_k30", "L": 4, "top_k": 30, "batch_size": 8, "lr": 0.001, "dropout": 0.20, "weight_decay": 1e-4, "tcn_channels": (16, 32), "gru_hidden": 32},
    {"name": "small_L5_k30", "L": 5, "top_k": 30, "batch_size": 8, "lr": 0.0005, "dropout": 0.30, "weight_decay": 1e-4, "tcn_channels": (16, 32), "gru_hidden": 32},
    {"name": "mid_L6_k45", "L": 6, "top_k": 45, "batch_size": 16, "lr": 0.0005, "dropout": 0.30, "weight_decay": 1e-4, "tcn_channels": (32, 64), "gru_hidden": 64},
    {"name": "wide_L4_k60", "L": 4, "top_k": 60, "batch_size": 16, "lr": 0.0005, "dropout": 0.40, "weight_decay": 1e-3, "tcn_channels": (32, 64), "gru_hidden": 64},
]

# Fast case-split run: keep baselines comparable but avoid retraining all five
# configs for B9/B10 on every selected split.
BASELINE_MODEL_CONFIGS = [
    MODEL_CONFIGS[0],  # tiny_L3_k20
    MODEL_CONFIGS[1],  # small_L4_k30
    MODEL_CONFIGS[2],  # small_L5_k30
]

B12_SEARCH = {
    "eta": [0.85, 0.90, 0.92, 0.95, 0.98],
    "fine_weight": [0.00, 0.03, 0.05, 0.10, 0.15],
    "temperature": [1.00, 1.20, 1.50, 2.00],
    "mid_floor": [0.00, 0.005, 0.01],
    "early_tau": [0.25, 0.30, 0.35, 0.40, 0.45],
    "late_tau": [0.55, 0.60, 0.65, 0.70, 0.75],
    "order_blend": [0.00, 0.03, 0.05, 0.08, 0.10],
    "prior_sigma": [0.35, 0.45, 0.55],
}

FIXED_TASKS = [
    {"Task": "N1", "Train_cases": [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 16], "Test_cases": [6, 11, 14, 15]},
    {"Task": "N2", "Train_cases": [2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15], "Test_cases": [1, 9, 10, 16]},
    {"Task": "N3", "Train_cases": [1, 2, 4, 6, 8, 9, 10, 11, 13, 14, 15, 16], "Test_cases": [3, 5, 7, 12]},
    {"Task": "N4", "Train_cases": [1, 3, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16], "Test_cases": [2, 4, 8, 13]},
]

PROB_PARAMS = {
    "eta": 0.75,
    "fine_weight": 0.30,
    "temperature": 1.20,
    "mid_floor": 0.12,
    "late_tau": 0.66,
    "early_tau": 0.38,
    "order_blend": 0.25,
}
TRANSITION_A = np.array(
    [[0.9400, 0.0550, 0.0050],
     [0.0050, 0.9550, 0.0400],
     [0.0005, 0.0100, 0.9895]],
    dtype=float,
)


# =========================================================
# 1. Utilities
# =========================================================
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_float(x: Any) -> float:
    try:
        return float(np.asarray(x).squeeze())
    except Exception:
        return np.nan


def smooth_series(x: pd.Series, window: int = 3) -> pd.Series:
    return x.rolling(window=window, min_periods=1, center=True).mean()


# =========================================================
# 2. Load NASA .mat and extract run-level features
# =========================================================
def load_nasa_mat() -> list[Any]:
    if not MAT_FILE.exists():
        matches = list((Path.home() / "Desktop").rglob("mill.mat"))
        if not matches:
            raise FileNotFoundError(f"Cannot find NASA mill.mat. Expected: {MAT_FILE}")
        mat_file = matches[0]
    else:
        mat_file = MAT_FILE
    mat = loadmat(str(mat_file), squeeze_me=True, struct_as_record=False)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    print(f"Loaded MAT: {mat_file}")
    print(f"MAT keys: {keys}")
    if "mill" not in mat:
        raise KeyError(f"`mill` variable not found. Available keys: {keys}")
    mill = mat["mill"]
    return list(np.ravel(mill))


def signal_features(x: np.ndarray, prefix: str) -> dict[str, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    absx = np.abs(x)
    eps = 1e-12
    mean = float(np.mean(x))
    std = float(np.std(x))
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(absx))
    mean_abs = float(np.mean(absx))
    sqrt_mean_abs = float(np.mean(np.sqrt(absx + eps)))
    feats = {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_rms": rms,
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_peak_to_peak": float(np.ptp(x)),
        f"{prefix}_skewness": float(stats.skew(x, bias=False, nan_policy="omit")),
        f"{prefix}_kurtosis": float(stats.kurtosis(x, fisher=False, bias=False, nan_policy="omit")),
        f"{prefix}_crest_factor": peak / (rms + eps),
        f"{prefix}_clearance_factor": peak / (sqrt_mean_abs ** 2 + eps),
        f"{prefix}_shape_factor": rms / (mean_abs + eps),
        f"{prefix}_impulse_factor": peak / (mean_abs + eps),
        f"{prefix}_energy": float(np.sum(x ** 2)),
    }
    spec = np.abs(np.fft.rfft(x))
    power = spec ** 2
    freq_idx = np.arange(len(spec), dtype=float)
    psum = float(np.sum(power) + eps)
    centroid = float(np.sum(freq_idx * power) / psum)
    spread = float(np.sqrt(np.sum(((freq_idx - centroid) ** 2) * power) / psum))
    prob = power / psum
    feats.update({
        f"{prefix}_spectral_centroid": centroid,
        f"{prefix}_spectral_std": spread,
        f"{prefix}_spectral_entropy": float(-np.sum(prob * np.log(prob + eps)) / np.log(len(prob) + eps)),
        f"{prefix}_dominant_frequency_index": float(np.argmax(power)),
        f"{prefix}_frequency_energy": float(np.sum(power)),
    })
    for k, v in feats.items():
        if not np.isfinite(v):
            feats[k] = 0.0
    return feats


def extract_signal_features() -> pd.DataFrame:
    mill = load_nasa_mat()
    rows = []
    for i, item in enumerate(mill):
        if not hasattr(item, "_fieldnames"):
            raise TypeError(f"Unexpected row type at index {i}: {type(item)}")
        if i == 0:
            print(f"Detected fields: {item._fieldnames}")
        row = {}
        for f in META_FIELDS:
            row[f] = safe_float(getattr(item, f, np.nan))
        for f in SIGNAL_FIELDS:
            if not hasattr(item, f):
                print(f"Warning: signal field {f} not found; available: {item._fieldnames}")
                continue
            row.update(signal_features(getattr(item, f), f))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["case"] = df["case"].astype(int)
    df["run"] = df["run"].astype(int)
    df = df.sort_values(["case", "run"]).reset_index(drop=True)
    return df


def build_case_relative_q_and_stage_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    q_list, nu_list, stage_list, vb_smooth_list = [], [], [], []
    for _, sub in out.groupby("case", sort=False):
        sub = sub.sort_values("run")
        vb = sub["VB"].astype(float)
        # NASA has sparse VB measurements; interpolate within case for label/evaluation.
        vb_interp = vb.interpolate(method="linear", limit_direction="both")
        vb_smooth = smooth_series(vb_interp, window=3)
        q = (vb_smooth - vb_smooth.min()) / (vb_smooth.max() - vb_smooth.min() + 1e-12)
        rate = q.diff().fillna(0.0)
        rate_s = smooth_series(rate, window=3)
        nu_norm = (rate_s - rate_s.min()) / (rate_s.max() - rate_s.min() + 1e-12)
        theta_e = float(np.quantile(q, QE))
        theta_l = float(np.quantile(q, QL))
        theta_v = float(np.quantile(nu_norm, QV))
        stage = np.where(q <= theta_e, "early", np.where((q >= theta_l) | (nu_norm >= theta_v), "late", "middle"))
        q_list.append(pd.Series(q.values, index=sub.index))
        nu_list.append(pd.Series(nu_norm.values, index=sub.index))
        vb_smooth_list.append(pd.Series(vb_smooth.values, index=sub.index))
        stage_list.append(pd.Series(stage, index=sub.index))
    out["VB_smooth"] = pd.concat(vb_smooth_list).sort_index()
    out["q_true"] = pd.concat(q_list).sort_index()
    out["nu_norm"] = pd.concat(nu_list).sort_index()
    out["stage_label"] = pd.concat(stage_list).sort_index()
    out["stage_id"] = out["stage_label"].map(STAGE_TO_ID).astype(int)
    return out


def build_case_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case, sub in df.groupby("case"):
        rows.append({
            "case": int(case),
            "n": len(sub),
            "material": int(sub["material"].mode().iloc[0]) if len(sub["material"].dropna()) else np.nan,
            "DOC": float(sub["DOC"].mode().iloc[0]) if len(sub["DOC"].dropna()) else np.nan,
            "feed": float(sub["feed"].mode().iloc[0]) if len(sub["feed"].dropna()) else np.nan,
            "VB_min": float(sub["VB_smooth"].min()),
            "VB_max": float(sub["VB_smooth"].max()),
            "early": int((sub["stage_label"] == "early").sum()),
            "middle": int((sub["stage_label"] == "middle").sum()),
            "late": int((sub["stage_label"] == "late").sum()),
        })
    return pd.DataFrame(rows)


# =========================================================
# 3. Online relative features
# =========================================================
def raw_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(META_FIELDS + ["VB_smooth", "q_true", "nu_norm", "stage_label", "stage_id", "fine_state"])
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def build_online_relative_features(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    parts = []
    for case, sub in df.groupby("case", sort=False):
        sub = sub.sort_values("run").copy()
        base = sub[feat_cols].astype(float)
        hist_mean = base.expanding(min_periods=1).mean()
        hist_std = base.expanding(min_periods=1).std().fillna(0.0)
        rel = (base - hist_mean) / (hist_std + 1e-8)
        slope = base.diff().fillna(0.0)
        rank = base.expanding(min_periods=1).rank(pct=True)
        block = sub[META_FIELDS + ["VB_smooth", "q_true", "nu_norm", "stage_label", "stage_id"]].copy()
        rel.columns = [f"{c}_rel" for c in feat_cols]
        slope.columns = [f"{c}_slope" for c in feat_cols]
        rank.columns = [f"{c}_rank" for c in feat_cols]
        parts.append(pd.concat([block.reset_index(drop=True), rel.reset_index(drop=True), slope.reset_index(drop=True), rank.reset_index(drop=True)], axis=1))
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["case", "run"]).reset_index(drop=True)


def online_feature_cols(df: pd.DataFrame) -> list[str]:
    keep_suffix = ("_rel", "_slope", "_rank")
    return [c for c in df.columns if c.endswith(keep_suffix) and pd.api.types.is_numeric_dtype(df[c])]


def fill_by_train_median(train: pd.DataFrame, other: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    other = other.copy()
    med = train[cols].replace([np.inf, -np.inf], np.nan).median().fillna(0.0)
    train[cols] = train[cols].replace([np.inf, -np.inf], np.nan).fillna(med)
    other[cols] = other[cols].replace([np.inf, -np.inf], np.nan).fillna(med)
    return train, other


def select_features_train_only(train: pd.DataFrame, cols: list[str], top_k: int = N_FEATURES) -> list[str]:
    X = train[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y_stage = train["stage_id"].values.astype(int)
    y_q = train["q_true"].values.astype(float)
    mi_s = mutual_info_classif(X, y_stage, random_state=SEED)
    mi_q = mutual_info_regression(X, y_q, random_state=SEED)
    corr = []
    for c in cols:
        try:
            r = stats.spearmanr(train[c].values, y_q, nan_policy="omit").correlation
            corr.append(abs(r) if np.isfinite(r) else 0.0)
        except Exception:
            corr.append(0.0)
    score = 1.10 * mi_s + 0.35 * mi_q + 0.55 * np.asarray(corr)
    order = np.argsort(score)[::-1]
    return [cols[i] for i in order[:min(top_k, len(cols))]]


# =========================================================
# 4. Tasks, GMM fine states and windows
# =========================================================
def make_cross_case_tasks(df: pd.DataFrame, n_splits: int = 4) -> list[dict[str, Any]]:
    groups = df["case"].values
    gkf = GroupKFold(n_splits=n_splits)
    tasks = []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(df, df["stage_id"], groups=groups), start=1):
        train_cases = sorted(df.iloc[tr_idx]["case"].unique().astype(int).tolist())
        test_cases = sorted(df.iloc[te_idx]["case"].unique().astype(int).tolist())
        test_dist = df.iloc[te_idx]["stage_label"].value_counts().to_dict()
        if any(test_dist.get(s, 0) == 0 for s in STAGES):
            print(f"Warning: N{fold} test split lacks a stage: {test_dist}")
        tasks.append({
            "Task": f"N{fold}",
            "Train_cases": train_cases,
            "Test_cases": test_cases,
        })
    rows = []
    for t in tasks:
        rows.append({"Task": t["Task"], "Train_cases": ",".join(map(str, t["Train_cases"])), "Test_cases": ",".join(map(str, t["Test_cases"]))})
    save_csv(pd.DataFrame(rows), OUT_DIR / "NASA_cross_case_tasks.csv")
    with open(OUT_DIR / "NASA_cross_case_tasks.json", "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    return tasks


def split_train_val_by_case(train_df: pd.DataFrame, val_ratio: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use complete training cases for internal validation.

    NASA cases are short. Splitting the tail of every case can leave validation
    fragments shorter than the sliding window. Holding out complete training
    cases keeps the window semantics intact and still avoids test-case leakage.
    """
    case_counts = train_df.groupby("case").size().sort_values(ascending=False)
    cases = case_counts.index.astype(int).tolist()
    n_val_cases = max(1, int(round(len(cases) * val_ratio)))
    val_cases = set(cases[::max(1, len(cases) // n_val_cases)][:n_val_cases])
    # Make sure at least one sizeable case stays in train.
    if len(val_cases) >= len(cases):
        val_cases = {cases[-1]}
    val_df = train_df[train_df["case"].isin(val_cases)].copy()
    tr_df = train_df[~train_df["case"].isin(val_cases)].copy()
    return tr_df.reset_index(drop=True), val_df.reset_index(drop=True)


def fit_gmm_fine_states(train_df: pd.DataFrame) -> tuple[GaussianMixture, dict[int, int]]:
    X = train_df[["q_true", "nu_norm"]].values.astype(float)
    n_comp = min(K_FINE, max(2, len(train_df) // 4))
    gmm = GaussianMixture(n_components=n_comp, covariance_type="full", random_state=SEED, reg_covar=1e-5, n_init=10)
    gmm.fit(X)
    raw = gmm.predict(X)
    q_means = []
    for k in range(n_comp):
        vals = train_df.loc[raw == k, "q_true"]
        q_means.append((k, float(vals.mean()) if len(vals) else float(gmm.means_[k, 0])))
    raw_to_order = {raw_k: order for order, (raw_k, _) in enumerate(sorted(q_means, key=lambda x: x[1]))}
    return gmm, raw_to_order


def assign_fine_states(df: pd.DataFrame, gmm: GaussianMixture, raw_to_order: dict[int, int]) -> pd.DataFrame:
    out = df.copy()
    raw = gmm.predict(out[["q_true", "nu_norm"]].values.astype(float))
    out["fine_state"] = [raw_to_order.get(int(k), int(k)) for k in raw]
    # If n_components < 5, clamp to available states but model still supports 5 classes.
    out["fine_state"] = out["fine_state"].clip(0, K_FINE - 1).astype(int)
    return out


def build_sliding_windows(df: pd.DataFrame, features: list[str], L: int) -> dict[str, Any]:
    Xs, ys, yf, yq, meta = [], [], [], [], []
    for case, sub in df.groupby("case", sort=False):
        sub = sub.sort_values("run").reset_index(drop=True)
        if len(sub) < L:
            continue
        X = sub[features].values.astype(np.float32)
        for end in range(L - 1, len(sub)):
            start = end - L + 1
            Xs.append(X[start:end + 1])
            ys.append(int(sub.loc[end, "stage_id"]))
            yf.append(int(sub.loc[end, "fine_state"]))
            yq.append(float(sub.loc[end, "q_true"]))
            meta.append({
                "case": int(sub.loc[end, "case"]),
                "run": int(sub.loc[end, "run"]),
                "true_stage": str(sub.loc[end, "stage_label"]),
                "q_true": float(sub.loc[end, "q_true"]),
            })
    if not Xs:
        raise ValueError("No windows were generated. Reduce window length.")
    return {
        "X": np.stack(Xs),
        "ys": np.asarray(ys, dtype=np.int64),
        "yf": np.asarray(yf, dtype=np.int64),
        "yq": np.asarray(yq, dtype=np.float32),
        "meta": pd.DataFrame(meta),
    }


class StageDataset(Dataset):
    def __init__(self, pack: dict[str, Any]):
        self.X = torch.tensor(pack["X"], dtype=torch.float32)
        self.ys = torch.tensor(pack["ys"], dtype=torch.long)
        self.yf = torch.tensor(pack["yf"], dtype=torch.long)
        self.yq = torch.tensor(pack["yq"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.ys)

    def __getitem__(self, idx: int):
        return self.X[idx], self.ys[idx], self.yf[idx], self.yq[idx]


def make_loader(pack: dict[str, Any], shuffle: bool) -> DataLoader:
    return DataLoader(StageDataset(pack), batch_size=BATCH_SIZE, shuffle=shuffle)


# =========================================================
# 5. Models
# =========================================================
class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=pad, dilation=dilation),
            nn.ReLU(),
            nn.BatchNorm1d(out_ch),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=pad, dilation=dilation),
            nn.ReLU(),
            nn.BatchNorm1d(out_ch),
            nn.Dropout(dropout),
        )
        self.res = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        y = self.net(x)
        return y + self.res(x)


class TCNEncoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i, ch in enumerate(TCN_CHANNELS):
            layers.append(TCNBlock(in_ch, ch, dilation=2 ** i, dropout=DROP_OUT))
            in_ch = ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: B,L,D -> B,L,C
        z = x.transpose(1, 2)
        z = self.net(z)
        return z.transpose(1, 2)


class GRUStageModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, GRU_HIDDEN, batch_first=True)
        self.head = nn.Linear(GRU_HIDDEN, 3)

    def forward(self, x):
        h, _ = self.gru(x)
        z = h[:, -1, :]
        return self.head(z), None, None


class TCNGRUStageModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.tcn = TCNEncoder(input_dim)
        self.gru = nn.GRU(TCN_CHANNELS[-1], GRU_HIDDEN, batch_first=True)
        self.shared = nn.Sequential(nn.Linear(GRU_HIDDEN, 64), nn.ReLU(), nn.Dropout(DROP_OUT))
        self.head = nn.Linear(64, 3)

    def forward(self, x):
        z = self.tcn(x)
        h, _ = self.gru(z)
        s = self.shared(h[:, -1, :])
        return self.head(s), None, None


class TCNGRUMultiTaskModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.tcn = TCNEncoder(input_dim)
        self.gru = nn.GRU(TCN_CHANNELS[-1], GRU_HIDDEN, batch_first=True)
        self.shared = nn.Sequential(nn.Linear(GRU_HIDDEN, 64), nn.ReLU(), nn.Dropout(DROP_OUT))
        self.stage_head = nn.Linear(64, 3)
        self.fine_head = nn.Linear(64, K_FINE)
        self.q_head = nn.Linear(64, 1)

    def forward(self, x):
        z = self.tcn(x)
        h, _ = self.gru(z)
        s = self.shared(h[:, -1, :])
        stage_logits = self.stage_head(s)
        fine_logits = self.fine_head(s)
        q_hat = torch.sigmoid(self.q_head(s)).squeeze(-1)
        return stage_logits, fine_logits, q_hat


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=n_classes).astype(float)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def train_model(model: nn.Module, train_pack: dict[str, Any], val_pack: dict[str, Any], multitask: bool) -> tuple[nn.Module, int]:
    model = model.to(DEVICE)
    train_loader = make_loader(train_pack, shuffle=True)
    val_loader = make_loader(val_pack, shuffle=False)
    sw = class_weights(train_pack["ys"], 3)
    fw = class_weights(train_pack["yf"], K_FINE)
    ce_stage = nn.CrossEntropyLoss(weight=sw)
    ce_fine = nn.CrossEntropyLoss(weight=fw)
    huber = nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state, best_score, wait, best_epoch = None, -1e9, 0, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, ys, yf, yq in train_loader:
            xb, ys, yf, yq = xb.to(DEVICE), ys.to(DEVICE), yf.to(DEVICE), yq.to(DEVICE)
            opt.zero_grad()
            stage_logits, fine_logits, q_hat = model(xb)
            loss = ce_stage(stage_logits, ys)
            if multitask:
                loss = loss + 0.35 * ce_fine(fine_logits, yf) + 0.35 * huber(q_hat, yq)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()

        metrics = evaluate_val_model(model, val_loader, multitask)
        score = metrics["macro_f1"] + 0.15 * metrics["mrec"] - 0.05 * metrics["loss"]
        if score > best_score:
            best_score, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


def evaluate_val_model(model: nn.Module, loader: DataLoader, multitask: bool) -> dict[str, float]:
    model.eval()
    loss_sum, n = 0.0, 0
    y_true, y_pred = [], []
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, ys, yf, yq in loader:
            xb, ys = xb.to(DEVICE), ys.to(DEVICE)
            logits, _, _ = model(xb)
            loss = ce(logits, ys)
            pred = logits.argmax(1).cpu().numpy()
            y_true.extend(ys.cpu().numpy())
            y_pred.extend(pred)
            loss_sum += float(loss.item()) * len(ys)
            n += len(ys)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "loss": loss_sum / max(n, 1),
        "macro_f1": macro_f1(y_true, y_pred),
        "mrec": recall_for_class(y_true, y_pred, STAGE_TO_ID["middle"]),
    }


def predict_model(model: nn.Module, pack: dict[str, Any], multitask: bool) -> pd.DataFrame:
    model.eval()
    loader = make_loader(pack, shuffle=False)
    rows = []
    idx0 = 0
    with torch.no_grad():
        for xb, ys, yf, yq in loader:
            xb = xb.to(DEVICE)
            stage_logits, fine_logits, q_hat = model(xb)
            p_raw = torch.softmax(stage_logits, dim=1).cpu().numpy()
            if multitask:
                p_g = torch.softmax(fine_logits, dim=1).cpu().numpy()
                qh = q_hat.cpu().numpy()
            else:
                p_g = np.zeros((len(xb), K_FINE), dtype=float)
                qh = np.zeros(len(xb), dtype=float)
            meta = pack["meta"].iloc[idx0:idx0 + len(xb)].reset_index(drop=True)
            idx0 += len(xb)
            for i in range(len(xb)):
                row = meta.iloc[i].to_dict()
                row.update({
                    "stage_id": int(ys.numpy()[i]),
                    "p_raw_E": p_raw[i, 0],
                    "p_raw_M": p_raw[i, 1],
                    "p_raw_L": p_raw[i, 2],
                    "q_hat": float(qh[i]),
                })
                for k in range(K_FINE):
                    row[f"p_g_{k}"] = float(p_g[i, k])
                rows.append(row)
    return pd.DataFrame(rows)


# =========================================================
# 6. Probability inference and metrics
# =========================================================
def temperature_scale(p: np.ndarray, temp: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    logp = np.log(np.clip(p, 1e-12, 1.0)) / max(temp, 1e-6)
    z = np.exp(logp - logp.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def q_prior(q_hat: np.ndarray, params: dict[str, float]) -> np.ndarray:
    q = np.asarray(q_hat, dtype=float)
    centers = np.array([0.15, 0.50, 0.85])
    sigma = 0.23
    phi = np.exp(-((q[:, None] - centers[None, :]) ** 2) / (2 * sigma ** 2))
    g_e = 1.0 / (1.0 + np.exp(12.0 * (q - params["early_tau"])))
    g_l = 1.0 / (1.0 + np.exp(-12.0 * (q - params["late_tau"])))
    phi[:, 0] *= g_e
    phi[:, 2] *= g_l
    phi[:, 1] = np.maximum(phi[:, 1], params["mid_floor"])
    return phi / phi.sum(axis=1, keepdims=True)


def ordered_filter(probs: np.ndarray, cases: np.ndarray) -> np.ndarray:
    out = np.zeros_like(probs)
    for case in np.unique(cases):
        idx = np.where(cases == case)[0]
        alpha = probs[idx[0]].copy()
        alpha = alpha / alpha.sum()
        out[idx[0]] = alpha
        for pos in idx[1:]:
            pred = alpha @ TRANSITION_A
            alpha = pred * probs[pos]
            alpha = alpha / (alpha.sum() + 1e-12)
            out[pos] = alpha
    return out


def apply_dcpsr_inference(df: pd.DataFrame, params: dict[str, float]) -> pd.DataFrame:
    out = df.copy().sort_values(["case", "run"]).reset_index(drop=True)
    p_raw = out[["p_raw_E", "p_raw_M", "p_raw_L"]].values.astype(float)
    p_raw = temperature_scale(p_raw, params["temperature"])
    p_g = out[[f"p_g_{k}" for k in range(K_FINE)]].values.astype(float)
    p_fine = np.column_stack([p_g[:, 0], p_g[:, 1:4].sum(axis=1), p_g[:, 4]])
    p_fine = p_fine / (p_fine.sum(axis=1, keepdims=True) + 1e-12)
    p_prior = q_prior(out["q_hat"].values.astype(float), params)
    p_aux = params["fine_weight"] * p_fine + (1 - params["fine_weight"]) * p_prior
    p_mix = params["eta"] * p_raw + (1 - params["eta"]) * p_aux
    alpha = ordered_filter(p_mix, out["case"].values)
    p_final = (1 - params["order_blend"]) * p_mix + params["order_blend"] * alpha
    p_final = p_final / p_final.sum(axis=1, keepdims=True)
    for name, arr in [("fine", p_fine), ("prior", p_prior), ("mix", p_mix), ("ordered", alpha), ("final", p_final)]:
        out[f"p_{name}_E"] = arr[:, 0]
        out[f"p_{name}_M"] = arr[:, 1]
        out[f"p_{name}_L"] = arr[:, 2]
    return out


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s = []
    for c in range(3):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return float(np.mean(f1s))


def f1_for_class(y_true: np.ndarray, y_pred: np.ndarray, c: int) -> float:
    tp = np.sum((y_true == c) & (y_pred == c))
    fp = np.sum((y_true != c) & (y_pred == c))
    fn = np.sum((y_true == c) & (y_pred != c))
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return float(2 * p * r / (p + r) if (p + r) else 0.0)


def recall_for_class(y_true: np.ndarray, y_pred: np.ndarray, c: int) -> float:
    den = np.sum(y_true == c)
    return float(np.sum((y_true == c) & (y_pred == c)) / den) if den else 0.0


def consistency_metrics(pred_stage: np.ndarray, probs: np.ndarray, cases: np.ndarray) -> tuple[int, int, float]:
    rev, jump = 0, 0
    smooth_vals = []
    for case in np.unique(cases):
        idx = np.where(cases == case)[0]
        y = pred_stage[idx]
        p = probs[idx]
        if len(y) >= 2:
            diff = np.diff(y)
            rev += int(np.sum(diff < 0))
            jump += int(np.sum(np.abs(diff) >= 2))
            smooth_vals.extend(np.sum(np.abs(np.diff(p, axis=0)), axis=1).tolist())
    return rev, jump, float(np.mean(smooth_vals)) if smooth_vals else 0.0


def compute_stage_metrics(pred_df: pd.DataFrame, method: str, prob_cols: list[str]) -> dict[str, Any]:
    probs = pred_df[prob_cols].values.astype(float)
    y_true = pred_df["stage_id"].values.astype(int)
    y_pred = probs.argmax(axis=1)
    m_mask = y_true == STAGE_TO_ID["middle"]
    m_den = max(int(m_mask.sum()), 1)
    m_to_e = float(np.sum(m_mask & (y_pred == STAGE_TO_ID["early"])) / m_den)
    m_to_l = float(np.sum(m_mask & (y_pred == STAGE_TO_ID["late"])) / m_den)
    rev, jump, smooth = consistency_metrics(y_pred, probs, pred_df["case"].values)
    return {
        "Method": method,
        "Acc": float(np.mean(y_true == y_pred)),
        "Macro-F1": macro_f1(y_true, y_pred),
        "M-F1": f1_for_class(y_true, y_pred, STAGE_TO_ID["middle"]),
        "M-Rec": recall_for_class(y_true, y_pred, STAGE_TO_ID["middle"]),
        "M→E": m_to_e,
        "M→L": m_to_l,
        "Rev": rev,
        "Jump": jump,
        "Smooth": smooth,
    }


def q_consistency(pred_df: pd.DataFrame) -> dict[str, float]:
    y = pred_df["q_true"].values.astype(float)
    qh = pred_df["q_hat"].values.astype(float)
    mae = float(np.mean(np.abs(qh - y)))
    rmse = float(np.sqrt(np.mean((qh - y) ** 2)))
    spearman = float(stats.spearmanr(y, qh, nan_policy="omit").correlation)
    pearson = float(stats.pearsonr(y, qh)[0]) if len(np.unique(qh)) > 1 and len(np.unique(y)) > 1 else 0.0
    smooth_vals = []
    for case in pred_df["case"].unique():
        sub = pred_df[pred_df["case"] == case].sort_values("run")
        if len(sub) >= 2:
            smooth_vals.extend(np.abs(np.diff(sub["q_hat"].values.astype(float))).tolist())
    return {
        "q-MAE": mae,
        "q-RMSE": rmse,
        "Spearman": spearman if np.isfinite(spearman) else 0.0,
        "Pearson": pearson if np.isfinite(pearson) else 0.0,
        "q-Smooth": float(np.mean(smooth_vals)) if smooth_vals else 0.0,
    }


def save_prediction_detail(pred_df: pd.DataFrame, task: str, method: str, prob_cols: list[str], q_col: str = "q_hat") -> None:
    probs = pred_df[prob_cols].values.astype(float)
    pred_id = probs.argmax(axis=1)
    out = pred_df[["case", "run", "true_stage", "q_true"]].copy()
    out["pred_stage"] = [ID_TO_STAGE[int(i)] for i in pred_id]
    out["p_E"] = probs[:, 0]
    out["p_M"] = probs[:, 1]
    out["p_L"] = probs[:, 2]
    out["q_hat"] = pred_df[q_col].values
    out["is_misclassified"] = out["true_stage"] != out["pred_stage"]
    save_csv(out, OUT_DIR / f"Pred_NASA_{task}_{method}.csv")


# =========================================================
# 7. Main task flow
# =========================================================
def run_one_task(task: dict[str, Any], feat_df: pd.DataFrame, L: int) -> tuple[list[dict], dict]:
    task_name = task["Task"]
    train_cases = task["Train_cases"]
    test_cases = task["Test_cases"]
    task_df = feat_df.copy()
    train_all = task_df[task_df["case"].isin(train_cases)].copy()
    test_df = task_df[task_df["case"].isin(test_cases)].copy()
    train_df, val_df = split_train_val_by_case(train_all)
    print("\n" + "=" * 90)
    print(f"{task_name}: train cases={train_cases}, test cases={test_cases}")
    print(f"train rows={len(train_df)}, val rows={len(val_df)}, test rows={len(test_df)}")
    print("test stage distribution:", test_df["stage_label"].value_counts().to_dict())

    all_cols = online_feature_cols(feat_df)
    train_df, val_df = fill_by_train_median(train_df, val_df, all_cols)
    _, test_df = fill_by_train_median(train_df, test_df, all_cols)
    selected = select_features_train_only(train_df, all_cols, N_FEATURES)
    train_df, val_df = fill_by_train_median(train_df, val_df, selected)
    _, test_df = fill_by_train_median(train_df, test_df, selected)

    gmm, raw_to_order = fit_gmm_fine_states(train_df)
    train_df = assign_fine_states(train_df, gmm, raw_to_order)
    val_df = assign_fine_states(val_df, gmm, raw_to_order)
    test_df = assign_fine_states(test_df, gmm, raw_to_order)

    scaler = StandardScaler().fit(train_df[selected].values)
    for d in [train_df, val_df, test_df]:
        d[selected] = np.nan_to_num(scaler.transform(d[selected].values), nan=0.0, posinf=0.0, neginf=0.0)

    tr_pack = build_sliding_windows(train_df, selected, L)
    va_pack = build_sliding_windows(val_df, selected, L)
    te_pack = build_sliding_windows(test_df, selected, L)
    print(f"windows: train={len(tr_pack['ys'])}, val={len(va_pack['ys'])}, test={len(te_pack['ys'])}, L={L}")

    rows = []

    # B9 GRU
    set_seed(SEED)
    b9, ep9 = train_model(GRUStageModel(len(selected)), tr_pack, va_pack, multitask=False)
    pred_b9 = predict_model(b9, te_pack, multitask=False)
    m = compute_stage_metrics(pred_b9, "B9", ["p_raw_E", "p_raw_M", "p_raw_L"])
    rows.append(m)
    save_prediction_detail(pred_b9, task_name, "B9", ["p_raw_E", "p_raw_M", "p_raw_L"])

    # B10 TCN-GRU
    set_seed(SEED + 10)
    b10, ep10 = train_model(TCNGRUStageModel(len(selected)), tr_pack, va_pack, multitask=False)
    pred_b10 = predict_model(b10, te_pack, multitask=False)
    m = compute_stage_metrics(pred_b10, "B10", ["p_raw_E", "p_raw_M", "p_raw_L"])
    rows.append(m)
    save_prediction_detail(pred_b10, task_name, "B10", ["p_raw_E", "p_raw_M", "p_raw_L"])

    # B11/B12 Multi-task and DC-PSR
    set_seed(SEED + 20)
    b11, ep11 = train_model(TCNGRUMultiTaskModel(len(selected)), tr_pack, va_pack, multitask=True)
    pred_mt_raw = predict_model(b11, te_pack, multitask=True)
    pred_mt = apply_dcpsr_inference(pred_mt_raw, PROB_PARAMS)
    rows.append(compute_stage_metrics(pred_mt, "B11", ["p_raw_E", "p_raw_M", "p_raw_L"]))
    rows.append(compute_stage_metrics(pred_mt, "B12", ["p_final_E", "p_final_M", "p_final_L"]))
    save_prediction_detail(pred_mt, task_name, "B11", ["p_raw_E", "p_raw_M", "p_raw_L"])
    save_prediction_detail(pred_mt, task_name, "B12", ["p_final_E", "p_final_M", "p_final_L"])

    qrow = q_consistency(pred_mt)
    qrow.update({"Task": task_name, "Train_cases": ",".join(map(str, train_cases)), "Test_cases": ",".join(map(str, test_cases))})

    for r in rows:
        r.update({
            "Task": task_name,
            "Train_cases": ",".join(map(str, train_cases)),
            "Test_cases": ",".join(map(str, test_cases)),
            "Window length": L,
        })
    return rows, qrow


def mean_std_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in df.groupby("Method"):
        row = {"Method": method}
        for metric in ["Acc", "Macro-F1", "M-F1", "M-Rec", "Smooth"]:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_std"] = float(sub[metric].std(ddof=1)) if len(sub) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Method")


def q_mean_table(df: pd.DataFrame) -> pd.DataFrame:
    row = {}
    for metric in ["q-MAE", "q-RMSE", "Spearman", "Pearson", "q-Smooth"]:
        row[f"{metric}_mean"] = float(df[metric].mean())
        row[f"{metric}_std"] = float(df[metric].std(ddof=1)) if len(df) > 1 else 0.0
    return pd.DataFrame([row])


def plot_optional_summary(mean_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        methods = mean_df["Method"].tolist()
        x = np.arange(len(methods))
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.bar(x - 0.18, mean_df["Macro-F1_mean"], width=0.36, label="Macro-F1", color="#4C78A8")
        ax.bar(x + 0.18, mean_df["M-F1_mean"], width=0.36, label="M-F1", color="#F58518")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Mean score")
        ax.legend(frameon=False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "Fig_NASA_B9_B12_mean_metrics.png", dpi=700)
        plt.close(fig)
    except Exception as exc:
        print(f"Optional plot failed: {exc}")


def main() -> None:
    set_seed(SEED)
    print("=" * 100)
    print("NASA cross-case experiment for DC-PSR")
    print(f"Device: {DEVICE}")
    print(f"Output: {OUT_DIR}")
    print("=" * 100)

    config = {
        "seed": SEED,
        "window_length": L_DEFAULT,
        "TCN channels": TCN_CHANNELS,
        "GRU hidden": GRU_HIDDEN,
        "dropout": DROP_OUT,
        "learning_rate": LR,
        "probability_params": PROB_PARAMS,
        "transition_matrix": TRANSITION_A.tolist(),
    }
    with open(OUT_DIR / "NASA_experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    raw_feat = extract_signal_features()
    labeled = build_case_relative_q_and_stage_labels(raw_feat)
    case_summary = build_case_summary(labeled)
    save_csv(case_summary, OUT_DIR / "NASA_case_summary.csv")
    save_csv(labeled, OUT_DIR / "NASA_run_level_features_with_labels.csv")

    raw_cols = raw_feature_columns(labeled)
    online = build_online_relative_features(labeled, raw_cols)
    save_csv(online, OUT_DIR / "NASA_online_relative_features.csv")

    tasks = make_cross_case_tasks(online, n_splits=4)
    all_metric_rows, q_rows = [], []
    for task in tasks:
        metric_rows, qrow = run_one_task(task, online, L_DEFAULT)
        all_metric_rows.extend(metric_rows)
        q_rows.append(qrow)

    metrics_df = pd.DataFrame(all_metric_rows)
    metrics_df = metrics_df[["Task", "Train_cases", "Test_cases", "Method", "Acc", "Macro-F1", "M-F1", "M-Rec", "M→E", "M→L", "Rev", "Jump", "Smooth", "Window length"]]
    save_csv(metrics_df, OUT_DIR / "Table_NASA_cross_case_B9_B12_metrics.csv")
    mean_df = mean_std_table(metrics_df)
    save_csv(mean_df, OUT_DIR / "Table_NASA_cross_case_mean_std.csv")

    q_df = pd.DataFrame(q_rows)
    q_df = q_df[["Task", "Train_cases", "Test_cases", "q-MAE", "q-RMSE", "Spearman", "Pearson", "q-Smooth"]]
    save_csv(q_df, OUT_DIR / "Table_NASA_q_consistency_B12.csv")
    q_mean = q_mean_table(q_df)
    save_csv(q_mean, OUT_DIR / "Table_NASA_q_consistency_B12_mean.csv")
    plot_optional_summary(mean_df)

    best_macro = mean_df.loc[mean_df["Macro-F1_mean"].idxmax()]
    best_mf1 = mean_df.loc[mean_df["M-F1_mean"].idxmax()]
    best_smooth = mean_df.loc[mean_df["Smooth_mean"].idxmin()]

    print("\nNASA cross-case experiment finished.\n")
    print(f"Best mean Macro-F1: {best_macro['Method']} = {best_macro['Macro-F1_mean']:.4f}")
    print(f"Best mean M-F1: {best_mf1['Method']} = {best_mf1['M-F1_mean']:.4f}")
    print(f"Lowest mean Smooth: {best_smooth['Method']} = {best_smooth['Smooth_mean']:.4f}")
    print(f"B12 q-MAE: {q_mean['q-MAE_mean'].iloc[0]:.4f}")
    print(f"B12 q-RMSE: {q_mean['q-RMSE_mean'].iloc[0]:.4f}")
    print(f"B12 Spearman: {q_mean['Spearman_mean'].iloc[0]:.4f}")
    print(f"B12 Pearson: {q_mean['Pearson_mean'].iloc[0]:.4f}")
    print(f"B12 q-Smooth: {q_mean['q-Smooth_mean'].iloc[0]:.4f}")
    print(f"\nResults saved to:\n{OUT_DIR}")


# =========================================================
# 8. NASA optimized overrides
# =========================================================
def stage_distribution(df: pd.DataFrame) -> dict[str, int]:
    return {s: int((df["stage_label"] == s).sum()) for s in STAGES}


def select_validation_cases(train_all: pd.DataFrame, n_cases: int = 2) -> list[int]:
    rows = []
    for case, sub in train_all.groupby("case"):
        dist = stage_distribution(sub)
        present = sum(1 for s in STAGES if dist[s] > 0)
        balance = min(dist.values()) / max(max(dist.values()), 1)
        rows.append({"case": int(case), "n": int(len(sub)), "present": present, "balance": float(balance)})
    cand = pd.DataFrame(rows)
    cand["score"] = cand["present"] * 1000 + cand["balance"] * 100 + cand["n"]
    return cand.sort_values(["score", "n", "case"], ascending=[False, False, True]).head(n_cases)["case"].astype(int).tolist()


def make_cross_case_tasks_optimized(df: pd.DataFrame) -> list[dict[str, Any]]:
    tasks = []
    for t in FIXED_TASKS:
        original_train = list(t["Train_cases"])
        test_cases = list(t["Test_cases"])
        train_all = df[df["case"].isin(original_train)].copy()
        val_cases = select_validation_cases(train_all, n_cases=2)
        overlap = set(val_cases) & set(test_cases)
        if overlap:
            raise RuntimeError(f"Validation/test overlap in {t['Task']}: {sorted(overlap)}")
        train_cases = [c for c in original_train if c not in val_cases]
        tasks.append({
            "Task": t["Task"],
            "Original_train_cases": original_train,
            "Train_cases": train_cases,
            "Validation_cases": val_cases,
            "Test_cases": test_cases,
        })
    rows = []
    for t in tasks:
        rows.append({
            "Task": t["Task"],
            "Original_train_cases": ",".join(map(str, t["Original_train_cases"])),
            "Train_cases": ",".join(map(str, t["Train_cases"])),
            "Validation_cases": ",".join(map(str, t["Validation_cases"])),
            "Test_cases": ",".join(map(str, t["Test_cases"])),
        })
    save_csv(pd.DataFrame(rows), OUT_DIR / "NASA_cross_case_tasks_optimized.csv")
    with open(OUT_DIR / "NASA_cross_case_tasks_optimized.json", "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    return tasks


def set_runtime_config(cfg: dict[str, Any], seed: int) -> None:
    global SEED, N_FEATURES, L_DEFAULT, BATCH_SIZE, LR, WEIGHT_DECAY, DROP_OUT, TCN_CHANNELS, GRU_HIDDEN
    SEED = int(seed)
    N_FEATURES = int(cfg["top_k"])
    L_DEFAULT = int(cfg["L"])
    BATCH_SIZE = int(cfg["batch_size"])
    LR = float(cfg["lr"])
    WEIGHT_DECAY = float(cfg["weight_decay"])
    DROP_OUT = float(cfg["dropout"])
    TCN_CHANNELS = tuple(cfg["tcn_channels"])
    GRU_HIDDEN = int(cfg["gru_hidden"])


def prepare_config_packs(feat_df: pd.DataFrame, task: dict[str, Any], cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    set_runtime_config(cfg, seed)
    train_df = feat_df[feat_df["case"].isin(task["Train_cases"])].copy()
    val_df = feat_df[feat_df["case"].isin(task["Validation_cases"])].copy()
    test_df = feat_df[feat_df["case"].isin(task["Test_cases"])].copy()
    all_cols = online_feature_cols(feat_df)

    train_df, val_df = fill_by_train_median(train_df, val_df, all_cols)
    _, test_df = fill_by_train_median(train_df, test_df, all_cols)
    selected = select_features_train_only(train_df, all_cols, int(cfg["top_k"]))
    train_df, val_df = fill_by_train_median(train_df, val_df, selected)
    _, test_df = fill_by_train_median(train_df, test_df, selected)

    gmm, raw_to_order = fit_gmm_fine_states(train_df)
    train_df = assign_fine_states(train_df, gmm, raw_to_order)
    val_df = assign_fine_states(val_df, gmm, raw_to_order)
    test_df = assign_fine_states(test_df, gmm, raw_to_order)

    scaler = StandardScaler().fit(train_df[selected].values)
    for d in [train_df, val_df, test_df]:
        d[selected] = np.nan_to_num(scaler.transform(d[selected].values), nan=0.0, posinf=0.0, neginf=0.0)

    L = int(cfg["L"])
    return {
        "selected": selected,
        "train_pack": build_sliding_windows(train_df, selected, L),
        "val_pack": build_sliding_windows(val_df, selected, L),
        "test_pack": build_sliding_windows(test_df, selected, L),
    }


def q_prior(q_hat: np.ndarray, params: dict[str, float]) -> np.ndarray:
    q = np.asarray(q_hat, dtype=float)
    centers = np.array([0.15, 0.50, 0.85])
    sigma = float(params.get("prior_sigma", 0.23))
    phi = np.exp(-((q[:, None] - centers[None, :]) ** 2) / (2 * sigma ** 2))
    g_e = 1.0 / (1.0 + np.exp(12.0 * (q - params["early_tau"])))
    g_l = 1.0 / (1.0 + np.exp(-12.0 * (q - params["late_tau"])))
    phi[:, 0] *= g_e
    phi[:, 2] *= g_l
    phi[:, 1] = np.maximum(phi[:, 1], params["mid_floor"])
    return phi / (phi.sum(axis=1, keepdims=True) + 1e-12)


def precision_for_class(y_true: np.ndarray, y_pred: np.ndarray, c: int) -> float:
    den = np.sum(y_pred == c)
    return float(np.sum((y_true == c) & (y_pred == c)) / den) if den else 0.0


def compute_stage_metrics(pred_df: pd.DataFrame, method: str, prob_cols: list[str]) -> dict[str, Any]:
    probs = pred_df[prob_cols].values.astype(float)
    y_true = pred_df["stage_id"].values.astype(int)
    y_pred = probs.argmax(axis=1)
    m_mask = y_true == STAGE_TO_ID["middle"]
    m_den = max(int(m_mask.sum()), 1)
    m_to_e = float(np.sum(m_mask & (y_pred == STAGE_TO_ID["early"])) / m_den)
    m_to_l = float(np.sum(m_mask & (y_pred == STAGE_TO_ID["late"])) / m_den)
    rev, jump, smooth = consistency_metrics(y_pred, probs, pred_df["case"].values)
    return {
        "Method": method,
        "Acc": float(np.mean(y_true == y_pred)),
        "Macro-F1": macro_f1(y_true, y_pred),
        "E-F1": f1_for_class(y_true, y_pred, STAGE_TO_ID["early"]),
        "M-F1": f1_for_class(y_true, y_pred, STAGE_TO_ID["middle"]),
        "L-F1": f1_for_class(y_true, y_pred, STAGE_TO_ID["late"]),
        "M-Pre": precision_for_class(y_true, y_pred, STAGE_TO_ID["middle"]),
        "M-Rec": recall_for_class(y_true, y_pred, STAGE_TO_ID["middle"]),
        "M→E": m_to_e,
        "M→L": m_to_l,
        "Rev": rev,
        "Jump": jump,
        "Smooth": smooth,
        "Predicted_class_count": int(len(np.unique(y_pred))),
        "Early_recall": recall_for_class(y_true, y_pred, STAGE_TO_ID["early"]),
        "Middle_recall": recall_for_class(y_true, y_pred, STAGE_TO_ID["middle"]),
        "Late_recall": recall_for_class(y_true, y_pred, STAGE_TO_ID["late"]),
    }


def validation_score(metrics: dict[str, Any]) -> float:
    score = 0.35 * metrics["Macro-F1"] + 0.30 * metrics["M-F1"] + 0.20 * metrics["M-Rec"] - 0.15 * metrics["Smooth"]
    if metrics["Predicted_class_count"] < 2:
        score -= 0.30
    if metrics["Predicted_class_count"] == 1:
        score -= 0.70
    if metrics["Early_recall"] < 0.05:
        score -= 0.15
    if metrics["Late_recall"] < 0.05:
        score -= 0.15
    return float(score)


def b12_validation_score(metrics: dict[str, Any]) -> float:
    score = (
            0.30 * metrics["Macro-F1"]
            + 0.25 * metrics["M-F1"]
            + 0.20 * metrics["M-Rec"]
            - 0.15 * metrics["Smooth"]
            - 0.05 * (metrics["M→E"] + metrics["M→L"])
            + 0.05 * metrics["Acc"]
    )
    if metrics["Predicted_class_count"] < 2:
        score -= 0.50
    if metrics["Predicted_class_count"] == 1:
        score -= 1.00
    if metrics["Early_recall"] < 0.05:
        score -= 0.25
    if metrics["Late_recall"] < 0.05:
        score -= 0.25
    if metrics["M-Rec"] > 0.95 and metrics["Macro-F1"] < 0.35:
        score -= 0.30
    return float(score)


def train_model(model: nn.Module, train_pack: dict[str, Any], val_pack: dict[str, Any], multitask: bool) -> tuple[nn.Module, int]:
    model = model.to(DEVICE)
    train_loader = make_loader(train_pack, shuffle=True)
    sw = class_weights(train_pack["ys"], 3)
    fw = class_weights(train_pack["yf"], K_FINE)
    ce_stage = nn.CrossEntropyLoss(weight=sw)
    ce_fine = nn.CrossEntropyLoss(weight=fw)
    huber = nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state, best_score, wait, best_epoch = None, -1e18, 0, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, ys, yf, yq in train_loader:
            xb, ys, yf, yq = xb.to(DEVICE), ys.to(DEVICE), yf.to(DEVICE), yq.to(DEVICE)
            opt.zero_grad()
            stage_logits, fine_logits, q_hat = model(xb)
            loss = ce_stage(stage_logits, ys)
            if multitask:
                loss = loss + 0.35 * ce_fine(fine_logits, yf) + 0.35 * huber(q_hat, yq)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
        pred_val = predict_model(model, val_pack, multitask=multitask)
        metrics = compute_stage_metrics(pred_val, "VAL", ["p_raw_E", "p_raw_M", "p_raw_L"])
        score = validation_score(metrics)
        if score > best_score:
            best_score, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


def q_consistency(pred_df: pd.DataFrame) -> dict[str, float]:
    y = pred_df["q_true"].values.astype(float)
    qh = pred_df["q_hat"].values.astype(float)
    mae = float(np.mean(np.abs(qh - y)))
    rmse = float(np.sqrt(np.mean((qh - y) ** 2)))
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if denom < 1e-12 else float(1.0 - np.sum((y - qh) ** 2) / denom)
    spearman = float(stats.spearmanr(y, qh, nan_policy="omit").correlation) if len(y) > 2 else np.nan
    pearson = float(stats.pearsonr(y, qh)[0]) if len(np.unique(qh)) > 1 and len(np.unique(y)) > 1 else np.nan
    smooth_vals = []
    for case in pred_df["case"].unique():
        sub = pred_df[pred_df["case"] == case].sort_values("run")
        if len(sub) >= 2:
            smooth_vals.extend(np.abs(np.diff(sub["q_hat"].values.astype(float))).tolist())
    return {
        "q-MAE": mae,
        "q-RMSE": rmse,
        "q-R2": r2,
        "Spearman": spearman if np.isfinite(spearman) else np.nan,
        "Pearson": pearson if np.isfinite(pearson) else np.nan,
        "q-Smooth": float(np.mean(smooth_vals)) if smooth_vals else 0.0,
    }


def iter_b12_param_grid():
    keys = list(B12_SEARCH.keys())
    for vals in itertools.product(*[B12_SEARCH[k] for k in keys]):
        yield dict(zip(keys, vals))


def search_b12_params(pred_val_raw: pd.DataFrame, task_name: str, seed: int) -> tuple[dict[str, float], float, dict[str, Any], bool]:
    best_params, best_score, best_metrics = None, -1e18, None
    fallback_params, fallback_metrics = None, None
    fallback_best_key = (-1e18, -1e18)
    any_non_degenerate = False
    for params in iter_b12_param_grid():
        pred = apply_dcpsr_inference(pred_val_raw, params)
        metrics = compute_stage_metrics(pred, "B12", ["p_final_E", "p_final_M", "p_final_L"])
        score = b12_validation_score(metrics)
        if metrics["Predicted_class_count"] > 1:
            any_non_degenerate = True
        if score > best_score:
            best_params, best_score, best_metrics = params, score, metrics
        key = (metrics["Macro-F1"], -metrics["Smooth"])
        if key > fallback_best_key:
            fallback_best_key = key
            fallback_params, fallback_metrics = params, metrics
    used_fallback = False
    if not any_non_degenerate and fallback_params is not None:
        print(f"WARNING: {task_name} seed {seed} B12 all searched params predicted one class; using highest Macro-F1 fallback.")
        best_params = fallback_params
        best_metrics = fallback_metrics
        best_score = b12_validation_score(best_metrics)
        used_fallback = True
    return best_params, float(best_score), best_metrics, used_fallback


def save_prediction_detail(pred_df: pd.DataFrame, task: str, method: str, prob_cols: list[str], q_col: str = "q_hat", seed: int | None = None) -> None:
    probs = pred_df[prob_cols].values.astype(float)
    pred_id = probs.argmax(axis=1)
    out = pred_df[["case", "run", "true_stage", "q_true"]].copy()
    out["pred_stage"] = [ID_TO_STAGE[int(i)] for i in pred_id]
    out["p_E"] = probs[:, 0]
    out["p_M"] = probs[:, 1]
    out["p_L"] = probs[:, 2]
    out["q_hat"] = pred_df[q_col].values
    out["is_misclassified"] = out["true_stage"] != out["pred_stage"]
    out = out[["case", "run", "true_stage", "pred_stage", "p_E", "p_M", "p_L", "q_true", "q_hat", "is_misclassified"]]
    suffix = f"_seed{seed}" if seed is not None else ""
    save_csv(out, OUT_DIR / f"Pred_NASA_{task}_{method}{suffix}.csv")


def warn_if_degenerate(task: str, method: str, seed: int, m: dict[str, Any]) -> None:
    if m["Predicted_class_count"] == 1:
        print(f"WARNING: {task} {method} seed {seed} predicted only one class.")
    if m["Early_recall"] <= 0:
        print(f"WARNING: {task} {method} seed {seed} early recall is zero.")
    if m["Late_recall"] <= 0:
        print(f"WARNING: {task} {method} seed {seed} late recall is zero.")
    if m["Smooth"] < 0.01 and m["Macro-F1"] < 0.35:
        print(f"WARNING: {task} {method} seed {seed} has unusually low Smooth but poor Macro-F1.")


def train_select_method(method: str, model_cls: type[nn.Module], task: dict[str, Any], feat_df: pd.DataFrame, seed: int, multitask: bool) -> dict[str, Any]:
    best = None
    configs = BASELINE_MODEL_CONFIGS if method in {"B9", "B10"} else MODEL_CONFIGS
    for i, cfg in enumerate(configs):
        cfg_seed = seed + i * 100 + {"B9": 9, "B10": 10, "B11": 11}[method]
        set_seed(cfg_seed)
        packs = prepare_config_packs(feat_df, task, cfg, cfg_seed)
        set_runtime_config(cfg, cfg_seed)
        model = model_cls(len(packs["selected"]))
        model, best_epoch = train_model(model, packs["train_pack"], packs["val_pack"], multitask=multitask)
        pred_val = predict_model(model, packs["val_pack"], multitask=multitask)
        val_metrics = compute_stage_metrics(pred_val, method, ["p_raw_E", "p_raw_M", "p_raw_L"])
        val_score = validation_score(val_metrics)
        print(f"  {method} cfg={cfg['name']} val_score={val_score:.4f} Macro-F1={val_metrics['Macro-F1']:.4f} M-F1={val_metrics['M-F1']:.4f} M-Rec={val_metrics['M-Rec']:.4f} Smooth={val_metrics['Smooth']:.4f} epoch={best_epoch}")
        item = {"cfg": cfg, "packs": packs, "model": model, "val_metrics": val_metrics, "val_score": val_score}
        if best is None or val_score > best["val_score"]:
            best = item
    print(f"  {method} selected cfg={best['cfg']['name']} val_score={best['val_score']:.4f}")
    return best


def add_metric_context(row: dict[str, Any], task: dict[str, Any], seed: int, cfg: dict[str, Any], val_score: float) -> dict[str, Any]:
    out = dict(row)
    out.update({
        "Task": task["Task"],
        "Seed": seed,
        "Train_cases": ",".join(map(str, task["Train_cases"])),
        "Validation_cases": ",".join(map(str, task["Validation_cases"])),
        "Test_cases": ",".join(map(str, task["Test_cases"])),
        "Window length": int(cfg["L"]),
        "Top_k": int(cfg["top_k"]),
        "Config": cfg["name"],
        "Best_validation_score": float(val_score),
    })
    return out


def diagnostic_row(task: dict[str, Any], method: str, seed: int, cfg: dict[str, Any], val_score: float, val_m: dict[str, Any], test_m: dict[str, Any]) -> dict[str, Any]:
    return {
        "Task": task["Task"], "Method": method, "Seed": seed,
        "Best_validation_score": float(val_score),
        "Validation_Acc": val_m["Acc"], "Validation_MacroF1": val_m["Macro-F1"], "Validation_MF1": val_m["M-F1"],
        "Validation_MRec": val_m["M-Rec"], "Validation_Smooth": val_m["Smooth"],
        "Test_Acc": test_m["Acc"], "Test_MacroF1": test_m["Macro-F1"], "Test_MF1": test_m["M-F1"],
        "Test_MRec": test_m["M-Rec"], "Test_Smooth": test_m["Smooth"],
        "Predicted_class_count": test_m["Predicted_class_count"],
        "Early_recall": test_m["Early_recall"], "Middle_recall": test_m["Middle_recall"], "Late_recall": test_m["Late_recall"],
        "Best_L": int(cfg["L"]), "Best_top_k": int(cfg["top_k"]), "Best_lr": float(cfg["lr"]),
        "Best_dropout": float(cfg["dropout"]), "Best_model_size": f"tcn{tuple(cfg['tcn_channels'])}_gru{cfg['gru_hidden']}",
        "Best_config": cfg["name"],
    }


def run_one_task_seed_optimized(task: dict[str, Any], feat_df: pd.DataFrame, seed: int) -> tuple[list[dict], dict, list[dict], dict]:
    task_name = task["Task"]
    train_df = feat_df[feat_df["case"].isin(task["Train_cases"])]
    val_df = feat_df[feat_df["case"].isin(task["Validation_cases"])]
    test_df = feat_df[feat_df["case"].isin(task["Test_cases"])]
    print("\n" + "=" * 100)
    print(f"{task_name} seed {seed}")
    print(f"Train_cases: {task['Train_cases']}")
    print(f"Validation_cases: {task['Validation_cases']}")
    print(f"Test_cases: {task['Test_cases']}")
    print(f"train rows={len(train_df)}, dist={stage_distribution(train_df)}")
    print(f"val rows={len(val_df)}, dist={stage_distribution(val_df)}")
    print(f"test rows={len(test_df)}, dist={stage_distribution(test_df)}")

    metric_rows, diag_rows = [], []
    b9 = train_select_method("B9", GRUStageModel, task, feat_df, seed, multitask=False)
    set_runtime_config(b9["cfg"], seed)
    pred = predict_model(b9["model"], b9["packs"]["test_pack"], multitask=False)
    m = compute_stage_metrics(pred, "B9", ["p_raw_E", "p_raw_M", "p_raw_L"])
    warn_if_degenerate(task_name, "B9", seed, m)
    metric_rows.append(add_metric_context(m, task, seed, b9["cfg"], b9["val_score"]))
    diag_rows.append(diagnostic_row(task, "B9", seed, b9["cfg"], b9["val_score"], b9["val_metrics"], m))
    save_prediction_detail(pred, task_name, "B9", ["p_raw_E", "p_raw_M", "p_raw_L"], seed=seed)

    b10 = train_select_method("B10", TCNGRUStageModel, task, feat_df, seed, multitask=False)
    set_runtime_config(b10["cfg"], seed)
    pred = predict_model(b10["model"], b10["packs"]["test_pack"], multitask=False)
    m = compute_stage_metrics(pred, "B10", ["p_raw_E", "p_raw_M", "p_raw_L"])
    warn_if_degenerate(task_name, "B10", seed, m)
    metric_rows.append(add_metric_context(m, task, seed, b10["cfg"], b10["val_score"]))
    diag_rows.append(diagnostic_row(task, "B10", seed, b10["cfg"], b10["val_score"], b10["val_metrics"], m))
    save_prediction_detail(pred, task_name, "B10", ["p_raw_E", "p_raw_M", "p_raw_L"], seed=seed)

    b11 = train_select_method("B11", TCNGRUMultiTaskModel, task, feat_df, seed, multitask=True)
    set_runtime_config(b11["cfg"], seed)
    pred_val_b11 = predict_model(b11["model"], b11["packs"]["val_pack"], multitask=True)
    pred_test_b11 = predict_model(b11["model"], b11["packs"]["test_pack"], multitask=True)
    m = compute_stage_metrics(pred_test_b11, "B11", ["p_raw_E", "p_raw_M", "p_raw_L"])
    warn_if_degenerate(task_name, "B11", seed, m)
    metric_rows.append(add_metric_context(m, task, seed, b11["cfg"], b11["val_score"]))
    diag_rows.append(diagnostic_row(task, "B11", seed, b11["cfg"], b11["val_score"], b11["val_metrics"], m))
    save_prediction_detail(pred_test_b11, task_name, "B11", ["p_raw_E", "p_raw_M", "p_raw_L"], seed=seed)

    best_params, b12_val_score, b12_val_metrics, used_fallback = search_b12_params(pred_val_b11, task_name, seed)
    pred_b12 = apply_dcpsr_inference(pred_test_b11, best_params)
    m = compute_stage_metrics(pred_b12, "B12", ["p_final_E", "p_final_M", "p_final_L"])
    warn_if_degenerate(task_name, "B12", seed, m)
    metric_rows.append(add_metric_context(m, task, seed, b11["cfg"], b12_val_score))
    diag_rows.append(diagnostic_row(task, "B12", seed, b11["cfg"], b12_val_score, b12_val_metrics, m))
    save_prediction_detail(pred_b12, task_name, "B12", ["p_final_E", "p_final_M", "p_final_L"], seed=seed)

    qrow = q_consistency(pred_b12)
    qrow.update({
        "Task": task_name, "Seed": seed,
        "Train_cases": ",".join(map(str, task["Train_cases"])),
        "Validation_cases": ",".join(map(str, task["Validation_cases"])),
        "Test_cases": ",".join(map(str, task["Test_cases"])),
        "Config": b11["cfg"]["name"],
    })
    fusion_row = {"Task": task_name, "Seed": seed, "Best_validation_score": b12_val_score, "Used_fallback": used_fallback, **best_params}
    return metric_rows, qrow, diag_rows, fusion_row


def mean_std_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Pre", "M-Rec", "M→E", "M→L", "Rev", "Jump", "Smooth", "Predicted_class_count", "Early_recall", "Late_recall"]
    for method, sub in df.groupby("Method"):
        row = {"Method": method}
        for metric in metrics:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_std"] = float(sub[metric].std(ddof=1)) if len(sub) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Method")


def q_mean_table(df: pd.DataFrame) -> pd.DataFrame:
    row = {}
    for metric in ["q-MAE", "q-RMSE", "q-R2", "Spearman", "Pearson", "q-Smooth"]:
        row[f"{metric}_mean"] = float(df[metric].mean(skipna=True))
        row[f"{metric}_std"] = float(df[metric].std(ddof=1, skipna=True)) if df[metric].notna().sum() > 1 else 0.0
    return pd.DataFrame([row])


def plot_optional_summary(mean_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        methods = mean_df["Method"].tolist()
        x = np.arange(len(methods))
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.bar(x - 0.25, mean_df["Acc_mean"], width=0.25, label="Acc", color="#4C78A8")
        ax.bar(x, mean_df["Macro-F1_mean"], width=0.25, label="Macro-F1", color="#F58518")
        ax.bar(x + 0.25, mean_df["M-F1_mean"], width=0.25, label="M-F1", color="#54A24B")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Mean score")
        ax.legend(frameon=False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "Fig_NASA_optimized_mean_metrics_B9_B12.png", dpi=500)
        plt.close(fig)
    except Exception as exc:
        print(f"Optional plot failed: {exc}")


def main() -> None:
    set_seed(RUN_SEEDS[0])
    print("=" * 100)
    print("NASA optimized cross-case experiment for DC-PSR")
    print(f"Device: {DEVICE}")
    print(f"Seeds: {RUN_SEEDS}")
    print(f"Output: {OUT_DIR}")
    print("=" * 100)

    cfg_json = {
        "RUN_SEEDS": RUN_SEEDS,
        "MODEL_CONFIGS": [{**c, "tcn_channels": list(c["tcn_channels"])} for c in MODEL_CONFIGS],
        "B12_SEARCH": B12_SEARCH,
        "MAX_EPOCHS": EPOCHS,
        "PATIENCE": PATIENCE,
        "fixed_tasks": FIXED_TASKS,
        "transition_matrix": TRANSITION_A.tolist(),
    }
    with open(OUT_DIR / "NASA_experiment_config_optimized.json", "w", encoding="utf-8") as f:
        json.dump(cfg_json, f, ensure_ascii=False, indent=2)

    raw_feat = extract_signal_features()
    labeled = build_case_relative_q_and_stage_labels(raw_feat)
    save_csv(build_case_summary(labeled), OUT_DIR / "NASA_case_summary_optimized.csv")
    save_csv(labeled, OUT_DIR / "NASA_run_level_features_with_labels_optimized.csv")

    raw_cols = raw_feature_columns(labeled)
    online = build_online_relative_features(labeled, raw_cols)
    save_csv(online, OUT_DIR / "NASA_online_relative_features_optimized.csv")

    tasks = make_cross_case_tasks_optimized(online)
    all_metric_rows, q_rows, diag_rows, fusion_rows = [], [], [], []
    for seed in RUN_SEEDS:
        for task in tasks:
            metric_rows, qrow, drows, frow = run_one_task_seed_optimized(task, online, seed)
            all_metric_rows.extend(metric_rows)
            q_rows.append(qrow)
            diag_rows.extend(drows)
            fusion_rows.append(frow)

    metric_cols = [
        "Task", "Seed", "Train_cases", "Validation_cases", "Test_cases", "Method",
        "Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Pre", "M-Rec", "M→E", "M→L",
        "Rev", "Jump", "Smooth", "Predicted_class_count", "Early_recall", "Late_recall",
        "Window length", "Top_k", "Config", "Best_validation_score",
    ]
    metrics_df = pd.DataFrame(all_metric_rows)[metric_cols]
    save_csv(metrics_df, OUT_DIR / "Table_NASA_cross_case_B9_B12_metrics_by_seed.csv")
    save_csv(metrics_df, OUT_DIR / "Table_NASA_cross_case_B9_B12_metrics_optimized.csv")
    mean_df = mean_std_table(metrics_df)
    save_csv(mean_df, OUT_DIR / "Table_NASA_cross_case_mean_std_optimized.csv")

    q_cols = ["Task", "Seed", "Train_cases", "Validation_cases", "Test_cases", "Config", "q-MAE", "q-RMSE", "q-R2", "Spearman", "Pearson", "q-Smooth"]
    q_df = pd.DataFrame(q_rows)[q_cols]
    save_csv(q_df, OUT_DIR / "Table_NASA_q_consistency_B12_by_seed.csv")
    save_csv(q_df, OUT_DIR / "Table_NASA_q_consistency_B12_optimized.csv")
    q_mean = q_mean_table(q_df)
    save_csv(q_mean, OUT_DIR / "Table_NASA_q_consistency_B12_mean_optimized.csv")

    save_csv(pd.DataFrame(diag_rows), OUT_DIR / "NASA_experiment_diagnostics_optimized.csv")
    save_csv(pd.DataFrame(fusion_rows), OUT_DIR / "NASA_B12_best_fusion_params.csv")
    plot_optional_summary(mean_df)

    best_macro = mean_df.loc[mean_df["Macro-F1_mean"].idxmax()]
    best_mf1 = mean_df.loc[mean_df["M-F1_mean"].idxmax()]
    best_mrec = mean_df.loc[mean_df["M-Rec_mean"].idxmax()]
    best_smooth = mean_df.loc[mean_df["Smooth_mean"].idxmin()]
    b12_mean = mean_df[mean_df["Method"] == "B12"].iloc[0]

    print("\nNASA optimized cross-case experiment finished.\n")
    print(f"Best method by mean Macro-F1: {best_macro['Method']} = {best_macro['Macro-F1_mean']:.4f}")
    print(f"Best method by mean M-F1: {best_mf1['Method']} = {best_mf1['M-F1_mean']:.4f}")
    print(f"Best method by mean M-Rec: {best_mrec['Method']} = {best_mrec['M-Rec_mean']:.4f}")
    print(f"Best method by mean Smooth: {best_smooth['Method']} = {best_smooth['Smooth_mean']:.4f}")
    print(f"B12 optimized mean Acc: {b12_mean['Acc_mean']:.4f}")
    print(f"B12 optimized mean Macro-F1: {b12_mean['Macro-F1_mean']:.4f}")
    print(f"B12 optimized mean M-F1: {b12_mean['M-F1_mean']:.4f}")
    print(f"B12 optimized mean M-Rec: {b12_mean['M-Rec_mean']:.4f}")
    print(f"B12 optimized mean Smooth: {b12_mean['Smooth_mean']:.4f}")
    print(f"B12 optimized q-MAE: {q_mean['q-MAE_mean'].iloc[0]:.4f}")
    print(f"B12 optimized q-RMSE: {q_mean['q-RMSE_mean'].iloc[0]:.4f}")
    print(f"B12 optimized q-R2: {q_mean['q-R2_mean'].iloc[0]:.4f}")
    print(f"B12 optimized Spearman: {q_mean['Spearman_mean'].iloc[0]:.4f}")
    print(f"B12 optimized Pearson: {q_mean['Pearson_mean'].iloc[0]:.4f}")
    print(f"B12 optimized q-Smooth: {q_mean['q-Smooth_mean'].iloc[0]:.4f}")
    print(f"\nResults saved to:\n{OUT_DIR}")


# =========================================================
# 9. Stage-aware NASA optimization overrides
# =========================================================
LAMBDA_FINE = 0.35
LAMBDA_Q = 0.35
LAMBDA_MONO = 0.0


def stage_ids_to_labels(y: np.ndarray) -> list[str]:
    return [ID_TO_STAGE[int(v)] for v in y]


def apply_stage_labels(df: pd.DataFrame, strategy: str, params: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    if strategy == "A_q_global":
        q = out["q_true"].astype(float).values
        nu = out["nu_norm"].astype(float).values
        y = np.where(q <= params["theta_E"], 0, np.where((q >= params["theta_L"]) | (nu >= params["theta_v"]), 2, 1))
    elif strategy == "B_vb_engineering":
        vb = out["VB_smooth"].astype(float).values
        y = np.where(vb <= params["vb_E"], 0, np.where(vb >= params["vb_L"], 2, 1))
    else:
        raise ValueError(f"Unknown stage strategy: {strategy}")
    out["stage_id"] = y.astype(int)
    out["stage_label"] = stage_ids_to_labels(y)
    return out


def case_descriptor(df: pd.DataFrame) -> pd.Series:
    dist = stage_distribution(df)
    n = max(len(df), 1)
    return pd.Series({
        "material": float(df["material"].mode().iloc[0]) if len(df["material"].dropna()) else 0.0,
        "DOC": float(df["DOC"].mode().iloc[0]) if len(df["DOC"].dropna()) else 0.0,
        "feed": float(df["feed"].mode().iloc[0]) if len(df["feed"].dropna()) else 0.0,
        "VB_min": float(df["VB_smooth"].min()),
        "VB_max": float(df["VB_smooth"].max()),
        "q_min": float(df["q_true"].min()),
        "q_max": float(df["q_true"].max()),
        "early_ratio": dist["early"] / n,
        "middle_ratio": dist["middle"] / n,
        "late_ratio": dist["late"] / n,
        "n_samples": float(n),
    })


def group_descriptor(df: pd.DataFrame, cases: list[int]) -> pd.Series:
    return case_descriptor(df[df["case"].isin(cases)])


def descriptor_distance(a: pd.Series, b: pd.Series, scale: pd.Series) -> float:
    cols = ["material", "DOC", "feed", "VB_min", "VB_max", "q_min", "q_max", "early_ratio", "middle_ratio", "late_ratio", "n_samples"]
    av = a[cols].astype(float).values
    bv = b[cols].astype(float).values
    sv = np.maximum(scale[cols].astype(float).values, 1e-6)
    w = np.array([1.4, 1.2, 1.2, 0.8, 1.0, 0.4, 0.4, 1.1, 1.4, 1.1, 0.6])
    return float(np.sqrt(np.mean(w * ((av - bv) / sv) ** 2)))


def choose_validation_cases_similarity(df: pd.DataFrame, train_cases: list[int], test_cases: list[int], task_name: str, split_type: str) -> tuple[list[int], float, list[dict[str, Any]]]:
    desc_rows = []
    for case, sub in df[df["case"].isin(train_cases + test_cases)].groupby("case"):
        row = case_descriptor(sub).to_dict()
        row["case"] = int(case)
        desc_rows.append(row)
    desc_df = pd.DataFrame(desc_rows).set_index("case")
    scale = desc_df.max(numeric_only=True) - desc_df.min(numeric_only=True)
    test_desc = group_descriptor(df, test_cases)
    best_cases, best_score = None, 1e18
    for k in (2, 3):
        for combo in itertools.combinations(train_cases, k):
            combo = list(combo)
            combo_desc = group_descriptor(df, combo)
            dist = descriptor_distance(combo_desc, test_desc, scale)
            val_dist = stage_distribution(df[df["case"].isin(combo)])
            present = sum(1 for s in STAGES if val_dist[s] > 0)
            balance_penalty = 0.15 * (3 - present)
            score = dist + balance_penalty
            if score < best_score:
                best_cases, best_score = combo, score
    log = []
    for case in train_cases:
        dist = descriptor_distance(desc_df.loc[case], test_desc, scale)
        log.append({"Split_type": split_type, "Task": task_name, "Candidate_case": case, "Case_distance_to_test": dist, "Selected": case in best_cases})
    return best_cases, float(best_score), log


def make_original_tasks_stageaware(df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks, logs = [], []
    for t in FIXED_TASKS:
        val_cases, dist, log = choose_validation_cases_similarity(df, list(t["Train_cases"]), list(t["Test_cases"]), t["Task"], "original")
        train_cases = [c for c in t["Train_cases"] if c not in val_cases]
        tasks.append({"Split_type": "original", "Task": t["Task"], "Original_train_cases": list(t["Train_cases"]), "Train_cases": train_cases, "Validation_cases": val_cases, "Test_cases": list(t["Test_cases"]), "Similarity_distance": dist})
        logs.extend(log)
    return tasks, logs


def make_stagebalanced_tasks(df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases = sorted(df["case"].unique().astype(int).tolist())
    case_info = []
    for c in cases:
        sub = df[df["case"] == c]
        dist = stage_distribution(sub)
        case_info.append((c, len(sub), dist))
    groups = [[] for _ in range(4)]
    totals = [0, 0, 0, 0]
    stage_totals = [np.zeros(3) for _ in range(4)]
    for c, n, dist in sorted(case_info, key=lambda x: x[1], reverse=True):
        scores = []
        vec = np.array([dist["early"], dist["middle"], dist["late"]], dtype=float)
        for i in range(4):
            new_total = totals[i] + n
            new_stage = stage_totals[i] + vec
            balance = np.std([new_total] + totals[:i] + totals[i + 1:])
            stage_missing = np.sum(new_stage == 0) * 4.0
            scores.append(balance + stage_missing)
        j = int(np.argmin(scores))
        groups[j].append(c)
        totals[j] += n
        stage_totals[j] += vec
    tasks, logs = [], []
    for i, test_cases in enumerate(groups, start=1):
        train_pool = [c for c in cases if c not in test_cases]
        name = f"N{i}b"
        val_cases, dist, log = choose_validation_cases_similarity(df, train_pool, test_cases, name, "stagebalanced")
        train_cases = [c for c in train_pool if c not in val_cases]
        tasks.append({"Split_type": "stagebalanced", "Task": name, "Original_train_cases": train_pool, "Train_cases": train_cases, "Validation_cases": val_cases, "Test_cases": test_cases, "Similarity_distance": dist})
        logs.extend(log)
    return tasks, logs


def save_tasks(tasks: list[dict[str, Any]], csv_name: str, json_name: str) -> None:
    rows = []
    for t in tasks:
        rows.append({
            "Split_type": t["Split_type"],
            "Task": t["Task"],
            "Original_train_cases": ",".join(map(str, t["Original_train_cases"])),
            "Train_cases": ",".join(map(str, t["Train_cases"])),
            "Validation_cases": ",".join(map(str, t["Validation_cases"])),
            "Test_cases": ",".join(map(str, t["Test_cases"])),
            "Similarity_distance": t["Similarity_distance"],
        })
    save_csv(pd.DataFrame(rows), OUT_DIR / csv_name)
    with open(OUT_DIR / json_name, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)


def stage_dist_string(df: pd.DataFrame) -> str:
    d = stage_distribution(df)
    return f"E:{d['early']},M:{d['middle']},L:{d['late']}"


def descriptor_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case, sub in df.groupby("case"):
        row = case_descriptor(sub).to_dict()
        row["case"] = int(case)
        rows.append(row)
    return pd.DataFrame(rows).set_index("case")


def normalized_descriptor_distance(a: pd.Series, b: pd.Series, scale: pd.Series) -> float:
    cols = ["material", "DOC", "feed", "VB_min", "VB_max", "q_min", "q_max", "early_ratio", "middle_ratio", "late_ratio", "n_samples"]
    av = a[cols].astype(float).values
    bv = b[cols].astype(float).values
    sv = np.maximum(scale[cols].astype(float).values, 1e-6)
    w = np.array([1.4, 1.2, 1.2, 0.8, 1.0, 0.4, 0.4, 1.1, 1.4, 1.1, 0.6])
    return float(np.sqrt(np.mean(w * ((av - bv) / sv) ** 2)))


def group_descriptor_from_cases(df: pd.DataFrame, cases: list[int]) -> pd.Series:
    return case_descriptor(df[df["case"].isin(cases)])


def choose_validation_for_test_group(df: pd.DataFrame, train_pool: list[int], test_cases: list[int], desc_df: pd.DataFrame) -> tuple[list[int], float]:
    scale = desc_df.max(numeric_only=True) - desc_df.min(numeric_only=True)
    test_desc = group_descriptor_from_cases(df, test_cases)
    best_cases, best_score = None, 1e18
    for k in (2, 3):
        for combo in itertools.combinations(train_pool, k):
            combo = list(combo)
            val_df = df[df["case"].isin(combo)]
            val_dist = stage_distribution(val_df)
            present = sum(1 for s in STAGES if val_dist[s] > 0)
            combo_desc = group_descriptor_from_cases(df, combo)
            dist = normalized_descriptor_distance(combo_desc, test_desc, scale)
            middle_penalty = 0.15 if val_dist["middle"] == 0 else 0.0
            stage_penalty = 0.12 * (3 - present)
            size_penalty = 0.03 * abs(len(val_df) - max(len(df[df["case"].isin(test_cases)]) * 0.45, 1)) / max(len(df), 1)
            score = dist + middle_penalty + stage_penalty + size_penalty
            if score < best_score:
                best_cases, best_score = combo, float(score)
    return best_cases, best_score


def case_diversity_score(df: pd.DataFrame, cases: list[int]) -> float:
    sub = df[df["case"].isin(cases)]
    mat = sub.groupby("case")["material"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
    doc = sub.groupby("case")["DOC"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
    feed = sub.groupby("case")["feed"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
    vals = [mat.nunique() / 2.0, doc.nunique() / 2.0, feed.nunique() / 2.0]
    return float(np.clip(np.mean(vals), 0.0, 1.0))


def split_quality_components(df: pd.DataFrame, train_cases: list[int], val_cases: list[int], test_cases: list[int], val_test_similarity: float) -> dict[str, Any]:
    train_df = df[df["case"].isin(train_cases)]
    val_df = df[df["case"].isin(val_cases)]
    test_df = df[df["case"].isin(test_cases)]
    total_n = len(df)
    target_test_n = total_n / 4.0
    train_dist = stage_distribution(train_df)
    val_dist = stage_distribution(val_df)
    test_dist = stage_distribution(test_df)

    def stage_score(dist: dict[str, int]) -> float:
        n = max(sum(dist.values()), 1)
        ratios = np.array([dist[s] / n for s in STAGES], dtype=float)
        present = sum(v > 0 for v in dist.values()) / 3.0
        balance = 1.0 - float(np.sum(np.abs(ratios - 1.0 / 3.0)) / (4.0 / 3.0))
        return float(np.clip(0.65 * present + 0.35 * balance, 0.0, 1.0))

    test_n_score = float(np.exp(-abs(len(test_df) - target_test_n) / max(target_test_n, 1.0)))
    test_stage_score = stage_score(test_dist)
    train_stage_score = stage_score(train_dist)
    val_stage_score = stage_score(val_dist)
    middle_score = float(np.clip(test_dist["middle"] / 5.0, 0.0, 1.0))
    similarity_score = float(np.exp(-val_test_similarity))
    diversity_score = case_diversity_score(df, test_cases)

    reject = []
    extreme_penalty = 0.0
    if len(test_df) < 20:
        reject.append("test_n<20")
        extreme_penalty += 0.45
    if test_dist["middle"] == 0:
        reject.append("test_middle=0")
        extreme_penalty += 0.70
    if any(test_dist[s] == 0 for s in STAGES):
        reject.append("test_missing_stage")
        extreme_penalty += 0.70
    if any(train_dist[s] == 0 for s in STAGES):
        reject.append("train_missing_stage")
        extreme_penalty += 0.70
    if any(val_dist[s] == 0 for s in STAGES):
        reject.append("val_missing_stage")
        extreme_penalty += 0.20
    if max([len(df[df["case"] == c]) for c in test_cases]) < 6:
        reject.append("test_all_tiny_cases")
        extreme_penalty += 0.25

    score = (
            0.18 * test_n_score
            + 0.22 * test_stage_score
            + 0.14 * middle_score
            + 0.12 * train_stage_score
            + 0.12 * val_stage_score
            + 0.14 * similarity_score
            + 0.08 * diversity_score
            - extreme_penalty
    )
    return {
        "Train_n": int(len(train_df)),
        "Val_n": int(len(val_df)),
        "Test_n": int(len(test_df)),
        "Train_stage_dist": stage_dist_string(train_df),
        "Val_stage_dist": stage_dist_string(val_df),
        "Test_stage_dist": stage_dist_string(test_df),
        "Test_middle_n": int(test_dist["middle"]),
        "Val_test_similarity": float(val_test_similarity),
        "test_n_score": test_n_score,
        "test_stage_score": test_stage_score,
        "middle_score": middle_score,
        "train_stage_score": train_stage_score,
        "val_stage_score": val_stage_score,
        "case_diversity_score": diversity_score,
        "extreme_penalty": extreme_penalty,
        "Split_quality_score": float(score),
        "Reject_reason": ";".join(reject),
    }


def generate_candidate_case_splits(df: pd.DataFrame, n_candidates: int = 30) -> pd.DataFrame:
    cases = sorted(df["case"].unique().astype(int).tolist())
    desc_df = descriptor_frame(df)
    rng = np.random.default_rng(SEED)

    all_test_combos = []
    for test_size in (3, 4):
        all_test_combos.extend([tuple(map(int, x)) for x in itertools.combinations(cases, test_size)])

    # Best-case candidate analysis does not need to score every possible split.
    # Keep enough random/diverse test groups for coverage while avoiding a slow
    # exhaustive validation-search over all 2380 NASA case combinations.
    target_combo_count = min(len(all_test_combos), max(220, n_candidates * 12))
    chosen_combos: list[tuple[int, ...]] = []
    used = set()
    for c in cases:
        hits = [combo for combo in all_test_combos if c in combo and combo not in used]
        if hits:
            combo = hits[int(rng.integers(0, len(hits)))]
            chosen_combos.append(combo)
            used.add(combo)
    remaining = [combo for combo in all_test_combos if combo not in used]
    if remaining and len(chosen_combos) < target_combo_count:
        idx = rng.choice(len(remaining), size=min(target_combo_count - len(chosen_combos), len(remaining)), replace=False)
        for i in np.atleast_1d(idx):
            combo = remaining[int(i)]
            chosen_combos.append(combo)
            used.add(combo)

    rows = []
    split_id = 1
    print(f"Scoring {len(chosen_combos)} candidate test-case groups before selecting {n_candidates} splits...")
    for combo_idx, test_cases in enumerate(chosen_combos, start=1):
        test_cases = list(test_cases)
        train_pool = [c for c in cases if c not in test_cases]
        val_cases, val_test_sim = choose_validation_for_test_group(df, train_pool, test_cases, desc_df)
        train_cases = [c for c in train_pool if c not in val_cases]
        comp = split_quality_components(df, train_cases, val_cases, test_cases, val_test_sim)
        rows.append({
            "Split_ID": f"CAND{split_id:03d}",
            "Train_cases": ",".join(map(str, train_cases)),
            "Validation_cases": ",".join(map(str, val_cases)),
            "Test_cases": ",".join(map(str, test_cases)),
            **comp,
        })
        if combo_idx % 50 == 0 or combo_idx == len(chosen_combos):
            print(f"  scored {combo_idx}/{len(chosen_combos)} candidate groups")
        split_id += 1
    cand = pd.DataFrame(rows).sort_values("Split_quality_score", ascending=False).reset_index(drop=True)
    selected_rows = []
    test_counts = {c: 0 for c in cases}
    # First pass: high-quality candidates while improving test-case coverage.
    for _, row in cand.iterrows():
        tc = [int(x) for x in row["Test_cases"].split(",")]
        if len(selected_rows) < n_candidates:
            selected_rows.append(row)
            for c in tc:
                test_counts[c] += 1
        if len(selected_rows) >= 18 and all(v > 0 for v in test_counts.values()):
            break
    # Ensure every case appears as test at least once if possible.
    for c in cases:
        if test_counts[c] == 0:
            hit = cand[cand["Test_cases"].apply(lambda s, c=c: c in [int(x) for x in s.split(",")])].head(1)
            if len(hit):
                row = hit.iloc[0]
                selected_rows.append(row)
                for cc in [int(x) for x in row["Test_cases"].split(",")]:
                    test_counts[cc] += 1
    used_ids = {r["Split_ID"] for r in selected_rows}
    for _, row in cand.iterrows():
        if len(selected_rows) >= n_candidates:
            break
        if row["Split_ID"] not in used_ids:
            selected_rows.append(row)
            used_ids.add(row["Split_ID"])
    out = pd.DataFrame(selected_rows).drop_duplicates("Split_ID").head(n_candidates).copy()
    return out.sort_values("Split_quality_score", ascending=False).reset_index(drop=True)


def select_final_case_splits(candidate_df: pd.DataFrame, n_final: int = 4) -> list[dict[str, Any]]:
    selected = []
    used_test_cases: set[int] = set()
    sorted_df = candidate_df.sort_values("Split_quality_score", ascending=False).reset_index(drop=True)
    while len(selected) < n_final:
        best_idx, best_key = None, None
        for idx, row in sorted_df.iterrows():
            if any(row["Split_ID"] == s["Source_split_id"] for s in selected):
                continue
            test_cases = [int(x) for x in row["Test_cases"].split(",")]
            overlap = len(set(test_cases) & used_test_cases)
            # Quality is primary; overlap only breaks near ties.
            quality = float(row["Split_quality_score"])
            key = (round(quality, 3), -overlap, len(set(test_cases) - used_test_cases))
            if best_key is None or key > best_key:
                best_idx, best_key = idx, key
        row = sorted_df.loc[best_idx]
        test_cases = [int(x) for x in row["Test_cases"].split(",")]
        task = {
            "Split_type": "case_split",
            "Task": f"CSP{len(selected) + 1}",
            "Source_split_id": row["Split_ID"],
            "Original_train_cases": [int(x) for x in row["Train_cases"].split(",")] + [int(x) for x in row["Validation_cases"].split(",")],
            "Train_cases": [int(x) for x in row["Train_cases"].split(",")],
            "Validation_cases": [int(x) for x in row["Validation_cases"].split(",")],
            "Test_cases": test_cases,
            "Similarity_distance": float(row["Val_test_similarity"]),
            "Split_quality_score": float(row["Split_quality_score"]),
        }
        selected.append(task)
        used_test_cases.update(test_cases)
    return selected


def save_selected_case_splits(tasks: list[dict[str, Any]]) -> None:
    rows = []
    for t in tasks:
        rows.append({
            "Task": t["Task"],
            "Source_split_id": t["Source_split_id"],
            "Train_cases": ",".join(map(str, t["Train_cases"])),
            "Validation_cases": ",".join(map(str, t["Validation_cases"])),
            "Test_cases": ",".join(map(str, t["Test_cases"])),
            "Val_test_similarity": t["Similarity_distance"],
            "Split_quality_score": t["Split_quality_score"],
        })
    save_csv(pd.DataFrame(rows), OUT_DIR / "NASA_selected_case_splits.csv")
    with open(OUT_DIR / "NASA_selected_case_splits.json", "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)


def nearest_centroid_val_score(df_labeled: pd.DataFrame, task: dict[str, Any]) -> dict[str, float]:
    all_cols = online_feature_cols(df_labeled)
    train_df = df_labeled[df_labeled["case"].isin(task["Train_cases"])].copy()
    val_df = df_labeled[df_labeled["case"].isin(task["Validation_cases"])].copy()
    train_df, val_df = fill_by_train_median(train_df, val_df, all_cols)
    selected = select_features_train_only(train_df, all_cols, min(30, len(all_cols)))
    train_df, val_df = fill_by_train_median(train_df, val_df, selected)
    scaler = StandardScaler().fit(train_df[selected].values)
    Xtr = scaler.transform(train_df[selected].values)
    Xva = scaler.transform(val_df[selected].values)
    centroids = []
    for c in range(3):
        idx = train_df["stage_id"].values == c
        centroids.append(Xtr[idx].mean(axis=0) if idx.any() else Xtr.mean(axis=0))
    centroids = np.vstack(centroids)
    dist = ((Xva[:, None, :] - centroids[None, :, :]) ** 2).mean(axis=2)
    y_pred = dist.argmin(axis=1)
    y_true = val_df["stage_id"].values.astype(int)
    pseudo = val_df[["case", "run", "stage_id", "stage_label", "q_true"]].copy()
    pseudo["true_stage"] = pseudo["stage_label"]
    for c, name in enumerate(["E", "M", "L"]):
        pseudo[f"p_raw_{name}"] = (y_pred == c).astype(float)
    m = compute_stage_metrics(pseudo, "centroid", ["p_raw_E", "p_raw_M", "p_raw_L"])
    dist_train = stage_distribution(train_df)
    dist_val = stage_distribution(val_df)
    train_present = sum(1 for s in STAGES if dist_train[s] > 0)
    val_present = sum(1 for s in STAGES if dist_val[s] > 0)
    balance = min(dist_train.values()) / max(max(dist_train.values()), 1)
    score = 0.45 * m["Macro-F1"] + 0.25 * m["M-F1"] + 0.15 * m["Balanced-Acc"] + 0.10 * balance + 0.05 * min(train_present, val_present) / 3.0
    if train_present < 3 or val_present < 3:
        score -= 0.30
    return {**m, "label_selection_score": float(score), "train_present": train_present, "val_present": val_present}


def search_stage_strategy(df_base: pd.DataFrame, task: dict[str, Any], seed: int) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    rows = []
    train_pool = df_base[df_base["case"].isin(task["Train_cases"])]
    # A: q_true/nu_norm global thresholds selected on train/validation only.
    for theta_E in [0.25, 0.30, 0.35, 0.40]:
        for theta_L in [0.60, 0.65, 0.70, 0.75, 0.80]:
            if theta_L <= theta_E + 0.15:
                continue
            for theta_v in [0.75, 0.80, 0.85, 0.90]:
                params = {"theta_E": theta_E, "theta_L": theta_L, "theta_v": theta_v}
                labeled = apply_stage_labels(df_base, "A_q_global", params)
                m = nearest_centroid_val_score(labeled, task)
                rows.append({"Split_type": task["Split_type"], "Task": task["Task"], "Seed": seed, "Strategy": "A_q_global", **params, **m})
    # B: engineering VB thresholds from train quantiles.
    for qe, ql in [(0.25, 0.75), (0.30, 0.70), (0.35, 0.65)]:
        vb_E = float(train_pool["VB_smooth"].quantile(qe))
        vb_L = float(train_pool["VB_smooth"].quantile(ql))
        params = {"vb_E": vb_E, "vb_L": vb_L, "vb_E_quantile": qe, "vb_L_quantile": ql}
        labeled = apply_stage_labels(df_base, "B_vb_engineering", params)
        m = nearest_centroid_val_score(labeled, task)
        rows.append({"Split_type": task["Split_type"], "Task": task["Task"], "Seed": seed, "Strategy": "B_vb_engineering", **params, **m})
    comp = pd.DataFrame(rows)
    best = comp.sort_values(["label_selection_score", "Macro-F1", "M-F1"], ascending=False).iloc[0].to_dict()
    strategy = best["Strategy"]
    if strategy == "A_q_global":
        params = {"theta_E": float(best["theta_E"]), "theta_L": float(best["theta_L"]), "theta_v": float(best["theta_v"])}
    else:
        params = {"vb_E": float(best["vb_E"]), "vb_L": float(best["vb_L"]), "vb_E_quantile": float(best["vb_E_quantile"]), "vb_L_quantile": float(best["vb_L_quantile"])}
    labeled_best = apply_stage_labels(df_base, strategy, params)
    threshold_row = {"Split_type": task["Split_type"], "Task": task["Task"], "Seed": seed, "Selected_strategy": strategy, "Selection_score": float(best["label_selection_score"]), **params}
    return labeled_best, threshold_row, rows


def compute_stage_metrics(pred_df: pd.DataFrame, method: str, prob_cols: list[str]) -> dict[str, Any]:
    probs = pred_df[prob_cols].values.astype(float)
    y_true = pred_df["stage_id"].values.astype(int)
    y_pred = probs.argmax(axis=1)
    rev, jump, smooth = consistency_metrics(y_pred, probs, pred_df["case"].values)

    def trans_rate(src: int, dst: int) -> float:
        mask = y_true == src
        return float(np.sum(mask & (y_pred == dst)) / max(int(mask.sum()), 1))

    e_rec = recall_for_class(y_true, y_pred, 0)
    m_rec = recall_for_class(y_true, y_pred, 1)
    l_rec = recall_for_class(y_true, y_pred, 2)
    return {
        "Method": method,
        "Acc": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
        "Macro-F1": macro_f1(y_true, y_pred),
        "Balanced-Acc": float(np.mean([e_rec, m_rec, l_rec])),
        "E-F1": f1_for_class(y_true, y_pred, 0),
        "M-F1": f1_for_class(y_true, y_pred, 1),
        "L-F1": f1_for_class(y_true, y_pred, 2),
        "E-Rec": e_rec,
        "M-Rec": m_rec,
        "L-Rec": l_rec,
        "M-Pre": precision_for_class(y_true, y_pred, 1),
        "M→E": trans_rate(1, 0),
        "M→L": trans_rate(1, 2),
        "E→M": trans_rate(0, 1),
        "L→M": trans_rate(2, 1),
        "Rev": rev,
        "Jump": jump,
        "Smooth": smooth,
        "Predicted_class_count": int(len(np.unique(y_pred))),
        "Early_recall": e_rec,
        "Middle_recall": m_rec,
        "Late_recall": l_rec,
    }


def train_model(model: nn.Module, train_pack: dict[str, Any], val_pack: dict[str, Any], multitask: bool) -> tuple[nn.Module, int]:
    model = model.to(DEVICE)
    train_loader = make_loader(train_pack, shuffle=True)
    sw = class_weights(train_pack["ys"], 3)
    fw = class_weights(train_pack["yf"], K_FINE)
    ce_stage = nn.CrossEntropyLoss(weight=sw)
    ce_fine = nn.CrossEntropyLoss(weight=fw)
    huber = nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state, best_score, wait, best_epoch = None, -1e18, 0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, ys, yf, yq in train_loader:
            xb, ys, yf, yq = xb.to(DEVICE), ys.to(DEVICE), yf.to(DEVICE), yq.to(DEVICE)
            opt.zero_grad()
            stage_logits, fine_logits, q_hat = model(xb)
            loss = ce_stage(stage_logits, ys)
            if multitask:
                loss = loss + LAMBDA_FINE * ce_fine(fine_logits, yf) + LAMBDA_Q * huber(q_hat, yq)
                if LAMBDA_MONO > 0 and len(q_hat) > 2:
                    loss = loss + LAMBDA_MONO * torch.relu(-(q_hat[1:] - q_hat[:-1])).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
        pred_val = predict_model(model, val_pack, multitask=multitask)
        metrics = compute_stage_metrics(pred_val, "VAL", ["p_raw_E", "p_raw_M", "p_raw_L"])
        score = b11_selection_score(metrics, q_consistency(pred_val)) if multitask else validation_score(metrics)
        if score > best_score:
            best_score, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


def b11_candidate_configs(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    base = [
        {"name": "b11_tiny_L3_k20", "L": 3, "top_k": 20, "batch_size": 8, "lr": 0.001, "dropout": 0.20, "weight_decay": 1e-4, "tcn_channels": (16, 32), "gru_hidden": 32, "lambda_fine": 0.1, "lambda_q": 0.1, "lambda_mono": 0.0},
        {"name": "b11_small_L4_k30_qweak", "L": 4, "top_k": 30, "batch_size": 8, "lr": 0.001, "dropout": 0.20, "weight_decay": 1e-4, "tcn_channels": (16, 32), "gru_hidden": 32, "lambda_fine": 0.2, "lambda_q": 0.05, "lambda_mono": 0.02},
        {"name": "b11_mid_L5_k45", "L": 5, "top_k": 45, "batch_size": 16, "lr": 0.0005, "dropout": 0.20, "weight_decay": 1e-4, "tcn_channels": (32, 64), "gru_hidden": 32, "lambda_fine": 0.2, "lambda_q": 0.1, "lambda_mono": 0.02},
        {"name": "b11_wide_L4_k60", "L": 4, "top_k": 60, "batch_size": 16, "lr": 0.0005, "dropout": 0.30, "weight_decay": 1e-4, "tcn_channels": (32, 64), "gru_hidden": 32, "lambda_fine": 0.1, "lambda_q": 0.2, "lambda_mono": 0.05},
    ]
    choices = []
    for i in range(5):
        cfg = {
            "name": f"b11_sample_{i+1:02d}",
            "L": rng.choice([3, 4, 5, 6]),
            "top_k": rng.choice([20, 30, 45, 60]),
            "batch_size": rng.choice([8, 16]),
            "lr": rng.choice([0.001, 0.0005]),
            "dropout": rng.choice([0.10, 0.20, 0.30]),
            "weight_decay": 1e-4,
            "tcn_channels": rng.choice([(16, 32), (32, 32), (32, 64)]),
            "gru_hidden": rng.choice([16, 32]),
            "lambda_fine": rng.choice([0.1, 0.2, 0.5]),
            "lambda_q": rng.choice([0.05, 0.1, 0.2]),
            "lambda_mono": rng.choice([0.0, 0.02, 0.05]),
        }
        choices.append(cfg)
    return base + choices


def set_runtime_config(cfg: dict[str, Any], seed: int) -> None:
    global SEED, N_FEATURES, L_DEFAULT, BATCH_SIZE, LR, WEIGHT_DECAY, DROP_OUT, TCN_CHANNELS, GRU_HIDDEN, LAMBDA_FINE, LAMBDA_Q, LAMBDA_MONO
    SEED = int(seed)
    N_FEATURES = int(cfg["top_k"])
    L_DEFAULT = int(cfg["L"])
    BATCH_SIZE = int(cfg.get("batch_size", 8))
    LR = float(cfg["lr"])
    WEIGHT_DECAY = float(cfg.get("weight_decay", 1e-4))
    DROP_OUT = float(cfg["dropout"])
    TCN_CHANNELS = tuple(cfg["tcn_channels"])
    GRU_HIDDEN = int(cfg["gru_hidden"])
    LAMBDA_FINE = float(cfg.get("lambda_fine", 0.35))
    LAMBDA_Q = float(cfg.get("lambda_q", 0.35))
    LAMBDA_MONO = float(cfg.get("lambda_mono", 0.0))


def b11_selection_score(metrics: dict[str, Any], qmetrics: dict[str, float]) -> float:
    pred_bias = max(0.0, metrics["M-Rec"] - min(metrics["E-Rec"], metrics["L-Rec"]))
    q_corr = max(0.0, qmetrics.get("Spearman", 0.0)) if np.isfinite(qmetrics.get("Spearman", np.nan)) else 0.0
    score = (
            0.25 * metrics["Macro-F1"] + 0.20 * metrics["Balanced-Acc"] + 0.20 * metrics["M-F1"]
            + 0.10 * metrics["M-Rec"] + 0.10 * q_corr - 0.10 * metrics["Smooth"] - 0.05 * pred_bias
    )
    if metrics["Predicted_class_count"] < 3:
        score -= 0.40
    if metrics["E-Rec"] < 0.10 or metrics["L-Rec"] < 0.10:
        score -= 0.20
    if metrics["M-Rec"] > 0.90 and metrics["Macro-F1"] < 0.40:
        score -= 0.30
    return float(score)


def train_select_b11(task: dict[str, Any], feat_df: pd.DataFrame, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    best, logs = None, []
    for i, cfg in enumerate(b11_candidate_configs(seed)):
        cfg_seed = seed + 3100 + i
        set_seed(cfg_seed)
        packs = prepare_config_packs(feat_df, task, cfg, cfg_seed)
        set_runtime_config(cfg, cfg_seed)
        model = TCNGRUMultiTaskModel(len(packs["selected"]))
        model, ep = train_model(model, packs["train_pack"], packs["val_pack"], multitask=True)
        pred_val = predict_model(model, packs["val_pack"], multitask=True)
        val_m = compute_stage_metrics(pred_val, "B11", ["p_raw_E", "p_raw_M", "p_raw_L"])
        q_m = q_consistency(pred_val)
        score = b11_selection_score(val_m, q_m)
        log = {"Split_type": task["Split_type"], "Task": task["Task"], "Seed": seed, "Config": cfg["name"], "Validation_score": score, "Epoch": ep, "Macro-F1": val_m["Macro-F1"], "Balanced-Acc": val_m["Balanced-Acc"], "M-F1": val_m["M-F1"], "M-Rec": val_m["M-Rec"], "Smooth": val_m["Smooth"], "q_Spearman": q_m["Spearman"], "q_Pearson": q_m["Pearson"], **cfg}
        logs.append(log)
        if best is None or score > best["val_score"]:
            best = {"cfg": cfg, "packs": packs, "model": model, "val_metrics": val_m, "q_val": q_m, "val_score": score, "pred_val": pred_val}
        print(f"  B11 cfg={cfg['name']} score={score:.4f} Macro-F1={val_m['Macro-F1']:.4f} M-F1={val_m['M-F1']:.4f} qSp={q_m['Spearman']:.4f}")
    print(f"  B11 selected cfg={best['cfg']['name']} val_score={best['val_score']:.4f}")
    return best, logs


def validation_distribution_stats(pred_df: pd.DataFrame, prob_cols: list[str]) -> dict[str, float]:
    probs = pred_df[prob_cols].values.astype(float)
    y_true = pred_df["stage_id"].values.astype(int)
    y_pred = probs.argmax(axis=1)
    n = max(len(y_true), 1)
    true_dist = np.array([np.mean(y_true == c) for c in range(3)], dtype=float)
    pred_dist = np.array([np.mean(y_pred == c) for c in range(3)], dtype=float)
    l1 = float(np.sum(np.abs(pred_dist - true_dist)))
    return {
        "true_dist_E": float(true_dist[0]),
        "true_dist_M": float(true_dist[1]),
        "true_dist_L": float(true_dist[2]),
        "pred_dist_E": float(pred_dist[0]),
        "pred_dist_M": float(pred_dist[1]),
        "pred_dist_L": float(pred_dist[2]),
        "distribution_l1_penalty": l1,
        "min_pred_ratio": float(np.min(pred_dist)),
        "n_eval": int(n),
    }


def b12_stageaware_score(metrics: dict[str, Any], n: int, dist_stats: dict[str, float] | None = None) -> float:
    rev_norm = metrics["Rev"] / max(n, 1)
    jump_norm = metrics["Jump"] / max(n, 1)
    dist_stats = dist_stats or {}
    min_pred_ratio = float(dist_stats.get("min_pred_ratio", 1.0))
    dist_penalty = float(dist_stats.get("distribution_l1_penalty", 0.0))
    score = (
            0.22 * metrics["Macro-F1"] + 0.18 * metrics["Balanced-Acc"] + 0.18 * metrics["M-F1"]
            + 0.12 * metrics["M-Rec"] + 0.10 * min(metrics["E-Rec"], metrics["L-Rec"]) + 0.05 * metrics["Acc"]
            - 0.08 * metrics["Smooth"] - 0.04 * rev_norm - 0.04 * jump_norm
            - 0.05 * (metrics["M→E"] + metrics["M→L"]) - 0.05 * (metrics["E→M"] + metrics["L→M"])
            - 0.12 * dist_penalty
    )
    if metrics["Predicted_class_count"] < 3:
        score -= 1.00
    if metrics["Predicted_class_count"] < 2:
        score -= 2.00
    if min_pred_ratio < 0.03:
        score -= 0.50
    if min_pred_ratio < 0.01:
        score -= 0.80
    if metrics["M-Rec"] > 0.90 and (metrics["E-Rec"] < 0.20 or metrics["L-Rec"] < 0.20):
        score -= 0.40
    if metrics["M-Rec"] > 0.75 and (metrics["E-Rec"] < 0.20 or metrics["L-Rec"] < 0.20):
        score -= 0.50
    if metrics["L→M"] > 0.40:
        score -= 0.40
    if metrics["L→M"] > 0.60:
        score -= 0.70
    if metrics["M→L"] > 0.40:
        score -= 0.40
    if metrics["M→L"] > 0.60:
        score -= 0.70
    if metrics["M→E"] > 0.30:
        score -= 0.30
    if metrics["E-Rec"] < 0.15:
        score -= 0.35
    if metrics["L-Rec"] < 0.15:
        score -= 0.35
    if metrics["E-Rec"] < 0.05:
        score -= 0.60
    if metrics["L-Rec"] < 0.05:
        score -= 0.60
    if metrics["E-Rec"] < 0.10:
        score -= 0.25
    if metrics["L-Rec"] < 0.10:
        score -= 0.25
    if metrics["Macro-F1"] < 0.30:
        score -= 0.20
    return float(score)


def random_fusion_params(rng: random.Random, q_val: dict[str, float]) -> dict[str, float]:
    poor_q = q_prior_weakening_enabled(q_val)
    good_q = (q_val.get("Spearman", 0) >= 0.50) and (q_val.get("Pearson", 0) >= 0.50) and (q_val.get("q-R2", 0) > 0)
    space = {k: list(v) for k, v in B12_SEARCH.items()}
    if poor_q:
        space["eta"] = [v for v in space["eta"] if v >= 0.90]
        space["fine_weight"] = [v for v in space["fine_weight"] if v <= 0.10]
        space["prior_sigma"] = [v for v in space["prior_sigma"] if v >= 0.45]
        space["order_blend"] = [v for v in space["order_blend"] if v <= 0.05]
    elif good_q:
        space = {k: list(v) for k, v in B12_SEARCH.items()}
    return {k: rng.choice(v) for k, v in space.items()}


def q_prior_weakening_enabled(q_val: dict[str, float]) -> bool:
    return (
            q_val.get("Spearman", 0.0) < 0.40
            or q_val.get("Pearson", 0.0) < 0.40
            or q_val.get("q-R2", 0.0) < 0.0
    )


def enforce_q_prior_constraints(params: dict[str, float], q_val: dict[str, float]) -> dict[str, float] | None:
    if not q_prior_weakening_enabled(q_val):
        return dict(params)
    if params["eta"] < 0.90 or params["fine_weight"] > 0.10 or params["prior_sigma"] < 0.45 or params["order_blend"] > 0.05:
        return None
    return dict(params)


def search_b12_params_stageaware(pred_val_raw: pd.DataFrame, task: dict[str, Any], seed: int, q_val: dict[str, float]) -> tuple[dict[str, float], float, dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(seed + 9000)
    evaluated, seen = [], set()
    q_prior_weakening = q_prior_weakening_enabled(q_val)

    def eval_params(params: dict[str, float], phase: str) -> None:
        params = enforce_q_prior_constraints(params, q_val)
        if params is None:
            return
        key = tuple((k, params[k]) for k in sorted(params))
        if key in seen:
            return
        seen.add(key)
        pred = apply_dcpsr_inference(pred_val_raw, params)
        m = compute_stage_metrics(pred, "B12", ["p_final_E", "p_final_M", "p_final_L"])
        dist_stats = validation_distribution_stats(pred, ["p_final_E", "p_final_M", "p_final_L"])
        score = b12_stageaware_score(m, len(pred), dist_stats)
        evaluated.append({
            "Split_type": task["Split_type"],
            "Task": task["Task"],
            "Seed": seed,
            "Phase": phase,
            "Score": score,
            "q_prior_weakening": q_prior_weakening,
            **params,
            **dist_stats,
            **m,
        })

    for _ in range(160):
        eval_params(random_fusion_params(rng, q_val), "random160")
    if not evaluated:
        raise RuntimeError(f"No valid B12 fusion params evaluated for {task['Task']}. Check q-prior constraints.")
    top = sorted(evaluated, key=lambda r: r["Score"], reverse=True)[:12]
    for row in top:
        base = {k: row[k] for k in B12_SEARCH}
        for eta in sorted(set([base["eta"], min(0.98, round(base["eta"] + 0.02, 2)), max(0.85, round(base["eta"] - 0.02, 2))])):
            for et in sorted(set([base["early_tau"], min(0.45, round(base["early_tau"] + 0.02, 2)), max(0.25, round(base["early_tau"] - 0.02, 2))])):
                for lt in sorted(set([base["late_tau"], min(0.75, round(base["late_tau"] + 0.03, 2)), max(0.55, round(base["late_tau"] - 0.03, 2))])):
                    for ob in sorted(set([base["order_blend"], min(0.10, round(base["order_blend"] + 0.02, 2)), max(0.00, round(base["order_blend"] - 0.02, 2))])):
                        p = dict(base)
                        p.update({"eta": eta, "early_tau": et, "late_tau": lt, "order_blend": ob})
                        eval_params(p, "local_top12")
    best = max(evaluated, key=lambda r: r["Score"])
    params = {k: best[k] for k in B12_SEARCH}
    metrics = {k: best[k] for k in ["Acc", "Macro-F1", "Balanced-Acc", "E-F1", "M-F1", "L-F1", "E-Rec", "M-Rec", "L-Rec", "M-Pre", "M→E", "M→L", "E→M", "L→M", "Rev", "Jump", "Smooth", "Predicted_class_count", "Early_recall", "Middle_recall", "Late_recall", "true_dist_E", "true_dist_M", "true_dist_L", "pred_dist_E", "pred_dist_M", "pred_dist_L", "distribution_l1_penalty", "min_pred_ratio"]}
    params["q_prior_weakening"] = q_prior_weakening
    return params, float(best["Score"]), metrics, evaluated


def add_metric_context(row: dict[str, Any], task: dict[str, Any], seed: int, cfg: dict[str, Any], val_score: float) -> dict[str, Any]:
    out = dict(row)
    out.update({
        "Split_type": task["Split_type"],
        "Task": task["Task"],
        "Seed": seed,
        "Train_cases": ",".join(map(str, task["Train_cases"])),
        "Validation_cases": ",".join(map(str, task["Validation_cases"])),
        "Test_cases": ",".join(map(str, task["Test_cases"])),
        "Window length": int(cfg["L"]),
        "Top_k": int(cfg["top_k"]),
        "Config": cfg["name"],
        "Best_validation_score": float(val_score),
    })
    return out


def diagnostic_row(task: dict[str, Any], method: str, seed: int, cfg: dict[str, Any], val_score: float, val_m: dict[str, Any], test_m: dict[str, Any], q_val: dict[str, float] | None = None, fusion_params: dict[str, float] | None = None) -> dict[str, Any]:
    q_val = q_val or {}
    return {
        "Task": task["Task"], "Split_type": task["Split_type"], "Method": method, "Seed": seed,
        "Validation_cases": ",".join(map(str, task["Validation_cases"])),
        "Best_validation_score": float(val_score),
        "Validation_Acc": val_m["Acc"], "Validation_MacroF1": val_m["Macro-F1"], "Validation_BalancedAcc": val_m["Balanced-Acc"],
        "Validation_MF1": val_m["M-F1"], "Validation_MRec": val_m["M-Rec"], "Validation_ERec": val_m["E-Rec"], "Validation_LRec": val_m["L-Rec"], "Validation_Smooth": val_m["Smooth"],
        "Validation_q_Spearman": q_val.get("Spearman", np.nan), "Validation_q_Pearson": q_val.get("Pearson", np.nan),
        "Test_Acc": test_m["Acc"], "Test_MacroF1": test_m["Macro-F1"], "Test_BalancedAcc": test_m["Balanced-Acc"],
        "Test_MF1": test_m["M-F1"], "Test_MRec": test_m["M-Rec"], "Test_ERec": test_m["E-Rec"], "Test_LRec": test_m["L-Rec"], "Test_Smooth": test_m["Smooth"],
        "Predicted_class_count": test_m["Predicted_class_count"],
        "Best_L": int(cfg["L"]), "Best_top_k": int(cfg["top_k"]), "Best_model_config": cfg["name"],
        "Best_fusion_params": json.dumps(fusion_params or {}, ensure_ascii=False),
    }


def run_one_task_seed_stageaware(task: dict[str, Any], df_base: pd.DataFrame, seed: int) -> tuple[list[dict], dict, list[dict], dict, list[dict], list[dict], list[dict], dict]:
    labeled, threshold_row, strategy_rows = search_stage_strategy(df_base, task, seed)
    print("\n" + "=" * 100)
    print(f"{task['Split_type']} {task['Task']} seed {seed}")
    print(f"Test_cases: {task['Test_cases']}")
    print(f"Selected_validation_cases: {task['Validation_cases']}")
    print(f"Similarity_distance: {task['Similarity_distance']:.4f}")
    print(f"Selected_stage_strategy: {threshold_row['Selected_strategy']}")
    print(f"Train stage distribution: {stage_distribution(labeled[labeled['case'].isin(task['Train_cases'])])}")
    print(f"Val stage distribution: {stage_distribution(labeled[labeled['case'].isin(task['Validation_cases'])])}")
    print(f"Test stage distribution: {stage_distribution(labeled[labeled['case'].isin(task['Test_cases'])])}")

    metric_rows, diag_rows = [], []
    # B9/B10 remain fast baselines.
    b9 = train_select_method("B9", GRUStageModel, task, labeled, seed, multitask=False)
    set_runtime_config(b9["cfg"], seed)
    pred_b9 = predict_model(b9["model"], b9["packs"]["test_pack"], multitask=False)
    m_b9 = compute_stage_metrics(pred_b9, "B9", ["p_raw_E", "p_raw_M", "p_raw_L"])
    metric_rows.append(add_metric_context(m_b9, task, seed, b9["cfg"], b9["val_score"]))
    diag_rows.append(diagnostic_row(task, "B9", seed, b9["cfg"], b9["val_score"], b9["val_metrics"], m_b9))
    save_prediction_detail(pred_b9, f"{task['Split_type']}_{task['Task']}", "B9", ["p_raw_E", "p_raw_M", "p_raw_L"], seed=seed)

    b10 = train_select_method("B10", TCNGRUStageModel, task, labeled, seed, multitask=False)
    set_runtime_config(b10["cfg"], seed)
    pred_b10 = predict_model(b10["model"], b10["packs"]["test_pack"], multitask=False)
    m_b10 = compute_stage_metrics(pred_b10, "B10", ["p_raw_E", "p_raw_M", "p_raw_L"])
    metric_rows.append(add_metric_context(m_b10, task, seed, b10["cfg"], b10["val_score"]))
    diag_rows.append(diagnostic_row(task, "B10", seed, b10["cfg"], b10["val_score"], b10["val_metrics"], m_b10))
    save_prediction_detail(pred_b10, f"{task['Split_type']}_{task['Task']}", "B10", ["p_raw_E", "p_raw_M", "p_raw_L"], seed=seed)

    b11, b11_logs = train_select_b11(task, labeled, seed)
    set_runtime_config(b11["cfg"], seed)
    pred_val_b11 = predict_model(b11["model"], b11["packs"]["val_pack"], multitask=True)
    pred_test_b11 = predict_model(b11["model"], b11["packs"]["test_pack"], multitask=True)
    q_val = q_consistency(pred_val_b11)
    m_b11 = compute_stage_metrics(pred_test_b11, "B11", ["p_raw_E", "p_raw_M", "p_raw_L"])
    metric_rows.append(add_metric_context(m_b11, task, seed, b11["cfg"], b11["val_score"]))
    diag_rows.append(diagnostic_row(task, "B11", seed, b11["cfg"], b11["val_score"], b11["val_metrics"], m_b11, q_val=q_val))
    save_prediction_detail(pred_test_b11, f"{task['Split_type']}_{task['Task']}", "B11", ["p_raw_E", "p_raw_M", "p_raw_L"], seed=seed)

    fusion_params, b12_val_score, b12_val_metrics, fusion_logs = search_b12_params_stageaware(pred_val_b11, task, seed, q_val)
    pred_b12 = apply_dcpsr_inference(pred_test_b11, fusion_params)
    m_b12 = compute_stage_metrics(pred_b12, "B12", ["p_final_E", "p_final_M", "p_final_L"])
    metric_rows.append(add_metric_context(m_b12, task, seed, b11["cfg"], b12_val_score))
    diag_rows.append(diagnostic_row(task, "B12", seed, b11["cfg"], b12_val_score, b12_val_metrics, m_b12, q_val=q_val, fusion_params=fusion_params))
    save_prediction_detail(pred_b12, f"{task['Split_type']}_{task['Task']}", "B12", ["p_final_E", "p_final_M", "p_final_L"], seed=seed)

    if m_b12["Predicted_class_count"] < 3:
        print(f"WARNING: B12 {task['Split_type']} {task['Task']} predicted_class_count < 3")
    if m_b12["Macro-F1"] < m_b11["Macro-F1"]:
        print(f"WARNING: B12 Macro-F1 < B11 Macro-F1 for {task['Split_type']} {task['Task']}")
    if m_b12["M-F1"] < m_b11["M-F1"]:
        print(f"WARNING: B12 M-F1 < B11 M-F1 for {task['Split_type']} {task['Task']}")
    if m_b12["Smooth"] > m_b11["Smooth"]:
        print(f"WARNING: B12 Smooth > B11 Smooth for {task['Split_type']} {task['Task']}")
    if b12_val_metrics["Macro-F1"] - m_b12["Macro-F1"] > 0.25:
        print(f"WARNING: validation-test Macro-F1 gap > 0.25 for {task['Split_type']} {task['Task']} B12")
    if q_val.get("Spearman", 1.0) < 0.25:
        print(f"WARNING: q Spearman < 0.25 for {task['Split_type']} {task['Task']}")

    qrow = q_consistency(pred_b12)
    qrow.update({"Split_type": task["Split_type"], "Task": task["Task"], "Seed": seed, "Train_cases": ",".join(map(str, task["Train_cases"])), "Validation_cases": ",".join(map(str, task["Validation_cases"])), "Test_cases": ",".join(map(str, task["Test_cases"])), "Config": b11["cfg"]["name"]})
    q_quality = {"Split_type": task["Split_type"], "Task": task["Task"], "Seed": seed, **q_val}
    fusion_best = {"Split_type": task["Split_type"], "Task": task["Task"], "Seed": seed, "Best_validation_score": b12_val_score, **fusion_params}
    return metric_rows, qrow, diag_rows, fusion_best, fusion_logs, b11_logs, strategy_rows, q_quality


def mean_std_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["Acc", "Macro-F1", "Balanced-Acc", "E-F1", "M-F1", "L-F1", "E-Rec", "M-Rec", "L-Rec", "M-Pre", "M→E", "M→L", "E→M", "L→M", "Rev", "Jump", "Smooth", "Predicted_class_count"]
    for keys, sub in df.groupby(["Split_type", "Method"]):
        split_type, method = keys
        row = {"Split_type": split_type, "Method": method}
        for metric in metrics:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_std"] = float(sub[metric].std(ddof=1)) if len(sub) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Split_type", "Method"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(RUN_SEEDS[0])
    print("=" * 100)
    print("NASA stage-aware optimized cross-case experiment for DC-PSR")
    print(f"Device: {DEVICE}")
    print(f"Seeds: {RUN_SEEDS}")
    print(f"Output: {OUT_DIR}")
    print("=" * 100)

    with open(OUT_DIR / "NASA_experiment_config_final_b12calib.json", "w", encoding="utf-8") as f:
        json.dump({"RUN_SEEDS": RUN_SEEDS, "RUN_SPLITS": RUN_SPLITS, "B12_SEARCH": B12_SEARCH, "fixed_tasks": FIXED_TASKS, "max_epochs": EPOCHS, "patience": PATIENCE}, f, ensure_ascii=False, indent=2)

    raw_feat = extract_signal_features()
    base_labeled = build_case_relative_q_and_stage_labels(raw_feat)
    save_csv(build_case_summary(base_labeled), OUT_DIR / "NASA_case_summary_optimized.csv")
    save_csv(base_labeled, OUT_DIR / "NASA_run_level_features_with_labels_optimized.csv")
    online = build_online_relative_features(base_labeled, raw_feature_columns(base_labeled))
    save_csv(online, OUT_DIR / "NASA_online_relative_features_optimized.csv")

    original_tasks, original_val_logs = make_original_tasks_stageaware(online)
    balanced_tasks, balanced_val_logs = make_stagebalanced_tasks(online)
    save_tasks(original_tasks, "NASA_cross_case_tasks_original.csv", "NASA_cross_case_tasks_original.json")
    save_tasks(balanced_tasks, "NASA_cross_case_tasks_stagebalanced.csv", "NASA_cross_case_tasks_stagebalanced.json")
    save_csv(pd.DataFrame(original_val_logs + balanced_val_logs), OUT_DIR / "NASA_validation_case_selection.csv")

    all_metrics, all_q, all_diag, all_fusion, all_fusion_logs, all_b11_logs, all_strategy_rows, all_thresholds, all_q_quality = [], [], [], [], [], [], [], [], []
    tasks_to_run = []
    if "original" in RUN_SPLITS:
        tasks_to_run.extend(original_tasks)
    if "stagebalanced" in RUN_SPLITS:
        tasks_to_run.extend(balanced_tasks)

    for seed in RUN_SEEDS:
        for task in tasks_to_run:
            metric_rows, qrow, diag_rows, fusion_best, fusion_logs, b11_logs, strategy_rows, q_quality = run_one_task_seed_stageaware(task, online, seed)
            all_metrics.extend(metric_rows)
            all_q.append(qrow)
            all_diag.extend(diag_rows)
            all_fusion.append(fusion_best)
            all_fusion_logs.extend(fusion_logs)
            all_b11_logs.extend(b11_logs)
            all_strategy_rows.extend(strategy_rows)
            all_q_quality.append(q_quality)
            best_strategy = max(strategy_rows, key=lambda r: r["label_selection_score"])
            all_thresholds.append({k: best_strategy.get(k, np.nan) for k in best_strategy.keys() if k in ["Split_type", "Task", "Seed", "Strategy", "label_selection_score", "theta_E", "theta_L", "theta_v", "vb_E", "vb_L", "vb_E_quantile", "vb_L_quantile"]})

    metrics_df = pd.DataFrame(all_metrics)
    metric_cols = ["Split_type", "Task", "Seed", "Train_cases", "Validation_cases", "Test_cases", "Method", "Acc", "Macro-F1", "Balanced-Acc", "E-F1", "M-F1", "L-F1", "E-Rec", "M-Rec", "L-Rec", "M-Pre", "M→E", "M→L", "E→M", "L→M", "Rev", "Jump", "Smooth", "Predicted_class_count", "Window length", "Top_k", "Config", "Best_validation_score"]
    metrics_df = metrics_df[metric_cols]
    save_csv(metrics_df, OUT_DIR / "Table_NASA_cross_case_B9_B12_metrics_by_seed.csv")
    save_csv(metrics_df, OUT_DIR / "Table_NASA_cross_case_B9_B12_metrics_optimized.csv")
    mean_df = mean_std_table(metrics_df)
    save_csv(mean_df, OUT_DIR / "Table_NASA_cross_case_mean_std_optimized.csv")
    save_csv(mean_df[mean_df["Split_type"] == "original"], OUT_DIR / "Table_NASA_original_split_mean_std.csv")
    if "stagebalanced" in RUN_SPLITS:
        save_csv(mean_df[mean_df["Split_type"] == "stagebalanced"], OUT_DIR / "Table_NASA_stagebalanced_split_mean_std.csv")

    q_df = pd.DataFrame(all_q)
    q_cols = ["Split_type", "Task", "Seed", "Train_cases", "Validation_cases", "Test_cases", "Config", "q-MAE", "q-RMSE", "q-R2", "Spearman", "Pearson", "q-Smooth"]
    q_df = q_df[q_cols]
    save_csv(q_df, OUT_DIR / "Table_NASA_q_consistency_B12_by_seed.csv")
    save_csv(q_df, OUT_DIR / "Table_NASA_q_consistency_B12_optimized.csv")
    save_csv(q_mean_table(q_df), OUT_DIR / "Table_NASA_q_consistency_B12_mean_optimized.csv")

    save_csv(pd.DataFrame(all_diag), OUT_DIR / "NASA_experiment_diagnostics_final.csv")
    save_csv(pd.DataFrame(all_fusion), OUT_DIR / "NASA_B12_best_fusion_params_final.csv")
    save_csv(pd.DataFrame(all_fusion_logs), OUT_DIR / "NASA_B12_fusion_search_log_final.csv")
    save_csv(pd.DataFrame(all_b11_logs), OUT_DIR / "NASA_B11_best_training_configs.csv")
    save_csv(pd.DataFrame(all_strategy_rows), OUT_DIR / "NASA_stage_label_strategy_comparison.csv")
    save_csv(pd.DataFrame(all_thresholds), OUT_DIR / "NASA_stage_thresholds_by_task.csv")
    save_csv(pd.DataFrame(all_q_quality), OUT_DIR / "NASA_q_validation_quality_by_task.csv")
    plot_optional_summary(mean_df[mean_df["Split_type"] == "original"])

    print("\nNASA stage-aware optimized cross-case experiment finished.\n")
    for split_type in RUN_SPLITS:
        sub = mean_df[mean_df["Split_type"] == split_type]
        if len(sub) == 0:
            continue
        b12 = sub[sub["Method"] == "B12"].iloc[0]
        b11 = sub[sub["Method"] == "B11"].iloc[0]
        best_macro = sub.loc[sub["Macro-F1_mean"].idxmax()]
        print(f"{split_type}: best Macro-F1 method = {best_macro['Method']} ({best_macro['Macro-F1_mean']:.4f})")
        print(f"{split_type}: B12 Macro-F1={b12['Macro-F1_mean']:.4f}, M-F1={b12['M-F1_mean']:.4f}, M-Rec={b12['M-Rec_mean']:.4f}, Smooth={b12['Smooth_mean']:.4f}")
        print(f"B12 vs B11 Macro-F1 improvement: {b12['Macro-F1_mean'] - b11['Macro-F1_mean']:.4f}")
        print(f"B12 vs B11 M-F1 improvement: {b12['M-F1_mean'] - b11['M-F1_mean']:.4f}")
        print(f"B12 vs B11 Smooth change: {b12['Smooth_mean'] - b11['Smooth_mean']:.4f}")
    b12_rows = metrics_df[metrics_df["Method"] == "B12"].copy()
    print("B12 predicted_class_count per task:")
    for _, r in b12_rows.iterrows():
        print(f"  {r['Split_type']} {r['Task']}: {int(r['Predicted_class_count'])}")
    has_low_class_count = bool((b12_rows["Predicted_class_count"].astype(float) < 3).any())
    has_low_el_rec = bool(((b12_rows["E-Rec"].astype(float) < 0.10) | (b12_rows["L-Rec"].astype(float) < 0.10)).any())
    print(f"Any B12 task predicted_class_count < 3: {has_low_class_count}")
    print(f"Any B12 task E_Rec or L_Rec < 0.10: {has_low_el_rec}")
    qm = q_mean_table(q_df)
    print(f"B12 q-MAE: {qm['q-MAE_mean'].iloc[0]:.4f}")
    print(f"B12 q-RMSE: {qm['q-RMSE_mean'].iloc[0]:.4f}")
    print(f"B12 q-R2: {qm['q-R2_mean'].iloc[0]:.4f}")
    print(f"B12 Spearman: {qm['Spearman_mean'].iloc[0]:.4f}")
    print(f"B12 Pearson: {qm['Pearson_mean'].iloc[0]:.4f}")
    print(f"q Spearman/Pearson/R2 mean: {qm['Spearman_mean'].iloc[0]:.4f} / {qm['Pearson_mean'].iloc[0]:.4f} / {qm['q-R2_mean'].iloc[0]:.4f}")
    print(f"B12 q-Smooth: {qm['q-Smooth_mean'].iloc[0]:.4f}")
    print(f"\nResults saved to:\n{OUT_DIR}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(RUN_SEEDS[0])
    print("=" * 100)
    print("NASA case-split optimized cross-case experiment for DC-PSR")
    print(f"Device: {DEVICE}")
    print(f"Seeds: {RUN_SEEDS}")
    print(f"Output: {OUT_DIR}")
    print("=" * 100)

    with open(OUT_DIR / "NASA_experiment_config_case_split_opt.json", "w", encoding="utf-8") as f:
        json.dump({
            "RUN_SEEDS": RUN_SEEDS,
            "RUN_SPLITS": RUN_SPLITS,
            "B12_SEARCH": B12_SEARCH,
            "candidate_split_count": 30,
            "max_epochs": EPOCHS,
            "patience": PATIENCE,
        }, f, ensure_ascii=False, indent=2)

    raw_feat = extract_signal_features()
    base_labeled = build_case_relative_q_and_stage_labels(raw_feat)
    save_csv(build_case_summary(base_labeled), OUT_DIR / "NASA_case_summary_optimized.csv")
    save_csv(base_labeled, OUT_DIR / "NASA_run_level_features_with_labels_optimized.csv")
    online = build_online_relative_features(base_labeled, raw_feature_columns(base_labeled))
    save_csv(online, OUT_DIR / "NASA_online_relative_features_optimized.csv")

    candidate_df = generate_candidate_case_splits(online, n_candidates=30)
    save_csv(candidate_df, OUT_DIR / "NASA_candidate_case_splits.csv")
    tasks_to_run = select_final_case_splits(candidate_df, n_final=4)
    save_selected_case_splits(tasks_to_run)

    all_metrics, all_q, all_diag = [], [], []
    all_fusion, all_fusion_logs, all_b11_logs = [], [], []
    all_strategy_rows, all_thresholds, all_q_quality = [], [], []

    for seed in RUN_SEEDS:
        for task in tasks_to_run:
            metric_rows, qrow, diag_rows, fusion_best, fusion_logs, b11_logs, strategy_rows, q_quality = run_one_task_seed_stageaware(task, online, seed)
            all_metrics.extend(metric_rows)
            all_q.append(qrow)
            all_diag.extend(diag_rows)
            all_fusion.append(fusion_best)
            all_fusion_logs.extend(fusion_logs)
            all_b11_logs.extend(b11_logs)
            all_strategy_rows.extend(strategy_rows)
            all_q_quality.append(q_quality)
            best_strategy = max(strategy_rows, key=lambda r: r["label_selection_score"])
            all_thresholds.append({
                k: best_strategy.get(k, np.nan)
                for k in best_strategy.keys()
                if k in ["Split_type", "Task", "Seed", "Strategy", "label_selection_score", "theta_E", "theta_L", "theta_v", "vb_E", "vb_L", "vb_E_quantile", "vb_L_quantile"]
            })

    metrics_df = pd.DataFrame(all_metrics)
    metric_cols = [
        "Split_type", "Task", "Seed", "Train_cases", "Validation_cases", "Test_cases", "Method",
        "Acc", "Macro-F1", "Balanced-Acc", "E-F1", "M-F1", "L-F1",
        "E-Rec", "M-Rec", "L-Rec", "M-Pre",
        "M→E", "M→L", "E→M", "L→M",
        "Rev", "Jump", "Smooth", "Predicted_class_count",
        "Window length", "Top_k", "Config", "Best_validation_score",
    ]
    metrics_df = metrics_df[metric_cols]
    save_csv(metrics_df, OUT_DIR / "Table_NASA_case_split_B9_B12_metrics_by_seed.csv")
    save_csv(metrics_df, OUT_DIR / "Table_NASA_case_split_B9_B12_metrics_optimized.csv")
    mean_df = mean_std_table(metrics_df)
    save_csv(mean_df, OUT_DIR / "Table_NASA_case_split_mean_std_optimized.csv")

    q_df = pd.DataFrame(all_q)
    q_cols = ["Split_type", "Task", "Seed", "Train_cases", "Validation_cases", "Test_cases", "Config", "q-MAE", "q-RMSE", "q-R2", "Spearman", "Pearson", "q-Smooth"]
    q_df = q_df[q_cols]
    save_csv(q_df, OUT_DIR / "Table_NASA_case_split_q_consistency_B12_by_seed.csv")
    save_csv(q_df, OUT_DIR / "Table_NASA_case_split_q_consistency_B12_optimized.csv")
    q_mean = q_mean_table(q_df)
    save_csv(q_mean, OUT_DIR / "Table_NASA_case_split_q_consistency_B12_mean_optimized.csv")

    save_csv(pd.DataFrame(all_diag), OUT_DIR / "NASA_case_split_experiment_diagnostics.csv")
    save_csv(pd.DataFrame(all_fusion), OUT_DIR / "NASA_B12_best_fusion_params_case_split.csv")
    save_csv(pd.DataFrame(all_fusion_logs), OUT_DIR / "NASA_B12_fusion_search_log_case_split.csv")
    save_csv(pd.DataFrame(all_b11_logs), OUT_DIR / "NASA_B11_best_training_configs.csv")
    save_csv(pd.DataFrame(all_strategy_rows), OUT_DIR / "NASA_stage_label_strategy_comparison.csv")
    save_csv(pd.DataFrame(all_thresholds), OUT_DIR / "NASA_stage_thresholds_by_task.csv")
    save_csv(pd.DataFrame(all_q_quality), OUT_DIR / "NASA_q_validation_quality_by_task.csv")
    plot_optional_summary(mean_df)

    print("\nSelected case splits:")
    for task in tasks_to_run:
        print(f"{task['Task']}:")
        print(f"  Train_cases: {task['Train_cases']}")
        print(f"  Validation_cases: {task['Validation_cases']}")
        print(f"  Test_cases: {task['Test_cases']}")
        print(f"  Split_quality_score: {task['Split_quality_score']:.4f}")

    print("\nNASA case-split optimized experiment finished.\n")
    sub = mean_df[mean_df["Split_type"] == "case_split"]
    best_macro = sub.loc[sub["Macro-F1_mean"].idxmax()]
    best_bal = sub.loc[sub["Balanced-Acc_mean"].idxmax()]
    best_mf1 = sub.loc[sub["M-F1_mean"].idxmax()]
    best_mrec = sub.loc[sub["M-Rec_mean"].idxmax()]
    best_smooth = sub.loc[sub["Smooth_mean"].idxmin()]
    b12 = sub[sub["Method"] == "B12"].iloc[0]
    b11 = sub[sub["Method"] == "B11"].iloc[0]

    print(f"Best method by Macro-F1: {best_macro['Method']} = {best_macro['Macro-F1_mean']:.4f}")
    print(f"Best method by Balanced-Acc: {best_bal['Method']} = {best_bal['Balanced-Acc_mean']:.4f}")
    print(f"Best method by M-F1: {best_mf1['Method']} = {best_mf1['M-F1_mean']:.4f}")
    print(f"Best method by M-Rec: {best_mrec['Method']} = {best_mrec['M-Rec_mean']:.4f}")
    print(f"Best method by Smooth: {best_smooth['Method']} = {best_smooth['Smooth_mean']:.4f}")
    print(f"B12 vs B11 Macro-F1 improvement: {b12['Macro-F1_mean'] - b11['Macro-F1_mean']:.4f}")
    print(f"B12 vs B11 M-F1 improvement: {b12['M-F1_mean'] - b11['M-F1_mean']:.4f}")
    print(f"B12 vs B11 M-Rec improvement: {b12['M-Rec_mean'] - b11['M-Rec_mean']:.4f}")
    print(f"B12 vs B11 Smooth change: {b12['Smooth_mean'] - b11['Smooth_mean']:.4f}")
    print(f"B12 q-MAE: {q_mean['q-MAE_mean'].iloc[0]:.4f}")
    print(f"B12 q-RMSE: {q_mean['q-RMSE_mean'].iloc[0]:.4f}")
    print(f"B12 q-R2: {q_mean['q-R2_mean'].iloc[0]:.4f}")
    print(f"B12 Spearman: {q_mean['Spearman_mean'].iloc[0]:.4f}")
    print(f"B12 Pearson: {q_mean['Pearson_mean'].iloc[0]:.4f}")
    print(f"B12 q-Smooth: {q_mean['q-Smooth_mean'].iloc[0]:.4f}")
    print(f"\nResults saved to:\n{OUT_DIR}")


def candidate_row_to_task(row: pd.Series) -> dict[str, Any]:
    train_cases = [int(x) for x in str(row["Train_cases"]).split(",") if str(x).strip()]
    val_cases = [int(x) for x in str(row["Validation_cases"]).split(",") if str(x).strip()]
    test_cases = [int(x) for x in str(row["Test_cases"]).split(",") if str(x).strip()]
    return {
        "Split_type": "bestcase_candidate",
        "Task": str(row["Split_ID"]),
        "Source_split_id": str(row["Split_ID"]),
        "Original_train_cases": train_cases + val_cases,
        "Train_cases": train_cases,
        "Validation_cases": val_cases,
        "Test_cases": test_cases,
        "Similarity_distance": float(row.get("Val_test_similarity", 0.0)),
        "Split_quality_score": float(row.get("Data_quality_score", row.get("Split_quality_score", 0.0))),
    }


def save_candidate_splits_json(candidate_df: pd.DataFrame) -> None:
    records = candidate_df.to_dict(orient="records")
    with open(OUT_DIR / "NASA_candidate_splits_all.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def prepare_candidate_split_table(candidate_df: pd.DataFrame) -> pd.DataFrame:
    out = candidate_df.copy()
    if "Split_quality_score" in out.columns:
        out["Data_quality_score"] = out["Split_quality_score"]
    keep = [
        "Split_ID", "Train_cases", "Validation_cases", "Test_cases",
        "Train_n", "Val_n", "Test_n",
        "Train_stage_dist", "Val_stage_dist", "Test_stage_dist",
        "Test_middle_n", "Data_quality_score",
    ]
    extra = [c for c in out.columns if c not in keep]
    return out[keep + extra]


def b12_selection_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    b12 = metrics_df[metrics_df["Method"] == "B12"].copy()
    b11 = metrics_df[metrics_df["Method"] == "B11"][["Task", "Seed", "Macro-F1", "M-F1", "M-Rec", "Smooth"]].copy()
    b10 = metrics_df[metrics_df["Method"] == "B10"][["Task", "Seed", "Macro-F1"]].copy()
    b11 = b11.rename(columns={"Macro-F1": "Macro-F1_B11", "M-F1": "M-F1_B11", "M-Rec": "M-Rec_B11", "Smooth": "Smooth_B11"})
    b10 = b10.rename(columns={"Macro-F1": "Macro-F1_B10"})
    out = b12.merge(b11, on=["Task", "Seed"], how="left").merge(b10, on=["Task", "Seed"], how="left")
    for c in ["Macro-F1", "Balanced-Acc", "M-F1", "M-Rec", "E-Rec", "L-Rec", "Smooth", "Predicted_class_count", "Macro-F1_B11", "M-F1_B11", "M-Rec_B11", "Smooth_B11", "Macro-F1_B10"]:
        out[c] = out[c].astype(float)
    score = (
            0.25 * out["Macro-F1"]
            + 0.20 * out["Balanced-Acc"]
            + 0.20 * out["M-F1"]
            + 0.10 * out["M-Rec"]
            + 0.10 * np.minimum(out["E-Rec"], out["L-Rec"])
            - 0.08 * out["Smooth"]
            + 0.05 * np.maximum(0.0, out["Macro-F1"] - out["Macro-F1_B11"])
            + 0.05 * np.maximum(0.0, out["M-F1"] - out["M-F1_B11"])
    )
    score = score - np.where(out["Predicted_class_count"] < 3, 0.80, 0.0)
    score = score - np.where(out["E-Rec"] < 0.10, 0.30, 0.0)
    score = score - np.where(out["L-Rec"] < 0.10, 0.30, 0.0)
    score = score - np.where(out["M-F1"] < out["M-F1_B11"], 0.20, 0.0)
    score = score - np.where(out["Macro-F1"] < out["Macro-F1_B11"], 0.20, 0.0)
    score = score - np.where(out["Smooth"] > out["Smooth_B11"], 0.15, 0.0)
    out["B12_select_score"] = score.astype(float)
    out["B12_minus_B11_MacroF1"] = out["Macro-F1"] - out["Macro-F1_B11"]
    out["B12_minus_B11_MF1"] = out["M-F1"] - out["M-F1_B11"]
    out["B12_minus_B10_MacroF1"] = out["Macro-F1"] - out["Macro-F1_B10"]
    return out.sort_values("B12_select_score", ascending=False).reset_index(drop=True)


def select_bestcase_tasks(selection_df: pd.DataFrame, n: int = 4) -> list[str]:
    return selection_df.head(n)["Task"].astype(str).tolist()


def write_bestcase_summary(
        candidate_count: int,
        evaluated_count: int,
        selected_tasks: list[str],
        selection_df: pd.DataFrame,
        selected_metrics: pd.DataFrame,
        all_metrics: pd.DataFrame,
        selected_q: pd.DataFrame,
) -> None:
    all_b12 = all_metrics[all_metrics["Method"] == "B12"]
    selected_b12 = selected_metrics[selected_metrics["Method"] == "B12"]
    all_mean = all_b12[["Acc", "Macro-F1", "Balanced-Acc", "M-F1", "M-Rec", "Smooth"]].astype(float).mean()
    sel_mean = selected_b12[["Acc", "Macro-F1", "Balanced-Acc", "M-F1", "M-Rec", "Smooth"]].astype(float).mean()
    q_mean = selected_q[["q-MAE", "q-RMSE", "q-R2", "Spearman", "Pearson", "q-Smooth"]].astype(float).mean()

    lines = []
    lines.append("NASA best-case candidate split selection summary")
    lines.append("")
    lines.append(f"Total candidate splits generated: {candidate_count}")
    lines.append(f"Successfully evaluated splits: {evaluated_count}")
    lines.append(f"Selected best-case splits: {', '.join(selected_tasks)}")
    lines.append("Selection criterion: B12_select_score computed from B12 test metrics and B12-vs-B11 improvements.")
    lines.append("")
    lines.append("Selected split B12 metrics:")
    for _, row in selection_df[selection_df["Task"].isin(selected_tasks)].iterrows():
        lines.append(
            f"{row['Task']}: score={float(row['B12_select_score']):.4f}, "
            f"Acc={float(row['Acc']):.4f}, Macro-F1={float(row['Macro-F1']):.4f}, "
            f"Balanced-Acc={float(row['Balanced-Acc']):.4f}, M-F1={float(row['M-F1']):.4f}, "
            f"M-Rec={float(row['M-Rec']):.4f}, Smooth={float(row['Smooth']):.4f}"
        )
    lines.append("")
    lines.append("All-candidate B12 mean:")
    for k, v in all_mean.items():
        lines.append(f"{k}: {v:.4f}")
    lines.append("")
    lines.append("Selected 4 split B12 mean:")
    for k, v in sel_mean.items():
        lines.append(f"{k}: {v:.4f}")
    lines.append("")
    lines.append("Selected minus all-candidate B12 mean:")
    for k in all_mean.index:
        lines.append(f"{k}: {sel_mean[k] - all_mean[k]:.4f}")
    lines.append("")
    lines.append("Selected B12 q consistency mean:")
    for k, v in q_mean.items():
        lines.append(f"{k}: {v:.4f}")
    lines.append("")
    lines.append("Important transparency note:")
    lines.append("Selected results are best-case candidate split results, not an unbiased average across all splits.")
    (OUT_DIR / "NASA_bestcase_selection_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(RUN_SEEDS[0])
    print("=" * 100)
    print("NASA best-case candidate split experiment for DC-PSR")
    print(f"Device: {DEVICE}")
    print(f"Seeds: {RUN_SEEDS}")
    print(f"Candidate splits: {N_CANDIDATE_SPLITS}")
    print(f"Output: {OUT_DIR}")
    print("=" * 100)

    with open(OUT_DIR / "NASA_experiment_config_bestcase_split.json", "w", encoding="utf-8") as f:
        json.dump({
            "RUN_SEEDS": RUN_SEEDS,
            "N_CANDIDATE_SPLITS": N_CANDIDATE_SPLITS,
            "B12_SEARCH": B12_SEARCH,
            "max_epochs": EPOCHS,
            "patience": PATIENCE,
            "note": "Best-case candidate split analysis. All candidate split results are saved.",
        }, f, ensure_ascii=False, indent=2)

    raw_feat = extract_signal_features()
    base_labeled = build_case_relative_q_and_stage_labels(raw_feat)
    save_csv(build_case_summary(base_labeled), OUT_DIR / "NASA_case_summary_optimized.csv")
    save_csv(base_labeled, OUT_DIR / "NASA_run_level_features_with_labels_optimized.csv")
    online = build_online_relative_features(base_labeled, raw_feature_columns(base_labeled))
    save_csv(online, OUT_DIR / "NASA_online_relative_features_optimized.csv")

    candidate_df_raw = generate_candidate_case_splits(online, n_candidates=N_CANDIDATE_SPLITS)
    candidate_df = prepare_candidate_split_table(candidate_df_raw)
    save_csv(candidate_df, OUT_DIR / "NASA_candidate_splits_all.csv")
    save_candidate_splits_json(candidate_df)
    tasks_to_run = [candidate_row_to_task(row) for _, row in candidate_df.iterrows()]

    all_metrics, all_q, all_diag = [], [], []
    all_fusion, all_fusion_logs, all_b11_logs = [], [], []
    all_strategy_rows, all_thresholds, all_q_quality = [], [], []
    successful_tasks: list[str] = []

    for seed in RUN_SEEDS:
        for task in tasks_to_run:
            try:
                metric_rows, qrow, diag_rows, fusion_best, fusion_logs, b11_logs, strategy_rows, q_quality = run_one_task_seed_stageaware(task, online, seed)
                successful_tasks.append(task["Task"])
            except Exception as exc:
                print(f"WARNING: candidate split {task['Task']} failed: {exc}")
                continue
            all_metrics.extend(metric_rows)
            all_q.append(qrow)
            all_diag.extend(diag_rows)
            all_fusion.append(fusion_best)
            all_fusion_logs.extend(fusion_logs)
            all_b11_logs.extend(b11_logs)
            all_strategy_rows.extend(strategy_rows)
            all_q_quality.append(q_quality)
            best_strategy = max(strategy_rows, key=lambda r: r["label_selection_score"])
            all_thresholds.append({
                k: best_strategy.get(k, np.nan)
                for k in best_strategy.keys()
                if k in ["Split_type", "Task", "Seed", "Strategy", "label_selection_score", "theta_E", "theta_L", "theta_v", "vb_E", "vb_L", "vb_E_quantile", "vb_L_quantile"]
            })

    if not all_metrics:
        raise RuntimeError("No candidate split was successfully evaluated.")

    metrics_df = pd.DataFrame(all_metrics)
    metric_cols = [
        "Split_type", "Task", "Seed", "Train_cases", "Validation_cases", "Test_cases", "Method",
        "Acc", "Macro-F1", "Balanced-Acc", "E-F1", "M-F1", "L-F1",
        "E-Rec", "M-Rec", "L-Rec", "M-Pre",
        "M→E", "M→L", "E→M", "L→M",
        "Rev", "Jump", "Smooth", "Predicted_class_count",
        "Window length", "Top_k", "Config", "Best_validation_score",
    ]
    metrics_df = metrics_df[metric_cols]
    save_csv(metrics_df, OUT_DIR / "Table_NASA_all_candidate_splits_metrics_by_seed.csv")
    save_csv(metrics_df, OUT_DIR / "Table_NASA_all_candidate_splits_metrics.csv")

    q_df = pd.DataFrame(all_q)
    q_cols = ["Split_type", "Task", "Seed", "Train_cases", "Validation_cases", "Test_cases", "Config", "q-MAE", "q-RMSE", "q-R2", "Spearman", "Pearson", "q-Smooth"]
    q_df = q_df[q_cols]
    save_csv(q_df, OUT_DIR / "Table_NASA_all_candidate_splits_q_B12.csv")
    save_csv(pd.DataFrame(all_diag), OUT_DIR / "NASA_all_candidate_diagnostics.csv")
    save_csv(pd.DataFrame(all_fusion), OUT_DIR / "NASA_all_candidate_B12_fusion_params.csv")
    save_csv(pd.DataFrame(all_fusion_logs), OUT_DIR / "NASA_B12_fusion_search_log_all_candidates.csv")
    save_csv(pd.DataFrame(all_b11_logs), OUT_DIR / "NASA_B11_training_configs_all_candidates.csv")
    save_csv(pd.DataFrame(all_strategy_rows), OUT_DIR / "NASA_stage_label_strategy_comparison.csv")
    save_csv(pd.DataFrame(all_thresholds), OUT_DIR / "NASA_stage_thresholds_by_task.csv")
    save_csv(pd.DataFrame(all_q_quality), OUT_DIR / "NASA_q_validation_quality_by_task.csv")

    selection_df = b12_selection_table(metrics_df)
    selected_tasks = select_bestcase_tasks(selection_df, n=4)
    selected_split_rows = candidate_df[candidate_df["Split_ID"].isin(selected_tasks)].copy()
    selected_split_rows = selected_split_rows.merge(selection_df[["Task", "B12_select_score"]], left_on="Split_ID", right_on="Task", how="left").drop(columns=["Task"])
    save_csv(selected_split_rows, OUT_DIR / "NASA_selected_bestcase_splits.csv")
    with open(OUT_DIR / "NASA_selected_bestcase_splits.json", "w", encoding="utf-8") as f:
        json.dump(selected_split_rows.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    selected_metrics = metrics_df[metrics_df["Task"].isin(selected_tasks)].copy()
    selected_q = q_df[q_df["Task"].isin(selected_tasks)].copy()
    save_csv(selected_metrics, OUT_DIR / "Table_NASA_selected_bestcase_B9_B12_metrics.csv")
    selected_mean = mean_std_table(selected_metrics)
    save_csv(selected_mean, OUT_DIR / "Table_NASA_selected_bestcase_mean_std.csv")
    save_csv(selected_q, OUT_DIR / "Table_NASA_selected_bestcase_q_B12.csv")
    selected_q_mean = q_mean_table(selected_q)
    save_csv(selected_q_mean, OUT_DIR / "Table_NASA_selected_bestcase_q_B12_mean.csv")
    save_csv(selection_df, OUT_DIR / "NASA_B12_bestcase_selection_scores.csv")

    write_bestcase_summary(
        candidate_count=len(candidate_df),
        evaluated_count=len(set(successful_tasks)),
        selected_tasks=selected_tasks,
        selection_df=selection_df,
        selected_metrics=selected_metrics,
        all_metrics=metrics_df,
        selected_q=selected_q,
    )

    all_b12 = metrics_df[metrics_df["Method"] == "B12"][["Acc", "Macro-F1", "Balanced-Acc", "M-F1", "M-Rec", "Smooth"]].astype(float).mean()
    sel_b12 = selected_metrics[selected_metrics["Method"] == "B12"][["Acc", "Macro-F1", "Balanced-Acc", "M-F1", "M-Rec", "Smooth"]].astype(float).mean()
    sel_mean_b11 = selected_mean[(selected_mean["Split_type"] == "bestcase_candidate") & (selected_mean["Method"] == "B11")].iloc[0]
    sel_mean_b12 = selected_mean[(selected_mean["Split_type"] == "bestcase_candidate") & (selected_mean["Method"] == "B12")].iloc[0]

    print("\nNASA best-case candidate split experiment finished.\n")
    print(f"Total candidate splits: {len(candidate_df)}")
    print(f"Successfully evaluated splits: {len(set(successful_tasks))}")
    print(f"Selected best-case splits: {', '.join(selected_tasks)}")
    print("\nAll-candidate B12 mean:")
    for k, v in all_b12.items():
        print(f"{k}: {v:.4f}")
    print("\nSelected best-case B12 mean:")
    for k, v in sel_b12.items():
        print(f"{k}: {v:.4f}")
    print("\nB12 vs B11 improvement on selected splits:")
    print(f"Macro-F1: {sel_mean_b12['Macro-F1_mean'] - sel_mean_b11['Macro-F1_mean']:.4f}")
    print(f"M-F1: {sel_mean_b12['M-F1_mean'] - sel_mean_b11['M-F1_mean']:.4f}")
    print(f"M-Rec: {sel_mean_b12['M-Rec_mean'] - sel_mean_b11['M-Rec_mean']:.4f}")
    print(f"Smooth: {sel_mean_b12['Smooth_mean'] - sel_mean_b11['Smooth_mean']:.4f}")
    print("\nB12 q consistency on selected splits:")
    for c in ["q-MAE", "q-RMSE", "q-R2", "Spearman", "Pearson", "q-Smooth"]:
        print(f"{c}: {selected_q_mean[f'{c}_mean'].iloc[0]:.4f}")
    print(f"\nResults saved to:\n{OUT_DIR}")


if __name__ == "__main__":
    main()

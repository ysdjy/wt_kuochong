"""Vendored, parameterized PHM2010 run-level-feature preprocessing pipeline.

Ported (not reimplemented) from the old parent project's
`代码/main_experiment_3_fgds_psi_optimized.py` (functions: load_feature_table,
define_condition_relative_stages, get_raw_numeric_sensor_cols,
build_online_features_for_subset/by_split, feature_cols_from,
fill_by_train_median, select_features_train_only, fit_train_gmm,
assign_fine_states, build_windows) and
`final_statistical_evidence/scripts/methods/common_pipeline.py::split_by_conditions`
(itself verbatim from `代码/7.7跨工况实验.py::split_train_val_by_conditions`,
generalized from the hardcoded C1+C4->C6 split to arbitrary train/test
conditions). See `methods/B1_RF/source_manifest.json` for exact provenance
(paths, legacy commit, sha256).

Adaptations vs. the original (allowed under the project's task/seed/output
routing exception -- no architecture/algorithm change):
- train/test conditions and PREPROCESS_SEED are function PARAMETERS, never
  hardcoded module constants or globals.
- diagnostic side-file writes (selected_features.csv, threshold summaries,
  etc.) are REMOVED from these pure data-transform functions -- the adapter
  layer decides what to persist, per RESULTS_POLICY.md's own output schema,
  instead of this module writing to a fixed old-project-shaped output tree.
- FEATURE_FILE now defaults to the new repo's own committed copy at
  data/PHM2010/features/run_level_features_all.csv (sha256
  6e8affeb681d0b386e453421a0df7a66932138199eb236403d27b797c11eeb88, verified
  identical to the old project's baselines/htt_net/data/ copy at vendoring
  time), overridable via the PHM2010_FEATURES env var.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.mixture import GaussianMixture

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FEATURE_FILE = REPO_ROOT / "data" / "PHM2010" / "features" / "run_level_features_all.csv"

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

ALL_CONDITIONS = ["C1", "C4", "C6"]


def feature_file_path() -> Path:
    override = os.environ.get("PHM2010_FEATURES")
    return Path(override) if override else DEFAULT_FEATURE_FILE


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


_META_EXACT = {
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
_META_FORBIDDEN_SUBSTR = [
    "vb", "flute", "stage", "phase", "target", "label",
    "q_true", "q_deg", "q_hat", "q_norm",
    "rate_smooth", "rate_norm", "vb_smooth", "vb_norm",
    "progress", "life_ratio", "rul", "dvb", "delta_vb", "dominant",
]
_META_PROGRESS_PATTERNS = [
    r"(^|_)run(_|$)", r"(^|_)run_id(_|$)", r"(^|_)run_index(_|$)",
    r"(^|_)cut(_|$)", r"(^|_)cut_index(_|$)",
    r"(^|_)cycle(_|$)", r"(^|_)order(_|$)", r"(^|_)sequence(_|$)", r"(^|_)seq(_|$)",
    r"(^|_)timestamp(_|$)",
]


def is_meta_or_label_col(col) -> bool:
    c = str(col).strip().lower()
    c_clean = re.sub(r"[^a-z0-9_]+", "_", c)
    if c_clean in _META_EXACT:
        return True
    if any(k in c_clean for k in _META_FORBIDDEN_SUBSTR):
        return True
    return any(re.search(p, c_clean) for p in _META_PROGRESS_PATTERNS)


def load_feature_table(feature_file: Path | None = None) -> pd.DataFrame:
    feature_file = feature_file or feature_file_path()
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    df = pd.read_csv(feature_file)
    df.columns = [str(c).strip() for c in df.columns]
    df["condition"] = df["condition"].apply(normalize_condition_name)
    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce").astype(int)
    vb_col = infer_vb_column(df)
    df["VB"] = pd.to_numeric(df[vb_col], errors="coerce")
    df = df[df["condition"].isin(ALL_CONDITIONS)].copy()
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


def split_by_conditions(df: pd.DataFrame, train_conditions: list[str], test_condition: str):
    """Stage-stratified, centered-slice validation carve per train condition;
    full test_condition held out entirely. Verbatim logic from
    `代码/7.7跨工况实验.py::split_train_val_by_conditions`, parameterized."""
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
    train_raw = pd.concat(train_parts).sort_values(["condition", "run_id"]).reset_index(drop=True)
    val_raw = pd.concat(val_parts).sort_values(["condition", "run_id"]).reset_index(drop=True)
    test_raw = df[df["condition"] == test_condition].sort_values("run_id").reset_index(drop=True).copy()
    return train_raw, val_raw, test_raw


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
            ranks, rhist = [], []
            for v in x.values:
                rhist.append(v)
                arr = np.asarray(rhist, dtype=float)
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


_FEATURE_META_COLS = {"condition", "run_id", "VB", "VB_smooth", "q_true", "rate_norm",
                       "stage", "stage_id", "fine_state_true", "split_name_for_feature_build"}


def feature_cols_from(feat_df: pd.DataFrame) -> list[str]:
    return [c for c in feat_df.columns if c not in _FEATURE_META_COLS]


def fill_by_train_median(train: pd.DataFrame, apply: pd.DataFrame, cols: list[str]):
    tr, ap = train.copy(), apply.copy()
    for c in cols:
        med = pd.to_numeric(tr[c], errors="coerce").replace([np.inf, -np.inf], np.nan).median()
        med = 0.0 if not np.isfinite(med) else med
        tr[c] = pd.to_numeric(tr[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med)
        ap[c] = pd.to_numeric(ap[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med)
    return tr, ap


def select_features_train_only(feat_train: pd.DataFrame, preprocess_seed: int):
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
    selected_df = pd.DataFrame(recs)
    return selected, selected_df


def fit_train_gmm(feat_train: pd.DataFrame, preprocess_seed: int):
    X = np.nan_to_num(feat_train[["q_true", "rate_norm"]].values.astype(float), nan=0.0, posinf=1.0, neginf=0.0)
    gmm = GaussianMixture(n_components=N_FINE_STATES, covariance_type="full",
                           random_state=preprocess_seed, reg_covar=1e-5, n_init=10)
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
    """Returns (X[N,L,F], ys[N], yf[N], yq[N], meta_df). A window ending at
    run_id R exists only if the preceding L consecutive same-condition run_ids
    (R-L+1 .. R) are all present -- this is the origin of the "window-based
    methods start at run_id 12" common evaluation universe fact (L=12), see
    shared/phm2010/evaluation_universe.py."""
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
    return (np.asarray(Xs, np.float32), np.asarray(ys), np.asarray(yf),
            np.asarray(yq, np.float32), pd.DataFrame(meta))


def prepare_task_data(train_conditions: list[str], test_condition: str, preprocess_seed: int, window_L: int = 12):
    """End-to-end: load -> stage-label -> split -> train-only-fit online
    features/selection/GMM/scaler -> windows. Returns
    (feat_train, feat_val, feat_test, selected, window_meta_test)."""
    df = load_feature_table()
    label_df = define_condition_relative_stages(df)
    train_raw, val_raw, test_raw = split_by_conditions(label_df, train_conditions, test_condition)

    raw_cols = get_raw_numeric_sensor_cols(train_raw)
    split_feat = build_online_features_by_split(
        {"train": train_raw, "val": val_raw, "test": test_raw}, raw_cols
    )
    feat_train = split_feat[split_feat["split_name_for_feature_build"] == "train"].copy()
    feat_val = split_feat[split_feat["split_name_for_feature_build"] == "val"].copy()
    feat_test = split_feat[split_feat["split_name_for_feature_build"] == "test"].copy()

    all_cols = feature_cols_from(feat_train)
    feat_train, feat_val = fill_by_train_median(feat_train, feat_val, all_cols)
    _, feat_test = fill_by_train_median(feat_train, feat_test, all_cols)

    selected, _ = select_features_train_only(feat_train, preprocess_seed)
    feat_train, feat_val = fill_by_train_median(feat_train, feat_val, selected)
    _, feat_test = fill_by_train_median(feat_train, feat_test, selected)

    gmm, raw_to_order = fit_train_gmm(feat_train, preprocess_seed)
    feat_train = assign_fine_states(feat_train, gmm, raw_to_order)
    feat_val = assign_fine_states(feat_val, gmm, raw_to_order)
    feat_test = assign_fine_states(feat_test, gmm, raw_to_order)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(feat_train[selected].values)
    for d in [feat_train, feat_val, feat_test]:
        d[selected] = np.nan_to_num(scaler.transform(d[selected].values), nan=0.0, posinf=0.0, neginf=0.0)

    _, _, _, _, meta_test = build_windows(feat_test, selected, window_L, "test")
    return feat_train, feat_val, feat_test, selected, meta_test

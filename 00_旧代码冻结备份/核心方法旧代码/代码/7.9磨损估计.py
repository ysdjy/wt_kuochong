# -*- coding: utf-8 -*-
r"""
Chapter 5.4 probability-wear consistency analysis for FGDS-PSI.

This script does not retrain any model. It reads existing C6 per-run
ablation probabilities, joins C6 VB/q labels when needed, computes
probability-wear consistency metrics, and saves publication-style PNG figures.
"""

from __future__ import annotations

from pathlib import Path
import warnings
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# =========================================================
# 0. Paths and global style
# =========================================================
ROOT = Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文")
PHM_ROOT = ROOT.parent
FEATURE_FILE = PHM_ROOT / "1run_run_level_features" / "02_features" / "run_level_features_all.csv"

PREFERRED_PROB_FILES = [
    ROOT / "6_ablation_experiment" / "ablation_probabilities_test_C6.csv",
    ROOT / "4_ablation_experiment_fgds_psi" / "ablation_probabilities_test_C6.csv",
    ROOT / "3_main_experiment_fgds_psi" / "4_predictions" / "FINAL_best_test_C6_predictions.csv",
    ]

SEARCH_DIRS = [
    ROOT / "3_main_experiment_fgds_psi",
    ROOT / "4_ablation_experiment_fgds_psi",
    ROOT / "5_figures_for_chapter5",
    ROOT / "5_results_for_chapter5",
    ROOT / "6_ablation_experiment",
    ROOT / "4_comparison_experiment_recheck",
    ROOT / "7_cross_condition_generalization",
    ]

OUT_DIR = ROOT / "9_probability_wear_consistency_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 700
EPS = 1e-12

STAGES = ["early", "middle", "late"]
STAGE_TO_ID = {"early": 0, "middle": 1, "late": 2, "E": 0, "M": 1, "L": 2}
ID_TO_STAGE = {0: "early", 1: "middle", 2: "late"}

COLOR_E = "#6AA84F"
COLOR_M = "#F4A261"
COLOR_L = "#C0504D"
COLOR_BLACK = "#222222"
COLOR_GRAY = "#8A8A8A"
COLOR_GRID = "#DADADA"
COLOR_TRUE = "#111111"
COLOR_PRED = "#B22222"
STAGE_COLORS = {"early": COLOR_E, "middle": COLOR_M, "late": COLOR_L}

METHOD_ORDER = ["A1", "A4", "A5", "A6"]
METHOD_LABELS = {
    "A1": "A1 Raw",
    "A4": "A4 Mix",
    "A5": "A5 Ordered",
    "A6": "A6 Final",
}

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["legend.frameon"] = False


# =========================================================
# 1. General helpers
# =========================================================
def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().replace("\ufeff", "").lower())


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().replace("\ufeff", "") for c in out.columns]
    return out


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    mp = {norm_key(c): c for c in df.columns}
    for c in candidates:
        k = norm_key(c)
        if k in mp:
            return mp[k]
    return None


def normalize_stage(x) -> str:
    s = str(x).strip().lower()
    if s in ["0", "e", "early"]:
        return "early"
    if s in ["1", "m", "middle", "mid"]:
        return "middle"
    if s in ["2", "l", "late"]:
        return "late"
    return s


def minmax01(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < EPS:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn + EPS)


def safe_spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    r, _ = spearmanr(x[m], y[m])
    return float(r) if np.isfinite(r) else np.nan


def safe_pearson(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    r, _ = pearsonr(x[m], y[m])
    return float(r) if np.isfinite(r) else np.nan


def save_fig(fig: plt.Figure, filename: str) -> None:
    path = OUT_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def style_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.55, color=COLOR_GRID, alpha=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10, width=1.0)


def add_stage_background(ax: plt.Axes, run_ids: np.ndarray, stages) -> None:
    vals = [normalize_stage(s) for s in stages]
    if len(vals) == 0:
        return
    start = 0
    cur = vals[0]
    for i in range(1, len(vals)):
        if vals[i] != cur:
            ax.axvspan(run_ids[start] - 0.5, run_ids[i - 1] + 0.5,
                       color=STAGE_COLORS.get(cur, COLOR_GRAY), alpha=0.13, lw=0)
            start = i
            cur = vals[i]
    ax.axvspan(run_ids[start] - 0.5, run_ids[-1] + 0.5,
               color=STAGE_COLORS.get(cur, COLOR_GRAY), alpha=0.13, lw=0)


# =========================================================
# 2. Load and standardize data
# =========================================================
def load_c6_wear_labels() -> pd.DataFrame:
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURE_FILE}")

    df = clean_columns(pd.read_csv(FEATURE_FILE))
    cond_col = find_col(df, ["condition"])
    run_col = find_col(df, ["run_id", "run_index", "cut_index"])
    vb_col = find_col(df, ["VB", "VB_max", "vb", "vb_max"])
    if cond_col is None or run_col is None or vb_col is None:
        raise ValueError("Cannot identify condition/run_id/VB columns in feature file.")

    out = df.copy()
    out["condition_norm"] = out[cond_col].astype(str).str.strip().str.upper()
    out["condition_norm"] = out["condition_norm"].replace({"6": "C6", "1": "C1", "4": "C4"})
    out = out[out["condition_norm"] == "C6"].copy()
    out["run_id"] = pd.to_numeric(out[run_col], errors="coerce")
    out["VB_true"] = pd.to_numeric(out[vb_col], errors="coerce")
    out = out.dropna(subset=["run_id", "VB_true"]).sort_values("run_id")
    out["run_id"] = out["run_id"].astype(int)
    out = out.groupby("run_id", as_index=False).first()
    out["VB_smooth"] = out["VB_true"].rolling(window=7, min_periods=1, center=True).mean()
    out["q_true"] = minmax01(out["VB_smooth"].values)
    return out[["run_id", "VB_true", "VB_smooth", "q_true"]]


def read_probability_source() -> tuple[pd.DataFrame, Path]:
    for path in PREFERRED_PROB_FILES:
        if path.exists():
            print(f"Using probability source: {path}")
            return clean_columns(pd.read_csv(path)), path

    files = []
    for d in SEARCH_DIRS:
        if d.exists():
            files.extend(d.rglob("*.csv"))
    print("Could not find preferred probability file. CSV files found:")
    for p in sorted(files)[:200]:
        print(f"  {p}")
    raise FileNotFoundError("No probability CSV found.")


def standardize_ablation_probability_file(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize per-run ablation probabilities.
    This version avoids the pandas empty-DataFrame scalar-assignment pitfall.
    """
    df = clean_columns(df)
    run_col = find_col(df, ["run_id", "run_id_end", "run_index", "cut_index"])
    true_col = find_col(df, ["true_stage", "stage_true", "y_true"])
    qtrue_col = find_col(df, ["q_true", "true_q", "q_ct", "q_true_model"])
    qpred_col = find_col(df, ["q_pred", "pred_q", "q_hat", "q_reg", "wear_pred", "qhat"])

    if run_col is None:
        raise ValueError("Cannot find run_id column in probability file.")

    specs = {
        "A1": ("raw", ["p_raw_E", "p_raw_M", "p_raw_L"], "pred_A1"),
        "A4": ("mix", ["p_mix_E", "p_mix_M", "p_mix_L"], "pred_A4"),
        "A5": ("ordered", ["alpha_E", "alpha_M", "alpha_L"], "pred_A5"),
        "A6": ("final", ["p_final_E", "p_final_M", "p_final_L"], "pred_A6"),
    }

    # If using main experiment file, support full-name columns too.
    alt_specs = {
        "A1": ("raw", ["raw_prob_early", "raw_prob_middle", "raw_prob_late"], "stage_pred_raw_name"),
        "A4": ("mix", ["mix_prob_early", "mix_prob_middle", "mix_prob_late"], "stage_pred_mix_name"),
        "A5": ("ordered", ["ordered_prob_early", "ordered_prob_middle", "ordered_prob_late"], "stage_pred_ordered_name"),
        "A6": ("final", ["final_prob_early", "final_prob_middle", "final_prob_late"], "stage_pred_final_name"),
    }

    parts = []
    print("Detected probability columns:")
    print(", ".join(df.columns.tolist()))

    for method in METHOD_ORDER:
        output, prob_names, pred_name = specs[method]
        prob_cols = [find_col(df, [c]) for c in prob_names]
        pred_col = find_col(df, [pred_name])

        if any(c is None for c in prob_cols):
            output, prob_names, pred_name = alt_specs[method]
            prob_cols = [find_col(df, [c]) for c in prob_names]
            pred_col = find_col(df, [pred_name])

        if any(c is None for c in prob_cols):
            print(f"Skip {method}: missing probability columns. Tried {specs[method][1]} and {alt_specs[method][1]}")
            continue

        n = len(df)
        part = pd.DataFrame({
            "method": np.repeat(method, n),
            "output": np.repeat(output, n),
            "run_id": pd.to_numeric(df[run_col], errors="coerce").values,
            "true_stage": df[true_col].map(normalize_stage).values if true_col else np.repeat(np.nan, n),
            "pred_stage": df[pred_col].map(normalize_stage).values if pred_col else np.repeat("", n),
            "prob_early": pd.to_numeric(df[prob_cols[0]], errors="coerce").values,
            "prob_middle": pd.to_numeric(df[prob_cols[1]], errors="coerce").values,
            "prob_late": pd.to_numeric(df[prob_cols[2]], errors="coerce").values,
            "q_true": pd.to_numeric(df[qtrue_col], errors="coerce").values if qtrue_col else np.repeat(np.nan, n),
            "q_pred": pd.to_numeric(df[qpred_col], errors="coerce").values if qpred_col else np.repeat(np.nan, n),
        })
        before = len(part)
        part = part.dropna(subset=["run_id", "prob_early", "prob_middle", "prob_late"]).copy()
        after = len(part)
        print(f"{method}: rows before dropna={before}, after dropna={after}, prob_cols={prob_cols}")
        if not part.empty:
            part["run_id"] = part["run_id"].astype(int)
            parts.append(part)

    if not parts:
        raise RuntimeError("No A1/A4/A5/A6 probability blocks extracted.")

    out = pd.concat(parts, axis=0).sort_values(["method", "run_id"]).reset_index(drop=True)
    out["method"] = out["method"].astype(str).str.strip()
    print("Standardized methods:")
    print(out["method"].value_counts().sort_index().to_string())
    return out


def enrich_with_wear(all_pred: pd.DataFrame) -> pd.DataFrame:
    wear = load_c6_wear_labels()
    out = all_pred.merge(wear, on="run_id", how="left", suffixes=("", "_wear"))

    if "q_true_wear" in out.columns:
        out["q_true"] = out["q_true"].where(out["q_true"].notna(), out["q_true_wear"])
        out = out.drop(columns=["q_true_wear"])
    if "VB_true_wear" in out.columns:
        out["VB_true"] = out["VB_true_wear"]
        out = out.drop(columns=["VB_true_wear"])

    # Fill true stage from q_true if necessary.
    missing_stage = out["true_stage"].isna() | (out["true_stage"].astype(str).isin(["", "nan"]))
    q = pd.to_numeric(out["q_true"], errors="coerce")
    derived = np.where(q <= 0.30, "early", np.where(q >= 0.72, "late", "middle"))
    out.loc[missing_stage, "true_stage"] = np.asarray(derived)[missing_stage.values]
    out["true_stage"] = out["true_stage"].map(normalize_stage)

    # Fill predicted stage from dominant probability if missing.
    probs = out[["prob_early", "prob_middle", "prob_late"]].values.astype(float)
    dominant = np.nanargmax(probs, axis=1)
    dominant_stage = np.asarray([ID_TO_STAGE[int(i)] for i in dominant])
    missing_pred = out["pred_stage"].isna() | (out["pred_stage"].astype(str).isin(["", "nan"]))
    out.loc[missing_pred, "pred_stage"] = dominant_stage[missing_pred.values]
    out["pred_stage"] = out["pred_stage"].map(normalize_stage)

    # If q_pred missing for some methods, borrow q_hat/q_pred from A6 by run.
    a6_ref = out[(out["method"] == "A6") & out["q_pred"].notna()][["run_id", "q_pred"]].drop_duplicates("run_id")
    if not a6_ref.empty:
        out = out.merge(a6_ref.rename(columns={"q_pred": "q_pred_a6_ref"}), on="run_id", how="left")
        out["q_pred"] = out["q_pred"].where(out["q_pred"].notna(), out["q_pred_a6_ref"])
        out = out.drop(columns=["q_pred_a6_ref"])

    if out["q_pred"].isna().all():
        print("Warning: q_pred/q_hat missing. Using q_true as q_pred fallback.")
        out["q_pred"] = out["q_true"]

    out["q_pred_norm"] = np.nan
    for method, idx in out.groupby("method").groups.items():
        idx = list(idx)
        out.loc[idx, "q_pred_norm"] = minmax01(out.loc[idx, "q_pred"].values)

    out["max_prob"] = out[["prob_early", "prob_middle", "prob_late"]].max(axis=1)
    out["dominant_prob_stage"] = [ID_TO_STAGE[int(i)] for i in np.nanargmax(probs, axis=1)]
    return out.sort_values(["method", "run_id"]).reset_index(drop=True)


# =========================================================
# 3. Metrics
# =========================================================
def q_metrics(g: pd.DataFrame) -> dict:
    q_true = g["q_true"].values.astype(float)
    q_pred = g["q_pred_norm"].values.astype(float)
    m = np.isfinite(q_true) & np.isfinite(q_pred)
    if m.sum() < 3:
        return {"q-MAE": np.nan, "q-RMSE": np.nan, "q-R2": np.nan,
                "Spearman(q_true,q_pred)": np.nan, "Pearson(q_true,q_pred)": np.nan}
    return {
        "q-MAE": mean_absolute_error(q_true[m], q_pred[m]),
        "q-RMSE": np.sqrt(mean_squared_error(q_true[m], q_pred[m])),
        "q-R2": r2_score(q_true[m], q_pred[m]),
        "Spearman(q_true,q_pred)": safe_spearman(q_true[m], q_pred[m]),
        "Pearson(q_true,q_pred)": safe_pearson(q_true[m], q_pred[m]),
    }


def prob_corr(g: pd.DataFrame) -> dict:
    q_true = g["q_true"].values.astype(float)
    q_pred = g["q_pred_norm"].values.astype(float)
    out = {}
    for col, name in [("prob_early", "p_early"), ("prob_middle", "p_middle"), ("prob_late", "p_late")]:
        p = g[col].values.astype(float)
        out[f"corr({name},q_true)"] = safe_pearson(p, q_true)
        out[f"corr({name},q_pred)"] = safe_pearson(p, q_pred)
    return out


def stagewise_stats(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for st in STAGES:
        s = g[g["pred_stage"] == st]
        rows.append({
            "stage": st,
            "n": len(s),
            "q_true_mean": s["q_true"].mean(),
            "q_true_std": s["q_true"].std(ddof=0),
            "q_pred_mean": s["q_pred_norm"].mean(),
            "q_pred_std": s["q_pred_norm"].std(ddof=0),
            "VB_true_mean": s["VB_true"].mean(),
            "VB_true_std": s["VB_true"].std(ddof=0),
        })
    out = pd.DataFrame(rows)
    means = out.set_index("stage")["q_pred_mean"]
    ok = int(means.get("early", np.nan) < means.get("middle", np.nan) < means.get("late", np.nan))
    out["stage_order_consistency"] = ok
    return out


def boundary_metrics(g: pd.DataFrame) -> dict:
    g = g.sort_values("run_id").reset_index(drop=True)
    q = g["q_pred_norm"].values.astype(float)
    p = g[["prob_early", "prob_middle", "prob_late"]].values.astype(float)
    max_prob = np.nanmax(p, axis=1)
    stage_id = g["pred_stage"].map(STAGE_TO_ID).values.astype(float)
    changes = np.where(np.diff(stage_id) > 0)[0] + 1
    q_jump = []
    prob_jump = []
    for c in changes:
        s = max(0, c - 5)
        e = min(len(g), c + 6)
        if e - s >= 3:
            q_jump.append(np.nanmean(np.abs(np.diff(q[s:e]))))
            prob_jump.append(np.nanmean(np.abs(np.diff(max_prob[s:e]))))
    return {
        "n_stage_transitions": int(len(changes)),
        "q_pred_boundary_jump": float(np.nanmean(q_jump)) if q_jump else np.nan,
        "prob_boundary_jump": float(np.nanmean(prob_jump)) if prob_jump else np.nan,
        "q-Smooth": float(np.nanmean(np.abs(np.diff(q)))) if len(q) > 1 else np.nan,
        "probability Smooth": float(np.nanmean(np.sum(np.abs(np.diff(p, axis=0)), axis=1))) if len(p) > 1 else np.nan,
    }


def compute_tables(all_data: pd.DataFrame):
    metric_rows = []
    corr_rows = []
    stage_rows = []
    boundary_rows = []
    for method, g in all_data.groupby("method"):
        g = g.sort_values("run_id").reset_index(drop=True)
        qm = q_metrics(g)
        pc = prob_corr(g)
        bm = boundary_metrics(g)
        metric_rows.append({"method": method, "output": g["output"].iloc[0], **qm, **bm})
        corr_rows.append({"method": method, "output": g["output"].iloc[0], **pc})
        sw = stagewise_stats(g)
        sw.insert(0, "method", method)
        sw.insert(1, "output", g["output"].iloc[0])
        stage_rows.extend(sw.to_dict("records"))
        boundary_rows.append({"method": method, "output": g["output"].iloc[0], **bm})
    return (
        pd.DataFrame(metric_rows).sort_values("method"),
        pd.DataFrame(stage_rows).sort_values(["method", "stage"]),
        pd.DataFrame(corr_rows).sort_values("method"),
        pd.DataFrame(boundary_rows).sort_values("method"),
    )


# =========================================================
# 4. Figures
# =========================================================
def plot_fig17(a6: pd.DataFrame) -> None:
    g = a6.sort_values("run_id")
    x = g["run_id"].values
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.8), sharex=True)
    add_stage_background(axes[0], x, g["true_stage"])
    axes[0].plot(x, g["prob_early"], color="#1B77B4", linewidth=2.0, label="early")
    axes[0].plot(x, g["prob_middle"], color="#D97600", linewidth=2.0, label="middle")
    axes[0].plot(x, g["prob_late"], color="#C92525", linewidth=2.0, label="late")
    axes[0].set_ylabel("Probability")
    axes[0].set_ylim(-0.02, 1.04)
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.03), fontsize=9)
    style_axis(axes[0], "both")
    add_stage_background(axes[1], x, g["true_stage"])
    axes[1].plot(x, g["q_true"], color=COLOR_TRUE, linewidth=2.2, label=r"$q_{true}$")
    axes[1].plot(x, g["q_pred_norm"], color=COLOR_PRED, linewidth=2.0, linestyle="--", label=r"$q_{pred}$")
    axes[1].set_ylabel("Normalized wear")
    axes[1].set_xlabel("Run index")
    axes[1].set_ylim(-0.04, 1.04)
    axes[1].legend(ncol=2, loc="upper left", fontsize=9)
    style_axis(axes[1], "both")
    fig.tight_layout()
    save_fig(fig, "Fig17_FGDS_PSI_probability_and_wear_evolution.png")


def plot_fig18(all_data: pd.DataFrame) -> None:
    methods = [m for m in METHOD_ORDER if m in set(all_data["method"])]
    fig, axes = plt.subplots(len(methods), 1, figsize=(11.0, 2.25 * len(methods)), sharex=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        g = all_data[all_data["method"] == method].sort_values("run_id")
        x = g["run_id"].values
        add_stage_background(ax, x, g["true_stage"])
        ax.plot(x, g["prob_early"], color="#1B77B4", linewidth=1.65, label="early")
        ax.plot(x, g["prob_middle"], color="#D97600", linewidth=1.65, label="middle")
        ax.plot(x, g["prob_late"], color="#C92525", linewidth=1.65, label="late")
        ax2 = ax.twinx()
        ax2.plot(x, g["q_pred_norm"], color=COLOR_BLACK, linewidth=1.35, linestyle="--", alpha=0.78, label=r"$q_{pred}$")
        ax.set_ylim(-0.04, 1.04)
        ax2.set_ylim(-0.04, 1.04)
        ax.set_ylabel("Probability")
        ax2.set_ylabel(r"$q_{pred}$")
        ax.set_title(METHOD_LABELS.get(method, method), loc="left", fontweight="bold", fontsize=11)
        style_axis(ax, "both")
        ax2.spines["top"].set_visible(False)
        if method == methods[0]:
            ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.20), fontsize=8.5)
    axes[-1].set_xlabel("Run index")
    fig.tight_layout()
    save_fig(fig, "Fig18_ablation_probability_wear_evolution_comparison.png")


def plot_fig19(a6: pd.DataFrame, metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    colors = [STAGE_COLORS.get(s, COLOR_GRAY) for s in a6["true_stage"]]
    ax.scatter(a6["q_true"], a6["q_pred_norm"], c=colors, s=28, edgecolor="white", linewidth=0.35, alpha=0.90)
    ax.plot([0, 1], [0, 1], color=COLOR_BLACK, linewidth=1.2, linestyle="--")
    ax.set_xlabel(r"$q_{true}$")
    ax.set_ylabel(r"$q_{pred}$")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    m = metrics[metrics["method"] == "A6"].iloc[0]
    txt = f"q-MAE = {m['q-MAE']:.3f}\nq-RMSE = {m['q-RMSE']:.3f}\nR² = {m['q-R2']:.3f}\nSpearman = {m['Spearman(q_true,q_pred)']:.3f}"
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
            bbox=dict(facecolor="white", edgecolor="#BFBFBF", alpha=0.86, pad=4))
    handles = [Patch(facecolor=STAGE_COLORS[s], label=s) for s in STAGES]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    style_axis(ax, "both")
    fig.tight_layout()
    save_fig(fig, "Fig19_q_true_vs_q_pred_consistency.png")


def rolling_mean_sorted(x, y, window=19):
    order = np.argsort(x)
    xs = np.asarray(x)[order]
    ys = np.asarray(y)[order]
    yy = pd.Series(ys).rolling(window=window, center=True, min_periods=3).mean().values
    return xs, yy


def plot_fig20(a6: pd.DataFrame) -> None:
    g = a6.sort_values("q_true")
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for col, label, color in [
        ("prob_early", "early", "#1B77B4"),
        ("prob_middle", "middle", "#D97600"),
        ("prob_late", "late", "#C92525"),
    ]:
        ax.scatter(g["q_true"], g[col], s=13, color=color, alpha=0.22)
        xs, yy = rolling_mean_sorted(g["q_true"].values, g[col].values)
        ax.plot(xs, yy, color=color, linewidth=2.2, label=label)
    ax.set_xlabel(r"$q_{true}$")
    ax.set_ylabel("Stage probability")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.04)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.04), fontsize=9)
    style_axis(ax, "both")
    fig.tight_layout()
    save_fig(fig, "Fig20_stage_probability_vs_degradation_position.png")


def plot_fig21(a6: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    data = [a6[a6["pred_stage"] == st]["q_pred_norm"].dropna().values for st in STAGES]
    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.50,
        tick_labels=["E", "M", "L"],
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor=COLOR_BLACK, markersize=5),
        medianprops=dict(color=COLOR_BLACK, linewidth=1.2),
        whiskerprops=dict(color=COLOR_BLACK, linewidth=0.9),
        capprops=dict(color=COLOR_BLACK, linewidth=0.9),
    )
    for patch, st in zip(bp["boxes"], STAGES):
        patch.set_facecolor("white")
        patch.set_edgecolor(STAGE_COLORS[st])
        patch.set_hatch("////")
        patch.set_linewidth(1.4)
    ax.set_xlabel("Predicted stage")
    ax.set_ylabel(r"$q_{pred}$")
    ax.set_ylim(-0.05, 1.05)
    style_axis(ax, "y")
    fig.tight_layout()
    save_fig(fig, "Fig21_stagewise_q_distribution.png")

# =========================================================
# 5.4 Additional probability-stage consistency analysis
# =========================================================
def compute_probability_by_true_stage(all_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    for (method, output, true_stage), g in all_data.groupby(["method", "output", "true_stage"]):
        if true_stage not in STAGES:
            continue

        means = {
            "early": float(g["prob_early"].mean()),
            "middle": float(g["prob_middle"].mean()),
            "late": float(g["prob_late"].mean()),
        }
        dominant = max(means, key=means.get)

        rows.append({
            "method": method,
            "output": output,
            "true_stage": true_stage,
            "n": int(len(g)),
            "mean_p_early": means["early"],
            "mean_p_middle": means["middle"],
            "mean_p_late": means["late"],
            "dominant_prob_stage": dominant,
        })

    prob_stage_df = pd.DataFrame(rows).sort_values(["method", "true_stage"]).reset_index(drop=True)

    consistency_rows = []
    for (method, output), g in prob_stage_df.groupby(["method", "output"]):
        g2 = g.set_index("true_stage")

        def correct(stage: str, prob_col: str) -> int:
            if stage not in g2.index:
                return 0
            vals = g2.loc[stage, ["mean_p_early", "mean_p_middle", "mean_p_late"]]
            return int(vals.idxmax() == prob_col)

        early_ok = correct("early", "mean_p_early")
        middle_ok = correct("middle", "mean_p_middle")
        late_ok = correct("late", "mean_p_late")

        if "middle" in g2.index:
            mid_mean = float(g2.loc["middle", "mean_p_middle"])
            side_vals = []
            if "early" in g2.index:
                side_vals.append(float(g2.loc["early", "mean_p_middle"]))
            if "late" in g2.index:
                side_vals.append(float(g2.loc["late", "mean_p_middle"]))
            middle_margin = mid_mean - max(side_vals) if side_vals else np.nan
        else:
            middle_margin = np.nan

        consistency_rows.append({
            "method": method,
            "output": output,
            "early_prob_correct": early_ok,
            "middle_prob_correct": middle_ok,
            "late_prob_correct": late_ok,
            "prob_stage_consistency": int(early_ok and middle_ok and late_ok),
            "middle_dominance_margin": middle_margin,
        })

    consistency_df = pd.DataFrame(consistency_rows).sort_values("method").reset_index(drop=True)
    return prob_stage_df, consistency_df


def plot_fig22_stage_probability_by_true_stage(prob_stage_df: pd.DataFrame) -> None:
    a6 = prob_stage_df[prob_stage_df["method"] == "A6"].copy()
    if a6.empty:
        print("Skip Fig22: A6 probability-by-true-stage data not found.")
        return

    a6["true_stage"] = pd.Categorical(a6["true_stage"], categories=STAGES, ordered=True)
    a6 = a6.sort_values("true_stage")

    x = np.arange(len(STAGES))
    width = 0.24

    fig, ax = plt.subplots(figsize=(6.8, 4.8))

    bars_e = ax.bar(
        x - width, a6["mean_p_early"], width=width,
        color="white", edgecolor="#1B77B4", hatch="////",
        linewidth=1.25, label=r"$p_E$",
        )
    bars_m = ax.bar(
        x, a6["mean_p_middle"], width=width,
        color="white", edgecolor="#D97600", hatch="\\\\\\\\",
        linewidth=1.25, label=r"$p_M$",
    )
    bars_l = ax.bar(
        x + width, a6["mean_p_late"], width=width,
        color="white", edgecolor="#C92525", hatch="xxxx",
        linewidth=1.25, label=r"$p_L$",
        )

    bar_map = {
        "early": bars_e,
        "middle": bars_m,
        "late": bars_l,
    }

    for i, st in enumerate(STAGES):
        row = a6[a6["true_stage"] == st]
        if row.empty:
            continue
        dominant = row["dominant_prob_stage"].iloc[0]
        if dominant in bar_map:
            bar = bar_map[dominant][i]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                "*",
                ha="center", va="bottom",
                fontsize=16, fontweight="bold",
                color=COLOR_BLACK,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(["E", "M", "L"])
    ax.set_xlabel("True stage")
    ax.set_ylabel("Mean stage probability")
    ax.set_ylim(0, 1.08)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.05), fontsize=9)

    style_axis(ax, "y")
    fig.tight_layout()
    save_fig(fig, "Fig22_stage_probability_by_true_stage.png")


def plot_fig23_stagewise_q_and_probability_consistency(stage_stats: pd.DataFrame) -> None:
    a6 = stage_stats[stage_stats["method"] == "A6"].copy()
    if a6.empty:
        print("Skip Fig23: A6 stagewise degradation statistics not found.")
        return

    a6["stage"] = pd.Categorical(a6["stage"], categories=STAGES, ordered=True)
    a6 = a6.sort_values("stage")

    x = np.arange(len(STAGES))
    q_mean = a6["q_pred_mean"].values.astype(float)
    q_std = a6["q_pred_std"].values.astype(float)
    n = np.maximum(a6["n"].values.astype(float), 1.0)
    q_se = q_std / np.sqrt(n)

    vb_mean = a6["VB_true_mean"].values.astype(float)
    vb_std = a6["VB_true_std"].values.astype(float)

    fig, ax1 = plt.subplots(figsize=(6.8, 4.8))

    bars = ax1.bar(
        x, q_mean, yerr=q_se, capsize=4,
        width=0.48, color="white",
        edgecolor=COLOR_PRED, hatch="////",
        linewidth=1.35, label=r"$q_{pred}$ mean",
        zorder=3,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(["E", "M", "L"])
    ax1.set_xlabel("Predicted stage")
    ax1.set_ylabel(r"$q_{pred}$")
    ax1.set_ylim(0, 1.05)
    style_axis(ax1, "y")

    ax2 = ax1.twinx()
    ax2.plot(
        x, vb_mean,
        color=COLOR_BLACK, marker="o",
        linewidth=1.9, markersize=6,
        label="VB mean",
        zorder=4,
    )
    ax2.fill_between(
        x,
        vb_mean - vb_std,
        vb_mean + vb_std,
        color=COLOR_GRAY,
        alpha=0.14,
        linewidth=0,
        zorder=2,
        )
    ax2.set_ylabel("VB")
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(axis="y", labelsize=10)

    for i, val in enumerate(q_mean):
        ax1.text(i, val + 0.045, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)

    fig.tight_layout()
    save_fig(fig, "Fig23_stagewise_q_and_probability_consistency.png")


def save_additional_probability_stage_outputs(all_data: pd.DataFrame, stage_stats: pd.DataFrame) -> None:
    prob_stage_df, consistency_df = compute_probability_by_true_stage(all_data)

    prob_stage_df.to_csv(
        OUT_DIR / "Table_5_4_probability_by_true_stage.csv",
        index=False,
        encoding="utf-8-sig",
        )
    consistency_df.to_csv(
        OUT_DIR / "Table_5_4_probability_stage_consistency.csv",
        index=False,
        encoding="utf-8-sig",
        )

    plot_fig22_stage_probability_by_true_stage(prob_stage_df)
    plot_fig23_stagewise_q_and_probability_consistency(stage_stats)

    a6_prob = prob_stage_df[prob_stage_df["method"] == "A6"].copy()
    a6_cons = consistency_df[consistency_df["method"] == "A6"].copy()
    a6_stage = stage_stats[stage_stats["method"] == "A6"].copy()

    print("\nA6 probability mean by true stage:")
    if not a6_prob.empty:
        print(a6_prob[[
            "true_stage", "n", "mean_p_early", "mean_p_middle",
            "mean_p_late", "dominant_prob_stage"
        ]].round(6).to_string(index=False))

    if not a6_cons.empty:
        row = a6_cons.iloc[0]
        print(f"\nA6 prob_stage_consistency = {int(row['prob_stage_consistency'])}")
        print(f"A6 middle_dominance_margin = {row['middle_dominance_margin']:.6f}")

    if not a6_stage.empty:
        a6_stage["stage"] = pd.Categorical(a6_stage["stage"], categories=STAGES, ordered=True)
        a6_stage = a6_stage.sort_values("stage")
        q_means = a6_stage["q_pred_mean"].values
        vb_means = a6_stage["VB_true_mean"].values
        q_order = bool(len(q_means) == 3 and q_means[0] < q_means[1] < q_means[2])
        vb_order = bool(len(vb_means) == 3 and vb_means[0] < vb_means[1] < vb_means[2])

        print("\nA6 q_pred_mean order under predicted stages:")
        print(a6_stage[["stage", "q_pred_mean"]].round(6).to_string(index=False))
        print(f"early < middle < late: {q_order}")

        print("\nA6 VB_true_mean order under predicted stages:")
        print(a6_stage[["stage", "VB_true_mean"]].round(6).to_string(index=False))
        print(f"early < middle < late: {vb_order}")


# =========================================================
# 5. Main
# =========================================================
def main() -> None:
    print("=" * 100)
    print("Chapter 5.4 probability-wear consistency analysis")
    print(f"ROOT    : {ROOT}")
    print(f"OUT_DIR : {OUT_DIR}")
    print("=" * 100)

    raw, src = read_probability_source()
    pred = standardize_ablation_probability_file(raw)
    pred["method"] = pred["method"].astype(str).str.strip()
    if "A6" not in set(pred["method"]):
        raise RuntimeError("A6/Final output was not found after standardization.")

    all_data = enrich_with_wear(pred)
    metrics, stage_stats, corr_df, boundary_df = compute_tables(all_data)

    a6 = all_data[all_data["method"] == "A6"].sort_values("run_id").copy()

    # Save per-run data.
    a6_cols = ["run_id", "VB_true", "VB_smooth", "q_true", "q_pred", "q_pred_norm", "true_stage", "pred_stage",
               "prob_early", "prob_middle", "prob_late", "max_prob", "dominant_prob_stage"]
    a6[a6_cols].to_csv(OUT_DIR / "Data_5_4_A6_probability_wear_trajectory.csv", index=False, encoding="utf-8-sig")
    all_cols = ["method", "output", "run_id", "q_true", "q_pred", "q_pred_norm", "true_stage", "pred_stage",
                "prob_early", "prob_middle", "prob_late", "max_prob"]
    all_data[all_cols].to_csv(OUT_DIR / "Data_5_4_all_ablation_probability_wear_trajectory.csv", index=False, encoding="utf-8-sig")

    # Save tables.
    metrics.to_csv(OUT_DIR / "Table_5_4_probability_wear_consistency_metrics.csv", index=False, encoding="utf-8-sig")
    stage_stats.to_csv(OUT_DIR / "Table_5_4_stagewise_degradation_statistics.csv", index=False, encoding="utf-8-sig")
    corr_df.to_csv(OUT_DIR / "Table_5_4_probability_q_correlations.csv", index=False, encoding="utf-8-sig")
    boundary_df.to_csv(OUT_DIR / "Table_5_4_boundary_smoothness_metrics.csv", index=False, encoding="utf-8-sig")
    save_additional_probability_stage_outputs(all_data, stage_stats)

    # Figures.
    plot_fig17(a6)
    plot_fig18(all_data)
    plot_fig19(a6, metrics)
    plot_fig20(a6)
    plot_fig21(a6)

    print("\nKey A6 results:")
    m = metrics[metrics["method"] == "A6"].iloc[0]
    c = corr_df[corr_df["method"] == "A6"].iloc[0]
    s = stage_stats[stage_stats["method"] == "A6"]
    print(f"A6 q-MAE  = {m['q-MAE']:.6f}")
    print(f"A6 q-RMSE = {m['q-RMSE']:.6f}")
    print(f"A6 q-R2   = {m['q-R2']:.6f}")
    print(f"A6 Spearman(q_true, q_pred) = {m['Spearman(q_true,q_pred)']:.6f}")
    print(f"A6 corr(p_early, q_true)  = {c['corr(p_early,q_true)']:.6f}")
    print(f"A6 corr(p_middle, q_true) = {c['corr(p_middle,q_true)']:.6f}")
    print(f"A6 corr(p_late, q_true)   = {c['corr(p_late,q_true)']:.6f}")
    print("A6 stagewise q_pred mean:")
    print(s[["stage", "q_pred_mean"]].to_string(index=False))
    print(f"A6 q-Smooth = {m['q-Smooth']:.6f}")
    print(f"A6 probability Smooth = {m['probability Smooth']:.6f}")
    print(f"\nPrediction source: {src}")
    print(f"Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

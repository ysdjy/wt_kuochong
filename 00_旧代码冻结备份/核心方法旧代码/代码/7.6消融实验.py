# -*- coding: utf-8 -*-
r"""
Ablation experiment for FGDS-PSI probability inference module.

This script is independent from the main experiment script. It reuses the same
data loading, no-leak online feature construction, train/validation/test split,
feature selection, GMM fine-state construction, TCN-GRU multi-task network and
random seed settings from the current FGDS-PSI main experiment code.

Only the probability output module is ablated:
    A1 Raw
    A2 Raw + Fine
    A3 Raw + Prior
    A4 Mix
    A5 Ordered
    A6 Final

Outputs:
    C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文\6_ablation_experiment
"""

from __future__ import annotations

from pathlib import Path
import os
import copy
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =========================================================
# 0. Output path and import main-experiment implementation
# =========================================================
OUT_ROOT = Path(
    os.environ.get(
        "FGDS_ABLATION_DIR",
        r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文\6_ablation_experiment",
    )
)
DIR_FIG = OUT_ROOT / "figures"
DIR_MODEL = OUT_ROOT / "models"
DIR_PRED = OUT_ROOT / "predictions"
DIR_INTERNAL = OUT_ROOT / "intermediate_main_outputs"
for d in [OUT_ROOT, DIR_FIG, DIR_MODEL, DIR_PRED, DIR_INTERNAL]:
    d.mkdir(parents=True, exist_ok=True)

# Make sure importing the main module never writes into the old main output dir.
os.environ.setdefault("FGDS_RUN_DIR", str(DIR_INTERNAL))

# Locate the main experiment code even when this script is copied to another folder.
SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_CODE_CANDIDATES = [
    SCRIPT_DIR / "main_experiment_3_fgds_psi_optimized.py",
    Path.cwd() / "main_experiment_3_fgds_psi_optimized.py",
    Path(r"C:\Users\wangting\Documents\Codex\2026-05-17\files-mentioned-by-the-user-docx\main_experiment_3_fgds_psi_optimized.py"),
    Path(r"D:\CodeTou-Download\pythonDemo\pythonDemo\阶段信息小论文\main_experiment_3_fgds_psi_optimized.py"),
    ]

MAIN_CODE_PATH = None
for p in MAIN_CODE_CANDIDATES:
    if p.exists():
        MAIN_CODE_PATH = p
        break

if MAIN_CODE_PATH is None:
    raise FileNotFoundError(
        "Cannot find main_experiment_3_fgds_psi_optimized.py. "
        "Please put it in the same folder as this ablation script, or update MAIN_CODE_CANDIDATES."
    )

sys.path.insert(0, str(MAIN_CODE_PATH.parent))
import main_experiment_3_fgds_psi_optimized as base  # noqa: E402


# Correct paths with proper Chinese characters for running outside this workspace.
base.ROOT = Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM")
base.FEATURE_FILE = base.ROOT / "PHM实验" / "1run_run_level_features" / "02_features" / "run_level_features_all.csv"
base.RUN_DIR = OUT_ROOT
base.DIR_RESULT = DIR_INTERNAL
base.DIR_MODEL = DIR_MODEL
base.DIR_FIG = DIR_FIG
base.DIR_PRED = DIR_PRED
for d in [base.DIR_RESULT, base.DIR_MODEL, base.DIR_FIG, base.DIR_PRED]:
    d.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1. Global style
# =========================================================
DPI = 900
STAGE_NAMES = base.STAGE_NAMES
STAGE_TO_ID = base.STAGE_TO_ID
ID_TO_STAGE = base.ID_TO_STAGE

METHODS = ["A1", "A2", "A3", "A4", "A5", "A6"]
OUTPUTS = ["raw", "raw_fine", "raw_prior", "mix", "ordered", "final"]
METHOD_NAME = {
    "A1": "Raw",
    "A2": "Raw + Fine",
    "A3": "Raw + Prior",
    "A4": "Mix",
    "A5": "Ordered",
    "A6": "Final",
}
OUTPUT_TO_METHOD = dict(zip(OUTPUTS, METHODS))

FIXED_MAIN_PROB_PARAMS = {
    "eta": 0.75,
    "fine_weight": 0.30,
    "temperature": 1.20,
    "mid_floor": 0.12,
    "late_tau": 0.66,
    "early_tau": 0.38,
    "order_blend": 0.25,
}

EXPECTED_A6 = {
    "Acc": 0.986842105,
    "Macro-F1": 0.987102093,
    "M-F1": 0.984375,
    "M→E": 0.023255814,
    "M→L": 0.0,
    "Smooth": 0.018776274,
}

COLOR_E = "#6AA84F"
COLOR_M = "#F4A261"
COLOR_L = "#C0504D"
COLOR_RED = "#B22222"
COLOR_BLUE = "#3C5CCF"
COLOR_GREEN = "#4FA36C"
COLOR_ORANGE = "#E39B2E"
COLOR_PURPLE = "#7E57C2"
COLOR_CYAN = "#4AA3A1"
COLOR_GRAY = "#8A8A8A"
COLOR_BLACK = "#222222"
COLOR_GRID = "#DADADA"
STAGE_BG = {"early": "#DDF0DD", "middle": "#FCE6C7", "late": "#F7D6D6"}
STAGE_COLORS = {"early": COLOR_E, "middle": COLOR_M, "late": COLOR_L}

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["legend.frameon"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# =========================================================
# 2. Metric and table helpers
# =========================================================
def fix_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize possible mojibake arrow columns from the imported main module."""
    rename = {}
    for col in df.columns:
        fixed = str(col).replace("M鈫扙", "M→E").replace("M鈫扡", "M→L")
        fixed = fixed.replace("M->E", "M→E").replace("M->L", "M→L")
        rename[col] = fixed
    return df.rename(columns=rename)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path}")


def metric_row(pred_df: pd.DataFrame, output: str) -> dict:
    row = base.manuscript_metric_row(
        pred_df,
        output,
        method_id=OUTPUT_TO_METHOD[output],
        split="test_C6",
    )
    row = fix_metric_columns(pd.DataFrame([row])).iloc[0].to_dict()
    keep = ["Method", "Output", "Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Pre", "M-Rec", "M→E", "M→L", "Rev", "Jump", "Smooth"]
    return {k: row[k] for k in keep}


def classification_report_rows(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y_true = pred_df["stage_true_id"].values.astype(int)
    for method, output in zip(METHODS, OUTPUTS):
        pred_col = "stage_pred_raw" if output == "raw" else f"stage_pred_{output}"
        y_pred = pred_df[pred_col].values.astype(int)
        rep = classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=STAGE_NAMES,
            output_dict=True,
            zero_division=0,
        )
        for label, item in rep.items():
            if isinstance(item, dict):
                rows.append({
                    "Method": method,
                    "Output": output,
                    "Stage": label,
                    "Precision": item.get("precision", np.nan),
                    "Recall": item.get("recall", np.nan),
                    "F1-score": item.get("f1-score", np.nan),
                    "Support": item.get("support", np.nan),
                })
            else:
                rows.append({
                    "Method": method,
                    "Output": output,
                    "Stage": label,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "F1-score": item,
                    "Support": len(y_true),
                })
    return pd.DataFrame(rows)


def confusion_rows(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y_true = pred_df["stage_true_id"].values.astype(int)
    for method, output in zip(METHODS, OUTPUTS):
        pred_col = "stage_pred_raw" if output == "raw" else f"stage_pred_{output}"
        y_pred = pred_df[pred_col].values.astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        row_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
        for i, true_stage in enumerate(STAGE_NAMES):
            for j, pred_stage in enumerate(STAGE_NAMES):
                rows.append({
                    "Method": method,
                    "True_stage": true_stage,
                    "Pred_stage": pred_stage,
                    "Count": int(cm[i, j]),
                    "Row_norm": float(row_norm[i, j]),
                })
    return pd.DataFrame(rows)


def build_probability_table(pred_test: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "condition": pred_test["condition"],
        "run_id": pred_test["cut_index"],
        "true_stage": pred_test["stage_true"],
        "q_hat": pred_test["q_hat"],
    })
    prefix_map = {
        "raw": "p_raw",
        "fine": "p_fine",
        "prior": "p_prior",
        "raw_fine": "p_raw_fine",
        "raw_prior": "p_raw_prior",
        "mix": "p_mix",
        "ordered": "alpha",
        "final": "p_final",
    }
    for source, prefix in prefix_map.items():
        for st, short in zip(STAGE_NAMES, ["E", "M", "L"]):
            col = f"raw_prob_{st}" if source == "raw" else f"{source}_prob_{st}"
            out[f"{prefix}_{short}"] = pred_test[col].values
    for method, output in zip(METHODS, OUTPUTS):
        col = "stage_pred_raw_name" if output == "raw" else f"stage_pred_{output}_name"
        out[f"pred_{method}"] = pred_test[col].values
    return out


def transition_matrix_string() -> str:
    A = np.array([[0.9400, 0.0550, 0.0050], [0.0050, 0.9550, 0.0400], [0.0005, 0.0100, 0.9895]])
    return np.array2string(A, precision=4, separator=", ")


# =========================================================
# 3. Plot helpers
# =========================================================
def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def polish(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_BLACK)
    ax.spines["bottom"].set_color(COLOR_BLACK)
    ax.tick_params(axis="both", colors=COLOR_BLACK, labelsize=10)
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.55, alpha=0.58, color=COLOR_GRID)
    ax.set_axisbelow(True)


def add_axis_arrows(ax: plt.Axes, x_pad: float = 0.025, y_pad: float = 0.035) -> None:
    ax.annotate(
        "",
        xy=(1.0 + x_pad, 0.0),
        xytext=(0.0, 0.0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color=COLOR_BLACK, shrinkA=0, shrinkB=0),
        clip_on=False,
        zorder=10,
    )
    ax.annotate(
        "",
        xy=(0.0, 1.0 + y_pad),
        xytext=(0.0, 0.0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color=COLOR_BLACK, shrinkA=0, shrinkB=0),
        clip_on=False,
        zorder=10,
    )


def add_a6_band(ax: plt.Axes, idx: int = 5, width: float = 0.86) -> None:
    ax.axvspan(idx - width / 2, idx + width / 2, color=COLOR_RED, alpha=0.08, zorder=0)


def stage_segments(stage_values: pd.Series | np.ndarray) -> list[tuple[int, int, str]]:
    values = [str(s) for s in stage_values]
    if not values:
        return []
    segs, start, current = [], 0, values[0]
    for i in range(1, len(values)):
        if values[i] != current:
            segs.append((start, i - 1, current))
            start, current = i, values[i]
    segs.append((start, len(values) - 1, current))
    return segs


def add_stage_background(ax: plt.Axes, x: np.ndarray, stage_values: pd.Series | np.ndarray) -> None:
    for s, e, st in stage_segments(stage_values):
        ax.axvspan(x[s] - 0.5, x[e] + 0.5, color=STAGE_BG.get(st, "#EFEFEF"), alpha=0.55, lw=0, zorder=0)


# =========================================================
# 4. Figures
# =========================================================
def plot_fig10_performance_profile(metric_df: pd.DataFrame) -> None:
    metrics = ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Pre", "M-Rec"]
    colors = [COLOR_GRAY, COLOR_GREEN, COLOR_ORANGE, COLOR_BLUE, COLOR_PURPLE, COLOR_RED]
    markers = ["o", "s", "^", "D", "v", "*"]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(9.8, 4.9))
    for i, (_, row) in enumerate(metric_df.iterrows()):
        method = row["Method"]
        lw = 2.8 if method == "A6" else 1.65
        ms = 12 if method == "A6" else 6.2
        ax.plot(
            x,
            row[metrics].values.astype(float),
            color=colors[i],
            marker=markers[i],
            linewidth=lw,
            markersize=ms,
            label=f"{method} {METHOD_NAME[method]}",
            zorder=5 if method == "A6" else 3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=0, fontstyle="normal")
    ax.set_ylim(0.45, 1.03)
    ax.set_ylabel("Score")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32), fontsize=8.8)
    polish(ax)
    add_axis_arrows(ax)
    savefig(DIR_FIG / "Fig10_ablation_performance_profile.png")


def add_bar_caps(ax: plt.Axes, xs: np.ndarray, heights: np.ndarray, width: float, color: str) -> None:
    for x, h in zip(xs, heights):
        ax.plot([x - width * 0.28, x + width * 0.28], [h, h], color=color, linewidth=1.0, zorder=5)


def plot_fig11_middle_analysis(metric_df: pd.DataFrame) -> None:
    methods = metric_df["Method"].tolist()
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    add_a6_band(ax)
    metrics = ["M-Pre", "M-Rec", "M-F1"]
    colors = [COLOR_BLUE, COLOR_GREEN, COLOR_ORANGE]
    hatches = ["////", "\\\\\\\\", "...."]
    w = 0.22
    for i, (metric, color, hatch) in enumerate(zip(metrics, colors, hatches)):
        xs = x + (i - 1) * w
        vals = metric_df[metric].values.astype(float)
        ax.bar(xs, vals, width=w, facecolor="white", edgecolor=color, linewidth=1.35, hatch=hatch, label=metric, zorder=3)
        add_bar_caps(ax, xs, vals, w, color)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Score")
    ax.set_ylim(0.88, 1.02)
    ax.set_title("(a) Middle-stage recognition", loc="left", fontweight="bold")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.27), fontsize=8.8)
    polish(ax)
    add_axis_arrows(ax)

    ax = axes[1]
    add_a6_band(ax)
    metrics = ["M→E", "M→L"]
    colors = [COLOR_BLUE, COLOR_RED]
    hatches = ["////", "\\\\\\\\"]
    w = 0.30
    for i, (metric, color, hatch) in enumerate(zip(metrics, colors, hatches)):
        xs = x + (i - 0.5) * w
        vals = metric_df[metric].values.astype(float)
        ax.bar(xs, vals, width=w, facecolor="white", edgecolor=color, linewidth=1.35, hatch=hatch, label=metric, zorder=3)
        add_bar_caps(ax, xs, vals, w, color)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Misclassification rate")
    ax.set_ylim(0, max(0.07, metric_df[metrics].values.max() * 1.25))
    ax.set_title("(b) Middle-stage misclassification", loc="left", fontweight="bold")
    ax.legend(ncol=2, loc="upper right", fontsize=8.8)
    polish(ax)
    add_axis_arrows(ax)

    savefig(DIR_FIG / "Fig11_ablation_middle_analysis.png")


def plot_fig12_consistency(metric_df: pd.DataFrame) -> None:
    methods = metric_df["Method"].tolist()
    x = np.arange(len(methods))
    fig, ax1 = plt.subplots(figsize=(8.6, 4.7))
    add_a6_band(ax1)
    w = 0.32
    ax1.bar(x - w / 2, metric_df["Rev"], width=w, color="#D7E3F4", edgecolor=COLOR_BLUE, linewidth=1.1, hatch="////", label="Rev")
    ax1.bar(x + w / 2, metric_df["Jump"], width=w, color="#F7D6D6", edgecolor=COLOR_RED, linewidth=1.1, hatch="\\\\\\\\", label="Jump")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.set_ylabel("Transition count")
    polish(ax1)
    add_axis_arrows(ax1)

    ax2 = ax1.twinx()
    ax2.plot(x, metric_df["Smooth"], color=COLOR_PURPLE, marker="o", linewidth=2.0, markersize=6, label="Smooth", zorder=4)
    ax2.scatter([5], [metric_df.loc[metric_df["Method"] == "A6", "Smooth"].iloc[0]], marker="*", s=160, color=COLOR_RED, edgecolor=COLOR_BLACK, linewidth=0.5, zorder=6)
    ax2.set_ylabel("Smooth")
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(axis="y", colors=COLOR_BLACK, labelsize=10)
    ax2.grid(False)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, ncol=3, loc="upper right", fontsize=9)
    savefig(DIR_FIG / "Fig12_ablation_consistency.png")


def plot_probability_panel(ax: plt.Axes, prob_df: pd.DataFrame, method: str, title: str) -> None:
    x = prob_df["run_id"].values
    add_stage_background(ax, x, prob_df["true_stage"])
    prefix = {
        "A1": "p_raw",
        "A4": "p_mix",
        "A5": "alpha",
        "A6": "p_final",
    }[method]
    ax.plot(x, prob_df[f"{prefix}_E"], color=COLOR_E, linewidth=1.8, label="E probability")
    ax.plot(x, prob_df[f"{prefix}_M"], color=COLOR_M, linewidth=1.8, label="M probability")
    ax.plot(x, prob_df[f"{prefix}_L"], color=COLOR_L, linewidth=1.8, label="L probability")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel("Probability")
    polish(ax)


def plot_fig13_probability_evolution(prob_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.3), sharex=True, sharey=True)
    panel_info = [
        ("A1", "A1 Raw"),
        ("A4", "A4 Mix"),
        ("A5", "A5 Ordered"),
        ("A6", "A6 Final"),
    ]
    for ax, (method, title) in zip(axes.ravel(), panel_info):
        plot_probability_panel(ax, prob_df, method, title)
    for ax in axes[-1, :]:
        ax.set_xlabel("Run index")
    axes[0, 0].legend(ncol=3, loc="upper center", bbox_to_anchor=(1.05, 1.23), fontsize=9)
    savefig(DIR_FIG / "Fig13_ablation_probability_evolution.png")


def plot_fig14_stage_sequence(prob_df: pd.DataFrame) -> None:
    rows = ["True", "A1", "A2", "A3", "A4", "A5", "A6"]
    run_ids = prob_df["run_id"].values
    ribbon = [prob_df["true_stage"].map(STAGE_TO_ID).values]
    for method in METHODS:
        ribbon.append(prob_df[f"pred_{method}"].map(STAGE_TO_ID).values)
    mat = np.vstack(ribbon).astype(float)
    cmap = ListedColormap([COLOR_E, COLOR_M, COLOR_L])
    fig, ax = plt.subplots(figsize=(11.0, 3.6))
    ax.imshow(
        mat,
        aspect="auto",
        cmap=cmap,
        vmin=-0.5,
        vmax=2.5,
        interpolation="nearest",
        extent=[run_ids.min() - 0.5, run_ids.max() + 0.5, len(rows) - 0.5, -0.5],
    )
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("Run index")
    ax.set_title("Ablation stage sequence comparison on C6", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color(COLOR_BLACK)
        spine.set_linewidth(0.8)
    handles = [Patch(facecolor=STAGE_COLORS[s], label=s) for s in STAGE_NAMES]
    ax.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.24), fontsize=9)
    savefig(DIR_FIG / "Fig14_ablation_stage_sequence.png")


# =========================================================
# 5. Main experiment flow
# =========================================================
def run_ablation_experiment() -> None:
    base.set_seed(base.RANDOM_SEED)
    print("=" * 110)
    print("FGDS-PSI ablation experiment: probability output module")
    print("=" * 110)
    print(f"Device      : {base.DEVICE}")
    print(f"Feature file: {base.FEATURE_FILE}")
    print(f"Output dir  : {OUT_ROOT}")

    # 1. Strictly reuse main-experiment data logic.
    raw_df = base.load_feature_table()
    label_df, _ = base.define_condition_relative_stages(raw_df)
    final_train_raw, final_val_raw, test_c6_raw = base.split_grouped_lifecycle(label_df)
    raw_cols = base.get_raw_numeric_sensor_cols(final_train_raw)

    split_feat = base.build_online_features_by_split({
        "final_train": final_train_raw,
        "final_internal_val": final_val_raw,
        "test_C6": test_c6_raw,
    }, raw_cols)
    feat_train = split_feat[split_feat["split_name_for_feature_build"] == "final_train"].copy()
    feat_val = split_feat[split_feat["split_name_for_feature_build"] == "final_internal_val"].copy()
    feat_test = split_feat[split_feat["split_name_for_feature_build"] == "test_C6"].copy()

    all_cols = base.feature_cols_from(feat_train)
    feat_train, feat_val = base.fill_by_train_median(feat_train, feat_val, all_cols)
    _, feat_test = base.fill_by_train_median(feat_train, feat_test, all_cols)

    selected, selected_df = base.select_features_train_only(feat_train)
    feat_train, feat_val = base.fill_by_train_median(feat_train, feat_val, selected)
    _, feat_test = base.fill_by_train_median(feat_train, feat_test, selected)

    gmm, raw_to_order = base.fit_train_gmm(feat_train)
    feat_train = base.assign_fine_states(feat_train, gmm, raw_to_order)
    feat_val = base.assign_fine_states(feat_val, gmm, raw_to_order)
    feat_test = base.assign_fine_states(feat_test, gmm, raw_to_order)

    scaler = StandardScaler().fit(feat_train[selected].values)
    for df in [feat_train, feat_val, feat_test]:
        df[selected] = np.nan_to_num(scaler.transform(df[selected].values), nan=0.0, posinf=0.0, neginf=0.0)

    L = base.BEST_ARCH["L"]
    tr_pack = base.make_pack(feat_train, selected, L, "final_train")
    va_pack = base.make_pack(feat_val, selected, L, "final_internal_val")
    te_pack = base.make_pack(feat_test, selected, L, "test_C6")

    # 2. Train the same multi-task TCN-GRU model as the main experiment.
    model, hist, best_score, best_epoch = base.train_model(tr_pack, va_pack, len(selected))
    hist.to_csv(OUT_ROOT / "training_history.csv", index=False, encoding="utf-8-sig")
    import torch
    torch.save(model.state_dict(), DIR_MODEL / "ablation_tcn_gru_multitask_best_model.pth")

    pred_val_raw = base.predict_model(model, va_pack)
    pred_test_raw = base.predict_model(model, te_pack)

    # 3. Use the final probability parameters fixed by the main experiment.
    # Do not re-search probability parameters in the ablation experiment.
    best_params = copy.deepcopy(FIXED_MAIN_PROB_PARAMS)
    save_csv(pd.DataFrame([best_params]), OUT_ROOT / "fixed_main_probability_params.csv")
    pred_test = base.apply_probability_inference(pred_test_raw, best_params)

    # 4. Tables.
    metric_df = pd.DataFrame([metric_row(pred_test, output) for output in OUTPUTS])
    save_csv(metric_df, OUT_ROOT / "Table10_ablation_summary.csv")
    save_csv(classification_report_rows(pred_test), OUT_ROOT / "Table11_ablation_classification_report.csv")
    save_csv(confusion_rows(pred_test), OUT_ROOT / "Table12_ablation_confusion_matrix.csv")
    prob_df = build_probability_table(pred_test)
    save_csv(prob_df, OUT_ROOT / "ablation_probabilities_test_C6.csv")

    summary = pd.DataFrame([{
        "train_conditions": "C1+C4 final_train blocks",
        "test_condition": "C6",
        "selected_features": len(selected),
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "eta": best_params["eta"],
        "fine_weight": best_params["fine_weight"],
        "temperature": best_params["temperature"],
        "mid_floor": best_params["mid_floor"],
        "late_tau": best_params["late_tau"],
        "early_tau": best_params["early_tau"],
        "order_blend": best_params["order_blend"],
        "transition_matrix": transition_matrix_string(),
        "probability_parameter_policy": "Fixed to the final main-experiment parameters; no ablation-side search.",
    }])
    save_csv(summary, OUT_ROOT / "ablation_experiment_summary.csv")

    # 5. Figures.
    plot_fig10_performance_profile(metric_df)
    plot_fig11_middle_analysis(metric_df)
    plot_fig12_consistency(metric_df)
    plot_fig13_probability_evolution(prob_df)
    plot_fig14_stage_sequence(prob_df)

    # 6. Console report.
    a4 = metric_df[metric_df["Method"] == "A4"].iloc[0].to_dict()
    a5 = metric_df[metric_df["Method"] == "A5"].iloc[0].to_dict()
    a6 = metric_df[metric_df["Method"] == "A6"].iloc[0].to_dict()
    a6_differs_from_a4 = not np.allclose(
        pred_test[[f"mix_prob_{s}" for s in STAGE_NAMES]].values,
        pred_test[[f"final_prob_{s}" for s in STAGE_NAMES]].values,
        atol=1e-12,
    )
    params_match = all(abs(float(best_params[k]) - float(FIXED_MAIN_PROB_PARAMS[k])) < 1e-12 for k in FIXED_MAIN_PROB_PARAMS)
    a6_match = all(abs(float(a6[k]) - float(EXPECTED_A6[k])) < (2e-3 if k == "Smooth" else 1e-6) for k in EXPECTED_A6)

    print("\nAblation experiment finished.\n")
    print(metric_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nA4 Mix:")
    print(f"Acc={a4['Acc']:.4f}, Macro-F1={a4['Macro-F1']:.4f}, M-F1={a4['M-F1']:.4f}, M→E={a4['M→E']:.4f}, M→L={a4['M→L']:.4f}, Smooth={a4['Smooth']:.6f}")
    print("\nA5 Ordered:")
    print(f"Acc={a5['Acc']:.4f}, Macro-F1={a5['Macro-F1']:.4f}, M-F1={a5['M-F1']:.4f}, M→E={a5['M→E']:.4f}, M→L={a5['M→L']:.4f}, Smooth={a5['Smooth']:.6f}")
    print("\nA6 Final:")
    print(f"Acc={a6['Acc']:.4f}, Macro-F1={a6['Macro-F1']:.4f}, M-F1={a6['M-F1']:.4f}, M→E={a6['M→E']:.4f}, M→L={a6['M→L']:.4f}, Smooth={a6['Smooth']:.6f}")
    print(f"\nA6 differs from A4: {a6_differs_from_a4}")
    print(f"Fixed probability parameters match main experiment: {params_match}")
    print(f"A6 matches expected B12 metrics: {a6_match}")
    if not a6_match:
        print("\nA6 consistency check details:")
        print(f"  Parameters used: {best_params}")
        print(f"  Expected A6/B12: {EXPECTED_A6}")
        print("  Current A6     : " + str({k: a6[k] for k in EXPECTED_A6}))
        print("  A6 formula     : p_final = (1 - order_blend) * p_mix + order_blend * alpha_t")
        print("  Check whether the training checkpoint is identical to the main experiment checkpoint; this script retrains the model unless you load the exact main checkpoint.")
    print(f"\nResults saved to:\n{OUT_ROOT}")


if __name__ == "__main__":
    run_ablation_experiment()

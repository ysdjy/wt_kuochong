# -*- coding: utf-8 -*-
r"""
Section 5.1 visualization suite for the main cross-condition comparison.

This script visualizes the comparison results of B1-B12 and produces multiple
candidate journal-style figures for manuscript selection.

Suggested captions:
Fig. 7A. Multi-metric performance profile of representative methods.
Fig. 7B. Comprehensive metric heatmap of all comparison methods.
Fig. 7C. Formal grouped-bar panel for overall and stage-wise performance.
Fig. 8A. Middle-stage recognition profile of representative methods.
Fig. 8B. Directional middle-stage misclassification comparison.
Fig. 8C. Row-normalized confusion matrices of representative methods.
Fig. 9A. Performance-consistency tradeoff of comparison methods.
Fig. 9B. Stage consistency comparison of comparison methods.
Fig. 9C. Pareto-style comparison between M-F1 and probability smoothness.
Fig. 10A. B12 stage probability evolution over the C6 lifecycle.
Fig. 10B. Probability evolution comparison between B11 and B12.
Fig. 10C. Local transition probability panels of B12.
Fig. 11. Stage timeline ribbon of true, B11, and B12 stage sequences.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Rectangle, Patch


# =========================================================
# 0. Paths and global style
# =========================================================
EXPERIMENT_ROOT = Path(
    os.environ.get(
        "CH5_COMPARISON_ROOT",
        r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文\4_comparison_experiment_recheck",
    )
)

LOCAL_ROOT = Path(r"C:\Users\wangting\Documents\Codex\2026-05-17\files-mentioned-by-the-user-docx")
LOCAL_EXPERIMENT_ROOT = LOCAL_ROOT / "_comparison_experiment_recheck"

OUT_DIR = Path(
    os.environ.get(
        "CH5_VIS_SUITE_OUT_DIR",
        r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文\5_figures_for_chapter5\5_1_visualization_suite",
    )
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 900
SAVE_PDF = False

COLOR_RED = "#D94B4B"
COLOR_BLUE = "#3C5CCF"
COLOR_GREEN = "#4FA36C"
COLOR_ORANGE = "#E39B2E"
COLOR_PURPLE = "#7E57C2"
COLOR_CYAN = "#4AA3A1"
COLOR_GRAY = "#8A8A8A"
COLOR_BLACK = "#222222"
COLOR_B12 = "#B22222"

COLOR_EARLY = "#6AA84F"
COLOR_MIDDLE = "#F4A261"
COLOR_LATE = "#C0504D"
COLOR_GRID = "#DADADA"

METHOD_ORDER = [f"B{i}" for i in range(1, 13)]
STAGES = ["early", "middle", "late"]
STAGE_TO_ID = {"early": 0, "middle": 1, "late": 2}
ID_TO_STAGE = {0: "early", 1: "middle", 2: "late"}
STAGE_COLORS = {"early": COLOR_EARLY, "middle": COLOR_MIDDLE, "late": COLOR_LATE}

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["legend.frameon"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


TABLE_A = """Method,Method_name,Stage_definition,Model_type,Acc,Macro-F1,E-F1,M-F1,L-F1,M-Pre,M-Rec,M→E,M→L,Rev,Jump,Smooth
B1,Fixed-stage Rule,Fixed-stage,wear-threshold reference,0.6447,0.6145,0.4587,0.5970,0.7879,0.5755,0.6202,0.0000,0.3798,0,0,0.0132
B2,Relative-stage Proxy Rule,Relative-stage,sensor-derived q_proxy rule,0.5757,0.4871,0.7304,0.0000,0.7309,0.0000,0.0000,0.4806,0.5194,5,11,0.0726
B3,Fixed-stage RF,Fixed-stage,RF,0.9474,0.9490,0.9231,0.9350,0.9889,0.9829,0.8915,0.1085,0.0000,5,0,0.1096
B4,Relative-stage SVM,Relative-stage,SVM,0.8651,0.8695,0.8737,0.8379,0.8970,0.8548,0.8217,0.1783,0.0000,9,0,0.1578
B5,Relative-stage RF,Relative-stage,RF,0.9770,0.9773,0.9651,0.9723,0.9945,0.9919,0.9535,0.0388,0.0078,5,0,0.1105
B6,Relative-stage XGBoost,Relative-stage,xgboost,0.8783,0.8765,0.8054,0.8702,0.9540,0.7949,0.9612,0.0388,0.0000,9,0,0.1390
B7,Relative-stage MLP,Relative-stage,MLP,0.8618,0.8651,0.8889,0.8174,0.8889,0.9307,0.7287,0.1628,0.1085,16,0,0.1554
B8,Relative-stage TCN,Relative-stage,TCN,0.9539,0.9550,0.9385,0.9426,0.9838,1.0000,0.8915,0.0853,0.0233,10,0,0.1245
B9,Relative-stage GRU,Relative-stage,GRU,0.9539,0.9560,0.9825,0.9426,0.9430,1.0000,0.8915,0.0233,0.0853,1,0,0.0398
B10,Relative-stage TCN-GRU,Relative-stage,TCN-GRU,0.8684,0.8746,0.8317,0.8261,0.9659,0.9406,0.7364,0.2636,0.0000,0,0,0.0219
B11,Multi-task TCN-GRU,Relative-stage,TCN-GRU + auxiliary heads,0.9901,0.9902,0.9825,0.9882,1.0000,1.0000,0.9767,0.0233,0.0000,0,0,0.0236
B12,FGDS-PSI,Relative-stage,Proposed,0.9868,0.9871,0.9825,0.9844,0.9945,0.9921,0.9767,0.0233,0.0000,0,0,0.0188
"""

TABLE_B = """Method,True_stage,Pred_stage,count,row_norm
B1,early,early,25,0.2976
B1,early,middle,59,0.7024
B1,early,late,0,0
B1,middle,early,0,0
B1,middle,middle,80,0.6202
B1,middle,late,49,0.3798
B1,late,early,0,0
B1,late,middle,0,0
B1,late,late,91,1
B2,early,early,84,1
B2,early,middle,0,0
B2,early,late,0,0
B2,middle,early,62,0.4806
B2,middle,middle,0,0
B2,middle,late,67,0.5194
B2,late,early,0,0
B2,late,middle,0,0
B2,late,late,91,1
B3,early,early,84,1
B3,early,middle,0,0
B3,early,late,0,0
B3,middle,early,14,0.1085
B3,middle,middle,115,0.8915
B3,middle,late,0,0
B3,late,early,0,0
B3,late,middle,2,0.022
B3,late,late,89,0.978
B4,early,early,83,0.9881
B4,early,middle,1,0.0119
B4,early,late,0,0
B4,middle,early,23,0.1783
B4,middle,middle,106,0.8217
B4,middle,late,0,0
B4,late,early,0,0
B4,late,middle,17,0.1868
B4,late,late,74,0.8132
B5,early,early,83,0.9881
B5,early,middle,1,0.0119
B5,early,late,0,0
B5,middle,early,5,0.0388
B5,middle,middle,123,0.9535
B5,middle,late,1,0.0078
B5,late,early,0,0
B5,late,middle,0,0
B5,late,late,91,1
B6,early,early,60,0.7143
B6,early,middle,24,0.2857
B6,early,late,0,0
B6,middle,early,5,0.0388
B6,middle,middle,124,0.9612
B6,middle,late,0,0
B6,late,early,0,0
B6,late,middle,8,0.0879
B6,late,late,83,0.9121
B7,early,early,84,1
B7,early,middle,0,0
B7,early,late,0,0
B7,middle,early,21,0.1628
B7,middle,middle,94,0.7287
B7,middle,late,14,0.1085
B7,late,early,0,0
B7,late,middle,7,0.0769
B7,late,late,84,0.9231
B8,early,early,84,1
B8,early,middle,0,0
B8,early,late,0,0
B8,middle,early,11,0.0853
B8,middle,middle,115,0.8915
B8,middle,late,3,0.0233
B8,late,early,0,0
B8,late,middle,0,0
B8,late,late,91,1
B9,early,early,84,1
B9,early,middle,0,0
B9,early,late,0,0
B9,middle,early,3,0.0233
B9,middle,middle,115,0.8915
B9,middle,late,11,0.0853
B9,late,early,0,0
B9,late,middle,0,0
B9,late,late,91,1
B10,early,early,84,1
B10,early,middle,0,0
B10,early,late,0,0
B10,middle,early,34,0.2636
B10,middle,middle,95,0.7364
B10,middle,late,0,0
B10,late,early,0,0
B10,late,middle,6,0.0659
B10,late,late,85,0.9341
B11,early,early,84,1
B11,early,middle,0,0
B11,early,late,0,0
B11,middle,early,3,0.0233
B11,middle,middle,126,0.9767
B11,middle,late,0,0
B11,late,early,0,0
B11,late,middle,0,0
B11,late,late,91,1
B12,early,early,84,1
B12,early,middle,0,0
B12,early,late,0,0
B12,middle,early,3,0.0233
B12,middle,middle,126,0.9767
B12,middle,late,0,0
B12,late,early,0,0
B12,late,middle,1,0.011
B12,late,late,90,0.989
"""


# =========================================================
# 1. Data loading helpers
# =========================================================
def existing_file(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def normalize_arrow_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        fixed = str(col).replace("M鈫扙", "M→E").replace("M鈫扡", "M→L")
        fixed = fixed.replace("M->E", "M→E").replace("M->L", "M→L")
        if fixed != col:
            rename[col] = fixed
    return df.rename(columns=rename)


def load_summary_data() -> pd.DataFrame:
    candidates = [
        EXPERIMENT_ROOT / "FINAL_comparison_results.csv",
        EXPERIMENT_ROOT / "1_results" / "FINAL_comparison_results.csv",
        LOCAL_EXPERIMENT_ROOT / "FINAL_comparison_results.csv",
        LOCAL_EXPERIMENT_ROOT / "1_results" / "FINAL_comparison_results.csv",
        ]
    path = existing_file(candidates)
    if path is None:
        df = pd.read_csv(StringIO(TABLE_A))
        print("Loaded embedded Table A.")
    else:
        df = pd.read_csv(path)
        print(f"Loaded summary data: {path}")
    df = normalize_arrow_columns(df)
    df["Method"] = pd.Categorical(df["Method"], categories=METHOD_ORDER, ordered=True)
    df = df.sort_values("Method").reset_index(drop=True)
    numeric_cols = ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Pre", "M-Rec", "M→E", "M→L", "Rev", "Jump", "Smooth"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_confusion_data() -> pd.DataFrame:
    candidates = [
        EXPERIMENT_ROOT / "FINAL_comparison_confusion_matrices_long.csv",
        EXPERIMENT_ROOT / "1_results" / "FINAL_comparison_confusion_matrices_long.csv",
        LOCAL_EXPERIMENT_ROOT / "FINAL_comparison_confusion_matrices_long.csv",
        LOCAL_EXPERIMENT_ROOT / "1_results" / "FINAL_comparison_confusion_matrices_long.csv",
        ]
    path = existing_file(candidates)
    if path is None:
        df = pd.read_csv(StringIO(TABLE_B))
        print("Loaded embedded Table B.")
    else:
        df = pd.read_csv(path)
        print(f"Loaded confusion data: {path}")
    # Support both old column names and embedded names.
    df = df.rename(columns={"true_stage": "True_stage", "pred_stage": "Pred_stage"})
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    df["row_norm"] = pd.to_numeric(df["row_norm"], errors="coerce").fillna(0.0)
    return df


def find_prediction_table() -> Path | None:
    candidates = [
        EXPERIMENT_ROOT / "FINAL_comparison_predictions.csv",
        EXPERIMENT_ROOT / "1_results" / "FINAL_comparison_predictions.csv",
        LOCAL_EXPERIMENT_ROOT / "FINAL_comparison_predictions.csv",
        LOCAL_EXPERIMENT_ROOT / "1_results" / "FINAL_comparison_predictions.csv",
        ]
    path = existing_file(candidates)
    if path is not None:
        return path
    for root in [EXPERIMENT_ROOT, LOCAL_EXPERIMENT_ROOT]:
        if root.exists():
            found = list(root.rglob("FINAL_comparison_predictions.csv"))
            if found:
                return found[0]
    return None


def to_standard_sequence(pred_df: pd.DataFrame, method: str) -> pd.DataFrame | None:
    prob_cols = [f"prob_E_{method}", f"prob_M_{method}", f"prob_L_{method}"]
    pred_col = f"pred_{method}"
    if pred_col not in pred_df.columns or any(c not in pred_df.columns for c in prob_cols):
        return None
    run_col = "run_id_end" if "run_id_end" in pred_df.columns else ("cut_index" if "cut_index" in pred_df.columns else None)
    if run_col is None:
        return None
    true_col = "true_stage" if "true_stage" in pred_df.columns else ("stage_true" if "stage_true" in pred_df.columns else None)
    if true_col is None:
        return None
    out = pd.DataFrame({
        "run_id": pd.to_numeric(pred_df[run_col], errors="coerce").astype(int),
        "true_stage": pred_df[true_col].astype(str),
        "pred_stage": pred_df[pred_col].astype(str),
        "p_early": pd.to_numeric(pred_df[prob_cols[0]], errors="coerce"),
        "p_middle": pd.to_numeric(pred_df[prob_cols[1]], errors="coerce"),
        "p_late": pd.to_numeric(pred_df[prob_cols[2]], errors="coerce"),
    }).sort_values("run_id").reset_index(drop=True)
    return out


def find_sequence_files() -> dict[str, pd.DataFrame]:
    """Load or export B10/B11/B12 per-run probability tables if possible."""
    seq = {}
    pred_path = find_prediction_table()
    if pred_path is None:
        note = (
            "Sequence-level plots were skipped because no FINAL_comparison_predictions.csv file was found.\n"
            "Needed columns: run_id_end (or cut_index), true_stage, pred_B10/B11/B12, "
            "prob_E_B10/B11/B12, prob_M_B10/B11/B12, prob_L_B10/B11/B12.\n"
        )
        (OUT_DIR / "missing_sequence_files_note.txt").write_text(note, encoding="utf-8")
        print("No comparison prediction file found. Sequence figures will be skipped.")
        return seq

    pred_df = pd.read_csv(pred_path)
    print(f"Loaded prediction table: {pred_path}")
    for method in ["B10", "B11", "B12"]:
        data = to_standard_sequence(pred_df, method)
        if data is not None:
            seq[method] = data
            out_csv = OUT_DIR / f"{method}_per_run_probability.csv"
            data.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"Exported per-run probability: {out_csv}")
    if "B12" not in seq:
        note = (
            "Sequence-level plots were skipped because B12 per-run probability columns were not found.\n"
            "Needed columns: run_id_end (or cut_index), true_stage, pred_B12, prob_E_B12, prob_M_B12, prob_L_B12.\n"
        )
        (OUT_DIR / "missing_sequence_files_note.txt").write_text(note, encoding="utf-8")
    return seq


# =========================================================
# 2. Common plotting helpers
# =========================================================
def save_figure(fig: plt.Figure, stem: str) -> None:
    """
    Save figure as PNG only.
    If Windows raises OSError for a long/unusual path, fall back to a shorter safe filename.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    png = OUT_DIR / f"{stem}.png"

    try:
        fig.savefig(str(png), dpi=DPI, bbox_inches="tight")
        print(f"Saved: {png}")
    except OSError as e:
        print(f"Warning: failed to save with original filename:\n{png}")
        print(f"Reason: {e}")

        safe_stem = (
            stem.replace("accuracy_smoothness_tradeoff", "tradeoff")
            .replace("probability_smoothness", "smoothness")
            .replace("confusion_matrix_panels", "cm_panels")
            .replace("overall_multimetric_profile", "profile")
        )
        safe_png = OUT_DIR / f"{safe_stem}.png"

        fig.savefig(str(safe_png), dpi=DPI, bbox_inches="tight")
        print(f"Saved with fallback filename: {safe_png}")

    plt.close(fig)

def polish_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_BLACK)
    ax.spines["bottom"].set_color(COLOR_BLACK)
    ax.tick_params(axis="both", colors=COLOR_BLACK, labelsize=10)
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.55, alpha=0.6, color=COLOR_GRID)
    ax.set_axisbelow(True)


def add_axis_arrows(ax: plt.Axes, x_pad: float = 0.025, y_pad: float = 0.035) -> None:
    """Add subtle arrowheads to the positive x/y axes in axes-fraction coordinates."""
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


def add_y_axis_break(ax: plt.Axes) -> None:
    kwargs = dict(transform=ax.transAxes, color=COLOR_BLACK, clip_on=False, linewidth=0.9)
    ax.plot((-0.012, 0.012), (0.045, 0.075), **kwargs)
    ax.plot((-0.012, 0.012), (0.080, 0.110), **kwargs)


def mark_b12_xtick(ax: plt.Axes) -> None:
    for tick in ax.get_xticklabels():
        if tick.get_text() == "B12":
            tick.set_color(COLOR_B12)
            tick.set_fontweight("bold")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.01, 0.98, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=13, fontweight="bold", color=COLOR_BLACK)


def b12_background(ax: plt.Axes, x_pos: float = 11.0, width: float = 0.9) -> None:
    ax.axvspan(x_pos - width / 2, x_pos + width / 2, color=COLOR_B12, alpha=0.07, zorder=0)


def stage_segments(stage_values: pd.Series | np.ndarray) -> list[tuple[int, int, str]]:
    stages = [str(s) for s in stage_values]
    if not stages:
        return []
    segs = []
    start = 0
    current = stages[0]
    for i in range(1, len(stages)):
        if stages[i] != current:
            segs.append((start, i - 1, current))
            start = i
            current = stages[i]
    segs.append((start, len(stages) - 1, current))
    return segs


def add_true_stage_background(ax: plt.Axes, x: np.ndarray, true_stage: pd.Series) -> None:
    for start, end, st in stage_segments(true_stage):
        ax.axvspan(x[start] - 0.5, x[end] + 0.5, color=STAGE_COLORS.get(st, COLOR_GRAY), alpha=0.12, lw=0, zorder=0)


# =========================================================
# 3. Fig. 7
# =========================================================
def plot_fig7a_profile(df: pd.DataFrame) -> None:
    metrics = ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Pre", "M-Rec"]
    selected = [f"B{i}" for i in range(3, 13)]
    palette = {
        "B1": ("#5F6368", "o", ":", 1.35),
        "B2": ("#8A6F3D", "s", "--", 1.35),
        "B3": ("#7F7F7F", "^", "-.", 1.25),
        "B4": ("#8C6BB1", "D", ":", 1.25),
        "B5": (COLOR_GREEN, "v", "-", 1.45),
        "B6": (COLOR_ORANGE, "P", "--", 1.25),
        "B7": ("#B279A2", "X", "-.", 1.25),
        "B8": (COLOR_BLUE, "<", ":", 1.45),
        "B9": (COLOR_CYAN, ">", "--", 1.45),
        "B10": (COLOR_PURPLE, "h", "-.", 1.45),
        "B11": (COLOR_BLACK, "p", "-", 2.0),
        "B12": (COLOR_B12, "*", "-", 2.9),
    }
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(10.2, 5.15))
    for method in selected:
        row = df[df["Method"].astype(str) == method].iloc[0]
        color, marker, ls, lw = palette[method]
        ms = 11 if method == "B12" else 5.1
        alpha = 1.0 if method in ["B11", "B12"] else 0.82
        ax.plot(x, row[metrics].values.astype(float), color=color, marker=marker, linestyle=ls,
                linewidth=lw, markersize=ms, alpha=alpha, label=method, zorder=5 if method == "B12" else 3)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=0, fontstyle="normal")
    ax.set_ylim(0.50, 1.03)
    ax.set_ylabel("Score")
    ax.legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.27), fontsize=8.6)
    polish_axis(ax)
    add_axis_arrows(ax)
    save_figure(fig, "Fig7A_overall_multimetric_profile")


def plot_fig7b_heatmap(df: pd.DataFrame) -> None:
    heat = df.copy()
    heat["1-M→E"] = 1.0 - heat["M→E"]
    heat["1-M→L"] = 1.0 - heat["M→L"]
    metrics = ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Pre", "M-Rec", "1-M→E", "1-M→L"]
    mat = heat[metrics].values.astype(float)
    cmap = LinearSegmentedColormap.from_list("journal_blue_red", ["#F8FBFF", "#C7D8EE", "#5B7DBA", "#B22222"])
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["Method"].astype(str))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if mat[i, j] > 0.72 else COLOR_BLACK
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color=color)
    b12_idx = df.index[df["Method"].astype(str) == "B12"][0]
    ax.add_patch(Rectangle((-0.5, b12_idx - 0.5), len(metrics), 1, fill=False, edgecolor=COLOR_B12, linewidth=2.2))
    ax.set_title("Fig. 7B Comprehensive metric heatmap", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=9)
    save_figure(fig, "Fig7B_overall_metric_heatmap")


def plot_fig7c_barpanel(df: pd.DataFrame) -> None:
    methods = df["Method"].astype(str).tolist()
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.6))

    ax = axes[0]
    w = 0.34
    b12_background(ax)
    ax.bar(x - w / 2, df["Acc"], width=w, color=COLOR_BLUE, edgecolor=COLOR_BLACK, linewidth=0.35, label="Acc")
    ax.bar(x + w / 2, df["Macro-F1"], width=w, color=COLOR_RED, edgecolor=COLOR_BLACK, linewidth=0.35, label="Macro-F1")
    ax.set_ylim(0.45, 1.04)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0)
    ax.set_ylabel("Score")
    ax.set_title("(a) Overall performance", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    polish_axis(ax)
    mark_b12_xtick(ax)

    ax = axes[1]
    w = 0.25
    b12_background(ax)
    ax.bar(x - w, df["E-F1"], width=w, color=COLOR_EARLY, edgecolor=COLOR_BLACK, linewidth=0.25, label="E-F1")
    ax.bar(x, df["M-F1"], width=w, color=COLOR_MIDDLE, edgecolor=COLOR_BLACK, linewidth=0.25, label="M-F1")
    ax.bar(x + w, df["L-F1"], width=w, color=COLOR_LATE, edgecolor=COLOR_BLACK, linewidth=0.25, label="L-F1")
    ax.set_ylim(0, 1.04)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0)
    ax.set_title("(b) Stage-wise F1-score", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    polish_axis(ax)
    mark_b12_xtick(ax)

    fig.suptitle("Fig. 7C Formal performance comparison panel", y=1.02, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "Fig7C_overall_bar_panel")


# =========================================================
# 4. Fig. 8
# =========================================================
def plot_fig8a_middle_profile(df: pd.DataFrame) -> None:
    metrics = ["M-Pre", "M-Rec", "M-F1", "1-M→E", "1-M→L"]
    selected = ["B5", "B8", "B9", "B11", "B12"]
    tmp = df.copy()
    tmp["1-M→E"] = 1.0 - tmp["M→E"]
    tmp["1-M→L"] = 1.0 - tmp["M→L"]
    colors = {"B5": COLOR_GREEN, "B8": COLOR_BLUE, "B9": COLOR_CYAN, "B11": COLOR_BLACK, "B12": COLOR_B12}
    markers = {"B5": "o", "B8": "s", "B9": "D", "B11": "^", "B12": "*"}
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for method in selected:
        row = tmp[tmp["Method"].astype(str) == method].iloc[0]
        ax.plot(x, row[metrics].values.astype(float), marker=markers[method], color=colors[method],
                linewidth=2.7 if method == "B12" else 1.65,
                markersize=11 if method == "B12" else 5.7, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.65, 1.03)
    ax.set_ylabel("Score")
    ax.set_title("Fig. 8A Middle-stage focused profile", fontweight="bold")
    polish_axis(ax)
    save_figure(fig, "Fig8A_middle_stage_profile")


def plot_fig8b_misdirection(df: pd.DataFrame) -> None:
    methods = df["Method"].astype(str).tolist()
    x = np.arange(len(methods))
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.6), sharex=True)
    for ax, metric, color, title in [
        (axes[0], "M→E", COLOR_BLUE, "(a) Middle misclassified as early"),
        (axes[1], "M→L", COLOR_RED, "(b) Middle misclassified as late"),
    ]:
        b12_background(ax)
        vals = df[metric].values.astype(float)
        markerline, stemlines, baseline = ax.stem(x, vals, basefmt=" ", linefmt=color, markerfmt="o")
        plt.setp(stemlines, linewidth=1.7, color=color, alpha=0.85)
        plt.setp(markerline, markersize=6.5, markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.5)
        ax.scatter([11], [vals[11]], marker="*", s=170, color=COLOR_B12, edgecolor=COLOR_BLACK, linewidth=0.4, zorder=5)
        if metric == "M→L":
            ax.text(11, vals[11] + 0.018, "0", ha="center", va="bottom", color=COLOR_B12, fontweight="bold")
        ax.set_ylim(0, 0.56)
        ax.set_ylabel("Rate")
        ax.set_title(title, fontweight="bold", loc="left")
        polish_axis(ax)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    mark_b12_xtick(axes[1])
    fig.suptitle("Fig. 8B Middle-stage misclassification direction", y=1.01, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "Fig8B_middle_misclassification_direction")


def confusion_matrix_for(conf_df: pd.DataFrame, method: str) -> tuple[np.ndarray, np.ndarray]:
    sub = conf_df[conf_df["Method"].astype(str) == method]
    mat = np.zeros((3, 3), dtype=float)
    cnt = np.zeros((3, 3), dtype=int)
    for _, row in sub.iterrows():
        i = STAGE_TO_ID[str(row["True_stage"])]
        j = STAGE_TO_ID[str(row["Pred_stage"])]
        mat[i, j] = float(row["row_norm"])
        cnt[i, j] = int(row["count"])
    return mat, cnt


def plot_fig8c_confusion_panels(conf_df: pd.DataFrame) -> None:
    methods = ["B1", "B5", "B8", "B12"]

    fig, axes = plt.subplots(
        1, 4,
        figsize=(13.2, 3.65),
        sharex=True,
        sharey=True,
        constrained_layout=False
    )

    # Softer journal-style cyan-blue to red-brown colormap.
    cmap = LinearSegmentedColormap.from_list(
        "cm_journal_soft",
        ["#FFFFFF", "#D9EEF3", "#7FB8C9", "#4F6FAE", "#8C1D18"]
    )

    im = None

    for ax, method, label in zip(axes, methods, ["(a)", "(b)", "(c)", "(d)"]):
        mat, cnt = confusion_matrix_for(conf_df, method)
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1)

        ax.set_title(
            f"{label} {method}",
            fontweight="bold",
            loc="left",
            fontsize=11.5
        )

        ax.set_xticks(range(3))
        ax.set_xticklabels(["E", "M", "L"], fontsize=10)
        ax.set_yticks(range(3))
        ax.set_yticklabels(["E", "M", "L"], fontsize=10)

        for i in range(3):
            for j in range(3):
                text_color = "white" if mat[i, j] >= 0.58 else COLOR_BLACK
                ax.text(
                    j,
                    i,
                    f"{cnt[i, j]}\n{mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10.2,
                    color=text_color
                )

        ax.set_xlabel("Predicted stage", fontsize=10.5)

        for spine in ax.spines.values():
            spine.set_color(COLOR_BLACK)
            spine.set_linewidth(0.8)

    axes[0].set_ylabel("True stage", fontsize=10.5)

    fig.subplots_adjust(
        left=0.055,
        right=0.875,
        bottom=0.18,
        top=0.88,
        wspace=0.08
    )

    cax = fig.add_axes([0.905, 0.18, 0.012, 0.70])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Row-normalized value", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    save_figure(fig, "Fig8C_confusion_matrix_panels")

# =========================================================
# 5. Fig. 9
# =========================================================
def model_group(row: pd.Series) -> str:
    text = f"{row.get('Model_type', '')} {row.get('Method_name', '')}".lower()
    if "rule" in text or "reference" in text:
        return "Rule"
    if any(k in text for k in ["rf", "svm", "xgboost", "mlp"]):
        return "Traditional ML"
    if any(k in text for k in ["tcn", "gru"]):
        return "Deep sequence"
    return "Other"


def plot_fig9a_tradeoff(df: pd.DataFrame) -> None:
    group_colors = {"Rule": COLOR_GRAY, "Traditional ML": COLOR_GREEN, "Deep sequence": COLOR_BLUE, "Other": COLOR_PURPLE}
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for _, row in df.iterrows():
        method = str(row["Method"])
        group = model_group(row)
        size = 58 + 16 * max(1, int(row["Rev"] + row["Jump"] + 1))
        color = COLOR_B12 if method == "B12" else group_colors.get(group, COLOR_GRAY)
        edge = COLOR_BLACK if method == "B12" else "white"
        ax.scatter(row["Smooth"], row["Macro-F1"], s=size, color=color, alpha=0.88, edgecolor=edge, linewidth=1.1, zorder=4)
        ax.text(row["Smooth"] + 0.0025, row["Macro-F1"] + 0.003, method, fontsize=9.5,
                color=COLOR_B12 if method == "B12" else COLOR_BLACK,
                fontweight="bold" if method == "B12" else "normal")
    handles = [Patch(facecolor=c, label=g) for g, c in group_colors.items()]
    handles.append(Patch(facecolor=COLOR_B12, label="B12"))
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    ax.set_xlabel("Smoothness (lower is smoother)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Fig. 9A Accuracy-smoothness tradeoff", fontweight="bold")
    ax.set_xlim(0, max(0.18, df["Smooth"].max() + 0.02))
    ax.set_ylim(0.45, 1.02)
    polish_axis(ax, grid_axis="both")
    save_figure(fig, "Fig9A_accuracy_smoothness_tradeoff")


def plot_fig9b_consistency(df: pd.DataFrame) -> None:
    methods = df["Method"].astype(str).tolist()
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
    ax = axes[0]
    w = 0.34
    b12_background(ax)
    ax.bar(x - w / 2, df["Rev"], width=w, color="#5B7DBA", edgecolor=COLOR_BLACK, linewidth=0.35, label="Rev")
    ax.bar(x + w / 2, df["Jump"], width=w, color=COLOR_RED, edgecolor=COLOR_BLACK, linewidth=0.35, label="Jump")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Count")
    ax.set_title("(a) Reverse and non-adjacent transitions", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    polish_axis(ax)
    mark_b12_xtick(ax)

    ax = axes[1]
    colors = [COLOR_GRAY] * len(df)
    colors[-1] = COLOR_BLUE
    b12_background(ax)
    ax.bar(x, df["Smooth"], color=colors, edgecolor=COLOR_BLACK, linewidth=0.35)
    for idx in [10, 11]:
        ax.text(idx, df["Smooth"].iloc[idx] + 0.004, f"{df['Smooth'].iloc[idx]:.4f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Smoothness")
    ax.set_title("(b) Probability smoothness", fontweight="bold")
    ax.text(0.02, 0.96, "Lower is smoother; interpret with accuracy.", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color="#555555")
    polish_axis(ax)
    mark_b12_xtick(ax)

    fig.suptitle("Fig. 9B Stage consistency comparison", y=1.02, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "Fig9B_consistency_panel")


def plot_fig9c_pareto(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    x = df["Smooth"].values.astype(float)
    y = df["M-F1"].values.astype(float)
    for _, row in df.iterrows():
        method = str(row["Method"])
        color = COLOR_B12 if method == "B12" else (COLOR_BLACK if method == "B11" else COLOR_GRAY)
        marker = "*" if method == "B12" else ("D" if method == "B11" else "o")
        size = 165 if method == "B12" else (80 if method == "B11" else 45)
        ax.scatter(row["Smooth"], row["M-F1"], s=size, color=color, marker=marker, edgecolor=COLOR_BLACK, linewidth=0.45, zorder=4)
        ax.text(row["Smooth"] + 0.0022, row["M-F1"] + 0.004, method, fontsize=9, fontweight="bold" if method in ["B11", "B12"] else "normal")

    # A light Pareto-like boundary: sort by smoothness and keep increasing M-F1.
    order = np.argsort(x)
    best = -np.inf
    frontier = []
    for idx in order:
        if y[idx] > best:
            frontier.append(idx)
            best = y[idx]
    if len(frontier) >= 2:
        ax.plot(x[frontier], y[frontier], color=COLOR_B12, linestyle="--", linewidth=1.2, alpha=0.65, label="Pareto-like envelope")
        ax.legend(loc="lower right", fontsize=9)
    ax.set_xlabel("Smoothness (lower is better)")
    ax.set_ylabel("Middle-stage F1")
    ax.set_title("Fig. 9C Pareto-style performance-consistency view", fontweight="bold")
    ax.set_xlim(0, max(0.18, df["Smooth"].max() + 0.02))
    ax.set_ylim(0, 1.03)
    polish_axis(ax, grid_axis="both")
    save_figure(fig, "Fig9C_pareto_front_style")


# =========================================================
# 6. Fig. 10 and Fig. 11 sequence-level figures
# =========================================================
def plot_probability_evolution(ax: plt.Axes, seq: pd.DataFrame, title: str) -> None:
    x = seq["run_id"].values
    add_true_stage_background(ax, x, seq["true_stage"])
    ax.plot(x, seq["p_early"], color=COLOR_EARLY, linewidth=2.0, label="early")
    ax.plot(x, seq["p_middle"], color=COLOR_MIDDLE, linewidth=2.0, label="middle")
    ax.plot(x, seq["p_late"], color=COLOR_LATE, linewidth=2.0, label="late")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Stage probability")
    if title:
        ax.set_title(title, fontweight="bold", loc="left")
    polish_axis(ax)
    add_axis_arrows(ax)


def plot_fig10a_b12_evolution(seq_data: dict[str, pd.DataFrame]) -> None:
    if "B12" not in seq_data:
        print("Skipped Fig10A: B12 sequence data not available.")
        return
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    plot_probability_evolution(ax, seq_data["B12"], "")
    ax.set_xlabel("Run index")
    save_figure(fig, "Fig10A_B12_probability_evolution")


def plot_fig10b_b11_b12_evolution(seq_data: dict[str, pd.DataFrame]) -> None:
    if "B11" not in seq_data or "B12" not in seq_data:
        print("Skipped Fig10B: B11 or B12 sequence data not available.")
        return
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.8), sharex=True)
    plot_probability_evolution(axes[0], seq_data["B11"], "(a) B11 Multi-task TCN-GRU")
    plot_probability_evolution(axes[1], seq_data["B12"], "(b) B12 FGDS-PSI")
    axes[1].set_xlabel("Run index")
    fig.tight_layout()
    save_figure(fig, "Fig10B_B11_vs_B12_probability_evolution")


def transition_windows(seq: pd.DataFrame, radius: int = 18) -> list[tuple[int, int]]:
    ids = seq["true_stage"].map(STAGE_TO_ID).values
    changes = np.where(np.diff(ids) != 0)[0] + 1
    windows = []
    n = len(seq)
    for c in changes[:4]:
        start = max(0, c - radius)
        end = min(n - 1, c + radius)
        if end - start >= 10:
            windows.append((start, end))
    if not windows:
        anchors = [int(n * 0.25), int(n * 0.50), int(n * 0.75)]
        for c in anchors:
            windows.append((max(0, c - radius), min(n - 1, c + radius)))
    return windows[:4]


def plot_fig10c_local_panels(seq_data: dict[str, pd.DataFrame]) -> None:
    if "B12" not in seq_data:
        print("Skipped Fig10C: B12 sequence data not available.")
        return
    seq = seq_data["B12"].reset_index(drop=True)
    windows = transition_windows(seq)
    ncols = 2
    nrows = int(np.ceil(len(windows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.8, 3.5 * nrows), squeeze=False)
    for ax, (start, end), label in zip(axes.ravel(), windows, ["(a)", "(b)", "(c)", "(d)"]):
        sub = seq.iloc[start:end + 1].copy()
        x = sub["run_id"].values
        add_true_stage_background(ax, x, sub["true_stage"])
        ax.plot(x, sub["p_early"], color=COLOR_EARLY, linewidth=1.8, label="early")
        ax.plot(x, sub["p_middle"], color=COLOR_MIDDLE, linewidth=1.8, label="middle")
        ax.plot(x, sub["p_late"], color=COLOR_LATE, linewidth=1.8, label="late")
        ax.scatter(x, sub["pred_stage"].map(STAGE_TO_ID) / 2.0, s=12, color=COLOR_BLACK, alpha=0.45, label="pred/2")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{label} Runs {x.min()}-{x.max()}", fontweight="bold", loc="left")
        ax.set_xlabel("Run index")
        ax.set_ylabel("Probability")
        polish_axis(ax)
    for ax in axes.ravel()[len(windows):]:
        ax.axis("off")
    fig.suptitle("Fig. 10C Local transition probability panels of B12", y=1.01, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "Fig10C_local_transition_panels")


def plot_fig11_timeline(seq_data: dict[str, pd.DataFrame]) -> None:
    if "B11" not in seq_data or "B12" not in seq_data:
        print("Skipped Fig11: B11 or B12 sequence data not available.")
        return
    b12 = seq_data["B12"].copy()
    b11 = seq_data["B11"].copy()
    runs = b12["run_id"].values
    ribbon = np.vstack([
        b12["true_stage"].map(STAGE_TO_ID).values,
        b11["pred_stage"].map(STAGE_TO_ID).values,
        b12["pred_stage"].map(STAGE_TO_ID).values,
    ]).astype(float)
    cmap = ListedColormap([COLOR_EARLY, COLOR_MIDDLE, COLOR_LATE])
    fig, ax = plt.subplots(figsize=(10.8, 2.8))
    ax.imshow(ribbon, aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5,
              extent=[runs.min() - 0.5, runs.max() + 0.5, -0.5, 2.5], interpolation="nearest")
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["True stage", "B11 prediction", "B12 prediction"])
    ax.set_xlabel("Run index")
    ax.set_title("Fig. 11 Stage timeline ribbon on C6", fontweight="bold")
    ax.set_xlim(runs.min() - 0.5, runs.max() + 0.5)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color(COLOR_BLACK)
        spine.set_linewidth(0.8)
    handles = [Patch(facecolor=STAGE_COLORS[s], label=s) for s in STAGES]
    ax.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.32), fontsize=9)
    save_figure(fig, "Fig11_stage_timeline_ribbon")


# =========================================================
# 7. Main
# =========================================================
def main() -> None:
    print("=" * 90)
    print("Section 5.1 visualization suite")
    print(f"Experiment root: {EXPERIMENT_ROOT}")
    print(f"Output dir     : {OUT_DIR}")
    print("=" * 90)

    summary_df = load_summary_data()
    confusion_df = load_confusion_data()
    seq_data = find_sequence_files()

    plot_fig7a_profile(summary_df)
    plot_fig7b_heatmap(summary_df)
    plot_fig7c_barpanel(summary_df)
    plot_fig8a_middle_profile(summary_df)
    plot_fig8b_misdirection(summary_df)
    plot_fig8c_confusion_panels(confusion_df)
    plot_fig9a_tradeoff(summary_df)
    plot_fig9b_consistency(summary_df)
    plot_fig9c_pareto(summary_df)
    plot_fig10a_b12_evolution(seq_data)
    plot_fig10b_b11_b12_evolution(seq_data)
    plot_fig10c_local_panels(seq_data)
    plot_fig11_timeline(seq_data)

    if not seq_data:
        print("Advanced sequence figures were skipped. See missing_sequence_files_note.txt.")
    print("=" * 90)
    print("All available figures have been generated.")
    print("=" * 90)


if __name__ == "__main__":
    main()

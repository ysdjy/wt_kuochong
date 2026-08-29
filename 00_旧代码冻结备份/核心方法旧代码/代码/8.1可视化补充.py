# -*- coding: utf-8 -*-
r"""
Chapter 5 high-quality visualization upgrade for the FGDS-PSI paper.

This script does not retrain any model. It reuses existing CSV outputs under
the "小论文" project folder and generates publication-style composite figures.

Output root:
    小论文/10_第五章顶刊风格可视化
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, ConnectionPatch

warnings.filterwarnings("ignore")


# =========================================================
# 0. Paths and global style
# =========================================================
ROOT = Path(os.environ.get(
    "PAPER_ROOT",
    r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文",
))
OUT_ROOT = Path(os.environ.get(
    "CH5_VIS_OUT_ROOT",
    str(ROOT / "10_第五章顶刊风格可视化"),
))

DIRS = {
    "main": OUT_ROOT / "figures_main",
    "ablation": OUT_ROOT / "figures_ablation",
    "cross": OUT_ROOT / "figures_cross_condition",
    "prob": OUT_ROOT / "figures_probability",
    "q": OUT_ROOT / "figures_q_representation",
    "appendix": OUT_ROOT / "figures_appendix",
    "data": OUT_ROOT / "data_exports",
    "scripts": OUT_ROOT / "scripts",
    "logs": OUT_ROOT / "logs",
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

SAVE_FORMATS = ("png", "pdf", "svg")
DPI = 900

COLOR_E = "#5B8C5A"
COLOR_M = "#D98C2B"
COLOR_L = "#C23B3B"
COLOR_B12 = "#B22222"
COLOR_B11 = "#111111"
COLOR_BLUE = "#3C5CCF"
COLOR_CYAN = "#4AA3A1"
COLOR_ORANGE = "#E39B2E"
COLOR_PURPLE = "#7E57C2"
COLOR_GRAY = "#8A8A8A"
COLOR_GRID = "#DADADA"
COLOR_BLACK = "#222222"

STAGE_COLORS = {"early": COLOR_E, "middle": COLOR_M, "late": COLOR_L}
STAGE_BG = {"early": "#DDEEDC", "middle": "#F8E8CF", "late": "#F3D7D7"}
STAGE_TO_ID = {"early": 0, "middle": 1, "late": 2, "E": 0, "M": 1, "L": 2}
ID_TO_STAGE = {0: "early", 1: "middle", 2: "late"}

METHOD_COLORS = {
    "B1": "#9E9E9E", "B2": "#777777", "B3": "#6F8FAF", "B4": "#8C6BB1",
    "B5": "#4FA36C", "B6": "#E39B2E", "B7": "#B279A2", "B8": "#3C5CCF",
    "B9": "#4AA3A1", "B10": "#7E57C2", "B11": COLOR_B11, "B12": COLOR_B12,
}
ABL_COLORS = {
    "A1": "#555555", "A2": "#1F77B4", "A3": "#17BECF",
    "A4": "#2CA02C", "A5": "#FF9F1C", "A6": COLOR_B12,
}

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


MANIFEST: list[dict] = []
SKIPPED: list[str] = []


@dataclass
class DataBundle:
    comparison: pd.DataFrame | None = None
    comparison_pred: pd.DataFrame | None = None
    comparison_cm: pd.DataFrame | None = None
    ablation: pd.DataFrame | None = None
    ablation_prob: pd.DataFrame | None = None
    cross_dual: pd.DataFrame | None = None
    cross_single: pd.DataFrame | None = None
    cross_avg: pd.DataFrame | None = None
    cross_b12_prob: pd.DataFrame | None = None
    q_a6: pd.DataFrame | None = None
    q_all: pd.DataFrame | None = None


# =========================================================
# 1. Common helpers
# =========================================================
def read_csv(path: Path, required: bool = False) -> pd.DataFrame | None:
    if not path.exists():
        msg = f"Missing: {path}"
        if required:
            raise FileNotFoundError(msg)
        SKIPPED.append(msg)
        return None
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_arrows(df: pd.DataFrame) -> pd.DataFrame:
    repl = {}
    for c in df.columns:
        nc = str(c).replace("M->E", "M→E").replace("M->L", "M→L")
        nc = nc.replace("M鈫扙", "M→E").replace("M鈫扡", "M→L")
        repl[c] = nc
    return df.rename(columns=repl)


def numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def save_data(df: pd.DataFrame, name: str) -> None:
    path = DIRS["data"] / name
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_fig(fig: plt.Figure, outdir: Path, stem: str, section: str, topic: str,
             source: str, recommendation: str, conclusion: str, dpi: int = DPI) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in SAVE_FORMATS:
        path = outdir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi if fmt == "png" else None, bbox_inches="tight", pad_inches=0.04)
        saved.append(str(path))
    plt.close(fig)
    MANIFEST.append({
        "figure": stem,
        "topic": topic,
        "section": section,
        "source": source,
        "recommended_position": recommendation,
        "core_conclusion": conclusion,
        "png": str(outdir / f"{stem}.png"),
        "pdf": str(outdir / f"{stem}.pdf"),
        "svg": str(outdir / f"{stem}.svg"),
    })
    print(f"Saved: {outdir / (stem + '.png')}")


def style_axis(ax, grid_axis="both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_BLACK)
    ax.spines["bottom"].set_color(COLOR_BLACK)
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.55, color=COLOR_GRID, alpha=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=9.5, colors=COLOR_BLACK)


def panel_label(ax, text: str, x=0.02, y=0.96) -> None:
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color=COLOR_BLACK)


def add_axis_arrows(ax, xpad=0.02, ypad=0.035) -> None:
    ax.annotate("", xy=(1 + xpad, 0), xytext=(0, 0),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=COLOR_BLACK),
                clip_on=False)
    ax.annotate("", xy=(0, 1 + ypad), xytext=(0, 0),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=COLOR_BLACK),
                clip_on=False)


def stage_segments(run: np.ndarray, stages: pd.Series | np.ndarray) -> list[tuple[float, float, str]]:
    st = [str(s).lower() for s in stages]
    if len(st) == 0:
        return []
    out = []
    start = 0
    cur = st[0]
    for i in range(1, len(st)):
        if st[i] != cur:
            out.append((float(run[start]), float(run[i - 1]), cur))
            start = i
            cur = st[i]
    out.append((float(run[start]), float(run[-1]), cur))
    return out


def add_stage_background(ax, run: np.ndarray, stages: pd.Series | np.ndarray, alpha=0.32) -> None:
    for x0, x1, st in stage_segments(run, stages):
        ax.axvspan(x0 - 0.5, x1 + 0.5, color=STAGE_BG.get(st, "#EEEEEE"), alpha=alpha, lw=0, zorder=0)


def pred_to_id(s: pd.Series) -> np.ndarray:
    return s.astype(str).str.lower().map({"early": 0, "middle": 1, "late": 2, "e": 0, "m": 1, "l": 2}).fillna(0).astype(int).values


def load_all_data() -> DataBundle:
    b = DataBundle()
    comp_root = ROOT / "4_comparison_experiment_recheck" / "1_results"
    abl_root = ROOT / "6_ablation_experiment"
    cross_root = ROOT / "7_cross_condition_generalization"
    q_root = ROOT / "9_probability_wear_consistency_analysis"

    b.comparison = read_csv(comp_root / "FINAL_comparison_results.csv")
    if b.comparison is not None:
        b.comparison = normalize_arrows(b.comparison)
        b.comparison = numeric(b.comparison, ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Pre", "M-Rec", "M→E", "M→L", "Rev", "Jump", "Smooth"])

    b.comparison_pred = read_csv(comp_root / "FINAL_comparison_predictions.csv")
    b.comparison_cm = read_csv(comp_root / "FINAL_comparison_confusion_matrices_long.csv")
    if b.comparison_cm is not None:
        b.comparison_cm = b.comparison_cm.rename(columns={"true_stage": "True_stage", "pred_stage": "Pred_stage", "count": "count", "row_norm": "row_norm"})
        b.comparison_cm = numeric(b.comparison_cm, ["count", "row_norm"])

    b.ablation = read_csv(abl_root / "Table10_ablation_summary.csv")
    if b.ablation is not None:
        b.ablation = normalize_arrows(b.ablation)
        b.ablation = numeric(b.ablation, ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Pre", "M-Rec", "M→E", "M→L", "Rev", "Jump", "Smooth"])
    b.ablation_prob = read_csv(abl_root / "ablation_probabilities_test_C6.csv")

    b.cross_dual = read_csv(cross_root / "Table12_dual_source_cross_condition_results.csv")
    b.cross_single = read_csv(cross_root / "Table13_single_source_cross_condition_results.csv")
    b.cross_avg = read_csv(cross_root / "Table14_average_cross_condition_performance.csv")
    for attr in ["cross_dual", "cross_single", "cross_avg"]:
        df = getattr(b, attr)
        if df is not None:
            setattr(b, attr, normalize_arrows(df))
    b.cross_b12_prob = read_csv(cross_root / "cross_condition_B12_probabilities.csv")

    b.q_a6 = read_csv(q_root / "Data_5_4_A6_probability_wear_trajectory.csv")
    b.q_all = read_csv(q_root / "Data_5_4_all_ablation_probability_wear_trajectory.csv")

    return b


# =========================================================
# 2. Scheme 1: main comparison overview
# =========================================================
def metric_direction_table(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    out = df[["Method"] + [m for m in metrics if m in df.columns]].copy()
    small_good = {"M→E", "M→L", "Rev", "Jump", "Smooth", "q-MAE", "q-RMSE"}
    for m in metrics:
        if m not in out.columns:
            continue
        vals = pd.to_numeric(out[m], errors="coerce")
        if vals.notna().sum() == 0:
            continue
        lo, hi = vals.min(), vals.max()
        score = (vals - lo) / (hi - lo + 1e-12)
        if m in small_good:
            score = 1 - score
        out[m + "_score"] = score
    return out


def plot_main_heatmap(df: pd.DataFrame):
    metrics = ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Pre", "M-Rec", "M→E", "M→L", "Rev", "Jump", "Smooth"]
    score_df = metric_direction_table(df, metrics)
    score_cols = [m + "_score" for m in metrics if m + "_score" in score_df.columns]
    mat = score_df[score_cols].values.astype(float)
    labels = [c.replace("_score", "") for c in score_cols]

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    cmap = LinearSegmentedColormap.from_list("paper_heat", ["#F7F7F7", "#C9D8EF", "#5978B8", "#B22222"])
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(score_df)))
    ax.set_yticklabels(score_df["Method"].astype(str))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=7.6, color="white" if mat[i, j] > 0.70 else COLOR_BLACK)
    if "B12" in score_df["Method"].astype(str).values:
        idx = score_df.index[score_df["Method"].astype(str) == "B12"][0]
        ax.add_patch(Rectangle((-0.5, idx - 0.5), len(labels), 1, fill=False, ec=COLOR_B12, lw=2.2))
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Direction-unified score")
    save_data(score_df, "Fig5_1_main_overview_heatmap_data.csv")
    save_fig(fig, DIRS["main"], "Fig5_1_main_overview_heatmap",
             "5.1 主实验对比", "Direction-unified metric heatmap for B1-B12",
             "FINAL_comparison_results.csv", "第五章 5.1 主实验总览",
             "B12 在多数方向统一指标上保持高水平，兼顾性能与稳定性。")


def plot_main_profile(df: pd.DataFrame):
    metrics = ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Pre", "M-Rec", "1-M→E", "1-M→L", "Smooth*"]
    tmp = df.copy()
    tmp["1-M→E"] = 1 - tmp["M→E"]
    tmp["1-M→L"] = 1 - tmp["M→L"]
    s = tmp["Smooth"].astype(float)
    tmp["Smooth*"] = 1 - (s - s.min()) / (s.max() - s.min() + 1e-12)
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    for _, row in tmp.iterrows():
        method = str(row["Method"])
        color = METHOD_COLORS.get(method, COLOR_GRAY)
        lw = 2.8 if method == "B12" else (2.1 if method == "B11" else 1.0)
        ms = 11 if method == "B12" else (6 if method == "B11" else 3.8)
        alpha = 1.0 if method in ["B11", "B12"] else 0.52
        marker = "*" if method == "B12" else ("D" if method == "B11" else "o")
        ax.plot(x, row[metrics].astype(float).values, color=color, marker=marker,
                lw=lw, ms=ms, alpha=alpha, label=method, zorder=5 if method in ["B11", "B12"] else 2)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Score")
    ax.legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.30), fontsize=8)
    style_axis(ax)
    add_axis_arrows(ax)
    save_data(tmp[["Method"] + metrics], "Fig5_2_main_profile_plot_data.csv")
    save_fig(fig, DIRS["main"], "Fig5_2_main_profile_plot",
             "5.1 主实验对比", "Multi-metric profile plot",
             "FINAL_comparison_results.csv", "第五章 5.1 主实验总览或附图",
             "B12 的多指标轮廓稳定接近最优，middle 专项与误判抑制均衡。")


def plot_main_representative_bars(df: pd.DataFrame):
    methods = ["B1", "B5", "B8", "B11", "B12"]
    metrics = ["Acc", "Macro-F1", "M-F1", "M-Rec", "1-M→L"]
    tmp = df[df["Method"].isin(methods)].copy()
    tmp["1-M→L"] = 1 - tmp["M→L"]
    x = np.arange(len(methods))
    w = 0.15
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    for j, m in enumerate(metrics):
        vals = tmp.set_index("Method").loc[methods, m].values
        ax.bar(x + (j - 2) * w, vals, width=w, facecolor="white",
               edgecolor=[COLOR_BLUE, COLOR_CYAN, COLOR_M, COLOR_ORANGE, COLOR_B12][j],
               hatch=["////", "\\\\\\\\", "....", "xxxx", "----"][j],
               linewidth=1.1, label=m, zorder=3)
    ax.axvspan(3.5, 4.5, color=COLOR_B12, alpha=0.07, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=8.2)
    style_axis(ax)
    add_axis_arrows(ax)
    save_data(tmp[["Method"] + metrics], "Fig5_3_main_representative_bar_score_data.csv")
    save_fig(fig, DIRS["main"], "Fig5_3_main_representative_bar_score",
             "5.1 主实验对比", "Representative grouped bars",
             "FINAL_comparison_results.csv", "第五章 5.1 代表方法对比",
             "B12 相比规则、传统模型和深度基线在关键指标上更均衡。")


# =========================================================
# 3. Scheme 2: middle stage analysis
# =========================================================
def plot_middle_panel(df: pd.DataFrame):
    methods = df["Method"].astype(str).tolist()
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax = axes[0]
    w = 0.25
    for off, metric, color, hatch in [(-w, "M-Pre", COLOR_BLUE, "////"), (0, "M-Rec", COLOR_M, "\\\\\\\\"), (w, "M-F1", COLOR_B12, "....")]:
        ax.bar(x + off, df[metric], width=w, facecolor="white", edgecolor=color, hatch=hatch,
               linewidth=1.0, label=metric, zorder=3)
    ax.axvspan(len(methods) - 1.5, len(methods) - 0.5, color=COLOR_B12, alpha=0.07, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=0)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.legend(fontsize=8, loc="lower right"); panel_label(ax, "(a)")
    style_axis(ax)

    ax = axes[1]
    w = 0.34
    ax.bar(x - w/2, df["M→E"], width=w, color="#7FA6D8", edgecolor=COLOR_BLACK, lw=0.4, label="M→E")
    ax.bar(x + w/2, df["M→L"], width=w, color=COLOR_L, edgecolor=COLOR_BLACK, lw=0.4, label="M→L")
    ax.axvspan(len(methods) - 1.5, len(methods) - 0.5, color=COLOR_B12, alpha=0.07, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylim(0, max(0.55, df[["M→E", "M→L"]].max().max() * 1.1)); ax.set_ylabel("Misclassification rate")
    ax.legend(fontsize=8); panel_label(ax, "(b)")
    style_axis(ax)

    ax = axes[2]
    ratio = df["M-Rec"].values
    ax.plot(x, ratio, color=COLOR_M, marker="o", lw=1.8, label="M-Rec")
    ax.plot(x, df["M-F1"], color=COLOR_B12, marker="*", ms=10, lw=2.2, label="M-F1")
    ax.fill_between(x, ratio, df["M-F1"], color=COLOR_M, alpha=0.10)
    ax.axvline(len(methods)-1, color=COLOR_B12, lw=1.2, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Middle-stage score")
    ax.legend(fontsize=8, loc="lower right"); panel_label(ax, "(c)")
    style_axis(ax)
    fig.tight_layout()
    save_data(df[["Method", "M-Pre", "M-Rec", "M-F1", "M→E", "M→L"]], "Fig5_4_middle_stage_panel_data.csv")
    save_fig(fig, DIRS["main"], "Fig5_4_middle_stage_panel",
             "5.1/5.4 Middle 专项", "Middle-stage precision/recall/error panel",
             "FINAL_comparison_results.csv", "第五章 middle 阶段专项分析",
             "B12 对 middle 阶段保持高召回，并将 M→L 控制在低水平。")


def plot_middle_error_flow(df: pd.DataFrame):
    show = ["B5", "B8", "B11", "B12"]
    sub = df[df["Method"].isin(show)].set_index("Method").loc[show]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    y = np.arange(len(show))
    ax.hlines(y, 0, sub["M→E"], color="#7FA6D8", lw=7, alpha=0.75, label="M→E")
    ax.hlines(y + 0.18, 0, sub["M→L"], color=COLOR_L, lw=7, alpha=0.80, label="M→L")
    ax.scatter(sub["M→E"], y, color="#2F5D9B", s=55, zorder=3)
    ax.scatter(sub["M→L"], y + 0.18, color=COLOR_L, s=55, zorder=3)
    ax.set_yticks(y + 0.09)
    ax.set_yticklabels(show)
    ax.set_xlabel("Middle-stage misclassification rate")
    ax.set_xlim(0, max(0.12, sub[["M→E", "M→L"]].max().max() * 1.25))
    ax.legend(ncol=2, loc="upper right", fontsize=9)
    style_axis(ax, "x")
    add_axis_arrows(ax)
    save_data(sub.reset_index(), "Fig5_5_middle_error_flow_panel_data.csv")
    save_fig(fig, DIRS["main"], "Fig5_5_middle_error_flow_panel",
             "5.1 Middle 专项", "Middle-stage directional error lollipop",
             "FINAL_comparison_results.csv", "第五章 middle 误判方向分析",
             "B12 避免 middle 过早进入 late，同时保持较低 M→E。")


# =========================================================
# 4. Scheme 3: confusion matrix panels
# =========================================================
def confusion_matrix_for(cm: pd.DataFrame, method: str) -> tuple[np.ndarray, np.ndarray]:
    sub = cm[cm["Method"].astype(str) == method]
    mat = np.zeros((3, 3))
    cnt = np.zeros((3, 3), dtype=int)
    for _, r in sub.iterrows():
        ti = str(r["True_stage"]).lower()
        pj = str(r["Pred_stage"]).lower()
        i = {"early": 0, "middle": 1, "late": 2, "e": 0, "m": 1, "l": 2}[ti]
        j = {"early": 0, "middle": 1, "late": 2, "e": 0, "m": 1, "l": 2}[pj]
        mat[i, j] = float(r["row_norm"])
        cnt[i, j] = int(r["count"])
    return mat, cnt


def plot_confusion_panel(cm: pd.DataFrame, comp: pd.DataFrame, horizontal=False):
    methods = ["B1", "B5", "B11", "B12"]
    if horizontal:
        fig, axes = plt.subplots(1, 4, figsize=(14.8, 3.7))
        stem = "Fig5_S1_confusion_matrix_panel_1x4"
    else:
        fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.6))
        stem = "Fig5_6_confusion_matrix_panel_2x2"
    axes = np.ravel(axes)
    cmap = LinearSegmentedColormap.from_list("cm_soft", ["#FFFFFF", "#CFE1F2", "#5487C7", "#8B1E1E"])
    im = None
    for k, (ax, m) in enumerate(zip(axes, methods)):
        mat, cnt = confusion_matrix_for(cm, m)
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1)
        r = comp[comp["Method"].astype(str) == m].iloc[0]
        ax.set_title(f"({chr(97+k)}) {m}   Acc={r['Acc']:.3f}, M-F1={r['M-F1']:.3f}", loc="left", fontsize=10, fontweight="bold")
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["E", "M", "L"])
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["E", "M", "L"])
        ax.set_xlabel("Predicted stage"); ax.set_ylabel("True stage")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cnt[i,j]}\n{mat[i,j]:.2f}",
                        ha="center", va="center", fontsize=10.5,
                        color="white" if mat[i, j] > 0.62 else COLOR_BLACK)
        for s in ax.spines.values():
            s.set_linewidth(0.8); s.set_color(COLOR_BLACK)
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.022 if horizontal else 0.035, pad=0.02)
    cb.set_label("Row-normalized value")
    save_data(cm[cm["Method"].isin(methods)], stem + "_data.csv")
    save_fig(fig, DIRS["main"], stem,
             "5.1 混淆矩阵", "Confusion matrix panel",
             "FINAL_comparison_confusion_matrices_long.csv", "第五章 5.1 混淆矩阵分析",
             "B12 的错分主要集中在相邻阶段，middle→late 得到抑制。")


# =========================================================
# 5. Scheme 4: probability evolution
# =========================================================
def method_sequence(pred: pd.DataFrame, method: str) -> pd.DataFrame | None:
    cols = [f"prob_E_{method}", f"prob_M_{method}", f"prob_L_{method}", f"pred_{method}"]
    if any(c not in pred.columns for c in cols):
        return None
    return pd.DataFrame({
        "run_id": pd.to_numeric(pred["run_id_end"], errors="coerce"),
        "true_stage": pred["true_stage"].astype(str).str.lower(),
        "pred_stage": pred[f"pred_{method}"].astype(str).str.lower(),
        "p_E": pd.to_numeric(pred[f"prob_E_{method}"], errors="coerce"),
        "p_M": pd.to_numeric(pred[f"prob_M_{method}"], errors="coerce"),
        "p_L": pd.to_numeric(pred[f"prob_L_{method}"], errors="coerce"),
    }).dropna().sort_values("run_id")


def plot_prob_lines(ax, seq: pd.DataFrame, title: str, show_legend=False):
    x = seq["run_id"].values
    add_stage_background(ax, x, seq["true_stage"], alpha=0.28)
    ax.plot(x, seq["p_E"], color=COLOR_E, lw=1.8, label=r"$p_E$")
    ax.plot(x, seq["p_M"], color=COLOR_M, lw=1.8, label=r"$p_M$")
    ax.plot(x, seq["p_L"], color=COLOR_L, lw=1.8, label=r"$p_L$")
    ax.set_ylim(-0.03, 1.04)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    style_axis(ax)
    if show_legend:
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=8.5)


def plot_probability_b11_vs_b12(pred: pd.DataFrame):
    b11 = method_sequence(pred, "B11")
    b12 = method_sequence(pred, "B12")
    if b11 is None or b12 is None:
        SKIPPED.append("B11/B12 sequence columns missing for probability evolution.")
        return
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 6.4), sharex=True, sharey=True,
                             gridspec_kw={"height_ratios": [1, 0.34]})
    plot_prob_lines(axes[0, 0], b11, "(a) B11 probability evolution", show_legend=True)
    plot_prob_lines(axes[0, 1], b12, "(b) B12 probability evolution", show_legend=True)
    for ax, seq, title in [(axes[1,0], b11, "(c) B11 stage ribbon"), (axes[1,1], b12, "(d) B12 stage ribbon")]:
        run = seq["run_id"].values
        ribbon = np.vstack([pred_to_id(seq["true_stage"]), pred_to_id(seq["pred_stage"])])
        cmap = ListedColormap([COLOR_E, COLOR_M, COLOR_L])
        ax.imshow(ribbon, aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5,
                  extent=[run.min(), run.max(), -0.5, 1.5], interpolation="nearest")
        ax.set_yticks([1, 0]); ax.set_yticklabels(["True", "Pred"])
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Run index")
        ax.grid(False)
    axes[0,0].set_ylabel("Probability")
    axes[1,0].set_ylabel("Stage")
    save_data(b11.assign(method="B11"), "Fig5_7_probability_evolution_B11_data.csv")
    save_data(b12.assign(method="B12"), "Fig5_7_probability_evolution_B12_data.csv")
    save_fig(fig, DIRS["prob"], "Fig5_7_probability_evolution_B11_vs_B12",
             "5.4 阶段概率演化", "B11 vs B12 probability evolution and ribbon",
             "FINAL_comparison_predictions.csv", "第五章 5.4 阶段概率演化",
             "B12 概率随生命周期有序演化，阶段条带更连续。")


def plot_probability_stream_and_zoom(seq: pd.DataFrame):
    x = seq["run_id"].values
    fig, axes = plt.subplots(2, 1, figsize=(11.6, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.0, 0.82]})
    ax = axes[0]
    ax.stackplot(x, seq["p_E"], seq["p_M"], seq["p_L"], colors=[COLOR_E, COLOR_M, COLOR_L],
                 alpha=0.78, labels=[r"$p_E$", r"$p_M$", r"$p_L$"])
    ax.set_ylabel("Probability composition")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=8.5)
    panel_label(ax, "(a)")
    style_axis(ax)
    ax = axes[1]
    plot_prob_lines(ax, seq, "(b) Local transition zoom", show_legend=False)
    # highlight transition area around true-stage changes
    ids = pred_to_id(seq["true_stage"])
    changes = np.where(np.diff(ids) != 0)[0] + 1
    if len(changes) > 0:
        c = changes[min(1, len(changes)-1)]
        lo = max(0, c - 22); hi = min(len(seq)-1, c + 22)
        ax.set_xlim(x[lo], x[hi])
        con = ConnectionPatch(xyA=(x[lo], 0.02), coordsA=axes[0].transData,
                              xyB=(x[lo], 1.0), coordsB=ax.transData,
                              color=COLOR_GRAY, lw=0.8, ls="--")
        fig.add_artist(con)
    ax.set_xlabel("Run index"); ax.set_ylabel("Probability")
    fig.tight_layout()
    save_data(seq, "Fig5_8_probability_stream_ribbon_data.csv")
    save_fig(fig, DIRS["prob"], "Fig5_8_probability_stream_ribbon",
             "5.4 阶段概率演化", "B12 streamgraph and transition zoom",
             "FINAL_comparison_predictions.csv", "第五章 5.4 或正文核心图",
             "B12 阶段概率组成呈 early-middle-late 的生命周期流动。")


def plot_ablation_probability_2x3(abprob: pd.DataFrame):
    mapping = [
        ("A1 Raw", ["p_raw_E", "p_raw_M", "p_raw_L"], "pred_A1"),
        ("A2 Raw + Fine", ["p_raw_fine_E", "p_raw_fine_M", "p_raw_fine_L"], "pred_A2"),
        ("A3 Raw + Prior", ["p_raw_prior_E", "p_raw_prior_M", "p_raw_prior_L"], "pred_A3"),
        ("A4 Mix", ["p_mix_E", "p_mix_M", "p_mix_L"], "pred_A4"),
        ("A5 Ordered", ["alpha_E", "alpha_M", "alpha_L"], "pred_A5"),
        ("A6 Final", ["p_final_E", "p_final_M", "p_final_L"], "pred_A6"),
    ]
    if any(any(c not in abprob.columns for c in cols) for _, cols, _ in mapping):
        SKIPPED.append("Ablation probability columns missing.")
        return
    run = pd.to_numeric(abprob["run_id"], errors="coerce").values
    true_stage = abprob["true_stage"].astype(str).str.lower()
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 9.0), sharex=True, sharey=True)
    for ax, (title, cols, _) in zip(axes.ravel(), mapping):
        add_stage_background(ax, run, true_stage, alpha=0.25)
        ax.plot(run, abprob[cols[0]], color=COLOR_E, lw=1.55, label=r"$p_E$")
        ax.plot(run, abprob[cols[1]], color=COLOR_M, lw=1.55, label=r"$p_M$")
        ax.plot(run, abprob[cols[2]], color=COLOR_L, lw=1.55, label=r"$p_L$")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10.5)
        ax.set_ylim(-0.03, 1.04)
        style_axis(ax)
    axes[0, 0].legend(ncol=3, loc="upper center", bbox_to_anchor=(1.05, 1.22), fontsize=8.5)
    for ax in axes[:,0]:
        ax.set_ylabel("Probability")
    for ax in axes[-1,:]:
        ax.set_xlabel("Run index")
    fig.tight_layout()
    save_data(abprob, "Fig5_10_ablation_probability_evolution_2x3_data.csv")
    save_fig(fig, DIRS["ablation"], "Fig5_10_ablation_probability_evolution_2x3",
             "5.3 消融概率机制", "A1-A6 probability evolution small multiples",
             "ablation_probabilities_test_C6.csv", "第五章 5.3 消融机制图",
             "从 Raw 到 Final，概率输出逐步变得更有序、更平滑。")


def add_inset_probability(ax, run, probs, center_idx: int, width: int = 34):
    """Add a compact transition-zone inset to a probability axis."""
    lo = max(0, center_idx - width // 2)
    hi = min(len(run) - 1, center_idx + width // 2)
    ins = ax.inset_axes([0.58, 0.52, 0.34, 0.34])
    ins.plot(run[lo:hi + 1], probs[0][lo:hi + 1], color=COLOR_E, lw=1.0)
    ins.plot(run[lo:hi + 1], probs[1][lo:hi + 1], color=COLOR_M, lw=1.0)
    ins.plot(run[lo:hi + 1], probs[2][lo:hi + 1], color=COLOR_L, lw=1.0)
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_ylim(-0.03, 1.03)
    for spine in ins.spines.values():
        spine.set_color(COLOR_BLACK)
        spine.set_linewidth(0.75)
    ax.add_patch(Rectangle((run[lo], -0.02), run[hi] - run[lo], 1.04,
                           fill=False, ec=COLOR_B12, lw=0.9, ls="--", zorder=5))


def plot_ablation_probability_2x3_inset(abprob: pd.DataFrame):
    mapping = [
        ("A1 Raw", ["p_raw_E", "p_raw_M", "p_raw_L"]),
        ("A2 Raw + Fine", ["p_raw_fine_E", "p_raw_fine_M", "p_raw_fine_L"]),
        ("A3 Raw + Prior", ["p_raw_prior_E", "p_raw_prior_M", "p_raw_prior_L"]),
        ("A4 Mix", ["p_mix_E", "p_mix_M", "p_mix_L"]),
        ("A5 Ordered", ["alpha_E", "alpha_M", "alpha_L"]),
        ("A6 Final", ["p_final_E", "p_final_M", "p_final_L"]),
    ]
    if any(any(c not in abprob.columns for c in cols) for _, cols, _ in mapping):
        SKIPPED.append("Ablation inset figure skipped: probability columns missing.")
        return
    run = pd.to_numeric(abprob["run_id"], errors="coerce").values
    true_stage = abprob["true_stage"].astype(str).str.lower()
    ids = pred_to_id(true_stage)
    changes = np.where(np.diff(ids) != 0)[0] + 1
    center_idx = int(changes[-1]) if len(changes) else int(len(run) * 0.72)

    fig, axes = plt.subplots(2, 3, figsize=(14.8, 6.7), sharex=True, sharey=True)
    for ax, (title, cols), label in zip(axes.ravel(), mapping, list("abcdef")):
        add_stage_background(ax, run, true_stage, alpha=0.23)
        probs = [pd.to_numeric(abprob[c], errors="coerce").values for c in cols]
        ax.plot(run, probs[0], color=COLOR_E, lw=1.45, label=r"$p_E$")
        ax.plot(run, probs[1], color=COLOR_M, lw=1.45, label=r"$p_M$")
        ax.plot(run, probs[2], color=COLOR_L, lw=1.45, label=r"$p_L$")
        add_inset_probability(ax, run, probs, center_idx=center_idx, width=42)
        ax.set_title(f"({label}) {title}", loc="left", fontweight="bold", fontsize=10.5)
        ax.set_ylim(-0.03, 1.04)
        style_axis(ax)
    axes[0, 1].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.27), fontsize=8.6)
    for ax in axes[:, 0]:
        ax.set_ylabel("Probability")
    for ax in axes[-1, :]:
        ax.set_xlabel("Run index")
    fig.tight_layout()
    save_data(abprob, "Fig5_19_ablation_probability_inset_2x3_data.csv")
    save_fig(fig, DIRS["ablation"], "Fig5_19_ablation_probability_inset_2x3",
             "5.3 消融概率机制", "A1-A6 probability evolution with transition insets",
             "ablation_probabilities_test_C6.csv", "第五章 5.3 消融机制重点图",
             "局部放大显示 ordered/final 模块在过渡区抑制概率抖动。")


# =========================================================
# 6. Scheme 5 and 6: q representation + ablation overview
# =========================================================
def plot_q_consistency(qdf: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    ax = axes[0]
    colors = qdf["true_stage"].astype(str).str.lower().map(STAGE_COLORS).fillna(COLOR_GRAY)
    ax.scatter(qdf["q_true"], qdf["q_pred_norm"], c=colors, s=20, alpha=0.78, edgecolor="white", lw=0.2)
    ax.plot([0, 1], [0, 1], color=COLOR_BLACK, ls="--", lw=1.0)
    ax.set_xlabel(r"$q_{true}$"); ax.set_ylabel(r"$q_{pred}$")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    panel_label(ax, "(a)")
    style_axis(ax)

    ax = axes[1]
    run = qdf["run_id"].values
    add_stage_background(ax, run, qdf["true_stage"], alpha=0.28)
    ax.plot(run, qdf["q_true"], color=COLOR_BLACK, lw=1.8, label=r"$q_{true}$")
    ax.plot(run, qdf["q_pred_norm"], color=COLOR_B12, lw=1.8, ls="--", label=r"$q_{pred}$")
    ax.fill_between(run, qdf["q_true"], qdf["q_pred_norm"], color=COLOR_B12, alpha=0.08)
    ax.set_xlabel("Run index"); ax.set_ylabel("Degradation position")
    ax.legend(fontsize=8); panel_label(ax, "(b)")
    style_axis(ax)

    ax = axes[2]
    qdf2 = qdf.copy()
    qdf2["abs_error"] = np.abs(qdf2["q_pred_norm"] - qdf2["q_true"])
    ax.plot(run, qdf2["abs_error"], color=COLOR_BLUE, lw=1.3)
    ax.fill_between(run, 0, qdf2["abs_error"], color=COLOR_BLUE, alpha=0.14)
    ax.set_xlabel("Run index"); ax.set_ylabel(r"$|q_{pred}-q_{true}|$")
    panel_label(ax, "(c)")
    style_axis(ax)
    fig.tight_layout()
    save_data(qdf, "Fig5_13_q_evolution_consistency_data.csv")
    save_fig(fig, DIRS["q"], "Fig5_13_q_evolution_consistency",
             "5.4 q 表征一致性", "q_true vs q_pred scatter/evolution/error",
             "Data_5_4_A6_probability_wear_trajectory.csv", "第五章 5.4 q 一致性分析",
             "连续退化位置估计与真实退化位置同步演化。")


def plot_q_stage_distribution(qdf: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.7))
    for ax, stage_col, title, label in [(axes[0], "true_stage", "Grouped by true stage", "(a)"),
                                        (axes[1], "pred_stage", "Grouped by predicted stage", "(b)")]:
        data = []
        labels = []
        for st in ["early", "middle", "late"]:
            sub = qdf[qdf[stage_col].astype(str).str.lower() == st]["q_pred_norm"].dropna().values
            if len(sub):
                data.append(sub); labels.append(st[0].upper())
        bp = ax.boxplot(data, patch_artist=True, tick_labels=labels, widths=0.52, showmeans=True,
                        medianprops=dict(color=COLOR_BLACK, lw=1.2),
                        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor=COLOR_BLACK, markersize=4.5))
        for patch, st in zip(bp["boxes"], ["early", "middle", "late"]):
            patch.set_facecolor("white")
            patch.set_edgecolor(STAGE_COLORS[st])
            patch.set_hatch("////")
            patch.set_linewidth(1.25)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
        ax.set_ylabel(r"$q_{pred}$")
        panel_label(ax, label)
        style_axis(ax)
    fig.tight_layout()
    save_data(qdf, "Fig5_14_q_stage_distribution_data.csv")
    save_fig(fig, DIRS["q"], "Fig5_14_q_stage_distribution",
             "5.4 q 表征一致性", "Stage-wise q distribution",
             "Data_5_4_A6_probability_wear_trajectory.csv", "第五章 5.4 阶段语义递增分析",
             "预测阶段 E/M/L 对应退化位置递增，说明阶段概率具有退化语义。")


def plot_q_probability_phase(qdf: pd.DataFrame):
    q = qdf["q_pred_norm"].values
    y = qdf["prob_late"].values - qdf["prob_early"].values
    run = qdf["run_id"].values
    pts = np.array([q, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cmap = LinearSegmentedColormap.from_list("life", [COLOR_E, COLOR_M, COLOR_L])
    lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(run.min(), run.max()), lw=2.0)
    lc.set_array(run[:-1])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.add_collection(lc)
    ax.scatter(q, y, c=run, cmap=cmap, s=18, edgecolor="white", lw=0.2)
    ax.axhline(0, color=COLOR_GRAY, ls="--", lw=0.9)
    ax.axvline(0.5, color=COLOR_GRAY, ls="--", lw=0.9)
    ax.scatter(q[0], y[0], s=90, color=COLOR_E, edgecolor=COLOR_BLACK, label="Start")
    ax.scatter(q[-1], y[-1], s=150, marker="*", color=COLOR_L, edgecolor=COLOR_BLACK, label="End")
    ax.set_xlabel(r"$q_{pred}$")
    ax.set_ylabel(r"$p_L-p_E$")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=8)
    style_axis(ax)
    save_data(qdf, "Fig5_15_q_probability_phase_data.csv")
    save_fig(fig, DIRS["q"], "Fig5_15_q_probability_phase",
             "5.4 概率-q 联合表征", "q-probability phase trajectory",
             "Data_5_4_A6_probability_wear_trajectory.csv", "第五章 5.4 联合状态表征",
             "阶段概率差异与连续退化位置共同形成有序生命周期轨迹。")


def pca_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    arr = arr / scale
    u, s, vt = np.linalg.svd(arr, full_matrices=False)
    return arr @ vt[:2].T


def plot_representation_space_grid(pred: pd.DataFrame, qdf: pd.DataFrame | None = None):
    """Nine-panel PCA visualization using available probability-state proxies.

    If hidden features are not exported, this figure uses probability/degradation
    state vectors as a proxy representation and records this in the manifest.
    """
    required = ["prob_E_B5", "prob_M_B5", "prob_L_B5", "prob_E_B11", "prob_M_B11", "prob_L_B11", "prob_E_B12", "prob_M_B12", "prob_L_B12"]
    if any(c not in pred.columns for c in required):
        SKIPPED.append("Representation grid skipped: B5/B11/B12 probability columns missing.")
        return
    run = pd.to_numeric(pred["run_id_end"], errors="coerce").values
    run_norm = (run - np.nanmin(run)) / (np.nanmax(run) - np.nanmin(run) + 1e-12)
    true_stage = pred["true_stage"].astype(str).str.lower()
    stage_id = pred_to_id(true_stage)
    q_color = qdf["q_true"].values if qdf is not None and "q_true" in qdf.columns and len(qdf) == len(pred) else run_norm
    q_color = np.asarray(q_color, dtype=float)

    reps = [
        ("Baseline probability state", pred[["prob_E_B5", "prob_M_B5", "prob_L_B5"]].values),
        ("Multi-task raw state", pred[["prob_E_B11", "prob_M_B11", "prob_L_B11"]].values),
        ("FGDS-PSI final state", pred[["prob_E_B12", "prob_M_B12", "prob_L_B12"]].values),
    ]
    color_modes = ["Stage", r"$q$ / lifecycle", "Predicted stage"]
    fig, axes = plt.subplots(3, 3, figsize=(9.6, 9.0))
    for i, (rep_name, values) in enumerate(reps):
        features = np.column_stack([values, run_norm])
        emb = pca_2d(features)
        for j, mode in enumerate(color_modes):
            ax = axes[i, j]
            if mode == "Stage":
                colors = [STAGE_COLORS[ID_TO_STAGE[k]] for k in stage_id]
                ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=12, alpha=0.82, edgecolor="none")
            elif mode == "Predicted stage":
                method = ["B5", "B11", "B12"][i]
                pred_col = f"pred_{method}"
                pid = pred_to_id(pred[pred_col]) if pred_col in pred.columns else stage_id
                colors = [STAGE_COLORS[ID_TO_STAGE[k]] for k in pid]
                ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=12, alpha=0.82, edgecolor="none")
            else:
                sc = ax.scatter(emb[:, 0], emb[:, 1], c=q_color, cmap="viridis", s=12, alpha=0.86, edgecolor="none")
                if i == 0:
                    cb = fig.colorbar(sc, ax=axes[:, j], fraction=0.035, pad=0.02)
                    cb.set_label("Degradation position")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            ax.set_title(f"({chr(97+i*3+j)}) {rep_name}\ncolored by {mode}", fontsize=9.2, loc="left")
            style_axis(ax, "both")
    handles = [Patch(facecolor=STAGE_COLORS[s], label=s.capitalize()) for s in ["early", "middle", "late"]]
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.01), fontsize=9)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    export = pd.DataFrame({"run_id": run, "true_stage": true_stage, "stage_id": stage_id, "q_color": q_color})
    save_data(export, "Fig5_20_representation_space_grid_metadata.csv")
    save_fig(fig, DIRS["q"], "Fig5_20_representation_space_grid_proxy",
             "5.4 表征空间扩展", "PCA grid of probability-state proxy representations",
             "FINAL_comparison_predictions.csv", "第五章 5.4 表征空间备选图",
             "在未导出 hidden feature 的情况下，概率状态代理表征沿退化阶段呈连续结构。")


def plot_cross_qhat_small_multiples(prob: pd.DataFrame):
    if prob is None or prob.empty or "q_hat" not in prob.columns:
        return
    tasks = prob["Task"].astype(str).unique().tolist()[:6]
    if not tasks:
        return
    nrows = int(np.ceil(len(tasks) / 3))
    fig, axes = plt.subplots(nrows, 3, figsize=(15.0, 3.6 * nrows), sharey=True)
    axes = np.ravel(axes)
    for ax, task, label in zip(axes, tasks, list("abcdef")):
        sub = prob[prob["Task"].astype(str) == task].copy()
        run = pd.to_numeric(sub["run_id"], errors="coerce").values
        q = pd.to_numeric(sub["q_hat"], errors="coerce")
        qn = (q - q.min()) / (q.max() - q.min() + 1e-12)
        add_stage_background(ax, run, sub["true_stage"].astype(str).str.lower(), alpha=0.22)
        ax.plot(run, qn, color=COLOR_B12, lw=1.8, label=r"$\hat q$")
        ax.fill_between(run, 0, qn, color=COLOR_B12, alpha=0.08)
        # inset for late transition
        ids = pred_to_id(sub["true_stage"])
        changes = np.where(np.diff(ids) != 0)[0] + 1
        if len(changes):
            c = changes[-1]
            lo, hi = max(0, c-18), min(len(run)-1, c+18)
            ins = ax.inset_axes([0.53, 0.16, 0.38, 0.34])
            ins.plot(run[lo:hi+1], qn.iloc[lo:hi+1], color=COLOR_B12, lw=1.0)
            ins.set_xticks([]); ins.set_yticks([])
            for sp in ins.spines.values():
                sp.set_linewidth(0.7)
            ax.add_patch(Rectangle((run[lo], -0.02), run[hi]-run[lo], 1.04,
                                   fill=False, ec=COLOR_B12, ls="--", lw=0.9))
        ax.set_title(f"({label}) {task}", loc="left", fontsize=10.2, fontweight="bold")
        ax.set_ylim(-0.03, 1.04)
        ax.set_xlabel("Run index")
        style_axis(ax)
    for ax in axes[len(tasks):]:
        ax.axis("off")
    axes[0].set_ylabel("Normalized q estimate")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    save_data(prob, "Fig5_21_cross_condition_qhat_small_multiples_data.csv")
    save_fig(fig, DIRS["cross"], "Fig5_21_cross_condition_qhat_small_multiples",
             "5.2/5.4 跨工况 q 演化", "Cross-condition q-hat small multiples with insets",
             "cross_condition_B12_probabilities.csv", "第五章 5.2 跨工况案例或附录",
             "B12 在不同迁移任务下的退化位置估计均呈随生命周期上升的趋势。")


def plot_cross_probability_small_multiples(prob: pd.DataFrame):
    if prob is None or prob.empty:
        return
    tasks = prob["Task"].astype(str).unique().tolist()[:6]
    nrows = int(np.ceil(len(tasks) / 3))
    fig, axes = plt.subplots(nrows, 3, figsize=(15.0, 3.6 * nrows), sharey=True)
    axes = np.ravel(axes)
    for ax, task, label in zip(axes, tasks, list("abcdef")):
        sub = prob[prob["Task"].astype(str) == task].copy()
        run = pd.to_numeric(sub["run_id"], errors="coerce").values
        add_stage_background(ax, run, sub["true_stage"].astype(str).str.lower(), alpha=0.22)
        ax.plot(run, sub["p_E"], color=COLOR_E, lw=1.35)
        ax.plot(run, sub["p_M"], color=COLOR_M, lw=1.35)
        ax.plot(run, sub["p_L"], color=COLOR_L, lw=1.35)
        ax.set_title(f"({label}) {task}", loc="left", fontsize=10.2, fontweight="bold")
        ax.set_ylim(-0.03, 1.04)
        ax.set_xlabel("Run index")
        style_axis(ax)
    for ax in axes[len(tasks):]:
        ax.axis("off")
    axes[0].set_ylabel("Stage probability")
    axes[0].legend([r"$p_E$", r"$p_M$", r"$p_L$"], ncol=3, loc="upper center", bbox_to_anchor=(1.65, 1.22), fontsize=8)
    fig.tight_layout()
    save_data(prob, "Fig5_22_cross_condition_probability_small_multiples_data.csv")
    save_fig(fig, DIRS["cross"], "Fig5_22_cross_condition_probability_small_multiples",
             "5.2 跨工况概率演化", "Cross-condition probability small multiples",
             "cross_condition_B12_probabilities.csv", "第五章 5.2 跨工况阶段概率演化",
             "不同迁移任务中，FGDS-PSI 均形成 early-middle-late 的概率主导迁移。")


def plot_ablation_overview(ab: pd.DataFrame):
    methods = ab["Method"].astype(str).tolist()
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    ax = axes[0]
    metrics = ["Acc", "Macro-F1", "M-F1", "M-Rec"]
    for m, color in zip(metrics, [COLOR_BLUE, COLOR_B12, COLOR_M, COLOR_E]):
        ax.plot(x, ab[m], marker="o", lw=1.7, color=color, label=m)
    ax.axvspan(5-0.4, 5+0.4, color=COLOR_B12, alpha=0.08)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylim(0.94, 1.01); ax.set_ylabel("Score")
    ax.legend(fontsize=8, loc="lower right")
    panel_label(ax, "(a)"); style_axis(ax)

    ax = axes[1]
    ax.bar(x, ab["Smooth"], color=[ABL_COLORS[m] for m in methods], edgecolor=COLOR_BLACK, lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("Smooth")
    panel_label(ax, "(b)"); style_axis(ax)

    ax = axes[2]
    # normalized tradeoff score
    f = ab["Macro-F1"]; s = ab["Smooth"]
    score = 0.6 * ((f - f.min()) / (f.max() - f.min() + 1e-12)) + 0.4 * ((s.max() - s) / (s.max() - s.min() + 1e-12))
    ax.bar(x, score, facecolor="white", edgecolor=[ABL_COLORS[m] for m in methods], hatch="////", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("Trade-off score"); ax.set_ylim(0, 1.05)
    panel_label(ax, "(c)"); style_axis(ax)
    fig.tight_layout()
    save_data(ab.assign(Tradeoff=score), "Fig5_11_ablation_summary_panel_data.csv")
    save_fig(fig, DIRS["ablation"], "Fig5_11_ablation_summary_panel",
             "5.3 消融实验", "Ablation overview panel",
             "Table10_ablation_summary.csv", "第五章 5.3 消融结果总览",
             "A6 在分类性能与概率平滑之间取得折中。")


def plot_ablation_tradeoff(ab: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.4, 5.3))
    x = ab["Smooth"].values; y = ab["Macro-F1"].values
    xx, yy = np.meshgrid(np.linspace(x.min()-0.001, x.max()+0.001, 180), np.linspace(y.min()-0.003, y.max()+0.003, 180))
    score = 0.42*(x.max()+0.001-xx)/(x.max()-x.min()+0.002) + 0.58*(yy-(y.min()-0.003))/(y.max()-y.min()+0.006)
    cmap = LinearSegmentedColormap.from_list("trade_bg", ["#FAFAFA", "#EEF5FB", "#EAF4EC", "#F8EEEE"])
    ax.contourf(xx, yy, score, levels=18, cmap=cmap, alpha=0.75)
    ax.plot(x, y, color=COLOR_GRAY, lw=1.1, ls="--", alpha=0.65)
    for _, r in ab.iterrows():
        m = str(r["Method"]); marker = "*" if m == "A6" else ("D" if m == "A4" else ("P" if m == "A5" else "o"))
        size = 230 if m == "A6" else 95
        ax.scatter(r["Smooth"], r["Macro-F1"], s=size, marker=marker, color=ABL_COLORS[m],
                   edgecolor=COLOR_BLACK if m in ["A4","A5","A6"] else "white", zorder=5)
        ax.text(r["Smooth"]+0.00012, r["Macro-F1"]+0.0008, m, color=ABL_COLORS[m], fontsize=10, fontweight="bold")
    ax.set_xlabel("Smoothness")
    ax.set_ylabel("Macro-F1")
    style_axis(ax); add_axis_arrows(ax)
    save_data(ab, "Fig5_12_ablation_tradeoff_data.csv")
    save_fig(fig, DIRS["ablation"], "Fig5_12_ablation_tradeoff",
             "5.3 消融实验", "Accuracy-smoothness tradeoff",
             "Table10_ablation_summary.csv", "第五章 5.3 trade-off 分析",
             "A4 分类最高，A5 最平滑，A6 是最终折中输出。")


# =========================================================
# 7. Scheme 8: cross-condition generalization
# =========================================================
def combine_cross(dual: pd.DataFrame | None, single: pd.DataFrame | None) -> pd.DataFrame:
    parts = []
    if dual is not None:
        parts.append(dual.assign(Setting="Dual-source"))
    if single is not None:
        parts.append(single.assign(Setting="Single-source"))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = numeric(out, ["Acc", "Macro-F1", "M-F1", "M-Rec", "M→E", "M→L", "Smooth"])
    return out


def plot_cross_heatmap(cross: pd.DataFrame):
    if cross.empty:
        return
    metrics = ["Acc", "Macro-F1", "M-F1", "M-Rec", "Smooth"]
    tasks = cross["Task"].astype(str).unique().tolist()
    methods = ["B8", "B9", "B10", "B11", "B12"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16.0, 5.3), sharey=True)
    for ax, metric in zip(axes, metrics):
        mat = np.full((len(tasks), len(methods)), np.nan)
        for i, t in enumerate(tasks):
            for j, m in enumerate(methods):
                sub = cross[(cross["Task"].astype(str) == t) & (cross["Method"].astype(str) == m)]
                if len(sub):
                    val = float(sub.iloc[0][metric])
                    mat[i, j] = (1 - val) if metric == "Smooth" else val
        im = ax.imshow(mat, cmap="YlGnBu" if metric != "Smooth" else "YlOrRd", vmin=np.nanmin(mat), vmax=np.nanmax(mat), aspect="auto")
        ax.set_title(metric if metric != "Smooth" else "1-Smooth", fontsize=10.5, fontweight="bold")
        ax.set_xticks(np.arange(len(methods))); ax.set_xticklabels(methods, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(tasks))); ax.set_yticklabels(tasks)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    raw = 1 - mat[i, j] if metric == "Smooth" else mat[i, j]
                    ax.text(j, i, f"{raw:.2f}", ha="center", va="center", fontsize=8.2,
                            color="white" if mat[i, j] > np.nanmean(mat) else COLOR_BLACK)
        ax.add_patch(Rectangle((methods.index("B12")-0.5, -0.5), 1, len(tasks), fill=False, ec=COLOR_B12, lw=1.8))
    fig.tight_layout()
    save_data(cross, "Fig5_16_cross_condition_overview_heatmap_data.csv")
    save_fig(fig, DIRS["cross"], "Fig5_16_cross_condition_overview_heatmap",
             "5.2 跨工况泛化", "Cross-condition task-method heatmap",
             "Table12/Table13 cross condition results", "第五章 5.2 跨工况总览",
             "B12 在多种跨工况任务中保持较高泛化性能和平滑性。")


def plot_cross_mean_std(avg: pd.DataFrame | None):
    if avg is None or avg.empty:
        return
    methods = ["B8", "B9", "B10", "B11", "B12"]
    metrics = ["Macro-F1", "M-F1", "M-Rec"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    y = np.arange(len(methods))
    for ax, setting, color, marker in [(axes[0], "Dual-source", COLOR_BLUE, "o"), (axes[1], "Single-source", COLOR_B12, "s")]:
        sub = avg[avg["Setting"].astype(str) == setting].set_index("Method")
        for k, metric in enumerate(metrics):
            mean_col = metric + "_mean"
            std_col = metric + "_std"
            if mean_col not in sub.columns:
                # Some files use Macro-F1_mean etc.; already matched for these metrics.
                continue
            vals = sub.loc[methods, mean_col].astype(float).values
            errs = sub.loc[methods, std_col].astype(float).values if std_col in sub.columns else np.zeros_like(vals)
            ax.errorbar(vals, y + (k - 1) * 0.16, xerr=errs, fmt=marker,
                        color=color, ms=6, lw=1.2, capsize=3, label=metric)
        ax.set_yticks(y); ax.set_yticklabels(methods)
        ax.invert_yaxis()
        ax.set_xlabel("Mean ± std")
        ax.set_title(setting, loc="left", fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        style_axis(ax, "x")
    save_data(avg, "Fig5_17_cross_condition_mean_std_data.csv")
    save_fig(fig, DIRS["cross"], "Fig5_17_cross_condition_mean_std",
             "5.2 跨工况泛化", "Mean/std point-range plot",
             "Table14_average_cross_condition_performance.csv", "第五章 5.2 平均泛化性能",
             "B12 在平均意义下表现稳定，尤其在平滑性与 middle 识别上波动较小。")


def plot_cross_b12_cases(prob: pd.DataFrame | None):
    if prob is None or prob.empty:
        return
    tasks = prob["Task"].astype(str).unique().tolist()[:6]
    n = len(tasks)
    if n == 0:
        return
    fig, axes = plt.subplots(int(np.ceil(n/2)), 2, figsize=(12.6, 3.1*int(np.ceil(n/2))), sharey=True)
    axes = np.ravel(axes)
    for ax, t in zip(axes, tasks):
        sub = prob[prob["Task"].astype(str) == t].copy()
        run = pd.to_numeric(sub["run_id"], errors="coerce").values
        add_stage_background(ax, run, sub["true_stage"].astype(str).str.lower(), alpha=0.22)
        ax.plot(run, sub["p_E"], color=COLOR_E, lw=1.3)
        ax.plot(run, sub["p_M"], color=COLOR_M, lw=1.3)
        ax.plot(run, sub["p_L"], color=COLOR_L, lw=1.3)
        if "q_hat" in sub.columns:
            q = pd.to_numeric(sub["q_hat"], errors="coerce")
            q = (q - q.min()) / (q.max() - q.min() + 1e-12)
            ax.plot(run, q, color=COLOR_BLACK, lw=1.0, ls="--", alpha=0.65)
        ax.set_title(t, loc="left", fontsize=10.2, fontweight="bold")
        ax.set_ylim(-0.03, 1.04)
        style_axis(ax)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend([r"$p_E$", r"$p_M$", r"$p_L$", r"$q_{hat}$"], ncol=4, loc="upper center", bbox_to_anchor=(1.05, 1.22), fontsize=8)
    fig.tight_layout()
    save_data(prob, "Fig5_18_cross_condition_probability_cases_data.csv")
    save_fig(fig, DIRS["cross"], "Fig5_18_cross_condition_probability_cases",
             "5.2 跨工况泛化", "B12 probability cases across tasks",
             "cross_condition_B12_probabilities.csv", "第五章 5.2 跨工况案例图",
             "不同跨工况任务中，B12 的阶段概率仍呈有序演化。")


# =========================================================
# 8. Appendix: 3D surfaces
# =========================================================
def polish_3d(ax):
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1, 1, 1, 0.0))
        axis.pane.set_edgecolor((0.86, 0.86, 0.86, 0.45))
        axis._axinfo["grid"]["color"] = (0.76, 0.76, 0.76, 0.22)
        axis._axinfo["grid"]["linewidth"] = 0.50
    ax.tick_params(axis="both", labelsize=8.5, pad=1)


def plot_3d_surfaces(qdf: pd.DataFrame):
    x = qdf["run_id"].values.astype(float)
    probs = np.vstack([qdf["prob_early"].values, qdf["prob_middle"].values, qdf["prob_late"].values])
    y_dense = np.linspace(0, 2, 80)
    X, Y = np.meshgrid(x, y_dense)
    Z = np.zeros_like(X)
    for i in range(len(x)):
        Z[:, i] = np.interp(y_dense, [0,1,2], probs[:, i])
    fig = plt.figure(figsize=(9.0, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap=cm.turbo, lw=0, antialiased=True, alpha=0.90, cstride=4)
    ax.plot(x, np.zeros_like(x), qdf["prob_early"], color=COLOR_E, lw=2.5)
    ax.plot(x, np.ones_like(x), qdf["prob_middle"], color=COLOR_M, lw=2.5)
    ax.plot(x, np.ones_like(x)*2, qdf["prob_late"], color=COLOR_L, lw=2.5)
    ax.set_xlabel("Run index", labelpad=4); ax.set_ylabel("Stage axis", labelpad=5); ax.set_zlabel("Stage probability", labelpad=5)
    ax.set_yticks([0,1,2]); ax.set_yticklabels(["E","M","L"]); ax.set_zlim(0,1.05)
    ax.view_init(elev=28, azim=-58); ax.set_box_aspect((2.5,1,1.2)); polish_3d(ax)
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.13, label="Probability magnitude")
    save_fig(fig, DIRS["appendix"], "Fig5_S2_3D_stage_probability_surface",
             "5.4 附录扩展", "3D stage probability surface",
             "Data_5_4_A6_probability_wear_trajectory.csv", "附录或高质量备选图",
             "三维曲面展示阶段概率沿生命周期与阶段轴的连续演化。")

    q = qdf["q_pred_norm"].values.astype(float)
    z0 = qdf[["prob_early","prob_middle","prob_late"]].max(axis=1).values.astype(float)
    band = np.linspace(-0.075, 0.075, 38)
    X = np.tile(x, (len(band),1))
    Y = np.clip(q[None,:] + band[:,None], 0, 1)
    atten = np.exp(-0.5*(band[:,None]/0.045)**2)
    Z = z0[None,:]*(0.72+0.28*atten)
    fig = plt.figure(figsize=(9.0, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap=cm.turbo, lw=0, antialiased=True, alpha=0.90, cstride=4)
    ax.plot(x, q, z0, color=COLOR_BLACK, lw=2.2)
    ax.plot(x, q, np.zeros_like(q), color=COLOR_GRAY, lw=1.2, ls="--")
    ax.set_xlabel("Run index", labelpad=4); ax.set_ylabel(r"$q_{pred}$", labelpad=5); ax.set_zlabel("Max probability", labelpad=5)
    ax.set_ylim(0,1.02); ax.set_zlim(0,1.05)
    ax.view_init(elev=28, azim=-62); ax.set_box_aspect((2.5,1,1.2)); polish_3d(ax)
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.13, label="Probability magnitude")
    save_fig(fig, DIRS["appendix"], "Fig5_S3_3D_wear_probability_surface",
             "5.4 附录扩展", "3D confidence surface coupled with q_pred",
             "Data_5_4_A6_probability_wear_trajectory.csv", "附录或高质量备选图",
             "最大阶段概率与连续退化位置共同形成平滑置信曲面。")


# =========================================================
# 9. README and manifest
# =========================================================
def write_readme():
    manifest = pd.DataFrame(MANIFEST)
    manifest.to_csv(OUT_ROOT / "figure_manifest.csv", index=False, encoding="utf-8-sig")
    skipped_text = "\n".join(f"- {s}" for s in SKIPPED) if SKIPPED else "- None"
    readme = f"""# 第五章顶刊风格可视化升级

本目录由 `chapter5_visualization_upgrade.py` 自动生成，所有图片均基于已完成实验结果重绘，不重新训练模型。

## 子目录

- `figures_main`: 主实验、middle 专项、混淆矩阵
- `figures_ablation`: 消融实验与概率推断机制
- `figures_cross_condition`: 跨工况泛化
- `figures_probability`: 阶段概率演化
- `figures_q_representation`: q / 退化位置表征
- `figures_appendix`: 三维曲面与备选图
- `data_exports`: 每张图对应的绘图数据
- `scripts`: 本次绘图脚本备份
- `logs`: 运行日志

## 已生成图片数量

{len(MANIFEST)}

## 缺失或跳过项

{skipped_text}

## 使用建议

优先正文使用：

1. `Fig5_7_probability_evolution_B11_vs_B12`
2. `Fig5_10_ablation_probability_evolution_2x3`
3. `Fig5_6_confusion_matrix_panel_2x2`
4. `Fig5_4_middle_stage_panel`
5. `Fig5_1_main_overview_heatmap`
6. `Fig5_16_cross_condition_overview_heatmap`
7. `Fig5_13_q_evolution_consistency`

扩展或附录使用：

- `Fig5_S1_confusion_matrix_panel_1x4`
- `Fig5_S2_3D_stage_probability_surface`
- `Fig5_S3_3D_wear_probability_surface`
"""
    (OUT_ROOT / "README.md").write_text(readme, encoding="utf-8")


def copy_script_to_output():
    src = Path(__file__).resolve()
    dst = DIRS["scripts"] / src.name
    try:
        shutil.copy2(src, dst)
    except Exception as exc:
        SKIPPED.append(f"Could not copy script: {exc}")


# =========================================================
# 10. Main
# =========================================================
def main():
    print("=" * 110)
    print("Chapter 5 high-quality visualization upgrade")
    print(f"Output root: {OUT_ROOT}")
    print("=" * 110)
    b = load_all_data()

    if b.comparison is not None:
        plot_main_heatmap(b.comparison)
        plot_main_profile(b.comparison)
        plot_main_representative_bars(b.comparison)
        plot_middle_panel(b.comparison)
        plot_middle_error_flow(b.comparison)

    if b.comparison_cm is not None and b.comparison is not None:
        plot_confusion_panel(b.comparison_cm, b.comparison, horizontal=False)
        plot_confusion_panel(b.comparison_cm, b.comparison, horizontal=True)

    if b.comparison_pred is not None:
        plot_probability_b11_vs_b12(b.comparison_pred)
        b12seq = method_sequence(b.comparison_pred, "B12")
        if b12seq is not None:
            plot_probability_stream_and_zoom(b12seq)

    if b.ablation_prob is not None:
        plot_ablation_probability_2x3(b.ablation_prob)
        plot_ablation_probability_2x3_inset(b.ablation_prob)

    if b.ablation is not None:
        plot_ablation_overview(b.ablation)
        plot_ablation_tradeoff(b.ablation)

    if b.q_a6 is not None:
        b.q_a6 = numeric(b.q_a6, ["run_id", "q_true", "q_pred_norm", "prob_early", "prob_middle", "prob_late"])
        plot_q_consistency(b.q_a6)
        plot_q_stage_distribution(b.q_a6)
        plot_q_probability_phase(b.q_a6)
        plot_3d_surfaces(b.q_a6)

    if b.comparison_pred is not None:
        plot_representation_space_grid(b.comparison_pred, b.q_a6)

    cross = combine_cross(b.cross_dual, b.cross_single)
    if not cross.empty:
        plot_cross_heatmap(cross)
    if b.cross_avg is not None:
        plot_cross_mean_std(b.cross_avg)
    if b.cross_b12_prob is not None:
        plot_cross_b12_cases(b.cross_b12_prob)
        plot_cross_qhat_small_multiples(b.cross_b12_prob)
        plot_cross_probability_small_multiples(b.cross_b12_prob)

    copy_script_to_output()
    write_readme()
    print("=" * 110)
    print(f"Finished. Generated {len(MANIFEST)} figures.")
    print(f"Manifest: {OUT_ROOT / 'figure_manifest.csv'}")
    print(f"README  : {OUT_ROOT / 'README.md'}")
    print("=" * 110)


if __name__ == "__main__":
    main()

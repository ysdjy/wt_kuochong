# -*- coding: utf-8 -*-
r"""
消融实验 A1-A6：分类指标柱状图 + Smooth 折线 + true-stage probability 箱线图

为什么不用 Macro-F1 bootstrap 箱线图：
    当前 ablation_probabilities_test_C6.csv 中 A1-A4 的 argmax 预测标签完全一致，
    因此 Macro-F1 重采样分布会高度重合，放在图里容易误导。

本脚本改用每个测试窗口的 true-stage probability：
    p_true = p_E if true stage is early,
             p_M if true stage is middle,
             p_L if true stage is late.

这样箱线图表示每种概率输出策略对真实阶段的概率支持分布，能真实展示
Raw / Raw+Fine / Raw+Prior / Mix / Ordered / Final 的概率层面差异。

输出：
    C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文\11_第五章图像字号优化
    \Fig_ablation_key_methods_with_boxplot.png
"""

from __future__ import annotations

from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# =========================================================
# 0. Paths
# =========================================================
ROOT = Path.home() / "Desktop"
PROB_FILE_NAME = "ablation_probabilities_test_C6.csv"
OUT_DIR_NAME = "11_第五章图像字号优化"

METHOD_ORDER = ["A1", "A2", "A3", "A4", "A5", "A6"]
DPI = 900


def find_project_file(filename: str) -> Path:
    matches = list(ROOT.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Cannot find {filename} under {ROOT}")
    # Prefer the formal ablation experiment directory.
    for p in matches:
        if "6_ablation_experiment" in str(p):
            return p
    return matches[0]


PROB_FILE = find_project_file(PROB_FILE_NAME)
PROJECT_ROOT = PROB_FILE.parents[1]
OUT_DIR = PROJECT_ROOT / OUT_DIR_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "Fig_ablation_key_methods_with_boxplot.png"
TRUE_PROB_EXPORT = OUT_DIR / "ablation_true_stage_probability_distribution.csv"


# =========================================================
# 1. Mean data, same as the current visual baseline
# =========================================================
ABLATION_DATA = """Method,Acc,Macro-F1,E-F1,M-F1,L-F1,M-Pre,M-Rec,Smooth
A1,0.9705,0.9601,0.9625,0.9532,0.9880,0.9852,0.9561,0.0236
A2,0.9803,0.9732,0.9678,0.9772,0.9778,0.9834,0.9663,0.0199
A3,0.9816,0.9772,0.9725,0.9782,0.9876,0.9889,0.9678,0.0209
A4,0.9901,0.9902,0.9825,0.9882,1.0000,1.0000,0.9767,0.0236
A5,0.9770,0.9775,0.9711,0.9725,0.9889,0.9841,0.9612,0.0146
A6,0.9868,0.9871,0.9825,0.9844,0.9945,0.9921,0.9767,0.0188
"""


PROB_PREFIX = {
    "A1": "p_raw",
    "A2": "p_raw_fine",
    "A3": "p_raw_prior",
    "A4": "p_mix",
    "A5": "alpha",
    "A6": "p_final",
}

STAGE_SUFFIX = {
    "early": "E",
    "middle": "M",
    "late": "L",
}


def load_ablation_data() -> pd.DataFrame:
    df = pd.read_csv(StringIO(ABLATION_DATA.strip()))
    for col in ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Pre", "M-Rec", "Smooth"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_true_stage_probability_distribution() -> pd.DataFrame:
    df = pd.read_csv(PROB_FILE)
    if "true_stage" not in df.columns:
        raise ValueError(f"`true_stage` column is missing in {PROB_FILE}")

    rows = []
    true_stage = df["true_stage"].astype(str).str.lower()
    for method in METHOD_ORDER:
        prefix = PROB_PREFIX[method]
        needed = [f"{prefix}_E", f"{prefix}_M", f"{prefix}_L"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Missing probability columns for {method}: {missing}")

        for stage, suffix in STAGE_SUFFIX.items():
            mask = true_stage == stage
            vals = pd.to_numeric(df.loc[mask, f"{prefix}_{suffix}"], errors="coerce").dropna().values
            for v in vals:
                rows.append({
                    "Method": method,
                    "true_stage": stage,
                    "p_true_stage": float(v),
                })

    out = pd.DataFrame(rows)
    out.to_csv(TRUE_PROB_EXPORT, index=False, encoding="utf-8-sig")
    print(f"Saved true-stage probability distribution: {TRUE_PROB_EXPORT}")
    return out


def prepare_boxplot_data(true_prob_df: pd.DataFrame) -> list[np.ndarray]:
    return [
        true_prob_df.loc[true_prob_df["Method"] == method, "p_true_stage"].dropna().astype(float).values
        for method in METHOD_ORDER
    ]


# =========================================================
# 2. Style
# =========================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

plt.rcParams["font.size"] = 17
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["axes.titlesize"] = 19
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 13

COLOR_BLUE = "#1F4DFF"
COLOR_GREEN = "#18A558"
COLOR_CYAN = "#17BECF"
COLOR_ORANGE = "#FF9F1C"
COLOR_PURPLE = "#8E44AD"
COLOR_BLACK = "#111111"
COLOR_GRID = "#D9D9D9"
COLOR_B12 = "#B22222"
COLOR_BOX_EDGE = "#222222"
COLOR_BOX_FACE = "#FFF4CC"


# =========================================================
# 3. Axis helpers
# =========================================================
def add_left_y_axis_arrow(ax):
    ax.spines["left"].set_visible(False)
    ax.annotate(
        "",
        xy=(0, 1.035),
        xytext=(0, 0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            lw=1.15,
            color=COLOR_BLACK,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=13,
        ),
        clip_on=False,
        zorder=10,
    )


def style_left_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(COLOR_BLACK)
    ax.spines["bottom"].set_linewidth(1.10)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", colors=COLOR_BLACK, labelsize=16, width=1.1, length=5.0)
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.65, alpha=0.60, color=COLOR_GRID)
    ax.set_axisbelow(True)
    add_left_y_axis_arrow(ax)


def style_right_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.spines["right"].set_color(COLOR_BLACK)
    ax.spines["right"].set_linewidth(1.10)
    ax.tick_params(axis="y", colors=COLOR_BLACK, labelsize=16, width=1.1, length=5.0)


def add_bar_caps(ax, bars, color):
    for bar in bars:
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        cap_w = bar.get_width() * 0.45
        ax.plot([x - cap_w / 2, x + cap_w / 2], [y, y], color=color, linewidth=1.05, zorder=5)


def add_boxplot_overlay(ax, x: np.ndarray, box_data: list[np.ndarray]) -> Patch:
    bp = ax.boxplot(
        box_data,
        positions=x,
        widths=0.082,
        patch_artist=True,
        manage_ticks=False,
        showfliers=False,
        zorder=8,
        medianprops=dict(color=COLOR_BLACK, linewidth=1.35),
        whiskerprops=dict(color=COLOR_BOX_EDGE, linewidth=1.15),
        capprops=dict(color=COLOR_BOX_EDGE, linewidth=1.15),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(COLOR_BOX_FACE)
        patch.set_alpha(0.82)
        patch.set_edgecolor(COLOR_BOX_EDGE)
        patch.set_linewidth(1.2)
        patch.set_hatch("///")
        patch.set_zorder(8)
    for key in ["whiskers", "caps", "medians"]:
        for artist in bp[key]:
            artist.set_zorder(9)
    return Patch(
        facecolor=COLOR_BOX_FACE,
        edgecolor=COLOR_BOX_EDGE,
        alpha=0.82,
        hatch="///",
        label="True-stage probability distribution",
    )


# =========================================================
# 4. Plot
# =========================================================
def plot_ablation_with_boxplot(mean_df: pd.DataFrame, true_prob_df: pd.DataFrame):
    methods = mean_df["Method"].tolist()
    x = np.arange(len(methods))

    bar_metrics = ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1"]
    edge_colors = [COLOR_BLUE, COLOR_GREEN, COLOR_CYAN, COLOR_ORANGE, COLOR_PURPLE]
    hatches = ["////", "\\\\\\\\", "....", "xxxx", "----"]
    width = 0.13

    fig, ax1 = plt.subplots(figsize=(12.9, 6.15))

    for j, (metric, color, hatch) in enumerate(zip(bar_metrics, edge_colors, hatches)):
        xs = x + (j - 2) * width
        bars = ax1.bar(
            xs,
            mean_df[metric],
            width=width,
            facecolor="white",
            edgecolor=color,
            linewidth=1.35,
            hatch=hatch,
            label=metric,
            zorder=3,
        )
        add_bar_caps(ax1, bars, color)

    ax1.axvspan(5 - 0.48, 5 + 0.48, color=COLOR_B12, alpha=0.07, zorder=0)

    box_handle = add_boxplot_overlay(ax1, x, prepare_boxplot_data(true_prob_df))

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontstyle="normal", fontsize=16)
    ax1.set_ylabel("Classification metric", fontsize=18)
    ax1.set_ylim(0.60, 1.03)
    style_left_axis(ax1, grid_axis="y")

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        mean_df["Smooth"],
        color=COLOR_B12,
        marker="*",
        markersize=13,
        linewidth=2.3,
        label="Smooth",
        zorder=10,
    )
    ax2.set_ylabel("Smooth", fontsize=18, labelpad=10)
    ax2.set_ylim(0.012, 0.026)
    style_right_axis(ax2)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    handles = h1 + [box_handle] + h2
    labels = l1 + [box_handle.get_label()] + l2

    ax1.legend(
        handles,
        labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        fontsize=12.8,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.15,
        handletextpad=0.55,
    )

    fig.subplots_adjust(left=0.085, right=0.925, top=0.82, bottom=0.14)
    fig.savefig(OUT_FILE, dpi=DPI, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(f"Saved PNG: {OUT_FILE}")


def main():
    mean_df = load_ablation_data()
    true_prob_df = load_true_stage_probability_distribution()

    print("A1-A6 true-stage probability summary:")
    print(
        true_prob_df.groupby("Method")["p_true_stage"]
        .agg(["mean", "std", "min", "median", "max"])
        .reindex(METHOD_ORDER)
        .round(4)
        .to_string()
    )

    plot_ablation_with_boxplot(mean_df, true_prob_df)


if __name__ == "__main__":
    main()

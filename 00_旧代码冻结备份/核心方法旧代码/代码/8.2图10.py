# -*- coding: utf-8 -*-
r"""
三联图：
(a) Middle-stage recognition
(b) Middle-stage misclassification
(c) Transition stability and smoothness

修改：
1. 只去掉第三个小图 (c) 的 x 轴箭头；
2. 去掉第三个小图 (c) 的右侧 Smoothness 轴箭头；
3. 保留第三个小图左侧 y 轴箭头；
4. 其余内容不变。

输出：
C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文\11_第五章图像字号优化
\Fig_middle_stage_triptych_largefont_no_c_x_arrow.png
"""

from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 0. 输出路径
# =========================================================
OUT_DIR = Path(
    r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文\11_第五章图像字号优化"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 900


# =========================================================
# 1. 数据
# =========================================================
SUMMARY_DATA = """Method,Acc,Macro-F1,E-F1,M-F1,L-F1,M-Pre,M-Rec,M_E,M_L,Rev,Jump,Smooth
B1,0.6447,0.6145,0.4587,0.5970,0.7879,0.5755,0.6202,0.0000,0.3798,0,0,0.0132
B2,0.5757,0.4871,0.7304,0.0000,0.7309,0.0000,0.0000,0.4806,0.5194,5,11,0.0726
B3,0.9474,0.9490,0.9231,0.9350,0.9889,0.9829,0.8915,0.1085,0.0000,5,0,0.1096
B4,0.8651,0.8695,0.8737,0.8379,0.8970,0.8548,0.8217,0.1783,0.0000,9,0,0.1578
B5,0.9770,0.9773,0.9651,0.9723,0.9945,0.9919,0.9535,0.0388,0.0078,5,0,0.1105
B6,0.8783,0.8765,0.8054,0.8702,0.9540,0.7949,0.9612,0.0388,0.0000,9,0,0.1390
B7,0.8618,0.8651,0.8889,0.8174,0.8889,0.9307,0.7287,0.1628,0.1085,16,0,0.1554
B8,0.9539,0.9550,0.9385,0.9426,0.9838,1.0000,0.8915,0.0853,0.0233,10,0,0.1245
B9,0.9539,0.9560,0.9825,0.9426,0.9430,1.0000,0.8915,0.0233,0.0853,1,0,0.0398
B10,0.8684,0.8746,0.8317,0.8261,0.9659,0.9406,0.7364,0.2636,0.0000,0,0,0.0219
B11,0.9901,0.9902,0.9825,0.9882,1.0000,1.0000,0.9767,0.0233,0.0000,0,0,0.0236
B12,0.9868,0.9871,0.9825,0.9844,0.9945,0.9921,0.9767,0.0233,0.0000,0,0,0.0188
"""


def load_data() -> pd.DataFrame:
    df = pd.read_csv(StringIO(SUMMARY_DATA))
    numeric_cols = [
        "Acc", "Macro-F1", "E-F1", "M-F1", "L-F1",
        "M-Pre", "M-Rec", "M_E", "M_L", "Rev", "Jump", "Smooth"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["order"] = df["Method"].str.extract(r"B(\d+)").astype(int)
    df = df.sort_values("order").reset_index(drop=True)
    return df


# =========================================================
# 2. 全局绘图风格
# =========================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

plt.rcParams["font.size"] = 16
plt.rcParams["axes.labelsize"] = 17
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 12


# =========================================================
# 3. 颜色
# =========================================================
COLORS = {
    "M-Pre":  "#D55E00",
    "M-Rec":  "#0072B2",
    "M-F1":   "#009E73",
    "M→E":    "#CC79A7",
    "M→L":    "#E69F00",
    "Rev":    "#56B4E9",
    "Jump":   "#8C564B",
    "Smooth": "#222222",
}

HIGHLIGHT = "#F1E6E6"
GRID_COLOR = "#D8D8D8"
AXIS_COLOR = "#222222"


# =========================================================
# 4. 工具函数
# =========================================================
def add_axis_arrows(ax, x_pad=0.018, y_pad=0.020, show_x=True, show_y=True):
    """
    坐标轴箭头控制函数。
    show_x=False 时不画 x 轴箭头；
    show_y=False 时不画 y 轴箭头。
    """
    arrow_kw = dict(
        arrowstyle="-|>",
        lw=1.1,
        color=AXIS_COLOR,
        shrinkA=0,
        shrinkB=0,
        mutation_scale=10
    )

    if show_x:
        ax.annotate(
            "",
            xy=(1.0 + x_pad, 0.0),
            xytext=(0.0, 0.0),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=arrow_kw,
            clip_on=False,
            zorder=20,
        )

    if show_y:
        ax.annotate(
            "",
            xy=(0.0, 1.0 + y_pad),
            xytext=(0.0, 0.0),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=arrow_kw,
            clip_on=False,
            zorder=20,
        )


def style_axis(ax, add_arrows=True, show_x_arrow=True, show_y_arrow=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 如果有箭头，就隐藏对应轴线；没有箭头，则保留正常轴线
    ax.spines["left"].set_visible(not (add_arrows and show_y_arrow))
    ax.spines["bottom"].set_visible(not (add_arrows and show_x_arrow))

    if ax.spines["left"].get_visible():
        ax.spines["left"].set_color(AXIS_COLOR)
        ax.spines["left"].set_linewidth(1.0)

    if ax.spines["bottom"].get_visible():
        ax.spines["bottom"].set_color(AXIS_COLOR)
        ax.spines["bottom"].set_linewidth(1.0)

    ax.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.55,
        color=GRID_COLOR
    )
    ax.set_axisbelow(True)

    ax.tick_params(
        axis="both",
        direction="in",
        width=1.0,
        length=4.5,
        colors=AXIS_COLOR,
        labelsize=14
    )

    if add_arrows:
        add_axis_arrows(
            ax,
            show_x=show_x_arrow,
            show_y=show_y_arrow
        )


def add_highlight_for_last_group(ax, x_center, half_width=0.55):
    ax.axvspan(
        x_center - half_width,
        x_center + half_width,
        color=HIGHLIGHT,
        alpha=0.85,
        zorder=0
    )


# =========================================================
# 5. 子图 (a)
# =========================================================
def plot_panel_a(ax, df):
    methods = ["B1", "B5", "B8", "B12"]
    sub = df[df["Method"].isin(methods)].copy()
    sub["Method"] = pd.Categorical(sub["Method"], methods, ordered=True)
    sub = sub.sort_values("Method")

    x = np.arange(len(methods))
    w = 0.22

    add_highlight_for_last_group(ax, x_center=x[-1], half_width=0.55)

    ax.bar(
        x - w,
        sub["M-Pre"].values,
        width=w,
        facecolor="white",
        edgecolor=COLORS["M-Pre"],
        linewidth=1.6,
        hatch="////",
        label="M-Pre",
        zorder=3,
        )
    ax.bar(
        x,
        sub["M-Rec"].values,
        width=w,
        facecolor="white",
        edgecolor=COLORS["M-Rec"],
        linewidth=1.6,
        hatch="\\\\\\\\",
        label="M-Rec",
        zorder=3,
    )
    ax.bar(
        x + w,
        sub["M-F1"].values,
        width=w,
        facecolor="white",
        edgecolor=COLORS["M-F1"],
        linewidth=1.6,
        hatch="....",
        label="M-F1",
        zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=14)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Middle-stage score", fontsize=16)
    ax.set_title("(a) Middle-stage recognition", fontsize=18, fontweight="bold", pad=8)

    style_axis(ax, add_arrows=True)

    ax.legend(
        loc="upper left",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.01, 1.01),
        handlelength=1.6,
        columnspacing=1.0,
        handletextpad=0.45,
        fontsize=12
    )


# =========================================================
# 6. 子图 (b)
# =========================================================
def plot_panel_b(ax, df):
    methods = ["B1", "B5", "B8", "B12"]
    sub = df[df["Method"].isin(methods)].copy()
    sub["Method"] = pd.Categorical(sub["Method"], methods, ordered=True)
    sub = sub.sort_values("Method")

    x = np.arange(len(methods))
    w = 0.28

    add_highlight_for_last_group(ax, x_center=x[-1], half_width=0.55)

    ax.bar(
        x - w / 2,
        sub["M_E"].values,
        width=w,
        facecolor="white",
        edgecolor=COLORS["M→E"],
        linewidth=1.6,
        hatch="////",
        label="M→E",
        zorder=3,
        )
    ax.bar(
        x + w / 2,
        sub["M_L"].values,
        width=w,
        facecolor="white",
        edgecolor=COLORS["M→L"],
        linewidth=1.6,
        hatch="\\\\\\\\",
        label="M→L",
        zorder=3,
        )

    ax.text(
        x[-1] + w / 2,
        max(sub["M_L"].values[-1], 0.0) + 0.010,
        "0",
        color=COLORS["M→L"],
        fontsize=12,
        ha="center",
        va="bottom",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=14)
    ax.set_ylim(0, 0.43)
    ax.set_ylabel("Misclassification rate", fontsize=16)
    ax.set_title("(b) Middle-stage misclassification", fontsize=18, fontweight="bold", pad=8)

    style_axis(ax, add_arrows=True)

    ax.legend(
        loc="upper right",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.98, 1.01),
        handlelength=1.6,
        columnspacing=1.0,
        handletextpad=0.45,
        fontsize=12
    )


# =========================================================
# 7. 子图 (c)
# =========================================================
def plot_panel_c(ax, df):
    methods = [f"B{i}" for i in range(1, 13)]
    sub = df[df["Method"].isin(methods)].copy()
    sub["Method"] = pd.Categorical(sub["Method"], methods, ordered=True)
    sub = sub.sort_values("Method")

    x = np.arange(len(methods))
    w = 0.36

    add_highlight_for_last_group(ax, x_center=x[-1], half_width=0.55)

    ax.bar(
        x - w / 2,
        sub["Rev"].values,
        width=w,
        facecolor="white",
        edgecolor=COLORS["Rev"],
        linewidth=1.6,
        hatch="////",
        label="Rev",
        zorder=3,
        )
    ax.bar(
        x + w / 2,
        sub["Jump"].values,
        width=w,
        facecolor="white",
        edgecolor=COLORS["Jump"],
        linewidth=1.6,
        hatch="\\\\\\\\",
        label="Jump",
        zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylim(0, 18.8)
    ax.set_ylabel("Transition count", fontsize=16)
    ax.set_title("(c) Transition stability and smoothness", fontsize=18, fontweight="bold", pad=8)

    # 关键修改：
    # 第三个图是 B1-B12 类别轴，x 轴不加箭头；
    # 左 y 轴为 Transition count，保留箭头。
    style_axis(
        ax,
        add_arrows=True,
        show_x_arrow=False,
        show_y_arrow=True
    )

    # 右轴 Smooth
    ax2 = ax.twinx()
    ax2.plot(
        x,
        sub["Smooth"].values,
        color=COLORS["Smooth"],
        marker="o",
        linewidth=1.6,
        markersize=4.2,
        label="Smooth",
        zorder=5,
    )

    ax2.plot(
        x[-1],
        sub["Smooth"].values[-1],
        marker="*",
        markersize=10,
        color=COLORS["Smooth"],
        zorder=7,
    )
    ax2.text(
        x[-1] - 0.05,
        sub["Smooth"].values[-1] + 0.004,
        f"{sub['Smooth'].values[-1]:.4f}",
        color=COLORS["Smooth"],
        fontsize=11,
        ha="right",
        va="bottom",
        )

    ax2.set_ylim(0, 0.18)
    ax2.set_ylabel("Smoothness", fontsize=16)
    ax2.tick_params(axis="y", labelsize=12, colors=AXIS_COLOR)

    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color(AXIS_COLOR)
    ax2.spines["right"].set_linewidth(1.0)

    # 关键修改：
    # 右侧 Smoothness 是 secondary y-axis，不再加箭头，避免视觉过重。
    # 原来的 ax2.annotate(...) 已删除。

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper right",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.98, 1.01),
        handlelength=1.6,
        columnspacing=1.0,
        handletextpad=0.45,
        fontsize=12
    )


# =========================================================
# 8. 主函数
# =========================================================
def main():
    df = load_data()

    fig, axes = plt.subplots(
        1, 3,
        figsize=(19.2, 5.5),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.35]}
    )

    plot_panel_a(axes[0], df)
    plot_panel_b(axes[1], df)
    plot_panel_c(axes[2], df)

    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.88,
        bottom=0.18,
        wspace=0.28
    )

    out_path = OUT_DIR / "Fig_middle_stage_triptych_largefont_no_c_x_arrow.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print(f"Saved PNG: {out_path}")


if __name__ == "__main__":
    main()
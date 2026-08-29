# -*- coding: utf-8 -*-
r"""
消融实验：Smoothness / Macro-F1 trade-off 优化图

优化点：
1. 字体整体放大；
2. 背景 trade-off score 更淡；
3. 注释位置优化；
4. 去掉 x 轴箭头，仅保留左 y 轴箭头；
5. 图例放大并保持左下角紧凑；
6. 仅输出 PNG。

输出：
C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文\11_第五章图像字号优化
\Fig15A_ablation_accuracy_smoothness_tradeoff_largefont.png
"""

from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =========================================================
# 0. 输出路径
# =========================================================
OUT_DIR = Path(
    r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文\11_第五章图像字号优化"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "Fig15A_ablation_accuracy_smoothness_tradeoff_largefont.png"
DPI = 900


# =========================================================
# 1. 数据
# =========================================================
ABLATION_DATA = """Method,Method_name,Acc,Macro-F1,E-F1,M-F1,L-F1,M-Pre,M-Rec,Smooth
A1,Raw,0.9705,0.9601,0.9625,0.9532,0.9880,0.9852,0.9561,0.0236
A2,Raw + Fine,0.9803,0.9732,0.9678,0.9772,0.9778,0.9834,0.9663,0.0199
A3,Raw + Prior,0.9816,0.9772,0.9725,0.9782,0.9876,0.9889,0.9678,0.0209
A4,Mix,0.9901,0.9902,0.9825,0.9882,1.0000,1.0000,0.9767,0.0236
A5,Ordered,0.9770,0.9775,0.9711,0.9725,0.9889,0.9841,0.9612,0.0146
A6,Final,0.9868,0.9871,0.9825,0.9844,0.9945,0.9921,0.9767,0.0188
"""


def load_ablation_data() -> pd.DataFrame:
    df = pd.read_csv(StringIO(ABLATION_DATA.strip()))
    num_cols = ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Pre", "M-Rec", "Smooth"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# =========================================================
# 2. 全局风格
# =========================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

plt.rcParams["font.size"] = 16
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["axes.titlesize"] = 19
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["legend.fontsize"] = 12


# =========================================================
# 3. 配色
# =========================================================
COLOR_BLACK = "#222222"
COLOR_GRID = "#D7D7D7"

METHOD_COLORS = {
    "A1": "#6B6B6B",
    "A2": "#2F6DAE",
    "A3": "#2AA6B8",
    "A4": "#218C45",
    "A5": "#D98B16",
    "A6": "#B22222",
}

METHOD_MARKERS = {
    "A1": "o",
    "A2": "s",
    "A3": "^",
    "A4": "D",
    "A5": "P",
    "A6": "*",
}

METHOD_NAMES = {
    "A1": "Raw",
    "A2": "Raw + Fine",
    "A3": "Raw + Prior",
    "A4": "Mix",
    "A5": "Ordered",
    "A6": "Final",
}


# =========================================================
# 4. 坐标轴工具
# =========================================================
def add_axis_arrows(ax):
    """
    同时给 x 轴和左 y 轴加箭头。
    """
    # x 轴箭头
    ax.annotate(
        "",
        xy=(1.025, 0),
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
        zorder=20,
    )

    # y 轴箭头
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
        zorder=20,
    )


def style_axis(ax):
    """
    x 轴和左 y 轴都使用箭头形式。
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 隐藏左轴和下轴，用箭头替代
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.tick_params(
        axis="both",
        labelsize=15,
        colors=COLOR_BLACK,
        width=1.05,
        length=4.8,
    )

    ax.grid(
        True,
        axis="both",
        linestyle="--",
        linewidth=0.65,
        alpha=0.42,
        color=COLOR_GRID,
    )
    ax.set_axisbelow(True)

    add_axis_arrows(ax)

# =========================================================
# 5. 主绘图函数
# =========================================================
def plot_tradeoff(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9.0, 6.5))

    x_min, x_max = 0.0136, 0.0248
    y_min, y_max = 0.956, 0.993

    # -----------------------------------------------------
    # Soft trade-off background
    # 越靠左上越优：lower Smooth + higher Macro-F1
    # -----------------------------------------------------
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 260),
        np.linspace(y_min, y_max, 260),
    )

    smooth_score = (x_max - xx) / (x_max - x_min)
    f1_score = (yy - y_min) / (y_max - y_min)
    score = 0.42 * smooth_score + 0.58 * f1_score

    cmap = LinearSegmentedColormap.from_list(
        "tradeoff_bg",
        ["#FBFBFB", "#EEF5FB", "#EAF4EC", "#F7EAEA"],
    )

    ax.contourf(
        xx,
        yy,
        score,
        levels=18,
        cmap=cmap,
        alpha=0.58,
        zorder=0,
    )

    # 等值线更淡
    cs = ax.contour(
        xx,
        yy,
        score,
        levels=[0.35, 0.50, 0.65, 0.80],
        colors="#B8B8B8",
        linewidths=0.50,
        linestyles="--",
        alpha=0.40,
        zorder=1,
    )
    ax.clabel(
        cs,
        inline=True,
        fontsize=8.5,
        fmt="%.2f",
        colors="#999999",
    )

    # 高性能区域淡色提示
    ax.axvspan(x_min, 0.0192, color="#DCEEDD", alpha=0.12, lw=0, zorder=0)
    ax.axhspan(0.984, y_max, color="#F7D9D7", alpha=0.10, lw=0, zorder=0)

    # -----------------------------------------------------
    # Trade-off path
    # -----------------------------------------------------
    path_methods = ["A1", "A2", "A3", "A4", "A6", "A5"]
    path = df.set_index("Method").loc[path_methods]

    ax.plot(
        path["Smooth"],
        path["Macro-F1"],
        color="#8A8A8A",
        linestyle="--",
        linewidth=1.35,
        alpha=0.62,
        zorder=2,
        label="Trade-off path",
    )

    # -----------------------------------------------------
    # Points
    # -----------------------------------------------------
    for _, row in df.iterrows():
        method = row["Method"]
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]

        size = 310 if method == "A6" else (145 if method in ["A4", "A5"] else 105)
        edge = COLOR_BLACK if method in ["A4", "A5", "A6"] else "white"
        lw = 1.05 if method in ["A4", "A5", "A6"] else 0.70

        ax.scatter(
            row["Smooth"],
            row["Macro-F1"],
            s=size,
            marker=marker,
            color=color,
            edgecolor=edge,
            linewidth=lw,
            alpha=0.96,
            zorder=6 if method == "A6" else 5,
            label=f"{method} {METHOD_NAMES.get(method, '')}",
        )

        # method labels
        dx, dy = 0.00018, 0.00115
        if method == "A1":
            dx, dy = 0.00020, 0.00100
        elif method == "A2":
            dx, dy = -0.00035, 0.00110
        elif method == "A3":
            dx, dy = -0.00030, 0.00125
        elif method == "A4":
            dx, dy = 0.00022, 0.00120
        elif method == "A5":
            dx, dy = -0.00050, -0.0020
        elif method == "A6":
            dx, dy = 0.00024, 0.00118

        ax.text(
            row["Smooth"] + dx,
            row["Macro-F1"] + dy,
            method,
            fontsize=12,
            color=color,
            fontweight="bold" if method in ["A4", "A5", "A6"] else "normal",
            zorder=8,
            )

    # -----------------------------------------------------
    # Key annotations
    # -----------------------------------------------------
    a4 = df[df["Method"] == "A4"].iloc[0]
    a5 = df[df["Method"] == "A5"].iloc[0]
    a6 = df[df["Method"] == "A6"].iloc[0]

    ax.annotate(
        "highest Macro-F1",
        xy=(a4["Smooth"], a4["Macro-F1"]),
        xytext=(0.0216, 0.9916),
        arrowprops=dict(arrowstyle="->", lw=1.0, color=COLOR_BLACK),
        fontsize=11,
        color=COLOR_BLACK,
    )

    ax.annotate(
        "lowest Smooth",
        xy=(a5["Smooth"], a5["Macro-F1"]),
        xytext=(0.0158, 0.9713),
        arrowprops=dict(arrowstyle="->", lw=1.05, color=METHOD_COLORS["A5"]),
        fontsize=11,
        color="#1F4DFF",
    )

    ax.annotate(
        "balanced trade-off",
        xy=(a6["Smooth"], a6["Macro-F1"]),
        xytext=(0.0168, 0.9900),
        arrowprops=dict(arrowstyle="->", lw=1.10, color=METHOD_COLORS["A6"]),
        fontsize=11.3,
        color=METHOD_COLORS["A6"],
        fontweight="bold",
    )

    # Direction text and arrow
    ax.text(
        0.01405,
        0.98705,
        "higher Macro-F1",
        fontsize=11.2,
        color=COLOR_BLACK,
        fontweight="bold",
    )
    ax.text(
        0.01405,
        0.98535,
        "lower Smooth",
        fontsize=11.2,
        color=COLOR_BLACK,
        fontweight="bold",
    )

    ax.annotate(
        "",
        xy=(0.01470, 0.99170),
        xytext=(0.01720, 0.98425),
        arrowprops=dict(arrowstyle="->", lw=1.1, color=COLOR_BLACK),
        zorder=3,
    )

    # -----------------------------------------------------
    # Axes
    # -----------------------------------------------------
    ax.set_xlabel("Smoothness", fontsize=18)
    ax.set_ylabel("Macro-F1", fontsize=18)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    style_axis(ax)

    # -----------------------------------------------------
    # Legend
    # -----------------------------------------------------
    handles, labels = ax.get_legend_handles_labels()

    ax.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.018, 0.018),
        fontsize=10.5,
        frameon=True,
        borderpad=0.42,
        labelspacing=0.35,
        handletextpad=0.50,
        ncol=2,
    )

    fig.subplots_adjust(
        left=0.115,
        right=0.985,
        top=0.955,
        bottom=0.125,
    )

    fig.savefig(
        OUT_FILE,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.035,
    )
    plt.close(fig)

    print(f"Saved PNG: {OUT_FILE}")


# =========================================================
# 6. 运行
# =========================================================
if __name__ == "__main__":
    df = load_ablation_data()
    plot_tradeoff(df)
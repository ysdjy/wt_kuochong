# -*- coding: utf-8 -*-
"""
重新绘制 B3-B12 多指标性能轮廓图，并放大所有数字与文字字号。

输出目录：
C:\\Users\\wangting\\Desktop\\博士开题\\公开数据\\1PHM\\PHM实验\\小论文\\11_第五章图像字号优化

输出文件：
Fig7A_overall_multimetric_profile_largefont.png
Fig7A_overall_multimetric_profile_largefont.pdf
"""

from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 0. 输出路径
# =========================================================
ROOT = Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文")
OUT_DIR = ROOT / "11_第五章图像字号优化"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 900


# =========================================================
# 1. 数据
# =========================================================
SUMMARY_DATA = """Method,Method_name,Stage_definition,Model_type,Acc,Macro-F1,E-F1,M-F1,L-F1,M-Pre,M-Rec,M→E,M→L,Rev,Jump,Smooth
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
B12,DC-PSR,Relative-stage,Proposed,0.9868,0.9871,0.9825,0.9844,0.9945,0.9921,0.9767,0.0233,0.0000,0,0,0.0188
"""


def load_data() -> pd.DataFrame:
    df = pd.read_csv(StringIO(SUMMARY_DATA))
    numeric_cols = [
        "Acc", "Macro-F1", "E-F1", "M-F1", "L-F1",
        "M-Pre", "M-Rec", "M→E", "M→L", "Rev", "Jump", "Smooth"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["order"] = df["Method"].str.extract(r"B(\d+)").astype(int)
    df = df.sort_values("order").reset_index(drop=True)
    return df


# =========================================================
# 2. 全局绘图风格：字号整体放大
# =========================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

# 重点：整体字号放大
plt.rcParams["font.size"] = 18
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["legend.fontsize"] = 16


# =========================================================
# 3. 颜色与标记
# =========================================================
COLOR_GREEN = "#18A558"
COLOR_ORANGE = "#FF9F1C"
COLOR_BLUE = "#1F4DFF"
COLOR_CYAN = "#17BECF"
COLOR_PURPLE = "#8E44AD"
COLOR_BLACK = "#111111"
COLOR_B12 = "#B22222"
COLOR_GRID = "#DADADA"

PALETTE = {
    "B3":  ("#7F7F7F", "^", "-.", 2.0),
    "B4":  ("#8C6BB1", "D", ":", 2.0),
    "B5":  (COLOR_GREEN, "v", "-", 2.2),
    "B6":  (COLOR_ORANGE, "P", "--", 2.0),
    "B7":  ("#B279A2", "X", "-.", 2.0),
    "B8":  (COLOR_BLUE, "<", ":", 2.2),
    "B9":  (COLOR_CYAN, ">", "--", 2.2),
    "B10": (COLOR_PURPLE, "h", "-.", 2.2),
    "B11": (COLOR_BLACK, "p", "-", 3.0),
    "B12": (COLOR_B12, "*", "-", 4.0),
}


# =========================================================
# 4. 工具函数
# =========================================================
def add_axis_arrows(ax, x_pad=0.035, y_pad=0.045):
    """给坐标轴添加箭头。"""
    arrow_kw = dict(
        arrowstyle="-|>",
        lw=1.35,
        color="#222222",
        shrinkA=0,
        shrinkB=0,
        mutation_scale=14
    )

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


def style_axis(ax):
    """统一坐标轴风格。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 为了配合箭头，隐藏左、下边框
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=0.75,
        alpha=0.42,
        color=COLOR_GRID
    )
    ax.set_axisbelow(True)

    ax.tick_params(
        axis="both",
        direction="in",
        width=1.15,
        length=5.2,
        colors="#222222",
        labelsize=18
    )

    add_axis_arrows(ax)


def save_fig(fig, name):
    png_path = OUT_DIR / f"{name}.png"

    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PNG: {png_path}")


# =========================================================
# 5. 主图：B3-B12 多指标性能轮廓图
# =========================================================
def plot_multimetric_profile_largefont(df: pd.DataFrame):
    metrics = ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Pre", "M-Rec"]
    selected = [f"B{i}" for i in range(3, 13)]

    x = np.arange(len(metrics))

    # 图尺寸加大，适配大字号
    fig, ax = plt.subplots(figsize=(13.6, 7.4))

    for method in selected:
        row = df[df["Method"].astype(str) == method].iloc[0]
        y = row[metrics].values.astype(float)

        color, marker, linestyle, linewidth = PALETTE[method]

        is_b12 = method == "B12"
        is_b11 = method == "B11"

        ax.plot(
            x,
            y,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            markersize=16 if is_b12 else 10.5 if is_b11 else 8.2,
            markeredgewidth=1.2,
            alpha=1.0 if method in ["B11", "B12"] else 0.84,
            label=method,
            zorder=15 if is_b12 else 12 if is_b11 else 6,
        )

    # 横纵坐标
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=0, fontstyle="normal", fontsize=18)
    ax.set_ylabel("Score", fontsize=21, labelpad=14)

    # 坐标范围
    ax.set_ylim(0.50, 1.03)
    ax.set_xlim(-0.28, len(metrics) - 0.72)

    # y 轴数字字号
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set_yticklabels([f"{v:.1f}" for v in np.arange(0.5, 1.01, 0.1)], fontsize=18)

    style_axis(ax)

    # 图例：字号放大，位置下移，避免遮挡
    legend = ax.legend(
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.31),
        frameon=False,
        columnspacing=1.45,
        handlelength=2.4,
        handletextpad=0.55,
        fontsize=16,
    )

    for text in legend.get_texts():
        text.set_fontstyle("normal")

    # 留出底部图例空间
    fig.subplots_adjust(left=0.085, right=0.985, top=0.965, bottom=0.27)

    save_fig(fig, "Fig7A_overall_multimetric_profile_largefont")


# =========================================================
# 6. 运行
# =========================================================
if __name__ == "__main__":
    df = load_data()
    plot_multimetric_profile_largefont(df)
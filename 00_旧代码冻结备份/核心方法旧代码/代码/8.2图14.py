# -*- coding: utf-8 -*-
r"""
跨工况任务下双源/单源设置的分类性能与平滑性均值-标准差对比图
修改版：
1）颜色更接近原图
2）(a)(b) 小标题与 x 轴名称间距加大
3）只保存 PNG

输出：
C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文\11_第五章图像字号优化
\Fig_mean_std_dual_vs_single_largefont_v2.png
"""

from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =========================================================
# 0. 输出路径
# =========================================================
OUT_DIR = Path(
    r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文\11_第五章图像字号优化"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "Fig_mean_std_dual_vs_single_largefont_v2.png"
DPI = 900


# =========================================================
# 1. 数据
# =========================================================
SUMMARY_DATA = r"""Task,Setting,Train,Test,Method,Method_name,Acc,MacroF1,EF1,MF1,LF1,MPre,MRec,ME,ML,Rev,Jump,Smooth
D1,Dual-source,C1+C4,C6,B9,Relative-stage GRU,0.802631579,0.801260986,0.827586207,0.696969697,0.879227053,1.000000000,0.534883721,0.271317829,0.193798450,1,0,0.029992066
D1,Dual-source,C1+C4,C6,B10,Relative-stage TCN-GRU,0.792763158,0.790391983,0.827586207,0.676923077,0.866666667,1.000000000,0.511627907,0.271317829,0.217054264,1,0,0.039126985
D1,Dual-source,C1+C4,C6,B11,Multi-task TCN-GRU,0.986842105,0.987102093,0.982456140,0.984375000,0.994475138,0.992125984,0.976744186,0.023255814,0.000000000,0,0,0.024748008
D1,Dual-source,C1+C4,C6,B12,DC-PSR,0.983552632,0.983963259,0.982456140,0.980544747,0.988888889,0.984375000,0.976744186,0.023255814,0.000000000,0,0,0.019237527
D2,Dual-source,C4+C6,C1,B9,Relative-stage GRU,0.881578947,0.884037158,0.894736842,0.840707965,0.916666667,0.826086957,0.855855856,0.000000000,0.144144144,0,0,0.014450183
D2,Dual-source,C4+C6,C1,B10,Relative-stage TCN-GRU,0.881578947,0.884037158,0.894736842,0.840707965,0.916666667,0.826086957,0.855855856,0.000000000,0.144144144,0,0,0.018610958
D2,Dual-source,C4+C6,C1,B11,Multi-task TCN-GRU,0.855263158,0.857480582,0.900523560,0.796296296,0.875621891,0.819047619,0.774774775,0.000000000,0.225225225,0,0,0.013625672
D2,Dual-source,C4+C6,C1,B12,DC-PSR,0.865131579,0.867399279,0.900523560,0.812785388,0.888888889,0.824074074,0.801801802,0.000000000,0.198198198,0,0,0.015188771
S1,Single-source,C4,C1,B9,Relative-stage GRU,0.884868421,0.886920851,0.907216495,0.841628959,0.911917098,0.845454545,0.837837838,0.009009009,0.153153153,1,0,0.023214139
S1,Single-source,C4,C1,B10,Relative-stage TCN-GRU,0.914473684,0.917674984,0.885416667,0.889830508,0.977777778,0.840000000,0.945945946,0.018018018,0.036036036,1,0,0.020789772
S1,Single-source,C4,C1,B11,Multi-task TCN-GRU,0.914473684,0.917200106,0.894736842,0.889830508,0.967032967,0.840000000,0.945945946,0.000000000,0.054054054,0,0,0.014353601
S1,Single-source,C4,C1,B12,DC-PSR,0.924342105,0.927084975,0.900523560,0.902953586,0.977777778,0.849206349,0.963963964,0.000000000,0.036036036,0,0,0.013449135
S2,Single-source,C6,C1,B9,Relative-stage GRU,0.796052632,0.801749144,0.759825328,0.683673469,0.961748634,0.788235294,0.603603604,0.333333333,0.063063063,2,0,0.041191164
S2,Single-source,C6,C1,B10,Relative-stage TCN-GRU,0.881578947,0.884647468,0.813559322,0.857142857,0.983240223,0.765957447,0.972972973,0.000000000,0.027027027,0,0,0.018481182
S2,Single-source,C6,C1,B11,Multi-task TCN-GRU,0.937500000,0.940565847,0.906250000,0.921161826,0.994285714,0.853846154,1.000000000,0.000000000,0.000000000,0,0,0.019588977
S2,Single-source,C6,C1,B12,DC-PSR,0.930921053,0.934080510,0.911917098,0.913580247,0.976744186,0.840909091,1.000000000,0.000000000,0.000000000,0,0,0.014427599
"""


# =========================================================
# 2. 全局风格
# =========================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

plt.rcParams["font.size"] = 18
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 21
plt.rcParams["xtick.labelsize"] = 17
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["legend.fontsize"] = 14


# =========================================================
# 3. 参数
# =========================================================
METHODS = ["B9", "B10", "B11", "B12"]

# 更接近你原图的颜色：蓝、红更饱和一些
COLOR_DUAL = "#5A78E6"
COLOR_SINGLE = "#EA605C"

COLOR_GRID = "#D9D9D9"
COLOR_BLACK = "#222222"

METRICS = [
    ("Acc", "Acc", "*"),
    ("MacroF1", "Macro-F1", "o"),
    ("MF1", "M-F1", "D"),
    ("MRec", "M-Rec", "s"),
]


# =========================================================
# 4. 工具函数
# =========================================================
def load_summary() -> pd.DataFrame:
    df = pd.read_csv(StringIO(SUMMARY_DATA.strip()))
    numeric_cols = [
        "Acc", "MacroF1", "EF1", "MF1", "LF1",
        "MPre", "MRec", "ME", "ML", "Rev", "Jump", "Smooth"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def average_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setting in ["Dual-source", "Single-source"]:
        sub = summary[summary["Setting"] == setting]
        for method in METHODS:
            g = sub[sub["Method"] == method]
            row = {"Setting": setting, "Method": method}
            for col in ["Acc", "MacroF1", "MF1", "MRec", "Smooth"]:
                row[f"{col}_mean"] = float(g[col].mean())
                row[f"{col}_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def add_axis_arrows(ax, x_pad=0.030, y_pad=0.040):
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.annotate(
        "",
        xy=(1 + x_pad, 0),
        xytext=(0, 0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.15, color=COLOR_BLACK),
        clip_on=False,
        zorder=20,
    )
    ax.annotate(
        "",
        xy=(0, 1 + y_pad),
        xytext=(0, 0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.15, color=COLOR_BLACK),
        clip_on=False,
        zorder=20,
    )


def style_axis(ax, grid_axis="x"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_BLACK)
    ax.spines["bottom"].set_color(COLOR_BLACK)
    ax.spines["left"].set_linewidth(1.05)
    ax.spines["bottom"].set_linewidth(1.05)

    ax.grid(
        True,
        axis=grid_axis,
        linestyle="--",
        linewidth=0.75,
        color=COLOR_GRID,
        alpha=0.60,
    )
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=17, width=1.05, length=5.0)
    add_axis_arrows(ax)


# =========================================================
# 5. 主绘图函数
# =========================================================
def plot_mean_std_dual_vs_single(summary: pd.DataFrame):
    avg = average_table(summary)

    y_base = np.arange(len(METHODS))[::-1]
    offsets = np.linspace(-0.18, 0.18, len(METRICS))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16.8, 6.0),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    # -----------------------------------------------------
    # (a) Classification metrics
    # -----------------------------------------------------
    ax = axes[0]

    for k, (metric, metric_name, marker) in enumerate(METRICS):
        for setting, color in [
            ("Dual-source", COLOR_DUAL),
            ("Single-source", COLOR_SINGLE),
        ]:
            sub = avg[avg["Setting"] == setting].set_index("Method").reindex(METHODS)

            y = y_base + offsets[k]
            x = sub[f"{metric}_mean"].values
            err = sub[f"{metric}_std"].values

            ax.errorbar(
                x, y, xerr=err,
                fmt=marker,
                markersize=8.6 if marker != "*" else 11.0,
                color=color,
                ecolor=color,
                elinewidth=1.35,
                capsize=4.0,
                capthick=1.20,
                linewidth=1.20,
                alpha=0.95,
                zorder=4
            )

    ax.set_yticks(y_base)
    ax.set_yticklabels(METHODS, fontsize=18)
    ax.set_xlim(0.65, 1.02)

    # 这里把 xlabel 和下面标题分开一点
    ax.set_xlabel("Mean ± std", fontsize=20, labelpad=8)
    ax.set_title(
        "(a) Classification metrics",
        loc="center",
        fontsize=18,
        fontweight="bold",
        y=-0.33
    )

    style_axis(ax, grid_axis="x")

    # -----------------------------------------------------
    # (b) Probability smoothness
    # -----------------------------------------------------
    ax = axes[1]

    for setting, color, marker in [
        ("Dual-source", COLOR_DUAL, "D"),
        ("Single-source", COLOR_SINGLE, "D"),
    ]:
        sub = avg[avg["Setting"] == setting].set_index("Method").reindex(METHODS)

        ax.errorbar(
            sub["Smooth_mean"].values,
            y_base,
            xerr=sub["Smooth_std"].values,
            fmt=marker,
            markersize=8.4,
            color=color,
            ecolor=color,
            elinewidth=1.35,
            capsize=4.0,
            capthick=1.20,
            linewidth=1.20,
            alpha=0.95,
            label=setting,
            zorder=4,
        )

    ax.set_yticks(y_base)
    ax.set_yticklabels(METHODS, fontsize=18)
    ax.set_xlabel("Smoothness mean ± std", fontsize=20, labelpad=8)
    ax.set_title(
        "(b) Probability smoothness",
        loc="center",
        fontsize=18,
        fontweight="bold",
        y=-0.33
    )

    style_axis(ax, grid_axis="x")

    # -----------------------------------------------------
    # 图例
    # -----------------------------------------------------
    metric_handles = [
        Line2D(
            [0], [0],
            marker=marker,
            linestyle="None",
            color=COLOR_BLACK,
            markerfacecolor=COLOR_BLACK,
            markersize=9.0 if marker != "*" else 11.5,
            label=metric_name,
        )
        for _, metric_name, marker in METRICS
    ]

    setting_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            color=COLOR_DUAL,
            markerfacecolor=COLOR_DUAL,
            markersize=9,
            label="Dual-source",
        ),
        Line2D(
            [0], [0],
            marker="s",
            linestyle="None",
            color=COLOR_SINGLE,
            markerfacecolor=COLOR_SINGLE,
            markersize=9,
            label="Single-source",
        ),
    ]

    leg1 = axes[1].legend(
        handles=metric_handles,
        loc="center right",
        bbox_to_anchor=(0.98, 0.38),
        frameon=True,
        fontsize=13,
        borderpad=0.35,
        labelspacing=0.35,
        handletextpad=0.45,
    )
    leg1.get_frame().set_linewidth(0.7)
    leg1.get_frame().set_edgecolor("#C8C8C8")

    leg2 = axes[1].legend(
        handles=setting_handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.06),
        frameon=True,
        fontsize=13,
        borderpad=0.35,
        labelspacing=0.40,
        handletextpad=0.50,
    )
    leg2.get_frame().set_linewidth(0.7)
    leg2.get_frame().set_edgecolor("#C8C8C8")

    axes[1].add_artist(leg1)

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.93,
        bottom=0.27,   # 底部留白加大，给 (a)(b) 更多空间
        wspace=0.34,
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
    summary = load_summary()
    plot_mean_std_dual_vs_single(summary)
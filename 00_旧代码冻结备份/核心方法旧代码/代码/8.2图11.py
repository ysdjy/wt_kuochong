# -*- coding: utf-8 -*-
r"""
跨工况雷达图（放大字号 + 优化配色）

来源：
原始代码对应 7.8.1跨工况可视化优化.py 中的 plot_fig18_radar_by_task(summary)

输出：
C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文\11_第五章图像字号优化
\Fig18_cross_condition_radar_profile_largefont.png
"""

from __future__ import annotations

from pathlib import Path
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =========================================================
# 1. 输出路径
# =========================================================
OUT_DIR = Path(
    r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文\11_第五章图像字号优化"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 900


# =========================================================
# 2. 全局风格
# =========================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

# 整体字号显著放大，确保除圈线刻度外的其余组件清晰可见
plt.rcParams["font.size"] = 17
plt.rcParams["axes.titlesize"] = 21
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 16


# =========================================================
# 3. 颜色（更偏高级期刊风格，不再用大红）
# =========================================================
COLOR_GRID = "#D7D7D7"

# 方法配色：柔和但区分明显
COLOR_B9 = "#6C7BD0"   # muted blue-violet
COLOR_B10 = "#D39B2C"  # amber / ochre
COLOR_B11 = "#3E3E3E"  # charcoal
COLOR_B12 = "#2A9D8F"  # teal-green（替代红色）

METHODS = ["B9", "B10", "B11", "B12"]
TASKS = ["D1", "D2", "S1", "S2"]


# =========================================================
# 4. 数据
# =========================================================
SUMMARY_DATA = r"""Task,Setting,Train,Test,Method,Method_name,Acc,MacroF1,EF1,MF1,LF1,MPre,MRec,ME,ML,Rev,Jump,Smooth
D1,Dual-source,C1+C4,C6,B9,Relative-stage GRU,0.802631579,0.801260986,0.827586207,0.696969697,0.879227053,1.000000000,0.534883721,0.271317829,0.193798450,1,0,0.029992066
D1,Dual-source,C1+C4,C6,B10,Relative-stage TCN-GRU,0.792763158,0.790391983,0.827586207,0.676923077,0.866666667,1.000000000,0.511627907,0.271317829,0.217054264,1,0,0.039126985
D1,Dual-source,C1+C4,C6,B11,Multi-task TCN-GRU,0.986842105,0.987102093,0.982456140,0.984375000,0.994475138,0.992125984,0.976744186,0.023255814,0.000000000,0,0,0.024748008
D1,Dual-source,C1+C4,C6,B12,FGDS-PSI,0.983552632,0.983963259,0.982456140,0.980544747,0.988888889,0.984375000,0.976744186,0.023255814,0.000000000,0,0,0.019237527
D2,Dual-source,C4+C6,C1,B9,Relative-stage GRU,0.881578947,0.884037158,0.894736842,0.840707965,0.916666667,0.826086957,0.855855856,0.000000000,0.144144144,0,0,0.014450183
D2,Dual-source,C4+C6,C1,B10,Relative-stage TCN-GRU,0.881578947,0.884037158,0.894736842,0.840707965,0.916666667,0.826086957,0.855855856,0.000000000,0.144144144,0,0,0.018610958
D2,Dual-source,C4+C6,C1,B11,Multi-task TCN-GRU,0.855263158,0.857480582,0.900523560,0.796296296,0.875621891,0.819047619,0.774774775,0.000000000,0.225225225,0,0,0.013625672
D2,Dual-source,C4+C6,C1,B12,FGDS-PSI,0.865131579,0.867399279,0.900523560,0.812785388,0.888888889,0.824074074,0.801801802,0.000000000,0.198198198,0,0,0.015188771
S1,Single-source,C4,C1,B9,Relative-stage GRU,0.884868421,0.886920851,0.907216495,0.841628959,0.911917098,0.845454545,0.837837838,0.009009009,0.153153153,1,0,0.023214139
S1,Single-source,C4,C1,B10,Relative-stage TCN-GRU,0.914473684,0.917674984,0.885416667,0.889830508,0.977777778,0.840000000,0.945945946,0.018018018,0.036036036,1,0,0.020789772
S1,Single-source,C4,C1,B11,Multi-task TCN-GRU,0.914473684,0.917200106,0.894736842,0.889830508,0.967032967,0.840000000,0.945945946,0.000000000,0.054054054,0,0,0.014353601
S1,Single-source,C4,C1,B12,FGDS-PSI,0.924342105,0.927084975,0.900523560,0.902953586,0.977777778,0.849206349,0.963963964,0.000000000,0.036036036,0,0,0.013449135
S2,Single-source,C6,C1,B9,Relative-stage GRU,0.796052632,0.801749144,0.759825328,0.683673469,0.961748634,0.788235294,0.603603604,0.333333333,0.063063063,2,0,0.041191164
S2,Single-source,C6,C1,B10,Relative-stage TCN-GRU,0.881578947,0.884647468,0.813559322,0.857142857,0.983240223,0.765957447,0.972972973,0.000000000,0.027027027,0,0,0.018481182
S2,Single-source,C6,C1,B11,Multi-task TCN-GRU,0.937500000,0.940565847,0.906250000,0.921161826,0.994285714,0.853846154,1.000000000,0.000000000,0.000000000,0,0,0.019588977
S2,Single-source,C6,C1,B12,FGDS-PSI,0.930921053,0.934080510,0.911917098,0.913580247,0.976744186,0.840909091,1.000000000,0.000000000,0.000000000,0,0,0.014427599
"""


def load_summary():
    df = pd.read_csv(StringIO(SUMMARY_DATA.strip()))
    for col in ["Acc", "MacroF1", "MF1", "MRec", "ME", "ML", "Smooth"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# =========================================================
# 5. 绘图函数（基于原 plot_fig18_radar_by_task 改写）
# =========================================================
def plot_fig18_radar_by_task(summary: pd.DataFrame):
    metrics = ["Acc", "MacroF1", "MF1", "MRec", "1-ME", "1-ML", "1-Smooth"]
    labels = ["Acc", "Macro-F1", "M-F1", "M-Rec", "1-M→E", "1-M→L", "1-Smooth"]

    tmp = summary.copy()
    tmp["1-ME"] = 1.0 - tmp["ME"]
    tmp["1-ML"] = 1.0 - tmp["ML"]
    tmp["1-Smooth"] = 1.0 - tmp["Smooth"]

    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles = np.r_[angles, angles[0]]

    styles = {
        "B9": dict(color=COLOR_B9, linestyle="--", marker="o", linewidth=1.8, markersize=5.2),
        "B10": dict(color=COLOR_B10, linestyle="-.", marker="s", linewidth=1.8, markersize=5.2),
        "B11": dict(color=COLOR_B11, linestyle=":", marker="D", linewidth=1.9, markersize=5.2),
        "B12": dict(color=COLOR_B12, linestyle="-", marker="*", linewidth=2.4, markersize=8.8),
    }

    fig, axes = plt.subplots(
        1, 4,
        subplot_kw={"polar": True},
        figsize=(18.2, 5.6)
    )

    panel_titles = ["(a) D1", "(b) D2", "(c) S1", "(d) S2"]

    for ax, task, panel in zip(axes, TASKS, panel_titles):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.grid(False)
        ax.spines["polar"].set_visible(False)

        r_ticks = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

        # 圆形网格
        for r in r_ticks:
            ax.plot(angles, [r] * len(angles), color="#BEBEBE", linewidth=0.72, zorder=0)

        # 辐射线
        for a in angles[:-1]:
            ax.plot([a, a], [0.72, 1.0], color=COLOR_GRID, linewidth=0.62, zorder=0)

        sub = tmp[tmp["Task"] == task].set_index("Method").reindex(METHODS)

        for method in METHODS:
            vals = sub.loc[method, metrics].values.astype(float)
            vals = np.clip(vals, 0.72, 1.0)
            vals = np.r_[vals, vals[0]]
            st = styles[method]

            ax.plot(angles, vals, label=method, **st)
            ax.fill(
                angles, vals,
                color=st["color"],
                alpha=0.045 if method != "B12" else 0.085
            )

        ax.set_xticks(angles[:-1])
        # 变更为同步放大后的轴指标字号
        ax.set_xticklabels(labels, fontsize=14)
        ax.set_ylim(0.72, 1.0)
        ax.set_yticks(r_ticks)
        # 严格控制：0.75/0.80/0.85... 这一行圈线数值维持原版较小字号（不随rcParams进行放大）
        ax.set_yticklabels([f"{r:.2f}" for r in r_ticks], fontsize=10)
        ax.set_rlabel_position(90)
        # 变更为同步放大后的子图标题字号
        ax.set_title(panel, y=1.14, fontsize=19, fontweight="bold")

    handles = [
        Line2D(
            [0], [0],
            color=styles[m]["color"],
            linestyle=styles[m]["linestyle"],
            marker=styles[m]["marker"],
            linewidth=styles[m]["linewidth"],
            markersize=styles[m]["markersize"],
            label=m
        )
        for m in METHODS
    ]

    fig.legend(
        handles=handles,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
        fontsize=16,  # 下方图例字号等比例放大
        handlelength=2.2,
        columnspacing=1.6,
        handletextpad=0.55
    )

    fig.subplots_adjust(
        left=0.03,
        right=0.995,
        top=0.88,
        bottom=0.14,
        wspace=0.18
    )

    out_path = OUT_DIR / "Fig18_cross_condition_radar_profile_largefont.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {out_path}")


# =========================================================
# 6. 主程序
# =========================================================
def main():
    summary = load_summary()
    plot_fig18_radar_by_task(summary)


if __name__ == "__main__":
    main()
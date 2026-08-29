# -*- coding: utf-8 -*-
r"""
PHM + NASA 跨工况雷达图（两行八图，放大字号，仅输出 PNG）

说明：
1. 上排 4 个子图：PHM 数据（D1, D2, S1, S2）
2. 下排 4 个子图：NASA 数据（N1, N2, N3, N4）
3. 风格保持一致：配色、线型、填充、图例风格统一
4. 仅输出 PNG，不输出 PDF
5. 雷达图指标与原图一致：
   Acc, Macro-F1, M-F1, M-Rec, 1-M→E, 1-M→L, 1-Smooth
   （注：Rev 不进入雷达图，仍保留在原始表中）
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

OUT_PNG = OUT_DIR / "Fig18_cross_condition_radar_profile_PHM_NASA_final_largefont.png"
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

# 全局字号放大
plt.rcParams["font.size"] = 18
plt.rcParams["axes.titlesize"] = 22
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 17


# =========================================================
# 3. 配色与方法设置
# =========================================================
COLOR_GRID = "#D7D7D7"
COLOR_B9 = "#6C7BD0"   # muted blue-violet
COLOR_B10 = "#D39B2C"  # amber / ochre
COLOR_B11 = "#3E3E3E"  # charcoal
COLOR_B12 = "#2A9D8F"  # teal-green

METHODS = ["B9", "B10", "B11", "B12"]

STYLES = {
    "B9":  dict(color=COLOR_B9,  linestyle="--", marker="o", linewidth=1.9, markersize=5.6),
    "B10": dict(color=COLOR_B10, linestyle="-.", marker="s", linewidth=1.9, markersize=5.6),
    "B11": dict(color=COLOR_B11, linestyle=":",  marker="D", linewidth=2.0, markersize=5.6),
    "B12": dict(color=COLOR_B12, linestyle="-",  marker="*", linewidth=2.5, markersize=9.2),
}


# =========================================================
# 4. 数据
# =========================================================
PHM_DATA = r"""Task,Method,Acc,MacroF1,MF1,MRec,ME,ML,Rev,Smooth
D1,B9,0.8026,0.8013,0.6970,0.5349,0.2713,0.1938,1,0.0300
D1,B10,0.7928,0.7904,0.6769,0.5116,0.2713,0.2171,1,0.0391
D1,B11,0.9868,0.9871,0.9844,0.9767,0.0233,0.0000,0,0.0247
D1,B12,0.9836,0.9840,0.9805,0.9767,0.0233,0.0000,0,0.0192
D2,B9,0.8816,0.8840,0.8407,0.8559,0.0000,0.1441,0,0.0152
D2,B10,0.8816,0.8840,0.8407,0.8559,0.0000,0.1441,0,0.0186
D2,B11,0.8553,0.8575,0.7963,0.7748,0.0000,0.2252,0,0.0136
D2,B12,0.8651,0.8674,0.8128,0.8018,0.0000,0.1982,0,0.0145
S1,B9,0.8849,0.8869,0.8416,0.8378,0.0090,0.1532,1,0.0232
S1,B10,0.9145,0.9177,0.8898,0.9459,0.0180,0.0360,1,0.0208
S1,B11,0.9145,0.9172,0.8898,0.9459,0.0000,0.0541,0,0.0144
S1,B12,0.9243,0.9271,0.9030,0.9640,0.0000,0.0360,0,0.0134
S2,B9,0.7961,0.8017,0.6837,0.6036,0.3333,0.0631,2,0.0412
S2,B10,0.8816,0.8846,0.8571,0.9730,0.0000,0.0270,0,0.0185
S2,B11,0.9375,0.9406,0.9212,0.9876,0.0000,0.0000,0,0.0196
S2,B12,0.9309,0.9341,0.9136,1.0000,0.0000,0.0000,0,0.0144
"""

NASA_DATA = r"""Task,Method,Acc,MacroF1,MF1,MRec,ME,ML,Rev,Smooth
N1,B9,0.5385,0.5713,0.4545,0.3571,0.0714,0.5714,1,0.1462
N1,B10,0.5000,0.4179,0.6047,0.6842,0.1053,0.2632,1,0.1834
N1,B11,0.7059,0.7019,0.7300,0.7895,0.1053,0.1053,2,0.3315
N1,B12,0.7059,0.6873,0.7619,0.8421,0.0526,0.0526,0,0.0563
N2,B9,0.4074,0.3638,0.2222,0.5000,0.5000,0.0000,1,0.0737
N2,B10,0.5161,0.3761,0.0000,0.0000,0.5000,0.5000,1,0.1451
N2,B11,0.7037,0.6455,0.5000,0.5000,0.5000,0.0000,1,0.1781
N2,B12,0.7163,0.6895,0.6270,0.6305,0.5000,0.0000,0,0.0562
N3,B9,0.4848,0.4675,0.4286,0.3750,0.1250,0.5000,2,0.0797
N3,B10,0.4242,0.4334,0.5714,0.7500,0.1250,0.2222,4,0.1302
N3,B11,0.6676,0.6952,0.5167,0.6556,0.1111,0.3333,1,0.1561
N3,B12,0.7946,0.7720,0.7800,0.7667,0.1002,0.1250,0,0.0116
N4,B9,0.4706,0.4180,0.1429,0.0769,0.8462,0.0769,1,0.2211
N4,B10,0.4706,0.4651,0.3810,0.3077,0.6923,0.0000,0,0.0670
N4,B11,0.6667,0.5810,0.6897,0.8151,0.1297,0.0000,1,0.0805
N4,B12,0.6867,0.6956,0.7667,0.9091,0.0909,0.0000,0,0.0582
"""


# =========================================================
# 5. 数据读取
# =========================================================
def load_table(data_str: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(data_str.strip()))
    num_cols = ["Acc", "MacroF1", "MF1", "MRec", "ME", "ML", "Smooth"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# =========================================================
# 6. 雷达图绘制函数
# =========================================================
def draw_radar(ax, df_sub: pd.DataFrame, panel_title: str, radar_labels, rmin, rmax, r_ticks):
    """
    ax         : 极坐标子图
    df_sub     : 当前任务数据（4行，对应 B9-B12）
    panel_title: 子图标题
    radar_labels: 维度标签
    rmin/rmax  : 该子图径向范围
    r_ticks    : 该子图径向刻度
    """
    metrics = ["Acc", "MacroF1", "MF1", "MRec", "1-ME", "1-ML", "1-Smooth"]

    # 构造雷达用指标
    tmp = df_sub.copy()
    tmp["1-ME"] = 1.0 - tmp["ME"]
    tmp["1-ML"] = 1.0 - tmp["ML"]
    tmp["1-Smooth"] = 1.0 - tmp["Smooth"]

    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles_closed = np.r_[angles, angles[0]]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    # 圆形网格
    for r in r_ticks:
        ax.plot(angles_closed, [r] * len(angles_closed), color="#BEBEBE", linewidth=0.72, zorder=0)

    # 辐射线
    for a in angles:
        ax.plot([a, a], [rmin, rmax], color=COLOR_GRID, linewidth=0.62, zorder=0)

    # 按方法顺序绘图
    tmp = tmp.set_index("Method").reindex(METHODS)

    for method in METHODS:
        vals = tmp.loc[method, metrics].values.astype(float)
        vals = np.clip(vals, rmin, rmax)
        vals_closed = np.r_[vals, vals[0]]
        st = STYLES[method]

        ax.plot(angles_closed, vals_closed, label=method, **st)
        ax.fill(
            angles_closed, vals_closed,
            color=st["color"],
            alpha=0.045 if method != "B12" else 0.085
        )

    ax.set_xticks(angles)
    ax.set_xticklabels(radar_labels, fontsize=14)
    ax.set_ylim(rmin, rmax)
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([f"{x:.2f}" for x in r_ticks], fontsize=10)
    ax.set_rlabel_position(90)
    ax.set_title(panel_title, y=1.14, fontsize=21, fontweight="bold")


# =========================================================
# 7. 主绘图函数
# =========================================================
def plot_phm_nasa_radar():
    phm = load_table(PHM_DATA)
    nasa = load_table(NASA_DATA)

    # 标签与任务
    radar_labels = ["Acc", "Macro-F1", "M-F1", "M-Rec", "1-M→E", "1-M→L", "1-Smooth"]
    phm_tasks = ["D1", "D2", "S1", "S2"]
    nasa_tasks = ["N1", "N2", "N3", "N4"]

    panel_titles_top = ["(a) D1", "(b) D2", "(c) S1", "(d) S2"]
    panel_titles_bottom = ["(e) N1", "(f) N2", "(g) N3", "(h) N4"]

    # 创建 2x4 子图
    fig, axes = plt.subplots(
        2, 4,
        subplot_kw=dict(polar=True),
        figsize=(20.5, 11.2)
    )

    # -------------------------------
    # 上排：PHM
    # -------------------------------
    # PHM 指标整体较高，维持原风格的高区间展示
    phm_rmin = 0.72
    phm_rmax = 1.00
    phm_r_ticks = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

    for j, (task, title) in enumerate(zip(phm_tasks, panel_titles_top)):
        ax = axes[0, j]
        df_sub = phm[phm["Task"] == task]
        draw_radar(
            ax=ax,
            df_sub=df_sub,
            panel_title=title,
            radar_labels=radar_labels,
            rmin=phm_rmin,
            rmax=phm_rmax,
            r_ticks=phm_r_ticks
        )

    # -------------------------------
    # 下排：NASA
    # -------------------------------
    # NASA 指标明显更低，因此单独放宽显示范围，保证可读性
    nasa_rmin = 0.00
    nasa_rmax = 1.00
    nasa_r_ticks = [0.20, 0.40, 0.60, 0.80, 1.00]

    for j, (task, title) in enumerate(zip(nasa_tasks, panel_titles_bottom)):
        ax = axes[1, j]
        df_sub = nasa[nasa["Task"] == task]
        draw_radar(
            ax=ax,
            df_sub=df_sub,
            panel_title=title,
            radar_labels=radar_labels,
            rmin=nasa_rmin,
            rmax=nasa_rmax,
            r_ticks=nasa_r_ticks
        )

    # 图例
    legend_handles = [
        Line2D(
            [0], [0],
            color=STYLES[m]["color"],
            linestyle=STYLES[m]["linestyle"],
            marker=STYLES[m]["marker"],
            linewidth=STYLES[m]["linewidth"],
            markersize=STYLES[m]["markersize"],
            label=m
        )
        for m in METHODS
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
        fontsize=17,
        handlelength=2.4,
        columnspacing=2.0,
        handletextpad=0.7
    )

    fig.subplots_adjust(
        left=0.03,
        right=0.995,
        top=0.93,
        bottom=0.10,
        wspace=0.20,
        hspace=0.35
    )

    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print(f"Saved PNG: {OUT_PNG}")


# =========================================================
# 8. 主程序
# =========================================================
if __name__ == "__main__":
    plot_phm_nasa_radar()
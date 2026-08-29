# -*- coding: utf-8 -*-
r"""
跨工况任务下 B12/DC-PSR 混淆矩阵对比图
只输出 PNG，保存到：
C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文\11_第五章图像字号优化
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

OUT_FILE = OUT_DIR / "Fig_cross_condition_confusion_matrix_B12_largefont.png"
DPI = 900


# =========================================================
# 1. 数据：按你上传图中的数字填写
# =========================================================
CONFUSION_DATA = """Task,Title,True_stage,Pred_stage,count,row_norm
D1,(a) D1: C1+C4→C6,E,E,84,1.000
D1,(a) D1: C1+C4→C6,E,M,0,0.000
D1,(a) D1: C1+C4→C6,E,L,0,0.000
D1,(a) D1: C1+C4→C6,M,E,3,0.023
D1,(a) D1: C1+C4→C6,M,M,126,0.977
D1,(a) D1: C1+C4→C6,M,L,0,0.000
D1,(a) D1: C1+C4→C6,L,E,0,0.000
D1,(a) D1: C1+C4→C6,L,M,2,0.022
D1,(a) D1: C1+C4→C6,L,L,89,0.978
D2,(b) D2: C4+C6→C1,E,E,86,0.819
D2,(b) D2: C4+C6→C1,E,M,19,0.181
D2,(b) D2: C4+C6→C1,E,L,0,0.000
D2,(b) D2: C4+C6→C1,M,E,0,0.000
D2,(b) D2: C4+C6→C1,M,M,89,0.802
D2,(b) D2: C4+C6→C1,M,L,22,0.198
D2,(b) D2: C4+C6→C1,L,E,0,0.000
D2,(b) D2: C4+C6→C1,L,M,0,0.000
D2,(b) D2: C4+C6→C1,L,L,88,1.000
S1,(c) S1: C4→C1,E,E,86,0.819
S1,(c) S1: C4→C1,E,M,19,0.181
S1,(c) S1: C4→C1,E,L,0,0.000
S1,(c) S1: C4→C1,M,E,0,0.000
S1,(c) S1: C4→C1,M,M,107,0.964
S1,(c) S1: C4→C1,M,L,4,0.036
S1,(c) S1: C4→C1,L,E,0,0.000
S1,(c) S1: C4→C1,L,M,0,0.000
S1,(c) S1: C4→C1,L,L,88,1.000
S2,(d) S2: C6→C1,E,E,88,0.838
S2,(d) S2: C6→C1,E,M,17,0.162
S2,(d) S2: C6→C1,E,L,0,0.000
S2,(d) S2: C6→C1,M,E,0,0.000
S2,(d) S2: C6→C1,M,M,111,1.000
S2,(d) S2: C6→C1,M,L,0,0.000
S2,(d) S2: C6→C1,L,E,0,0.000
S2,(d) S2: C6→C1,L,M,4,0.045
S2,(d) S2: C6→C1,L,L,84,0.955
"""


def load_data() -> pd.DataFrame:
    df = pd.read_csv(StringIO(CONFUSION_DATA.strip()))
    df["count"] = pd.to_numeric(df["count"], errors="coerce").astype(int)
    df["row_norm"] = pd.to_numeric(df["row_norm"], errors="coerce")
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

# 字号放大
plt.rcParams["font.size"] = 16
plt.rcParams["axes.labelsize"] = 17
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["legend.fontsize"] = 14


# =========================================================
# 3. 配色：保持你原图的白—蓝—红风格
# =========================================================
CMAP_ORIGINAL_LIKE = LinearSegmentedColormap.from_list(
    "original_like_white_blue_red",
    [
        (0.00, "#FFFFFF"),  # 0: white
        (0.12, "#EEF5F7"),  # very light cyan
        (0.35, "#C9E2EC"),  # light blue
        (0.65, "#5275B7"),  # blue
        (0.84, "#51419A"),  # blue-purple
        (0.93, "#7A2C5F"),  # purple-red
        (1.00, "#9B1B1B"),  # deep red
    ],
    N=256,
)


# =========================================================
# 4. 绘图函数
# =========================================================
STAGES = ["E", "M", "L"]
TASK_ORDER = ["D1", "D2", "S1", "S2"]


def matrix_for_task(df: pd.DataFrame, task: str):
    sub = df[df["Task"] == task].copy()

    mat = np.zeros((3, 3), dtype=float)
    cnt = np.zeros((3, 3), dtype=int)

    for i, true_stage in enumerate(STAGES):
        for j, pred_stage in enumerate(STAGES):
            row = sub[
                (sub["True_stage"] == true_stage)
                & (sub["Pred_stage"] == pred_stage)
                ]
            if not row.empty:
                mat[i, j] = float(row["row_norm"].iloc[0])
                cnt[i, j] = int(row["count"].iloc[0])

    title = sub["Title"].iloc[0]
    return mat, cnt, title


def draw_confusion_matrix(ax, mat, cnt, title, show_ylabel=False):
    im = ax.imshow(mat, cmap=CMAP_ORIGINAL_LIKE, vmin=0.0, vmax=1.0)

    ax.set_title(title, loc="left", fontsize=18, fontweight="bold", pad=9)

    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))

    ax.set_xticklabels(STAGES, fontsize=15)
    ax.set_yticklabels(STAGES if show_ylabel else ["", "", ""], fontsize=15)

    ax.set_xlabel("Predicted stage", fontsize=16, labelpad=8)
    if show_ylabel:
        ax.set_ylabel("True stage", fontsize=16, labelpad=10)

    # 单元格白色分割线
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.45)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 外框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.15)
        spine.set_color("#333333")

    # 数字标注：count + row_norm
    for i in range(3):
        for j in range(3):
            value = mat[i, j]
            count = cnt[i, j]

            text_color = "white" if value >= 0.55 else "#222222"

            ax.text(
                j,
                i - 0.12,
                f"{count}",
                ha="center",
                va="center",
                fontsize=15,
                color=text_color,
                )
            ax.text(
                j,
                i + 0.22,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=14,
                color=text_color,
                )

    ax.tick_params(axis="both", length=0)

    return im


def plot_confusion_panel():
    df = load_data()

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(17.4, 4.55),
        constrained_layout=False,
    )

    last_im = None

    for idx, task in enumerate(TASK_ORDER):
        mat, cnt, title = matrix_for_task(df, task)
        last_im = draw_confusion_matrix(
            axes[idx],
            mat,
            cnt,
            title=title,
            show_ylabel=(idx == 0),
        )

    # 右侧 colorbar
    cax = fig.add_axes([0.925, 0.205, 0.014, 0.650])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("Row-normalized value", fontsize=15.5, labelpad=11)
    cb.ax.tick_params(labelsize=13.5, width=0.9, length=4.0)

    fig.subplots_adjust(
        left=0.055,
        right=0.905,
        bottom=0.185,
        top=0.850,
        wspace=0.090,
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
# 5. 运行
# =========================================================
if __name__ == "__main__":
    plot_confusion_panel()
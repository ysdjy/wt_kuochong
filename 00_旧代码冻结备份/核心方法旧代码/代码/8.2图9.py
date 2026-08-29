# -*- coding: utf-8 -*-
"""
Redraw representative confusion matrix panel: B1, B5, B8, B12.

只按原图风格重画：
1. 保持原图 B1/B5/B8/B12 顺序；
2. 保持原图数值；
3. 保持原图 E/M/L 标签；
4. 保持原图白-蓝-红配色风格；
5. 仅放大字体；
6. 仅输出 PNG；
7. 保存到“小论文/11_第五章图像字号优化”。

Output:
C:\\Users\\wangting\\Desktop\\博士开题\\公开数据\\1PHM\\PHM实验\\小论文
\\11_第五章图像字号优化\\figures_confusion_matrix
\\Fig_confusion_matrix_B1_B5_B8_B12_largefont.png
"""

from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =========================================================
# 0. Output path
# =========================================================
ROOT = Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文")
OUT_DIR = ROOT / "11_第五章图像字号优化" / "figures_confusion_matrix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 900


# =========================================================
# 1. Data
# 完全按你上传图中的数值
# =========================================================
CONFUSION_DATA = """Method,True_stage,Pred_stage,count,row_norm
B1,E,E,25,0.30
B1,E,M,59,0.70
B1,E,L,0,0.00
B1,M,E,0,0.00
B1,M,M,80,0.62
B1,M,L,49,0.38
B1,L,E,0,0.00
B1,L,M,0,0.00
B1,L,L,91,1.00
B5,E,E,83,0.99
B5,E,M,1,0.01
B5,E,L,0,0.00
B5,M,E,5,0.04
B5,M,M,123,0.95
B5,M,L,1,0.01
B5,L,E,0,0.00
B5,L,M,0,0.00
B5,L,L,91,1.00
B8,E,E,84,1.00
B8,E,M,0,0.00
B8,E,L,0,0.00
B8,M,E,11,0.09
B8,M,M,115,0.89
B8,M,L,3,0.02
B8,L,E,0,0.00
B8,L,M,0,0.00
B8,L,L,91,1.00
B12,E,E,84,1.00
B12,E,M,0,0.00
B12,E,L,0,0.00
B12,M,E,3,0.02
B12,M,M,126,0.98
B12,M,L,0,0.00
B12,L,E,0,0.00
B12,L,M,1,0.01
B12,L,L,90,0.99
"""


def load_confusion_data() -> pd.DataFrame:
    df = pd.read_csv(StringIO(CONFUSION_DATA.strip()))
    df["count"] = pd.to_numeric(df["count"], errors="coerce").astype(int)
    df["row_norm"] = pd.to_numeric(df["row_norm"], errors="coerce")
    return df


# =========================================================
# 2. Style
# =========================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

# 字号放大
plt.rcParams["font.size"] = 15
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 17
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14


# =========================================================
# 3. Colormap
# 贴近你原图：0 为白色，低-中值为浅蓝/蓝色，高值为深红
# =========================================================
CMAP_ORIGINAL_LIKE = LinearSegmentedColormap.from_list(
    "original_like_white_blue_red",
    [
        (0.00, "#FFFFFF"),  # 0: white
        (0.12, "#E9F3F5"),  # very light cyan
        (0.35, "#B7DCE5"),  # light blue
        (0.60, "#4D79B8"),  # blue
        (0.78, "#6B4B7E"),  # blue-purple
        (1.00, "#9B1B1B"),  # deep red
    ],
    N=256,
)


# =========================================================
# 4. Plot helpers
# =========================================================
STAGES = ["E", "M", "L"]
METHODS = ["B1", "B5", "B8", "B12"]
PANEL_LABELS = ["(a) B1", "(b) B5", "(c) B8", "(d) B12"]


def matrix_for_method(conf: pd.DataFrame, method: str):
    sub = conf[conf["Method"] == method].copy()

    mat = np.zeros((3, 3), dtype=float)
    cnt = np.zeros((3, 3), dtype=int)

    for i, true_stage in enumerate(STAGES):
        for j, pred_stage in enumerate(STAGES):
            row = sub[
                (sub["True_stage"] == true_stage) &
                (sub["Pred_stage"] == pred_stage)
                ]
            if not row.empty:
                mat[i, j] = float(row["row_norm"].iloc[0])
                cnt[i, j] = int(row["count"].iloc[0])

    return mat, cnt


def draw_confusion_matrix(ax, mat, cnt, title, show_ylabel=False):
    im = ax.imshow(mat, cmap=CMAP_ORIGINAL_LIKE, vmin=0.0, vmax=1.0)

    ax.set_title(title, loc="left", fontsize=17, fontweight="bold", pad=7)

    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))

    ax.set_xticklabels(STAGES, fontsize=14)
    ax.set_yticklabels(STAGES if show_ylabel else ["", "", ""], fontsize=14)

    ax.set_xlabel("Predicted stage", fontsize=15, labelpad=7)
    if show_ylabel:
        ax.set_ylabel("True stage", fontsize=15, labelpad=9)

    # 白色网格线，保持原图效果
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 外框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.05)
        spine.set_color("#333333")

    # 数字标注：数量 + 归一化值
    for i in range(3):
        for j in range(3):
            value = mat[i, j]
            count = cnt[i, j]

            # 按原图逻辑：深色区域白字，浅色区域黑字
            text_color = "white" if value >= 0.55 else "#222222"

            ax.text(
                j,
                i - 0.10,
                f"{count}",
                ha="center",
                va="center",
                fontsize=14,
                color=text_color,
                )
            ax.text(
                j,
                i + 0.20,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=13,
                color=text_color,
                )

    ax.tick_params(axis="both", length=0)

    return im


# =========================================================
# 5. Main figure
# =========================================================
def plot_confusion_panel():
    conf = load_confusion_data()

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(15.8, 4.2),
        constrained_layout=False,
    )

    last_im = None

    for idx, (method, label) in enumerate(zip(METHODS, PANEL_LABELS)):
        mat, cnt = matrix_for_method(conf, method)
        last_im = draw_confusion_matrix(
            axes[idx],
            mat,
            cnt,
            title=label,
            show_ylabel=(idx == 0),
        )

    # colorbar 保持右侧单独放置
    cax = fig.add_axes([0.925, 0.205, 0.014, 0.650])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("Row-normalized value", fontsize=15, labelpad=10)
    cb.ax.tick_params(labelsize=13, width=0.9, length=3.8)

    fig.subplots_adjust(
        left=0.055,
        right=0.905,
        bottom=0.180,
        top=0.850,
        wspace=0.070,
    )

    out_path = OUT_DIR / "Fig_confusion_matrix_B1_B5_B8_B12_largefont.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)

    print(f"Saved PNG: {out_path}")


# =========================================================
# 6. Run
# =========================================================
if __name__ == "__main__":
    plot_confusion_panel()
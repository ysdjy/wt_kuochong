# -*- coding: utf-8 -*-
"""
Visualization refinement for Section 5.1.

Generated figures:
1. Fig7A_all_methods_multimetric_profile.png
2. Fig8C_confusion_matrix_B1 / B5 / B8 / B12.png
3. Fig8A_middle_stage_bar_comparison.png
4. Fig9B_stage_consistency_combined.png

Output:
C:\\Users\\wangting\\Desktop\\博士开题\\公开数据\\1PHM\\PHM实验\\小论文\\5_figures_for_chapter5\\5_1_refined_visualization
"""

from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap


# =========================================================
# 0. Paths
# =========================================================
ROOT = Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文")
OUT_DIR = ROOT / "5_figures_for_chapter5" / "5_1_refined_visualization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 900


# =========================================================
# 1. Data
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
B12,FGDS-PSI,Relative-stage,Proposed,0.9868,0.9871,0.9825,0.9844,0.9945,0.9921,0.9767,0.0233,0.0000,0,0,0.0188
"""

CONFUSION_DATA = """Method,True_stage,Pred_stage,count,row_norm
B1,early,early,25,0.2976
B1,early,middle,59,0.7024
B1,early,late,0,0
B1,middle,early,0,0
B1,middle,middle,80,0.6202
B1,middle,late,49,0.3798
B1,late,early,0,0
B1,late,middle,0,0
B1,late,late,91,1
B5,early,early,83,0.9881
B5,early,middle,1,0.0119
B5,early,late,0,0
B5,middle,early,5,0.0388
B5,middle,middle,123,0.9535
B5,middle,late,1,0.0078
B5,late,early,0,0
B5,late,middle,0,0
B5,late,late,91,1
B8,early,early,84,1
B8,early,middle,0,0
B8,early,late,0,0
B8,middle,early,11,0.0853
B8,middle,middle,115,0.8915
B8,middle,late,3,0.0233
B8,late,early,0,0
B8,late,middle,0,0
B8,late,late,91,1
B12,early,early,84,1
B12,early,middle,0,0
B12,early,late,0,0
B12,middle,early,3,0.0233
B12,middle,middle,126,0.9767
B12,middle,late,0,0
B12,late,early,0,0
B12,late,middle,1,0.011
B12,late,late,90,0.989
"""


def load_data():
    df = pd.read_csv(StringIO(SUMMARY_DATA))
    conf = pd.read_csv(StringIO(CONFUSION_DATA))

    numeric_cols = [
        "Acc", "Macro-F1", "E-F1", "M-F1", "L-F1",
        "M-Pre", "M-Rec", "M→E", "M→L", "Rev", "Jump", "Smooth"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    conf["row_norm"] = pd.to_numeric(conf["row_norm"], errors="coerce")
    conf["count"] = pd.to_numeric(conf["count"], errors="coerce")

    df["order"] = df["Method"].str.extract(r"B(\d+)").astype(int)
    df = df.sort_values("order").reset_index(drop=True)

    return df, conf


# =========================================================
# 2. Global style
# =========================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 11.5
plt.rcParams["ytick.labelsize"] = 11.5
plt.rcParams["legend.fontsize"] = 10.5
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

COLOR_RED = "#E31A1C"
COLOR_BLUE = "#1F4DFF"
COLOR_BLACK = "#111111"
COLOR_GREEN = "#18A558"
COLOR_ORANGE = "#FF9F1C"
COLOR_PURPLE = "#8E44AD"
COLOR_B12 = "#B22222"

# brighter colors for 12 lines
METHOD_COLORS = {
    "B1": "#4D4D4D",
    "B2": "#999999",
    "B3": "#1F77B4",
    "B4": "#17BECF",
    "B5": "#2CA02C",
    "B6": "#FF7F0E",
    "B7": "#E377C2",
    "B8": "#0057FF",
    "B9": "#7B2CBF",
    "B10": "#F2B701",
    "B11": "#000000",
    "B12": COLOR_B12,
}

METHOD_MARKERS = {
    "B1": "o",
    "B2": "s",
    "B3": "^",
    "B4": "v",
    "B5": "D",
    "B6": "P",
    "B7": "X",
    "B8": "<",
    "B9": ">",
    "B10": "h",
    "B11": "p",
    "B12": "*",
}


def add_axis_arrows(ax, x_arrow=True, y_arrow=True):
    arrow_kw = dict(
        arrowstyle="-|>",
        lw=1.25,
        color="#222222",
        mutation_scale=12,
        shrinkA=0,
        shrinkB=0
    )

    if x_arrow:
        ax.annotate(
            "",
            xy=(1.018, 0),
            xytext=(0, 0),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=arrow_kw,
            clip_on=False,
            zorder=20
        )

    if y_arrow:
        ax.annotate(
            "",
            xy=(0, 1.025),
            xytext=(0, 0),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=arrow_kw,
            clip_on=False,
            zorder=20
        )


def add_right_y_arrow(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    arrow_kw = dict(
        arrowstyle="-|>",
        lw=1.25,
        color="#222222",
        mutation_scale=12,
        shrinkA=0,
        shrinkB=0
    )

    ax.annotate(
        "",
        xy=(1, 1.025),
        xytext=(1, 0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=arrow_kw,
        clip_on=False,
        zorder=20
    )

    ax.tick_params(axis="y", direction="in", width=1.0, length=4.2, colors="#222222")


def style_axis(ax, grid_axis="y", arrow=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if arrow:
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    else:
        ax.spines["left"].set_linewidth(1.1)
        ax.spines["bottom"].set_linewidth(1.1)
        ax.spines["left"].set_color("#222222")
        ax.spines["bottom"].set_color("#222222")

    ax.tick_params(axis="both", direction="in", width=1.0, length=4.2, colors="#222222")

    if grid_axis:
        ax.grid(axis=grid_axis, linestyle="--", linewidth=0.65, alpha=0.25, color="#808080")

    if arrow:
        add_axis_arrows(ax)


def save_fig(fig, name):
    png = OUT_DIR / f"{name}.png"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n{png}\n")


def add_bar_caps(ax, bars, color="#111111", cap_ratio=0.58, lw=1.25):
    """
    Add short horizontal cap lines on the top of bars, imitating error-bar caps.
    """
    for bar in bars:
        x = bar.get_x()
        w = bar.get_width()
        y = bar.get_height()
        cx = x + w / 2
        half = w * cap_ratio / 2
        ax.plot(
            [cx - half, cx + half],
            [y, y],
            color=color,
            linewidth=lw,
            solid_capstyle="butt",
            zorder=8
        )


# =========================================================
# 3. Figure 1: B1-B12 multi-metric performance profile
# =========================================================
def plot_all_methods_multimetric_profile(df):
    metrics = ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Pre", "M-Rec"]
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(14.2, 6.4))

    for _, row in df.iterrows():
        method = row["Method"]
        y = [row[m] for m in metrics]

        is_b12 = method == "B12"
        is_b11 = method == "B11"

        ax.plot(
            x, y,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markersize=14 if is_b12 else 10.2,
            markeredgewidth=1.15,
            linewidth=3.4 if is_b12 else 2.6 if is_b11 else 2.05,
            linestyle="-" if method in ["B11", "B12"] else "--",
            alpha=1.0 if method in ["B11", "B12"] else 0.86,
            label=method,
            zorder=12 if is_b12 else 10 if is_b11 else 5
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontstyle="normal", rotation=0)
    ax.set_ylabel("Score", fontstyle="normal")
    ax.set_ylim(-0.03, 1.06)
    ax.set_xlim(-0.25, len(metrics) - 0.75)

    style_axis(ax, grid_axis="y", arrow=True)

    legend = ax.legend(
        ncol=6,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.30),
        frameon=True,
        fancybox=False,
        edgecolor="#888888",
        columnspacing=1.35,
        handlelength=2.5
    )
    for text in legend.get_texts():
        text.set_fontstyle("normal")

    fig.tight_layout()
    save_fig(fig, "Fig7A_all_methods_multimetric_profile")


# =========================================================
# 4. Figure 2: four representative confusion matrices
# =========================================================
def get_confusion_matrix(conf, method):
    stage_order = ["early", "middle", "late"]
    mat = np.zeros((3, 3), dtype=float)

    sub = conf[conf["Method"] == method]
    for i, true_stage in enumerate(stage_order):
        for j, pred_stage in enumerate(stage_order):
            v = sub[
                (sub["True_stage"] == true_stage) &
                (sub["Pred_stage"] == pred_stage)
                ]["row_norm"]
            mat[i, j] = float(v.iloc[0]) if len(v) else np.nan

    return mat


def plot_single_confusion_matrix(conf, method, group_name):
    mat = get_confusion_matrix(conf, method)

    cmap = LinearSegmentedColormap.from_list(
        "cm_soft_blue",
        ["#FFFFFF", "#EAF3FB", "#9ECAE1", "#3182BD", "#08306B"]
    )

    fig, ax = plt.subplots(figsize=(4.65, 4.18))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    labels = ["E", "M", "L"]
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(labels, fontstyle="normal")
    ax.set_yticklabels(labels, fontstyle="normal")

    ax.set_xlabel("Predicted stage", fontstyle="normal")
    ax.set_ylabel("True stage", fontstyle="normal")

    title_color = COLOR_B12 if method == "B12" else "#111111"
    ax.set_title(
        f"{method} ({group_name})",
        fontweight="bold",
        color=title_color,
        fontstyle="normal",
        pad=8
    )

    for i in range(3):
        for j in range(3):
            val = mat[i, j]
            txt_color = "white" if val > 0.65 else "#111111"
            ax.text(
                j, i, f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=12,
                color=txt_color,
                fontstyle="normal"
            )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("#222222")

    ax.tick_params(axis="both", direction="in", width=1.0, length=3.5)

    if method == "B12":
        rect = Rectangle(
            (-0.5, -0.5), 3, 3,
            fill=False,
            edgecolor=COLOR_B12,
            linewidth=2.2
        )
        ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.035)
    cbar.set_label("Row-normalized value", fontstyle="normal")
    cbar.ax.tick_params(direction="in", width=0.8)

    fig.tight_layout()
    save_fig(fig, f"Fig8C_confusion_matrix_{method}")


def plot_four_confusion_matrices(conf):
    selected = [
        ("B1", "Fixed threshold"),
        ("B5", "Machine learning"),
        ("B8", "Deep learning"),
        ("B12", "Proposed")
    ]

    for method, group_name in selected:
        plot_single_confusion_matrix(conf, method, group_name)


# =========================================================
# 5. Figure 3: middle-stage bar comparison, two panels
# =========================================================
def plot_middle_stage_bar_comparison(df):
    methods = ["B1", "B5", "B8", "B12"]
    sub = df[df["Method"].isin(methods)].copy()
    sub["Method"] = pd.Categorical(sub["Method"], categories=methods, ordered=True)
    sub = sub.sort_values("Method")

    x = np.arange(len(methods))

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.9), gridspec_kw={"wspace": 0.18})

    # -----------------------------------------------------
    # (a) M-Pre / M-Rec / M-F1
    # -----------------------------------------------------
    ax = axes[0]
    width = 0.22

    b1 = ax.bar(
        x - width,
        sub["M-Pre"],
        width=width,
        label="M-Pre",
        facecolor="white",
        edgecolor=COLOR_RED,
        linewidth=1.9,
        hatch="////",
        zorder=4
    )
    b2 = ax.bar(
        x,
        sub["M-Rec"],
        width=width,
        label="M-Rec",
        facecolor="white",
        edgecolor=COLOR_BLUE,
        linewidth=1.9,
        hatch="\\\\\\\\",
        zorder=4
    )
    b3 = ax.bar(
        x + width,
        sub["M-F1"],
        width=width,
        label="M-F1",
        facecolor="white",
        edgecolor=COLOR_GREEN,
        linewidth=1.9,
        hatch="....",
        zorder=4
    )

    add_bar_caps(ax, b1, color=COLOR_RED)
    add_bar_caps(ax, b2, color=COLOR_BLUE)
    add_bar_caps(ax, b3, color=COLOR_GREEN)

    b12_idx = methods.index("B12")
    ax.axvspan(b12_idx - 0.48, b12_idx + 0.48, color=COLOR_B12, alpha=0.055, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontstyle="normal")
    ax.set_ylabel("Middle-stage score", fontstyle="normal")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.62, len(methods) - 0.38)
    ax.set_title("(a) Middle-stage recognition", fontweight="bold", fontstyle="normal")

    style_axis(ax, grid_axis="y", arrow=True)

    legend = ax.legend(
        ncol=3,
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="#888888",
        handlelength=1.9
    )
    for text in legend.get_texts():
        text.set_fontstyle("normal")

    # -----------------------------------------------------
    # (b) M→E / M→L
    # -----------------------------------------------------
    ax = axes[1]
    width2 = 0.28

    b4 = ax.bar(
        x - width2 / 2,
        sub["M→E"],
        width=width2,
        label="M→E",
        facecolor="white",
        edgecolor=COLOR_BLUE,
        linewidth=1.9,
        hatch="////",
        zorder=4
    )
    b5 = ax.bar(
        x + width2 / 2,
        sub["M→L"],
        width=width2,
        label="M→L",
        facecolor="white",
        edgecolor=COLOR_RED,
        linewidth=1.9,
        hatch="\\\\\\\\",
        zorder=4
    )

    add_bar_caps(ax, b4, color=COLOR_BLUE)
    add_bar_caps(ax, b5, color=COLOR_RED)

    ax.axvspan(b12_idx - 0.48, b12_idx + 0.48, color=COLOR_B12, alpha=0.055, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontstyle="normal")
    ax.set_ylabel("Misclassification rate", fontstyle="normal")
    ax.set_ylim(0, 0.43)
    ax.set_xlim(-0.62, len(methods) - 0.38)
    ax.set_title("(b) Middle-stage misclassification", fontweight="bold", fontstyle="normal")

    style_axis(ax, grid_axis="y", arrow=True)


    legend = ax.legend(
        ncol=2,
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="#888888",
        handlelength=1.9
    )
    for text in legend.get_texts():
        text.set_fontstyle("normal")

    # Annotate B12 zero M→L
    ax.text(
        b12_idx + width2 / 2,
        sub[sub["Method"] == "B12"]["M→L"].iloc[0] + 0.012,
        "0",
        ha="center",
        va="bottom",
        fontsize=11,
        color=COLOR_RED,
        fontweight="bold"
    )

    fig.tight_layout()
    save_fig(fig, "Fig8A_middle_stage_bar_comparison")


# =========================================================
# 6. Figure 4: Rev/Jump bars + Smooth line
# =========================================================
def plot_stage_consistency_combined(df):
    methods = df["Method"].tolist()
    x = np.arange(len(methods))
    width = 0.36

    fig, ax1 = plt.subplots(figsize=(13.5, 6.0))

    b1 = ax1.bar(
        x - width / 2,
        df["Rev"],
        width=width,
        label="Rev",
        facecolor="white",
        edgecolor=COLOR_BLUE,
        linewidth=1.8,
        hatch="////",
        zorder=4
    )
    b2 = ax1.bar(
        x + width / 2,
        df["Jump"],
        width=width,
        label="Jump",
        facecolor="white",
        edgecolor=COLOR_RED,
        linewidth=1.8,
        hatch="\\\\\\\\",
        zorder=4
    )

    add_bar_caps(ax1, b1, color=COLOR_BLUE)
    add_bar_caps(ax1, b2, color=COLOR_RED)

    ax1.set_ylabel("Transition count", fontstyle="normal")
    ax1.set_ylim(0, max(df["Rev"].max(), df["Jump"].max()) + 3.0)
    ax1.set_xlim(-0.65, len(methods) - 0.35)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontstyle="normal")

    style_axis(ax1, grid_axis="y", arrow=True)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        df["Smooth"],
        color=COLOR_BLACK,
        marker="o",
        markersize=7.2,
        linewidth=2.35,
        label="Smooth",
        zorder=6
    )

    b12_idx = methods.index("B12")
    ax2.scatter(
        b12_idx,
        df.loc[b12_idx, "Smooth"],
        s=220,
        marker="*",
        color=COLOR_B12,
        edgecolor="#111111",
        linewidth=0.9,
        zorder=8
    )

    ax1.axvspan(
        b12_idx - 0.48,
        b12_idx + 0.48,
        color=COLOR_B12,
        alpha=0.055,
        zorder=0
    )

    ax2.set_ylabel("Smoothness", fontstyle="normal")
    ax2.set_ylim(0, max(df["Smooth"].max() * 1.20, 0.18))
    add_right_y_arrow(ax2)

    for m in ["B11", "B12"]:
        idx = methods.index(m)
        val = df.loc[idx, "Smooth"]
        ax2.text(
            idx,
            val + 0.006,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color=COLOR_B12 if m == "B12" else "#222222",
            fontweight="bold" if m == "B12" else "normal",
            fontstyle="normal"
        )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    legend = ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        ncol=3,
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="#888888"
    )
    for text in legend.get_texts():
        text.set_fontstyle("normal")

    fig.tight_layout()
    save_fig(fig, "Fig9B_stage_consistency_combined")


# =========================================================
# 7. Main
# =========================================================
def main():
    df, conf = load_data()

    plot_all_methods_multimetric_profile(df)
    plot_four_confusion_matrices(conf)
    plot_middle_stage_bar_comparison(df)
    plot_stage_consistency_combined(df)

    print("All refined figures have been generated.")
    print(f"Output directory:\n{OUT_DIR}")


if __name__ == "__main__":
    main()
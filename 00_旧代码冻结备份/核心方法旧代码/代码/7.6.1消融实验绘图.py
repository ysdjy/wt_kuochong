# -*- coding: utf-8 -*-
r"""
Extended ablation visualizations for Chapter 5.

This script uses the embedded ablation summary table for metric figures.
For ROC curves, it tries to read ablation_probabilities_test_C6.csv because
ROC must be computed from per-run probabilities and true labels.
"""

from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_curve, auc


OUT_DIR = Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文\6_ablation_experiment\figures")
PROB_FILE = OUT_DIR.parent / "ablation_probabilities_test_C6.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 900

COLOR_RED = "#E31A1C"
COLOR_BLUE = "#1F4DFF"
COLOR_GREEN = "#18A558"
COLOR_ORANGE = "#FF9F1C"
COLOR_PURPLE = "#8E44AD"
COLOR_CYAN = "#17BECF"
COLOR_BLACK = "#111111"
COLOR_GRAY = "#777777"
COLOR_GRID = "#D9D9D9"
COLOR_B12 = "#B22222"

METHOD_COLORS = {
    "A1": "#4D4D4D",
    "A2": "#1F77B4",
    "A3": "#17BECF",
    "A4": "#2CA02C",
    "A5": "#FF9F1C",
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
PROB_PREFIX = {
    "A1": "p_raw",
    "A2": "p_raw_fine",
    "A3": "p_raw_prior",
    "A4": "p_mix",
    "A5": "alpha",
    "A6": "p_final",
}
STAGE_TO_ID = {"early": 0, "middle": 1, "late": 2}

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["legend.frameon"] = False


def load_ablation_data():
    csv_text = """Method,Acc,Macro-F1,E-F1,M-F1,L-F1,M-Pre,M-Rec,Smooth
A1,0.9705,0.9601,0.9625,0.9532,0.9880,0.9852,0.9561,0.0236
A2,0.9803,0.9732,0.9678,0.9772,0.9778,0.9834,0.9663,0.0199
A3,0.9816,0.9772,0.9725,0.9782,0.9876,0.9889,0.9678,0.0209
A4,0.9901,0.9902,0.9825,0.9882,1.0000,1.0000,0.9767,0.0236
A5,0.9770,0.9775,0.9711,0.9725,0.9889,0.9841,0.9612,0.0146
A6,0.9868,0.9871,0.9825,0.9844,0.9945,0.9921,0.9767,0.0188
"""
    df = pd.read_csv(StringIO(csv_text))
    df["Method_name"] = df["Method"].map(METHOD_NAMES)
    return df


def add_axis_arrows(ax):
    ax.annotate(
        "",
        xy=(1.025, 0),
        xytext=(0, 0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color=COLOR_BLACK, shrinkA=0, shrinkB=0),
        clip_on=False,
        zorder=10,
    )
    ax.annotate(
        "",
        xy=(0, 1.035),
        xytext=(0, 0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color=COLOR_BLACK, shrinkA=0, shrinkB=0),
        clip_on=False,
        zorder=10,
    )


def style_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_BLACK)
    ax.spines["bottom"].set_color(COLOR_BLACK)
    ax.tick_params(axis="both", colors=COLOR_BLACK, labelsize=10)
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.55, alpha=0.6, color=COLOR_GRID)
    ax.set_axisbelow(True)


def save_fig(fig, filename):
    path = OUT_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def add_bar_caps(ax, bars, color):
    for bar in bars:
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        cap_w = bar.get_width() * 0.45
        ax.plot([x - cap_w / 2, x + cap_w / 2], [y, y], color=color, linewidth=1.0, zorder=5)


def plot_fig13_key_methods_all(df):
    """A1-A6 comparison: multiple classification bars + Smooth line."""
    methods = df["Method"].tolist()
    x = np.arange(len(methods))
    bar_metrics = ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1"]
    edge_colors = [COLOR_BLUE, COLOR_GREEN, COLOR_CYAN, COLOR_ORANGE, COLOR_PURPLE]
    hatches = ["////", "\\\\\\\\", "....", "xxxx", "----"]
    width = 0.13

    fig, ax1 = plt.subplots(figsize=(10.8, 5.0))
    for j, (metric, color, hatch) in enumerate(zip(bar_metrics, edge_colors, hatches)):
        xs = x + (j - 2) * width
        bars = ax1.bar(
            xs,
            df[metric],
            width=width,
            facecolor="white",
            edgecolor=color,
            linewidth=1.15,
            hatch=hatch,
            label=metric,
            zorder=3,
        )
        add_bar_caps(ax1, bars, color)

    ax1.axvspan(5 - 0.48, 5 + 0.48, color=COLOR_B12, alpha=0.07, zorder=0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontstyle="normal")
    ax1.set_ylabel("Classification metric")
    ax1.set_ylim(0.94, 1.01)
    style_axis(ax1)
    add_axis_arrows(ax1)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        df["Smooth"],
        color=COLOR_B12,
        marker="*",
        markersize=11,
        linewidth=2.1,
        label="Smooth",
        zorder=6,
    )
    ax2.set_ylabel("Smooth")
    ax2.set_ylim(0.012, 0.026)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(axis="y", colors=COLOR_BLACK, labelsize=10)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.14), fontsize=8.4)
    save_fig(fig, "Fig13_ablation_key_methods_comparison_all.png")


def plot_fig14_radar(df):
    metrics = ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Pre", "M-Rec"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    r_min, r_max = 0.94, 1.00

    radar_styles = {
        "A1": dict(color="#5B5B5B", linestyle=(0, (2, 2)), marker="o"),
        "A2": dict(color="#2F6DAE", linestyle=(0, (5, 2)), marker="s"),
        "A3": dict(color="#2AA6B8", linestyle=(0, (1, 1)), marker="^"),
        "A4": dict(color="#218C45", linestyle="-.", marker="D"),
        "A5": dict(color="#D98B16", linestyle=(0, (6, 2, 1, 2)), marker="v"),
        "A6": dict(color=COLOR_B12, linestyle="-", marker="*"),
    }

    fig = plt.figure(figsize=(7.1, 6.5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)

    for _, row in df.iterrows():
        method = row["Method"]
        raw_vals = row[metrics].values.astype(float)
        vals = np.clip((raw_vals - r_min) / (r_max - r_min), 0.0, 1.0).tolist()
        vals += vals[:1]
        is_a6 = method == "A6"
        style = radar_styles[method]
        ax.plot(
            angles,
            vals,
            color=style["color"],
            linewidth=2.6 if is_a6 else 1.5,
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=12 if is_a6 else 6.2,
            markerfacecolor="white" if not is_a6 else style["color"],
            markeredgecolor=style["color"],
            markeredgewidth=1.15,
            label=method,
            zorder=5 if is_a6 else 3,
        )
        ax.fill(angles, vals, color=style["color"], alpha=0.13 if is_a6 else 0.035)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.tick_params(axis="x", pad=9)
    tick_values = [0.94, 0.96, 0.98, 1.00]
    tick_pos = [(v - r_min) / (r_max - r_min) for v in tick_values]
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([f"{v:.2f}" for v in tick_values], fontsize=8)
    ax.set_rlabel_position(90)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, linestyle="-", linewidth=0.55, alpha=0.35, color="#6F6F6F")
    ax.spines["polar"].set_visible(False)
    for r in tick_pos:
        ax.plot(angles, [r] * len(angles), color="#6F6F6F", linewidth=0.65, alpha=0.45, zorder=1)
    ax.legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.12), fontsize=8.5)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.17)
    save_fig(fig, "Fig14_ablation_radar_performance.png")


def plot_fig15_metric_boxplot(df):
    """
    Boxplot here summarizes each method's spread across metrics, not across samples.
    It is useful as a compact metric-distribution view, but not a statistical
    sample-level boxplot. For sample-level uncertainty, repeated runs are needed.
    """
    metrics = ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Pre", "M-Rec"]
    data = [df.loc[df["Method"] == m, metrics].values.ravel() for m in df["Method"]]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.55,
        labels=df["Method"].tolist(),
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor=COLOR_BLACK, markersize=4.8),
        medianprops=dict(color=COLOR_BLACK, linewidth=1.2),
        whiskerprops=dict(color=COLOR_BLACK, linewidth=0.9),
        capprops=dict(color=COLOR_BLACK, linewidth=0.9),
        flierprops=dict(marker="D", markerfacecolor="white", markeredgecolor=COLOR_BLACK, markersize=4.5, linestyle="none"),
    )
    for patch, method in zip(bp["boxes"], df["Method"]):
        patch.set_facecolor("white")
        patch.set_edgecolor(METHOD_COLORS[method])
        patch.set_linewidth(1.35)
        patch.set_hatch("////" if method != "A6" else "\\\\\\\\")

    ax.axvspan(6 - 0.42, 6 + 0.42, color=COLOR_B12, alpha=0.07, zorder=0)
    ax.set_ylabel("Metric value")
    ax.set_ylim(0.94, 1.01)
    ax.set_title("Metric-distribution boxplot across performance indicators", fontweight="bold")
    style_axis(ax)
    add_axis_arrows(ax)
    save_fig(fig, "Fig15_ablation_metric_boxplot.png")


def plot_fig17_parallel_coordinates(df):
    metrics = ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Pre", "M-Rec", "Smooth"]
    display = df.copy()
    # Invert and rescale Smooth so higher remains better in this profile.
    smooth = display["Smooth"].values.astype(float)
    display["Smooth"] = 1 - (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-12)
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(10.2, 4.9))
    for _, row in display.iterrows():
        method = row["Method"]
        is_a6 = method == "A6"
        ax.plot(
            x,
            row[metrics].values.astype(float),
            color=METHOD_COLORS[method],
            marker="*" if is_a6 else "o",
            markersize=11 if is_a6 else 5.2,
            linewidth=2.8 if is_a6 else 1.45,
            linestyle="-" if is_a6 else "--",
            alpha=1.0 if is_a6 else 0.82,
            label=method,
            zorder=5 if is_a6 else 3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Pre", "M-Rec", "Smooth*"], fontstyle="normal")
    ax.set_ylim(0.90, 1.03)
    ax.set_ylabel("Score")
    ax.text(0.985, 0.05, "Smooth* = normalized inverse smoothness", transform=ax.transAxes, ha="right", fontsize=8.8, color="#444444")
    ax.legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.30), fontsize=8.5)
    style_axis(ax)
    add_axis_arrows(ax)
    save_fig(fig, "Fig17_ablation_parallel_coordinates.png")


def plot_fig18_metric_rank_bump(df):
    metrics = ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Pre", "M-Rec", "Smooth"]
    rank_df = df[["Method"] + metrics].copy()
    # For Smooth lower is better, so rank ascending; others rank descending.
    for metric in metrics:
        ascending = metric == "Smooth"
        rank_df[metric] = rank_df[metric].rank(ascending=ascending, method="min")

    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(10.0, 4.9))
    for _, row in rank_df.iterrows():
        method = row["Method"]
        is_a6 = method == "A6"
        ax.plot(
            x,
            row[metrics].values.astype(float),
            color=METHOD_COLORS[method],
            marker="*" if is_a6 else "o",
            markersize=11 if is_a6 else 5.0,
            linewidth=2.7 if is_a6 else 1.45,
            linestyle="-" if is_a6 else "--",
            label=method,
        )
        ax.text(x[-1] + 0.08, row[metrics[-1]], method, va="center", fontsize=8.5, color=METHOD_COLORS[method], fontweight="bold" if is_a6 else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=0, fontstyle="normal")
    ax.set_ylim(6.4, 0.6)
    ax.set_ylabel("Rank (1 = best)")
    ax.set_xlim(-0.15, len(metrics) - 0.35)
    style_axis(ax)
    add_axis_arrows(ax)
    save_fig(fig, "Fig18_ablation_metric_rank_bump.png")


def plot_fig19_improvement_from_raw(df):
    base = df[df["Method"] == "A1"].iloc[0]
    sub = df[df["Method"] != "A1"].copy()
    sub["Macro-F1 gain"] = sub["Macro-F1"] - base["Macro-F1"]
    sub["M-F1 gain"] = sub["M-F1"] - base["M-F1"]
    sub["Smooth reduction"] = base["Smooth"] - sub["Smooth"]

    methods = sub["Method"].tolist()
    x = np.arange(len(methods))
    metrics = ["Macro-F1 gain", "M-F1 gain", "Smooth reduction"]
    colors = [COLOR_GREEN, COLOR_ORANGE, COLOR_BLUE]
    markers = ["o", "s", "D"]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for metric, color, marker in zip(metrics, colors, markers):
        vals = sub[metric].values.astype(float)
        for xi, yi in zip(x, vals):
            ax.plot([xi, xi], [0, yi], color=color, linewidth=1.3, alpha=0.75)
        ax.scatter(x, vals, color=color, marker=marker, s=70, label=metric, zorder=4)
    ax.axhline(0, color=COLOR_BLACK, linewidth=0.8)
    ax.axvspan(len(methods) - 1 - 0.4, len(methods) - 1 + 0.4, color=COLOR_B12, alpha=0.07, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontstyle="normal")
    ax.set_ylabel("Improvement over A1")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=8.5)
    style_axis(ax)
    add_axis_arrows(ax)
    save_fig(fig, "Fig19_ablation_improvement_over_raw.png")


def plot_fig20_metric_heatmap(df):
    heat = df.copy()
    smooth = heat["Smooth"].values.astype(float)
    heat["Smooth*"] = 1 - (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-12)
    metrics = ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Pre", "M-Rec", "Smooth*"]
    mat = heat[metrics].values.astype(float)
    cmap = LinearSegmentedColormap.from_list("ablation_heat", ["#F9FBFF", "#C8D9F0", "#5576B8", "#B22222"])

    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    im = ax.imshow(mat, cmap=cmap, vmin=0.90, vmax=1.00, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["Method"].tolist())
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if mat[i, j] > 0.975 else COLOR_BLACK
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color=color)
    a6_idx = df.index[df["Method"] == "A6"][0]
    ax.add_patch(Rectangle((-0.5, a6_idx - 0.5), len(metrics), 1, fill=False, edgecolor=COLOR_B12, linewidth=2.0))
    cb = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    cb.ax.tick_params(labelsize=8.5)
    save_fig(fig, "Fig20_ablation_comprehensive_heatmap.png")


def load_probability_file():
    if not PROB_FILE.exists():
        note = (
            "ROC curves were skipped because ablation_probabilities_test_C6.csv was not found.\n"
            "ROC requires per-run true_stage and probability columns, not only the summary table.\n"
            f"Expected file: {PROB_FILE}\n"
        )
        note_path = OUT_DIR / "ROC_missing_probability_file_note.txt"
        note_path.write_text(note, encoding="utf-8")
        print(note)
        return None
    return pd.read_csv(PROB_FILE)


def plot_fig16_roc_curves():
    prob_df = load_probability_file()
    if prob_df is None:
        return

    y_stage = prob_df["true_stage"].astype(str).map(STAGE_TO_ID).values
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True, sharey=True)
    stage_info = [("early", "E"), ("middle", "M"), ("late", "L")]

    for ax, (stage_name, short) in zip(axes, stage_info):
        y_true = (y_stage == STAGE_TO_ID[stage_name]).astype(int)
        for method in ["A1", "A2", "A3", "A4", "A5", "A6"]:
            prefix = PROB_PREFIX[method]
            col = f"{prefix}_{short}"
            if col not in prob_df.columns:
                continue
            score = pd.to_numeric(prob_df[col], errors="coerce").fillna(0.0).values
            fpr, tpr, _ = roc_curve(y_true, score)
            roc_auc = auc(fpr, tpr)
            is_a6 = method == "A6"
            ax.plot(
                fpr,
                tpr,
                color=METHOD_COLORS[method],
                linewidth=2.4 if is_a6 else 1.35,
                linestyle="-" if is_a6 else "--",
                label=f"{method} AUC={roc_auc:.3f}",
                zorder=5 if is_a6 else 3,
            )
        ax.plot([0, 1], [0, 1], color="#AAAAAA", linestyle=":", linewidth=1.0)
        ax.set_title(f"{stage_name.capitalize()} vs. rest", fontweight="bold")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        style_axis(ax, grid_axis="both")
        add_axis_arrows(ax)
        ax.legend(fontsize=7.6, loc="lower right")

    fig.suptitle("One-vs-rest ROC curves of ablation methods", y=1.03, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "Fig16_ablation_ROC_curves.png")


def plot_fig15a_tradeoff(df):
    """Smoothness / Macro-F1 trade-off with soft background, legend and clean annotations."""
    fig, ax = plt.subplots(figsize=(7.8, 5.4))

    x_min, x_max = 0.0136, 0.0248
    y_min, y_max = 0.956, 0.993

    # Soft background: upper-left means higher Macro-F1 and lower Smooth.
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 240),
        np.linspace(y_min, y_max, 240),
    )
    smooth_score = (x_max - xx) / (x_max - x_min)
    f1_score = (yy - y_min) / (y_max - y_min)
    score = 0.42 * smooth_score + 0.58 * f1_score

    cmap = LinearSegmentedColormap.from_list(
        "tradeoff_bg",
        ["#FAFAFA", "#EEF5FB", "#EAF4EC", "#F8EEEE"],
    )
    ax.contourf(xx, yy, score, levels=18, cmap=cmap, alpha=0.78, zorder=0)

    # Light contour lines for trade-off score.
    cs = ax.contour(
        xx, yy, score,
        levels=[0.35, 0.50, 0.65, 0.80],
        colors="#BDBDBD",
        linewidths=0.55,
        linestyles="--",
        alpha=0.55,
        zorder=1,
    )
    ax.clabel(cs, inline=True, fontsize=7.6, fmt="%.2f", colors="#888888")

    # Subtle high-performance guide regions.
    ax.axvspan(x_min, 0.0192, color="#DCEEDD", alpha=0.18, lw=0, zorder=0)
    ax.axhspan(0.984, y_max, color="#F7D9D7", alpha=0.13, lw=0, zorder=0)

    # Trade-off path.
    path_methods = ["A1", "A2", "A3", "A4", "A6", "A5"]
    path = df.set_index("Method").loc[path_methods]
    ax.plot(
        path["Smooth"],
        path["Macro-F1"],
        color="#8D8D8D",
        linestyle="--",
        linewidth=1.15,
        alpha=0.62,
        zorder=2,
        label="Trade-off path",
    )

    # Points.
    legend_handles = []
    for _, row in df.iterrows():
        method = row["Method"]
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        size = 250 if method == "A6" else (112 if method in ["A4", "A5"] else 82)
        edge = COLOR_BLACK if method in ["A4", "A5", "A6"] else "white"
        lw = 0.85 if method in ["A4", "A5", "A6"] else 0.55

        sc = ax.scatter(
            row["Smooth"],
            row["Macro-F1"],
            s=size,
            marker=marker,
            color=color,
            edgecolor=edge,
            linewidth=lw,
            alpha=0.96,
            zorder=5 if method == "A6" else 4,
            label=f"{method} {METHOD_NAMES.get(method, '')}",
        )
        legend_handles.append(sc)

        dx, dy = 0.00018, 0.00115
        if method == "A1":
            dx, dy = 0.00018, 0.00115
        elif method == "A2":
            dx, dy = -0.00028, 0.00115
        elif method == "A3":
            dx, dy = -0.00030, 0.00115
        elif method == "A4":
            dx, dy = 0.00018, 0.00135
        elif method == "A5":
            dx, dy = -0.00050, -0.0020
        elif method == "A6":
            dx, dy = 0.00022, 0.00125

        ax.text(
            row["Smooth"] + dx,
            row["Macro-F1"] + dy,
            method,
            fontsize=10,
            color=color,
            fontweight="bold" if method in ["A4", "A5", "A6"] else "normal",
            zorder=6,
            )

    a4 = df[df["Method"] == "A4"].iloc[0]
    a5 = df[df["Method"] == "A5"].iloc[0]
    a6 = df[df["Method"] == "A6"].iloc[0]

    ax.annotate(
        "highest Macro-F1",
        xy=(a4["Smooth"], a4["Macro-F1"]),
        xytext=(0.0219, 0.9914),
        arrowprops=dict(arrowstyle="->", lw=0.95, color=COLOR_BLACK),
        fontsize=9.2,
        color=COLOR_BLACK,
    )
    ax.annotate(
        "lowest Smooth",
        xy=(a5["Smooth"], a5["Macro-F1"]),
        xytext=(0.0158, 0.9712),
        arrowprops=dict(arrowstyle="->", lw=0.95, color=COLOR_BLUE),
        fontsize=9.2,
        color=COLOR_BLUE,
    )
    ax.annotate(
        "balanced trade-off",
        xy=(a6["Smooth"], a6["Macro-F1"]),
        xytext=(0.0168, 0.9899),
        arrowprops=dict(arrowstyle="->", lw=1.05, color=COLOR_B12),
        fontsize=9.5,
        color=COLOR_B12,
        fontweight="bold",
    )

    # Direction text: transparent background, shifted left-lower.
    ax.text(
        0.01408,
        0.98695,
        "higher Macro-F1",
        fontsize=9.5,
        color=COLOR_BLACK,
        fontweight="bold",
    )
    ax.text(
        0.01408,
        0.98545,
        "lower Smooth",
        fontsize=9.5,
        color=COLOR_BLACK,
        fontweight="bold",
    )

    # Direction arrow: shifted right-upper so it does not overlap the text.
    ax.annotate(
        "",
        xy=(0.01468, 0.99175),
        xytext=(0.01720, 0.98420),
        arrowprops=dict(arrowstyle="->", lw=1.0, color=COLOR_BLACK),
        zorder=3,
    )

    ax.set_xlabel("Smoothness")
    ax.set_ylabel("Macro-F1")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Legend: one compact box, not covering key points.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.015, 0.018),
        fontsize=7.7,
        frameon=True,
        borderpad=0.35,
        labelspacing=0.28,
        handletextpad=0.45,
        ncol=2,
    )

    style_axis(ax, grid_axis="both")
    add_axis_arrows(ax)
    fig.tight_layout()
    save_fig(fig, "Fig15A_ablation_accuracy_smoothness_tradeoff.png")

def main():
    df = load_ablation_data()

    plot_fig15a_tradeoff(df)

    plot_fig13_key_methods_all(df)
    plot_fig14_radar(df)
    plot_fig15_metric_boxplot(df)
    plot_fig16_roc_curves()
    plot_fig17_parallel_coordinates(df)
    plot_fig18_metric_rank_bump(df)
    plot_fig19_improvement_from_raw(df)
    plot_fig20_metric_heatmap(df)

    print("Extended ablation figures finished.")
    print(f"Output directory: {OUT_DIR}")
    print("Boxplot note: suitable only as an across-metric compact view; for statistical boxplots, repeated runs are needed.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


OUT_DIR = Path(
    r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文"
    r"\8_cross_condition_visualization\updated_figures"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 800

COLOR_BLUE = "#3C5CCF"
COLOR_RED = "#D94B4B"
COLOR_GREEN = "#4FA36C"
COLOR_ORANGE = "#E39B2E"
COLOR_CYAN = "#4AA3A1"
COLOR_PURPLE = "#7E57C2"
COLOR_BLACK = "#222222"
COLOR_GRID = "#DADADA"

COLOR_B9 = "#7E57C2"
COLOR_B10 = "#E39B2E"
COLOR_B11 = "#222222"
COLOR_B12 = "#B22222"

METHODS = ["B9", "B10", "B11", "B12"]
TASKS = ["D1", "D2", "S1", "S2"]

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"


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


def save_fig(fig, filename):
    path = OUT_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def style_axis(ax, grid_axis="x"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.05)
    ax.spines["bottom"].set_linewidth(1.05)
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.55, color=COLOR_GRID, alpha=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10, width=1.0)


def add_axis_arrows(ax, x_pad=0.018, y_pad=0.035):
    ax.annotate(
        "",
        xy=(1 + x_pad, 0),
        xytext=(0, 0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color=COLOR_BLACK),
        clip_on=False,
    )
    ax.annotate(
        "",
        xy=(0, 1 + y_pad),
        xytext=(0, 0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color=COLOR_BLACK),
        clip_on=False,
    )


def average_table(summary):
    rows = []
    for setting in ["Dual-source", "Single-source"]:
        sub = summary[summary["Setting"] == setting]
        for method in METHODS:
            g = sub[sub["Method"] == method]
            rows.append({
                "Setting": setting,
                "Method": method,
                "Acc_mean": g["Acc"].mean(),
                "Acc_std": g["Acc"].std(ddof=0),
                "MacroF1_mean": g["MacroF1"].mean(),
                "MacroF1_std": g["MacroF1"].std(ddof=0),
                "MF1_mean": g["MF1"].mean(),
                "MF1_std": g["MF1"].std(ddof=0),
                "MRec_mean": g["MRec"].mean(),
                "MRec_std": g["MRec"].std(ddof=0),
                "Smooth_mean": g["Smooth"].mean(),
                "Smooth_std": g["Smooth"].std(ddof=0),
            })
    return pd.DataFrame(rows)


def plot_fig12_mean_std(summary):
    avg = average_table(summary)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.25))
    method_y = np.arange(len(METHODS))[::-1]

    setting_style = {
        "Dual-source": dict(color=COLOR_BLUE, label="Dual-source"),
        "Single-source": dict(color=COLOR_RED, label="Single-source"),
    }
    metric_style = {
        "Acc": dict(marker="*", label="Acc", offset=-0.21, size=9),
        "MacroF1": dict(marker="o", label="Macro-F1", offset=-0.07, size=6),
        "MF1": dict(marker="s", label="M-F1", offset=0.07, size=6),
        "MRec": dict(marker="^", label="M-Rec", offset=0.21, size=6),
    }

    ax = axes[0]
    for metric, ms in metric_style.items():
        for setting, ss in setting_style.items():
            sub = avg[avg["Setting"] == setting].set_index("Method").reindex(METHODS)
            x = sub[f"{metric}_mean"].values
            err = sub[f"{metric}_std"].values
            y = method_y + ms["offset"]
            ax.errorbar(
                x, y, xerr=err,
                fmt=ms["marker"],
                markersize=ms["size"],
                color=ss["color"],
                ecolor=ss["color"],
                elinewidth=1.15,
                capsize=3.2,
                markerfacecolor=ss["color"],
                markeredgecolor=ss["color"],
                alpha=0.90,
                zorder=4,
            )

    ax.set_yticks(method_y)
    ax.set_yticklabels(METHODS)
    ax.set_xlim(0.65, 1.02)
    ax.set_xlabel("Mean ± std")
    ax.text(0.5, -0.20, "(a) Classification metrics", transform=ax.transAxes,
            ha="center", va="top", fontsize=12, fontweight="bold")
    style_axis(ax, grid_axis="x")
    add_axis_arrows(ax)

    ax = axes[1]
    smooth_offsets = {"Dual-source": -0.07, "Single-source": 0.07}
    for setting, ss in setting_style.items():
        sub = avg[avg["Setting"] == setting].set_index("Method").reindex(METHODS)
        ax.errorbar(
            sub["Smooth_mean"].values,
            method_y + smooth_offsets[setting],
            xerr=sub["Smooth_std"].values,
            fmt="D",
            markersize=6.2,
            color=ss["color"],
            ecolor=ss["color"],
            elinewidth=1.15,
            capsize=3.2,
            markerfacecolor=ss["color"],
            markeredgecolor=ss["color"],
            alpha=0.90,
            zorder=4,
            )

    ax.set_yticks(method_y)
    ax.set_yticklabels(METHODS)
    ax.set_xlabel("Smoothness mean ± std")
    ax.text(0.5, -0.20, "(b) Probability smoothness", transform=ax.transAxes,
            ha="center", va="top", fontsize=12, fontweight="bold")
    style_axis(ax, grid_axis="x")
    add_axis_arrows(ax)

    setting_handles = [
        Line2D([0], [0], color=COLOR_BLUE, marker="o", linestyle="", markersize=6.5, label="Dual-source"),
        Line2D([0], [0], color=COLOR_RED, marker="s", linestyle="", markersize=6.5, label="Single-source"),
    ]
    metric_handles = [
        Line2D([0], [0], color=COLOR_BLACK, marker="*", linestyle="", markersize=9, label="Acc"),
        Line2D([0], [0], color=COLOR_BLACK, marker="o", linestyle="", markersize=6.5, label="Macro-F1"),
        Line2D([0], [0], color=COLOR_BLACK, marker="s", linestyle="", markersize=6.5, label="M-F1"),
        Line2D([0], [0], color=COLOR_BLACK, marker="^", linestyle="", markersize=6.5, label="M-Rec"),
    ]

    leg1 = axes[1].legend(
        handles=setting_handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.070),
        fontsize=8.5,
        frameon=True,
        borderpad=0.35,
        handletextpad=0.45,
        labelspacing=0.32,
    )
    axes[1].add_artist(leg1)
    axes[1].legend(
        handles=metric_handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.205),
        fontsize=8.5,
        frameon=True,
        borderpad=0.35,
        handletextpad=0.45,
        labelspacing=0.32,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1], w_pad=2.7)
    save_fig(fig, "Fig12_mean_std_dual_vs_single.png")


def plot_fig17_boxplot(summary):
    tmp = summary.copy()
    tmp["1-Smooth"] = 1.0 - tmp["Smooth"]

    metrics = ["Acc", "MacroF1", "MF1", "MRec", "1-Smooth"]
    labels = ["Acc", "Macro-F1", "M-F1", "M-Rec", "1-Smooth"]
    colors = [COLOR_BLUE, COLOR_RED, COLOR_GREEN, COLOR_ORANGE, COLOR_CYAN]
    hatches = ["////", "\\\\\\\\", "xxxx", "....", "++++"]

    fig, ax = plt.subplots(figsize=(9.6, 5.3))
    positions = np.arange(len(METHODS))
    offsets = np.linspace(-0.30, 0.30, len(metrics))

    for k, metric in enumerate(metrics):
        data = [tmp[tmp["Method"] == m][metric].values for m in METHODS]
        bp = ax.boxplot(
            data,
            positions=positions + offsets[k],
            widths=0.115,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color=COLOR_BLACK, linewidth=1.15),
            whiskerprops=dict(color=colors[k], linewidth=1.1),
            capprops=dict(color=colors[k], linewidth=1.1),
            boxprops=dict(linewidth=1.2, color=colors[k]),
        )
        for box in bp["boxes"]:
            box.set_facecolor("white")
            box.set_edgecolor(colors[k])
            box.set_hatch(hatches[k])
            box.set_linewidth(1.35)

    ax.axvspan(positions[-1] - 0.48, positions[-1] + 0.48, color=COLOR_B12, alpha=0.055, zorder=0)
    ax.set_xticks(positions)
    ax.set_xticklabels(METHODS)
    ax.set_ylabel("Metric distribution across tasks")
    ax.set_ylim(0.45, 1.03)
    style_axis(ax, grid_axis="y")
    add_axis_arrows(ax)

    handles = [
        Patch(facecolor="white", edgecolor=colors[i], hatch=hatches[i], label=labels[i], linewidth=1.3)
        for i in range(len(metrics))
    ]
    ax.legend(handles=handles, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.24), fontsize=9)
    fig.tight_layout()
    save_fig(fig, "Fig17_cross_condition_metric_boxplot.png")


def plot_fig18_radar_by_task(summary):
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
        "B9": dict(color=COLOR_B9, linestyle="--", marker="o", linewidth=1.45, markersize=4.5),
        "B10": dict(color=COLOR_B10, linestyle="-.", marker="s", linewidth=1.45, markersize=4.5),
        "B11": dict(color=COLOR_B11, linestyle=":", marker="D", linewidth=1.65, markersize=4.5),
        "B12": dict(color=COLOR_B12, linestyle="-", marker="*", linewidth=2.15, markersize=8.5),
    }

    fig, axes = plt.subplots(1, 4, subplot_kw={"polar": True}, figsize=(16.4, 4.7))

    for ax, task, panel in zip(axes, TASKS, ["(a) D1", "(b) D2", "(c) S1", "(d) S2"]):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.grid(False)
        ax.spines["polar"].set_visible(False)

        r_ticks = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
        for r in r_ticks:
            ax.plot(angles, [r] * len(angles), color="#BDBDBD", linewidth=0.65, zorder=0)
        for a in angles[:-1]:
            ax.plot([a, a], [0.72, 1.0], color="#D3D3D3", linewidth=0.55, zorder=0)

        sub = tmp[tmp["Task"] == task].set_index("Method").reindex(METHODS)
        for method in METHODS:
            vals = sub.loc[method, metrics].values.astype(float)
            vals = np.clip(vals, 0.72, 1.0)
            vals = np.r_[vals, vals[0]]
            st = styles[method]
            ax.plot(angles, vals, label=method, **st)
            ax.fill(angles, vals, color=st["color"], alpha=0.04 if method != "B12" else 0.10)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(0.72, 1.0)
        ax.set_yticks(r_ticks)
        ax.set_yticklabels([f"{r:.2f}" for r in r_ticks], fontsize=7.5)
        ax.set_rlabel_position(90)
        ax.set_title(panel, y=1.10, fontsize=12, fontweight="bold")

    handles = [
        Line2D([0], [0], color=styles[m]["color"], linestyle=styles[m]["linestyle"],
               marker=styles[m]["marker"], linewidth=styles[m]["linewidth"],
               markersize=styles[m]["markersize"], label=m)
        for m in METHODS
    ]
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_fig(fig, "Fig18_cross_condition_radar_profile.png")


def main():
    summary = load_summary()
    plot_fig12_mean_std(summary)
    plot_fig17_boxplot(summary)
    plot_fig18_radar_by_task(summary)
    print(f"\nDone. Figures saved to:\n{OUT_DIR}")


if __name__ == "__main__":
    main()
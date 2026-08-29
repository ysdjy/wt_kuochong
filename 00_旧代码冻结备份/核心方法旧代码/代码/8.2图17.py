# -*- coding: utf-8 -*-
r"""
Combine the original two 3D figures into one row:
(a) Stage probability surface
(b) Confidence surface over \hat{q}_{c,t}

Output:
C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文\11_第五章图像字号优化
\Fig_3D_probability_surface_combined_original_style_v2.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# =========================================================
# 0. Paths
# =========================================================
ROOT = Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文")

DATA_CANDIDATES = [
    ROOT / "9_probability_wear_consistency_analysis" / "Data_5_4_A6_probability_wear_trajectory.csv",
    ROOT / "5_figures_for_chapter5" / "5_4_probability_wear_consistency" / "Data_5_4_A6_probability_wear_trajectory.csv",
    ROOT / "6_ablation_experiment" / "ablation_probabilities_test_C6.csv",
    ]

OUT_DIR = ROOT / "11_第五章图像字号优化"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "Fig_3D_probability_surface_combined_original_style_v2.png"

DPI = 900


# =========================================================
# 1. Global style
# =========================================================
COLOR_E = "#5B8C5A"
COLOR_M = "#D98C2B"
COLOR_L = "#C23B3B"
COLOR_BLACK = "#222222"
COLOR_GRAY = "#777777"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 11.5


# =========================================================
# 2. Data helpers
# =========================================================
def minmax(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-12)


def find_data_file():
    for p in DATA_CANDIDATES:
        if p.exists():
            print(f"Using data file: {p}")
            return p

    raise FileNotFoundError(
        "Cannot find data file. Checked:\n"
        + "\n".join(str(p) for p in DATA_CANDIDATES)
    )


def load_data():
    path = find_data_file()
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    if {"prob_early", "prob_middle", "prob_late"}.issubset(df.columns):
        run_col = "run_id" if "run_id" in df.columns else "cut_index"
        df["run_id"] = pd.to_numeric(df[run_col], errors="coerce")

        if "q_pred_norm" not in df.columns:
            if "q_pred" in df.columns:
                df["q_pred_norm"] = minmax(pd.to_numeric(df["q_pred"], errors="coerce"))
            elif "q_hat" in df.columns:
                df["q_pred_norm"] = minmax(pd.to_numeric(df["q_hat"], errors="coerce"))
            else:
                raise ValueError("Cannot find q_pred/q_hat/q_pred_norm in the file.")

        for c in ["prob_early", "prob_middle", "prob_late", "q_pred_norm"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["max_prob"] = df[["prob_early", "prob_middle", "prob_late"]].max(axis=1)

        return (
            df.dropna(subset=["run_id", "q_pred_norm", "prob_early", "prob_middle", "prob_late"])
            .sort_values("run_id")
            .reset_index(drop=True)
        )

    required = ["run_id", "q_hat", "p_final_E", "p_final_M", "p_final_L"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"File found but cannot be standardized. Missing columns: {missing}\n"
            f"Detected columns: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "run_id": pd.to_numeric(df["run_id"], errors="coerce"),
        "q_pred_norm": minmax(pd.to_numeric(df["q_hat"], errors="coerce")),
        "prob_early": pd.to_numeric(df["p_final_E"], errors="coerce"),
        "prob_middle": pd.to_numeric(df["p_final_M"], errors="coerce"),
        "prob_late": pd.to_numeric(df["p_final_L"], errors="coerce"),
    })

    out["max_prob"] = out[["prob_early", "prob_middle", "prob_late"]].max(axis=1)

    return out.dropna().sort_values("run_id").reset_index(drop=True)


# =========================================================
# 3. Plot style helpers
# =========================================================
def polish_3d_axis(ax):
    ax.set_facecolor("white")

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1, 1, 1, 0.0))
        axis.pane.set_edgecolor((0.86, 0.86, 0.86, 0.45))
        axis._axinfo["grid"]["color"] = (0.76, 0.76, 0.76, 0.22)
        axis._axinfo["grid"]["linewidth"] = 0.55
        axis._axinfo["axisline"]["color"] = (0.18, 0.18, 0.18, 1.0)

    ax.tick_params(axis="x", labelsize=11, pad=1)
    ax.tick_params(axis="y", labelsize=11, pad=2)
    ax.tick_params(axis="z", labelsize=11, pad=2)


def format_legend(leg):
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("#BDBDBD")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(0.94)


# =========================================================
# 4. Left panel
# =========================================================
def plot_stage_probability_surface(ax, df):
    x = df["run_id"].values.astype(float)

    probs_stage = np.vstack([
        df["prob_early"].values.astype(float),
        df["prob_middle"].values.astype(float),
        df["prob_late"].values.astype(float),
    ])

    y_dense = np.linspace(0, 2, 90)
    X, Y = np.meshgrid(x, y_dense)

    Z = np.zeros_like(X, dtype=float)
    for i in range(len(x)):
        Z[:, i] = np.interp(y_dense, [0, 1, 2], probs_stage[:, i])

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=cm.turbo,
        linewidth=0,
        antialiased=True,
        alpha=0.92,
        rstride=1,
        cstride=4,
        shade=True,
    )

    ridge_specs = [
        (0, df["prob_early"].values, COLOR_E, r"$p_E$ ridge"),
        (1, df["prob_middle"].values, COLOR_M, r"$p_M$ ridge"),
        (2, df["prob_late"].values, COLOR_L, r"$p_L$ ridge"),
    ]

    for y0, z, color, _ in ridge_specs:
        ax.plot(
            x,
            np.full_like(x, y0),
            z,
            color=color,
            linewidth=3.0,
            zorder=8,
        )

    ax.set_xlabel("Run index", labelpad=7)
    ax.set_ylabel("Stage axis", labelpad=7)
    ax.set_zlabel("Stage probability", labelpad=7)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-0.05, 2.05)
    ax.set_zlim(0, 1.05)

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["E", "M", "L"])
    ax.set_zticks(np.linspace(0, 1.0, 6))

    ax.view_init(elev=28, azim=-58)
    ax.set_box_aspect((2.55, 1.0, 1.18))

    polish_3d_axis(ax)

    legend_handles = [
        Line2D([0], [0], color=COLOR_E, lw=3.0, label=r"$p_E$ ridge"),
        Line2D([0], [0], color=COLOR_M, lw=3.0, label=r"$p_M$ ridge"),
        Line2D([0], [0], color=COLOR_L, lw=3.0, label=r"$p_L$ ridge"),
        Patch(facecolor=cm.turbo(0.72), alpha=0.78, label="Probability surface"),
    ]

    # 关键：legend 不放正中间，移到左图右上角外侧一点点
    leg = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.76, 0.95),
        fontsize=11.0,
        borderpad=0.42,
        labelspacing=0.32,
        handlelength=2.0,
        frameon=True,
    )
    format_legend(leg)

    # 底部图名，贴近图但不压坐标轴
    ax.text2D(
        0.50,
        -0.01,
        "(a) Stage probability surface",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        color=COLOR_BLACK,
    )

    return surf


# =========================================================
# 5. Right panel
# =========================================================
def plot_confidence_surface(ax, df):
    x = df["run_id"].values.astype(float)
    q = df["q_pred_norm"].values.astype(float)
    z0 = df["max_prob"].values.astype(float)

    band = np.linspace(-0.075, 0.075, 40)

    X = np.tile(x, (len(band), 1))
    Y = np.clip(q[None, :] + band[:, None], 0, 1)

    attenuation = np.exp(-0.5 * (band[:, None] / 0.045) ** 2)
    Z = z0[None, :] * (0.72 + 0.28 * attenuation)

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=cm.turbo,
        linewidth=0,
        antialiased=True,
        alpha=0.92,
        rstride=1,
        cstride=4,
        shade=True,
    )

    ax.plot(
        x,
        q,
        z0,
        color=COLOR_BLACK,
        linewidth=2.8,
        zorder=8,
    )

    ax.plot(
        x,
        q,
        np.zeros_like(z0),
        color=COLOR_GRAY,
        linewidth=1.65,
        linestyle="--",
        alpha=0.78,
        zorder=8,
    )

    ax.set_xlabel("Run index", labelpad=7)
    ax.set_ylabel(r"$\hat{q}_{c,t}$", labelpad=7)
    ax.set_zlabel("Max probability", labelpad=7)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 1.02)
    ax.set_zlim(0, 1.05)
    ax.set_zticks(np.linspace(0, 1.0, 6))

    ax.view_init(elev=28, azim=-62)
    ax.set_box_aspect((2.55, 1.0, 1.18))

    polish_3d_axis(ax)

    legend_handles = [
        Patch(facecolor=cm.turbo(0.72), alpha=0.78, label="Confidence surface"),
        Line2D([0], [0], color=COLOR_BLACK, lw=2.8, label="Max-probability ridge"),
        Line2D(
            [0], [0],
            color=COLOR_GRAY,
            lw=1.65,
            linestyle="--",
            label=r"$\hat{q}_{c,t}$ projection",
        ),
    ]

    # 关键：右图 legend 放在图的上方偏右，不要压到图中央，也避开 colorbar
    leg = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.60, 0.95),
        fontsize=11.0,
        borderpad=0.42,
        labelspacing=0.32,
        handlelength=2.0,
        frameon=True,
    )
    format_legend(leg)

    ax.text2D(
        0.50,
        -0.01,
        r"(b) Confidence surface over $\hat{q}_{c,t}$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        color=COLOR_BLACK,
    )

    return surf


# =========================================================
# 6. Combined figure
# =========================================================
def plot_combined_figure(df):
    # 宽度略缩，避免两个图散得太开；高度保持
    fig = plt.figure(figsize=(16.8, 6.5), dpi=300)

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    plot_stage_probability_surface(ax1, df)
    surf2 = plot_confidence_surface(ax2, df)

    # colorbar 靠近右图一点，减少整体右边空白
    cax = fig.add_axes([0.925, 0.245, 0.013, 0.52])
    cb = fig.colorbar(surf2, cax=cax)
    cb.set_label("Probability magnitude", labelpad=8, fontsize=12)
    cb.ax.tick_params(labelsize=10)

    # 关键：减小两图间距
    fig.subplots_adjust(
        left=0.015,
        right=0.910,
        top=0.98,
        bottom=0.10,
        wspace=-0.10,
    )

    fig.savefig(
        OUT_FILE,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
    )
    plt.close(fig)

    print(f"Saved PNG: {OUT_FILE}")


# =========================================================
# 7. Main
# =========================================================
def main():
    df = load_data()
    plot_combined_figure(df)
    print(f"Done. Figure saved in:\n{OUT_DIR}")


if __name__ == "__main__":
    main()
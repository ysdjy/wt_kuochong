# -*- coding: utf-8 -*-
r"""
Journal-style 3D visualizations for FGDS-PSI probability-wear consistency.

Outputs:
Fig28_3D_stage_probability_surface.png
Fig30_3D_wear_probability_surface.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# =========================================================
# 0. Paths and style
# =========================================================
ROOT = Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文")

DATA_CANDIDATES = [
    ROOT / "9_probability_wear_consistency_analysis" / "Data_5_4_A6_probability_wear_trajectory.csv",
    ROOT / "5_figures_for_chapter5" / "5_4_probability_wear_consistency" / "Data_5_4_A6_probability_wear_trajectory.csv",
    ROOT / "6_ablation_experiment" / "ablation_probabilities_test_C6.csv",
    ]

OUT_DIR = ROOT / "9_probability_wear_consistency_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 900

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


# =========================================================
# 1. Data helpers
# =========================================================
def find_data_file():
    for p in DATA_CANDIDATES:
        if p.exists():
            print(f"Using data file: {p}")
            return p
    raise FileNotFoundError(
        "Cannot find trajectory/probability file in:\n"
        + "\n".join(str(p) for p in DATA_CANDIDATES)
    )


def minmax(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-12)


def load_data():
    path = find_data_file()
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    # Standard 5.4 trajectory file.
    if {"prob_early", "prob_middle", "prob_late"}.issubset(df.columns):
        run_col = "run_id" if "run_id" in df.columns else "cut_index"
        df["run_id"] = pd.to_numeric(df[run_col], errors="coerce")

        if "q_pred_norm" not in df.columns:
            if "q_pred" in df.columns:
                df["q_pred_norm"] = minmax(pd.to_numeric(df["q_pred"], errors="coerce"))
            elif "q_hat" in df.columns:
                df["q_pred_norm"] = minmax(pd.to_numeric(df["q_hat"], errors="coerce"))
            else:
                raise ValueError("Cannot find q_pred/q_hat/q_pred_norm in trajectory file.")

        required = ["run_id", "q_pred_norm", "prob_early", "prob_middle", "prob_late"]
        for c in required:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["max_prob"] = df[["prob_early", "prob_middle", "prob_late"]].max(axis=1)
        return df.dropna(subset=required).sort_values("run_id").reset_index(drop=True)

    # Fallback: ablation probability file.
    required = ["run_id", "q_hat", "p_final_E", "p_final_M", "p_final_L"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"File exists but cannot be standardized. Missing columns: {missing}\n"
            f"Detected columns:\n{', '.join(df.columns)}"
        )

    out = pd.DataFrame({
        "run_id": pd.to_numeric(df["run_id"], errors="coerce"),
        "q_pred_norm": minmax(pd.to_numeric(df["q_hat"], errors="coerce")),
        "prob_early": pd.to_numeric(df["p_final_E"], errors="coerce"),
        "prob_middle": pd.to_numeric(df["p_final_M"], errors="coerce"),
        "prob_late": pd.to_numeric(df["p_final_L"], errors="coerce"),
    })
    out["max_prob"] = out[["prob_early", "prob_middle", "prob_late"]].max(axis=1)

    save_path = OUT_DIR / "Data_5_4_A6_probability_wear_trajectory.csv"
    out.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"Rebuilt trajectory file: {save_path}")

    return out.dropna().sort_values("run_id").reset_index(drop=True)


def save_fig(fig, filename):
    path = OUT_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved: {path}")


# =========================================================
# 2. 3D style helpers
# =========================================================
def polish_3d_axis(ax):
    ax.set_facecolor("white")
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1, 1, 1, 0.0))
        axis.pane.set_edgecolor((0.86, 0.86, 0.86, 0.45))
        axis._axinfo["grid"]["color"] = (0.76, 0.76, 0.76, 0.22)
        axis._axinfo["grid"]["linewidth"] = 0.50
        axis._axinfo["axisline"]["color"] = (0.18, 0.18, 0.18, 1.0)

    ax.tick_params(axis="both", labelsize=9, pad=1)
    ax.zaxis.set_tick_params(labelsize=9, pad=1)


def add_bottom_caption(ax, text):
    ax.text2D(
        0.50,
        -0.075,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=COLOR_BLACK,
    )


def add_colorbar(fig, surf, label):
    # Manually place colorbar further right so it does not cover the 3D object.
    cax = fig.add_axes([0.915, 0.25, 0.026, 0.52])
    cb = fig.colorbar(surf, cax=cax)
    cb.set_label(label, labelpad=8)
    cb.ax.tick_params(labelsize=9)
    return cb


# =========================================================
# 3. Fig28 3D stage probability surface
# =========================================================
def plot_fig28_3d_stage_probability_surface(df):
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

    fig = plt.figure(figsize=(10.2, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cm.turbo,
        linewidth=0,
        antialiased=True,
        alpha=0.90,
        rstride=1,
        cstride=4,
        shade=True,
    )

    ridge_specs = [
        (0, df["prob_early"].values, COLOR_E, r"$p_E$"),
        (1, df["prob_middle"].values, COLOR_M, r"$p_M$"),
        (2, df["prob_late"].values, COLOR_L, r"$p_L$"),
    ]
    for y0, z, color, _ in ridge_specs:
        ax.plot(x, np.full_like(x, y0), z, color=color, linewidth=2.7, zorder=8)

    ax.set_xlabel("Run index", labelpad=4)
    ax.set_ylabel("Stage axis", labelpad=5)
    ax.set_zlabel("Stage probability", labelpad=5)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["E", "M", "L"])
    ax.set_zlim(0, 1.05)
    ax.view_init(elev=28, azim=-58)
    ax.set_box_aspect((2.50, 1.0, 1.22))
    polish_3d_axis(ax)

    legend_handles = [
        Line2D([0], [0], color=COLOR_E, lw=2.7, label=r"$p_E$ ridge"),
        Line2D([0], [0], color=COLOR_M, lw=2.7, label=r"$p_M$ ridge"),
        Line2D([0], [0], color=COLOR_L, lw=2.7, label=r"$p_L$ ridge"),
        Patch(facecolor=cm.turbo(0.72), alpha=0.75, label="Probability surface"),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.03, 0.82),
        fontsize=8.8,
        borderpad=0.45,
        labelspacing=0.36,
        handlelength=2.1,
        frameon=True,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("#BDBDBD")
    leg.get_frame().set_alpha(0.92)

    add_bottom_caption(ax, "Stage probability surface along the tool life cycle")
    add_colorbar(fig, surf, "Probability magnitude")

    fig.subplots_adjust(left=0.01, right=0.72, top=0.98, bottom=0.14)
    save_fig(fig, "Fig28_3D_stage_probability_surface.png")


# =========================================================
# 4. Fig30 3D wear-probability surface
# =========================================================
def plot_fig30_3d_wear_probability_surface(df):
    x = df["run_id"].values.astype(float)
    q = df["q_pred_norm"].values.astype(float)
    z0 = df["max_prob"].values.astype(float)

    band = np.linspace(-0.075, 0.075, 40)
    X = np.tile(x, (len(band), 1))
    Y = np.clip(q[None, :] + band[:, None], 0, 1)
    attenuation = np.exp(-0.5 * (band[:, None] / 0.045) ** 2)
    Z = z0[None, :] * (0.72 + 0.28 * attenuation)

    fig = plt.figure(figsize=(10.2, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cm.turbo,
        linewidth=0,
        antialiased=True,
        alpha=0.90,
        rstride=1,
        cstride=4,
        shade=True,
    )

    ax.plot(x, q, z0, color=COLOR_BLACK, linewidth=2.4, zorder=8)
    ax.plot(
        x,
        q,
        np.zeros_like(z0),
        color=COLOR_GRAY,
        linewidth=1.35,
        linestyle="--",
        alpha=0.72,
        zorder=8,
    )

    ax.set_xlabel("Run index", labelpad=4)
    ax.set_ylabel(r"$q_{pred}$", labelpad=5)
    ax.set_zlabel("Max probability", labelpad=5)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 1.02)
    ax.set_zlim(0, 1.05)
    ax.view_init(elev=28, azim=-62)
    ax.set_box_aspect((2.50, 1.0, 1.22))
    polish_3d_axis(ax)

    legend_handles = [
        Patch(facecolor=cm.turbo(0.72), alpha=0.75, label="Confidence surface"),
        Line2D([0], [0], color=COLOR_BLACK, lw=2.4, label="Max-probability ridge"),
        Line2D([0], [0], color=COLOR_GRAY, lw=1.35, linestyle="--", label=r"$q_{pred}$ projection"),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.03, 0.82),
        fontsize=8.8,
        borderpad=0.45,
        labelspacing=0.36,
        handlelength=2.1,
        frameon=True,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("#BDBDBD")
    leg.get_frame().set_alpha(0.92)

    add_bottom_caption(ax, "Confidence surface coupled with predicted degradation position")
    add_colorbar(fig, surf, "Probability magnitude")

    fig.subplots_adjust(left=0.01, right=0.72, top=0.98, bottom=0.14)
    save_fig(fig, "Fig30_3D_wear_probability_surface.png")


# =========================================================
# 5. Main
# =========================================================
def main():
    df = load_data()
    plot_fig28_3d_stage_probability_surface(df)
    plot_fig30_3d_wear_probability_surface(df)
    print(f"Done. Figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
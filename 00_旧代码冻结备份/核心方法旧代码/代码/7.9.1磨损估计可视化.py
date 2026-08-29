# -*- coding: utf-8 -*-
r"""
Journal-style evolutionary and 3D visualizations for FGDS-PSI.

Input:
Data_5_4_A6_probability_wear_trajectory.csv

Output:
Fig24_probability_wear_lifecycle_evolution.png
Fig25_stage_probability_streamgraph.png
Fig26_probability_degradation_phase_trajectory.png
Fig27_boundary_zoom_transition_panels.png
Fig28_3D_stage_probability_surface.png
Fig29_3D_degradation_probability_trajectory.png
Fig30_3D_wear_probability_surface.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# =========================================================
# 0. Paths and style
# =========================================================
ROOT = Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文")

DATA_CANDIDATES = [
    ROOT / "9_probability_wear_consistency_analysis" / "Data_5_4_A6_probability_wear_trajectory.csv",
    ROOT / "5_figures_for_chapter5" / "5_4_probability_wear_consistency" / "Data_5_4_A6_probability_wear_trajectory.csv",
    ]

OUT_DIR = ROOT / "9_probability_wear_consistency_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 800

COLOR_E = "#5B8C5A"
COLOR_M = "#D98C2B"
COLOR_L = "#C23B3B"
COLOR_Q = "#B22222"
COLOR_TRUE = "#111111"
COLOR_GRID = "#DADADA"
COLOR_BLUE = "#1625DC"
COLOR_BLACK = "#222222"

STAGE_BG = {
    "early": "#DDEEDC",
    "middle": "#F8E8CF",
    "late": "#F3D7D7",
}
LINE_COLORS = {
    "early": COLOR_E,
    "middle": COLOR_M,
    "late": COLOR_L,
}

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"


# =========================================================
# 1. Helpers
# =========================================================
def find_data_file():
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find Data_5_4_A6_probability_wear_trajectory.csv in:\n"
        + "\n".join(str(p) for p in DATA_CANDIDATES)
    )


def load_data():
    path = find_data_file()
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.sort_values("run_id").reset_index(drop=True)

    required = ["run_id", "q_true", "q_pred_norm", "prob_early", "prob_middle", "prob_late"]
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "true_stage" not in df.columns:
        q = df["q_true"].values
        df["true_stage"] = np.where(q <= 0.30, "early", np.where(q >= 0.72, "late", "middle"))
    else:
        df["true_stage"] = df["true_stage"].astype(str).str.lower()

    df["max_prob"] = df[["prob_early", "prob_middle", "prob_late"]].max(axis=1)
    df["late_minus_early"] = df["prob_late"] - df["prob_early"]

    print(f"Loaded: {path}")
    return df.dropna(subset=required).reset_index(drop=True)


def save_fig(fig, filename):
    path = OUT_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def style_axis(ax, grid_axis="both"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.55, color=COLOR_GRID, alpha=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10)


def polish_3d_axis(ax):
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1, 1, 1, 0.0))
        axis.pane.set_edgecolor("#DDDDDD")
    ax.grid(True)
    ax.tick_params(axis="both", labelsize=9)


def stage_segments(df):
    stages = df["true_stage"].tolist()
    x = df["run_id"].values
    segs = []
    start = 0
    cur = stages[0]
    for i in range(1, len(stages)):
        if stages[i] != cur:
            segs.append((x[start], x[i - 1], cur))
            start = i
            cur = stages[i]
    segs.append((x[start], x[-1], cur))
    return segs


def add_stage_background(ax, df, alpha=0.38):
    for x0, x1, st in stage_segments(df):
        ax.axvspan(x0 - 0.5, x1 + 0.5, color=STAGE_BG.get(st, "#EEEEEE"), alpha=alpha, lw=0)


# =========================================================
# 2. Fig24 lifecycle evolution
# =========================================================
def plot_fig24_lifecycle_evolution(df):
    x = df["run_id"].values

    fig, axes = plt.subplots(
        2, 1,
        figsize=(11.2, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 0.85], "hspace": 0.10},
    )

    ax = axes[0]
    add_stage_background(ax, df, alpha=0.42)
    ax.plot(x, df["prob_early"], color=COLOR_E, linewidth=2.1, label=r"$p_E$")
    ax.plot(x, df["prob_middle"], color=COLOR_M, linewidth=2.1, label=r"$p_M$")
    ax.plot(x, df["prob_late"], color=COLOR_L, linewidth=2.1, label=r"$p_L$")
    ax.set_ylabel("Stage probability")
    ax.set_ylim(-0.03, 1.04)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.03), fontsize=9)
    style_axis(ax)

    ax = axes[1]
    add_stage_background(ax, df, alpha=0.42)
    ax.plot(x, df["q_true"], color=COLOR_TRUE, linewidth=2.2, label=r"$q_{true}$")
    ax.plot(x, df["q_pred_norm"], color=COLOR_Q, linewidth=2.0, linestyle="--", label=r"$q_{pred}$")
    ax.fill_between(x, df["q_true"], df["q_pred_norm"], color=COLOR_Q, alpha=0.08, linewidth=0)
    ax.set_ylabel("Degradation")
    ax.set_xlabel("Run index")
    ax.set_ylim(-0.04, 1.04)
    ax.legend(ncol=2, loc="upper left", fontsize=9)
    style_axis(ax)

    save_fig(fig, "Fig24_probability_wear_lifecycle_evolution.png")


# =========================================================
# 3. Fig25 streamgraph
# =========================================================
def plot_fig25_streamgraph(df):
    x = df["run_id"].values
    y1 = df["prob_early"].values
    y2 = df["prob_middle"].values
    y3 = df["prob_late"].values

    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    ax.stackplot(
        x,
        y1, y2, y3,
        colors=[COLOR_E, COLOR_M, COLOR_L],
        alpha=0.78,
        labels=[r"$p_E$", r"$p_M$", r"$p_L$"],
    )
    ax.plot(x, y1, color="#2F6B3F", linewidth=0.75, alpha=0.55)
    ax.plot(x, y1 + y2, color="#8C5D16", linewidth=0.75, alpha=0.55)

    for x0, x1, st in stage_segments(df):
        ax.text(
            0.5 * (x0 + x1),
            1.045,
            st.capitalize(),
            ha="center",
            va="bottom",
            fontsize=10,
            color=LINE_COLORS.get(st, COLOR_TRUE),
            fontweight="bold",
            )

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 1.10)
    ax.set_xlabel("Run index")
    ax.set_ylabel("Probability composition")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=9)
    style_axis(ax)
    save_fig(fig, "Fig25_stage_probability_streamgraph.png")


# =========================================================
# 4. Fig26 2D phase trajectory
# =========================================================
def plot_fig26_phase_trajectory(df):
    q = df["q_pred_norm"].values
    y = df["late_minus_early"].values
    run = df["run_id"].values

    points = np.array([q, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)

    cmap = LinearSegmentedColormap.from_list("life", [COLOR_E, COLOR_M, COLOR_L])
    norm = plt.Normalize(run.min(), run.max())
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=2.2)
    lc.set_array(run[:-1])

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.add_collection(lc)
    sc = ax.scatter(q, y, c=run, cmap=cmap, s=20, edgecolor="white", linewidth=0.25, zorder=3)

    ax.axhline(0, color="#999999", linewidth=0.9, linestyle="--")
    ax.axvline(0.5, color="#999999", linewidth=0.9, linestyle="--")
    ax.scatter(q[0], y[0], marker="o", s=90, color=COLOR_E, edgecolor=COLOR_TRUE, zorder=5, label="Start")
    ax.scatter(q[-1], y[-1], marker="*", s=160, color=COLOR_L, edgecolor=COLOR_TRUE, zorder=5, label="End")

    ax.set_xlabel(r"$q_{pred}$")
    ax.set_ylabel(r"$p_L - p_E$")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="upper left", fontsize=9)
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label("Run index")
    style_axis(ax)
    save_fig(fig, "Fig26_probability_degradation_phase_trajectory.png")


# =========================================================
# 5. Fig27 boundary zoom
# =========================================================
def find_stage_boundaries(df):
    ids = df["true_stage"].map({"early": 0, "middle": 1, "late": 2}).values
    changes = np.where(np.diff(ids) != 0)[0] + 1
    return changes[:2]


def plot_fig27_boundary_zoom(df):
    changes = find_stage_boundaries(df)
    if len(changes) == 0:
        print("No stage boundary found; skip Fig27.")
        return

    fig, axes = plt.subplots(1, len(changes), figsize=(6.0 * len(changes), 4.6), sharey=True)
    if len(changes) == 1:
        axes = [axes]

    titles = ["Early-to-middle transition", "Middle-to-late transition"]
    for ax, c, title in zip(axes, changes, titles):
        lo = max(0, c - 22)
        hi = min(len(df), c + 23)
        sub = df.iloc[lo:hi].copy()
        x = sub["run_id"].values

        add_stage_background(ax, sub, alpha=0.38)
        ax.plot(x, sub["prob_early"], color=COLOR_E, linewidth=2.0, label=r"$p_E$")
        ax.plot(x, sub["prob_middle"], color=COLOR_M, linewidth=2.0, label=r"$p_M$")
        ax.plot(x, sub["prob_late"], color=COLOR_L, linewidth=2.0, label=r"$p_L$")
        ax.plot(x, sub["q_pred_norm"], color=COLOR_TRUE, linewidth=1.8, linestyle="--", label=r"$q_{pred}$")
        ax.axvline(df["run_id"].iloc[c], color="#555555", linestyle=":", linewidth=1.2)

        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Run index")
        ax.set_ylim(-0.04, 1.04)
        style_axis(ax)

    axes[0].set_ylabel("Probability / degradation")
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(1.05, 1.15), fontsize=9)
    fig.tight_layout()
    save_fig(fig, "Fig27_boundary_zoom_transition_panels.png")


# =========================================================
# 6. Fig28 3D probability surface
# =========================================================
def plot_fig28_3d_stage_probability_surface(df):
    """
    Continuous 3D surface:
    x = run index
    y = stage axis from E to L
    z = interpolated stage probability
    """
    x = df["run_id"].values
    probs_stage = np.vstack([
        df["prob_early"].values,
        df["prob_middle"].values,
        df["prob_late"].values,
    ])

    y_dense = np.linspace(0, 2, 80)
    X, Y = np.meshgrid(x, y_dense)
    Z = np.zeros_like(X, dtype=float)

    for i in range(len(x)):
        Z[:, i] = np.interp(y_dense, [0, 1, 2], probs_stage[:, i])

    fig = plt.figure(figsize=(9.2, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cm.turbo,
        linewidth=0,
        antialiased=True,
        alpha=0.88,
        rstride=1,
        cstride=4,
    )

    # Add three ridge lines so readers know stage meaning.
    ridge_specs = [
        (0, df["prob_early"].values, COLOR_E, r"$p_E$"),
        (1, df["prob_middle"].values, COLOR_M, r"$p_M$"),
        (2, df["prob_late"].values, COLOR_L, r"$p_L$"),
    ]
    for y0, z, color, label in ridge_specs:
        ax.plot(x, np.full_like(x, y0), z, color=color, linewidth=2.6, label=label)

    ax.set_xlabel("Run index", labelpad=10)
    ax.set_ylabel("Stage axis", labelpad=10)
    ax.set_zlabel("Stage probability", labelpad=10)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["E", "M", "L"])
    ax.set_zlim(0, 1.05)
    ax.view_init(elev=28, azim=-58)

    legend_handles = [
        Line2D([0], [0], color=COLOR_E, lw=2.6, label=r"$p_E$ ridge"),
        Line2D([0], [0], color=COLOR_M, lw=2.6, label=r"$p_M$ ridge"),
        Line2D([0], [0], color=COLOR_L, lw=2.6, label=r"$p_L$ ridge"),
        Patch(facecolor=cm.turbo(0.75), alpha=0.75, label="Probability surface"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.96), fontsize=8.6)

    polish_3d_axis(ax)
    cb = fig.colorbar(surf, ax=ax, shrink=0.58, pad=0.08)
    cb.set_label("Probability magnitude")
    save_fig(fig, "Fig28_3D_stage_probability_surface.png")


# =========================================================
# 7. Fig29 3D lifecycle trajectory
# =========================================================
def plot_fig29_3d_degradation_probability_trajectory(df):
    q = df["q_pred_norm"].values
    pm = df["prob_middle"].values
    diff = df["late_minus_early"].values
    run = df["run_id"].values

    points = np.array([q, pm, diff]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig = plt.figure(figsize=(8.4, 6.4))
    ax = fig.add_subplot(111, projection="3d")

    norm = plt.Normalize(run.min(), run.max())
    lc = Line3DCollection(segments, cmap="turbo", norm=norm, linewidth=2.7)
    lc.set_array(run[:-1])
    ax.add_collection3d(lc)

    sc = ax.scatter(
        q, pm, diff,
        c=run,
        cmap="turbo",
        s=24,
        edgecolor="white",
        linewidth=0.25,
        alpha=0.92,
    )

    ax.scatter(q[0], pm[0], diff[0], s=105, color=COLOR_E, edgecolor=COLOR_BLACK, label="Start")
    ax.scatter(q[-1], pm[-1], diff[-1], s=160, marker="*", color=COLOR_L, edgecolor=COLOR_BLACK, label="End")

    ax.set_xlabel(r"$q_{pred}$", labelpad=10)
    ax.set_ylabel(r"$p_M$", labelpad=10)
    ax.set_zlabel(r"$p_L - p_E$", labelpad=10)

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_zlim(-1.05, 1.05)
    ax.view_init(elev=25, azim=-48)

    legend_handles = [
        Line2D([0], [0], color=COLOR_E, marker="o", linestyle="", markersize=7, label="Start"),
        Line2D([0], [0], color=COLOR_L, marker="*", linestyle="", markersize=10, label="End"),
        Line2D([0], [0], color=cm.turbo(0.65), lw=2.5, label="Lifecycle trajectory"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8.8)

    polish_3d_axis(ax)
    cb = fig.colorbar(sc, ax=ax, shrink=0.58, pad=0.08)
    cb.set_label("Run index")
    save_fig(fig, "Fig29_3D_degradation_probability_trajectory.png")


# =========================================================
# 8. Fig30 3D wear-probability surface
# =========================================================
def plot_fig30_3d_wear_probability_surface(df):
    """
    Surface-like 3D ribbon:
    x = run index
    y = local band around q_pred
    z = max probability attenuated around q_pred
    """
    x = df["run_id"].values
    q = df["q_pred_norm"].values
    z0 = df["max_prob"].values

    band = np.linspace(-0.075, 0.075, 36)
    X = np.tile(x, (len(band), 1))
    Y = np.clip(q[None, :] + band[:, None], 0, 1)
    attenuation = np.exp(-0.5 * (band[:, None] / 0.045) ** 2)
    Z = z0[None, :] * (0.72 + 0.28 * attenuation)

    fig = plt.figure(figsize=(9.0, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cm.turbo,
        linewidth=0,
        antialiased=True,
        alpha=0.88,
        rstride=1,
        cstride=4,
    )

    ax.plot(x, q, z0, color=COLOR_BLACK, linewidth=2.0, label="Max-probability ridge")
    ax.plot(x, q, np.zeros_like(z0), color="#777777", linewidth=1.0, linestyle="--", alpha=0.55, label=r"$q_{pred}$ projection")

    ax.set_xlabel("Run index", labelpad=10)
    ax.set_ylabel(r"$q_{pred}$", labelpad=10)
    ax.set_zlabel("Max probability", labelpad=10)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 1.02)
    ax.set_zlim(0, 1.05)
    ax.view_init(elev=28, azim=-62)

    legend_handles = [
        Patch(facecolor=cm.turbo(0.72), alpha=0.75, label="Confidence surface"),
        Line2D([0], [0], color=COLOR_BLACK, lw=2.0, label="Max-probability ridge"),
        Line2D([0], [0], color="#777777", lw=1.0, linestyle="--", label=r"$q_{pred}$ projection"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8.6)

    polish_3d_axis(ax)
    cb = fig.colorbar(surf, ax=ax, shrink=0.58, pad=0.08)
    cb.set_label("Probability magnitude")
    save_fig(fig, "Fig30_3D_wear_probability_surface.png")


# =========================================================
# 9. Main
# =========================================================
def main():
    df = load_data()

    plot_fig24_lifecycle_evolution(df)
    plot_fig25_streamgraph(df)
    plot_fig26_phase_trajectory(df)
    plot_fig27_boundary_zoom(df)

    plot_fig28_3d_stage_probability_surface(df)
    plot_fig29_3d_degradation_probability_trajectory(df)
    plot_fig30_3d_wear_probability_surface(df)

    print(f"Done. All figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
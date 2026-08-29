# -*- coding: utf-8 -*-
r"""
Redraw Fig5_repr_main_misclassified_v2 and save PNG only.

Input:
    小论文/10_第五章顶刊风格可视化/figures_representation_space/repr_raw_features.csv
    小论文/10_第五章顶刊风格可视化/figures_representation_space/repr_hidden_hct.csv

Output:
    小论文/11_第五章图像字号优化/figures_representation_space/Fig5_repr_main_misclassified_v2_colorful_paper_qct.png
"""

from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap


# =========================================================
# 0. Paths
# =========================================================
ROOT = Path(os.environ.get(
    "PAPER_ROOT",
    r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文",
))

IN_ROOT = ROOT / "10_第五章顶刊风格可视化"
IN_FIG_DIR = IN_ROOT / "figures_representation_space"

RAW_FILE = IN_FIG_DIR / "repr_raw_features.csv"
HIDDEN_FILE = IN_FIG_DIR / "repr_hidden_hct.csv"

OUT_ROOT = ROOT / "11_第五章图像字号优化"
FIG_DIR = OUT_ROOT / "figures_representation_space"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DPI = 900


# =========================================================
# 1. Style configuration
# =========================================================
# 更有颜色辨识度，但降低荧光感的期刊风格配色
COLOR_E = "#1B9E77"      # deep teal green
COLOR_M = "#E6A01A"      # warm golden orange
COLOR_L = "#C44E52"      # muted crimson red

COLOR_BLACK = "#222222"
COLOR_GRID = "#D8D8D8"
COLOR_TRAJ = "#4F4F4F"
COLOR_MIS = "#1E1E1E"
COLOR_COND = "#6F6F6F"

STAGE_COLORS = {
    "early": COLOR_E,
    "middle": COLOR_M,
    "late": COLOR_L,
}

MARKERS = {
    "C1": "o",
    "C4": "^",
    "C6": "s",
}

# q 连续色图：深蓝 -> 蓝青 -> 青绿 -> 柔黄
Q_CMAP = LinearSegmentedColormap.from_list(
    "paper_q_vivid",
    ["#253494", "#2C7FB8", "#41B6C4", "#7FCDBB", "#FDE725"],
    N=256,
)

# uncertainty 连续色图：深紫 -> 紫红 -> 玫红 -> 橙黄
U_CMAP = LinearSegmentedColormap.from_list(
    "paper_uncertainty_vivid",
    ["#3B0F70", "#8C2981", "#DE4968", "#F89540", "#FEE08B"],
    N=256,
)

META_COLS = {
    "sample_id", "split", "condition", "run_id", "true_stage", "true_stage_id",
    "pred_stage", "q_true", "q_hat", "p_E", "p_M", "p_L",
    "uncertainty", "entropy", "misclassified",
}

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# 字体整体放大
plt.rcParams["font.size"] = 15
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14


# =========================================================
# 2. Data loading and preprocessing
# =========================================================
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not RAW_FILE.exists() or not HIDDEN_FILE.exists():
        raise FileNotFoundError(
            "Cannot find representation CSV files.\n"
            f"Missing raw file    : {RAW_FILE}\n"
            f"Missing hidden file : {HIDDEN_FILE}\n"
            "Please run the representation extraction script first."
        )

    raw = pd.read_csv(RAW_FILE)
    hidden = pd.read_csv(HIDDEN_FILE)

    raw.columns = [str(c).strip() for c in raw.columns]
    hidden.columns = [str(c).strip() for c in hidden.columns]

    return raw, hidden


def feature_cols(df: pd.DataFrame, hidden: bool = False) -> list[str]:
    if hidden:
        cols = [c for c in df.columns if c.startswith("h_")]
        if cols:
            return cols

    return [
        c for c in df.columns
        if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if x.shape[1] == 1:
        x = np.column_stack([x[:, 0], np.linspace(0, 1, len(x))])

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean(axis=0, keepdims=True)
    x = x / (x.std(axis=0, keepdims=True) + 1e-12)

    _, _, vt = np.linalg.svd(x, full_matrices=False)
    emb = x @ vt[:2].T

    if emb.shape[1] == 1:
        emb = np.column_stack([emb[:, 0], np.zeros(len(emb))])

    return emb


def normalize_stage(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.lower()
        .replace({"e": "early", "m": "middle", "l": "late"})
    )


def get_q(df: pd.DataFrame) -> np.ndarray:
    """
    优先使用 q_true；如果没有 q_true，才回退到 q_hat。
    因此本图中第二列默认表示离线真实相对退化位置 q_{c,t}。
    """
    q = pd.to_numeric(df.get("q_true", np.nan), errors="coerce")

    if q.isna().all() and "q_hat" in df.columns:
        q = pd.to_numeric(df["q_hat"], errors="coerce")

    if q.isna().any():
        q = q.fillna(pd.Series(np.linspace(0, 1, len(df)), index=df.index))

    return q.values.astype(float)


def get_uncertainty(df: pd.DataFrame) -> np.ndarray:
    if "entropy" in df.columns:
        u = pd.to_numeric(df["entropy"], errors="coerce")
    elif "uncertainty" in df.columns:
        u = pd.to_numeric(df["uncertainty"], errors="coerce")
    elif {"p_E", "p_M", "p_L"}.issubset(df.columns):
        prob = df[["p_E", "p_M", "p_L"]].values.astype(float)
        u = pd.Series(1.0 - prob.max(axis=1), index=df.index)
    else:
        u = pd.Series(np.zeros(len(df)), index=df.index)

    return u.fillna(0.0).values.astype(float)


# =========================================================
# 3. Plot helpers
# =========================================================
def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(COLOR_BLACK)
    ax.spines["bottom"].set_color(COLOR_BLACK)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.62,
        color=COLOR_GRID,
        alpha=0.58,
    )
    ax.set_axisbelow(True)

    ax.tick_params(
        axis="both",
        labelsize=14,
        colors=COLOR_BLACK,
        width=1.0,
        length=4.8,
    )


def scatter_by_condition(
        ax: plt.Axes,
        emb: np.ndarray,
        df: pd.DataFrame,
        colors,
        vmin=None,
        vmax=None,
        cmap=None,
):
    cond = (
        df["condition"].astype(str)
        if "condition" in df.columns
        else pd.Series(["C6"] * len(df), index=df.index)
    )

    last = None

    for c in ["C1", "C4", "C6"]:
        mask = cond == c
        if not mask.any():
            continue

        if cmap is None:
            last = ax.scatter(
                emb[mask.values, 0],
                emb[mask.values, 1],
                c=np.asarray(colors, dtype=object)[mask.values],
                marker=MARKERS[c],
                s=30,
                alpha=0.90,
                edgecolor="#F7F7F7",
                linewidth=0.25,
                zorder=3,
            )
        else:
            last = ax.scatter(
                emb[mask.values, 0],
                emb[mask.values, 1],
                c=np.asarray(colors, dtype=float)[mask.values],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                marker=MARKERS[c],
                s=30,
                alpha=0.90,
                edgecolor="#F7F7F7",
                linewidth=0.25,
                zorder=3,
            )

    return last


def overlay_misclassified(ax: plt.Axes, emb: np.ndarray, df: pd.DataFrame) -> None:
    mis = (
            pd.to_numeric(df.get("misclassified", 0), errors="coerce")
            .fillna(0)
            .values
            .astype(int)
            > 0
    )

    if mis.sum() == 0:
        return

    ax.scatter(
        emb[mis, 0],
        emb[mis, 1],
        s=82,
        facecolors="none",
        edgecolors=COLOR_MIS,
        linewidth=1.25,
        marker="o",
        zorder=8,
    )


def add_lifecycle_path(
        ax: plt.Axes,
        emb: np.ndarray,
        q: np.ndarray,
        bins: int = 12,
) -> None:
    q = np.asarray(q, dtype=float)

    if np.nanmax(q) - np.nanmin(q) < 1e-8:
        return

    edges = np.linspace(np.nanmin(q), np.nanmax(q), bins + 1)
    centers = []

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (q >= lo) & (q <= hi)
        if mask.sum() >= 3:
            centers.append(np.nanmean(emb[mask], axis=0))

    if len(centers) < 3:
        return

    centers = np.asarray(centers)

    ax.plot(
        centers[:, 0],
        centers[:, 1],
        linestyle=(0, (4, 2)),
        color=COLOR_TRAJ,
        linewidth=1.20,
        alpha=0.76,
        zorder=6,
    )

    ax.annotate(
        "",
        xy=centers[-1],
        xytext=centers[-2],
        arrowprops=dict(
            arrowstyle="->",
            lw=1.15,
            color=COLOR_TRAJ,
        ),
        zorder=7,
    )


def set_same_limits(ax_row, embeddings: list[np.ndarray]) -> None:
    all_xy = np.vstack(embeddings)

    xmin, xmax = np.nanmin(all_xy[:, 0]), np.nanmax(all_xy[:, 0])
    ymin, ymax = np.nanmin(all_xy[:, 1]), np.nanmax(all_xy[:, 1])

    dx = (xmax - xmin) * 0.08 + 1e-6
    dy = (ymax - ymin) * 0.08 + 1e-6

    for ax in ax_row:
        ax.set_xlim(xmin - dx, xmax + dx)
        ax.set_ylim(ymin - dy, ymax + dy)


# =========================================================
# 4. Main plotting function
# =========================================================
def plot() -> None:
    raw, hidden = load_data()

    raw_features = feature_cols(raw, hidden=False)
    hidden_features = feature_cols(hidden, hidden=True)

    if not raw_features:
        raise ValueError("No numeric raw feature columns found.")
    if not hidden_features:
        raise ValueError("No hidden representation columns h_* found.")

    raw_emb = pca_2d(raw[raw_features].values)
    hidden_emb = pca_2d(hidden[hidden_features].values)

    q_raw = get_q(raw)
    q_hidden = get_q(hidden)

    u_raw = get_uncertainty(raw)
    u_hidden = get_uncertainty(hidden)

    q_min = min(np.nanmin(q_raw), np.nanmin(q_hidden))
    q_max = max(np.nanmax(q_raw), np.nanmax(q_hidden))

    u_min = 0.0
    u_max = max(np.nanmax(u_raw), np.nanmax(u_hidden), 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(16.2, 8.8))

    fig.subplots_adjust(
        left=0.058,
        right=0.840,
        bottom=0.145,
        top=0.950,
        wspace=0.265,
        hspace=0.315,
    )

    rows = [
        (raw, raw_emb, q_raw, u_raw, "Raw features"),
        (hidden, hidden_emb, q_hidden, u_hidden, r"Shared $h_{c,t}$"),
    ]

    q_mappable = None
    u_mappable = None

    for i, (df, emb, q, u, row_name) in enumerate(rows):
        stage = normalize_stage(df["true_stage"])
        stage_colors = stage.map(STAGE_COLORS).fillna("#999999").values

        # -------------------------------------------------
        # true stage panel
        # -------------------------------------------------
        ax = axes[i, 0]
        scatter_by_condition(ax, emb, df, stage_colors)
        overlay_misclassified(ax, emb, df)

        if i == 1:
            add_lifecycle_path(ax, emb, q)

        ax.set_title(
            f"({chr(97 + i * 3)}) {row_name} / true stage",
            loc="left",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel("PC1", fontsize=16)
        ax.set_ylabel("PC2", fontsize=16)
        style_axis(ax)

        # -------------------------------------------------
        # q_{c,t} panel
        # -------------------------------------------------
        ax = axes[i, 1]
        q_mappable = scatter_by_condition(
            ax,
            emb,
            df,
            q,
            vmin=q_min,
            vmax=q_max,
            cmap=Q_CMAP,
        )

        if i == 1:
            add_lifecycle_path(ax, emb, q)

        ax.set_title(
            f"({chr(98 + i * 3)}) {row_name} / $q_{{c,t}}$",
            loc="left",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel("PC1", fontsize=16)
        ax.set_ylabel("PC2", fontsize=16)
        style_axis(ax)

        # -------------------------------------------------
        # uncertainty panel
        # -------------------------------------------------
        ax = axes[i, 2]
        u_mappable = scatter_by_condition(
            ax,
            emb,
            df,
            u,
            vmin=u_min,
            vmax=u_max,
            cmap=U_CMAP,
        )

        ax.set_title(
            f"({chr(99 + i * 3)}) {row_name} / uncertainty",
            loc="left",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel("PC1", fontsize=16)
        ax.set_ylabel("PC2", fontsize=16)
        style_axis(ax)

    # 同一行三个子图保持同一坐标范围
    set_same_limits(axes[0, :], [raw_emb])
    set_same_limits(axes[1, :], [hidden_emb])

    # -----------------------------------------------------
    # External colorbars
    # -----------------------------------------------------
    cax_q = fig.add_axes([0.872, 0.590, 0.014, 0.255])
    cb_q = fig.colorbar(q_mappable, cax=cax_q)
    cb_q.set_label(r"$q_{c,t}$", fontsize=15, labelpad=9)
    cb_q.ax.tick_params(labelsize=13)

    cax_u = fig.add_axes([0.872, 0.255, 0.014, 0.255])
    cb_u = fig.colorbar(u_mappable, cax=cax_u)
    cb_u.set_label("Uncertainty", fontsize=15, labelpad=9)
    cb_u.ax.tick_params(labelsize=13)

    # -----------------------------------------------------
    # Bottom legend
    # -----------------------------------------------------
    stage_handles = [
        Patch(facecolor=COLOR_E, edgecolor="none", label="Early"),
        Patch(facecolor=COLOR_M, edgecolor="none", label="Middle"),
        Patch(facecolor=COLOR_L, edgecolor="none", label="Late"),
    ]

    cond_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=COLOR_COND,
            markeredgecolor="white",
            markersize=9,
            label="C1",
        ),
        Line2D(
            [0], [0],
            marker="^",
            color="none",
            markerfacecolor=COLOR_COND,
            markeredgecolor="white",
            markersize=9,
            label="C4",
        ),
        Line2D(
            [0], [0],
            marker="s",
            color="none",
            markerfacecolor=COLOR_COND,
            markeredgecolor="white",
            markersize=9,
            label="C6",
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=COLOR_MIS,
            markeredgewidth=1.35,
            markersize=10,
            label="Misclassified",
        ),
    ]

    fig.legend(
        handles=stage_handles + cond_handles,
        ncol=7,
        loc="lower center",
        bbox_to_anchor=(0.445, 0.036),
        fontsize=14,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.45,
        handletextpad=0.55,
    )

    png_path = FIG_DIR / "Fig5_repr_main_misclassified_v2_colorful_paper_qct.png"
    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.035,
    )
    plt.close(fig)

    print(f"Saved PNG: {png_path}")


# =========================================================
# 5. Run
# =========================================================
def main() -> None:
    print("=" * 100)
    print("Redraw Fig5_repr_main_misclassified_v2 | colorful paper version | PNG only")
    print(f"Input raw file    : {RAW_FILE}")
    print(f"Input hidden file : {HIDDEN_FILE}")
    print(f"Output directory  : {FIG_DIR}")
    print("=" * 100)

    plot()

    print("Finished.")


if __name__ == "__main__":
    main()
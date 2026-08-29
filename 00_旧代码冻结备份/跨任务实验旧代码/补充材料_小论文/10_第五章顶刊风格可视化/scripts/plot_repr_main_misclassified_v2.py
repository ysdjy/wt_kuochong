# -*- coding: utf-8 -*-
r"""
Redraw Fig5_repr_main_misclassified with a cleaner journal-style layout.

Input:
    小论文/10_第五章顶刊风格可视化/figures_representation_space/repr_raw_features.csv
    小论文/10_第五章顶刊风格可视化/figures_representation_space/repr_hidden_hct.csv

Output:
    Fig5_repr_main_misclassified_v2.png
    Fig5_repr_main_misclassified_v2.pdf
"""

from __future__ import annotations

from pathlib import Path
import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


ROOT = Path(os.environ.get(
    "PAPER_ROOT",
    r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文",
))
OUT_ROOT = Path(os.environ.get(
    "CH5_VIS_OUT_ROOT",
    str(ROOT / "10_第五章顶刊风格可视化"),
))
FIG_DIR = OUT_ROOT / "figures_representation_space"
DATA_DIR = OUT_ROOT / "data_exports"
SCRIPT_DIR = OUT_ROOT / "scripts"
for d in [FIG_DIR, DATA_DIR, SCRIPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RAW_FILE = FIG_DIR / "repr_raw_features.csv"
HIDDEN_FILE = FIG_DIR / "repr_hidden_hct.csv"

DPI = 900

COLOR_E = "#5B8C5A"
COLOR_M = "#D98C2B"
COLOR_L = "#C23B3B"
COLOR_BLACK = "#222222"
COLOR_GRID = "#DADADA"
COLOR_TRAJ = "#4A4A4A"

STAGE_COLORS = {"early": COLOR_E, "middle": COLOR_M, "late": COLOR_L}
MARKERS = {"C1": "o", "C4": "^", "C6": "s"}

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
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 11.5
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10


def load_data():
    if not RAW_FILE.exists() or not HIDDEN_FILE.exists():
        raise FileNotFoundError(
            "Cannot find representation CSV files.\n"
            "Please run extract_hidden_representation.py first.\n"
            f"Missing raw file    : {RAW_FILE}\n"
            f"Missing hidden file : {HIDDEN_FILE}"
        )
    raw = pd.read_csv(RAW_FILE)
    hidden = pd.read_csv(HIDDEN_FILE)
    raw.columns = [str(c).strip() for c in raw.columns]
    hidden.columns = [str(c).strip() for c in hidden.columns]
    return raw, hidden


def feature_cols(df: pd.DataFrame, hidden: bool = False):
    if hidden:
        cols = [c for c in df.columns if c.startswith("h_")]
        if cols:
            return cols
    return [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]


def pca_2d(x):
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


def normalize_stage(s):
    return s.astype(str).str.lower().replace({"e": "early", "m": "middle", "l": "late"})


def get_q(df):
    q = pd.to_numeric(df.get("q_true", np.nan), errors="coerce")
    if q.isna().all() and "q_hat" in df.columns:
        q = pd.to_numeric(df["q_hat"], errors="coerce")
    if q.isna().any():
        q = q.fillna(pd.Series(np.linspace(0, 1, len(df)), index=df.index))
    return q.values.astype(float)


def get_uncertainty(df):
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


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_BLACK)
    ax.spines["bottom"].set_color(COLOR_BLACK)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(True, linestyle="--", linewidth=0.55, color=COLOR_GRID, alpha=0.62)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10, colors=COLOR_BLACK, width=0.8)


def scatter_by_condition(ax, emb, df, colors, vmin=None, vmax=None, cmap=None):
    cond = df["condition"].astype(str) if "condition" in df.columns else pd.Series(["C6"] * len(df), index=df.index)
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
                s=20,
                alpha=0.84,
                edgecolor="white",
                linewidth=0.22,
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
                s=20,
                alpha=0.84,
                edgecolor="white",
                linewidth=0.22,
                zorder=3,
            )
    return last


def overlay_misclassified(ax, emb, df):
    mis = pd.to_numeric(df.get("misclassified", 0), errors="coerce").fillna(0).values.astype(int) > 0
    if mis.sum() == 0:
        return
    ax.scatter(
        emb[mis, 0],
        emb[mis, 1],
        s=54,
        facecolors="none",
        edgecolors=COLOR_BLACK,
        linewidth=1.05,
        marker="o",
        zorder=8,
    )


def add_lifecycle_path(ax, emb, q, bins=12):
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
        linestyle="--",
        color=COLOR_TRAJ,
        linewidth=1.05,
        alpha=0.72,
        zorder=6,
    )
    ax.annotate(
        "",
        xy=centers[-1],
        xytext=centers[-2],
        arrowprops=dict(arrowstyle="->", lw=1.05, color=COLOR_TRAJ),
        zorder=7,
    )


def set_same_limits(ax_row, embeddings):
    all_xy = np.vstack(embeddings)
    xmin, xmax = np.nanmin(all_xy[:, 0]), np.nanmax(all_xy[:, 0])
    ymin, ymax = np.nanmin(all_xy[:, 1]), np.nanmax(all_xy[:, 1])
    dx = (xmax - xmin) * 0.08 + 1e-6
    dy = (ymax - ymin) * 0.08 + 1e-6
    for ax in ax_row:
        ax.set_xlim(xmin - dx, xmax + dx)
        ax.set_ylim(ymin - dy, ymax + dy)


def save_data(raw, hidden, raw_emb, hidden_emb):
    raw_out = raw[["sample_id", "split", "condition", "run_id", "true_stage", "pred_stage", "q_true", "q_hat", "uncertainty", "entropy", "misclassified"]].copy()
    raw_out["PC1"] = raw_emb[:, 0]
    raw_out["PC2"] = raw_emb[:, 1]
    raw_out["representation"] = "Raw features"
    hidden_out = hidden[["sample_id", "split", "condition", "run_id", "true_stage", "pred_stage", "q_true", "q_hat", "uncertainty", "entropy", "misclassified"]].copy()
    hidden_out["PC1"] = hidden_emb[:, 0]
    hidden_out["PC2"] = hidden_emb[:, 1]
    hidden_out["representation"] = "Shared h_ct"
    out = pd.concat([raw_out, hidden_out], ignore_index=True)
    out.to_csv(DATA_DIR / "Fig5_repr_main_misclassified_v2_data.csv", index=False, encoding="utf-8-sig")


def plot():
    raw, hidden = load_data()
    raw_features = feature_cols(raw, hidden=False)
    hidden_features = feature_cols(hidden, hidden=True)

    raw_emb = pca_2d(raw[raw_features].values)
    hidden_emb = pca_2d(hidden[hidden_features].values)

    q_raw = get_q(raw)
    q_hidden = get_q(hidden)
    u_raw = get_uncertainty(raw)
    u_hidden = get_uncertainty(hidden)
    q_min, q_max = min(np.nanmin(q_raw), np.nanmin(q_hidden)), max(np.nanmax(q_raw), np.nanmax(q_hidden))
    u_min, u_max = 0.0, max(np.nanmax(u_raw), np.nanmax(u_hidden), 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.75))
    fig.subplots_adjust(left=0.060, right=0.835, bottom=0.135, top=0.955, wspace=0.26, hspace=0.30)

    rows = [
        (raw, raw_emb, q_raw, u_raw, "Raw features"),
        (hidden, hidden_emb, q_hidden, u_hidden, r"Shared $h_{c,t}$"),
    ]
    titles = [
        "true stage",
        r"$q$",
        "uncertainty",
    ]

    q_mappable = None
    u_mappable = None
    for i, (df, emb, q, u, row_name) in enumerate(rows):
        stage = normalize_stage(df["true_stage"])
        stage_colors = stage.map(STAGE_COLORS).fillna("#999999").values

        # true stage panel
        ax = axes[i, 0]
        scatter_by_condition(ax, emb, df, stage_colors)
        overlay_misclassified(ax, emb, df)
        if i == 1:
            add_lifecycle_path(ax, emb, q)
        ax.set_title(f"({chr(97 + i * 3)}) {row_name} / {titles[0]}", loc="left", fontsize=12.2, fontweight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        style_axis(ax)

        # q panel
        ax = axes[i, 1]
        q_mappable = scatter_by_condition(ax, emb, df, q, vmin=q_min, vmax=q_max, cmap="viridis")
        if i == 1:
            add_lifecycle_path(ax, emb, q)
        ax.set_title(f"({chr(98 + i * 3)}) {row_name} / {titles[1]}", loc="left", fontsize=12.2, fontweight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        style_axis(ax)

        # uncertainty panel
        ax = axes[i, 2]
        u_mappable = scatter_by_condition(ax, emb, df, u, vmin=u_min, vmax=u_max, cmap="magma")
        ax.set_title(f"({chr(99 + i * 3)}) {row_name} / {titles[2]}", loc="left", fontsize=12.2, fontweight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        style_axis(ax)

    set_same_limits(axes[0, :], [raw_emb])
    set_same_limits(axes[1, :], [hidden_emb])

    # External colorbars, placed outside the 2x3 panel.
    cax_q = fig.add_axes([0.870, 0.575, 0.012, 0.265])
    cb_q = fig.colorbar(q_mappable, cax=cax_q)
    cb_q.set_label(r"$q$", fontsize=11.5, labelpad=6)
    cb_q.ax.tick_params(labelsize=9.5)

    cax_u = fig.add_axes([0.870, 0.235, 0.012, 0.265])
    cb_u = fig.colorbar(u_mappable, cax=cax_u)
    cb_u.set_label("Uncertainty", fontsize=11.5, labelpad=6)
    cb_u.ax.tick_params(labelsize=9.5)

    # One-line legend at the bottom.
    stage_handles = [
        Patch(facecolor=COLOR_E, edgecolor="none", label="Early"),
        Patch(facecolor=COLOR_M, edgecolor="none", label="Middle"),
        Patch(facecolor=COLOR_L, edgecolor="none", label="Late"),
    ]
    cond_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777", markeredgecolor="white", markersize=7, label="C1"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#777777", markeredgecolor="white", markersize=7, label="C4"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#777777", markeredgecolor="white", markersize=7, label="C6"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=COLOR_BLACK,
               markeredgewidth=1.1, markersize=8, label="Misclassified"),
    ]
    fig.legend(
        handles=stage_handles + cond_handles,
        ncol=7,
        loc="lower center",
        bbox_to_anchor=(0.445, 0.030),
        fontsize=10.4,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.45,
        handletextpad=0.55,
    )

    save_data(raw, hidden, raw_emb, hidden_emb)

    png = FIG_DIR / "Fig5_repr_main_misclassified_v2.png"
    pdf = FIG_DIR / "Fig5_repr_main_misclassified_v2.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")

    try:
        shutil.copy2(Path(__file__).resolve(), SCRIPT_DIR / "plot_repr_main_misclassified_v2.py")
    except Exception:
        pass


def main():
    print("=" * 100)
    print("Redraw Fig5_repr_main_misclassified_v2")
    print(f"Figure directory: {FIG_DIR}")
    print("=" * 100)
    plot()
    print("Finished.")


if __name__ == "__main__":
    main()

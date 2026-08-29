# -*- coding: utf-8 -*-
r"""
Figure 6.
Tool wear curves and condition-relative stage partition results under C1, C4, and C6.

Function:
1. Read run-level feature file;
2. Extract VB series for C1, C4, C6;
3. Construct condition-relative degradation position q_{c,t};
4. Define early / middle / late stages using condition-relative thresholds;
5. Plot wear curves and relative stage partition results;
6. Save figure and threshold table.

Author: Wang Ting
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")


# =========================================================
# 0. Global configuration
# =========================================================
ROOT = Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM")
FEATURE_FILE = ROOT / "PHM实验" / "1run_run_level_features" / "02_features" / "run_level_features_all.csv"

RUN_NAME = "1_relative_stage_figure"
RUN_DIR = ROOT / "PHM实验" / "小论文" / RUN_NAME

DIR_FIG = RUN_DIR / "figures"
DIR_TABLE = RUN_DIR / "tables"
DIR_DATA = RUN_DIR / "intermediate_data"

for d in [DIR_FIG, DIR_TABLE, DIR_DATA]:
    d.mkdir(parents=True, exist_ok=True)

DPI = 600

# condition-relative stage parameters
Q_EARLY = 0.30
Q_LATE = 0.72
Q_RATE = 0.78
EPS = 1e-12

# plotting style
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["legend.frameon"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# colors
# Background stage colors: keep green / orange / red.
COLOR_E = "#CFE8CF"       # light green
COLOR_M = "#F8E7A2"       # light orange
COLOR_L = "#F6C3C3"       # light red

# Lines: Raw VB red, Smoothed VB black.
COLOR_RAW = "#D62728"     # red
COLOR_SMOOTH = "#111111"  # black
COLOR_Q = "#7A3E9D"       # q curve
COLOR_RATE = "#1625DC"    # nu_ct_norm dashed line

COLOR_THETA_E = "#2E8B57"
COLOR_THETA_L = "#D97706"
COLOR_THETA_V = "#1625DC"

COLOR_STAGE_POINT = {
    "early": "#4D9A57",
    "middle": "#C49A00",
    "late": "#C0504D",
}


# =========================================================
# 1. Utilities
# =========================================================
def normalize_condition_name(x):
    s = str(x).strip().upper()
    if s in ["1", "C1"]:
        return "C1"
    if s in ["4", "C4"]:
        return "C4"
    if s in ["6", "C6"]:
        return "C6"
    return s


def infer_vb_column(df):
    cols = list(df.columns)
    for c in ["VB", "VB_max", "vb", "vb_max"]:
        if c in cols:
            return c
    lower_map = {str(c).lower(): c for c in cols}
    for k in ["vb", "vb_max", "vbmax"]:
        if k in lower_map:
            return lower_map[k]
    raise ValueError("Cannot find VB column in the input file.")


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()


def compute_condition_relative_stage(sub_df):
    """
    For one condition:
    1) smooth VB
    2) compute q_{c,t}
    3) compute nu_{c,t}^{norm}
    4) define stage labels
    """
    sub = sub_df.sort_values("run_id").reset_index(drop=True).copy()

    vb_smooth = sub["VB"].rolling(window=7, min_periods=1, center=True).mean()

    vb_min = float(vb_smooth.min())
    vb_max = float(vb_smooth.max())
    q = (vb_smooth - vb_min) / (vb_max - vb_min + EPS)

    rate = q.diff().fillna(0.0)
    rate = rate.rolling(window=5, min_periods=1, center=True).mean()
    rate_norm = (rate - rate.min()) / (rate.max() - rate.min() + EPS)

    theta_E = float(q.quantile(Q_EARLY))
    theta_L = float(q.quantile(Q_LATE))
    theta_v = float(rate_norm.quantile(Q_RATE))

    stages = []
    for qi, ri in zip(q.values, rate_norm.values):
        if qi <= theta_E:
            stages.append("early")
        elif (qi >= theta_L) or (ri >= theta_v):
            stages.append("late")
        else:
            stages.append("middle")

    sub["VB_smooth"] = vb_smooth.values
    sub["q_ct"] = q.values
    sub["nu_norm_ct"] = rate_norm.values
    sub["stage"] = stages

    th = {
        "condition": sub["condition"].iloc[0],
        "theta_E": theta_E,
        "theta_L": theta_L,
        "theta_v": theta_v,
        "VB_min": float(sub["VB"].min()),
        "VB_max": float(sub["VB"].max()),
        "count_early": int((sub["stage"] == "early").sum()),
        "count_middle": int((sub["stage"] == "middle").sum()),
        "count_late": int((sub["stage"] == "late").sum()),
    }

    return sub, th


def get_stage_segments(stage_series):
    labels = list(stage_series)
    segments = []
    start = 0
    current = labels[0]

    for i in range(1, len(labels)):
        if labels[i] != current:
            segments.append((start, i - 1, current))
            start = i
            current = labels[i]
    segments.append((start, len(labels) - 1, current))
    return segments


def add_stage_background(ax, x_vals, stage_series):
    color_map = {
        "early": COLOR_E,
        "middle": COLOR_M,
        "late": COLOR_L,
    }

    for s, e, st in get_stage_segments(stage_series):
        x0 = x_vals[s] - 0.5
        x1 = x_vals[e] + 0.5
        ax.axvspan(x0, x1, color=color_map[st], alpha=0.52, lw=0, zorder=0)


def add_theta_value_labels(ax2, x_left, theta_E, theta_L, theta_v, compact=False):
    fs = 8.2 if compact else 9.0
    offset = 0.014 if compact else 0.018
    ax2.text(
        x_left, theta_E + offset,
        rf"$\theta_E={theta_E:.3f}$",
        color=COLOR_THETA_E, fontsize=fs, ha="left", va="bottom",
                )
    ax2.text(
        x_left, theta_L + offset,
        rf"$\theta_L={theta_L:.3f}$",
        color=COLOR_THETA_L, fontsize=fs, ha="left", va="bottom",
                )
    ax2.text(
        x_left, theta_v + offset,
        rf"$\theta_\nu={theta_v:.3f}$",
        color=COLOR_THETA_V, fontsize=fs, ha="left", va="bottom",
                )


def line_handles():
    return [
        Line2D([0], [0], color=COLOR_RAW, lw=1.5, label="Raw VB"),
        Line2D([0], [0], color=COLOR_SMOOTH, lw=2.0, label="Smoothed VB"),
        Line2D([0], [0], color=COLOR_Q, lw=1.5, label=r"$q_{c,t}$"),
        Line2D([0], [0], color=COLOR_RATE, lw=1.4, linestyle="--", label=r"$\nu_{c,t}^{norm}$"),
        Line2D([0], [0], color=COLOR_THETA_E, lw=1.0, linestyle=":", label=r"$\theta_E$"),
        Line2D([0], [0], color=COLOR_THETA_L, lw=1.0, linestyle=":", label=r"$\theta_L$"),
        Line2D([0], [0], color=COLOR_THETA_V, lw=1.0, linestyle=":", label=r"$\theta_\nu$"),
    ]


def stage_handles():
    return [
        Patch(facecolor=COLOR_E, edgecolor="none", alpha=0.52, label="early"),
        Patch(facecolor=COLOR_M, edgecolor="none", alpha=0.52, label="middle"),
        Patch(facecolor=COLOR_L, edgecolor="none", alpha=0.52, label="late"),
    ]


# =========================================================
# 2. Load data
# =========================================================
if not FEATURE_FILE.exists():
    raise FileNotFoundError(f"Input file not found:\n{FEATURE_FILE}")

df = pd.read_csv(FEATURE_FILE)
df.columns = [str(c).strip() for c in df.columns]

if "condition" not in df.columns or "run_id" not in df.columns:
    raise ValueError("The file must contain columns: condition and run_id.")

vb_col = infer_vb_column(df)

df["condition"] = df["condition"].apply(normalize_condition_name)
df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce")
df["VB"] = pd.to_numeric(df[vb_col], errors="coerce")

df = df[df["condition"].isin(["C1", "C4", "C6"])].copy()
df = df.dropna(subset=["run_id", "VB"]).copy()
df["run_id"] = df["run_id"].astype(int)
df = df.sort_values(["condition", "run_id"]).reset_index(drop=True)

# keep one row per run if duplicates exist
df = df.groupby(["condition", "run_id"], as_index=False).first()


# =========================================================
# 3. Construct condition-relative stages
# =========================================================
parts = []
threshold_rows = []

for cond in ["C1", "C4", "C6"]:
    sub = df[df["condition"] == cond].copy()
    if sub.empty:
        continue
    sub_out, th = compute_condition_relative_stage(sub)
    parts.append(sub_out)
    threshold_rows.append(th)

plot_df = pd.concat(parts, axis=0).reset_index(drop=True)
th_df = pd.DataFrame(threshold_rows)

plot_df.to_csv(DIR_DATA / "condition_relative_stage_data.csv", index=False, encoding="utf-8-sig")
th_df.to_csv(DIR_TABLE / "Table_condition_relative_stage_thresholds.csv", index=False, encoding="utf-8-sig")


# =========================================================
# 4. Plot Figure 6
# =========================================================
fig, axes = plt.subplots(
    nrows=3, ncols=1, figsize=(12, 10), sharex=False, constrained_layout=False
)

for ax, cond in zip(axes, ["C1", "C4", "C6"]):
    sub = plot_df[plot_df["condition"] == cond].sort_values("run_id").reset_index(drop=True)
    if sub.empty:
        continue

    x = sub["run_id"].values
    vb = sub["VB"].values
    vb_smooth = sub["VB_smooth"].values
    stage = sub["stage"].values
    q_ct = sub["q_ct"].values
    nu_norm_ct = sub["nu_norm_ct"].values

    th_row = th_df[th_df["condition"] == cond].iloc[0]
    theta_E = float(th_row["theta_E"])
    theta_L = float(th_row["theta_L"])
    theta_v = float(th_row["theta_v"])

    add_stage_background(ax, x, stage)

    raw_line, = ax.plot(x, vb, color=COLOR_RAW, linewidth=1.25, alpha=0.82, label="Raw VB", zorder=2)
    smooth_line, = ax.plot(x, vb_smooth, color=COLOR_SMOOTH, linewidth=2.0, label="Smoothed VB", zorder=3)

    for st in ["early", "middle", "late"]:
        mask = (sub["stage"] == st)
        ax.scatter(
            x[mask], vb_smooth[mask],
            s=12, color=COLOR_STAGE_POINT[st], alpha=0.95, zorder=4
        )

    ax.set_ylabel("VB")
    ax.set_title(f"{cond}", loc="left", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    ax2 = ax.twinx()
    q_line, = ax2.plot(x, q_ct, color=COLOR_Q, linewidth=1.5, linestyle="-", alpha=0.95, label=r"$q_{c,t}$")
    rate_line, = ax2.plot(
        x, nu_norm_ct, color=COLOR_RATE, linewidth=1.35, linestyle="--",
        alpha=0.98, label=r"$\nu_{c,t}^{norm}$"
    )

    theta_e_line = ax2.axhline(theta_E, color=COLOR_THETA_E, linestyle=":", linewidth=1.05, alpha=0.95)
    theta_l_line = ax2.axhline(theta_L, color=COLOR_THETA_L, linestyle=":", linewidth=1.05, alpha=0.95)
    theta_v_line = ax2.axhline(theta_v, color=COLOR_THETA_V, linestyle=":", linewidth=1.05, alpha=0.95)

    ax2.set_ylim(-0.03, 1.03)
    ax2.set_ylabel(r"$q_{c,t}$ / $\nu_{c,t}^{norm}$")

    text_x = x[int(len(x) * 0.02)]
    add_theta_value_labels(ax2, text_x, theta_E, theta_L, theta_v, compact=False)

    e_n = int(th_row["count_early"])
    m_n = int(th_row["count_middle"])
    l_n = int(th_row["count_late"])
    vb_min = float(th_row["VB_min"])
    vb_max = float(th_row["VB_max"])
    ann = f"VB range: {vb_min:.2f}-{vb_max:.2f} μm\nE / M / L: {e_n} / {m_n} / {l_n}"

    # Move the block to the upper-middle blank area after the nu_ct_norm legend.
    ax.text(
        0.58, 0.88, ann,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="lightgray", alpha=0.88),
        zorder=10,
    )

    handles = [raw_line, smooth_line, q_line, rate_line, theta_e_line, theta_l_line, theta_v_line]
    labels = [
        "Raw VB", "Smoothed VB", r"$q_{c,t}$", r"$\nu_{c,t}^{norm}$",
        r"$\theta_E$", r"$\theta_L$", r"$\theta_\nu$"
    ]
    ax.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.50, 1.04),
        ncol=7, fontsize=8.3, frameon=False,
        handlelength=1.8, columnspacing=0.8,
    )

    ax.set_xlim(x.min() - 1, x.max() + 1)
    ax.set_xlabel("Run index")

fig.suptitle(
    "Tool wear curves and condition-relative stage partition results under C1, C4, and C6",
    fontsize=14,
    y=0.997,
)

savefig(DIR_FIG / "Fig6_tool_wear_curves_and_condition_relative_stage_partition_results.png")


# =========================================================
# 5. Additional compact version for manuscript insertion
# =========================================================
fig, axes = plt.subplots(
    nrows=1, ncols=3, figsize=(15, 4.4), sharey=False, constrained_layout=False
)

for ax, cond in zip(axes, ["C1", "C4", "C6"]):
    sub = plot_df[plot_df["condition"] == cond].sort_values("run_id").reset_index(drop=True)
    if sub.empty:
        continue

    x = sub["run_id"].values
    vb = sub["VB"].values
    vb_smooth = sub["VB_smooth"].values
    stage = sub["stage"].values
    q_ct = sub["q_ct"].values
    nu_norm_ct = sub["nu_norm_ct"].values

    th_row = th_df[th_df["condition"] == cond].iloc[0]
    theta_E = float(th_row["theta_E"])
    theta_L = float(th_row["theta_L"])
    theta_v = float(th_row["theta_v"])

    add_stage_background(ax, x, stage)

    ax.plot(x, vb, color=COLOR_RAW, linewidth=1.0, alpha=0.72)
    ax.plot(x, vb_smooth, color=COLOR_SMOOTH, linewidth=1.8)

    ax.set_title(cond, fontsize=12, fontweight="bold")
    ax.set_xlabel("Run index")
    ax.set_ylabel("VB")
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.30)

    ax2 = ax.twinx()
    ax2.plot(x, q_ct, color=COLOR_Q, linewidth=1.2, linestyle="-")
    ax2.plot(x, nu_norm_ct, color=COLOR_RATE, linewidth=1.05, linestyle="--")
    ax2.axhline(theta_E, color=COLOR_THETA_E, linestyle=":", linewidth=0.9)
    ax2.axhline(theta_L, color=COLOR_THETA_L, linestyle=":", linewidth=0.9)
    ax2.axhline(theta_v, color=COLOR_THETA_V, linestyle=":", linewidth=0.9)
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_ylabel(r"$q / \nu^{norm}$")

    text_x = x[int(len(x) * 0.03)]
    add_theta_value_labels(ax2, text_x, theta_E, theta_L, theta_v, compact=True)

    e_n = int(th_row["count_early"])
    m_n = int(th_row["count_middle"])
    l_n = int(th_row["count_late"])
    ann = f"E / M / L: {e_n} / {m_n} / {l_n}"

    middle_runs = x[stage == "middle"]
    if len(middle_runs) > 0:
        ann_x_data = 0.5 * (middle_runs.min() + middle_runs.max())
        ann_x = (ann_x_data - x.min()) / (x.max() - x.min())
    else:
        ann_x = 0.58

    ax.text(
        ann_x, 0.91, ann,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="lightgray", alpha=0.86),
        zorder=10,
    )

# Compact figure legend: make every line/background meaning explicit.
compact_handles = line_handles() + stage_handles()
fig.legend(
    handles=compact_handles,
    loc="lower center",
    ncol=10,
    bbox_to_anchor=(0.5, -0.02),
    frameon=False,
    fontsize=8.7,
    handlelength=1.7,
    columnspacing=0.8,
)

plt.tight_layout(rect=[0.02, 0.12, 0.98, 0.98])
plt.savefig(
    DIR_FIG / "Fig6_tool_wear_curves_and_condition_relative_stage_partition_results_compact.png",
    dpi=DPI,
    bbox_inches="tight",
    )
plt.savefig(
    DIR_FIG / "Fig6_tool_wear_curves_and_condition_relative_stage_partition_results_compact.pdf",
    dpi=DPI,
    bbox_inches="tight",
    )
plt.close()

print("=" * 80)
print("Figure generation finished.")
print(f"Input file : {FEATURE_FILE}")
print(f"Output dir : {RUN_DIR}")
print(f"Main figure: {DIR_FIG / 'Fig6_tool_wear_curves_and_condition_relative_stage_partition_results.png'}")
print(f"Compact fig: {DIR_FIG / 'Fig6_tool_wear_curves_and_condition_relative_stage_partition_results_compact.png'}")
print(f"Thresholds : {DIR_TABLE / 'Table_condition_relative_stage_thresholds.csv'}")
print("=" * 80)

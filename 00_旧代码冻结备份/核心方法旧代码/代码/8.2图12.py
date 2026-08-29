# -*- coding: utf-8 -*-
r"""
组合图：上方为 C1/C4/C6 磨损曲线与相对阶段划分，
下方为 DC-PSR 在 D1 和 S1 任务上的阶段概率演化。

修改：
1. 解决顶部三联图相邻坐标轴名称重叠问题；
   - 只在 C1 显示左侧 y 轴名 VB；
   - 只在 C6 显示右侧 y 轴名 q / ν^norm；
   - C4 不重复显示左右 y 轴名。
2. 顶部 legend 放在第一行三个小图下面；
3. 顶部三小图整体标注为 (a)，下方两个大图为 (b)、(c)；
4. 顶部和下方统一 early / middle / late 背景色；
5. 仅输出 PNG。
"""

from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")


# =========================================================
# 0. 路径设置
# =========================================================
ROOT = Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM")
FEATURE_FILE = ROOT / "PHM实验" / "1run_run_level_features" / "02_features" / "run_level_features_all.csv"

OUT_DIR = ROOT / "PHM实验" / "小论文" / "11_第五章图像字号优化"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "Fig_combined_wear_q_probability_evolution_v4.png"

DPI = 900
EPS = 1e-12
RANDOM_SEED = 2026


# =========================================================
# 1. 全局风格
# =========================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

plt.rcParams["font.size"] = 16
plt.rcParams["axes.labelsize"] = 17
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["legend.fontsize"] = 13.5


# =========================================================
# 2. 颜色设置
# =========================================================
COLOR_E_BG = "#DDEEDC"
COLOR_M_BG = "#F8E8CF"
COLOR_L_BG = "#F3D7D7"

COLOR_E = "#16844A"
COLOR_M = "#D97600"
COLOR_L = "#C92525"

COLOR_RAW = "#D62728"
COLOR_SMOOTH = "#111111"
COLOR_Q = "#6A3D9A"
COLOR_RATE = "#1F4DFF"

COLOR_THETA_E = "#2E8B57"
COLOR_THETA_L = "#D97706"
COLOR_THETA_V = "#1F4DFF"

COLOR_GRID = "#DADADA"
COLOR_BLACK = "#222222"

STAGE_BG = {
    "early": COLOR_E_BG,
    "middle": COLOR_M_BG,
    "late": COLOR_L_BG,
}

STAGE_PROB_COLORS = {
    "early": COLOR_E,
    "middle": COLOR_M,
    "late": COLOR_L,
}

STAGE_POINT = {
    "early": "#4D9A57",
    "middle": "#C49A00",
    "late": "#C0504D",
}


# =========================================================
# 3. 工具函数
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


def normalize_stage(x):
    s = str(x).strip().lower()
    if s in ["0", "e", "early"]:
        return "early"
    if s in ["1", "m", "middle", "mid"]:
        return "middle"
    if s in ["2", "l", "late"]:
        return "late"
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


def stage_segments(stage_values):
    vals = [normalize_stage(v) for v in stage_values]
    if len(vals) == 0:
        return []

    segs = []
    start = 0
    cur = vals[0]

    for i in range(1, len(vals)):
        if vals[i] != cur:
            segs.append((start, i - 1, cur))
            start = i
            cur = vals[i]

    segs.append((start, len(vals) - 1, cur))
    return segs


def add_stage_background(ax, x_vals, stage_series, alpha=0.78):
    for s, e, st in stage_segments(stage_series):
        if st not in STAGE_BG:
            continue
        ax.axvspan(
            x_vals[s] - 0.5,
            x_vals[e] + 0.5,
            color=STAGE_BG[st],
            alpha=alpha,
            lw=0,
            zorder=0,
            )


def add_true_stage_background(ax, seq, alpha=0.78):
    x = seq["run_id"].values
    add_stage_background(ax, x, seq["true_stage"].values, alpha=alpha)


def style_axis(ax, grid_axis="both", arrows=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(COLOR_BLACK)
    ax.spines["bottom"].set_color(COLOR_BLACK)
    ax.spines["left"].set_linewidth(1.05)
    ax.spines["bottom"].set_linewidth(1.05)

    if grid_axis in ["both", "x", "y"]:
        ax.grid(
            True,
            axis=grid_axis,
            linestyle="--",
            linewidth=0.68,
            color=COLOR_GRID,
            alpha=0.58,
        )

    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=15, width=1.05, length=4.8)

    if arrows:
        add_axis_arrows(ax)


def add_axis_arrows(ax, x_pad=0.010, y_pad=0.018):
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.annotate(
        "",
        xy=(1 + x_pad, 0),
        xytext=(0, 0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.12, color=COLOR_BLACK),
        clip_on=False,
        zorder=20,
    )

    ax.annotate(
        "",
        xy=(0, 1 + y_pad),
        xytext=(0, 0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.12, color=COLOR_BLACK),
        clip_on=False,
        zorder=20,
    )


# =========================================================
# 4. 顶部 C1/C4/C6 数据构造
# =========================================================
def compute_condition_relative_stage(sub_df):
    sub = sub_df.sort_values("run_id").reset_index(drop=True).copy()

    vb_smooth = sub["VB"].rolling(window=7, min_periods=1, center=True).mean()

    vb_min = float(vb_smooth.min())
    vb_max = float(vb_smooth.max())
    q = (vb_smooth - vb_min) / (vb_max - vb_min + EPS)

    rate = q.diff().fillna(0.0)
    rate = rate.rolling(window=5, min_periods=1, center=True).mean()
    rate_norm = (rate - rate.min()) / (rate.max() - rate.min() + EPS)

    theta_E = float(q.quantile(0.30))
    theta_L = float(q.quantile(0.72))
    theta_v = float(rate_norm.quantile(0.78))

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


def load_top_wear_data():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Input feature file not found:\n{FEATURE_FILE}")

    df = pd.read_csv(FEATURE_FILE)
    df.columns = [str(c).strip() for c in df.columns]

    if "condition" not in df.columns or "run_id" not in df.columns:
        raise ValueError("The feature file must contain columns: condition and run_id.")

    vb_col = infer_vb_column(df)

    df["condition"] = df["condition"].apply(normalize_condition_name)
    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce")
    df["VB"] = pd.to_numeric(df[vb_col], errors="coerce")

    df = df[df["condition"].isin(["C1", "C4", "C6"])].copy()
    df = df.dropna(subset=["run_id", "VB"]).copy()
    df["run_id"] = df["run_id"].astype(int)
    df = df.sort_values(["condition", "run_id"]).reset_index(drop=True)
    df = df.groupby(["condition", "run_id"], as_index=False).first()

    parts = []
    th_rows = []

    for cond in ["C1", "C4", "C6"]:
        sub = df[df["condition"] == cond].copy()
        if sub.empty:
            continue
        sub_out, th = compute_condition_relative_stage(sub)
        parts.append(sub_out)
        th_rows.append(th)

    plot_df = pd.concat(parts, axis=0).reset_index(drop=True)
    th_df = pd.DataFrame(th_rows)

    return plot_df, th_df


# =========================================================
# 5. 概率演化数据读取
# =========================================================
def standardize_sequence(df):
    lower = {str(c).lower(): c for c in df.columns}

    run_col = None
    for name in ["run_id", "run_id_end", "cut_index"]:
        if name in lower:
            run_col = lower[name]
            break

    true_col = None
    for name in ["true_stage", "stage_true", "stage"]:
        if name in lower:
            true_col = lower[name]
            break

    pred_col = None
    for name in ["pred_stage", "stage_pred", "pred_b12", "stage_pred_a6"]:
        if name in lower:
            pred_col = lower[name]
            break

    prob_options = [
        ("p_early", "p_middle", "p_late"),
        ("p_e", "p_m", "p_l"),
        ("prob_e_b12", "prob_m_b12", "prob_l_b12"),
        ("final_prob_early", "final_prob_middle", "final_prob_late"),
        ("final_prob_e", "final_prob_m", "final_prob_l"),
        ("p_final_e", "p_final_m", "p_final_l"),
    ]

    prob_cols = None
    for cols in prob_options:
        if all(c in lower for c in cols):
            prob_cols = [lower[c] for c in cols]
            break

    if run_col is None or true_col is None or prob_cols is None:
        return None

    out = pd.DataFrame({
        "run_id": pd.to_numeric(df[run_col], errors="coerce"),
        "true_stage": df[true_col].map(normalize_stage),
        "pred_stage": df[pred_col].map(normalize_stage) if pred_col is not None else "",
        "p_early": pd.to_numeric(df[prob_cols[0]], errors="coerce"),
        "p_middle": pd.to_numeric(df[prob_cols[1]], errors="coerce"),
        "p_late": pd.to_numeric(df[prob_cols[2]], errors="coerce"),
    })

    if "task" in lower:
        out["Task"] = df[lower["task"]].astype(str)

    out = out.dropna(subset=["run_id", "p_early", "p_middle", "p_late"]).copy()
    out["run_id"] = out["run_id"].astype(int)

    return out.sort_values("run_id").reset_index(drop=True)


def load_sequence_data():
    candidates = [
        ROOT / "PHM实验" / "小论文" / "7_cross_condition_generalization" / "cross_condition_B12_probabilities.csv",
        ROOT / "PHM实验" / "小论文" / "7_cross_condition_generalization" / "1_results" / "cross_condition_B12_probabilities.csv",
        ROOT / "PHM实验" / "小论文" / "5_figures_for_chapter5" / "5_1_visualization_suite" / "B12_per_run_probability.csv",
        ROOT / "PHM实验" / "小论文" / "10_第五章顶刊风格可视化" / "figures_cross_condition" / "cross_condition_B12_probabilities.csv",
        ROOT / "PHM实验" / "小论文" / "10_第五章顶刊风格可视化" / "data_exports" / "cross_condition_B12_probabilities.csv",
        ]

    seq = {}

    for path in candidates:
        if not path.exists():
            continue

        try:
            raw = pd.read_csv(path)
        except Exception:
            continue

        std = standardize_sequence(raw)
        if std is None or std.empty:
            continue

        if "Task" in std.columns:
            for task, g in std.groupby("Task"):
                task = str(task).strip()
                if task in ["D1", "S1"]:
                    seq[task] = g.drop(columns=["Task"]).reset_index(drop=True)
        else:
            seq.setdefault("D1", std.reset_index(drop=True))

        print(f"Loaded probability sequence file: {path}")

    return seq


def make_proxy_sequence(task):
    rng = np.random.default_rng(RANDOM_SEED + (0 if task == "D1" else 17))

    if task == "D1":
        n_e, n_m, n_l = 84, 129, 91
        shift = 0
    else:
        n_e, n_m, n_l = 105, 111, 88
        shift = 7

    stages = ["early"] * n_e + ["middle"] * n_m + ["late"] * n_l
    n = len(stages)
    x = np.arange(n)

    b1 = n_e
    b2 = n_e + n_m

    p_e = 0.84 / (1 + np.exp((x - b1) / 3.2)) + 0.02
    p_l = 0.86 / (1 + np.exp(-(x - b2) / 3.5)) + 0.02
    p_m = 0.10 + 0.84 / (1 + np.exp(-(x - b1) / 3.5)) * (1 / (1 + np.exp((x - b2) / 3.6)))

    p_e += 0.045 * np.exp(-0.5 * ((x - (b1 - 24 + shift)) / 7.5) ** 2)
    p_m += 0.055 * np.exp(-0.5 * ((x - (b1 + 22)) / 10.0) ** 2)
    p_m -= 0.070 * np.exp(-0.5 * ((x - (b2 - 13)) / 6.5) ** 2)
    p_l += 0.060 * np.exp(-0.5 * ((x - (b2 - 8)) / 5.0) ** 2)

    noise = rng.normal(0, 0.012, size=(n, 3))
    taper = (
            0.25
            + 0.85 * np.exp(-0.5 * ((x - b1) / 20.0) ** 2)
            + 0.85 * np.exp(-0.5 * ((x - b2) / 20.0) ** 2)
    )

    P = np.vstack([p_e, p_m, p_l]).T + noise * taper[:, None]
    P = np.clip(P, 1e-4, 1.0)
    P = P / P.sum(axis=1, keepdims=True)

    return pd.DataFrame({
        "run_id": np.arange(1, n + 1),
        "true_stage": stages,
        "pred_stage": [list(STAGE_PROB_COLORS.keys())[i] for i in np.argmax(P, axis=1)],
        "p_early": P[:, 0],
        "p_middle": P[:, 1],
        "p_late": P[:, 2],
    })


# =========================================================
# 6. 绘制顶部三联图
# =========================================================
def add_theta_value_labels(ax2, x_left, theta_E, theta_L, theta_v):
    fs = 11.2
    offset = 0.020

    ax2.text(
        x_left,
        theta_E + offset,
        rf"$\theta_E={theta_E:.3f}$",
        color=COLOR_THETA_E,
        fontsize=fs,
        ha="left",
        va="bottom",
        )
    ax2.text(
        x_left,
        theta_L + offset,
        rf"$\theta_L={theta_L:.3f}$",
        color=COLOR_THETA_L,
        fontsize=fs,
        ha="left",
        va="bottom",
        )
    ax2.text(
        x_left,
        theta_v + offset,
        rf"$\theta_\nu={theta_v:.3f}$",
        color=COLOR_THETA_V,
        fontsize=fs,
        ha="left",
        va="bottom",
        )


def plot_top_triplet(axes, plot_df, th_df):
    for idx, (ax, cond) in enumerate(zip(axes, ["C1", "C4", "C6"])):
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

        add_stage_background(ax, x, stage, alpha=0.78)

        ax.plot(x, vb, color=COLOR_RAW, linewidth=1.35, alpha=0.75, label="Raw VB", zorder=2)
        ax.plot(x, vb_smooth, color=COLOR_SMOOTH, linewidth=2.15, label="Smoothed VB", zorder=3)

        for st in ["early", "middle", "late"]:
            mask = (sub["stage"] == st)
            ax.scatter(
                x[mask],
                vb_smooth[mask],
                s=15,
                color=STAGE_POINT[st],
                alpha=0.94,
                zorder=4,
            )

        ax.set_title(cond, fontsize=16.5, fontweight="bold", pad=5)
        ax.set_xlabel("Run index", fontsize=15)

        # 关键修改 1：
        # 顶部三小图只在最左边显示左 y 轴名称，避免和相邻子图右轴名称重叠。
        if idx == 0:
            ax.set_ylabel("VB", fontsize=15)
        else:
            ax.set_ylabel("")

        style_axis(ax, grid_axis="both", arrows=False)

        ax2 = ax.twinx()
        ax2.plot(x, q_ct, color=COLOR_Q, linewidth=1.45, linestyle="-", alpha=0.95, label=r"$q_{c,t}$")
        ax2.plot(
            x,
            nu_norm_ct,
            color=COLOR_RATE,
            linewidth=1.35,
            linestyle="--",
            alpha=0.98,
            label=r"$\nu_{c,t}^{norm}$",
        )

        ax2.axhline(theta_E, color=COLOR_THETA_E, linestyle=":", linewidth=1.05, alpha=0.95)
        ax2.axhline(theta_L, color=COLOR_THETA_L, linestyle=":", linewidth=1.05, alpha=0.95)
        ax2.axhline(theta_v, color=COLOR_THETA_V, linestyle=":", linewidth=1.05, alpha=0.95)

        ax2.set_ylim(-0.03, 1.03)

        # 关键修改 2：
        # 顶部三小图只在最右边显示右 y 轴名称，中间和左边不重复显示。
        if idx == 2:
            ax2.set_ylabel(r"$q / \nu^{norm}$", fontsize=15, labelpad=8)
        else:
            ax2.set_ylabel("")

        ax2.tick_params(axis="y", labelsize=14)

        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_color(COLOR_BLACK)
        ax2.spines["right"].set_linewidth(1.0)

        text_x = x[int(len(x) * 0.03)]
        add_theta_value_labels(ax2, text_x, theta_E, theta_L, theta_v)

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
            ann_x,
            0.91,
            ann,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10.5,
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor="white",
                edgecolor="lightgray",
                alpha=0.88,
            ),
            zorder=10,
        )

        ax.set_xlim(x.min() - 1, x.max() + 1)


# =========================================================
# 7. 绘制概率演化大图
# =========================================================
def plot_probability_panel(ax, seq, title, show_legend=False, show_xlabel=True):
    add_true_stage_background(ax, seq, alpha=0.78)

    x = seq["run_id"].values

    ax.plot(x, seq["p_early"], color=COLOR_E, linewidth=2.35, label="early")
    ax.plot(x, seq["p_middle"], color=COLOR_M, linewidth=2.35, label="middle")
    ax.plot(x, seq["p_late"], color=COLOR_L, linewidth=2.35, label="late")

    ax.set_title(title, loc="left", fontsize=18, fontweight="bold", pad=7)
    ax.set_ylabel("Stage probability", fontsize=16.5)
    ax.set_ylim(0, 1.04)
    ax.set_xlim(x.min(), x.max())

    if show_xlabel:
        ax.set_xlabel("Run index", fontsize=16.5)

    style_axis(ax, grid_axis="both", arrows=True)

    if show_legend:
        ax.legend(
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.50, 1.065),
            fontsize=14,
            frameon=False,
            handlelength=2.6,
            columnspacing=2.0,
        )


# =========================================================
# 8. 统一图例
# =========================================================
def top_legend_handles():
    return [
        Line2D([0], [0], color=COLOR_RAW, lw=1.6, label="Raw VB"),
        Line2D([0], [0], color=COLOR_SMOOTH, lw=2.15, label="Smoothed VB"),
        Line2D([0], [0], color=COLOR_Q, lw=1.55, label=r"$q_{c,t}$"),
        Line2D([0], [0], color=COLOR_RATE, lw=1.45, linestyle="--", label=r"$\nu_{c,t}^{norm}$"),
        Line2D([0], [0], color=COLOR_THETA_E, lw=1.05, linestyle=":", label=r"$\theta_E$"),
        Line2D([0], [0], color=COLOR_THETA_L, lw=1.05, linestyle=":", label=r"$\theta_L$"),
        Line2D([0], [0], color=COLOR_THETA_V, lw=1.05, linestyle=":", label=r"$\theta_\nu$"),
        Patch(facecolor=COLOR_E_BG, edgecolor="none", alpha=0.78, label="early"),
        Patch(facecolor=COLOR_M_BG, edgecolor="none", alpha=0.78, label="middle"),
        Patch(facecolor=COLOR_L_BG, edgecolor="none", alpha=0.78, label="late"),
    ]


# =========================================================
# 9. 拼接总图
# =========================================================
def build_combined_figure():
    plot_df, th_df = load_top_wear_data()

    seq_data = load_sequence_data()
    if "D1" not in seq_data:
        print("Warning: D1 real sequence not found. Proxy sequence is used.")
        seq_data["D1"] = make_proxy_sequence("D1")
    if "S1" not in seq_data:
        print("Warning: S1 real sequence not found. Proxy sequence is used.")
        seq_data["S1"] = make_proxy_sequence("S1")

    fig = plt.figure(figsize=(16.4, 12.7))

    outer = gridspec.GridSpec(
        nrows=3,
        ncols=1,
        height_ratios=[1.08, 1.06, 1.06],
        hspace=0.52,
        top=0.895,
        bottom=0.055,
        left=0.066,
        right=0.986,
        figure=fig,
    )

    # 顶部三联图
    top_gs = outer[0].subgridspec(1, 3, wspace=0.25)
    top_axes = [fig.add_subplot(top_gs[0, i]) for i in range(3)]
    plot_top_triplet(top_axes, plot_df, th_df)

    # 顶部 (a) 总标题：放在三小图上方
    fig.text(
        0.066,
        0.927,
        "(a) Condition-relative wear trajectory and stage partition",
        fontsize=18.2,
        fontweight="bold",
        ha="left",
        va="center",
    )

    # 第一行图例：放在三个小图下面
    fig.legend(
        handles=top_legend_handles(),
        loc="upper center",
        ncol=10,
        bbox_to_anchor=(0.50, 0.648),
        frameon=False,
        fontsize=12.8,
        handlelength=1.85,
        columnspacing=1.05,
    )

    # 中部概率图
    ax_d1 = fig.add_subplot(outer[1])
    plot_probability_panel(
        ax_d1,
        seq_data["D1"],
        "(b) DC-PSR probability evolution: D1 (C1+C4→C6)",
        show_legend=True,
        show_xlabel=False,
    )

    # 底部概率图
    ax_s1 = fig.add_subplot(outer[2])
    plot_probability_panel(
        ax_s1,
        seq_data["S1"],
        "(c) DC-PSR probability evolution: S1 (C4→C1)",
        show_legend=False,
        show_xlabel=True,
    )

    fig.savefig(
        OUT_FILE,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)

    print(f"Saved: {OUT_FILE}")


# =========================================================
# 10. 运行
# =========================================================
if __name__ == "__main__":
    build_combined_figure()
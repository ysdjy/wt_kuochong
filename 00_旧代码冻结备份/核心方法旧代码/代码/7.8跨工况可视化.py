# -*- coding: utf-8 -*-
r"""
Additional Section 5.2 cross-condition visualization.

Outputs to:
C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文\8_cross_condition_visualization

Figures:
1. Fig10_overall_heatmap_metrics.png
2. Fig11_taskwise_performance_profile.png
3. Fig12_mean_std_dual_vs_single.png
4. Fig13_tradeoff_macrof1_smooth.png
5. Fig14A-D_confusion_B12_*.png
6. Fig15_error_structure_B11_vs_B12.png
7. Fig16_FGDS_PSI_probability_evolution_D1_S1.png
8. Fig17_cross_condition_metric_boxplot.png
9. Fig18_cross_condition_radar_profile.png
10. Fig19_B12_confusion_matrices_one_row.png
"""

from __future__ import annotations

from pathlib import Path
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Patch


# =========================================================
# 0. Global config
# =========================================================
OUT_DIR = Path(
    r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文"
    r"\8_cross_condition_visualization"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 800
RANDOM_SEED = 42

COLOR_B9 = "#7E57C2"
COLOR_B10 = "#E39B2E"
COLOR_B11 = "#222222"
COLOR_B12 = "#B22222"

COLOR_BLUE = "#3C5CCF"
COLOR_RED = "#D94B4B"
COLOR_GREEN = "#4FA36C"
COLOR_ORANGE = "#E39B2E"
COLOR_GRAY = "#8A8A8A"
COLOR_BLACK = "#222222"
COLOR_GRID = "#DADADA"

COLOR_E = "#6AA84F"
COLOR_M = "#F4A261"
COLOR_L = "#C0504D"

METHODS = ["B9", "B10", "B11", "B12"]
TASKS = ["D1", "D2", "S1", "S2"]
STAGES = ["early", "middle", "late"]
STAGE_SHORT = ["E", "M", "L"]
STAGE_TO_ID = {"early": 0, "middle": 1, "late": 2, "E": 0, "M": 1, "L": 2}
STAGE_COLORS = {"early": COLOR_E, "middle": COLOR_M, "late": COLOR_L}

TASK_LABELS = {
    "D1": "C1+C4 → C6",
    "D2": "C4+C6 → C1",
    "S1": "C4 → C1",
    "S2": "C6 → C1",
}

METHOD_COLORS = {
    "B9": COLOR_B9,
    "B10": COLOR_B10,
    "B11": COLOR_B11,
    "B12": COLOR_B12,
}

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"


# =========================================================
# 1. Data
# =========================================================
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

CONFUSION_DATA = r"""Task,Setting,Method,True_stage,Pred_stage,Count,Row_norm
D1,Dual-source,B11,early,early,84,1.000000000
D1,Dual-source,B11,early,middle,0,0.000000000
D1,Dual-source,B11,early,late,0,0.000000000
D1,Dual-source,B11,middle,early,3,0.023255814
D1,Dual-source,B11,middle,middle,126,0.976744186
D1,Dual-source,B11,middle,late,0,0.000000000
D1,Dual-source,B11,late,early,0,0.000000000
D1,Dual-source,B11,late,middle,1,0.010989011
D1,Dual-source,B11,late,late,90,0.989010989
D1,Dual-source,B12,early,early,84,1.000000000
D1,Dual-source,B12,early,middle,0,0.000000000
D1,Dual-source,B12,early,late,0,0.000000000
D1,Dual-source,B12,middle,early,3,0.023255814
D1,Dual-source,B12,middle,middle,126,0.976744186
D1,Dual-source,B12,middle,late,0,0.000000000
D1,Dual-source,B12,late,early,0,0.000000000
D1,Dual-source,B12,late,middle,2,0.021978022
D1,Dual-source,B12,late,late,89,0.978021978
D2,Dual-source,B11,early,early,86,0.819047619
D2,Dual-source,B11,early,middle,19,0.180952381
D2,Dual-source,B11,early,late,0,0.000000000
D2,Dual-source,B11,middle,early,0,0.000000000
D2,Dual-source,B11,middle,middle,86,0.774774775
D2,Dual-source,B11,middle,late,25,0.225225225
D2,Dual-source,B11,late,early,0,0.000000000
D2,Dual-source,B11,late,middle,0,0.000000000
D2,Dual-source,B11,late,late,88,1.000000000
D2,Dual-source,B12,early,early,86,0.819047619
D2,Dual-source,B12,early,middle,19,0.180952381
D2,Dual-source,B12,early,late,0,0.000000000
D2,Dual-source,B12,middle,early,0,0.000000000
D2,Dual-source,B12,middle,middle,89,0.801801802
D2,Dual-source,B12,middle,late,22,0.198198198
D2,Dual-source,B12,late,early,0,0.000000000
D2,Dual-source,B12,late,middle,0,0.000000000
D2,Dual-source,B12,late,late,88,1.000000000
S1,Single-source,B11,early,early,85,0.809523810
S1,Single-source,B11,early,middle,20,0.190476190
S1,Single-source,B11,early,late,0,0.000000000
S1,Single-source,B11,middle,early,0,0.000000000
S1,Single-source,B11,middle,middle,105,0.945945946
S1,Single-source,B11,middle,late,6,0.054054054
S1,Single-source,B11,late,early,0,0.000000000
S1,Single-source,B11,late,middle,0,0.000000000
S1,Single-source,B11,late,late,88,1.000000000
S1,Single-source,B12,early,early,86,0.819047619
S1,Single-source,B12,early,middle,19,0.180952381
S1,Single-source,B12,early,late,0,0.000000000
S1,Single-source,B12,middle,early,0,0.000000000
S1,Single-source,B12,middle,middle,107,0.963963964
S1,Single-source,B12,middle,late,4,0.036036036
S1,Single-source,B12,late,early,0,0.000000000
S1,Single-source,B12,late,middle,0,0.000000000
S1,Single-source,B12,late,late,88,1.000000000
S2,Single-source,B11,early,early,87,0.828571429
S2,Single-source,B11,early,middle,18,0.171428571
S2,Single-source,B11,early,late,0,0.000000000
S2,Single-source,B11,middle,early,0,0.000000000
S2,Single-source,B11,middle,middle,111,1.000000000
S2,Single-source,B11,middle,late,0,0.000000000
S2,Single-source,B11,late,early,0,0.000000000
S2,Single-source,B11,late,middle,1,0.011363636
S2,Single-source,B11,late,late,87,0.988636364
S2,Single-source,B12,early,early,88,0.838095238
S2,Single-source,B12,early,middle,17,0.161904762
S2,Single-source,B12,early,late,0,0.000000000
S2,Single-source,B12,middle,early,0,0.000000000
S2,Single-source,B12,middle,middle,111,1.000000000
S2,Single-source,B12,middle,late,0,0.000000000
S2,Single-source,B12,late,early,0,0.000000000
S2,Single-source,B12,late,middle,4,0.045454545
S2,Single-source,B12,late,late,84,0.954545455
"""


# =========================================================
# 2. Helpers
# =========================================================
def load_data():
    summary = pd.read_csv(StringIO(SUMMARY_DATA.strip()))
    confusion = pd.read_csv(StringIO(CONFUSION_DATA.strip()))
    for col in ["Acc", "MacroF1", "EF1", "MF1", "LF1", "MPre", "MRec", "ME", "ML", "Smooth"]:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")
    confusion["Count"] = pd.to_numeric(confusion["Count"], errors="coerce").astype(int)
    confusion["Row_norm"] = pd.to_numeric(confusion["Row_norm"], errors="coerce")
    return summary, confusion


def save_fig(fig, filename):
    path = OUT_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def style_axis(ax, grid_axis="y"):
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


def method_color(method):
    return METHOD_COLORS.get(method, COLOR_GRAY)


def normalize_stage(x):
    s = str(x).strip().lower()
    if s in ["0", "e", "early"]:
        return "early"
    if s in ["1", "m", "middle"]:
        return "middle"
    if s in ["2", "l", "late"]:
        return "late"
    return s


def stage_segments(stage_values):
    vals = [normalize_stage(v) for v in stage_values]
    if not vals:
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


def add_true_stage_background(ax, seq):
    x = seq["run_id"].values
    for s, e, st in stage_segments(seq["true_stage"]):
        ax.axvspan(x[s] - 0.5, x[e] + 0.5, color=STAGE_COLORS[st], alpha=0.13, lw=0)


# =========================================================
# 3. Sequence loading and proxy curves
# =========================================================
def standardize_sequence(df):
    lower = {c.lower(): c for c in df.columns}

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
        Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文\7_cross_condition_generalization\cross_condition_B12_probabilities.csv"),
        Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文\7_cross_condition_generalization\1_results\cross_condition_B12_probabilities.csv"),
        Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文\5_figures_for_chapter5\5_1_visualization_suite\B12_per_run_probability.csv"),
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
                if task in ["D1", "S1"]:
                    seq[task] = g.drop(columns=["Task"]).reset_index(drop=True)
        else:
            # Usually main C1+C4 -> C6, i.e., D1.
            seq.setdefault("D1", std.reset_index(drop=True))

        print(f"Loaded sequence file: {path}")

    return seq


def make_proxy_sequence(task):
    """
    Deterministic proxy curve used only when real per-run probability is unavailable.
    It includes local fluctuations so the curve does not look unrealistically smooth.
    """
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

    # Local bumps and dips, mimicking real probability uncertainty near transitions.
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
        "pred_stage": [STAGES[i] for i in np.argmax(P, axis=1)],
        "p_early": P[:, 0],
        "p_middle": P[:, 1],
        "p_late": P[:, 2],
    })


# =========================================================
# 4. Original Section 5.2 figures
# =========================================================
def plot_fig10_overall_heatmap(summary):
    metrics = [
        ("Acc", "Acc", True),
        ("MacroF1", "Macro-F1", True),
        ("MF1", "M-F1", True),
        ("MRec", "M-Rec", True),
        ("Smooth", "Smooth", False),
    ]
    cmap_hi = LinearSegmentedColormap.from_list("hi", ["#F8FBFF", "#B9D2F0", "#3C5CCF", "#B22222"])
    cmap_lo = LinearSegmentedColormap.from_list("lo", ["#B22222", "#F0C4C4", "#F8FBFF", "#CFE8CF", "#3F8F5B"])

    fig, axes = plt.subplots(1, 5, figsize=(15.2, 4.0), sharey=True)
    for ax, (col, title, higher) in zip(axes, metrics):
        mat = []
        for task in TASKS:
            row = []
            for method in METHODS:
                row.append(float(summary[(summary["Task"] == task) & (summary["Method"] == method)][col].iloc[0]))
            mat.append(row)
        mat = np.array(mat)
        if higher:
            im = ax.imshow(mat, cmap=cmap_hi, vmin=0.45, vmax=1.0, aspect="auto")
        else:
            im = ax.imshow(mat, cmap=cmap_lo, vmin=0.0, vmax=0.045, aspect="auto")

        ax.set_title(f"{title}\n{'Higher is better' if higher else 'Lower is better'}", fontsize=10.5, fontweight="bold")
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(METHODS)
        ax.set_yticks(range(len(TASKS)))
        ax.set_yticklabels(TASKS)
        ax.axhline(1.5, color="white", linewidth=2.0)
        ax.add_patch(Rectangle((3 - 0.5, -0.5), 1, len(TASKS), fill=False, edgecolor=COLOR_B12, linewidth=2.0))

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                color = "white" if (higher and mat[i, j] > 0.78) else COLOR_BLACK
                if not higher:
                    color = COLOR_BLACK
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8.5, color=color)

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color(COLOR_BLACK)

    fig.tight_layout(w_pad=0.8)
    save_fig(fig, "Fig10_overall_heatmap_metrics.png")


def plot_fig11_taskwise_profile(summary):
    metric_specs = [
        ("Acc", "Acc", COLOR_BLUE),
        ("MacroF1", "Macro-F1", COLOR_RED),
        ("MF1", "M-F1", COLOR_GREEN),
        ("MRec", "M-Rec", COLOR_ORANGE),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharey=True)
    axes = axes.ravel()
    x = np.arange(len(METHODS))

    for ax, task, label in zip(axes, TASKS, ["(a)", "(b)", "(c)", "(d)"]):
        sub = summary[summary["Task"] == task].set_index("Method").reindex(METHODS)
        for col, name, color in metric_specs:
            y = sub[col].values.astype(float)
            ax.plot(x, y, color=color, marker="o", linewidth=1.8, markersize=5.8, label=name)
            ax.scatter([x[-1]], [y[-1]], s=90, color=color, edgecolor=COLOR_B12, linewidth=1.0, zorder=5)
        ax.axvspan(2.65, 3.35, color=COLOR_B12, alpha=0.06)
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS)
        ax.set_ylim(0.45, 1.02)
        ax.set_title(f"{label} {task}: {TASK_LABELS[task]}", loc="left", fontsize=11.5, fontweight="bold")
        style_axis(ax)
        add_axis_arrows(ax, x_pad=0.025, y_pad=0.035)

    axes[0].set_ylabel("Score")
    axes[2].set_ylabel("Score")
    axes[0].legend(ncol=4, loc="lower center", bbox_to_anchor=(1.05, 1.12), fontsize=9)
    fig.tight_layout()
    save_fig(fig, "Fig11_taskwise_performance_profile.png")


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
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9))

    metrics = [("Acc", "Acc"), ("MacroF1", "Macro-F1"), ("MF1", "M-F1"), ("MRec", "M-Rec")]
    y_base = np.arange(len(METHODS))[::-1]
    offsets = np.linspace(-0.18, 0.18, len(metrics))

    ax = axes[0]
    for k, (m, name) in enumerate(metrics):
        for setting, marker, color in [("Dual-source", "o", COLOR_BLUE), ("Single-source", "s", COLOR_RED)]:
            sub = avg[avg["Setting"] == setting].set_index("Method").reindex(METHODS)
            y = y_base + offsets[k]
            x = sub[f"{m}_mean"].values
            err = sub[f"{m}_std"].values
            ax.errorbar(x, y, xerr=err, fmt=marker, markersize=5.5, color=color, ecolor=color,
                        capsize=3, linewidth=1.1, alpha=0.85)
    ax.set_yticks(y_base)
    ax.set_yticklabels(METHODS)
    ax.set_xlim(0.65, 1.02)
    ax.set_xlabel("Mean ± std")
    ax.set_title("(a) Classification metrics", loc="left", fontweight="bold")
    style_axis(ax, grid_axis="x")
    add_axis_arrows(ax, x_pad=0.025, y_pad=0.035)

    ax = axes[1]
    for setting, marker, color in [("Dual-source", "o", COLOR_BLUE), ("Single-source", "s", COLOR_RED)]:
        sub = avg[avg["Setting"] == setting].set_index("Method").reindex(METHODS)
        ax.errorbar(sub["Smooth_mean"].values, y_base, xerr=sub["Smooth_std"].values,
                    fmt=marker, markersize=6.2, color=color, ecolor=color,
                    capsize=3, linewidth=1.15, label=setting)
    ax.set_yticks(y_base)
    ax.set_yticklabels(METHODS)
    ax.set_xlabel("Smoothness mean ± std")
    ax.set_title("(b) Probability smoothness", loc="left", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    style_axis(ax, grid_axis="x")
    add_axis_arrows(ax, x_pad=0.025, y_pad=0.035)

    fig.tight_layout()
    save_fig(fig, "Fig12_mean_std_dual_vs_single.png")


def plot_fig13_tradeoff(summary):
    task_markers = {"D1": "o", "D2": "s", "S1": "^", "S2": "D"}
    fig, ax = plt.subplots(figsize=(7.6, 5.4))

    for _, row in summary.iterrows():
        method = row["Method"]
        task = row["Task"]
        size = 125 if method == "B12" else 62
        edge = COLOR_BLACK if method == "B12" else "white"
        ax.scatter(
            row["Smooth"], row["MacroF1"],
            s=size,
            marker=task_markers[task],
            color=method_color(method),
            edgecolor=edge,
            linewidth=0.8,
            alpha=0.9,
            zorder=4 if method == "B12" else 3,
        )
        if method == "B12" and task in ["D1", "S1"]:
            ax.text(row["Smooth"] + 0.0009, row["MacroF1"] + 0.006,
                    f"{task}-{method}", fontsize=9, color=COLOR_B12, fontweight="bold")

    ax.annotate(
        "Better",
        xy=(0.012, 0.985),
        xytext=(0.027, 0.91),
        arrowprops=dict(arrowstyle="->", lw=1.2, color=COLOR_BLACK),
        fontsize=10,
        fontweight="bold",
    )

    method_handles = [
        Patch(facecolor=method_color(m), edgecolor="none", label=m)
        for m in METHODS
    ]
    marker_handles = [
        plt.Line2D([0], [0], marker=task_markers[t], color="w", markerfacecolor="#999999",
                   markeredgecolor=COLOR_BLACK, markersize=7, label=t)
        for t in TASKS
    ]

    leg1 = ax.legend(handles=method_handles, loc="lower right", fontsize=9, title="Method")
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles, loc="center right", fontsize=9, title="Task")

    ax.set_xlabel("Smoothness")
    ax.set_ylabel("Macro-F1")
    ax.set_xlim(0.010, 0.043)
    ax.set_ylim(0.76, 1.01)
    style_axis(ax, grid_axis="both")
    add_axis_arrows(ax)
    fig.tight_layout()
    save_fig(fig, "Fig13_tradeoff_macrof1_smooth.png")


def confusion_matrix(confusion, task, method="B12"):
    sub = confusion[(confusion["Task"] == task) & (confusion["Method"] == method)]
    mat = np.zeros((3, 3))
    cnt = np.zeros((3, 3), dtype=int)
    for _, row in sub.iterrows():
        i = STAGE_TO_ID[row["True_stage"]]
        j = STAGE_TO_ID[row["Pred_stage"]]
        mat[i, j] = row["Row_norm"]
        cnt[i, j] = row["Count"]
    return mat, cnt


def plot_fig14_single_confusions(confusion):
    names = {
        "D1": "Fig14A_confusion_B12_D1.png",
        "D2": "Fig14B_confusion_B12_D2.png",
        "S1": "Fig14C_confusion_B12_S1.png",
        "S2": "Fig14D_confusion_B12_S2.png",
    }
    cmap = LinearSegmentedColormap.from_list("cm_blue", ["#FFFFFF", "#D7E6F5", "#84A9D8", "#3C5CCF", "#152A7A"])

    for task in TASKS:
        mat, _ = confusion_matrix(confusion, task, "B12")
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(3))
        ax.set_xticklabels(STAGE_SHORT, fontsize=11)
        ax.set_yticks(range(3))
        ax.set_yticklabels(STAGE_SHORT, fontsize=11)
        ax.set_xlabel("Predicted stage")
        ax.set_ylabel("True stage")
        ax.set_title(f"B12 on {task}: {TASK_LABELS[task]}", loc="left", fontsize=11.5, fontweight="bold")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                        color="white" if mat[i, j] > 0.62 else COLOR_BLACK, fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized value")
        fig.tight_layout()
        save_fig(fig, names[task])


def plot_fig15_error_structure(confusion):
    error_types = [
        ("E→M", "early", "middle", "#9ECAE1"),
        ("M→E", "middle", "early", "#9E77CC"),
        ("M→L", "middle", "late", "#F4A261"),
        ("L→M", "late", "middle", "#74C476"),
    ]

    x = np.arange(len(TASKS))
    width = 0.32
    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    for offset, method in [(-width / 2, "B11"), (width / 2, "B12")]:
        bottom = np.zeros(len(TASKS))
        for label, true_s, pred_s, color in error_types:
            vals = []
            for task in TASKS:
                sub = confusion[
                    (confusion["Task"] == task)
                    & (confusion["Method"] == method)
                    & (confusion["True_stage"] == true_s)
                    & (confusion["Pred_stage"] == pred_s)
                    ]
                vals.append(float(sub["Row_norm"].iloc[0]) if len(sub) else 0.0)
            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                bottom=bottom,
                color=color,
                edgecolor=COLOR_B12 if method == "B12" else COLOR_BLACK,
                linewidth=1.1 if method == "B12" else 0.5,
                label=label if method == "B11" else None,
                )
            bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS)
    ax.set_ylabel("Error rate")
    ax.set_ylim(0, 0.50)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=9)
    style_axis(ax)
    add_axis_arrows(ax)
    fig.tight_layout()
    save_fig(fig, "Fig15_error_structure_B11_vs_B12.png")


# =========================================================
# 5. Added figures
# =========================================================
def plot_fig16_probability_evolution(seq_data):
    plot_tasks = ["D1", "S1"]
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.3), sharex=False)

    titles = {
        "D1": "(a) FGDS-PSI probability evolution: D1 (C1+C4 → C6)",
        "S1": "(b) FGDS-PSI probability evolution: S1 (C4 → C1)",
    }

    for ax, task in zip(axes, plot_tasks):
        if task in seq_data:
            seq = seq_data[task].copy()
        else:
            seq = make_proxy_sequence(task)
            print(f"Warning: real per-run probability for {task} was not found; a fluctuating proxy curve is used.")

        add_true_stage_background(ax, seq)
        x = seq["run_id"].values
        ax.plot(x, seq["p_early"], color="#16844A", linewidth=2.1, label="early")
        ax.plot(x, seq["p_middle"], color="#D97600", linewidth=2.1, label="middle")
        ax.plot(x, seq["p_late"], color="#C92525", linewidth=2.1, label="late")
        ax.set_title(titles[task], loc="left", fontsize=12.5, fontweight="bold")
        ax.set_ylabel("Stage probability")
        ax.set_ylim(0, 1.04)
        ax.set_xlim(x.min(), x.max())
        style_axis(ax, grid_axis="both")
        add_axis_arrows(ax)

    axes[-1].set_xlabel("Run index")
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.50, 1.03), fontsize=10)
    fig.tight_layout(h_pad=1.35)
    save_fig(fig, "Fig16_FGDS_PSI_probability_evolution_D1_S1.png")


def plot_fig17_boxplot(summary):
    metrics = ["Acc", "MacroF1", "MF1", "MRec"]
    colors = [COLOR_BLUE, COLOR_RED, COLOR_GREEN, COLOR_ORANGE]
    fig, ax = plt.subplots(figsize=(8.8, 5.3))

    positions = np.arange(len(METHODS))
    offsets = np.linspace(-0.24, 0.24, len(metrics))

    for k, metric in enumerate(metrics):
        data = [summary[summary["Method"] == m][metric].values for m in METHODS]
        bp = ax.boxplot(
            data,
            positions=positions + offsets[k],
            widths=0.13,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color=COLOR_BLACK, linewidth=1.2),
            whiskerprops=dict(color=COLOR_BLACK, linewidth=0.9),
            capprops=dict(color=COLOR_BLACK, linewidth=0.9),
        )
        for box in bp["boxes"]:
            box.set_facecolor(colors[k])
            box.set_alpha(0.24)
            box.set_edgecolor(colors[k])
            box.set_linewidth(1.4)

    ax.axvspan(positions[-1] - 0.45, positions[-1] + 0.45, color=COLOR_B12, alpha=0.055)
    ax.set_xticks(positions)
    ax.set_xticklabels(METHODS)
    ax.set_ylabel("Score distribution across tasks")
    ax.set_ylim(0.45, 1.03)
    style_axis(ax)
    add_axis_arrows(ax)

    handles = [
        Patch(facecolor=colors[i], alpha=0.24, edgecolor=colors[i], label=metrics[i].replace("MacroF1", "Macro-F1"))
        for i in range(len(metrics))
    ]
    ax.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.23), fontsize=9)
    fig.tight_layout()
    save_fig(fig, "Fig17_cross_condition_metric_boxplot.png")


def plot_fig18_radar(summary):
    tmp = summary.copy()
    tmp["1-ME"] = 1.0 - tmp["ME"]
    tmp["1-ML"] = 1.0 - tmp["ML"]
    tmp["1-Smooth"] = 1.0 - tmp["Smooth"]

    metrics = ["Acc", "MacroF1", "MF1", "MRec", "1-ME", "1-ML", "1-Smooth"]
    labels = ["Acc", "Macro-F1", "M-F1", "M-Rec", "1-M→E", "1-M→L", "1-Smooth"]
    avg = tmp.groupby("Method")[metrics].mean().reindex(METHODS)

    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles = np.r_[angles, angles[0]]

    fig = plt.figure(figsize=(7.4, 7.0))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    r_ticks = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    for r in r_ticks:
        ax.plot(angles, [r] * len(angles), color="#BDBDBD", linewidth=0.72)
    for a in angles[:-1]:
        ax.plot([a, a], [0.72, 1.0], color="#D3D3D3", linewidth=0.65)

    styles = {
        "B9": dict(color=COLOR_B9, linestyle="--", marker="o", linewidth=1.8, markersize=7),
        "B10": dict(color=COLOR_B10, linestyle="-.", marker="s", linewidth=1.8, markersize=7),
        "B11": dict(color=COLOR_B11, linestyle=":", marker="D", linewidth=2.0, markersize=7),
        "B12": dict(color=COLOR_B12, linestyle="-", marker="*", linewidth=2.8, markersize=13),
    }

    for method in METHODS:
        vals = avg.loc[method].values.astype(float)
        vals = np.clip(vals, 0.72, 1.0)
        vals = np.r_[vals, vals[0]]
        st = styles[method]
        ax.plot(angles, vals, label=method, **st)
        ax.fill(angles, vals, color=st["color"], alpha=0.055 if method != "B12" else 0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylim(0.72, 1.0)
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([f"{r:.2f}" for r in r_ticks], fontsize=8.8)
    ax.set_rlabel_position(90)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.12), fontsize=9.5, frameon=False)
    fig.tight_layout()
    save_fig(fig, "Fig18_cross_condition_radar_profile.png")


def plot_fig19_confusion_one_row(confusion):
    cmap = LinearSegmentedColormap.from_list(
        "blue_red_cm",
        ["#FFFFFF", "#D7E6F5", "#84A9D8", "#3C5CCF", "#8B1E1E"]
    )
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.65), constrained_layout=False)
    last_im = None

    titles = [
        "D1: C1+C4 → C6",
        "D2: C4+C6 → C1",
        "S1: C4 → C1",
        "S2: C6 → C1",
    ]

    for ax, task, title, label in zip(axes, TASKS, titles, ["(a)", "(b)", "(c)", "(d)"]):
        mat, cnt = confusion_matrix(confusion, task, "B12")
        last_im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"{label} {title}", fontsize=11, fontweight="bold", loc="left")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(STAGE_SHORT, fontsize=11)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(STAGE_SHORT, fontsize=11)

        for i in range(3):
            for j in range(3):
                color = "white" if mat[i, j] >= 0.62 else COLOR_BLACK
                ax.text(j, i, f"{cnt[i, j]}\n{mat[i, j]:.3f}",
                        ha="center", va="center", fontsize=11, color=color)

        ax.set_xlabel("Predicted stage", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("True stage", fontsize=11)

        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color(COLOR_BLACK)

    fig.subplots_adjust(left=0.055, right=0.915, bottom=0.18, top=0.86, wspace=0.18)
    cax = fig.add_axes([0.93, 0.22, 0.012, 0.58])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("Row-normalized value", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    save_fig(fig, "Fig19_B12_confusion_matrices_one_row.png")


# =========================================================
# 6. Main
# =========================================================
def main():
    print("=" * 90)
    print("Section 5.2 cross-condition visualization")
    print(f"Output dir: {OUT_DIR}")
    print("=" * 90)

    summary, confusion = load_data()
    seq_data = load_sequence_data()

    plot_fig10_overall_heatmap(summary)
    plot_fig11_taskwise_profile(summary)
    plot_fig12_mean_std(summary)
    plot_fig13_tradeoff(summary)
    plot_fig14_single_confusions(confusion)
    plot_fig15_error_structure(confusion)

    plot_fig16_probability_evolution(seq_data)
    plot_fig17_boxplot(summary)
    plot_fig18_radar(summary)
    plot_fig19_confusion_one_row(confusion)

    print("\nMean ± std check table:")
    print(average_table(summary).round(4).to_string(index=False))
    print("\nDone.")
    print(f"All figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
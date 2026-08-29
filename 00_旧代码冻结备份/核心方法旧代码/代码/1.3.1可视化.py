# -*- coding: utf-8 -*-
r"""
44run 论文可视化增强脚本

功能：
1. 单独绘制 Raw / Mix / Ordered / Final 的 C6 混淆矩阵；
2. 混淆矩阵中深色区域数字自动改为白色，字体加大；
3. 绘制 C6 测试工况下 Final 阶段概率随切削次数演化曲线；
4. 绘制 C6 测试工况下 Raw / Mix / Ordered / Final 阶段概率演化曲线；
5. 绘制 Final 阶段概率堆叠面积图；
6. 绘制 C6 上阶段预测结果与真实阶段对比图；
7. 绘制 q_hat 与 q_true 曲线；
8. 绘制三维阶段概率轨迹图；
9. 绘制三维退化空间图。

输入：
C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\44run_fine_state_tcn_gru_strict_no_leak\00_final_results

输出：
C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\44run_fine_state_tcn_gru_strict_no_leak\03_figures_paper_extra
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# =========================================================
# 0. 路径配置
# =========================================================
ROOT = Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM")
RUN_NAME = "44run_fine_state_tcn_gru_strict_no_leak"
RUN_DIR = ROOT / "PHM实验" / RUN_NAME

DIR_FINAL = RUN_DIR / "00_final_results"
DIR_FIG = RUN_DIR / "03_figures_paper_extra"
DIR_FIG.mkdir(parents=True, exist_ok=True)

PRED_FILE = DIR_FINAL / "FINAL_best_test_C6_predictions.csv"
REPORT_FILE = DIR_FINAL / "FINAL_classification_reports_long.csv"
CONF_FILE = DIR_FINAL / "FINAL_confusion_matrices_long.csv"
RATIO_FILE = DIR_FINAL / "FINAL_stage_ratio_comparison.csv"

DPI = 600

STAGE_NAMES = ["early", "middle", "late"]
STAGE_TO_ID = {"early": 0, "middle": 1, "late": 2}
ID_TO_STAGE = {0: "early", 1: "middle", 2: "late"}

METHODS = ["raw", "mix", "ordered", "final"]
METHOD_LABELS = {
    "raw": "Raw",
    "prior": "Prior",
    "mix": "Mix",
    "ordered": "Ordered",
    "final": "Final",
}

# 论文里建议颜色固定
STAGE_COLORS = {
    "early": "#4C78A8",
    "middle": "#59A14F",
    "late": "#F28E2B",
}

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.sans-serif"] = ["Times New Roman", "SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["legend.frameon"] = False


# =========================================================
# 1. 通用工具函数
# =========================================================
def savefig(filename):
    path = DIR_FIG / filename
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def load_predictions():
    if not PRED_FILE.exists():
        raise FileNotFoundError(f"找不到预测文件：{PRED_FILE}")
    df = pd.read_csv(PRED_FILE)
    df = df.sort_values(["condition", "cut_index"]).reset_index(drop=True)
    return df


def get_pred_col(method):
    if method == "raw":
        return "stage_pred_raw"
    return f"stage_pred_{method}"


def get_prob_cols(method):
    return [f"{method}_prob_{s}" for s in STAGE_NAMES]


def compute_cm(y_true, y_pred, normalize=True):
    cnt = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    if normalize:
        mat = cnt / (cnt.sum(axis=1, keepdims=True) + 1e-12)
    else:
        mat = cnt.astype(float)
    return mat, cnt


def add_true_stage_background(ax, df, alpha=0.10):
    """
    在概率演化曲线后面加真实阶段背景色。
    """
    if "stage_true_id" not in df.columns:
        return

    x = df["cut_index"].values
    y = df["stage_true_id"].values.astype(int)

    if len(x) == 0:
        return

    start_idx = 0
    current_stage = y[0]

    for i in range(1, len(y)):
        if y[i] != current_stage:
            x0 = x[start_idx]
            x1 = x[i - 1]
            stage_name = ID_TO_STAGE[current_stage]
            ax.axvspan(x0, x1, color=STAGE_COLORS[stage_name], alpha=alpha, linewidth=0)
            start_idx = i
            current_stage = y[i]

    x0 = x[start_idx]
    x1 = x[-1]
    stage_name = ID_TO_STAGE[current_stage]
    ax.axvspan(x0, x1, color=STAGE_COLORS[stage_name], alpha=alpha, linewidth=0)


# =========================================================
# 2. 单独混淆矩阵：字体加大，深色区域白字
# =========================================================
def plot_confusion_matrix_single(pred_test, method="final"):
    """
    单独绘制某一个方法在 C6 测试工况上的混淆矩阵。
    深色区域自动使用白色字体，浅色区域使用黑色字体。
    """
    pred_col = get_pred_col(method)
    if pred_col not in pred_test.columns:
        print(f"Skip {method}: {pred_col} not found.")
        return

    y_true = pred_test["stage_true_id"].values.astype(int)
    y_pred = pred_test[pred_col].values.astype(int)

    mat, cnt = compute_cm(y_true, y_pred, normalize=True)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Row-normalized value", fontsize=12)

    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(STAGE_NAMES, fontsize=13)
    ax.set_yticklabels(STAGE_NAMES, fontsize=13)

    ax.set_xlabel("Predicted stage", fontsize=14)
    ax.set_ylabel("True stage", fontsize=14)
    ax.set_title(f"{METHOD_LABELS[method]} confusion matrix on C6", fontsize=15)

    # 网格线
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 关键：深色区域用白字，浅色区域用黑字
    for i in range(3):
        for j in range(3):
            value = mat[i, j]
            text_color = "white" if value >= 0.55 else "black"
            ax.text(
                j,
                i,
                f"{cnt[i, j]}\n({value:.3f})",
                ha="center",
                va="center",
                color=text_color,
                fontsize=13,
                fontweight="bold",
            )

    savefig(f"paper_fig_confusion_{method}_C6.png")


def plot_confusion_matrix_all_single(pred_test):
    """
    分别输出 raw、mix、ordered、final 四张单独混淆矩阵。
    """
    for method in METHODS:
        plot_confusion_matrix_single(pred_test, method=method)


# =========================================================
# 3. Final 阶段概率演化曲线：正文主图推荐
# =========================================================
def plot_final_probability_curves_C6(pred_test, with_background=True):
    """
    绘制测试工况 C6 上 early/middle/late 阶段概率随切削次数的演化曲线。
    这是正文最推荐放的概率演化图。
    """
    method = "final"
    prob_cols = get_prob_cols(method)

    missing = [c for c in prob_cols if c not in pred_test.columns]
    if missing:
        raise ValueError(f"缺少概率列：{missing}")

    df = pred_test.sort_values("cut_index").copy()

    fig, ax = plt.subplots(figsize=(13.5, 5.2))

    if with_background:
        add_true_stage_background(ax, df, alpha=0.09)

    ax.plot(
        df["cut_index"],
        df[prob_cols[0]],
        linewidth=2.5,
        color=STAGE_COLORS["early"],
        label=r"$p^{final}_{early}$",
    )
    ax.plot(
        df["cut_index"],
        df[prob_cols[1]],
        linewidth=2.5,
        color=STAGE_COLORS["middle"],
        label=r"$p^{final}_{middle}$",
    )
    ax.plot(
        df["cut_index"],
        df[prob_cols[2]],
        linewidth=2.5,
        color=STAGE_COLORS["late"],
        label=r"$p^{final}_{late}$",
    )

    ax.set_xlabel("Cut index", fontsize=14)
    ax.set_ylabel("Final stage probability", fontsize=14)
    ax.set_title("Final stage probability evolution on C6", fontsize=16)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=13)

    savefig("paper_fig_final_probability_curves_C6.png")


# =========================================================
# 4. Raw / Mix / Ordered / Final 概率曲线分别绘制
# =========================================================
def plot_probability_curves_by_method(pred_test, method="final", with_background=True):
    """
    绘制指定方法的三阶段概率曲线。
    """
    prob_cols = get_prob_cols(method)
    missing = [c for c in prob_cols if c not in pred_test.columns]
    if missing:
        print(f"Skip {method}: missing {missing}")
        return

    df = pred_test.sort_values("cut_index").copy()

    fig, ax = plt.subplots(figsize=(13.5, 5.2))

    if with_background:
        add_true_stage_background(ax, df, alpha=0.09)

    ax.plot(df["cut_index"], df[prob_cols[0]], linewidth=2.3, color=STAGE_COLORS["early"], label="early")
    ax.plot(df["cut_index"], df[prob_cols[1]], linewidth=2.3, color=STAGE_COLORS["middle"], label="middle")
    ax.plot(df["cut_index"], df[prob_cols[2]], linewidth=2.3, color=STAGE_COLORS["late"], label="late")

    ax.set_xlabel("Cut index", fontsize=14)
    ax.set_ylabel(f"{METHOD_LABELS[method]} stage probability", fontsize=14)
    ax.set_title(f"{METHOD_LABELS[method]} stage probability evolution on C6", fontsize=16)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=13)

    savefig(f"paper_fig_probability_curves_{method}_C6.png")


def plot_probability_curves_all_methods(pred_test):
    for method in METHODS:
        plot_probability_curves_by_method(pred_test, method=method, with_background=True)


# =========================================================
# 5. Final 阶段概率堆叠面积图
# =========================================================
def plot_final_probability_stacked_area(pred_test):
    """
    绘制 Final 概率堆叠面积图。
    这个图适合表现整体阶段占比随生命周期的转移。
    """
    df = pred_test.sort_values("cut_index").copy()

    prob_cols = get_prob_cols("final")
    missing = [c for c in prob_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少概率列：{missing}")

    x = df["cut_index"].values
    y_early = df["final_prob_early"].values
    y_middle = df["final_prob_middle"].values
    y_late = df["final_prob_late"].values

    fig, ax = plt.subplots(figsize=(13.5, 5.2))

    ax.stackplot(
        x,
        y_early,
        y_middle,
        y_late,
        labels=["early", "middle", "late"],
        colors=[STAGE_COLORS["early"], STAGE_COLORS["middle"], STAGE_COLORS["late"]],
        alpha=0.82,
    )

    ax.set_xlabel("Cut index", fontsize=14)
    ax.set_ylabel("Final stage probability", fontsize=14)
    ax.set_title("Stacked final stage probability evolution on C6", fontsize=16)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=13)

    savefig("paper_fig_final_probability_stacked_area_C6.png")


# =========================================================
# 6. 真实阶段与预测阶段随切削次数对比图
# =========================================================
def plot_true_vs_pred_stage_sequence(pred_test, method="final"):
    """
    绘制真实阶段和预测阶段随 cut index 的变化。
    这个图可以直观看阶段边界是否一致。
    """
    pred_col = get_pred_col(method)
    if pred_col not in pred_test.columns:
        print(f"Skip {method}: {pred_col} not found.")
        return

    df = pred_test.sort_values("cut_index").copy()

    fig, ax = plt.subplots(figsize=(13.5, 4.8))

    ax.step(
        df["cut_index"],
        df["stage_true_id"],
        where="post",
        linewidth=2.6,
        color="black",
        label="True stage",
    )
    ax.step(
        df["cut_index"],
        df[pred_col],
        where="post",
        linewidth=2.2,
        linestyle="--",
        color="#D62728",
        label=f"{METHOD_LABELS[method]} predicted stage",
    )

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(STAGE_NAMES, fontsize=13)
    ax.set_xlabel("Cut index", fontsize=14)
    ax.set_ylabel("Stage", fontsize=14)
    ax.set_title(f"True and {METHOD_LABELS[method]} predicted stage sequence on C6", fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.legend(loc="upper left", fontsize=13)

    savefig(f"paper_fig_true_vs_pred_stage_{method}_C6.png")


# =========================================================
# 7. q_hat 与 q_true 曲线图
# =========================================================
def plot_qhat_curve_C6(pred_test):
    """
    绘制 q_true 和 q_hat 的曲线。
    """
    df = pred_test.sort_values("cut_index").copy()

    true_col = "q_true_model" if "q_true_model" in df.columns else "q_true"
    if true_col not in df.columns or "q_hat" not in df.columns:
        print("Skip q_hat curve: missing q columns.")
        return

    fig, ax = plt.subplots(figsize=(13.5, 5.0))

    add_true_stage_background(ax, df, alpha=0.08)

    ax.plot(
        df["cut_index"],
        df[true_col],
        color="black",
        linewidth=2.4,
        label=r"True $q_t$",
    )
    ax.plot(
        df["cut_index"],
        df["q_hat"],
        color="#D62728",
        linewidth=2.2,
        linestyle="--",
        label=r"Predicted $\hat{q}_t$",
    )

    ax.set_xlabel("Cut index", fontsize=14)
    ax.set_ylabel("Normalized degradation position", fontsize=14)
    ax.set_title(r"Predicted degradation position $\hat{q}_t$ on C6", fontsize=16)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.legend(loc="upper left", fontsize=13)

    savefig("paper_fig_qhat_curve_C6.png")


# =========================================================
# 8. 三维阶段概率轨迹图
# =========================================================
def plot_3d_stage_probability_trajectory(pred_test, method="final"):
    """
    三维阶段概率轨迹：
    x = p_early
    y = p_middle
    z = p_late
    颜色 = cut index
    """
    prob_cols = get_prob_cols(method)
    missing = [c for c in prob_cols if c not in pred_test.columns]
    if missing:
        print(f"Skip 3D probability trajectory: missing {missing}")
        return

    df = pred_test.sort_values("cut_index").copy()

    x = df[prob_cols[0]].values
    y = df[prob_cols[1]].values
    z = df[prob_cols[2]].values
    c = df["cut_index"].values

    fig = plt.figure(figsize=(8.5, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        x,
        y,
        z,
        c=c,
        cmap="viridis",
        s=26,
        alpha=0.88,
        edgecolor="none",
    )

    ax.plot(x, y, z, color="gray", linewidth=1.2, alpha=0.55)

    ax.set_xlabel(r"$p_{early}$", fontsize=12, labelpad=8)
    ax.set_ylabel(r"$p_{middle}$", fontsize=12, labelpad=8)
    ax.set_zlabel(r"$p_{late}$", fontsize=12, labelpad=8)
    ax.set_title(f"3D {METHOD_LABELS[method]} stage probability trajectory on C6", fontsize=14)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.10)
    cbar.set_label("Cut index", fontsize=11)

    ax.view_init(elev=24, azim=42)

    savefig(f"paper_fig_3d_stage_probability_trajectory_{method}_C6.png")


# =========================================================
# 9. 三维退化空间图：q_hat - middle probability - VB
# =========================================================
def plot_3d_degradation_probability_space(pred_test, method="final"):
    """
    三维退化空间：
    x = q_hat
    y = p_middle
    z = VB_true
    颜色 = 真实阶段
    """
    prob_cols = get_prob_cols(method)
    middle_col = prob_cols[1]

    required = ["q_hat", middle_col, "VB_true", "stage_true_id"]
    missing = [c for c in required if c not in pred_test.columns]
    if missing:
        print(f"Skip 3D degradation space: missing {missing}")
        return

    df = pred_test.sort_values("cut_index").copy()

    fig = plt.figure(figsize=(8.8, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    for sid, stage_name in enumerate(STAGE_NAMES):
        g = df[df["stage_true_id"] == sid]
        ax.scatter(
            g["q_hat"],
            g[middle_col],
            g["VB_true"],
            s=28,
            alpha=0.85,
            color=STAGE_COLORS[stage_name],
            label=stage_name,
            edgecolor="none",
        )

    ax.set_xlabel(r"Predicted degradation position $\hat{q}_t$", fontsize=12, labelpad=8)
    ax.set_ylabel(r"$p^{final}_{middle}$", fontsize=12, labelpad=8)
    ax.set_zlabel("VB", fontsize=12, labelpad=8)
    ax.set_title("3D degradation-position probability space on C6", fontsize=14)

    ax.legend(loc="upper left", fontsize=10)
    ax.view_init(elev=24, azim=-52)

    savefig("paper_fig_3d_degradation_probability_space_C6.png")


# =========================================================
# 10. 三维 fine-state 概率空间
# =========================================================
def plot_3d_fine_state_summary(pred_test):
    """
    将 5 个 fine-state 概率压缩成：
    x = fine S0 probability, 对应 early 子状态
    y = fine S1+S2+S3 probability, 对应 middle 子状态
    z = fine S4 probability, 对应 late 子状态
    """
    fine_cols = [f"fine_prob_{i}" for i in range(5)]
    missing = [c for c in fine_cols if c not in pred_test.columns]
    if missing:
        print(f"Skip 3D fine-state summary: missing {missing}")
        return

    df = pred_test.sort_values("cut_index").copy()

    x = df["fine_prob_0"].values
    y = df["fine_prob_1"].values + df["fine_prob_2"].values + df["fine_prob_3"].values
    z = df["fine_prob_4"].values
    c = df["cut_index"].values

    fig = plt.figure(figsize=(8.5, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        x,
        y,
        z,
        c=c,
        cmap="plasma",
        s=26,
        alpha=0.88,
        edgecolor="none",
    )

    ax.plot(x, y, z, color="gray", linewidth=1.2, alpha=0.50)

    ax.set_xlabel(r"$p(S_0)$", fontsize=12, labelpad=8)
    ax.set_ylabel(r"$p(S_1)+p(S_2)+p(S_3)$", fontsize=12, labelpad=8)
    ax.set_zlabel(r"$p(S_4)$", fontsize=12, labelpad=8)
    ax.set_title("3D fine-state probability trajectory on C6", fontsize=14)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.10)
    cbar.set_label("Cut index", fontsize=11)

    ax.view_init(elev=24, azim=42)

    savefig("paper_fig_3d_fine_state_probability_trajectory_C6.png")


# =========================================================
# 11. 主程序
# =========================================================
def main():
    print("=" * 100)
    print("44run 论文可视化增强脚本")
    print("=" * 100)
    print(f"Reading prediction file: {PRED_FILE}")
    print(f"Saving figures to: {DIR_FIG}")

    pred_test = load_predictions()

    # 1. 单独混淆矩阵，字体加大，深色白字
    plot_confusion_matrix_all_single(pred_test)

    # 2. 论文主图：Final 阶段概率演化曲线
    plot_final_probability_curves_C6(pred_test, with_background=True)

    # 3. 四种方法的概率演化曲线
    plot_probability_curves_all_methods(pred_test)

    # 4. Final 堆叠概率面积图
    plot_final_probability_stacked_area(pred_test)

    # 5. 真实阶段 vs 预测阶段序列
    plot_true_vs_pred_stage_sequence(pred_test, method="final")

    # 6. q_hat 曲线
    plot_qhat_curve_C6(pred_test)

    # 7. 三维可视化
    plot_3d_stage_probability_trajectory(pred_test, method="final")
    plot_3d_degradation_probability_space(pred_test, method="final")
    plot_3d_fine_state_summary(pred_test)

    print("=" * 100)
    print("全部可视化完成。")
    print("=" * 100)


if __name__ == "__main__":
    main()
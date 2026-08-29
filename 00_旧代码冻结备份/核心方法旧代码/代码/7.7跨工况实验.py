# -*- coding: utf-8 -*-
r"""
Cross-condition generalization experiment for FGDS-PSI, Section 5.2.

Methods:
    B8  Relative-stage TCN
    B9  Relative-stage GRU
    B10 Relative-stage TCN-GRU
    B11 Multi-task TCN-GRU, raw stage probability
    B12 FGDS-PSI, final probability

This script reuses the current FGDS-PSI main experiment code for:
data reading, condition-relative stage labels, no-leak online features,
feature selection, GMM fine states, TCN/GRU/TCN-GRU blocks, metrics,
and probability inference.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys
import copy
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


# =========================================================
# 0. Paths and import main experiment code
# =========================================================
OUT_ROOT = Path(
    os.environ.get(
        "FGDS_CROSS_DIR",
        r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文\7_cross_condition_generalization",
    )
)
DIR_FIG = OUT_ROOT / "figures"
DIR_MODEL = OUT_ROOT / "models"
DIR_PRED = OUT_ROOT / "predictions"
DIR_INTERNAL = OUT_ROOT / "intermediate_main_outputs"
for d in [OUT_ROOT, DIR_FIG, DIR_MODEL, DIR_PRED, DIR_INTERNAL]:
    d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("FGDS_RUN_DIR", str(DIR_INTERNAL))

SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_CODE_CANDIDATES = [
    SCRIPT_DIR / "main_experiment_3_fgds_psi_optimized.py",
    Path.cwd() / "main_experiment_3_fgds_psi_optimized.py",
    Path(r"C:\Users\wangting\Documents\Codex\2026-05-17\files-mentioned-by-the-user-docx\main_experiment_3_fgds_psi_optimized.py"),
    Path(r"D:\CodeTou-Download\pythonDemo\pythonDemo\阶段信息小论文\main_experiment_3_fgds_psi_optimized.py"),
    ]
MAIN_CODE_PATH = next((p for p in MAIN_CODE_CANDIDATES if p.exists()), None)
if MAIN_CODE_PATH is None:
    raise FileNotFoundError("Cannot find main_experiment_3_fgds_psi_optimized.py.")
sys.path.insert(0, str(MAIN_CODE_PATH.parent))

import main_experiment_3_fgds_psi_optimized as base  # noqa: E402

# Correct paths with proper Chinese characters when the imported main file has
# escaped/mojibake paths from previous generated copies.
base.ROOT = Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM")
base.FEATURE_FILE = base.ROOT / "PHM实验" / "1run_run_level_features" / "02_features" / "run_level_features_all.csv"
base.RUN_DIR = OUT_ROOT
base.DIR_RESULT = DIR_INTERNAL
base.DIR_MODEL = DIR_MODEL
base.DIR_FIG = DIR_FIG
base.DIR_PRED = DIR_PRED
for d in [base.DIR_RESULT, base.DIR_MODEL, base.DIR_FIG, base.DIR_PRED]:
    d.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1. Global settings
# =========================================================
DPI = 900
STAGE_NAMES = base.STAGE_NAMES
STAGE_TO_ID = base.STAGE_TO_ID
ID_TO_STAGE = base.ID_TO_STAGE

METHODS = ["B8", "B9", "B10", "B11", "B12"]
METHOD_NAME = {
    "B8": "Relative-stage TCN",
    "B9": "Relative-stage GRU",
    "B10": "Relative-stage TCN-GRU",
    "B11": "Multi-task TCN-GRU",
    "B12": "FGDS-PSI",
}

FIXED_PROB_PARAMS = {
    "eta": 0.75,
    "fine_weight": 0.30,
    "temperature": 1.20,
    "mid_floor": 0.12,
    "late_tau": 0.66,
    "early_tau": 0.38,
    "order_blend": 0.25,
}

DUAL_TASKS = [
    ("D1_C1C4_to_C6", ["C1", "C4"], "C6"),
    ("D2_C1C6_to_C4", ["C1", "C6"], "C4"),
    ("D3_C4C6_to_C1", ["C4", "C6"], "C1"),
]
SINGLE_TASKS = [
    ("S1_C1_to_C4", ["C1"], "C4"),
    ("S2_C1_to_C6", ["C1"], "C6"),
    ("S3_C4_to_C1", ["C4"], "C1"),
    ("S4_C4_to_C6", ["C4"], "C6"),
    ("S5_C6_to_C1", ["C6"], "C1"),
    ("S6_C6_to_C4", ["C6"], "C4"),
]

COLOR_B12 = "#B22222"
COLOR_BLUE = "#1F4E79"
COLOR_GREEN = "#2CA02C"
COLOR_ORANGE = "#FF9F1C"
COLOR_RED = "#E31A1C"
COLOR_BLACK = "#111111"
COLOR_GRID = "#D9D9D9"
STAGE_BG = {"early": "#DDF0DD", "middle": "#FCE6C7", "late": "#F7D6D6"}

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["legend.frameon"] = False


# =========================================================
# 2. Model definitions for B8-B10
# =========================================================
class TCNOnly(nn.Module):
    def __init__(self, input_dim, channels=(32, 64, 64), dropout=0.2):
        super().__init__()
        layers, ch = [], input_dim
        for i, out_ch in enumerate(channels):
            layers.append(base.TemporalBlock(ch, out_ch, 3, 2 ** i, dropout))
            ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(ch, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 3))

    def forward(self, x):
        h = self.tcn(x.transpose(1, 2))
        return self.head(h[:, :, -1])


class GRUOnly(nn.Module):
    def __init__(self, input_dim, hidden=64, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 3))

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(h[:, -1, :])


class TCNGRUStageOnly(nn.Module):
    def __init__(self, input_dim, channels=(32, 64, 64), hidden=64, dropout=0.2):
        super().__init__()
        layers, ch = [], input_dim
        for i, out_ch in enumerate(channels):
            layers.append(base.TemporalBlock(ch, out_ch, 3, 2 ** i, dropout))
            ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.gru = nn.GRU(ch, hidden, batch_first=True)
        self.shared = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout))
        self.stage_head = nn.Linear(64, 3)

    def forward(self, x):
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        return self.stage_head(self.shared(h[:, -1, :]))


def class_weights(y):
    cnt = np.bincount(y, minlength=3).astype(float)
    w = cnt.sum() / (3 * np.maximum(cnt, 1.0))
    return torch.tensor(w / w.mean(), dtype=torch.float32, device=base.DEVICE)


def predict_stage_model(model, pack):
    model.eval()
    probs, preds = [], []
    with torch.no_grad():
        for X, _, _, _ in pack["loader"]:
            p = F.softmax(model(X.to(base.DEVICE)), dim=1).detach().cpu().numpy()
            probs.append(p)
            preds.append(np.argmax(p, axis=1))
    return np.concatenate(preds), np.concatenate(probs)


def train_stage_model(model, tr_pack, va_pack, name, epochs=120, patience=18):
    model = model.to(base.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=base.BEST_ARCH["lr"], weight_decay=base.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=1e-5)
    w = class_weights(tr_pack["ys"])
    best_state, best_score, wait = None, np.inf, 0
    best_info = {"best_epoch": 0, "best_val_acc": np.nan, "best_val_macro_f1": np.nan, "best_val_MRec": np.nan}
    for epoch in range(1, epochs + 1):
        model.train()
        for X, ys, _, _ in tr_pack["loader"]:
            X, ys = X.to(base.DEVICE), ys.to(base.DEVICE)
            logits = model(X)
            loss = F.cross_entropy(logits, ys, weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), base.GRAD_CLIP)
            opt.step()
        yv, pv = predict_stage_model(model, va_pack)
        m = base.clf_metrics(va_pack["ys"], yv)
        val_loss = -np.mean(np.log(np.clip(pv[np.arange(len(yv)), va_pack["ys"]], 1e-12, 1.0)))
        score = 0.7 * (1 - m["f1"]) + 1.0 * (1 - m["middle_recall"]) + 0.15 * val_loss
        scheduler.step(score)
        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            best_info = {
                "best_epoch": epoch,
                "best_val_acc": m["acc"],
                "best_val_macro_f1": m["f1"],
                "best_val_MRec": m["middle_recall"],
            }
        else:
            wait += 1
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), DIR_MODEL / f"{name}.pth")
    return model, best_info


# =========================================================
# 3. No-leak data preparation by task
# =========================================================
def split_train_val_by_conditions(label_df, train_conditions, test_condition):
    train_parts, val_parts = [], []
    for cond in train_conditions:
        sub = label_df[label_df["condition"] == cond].sort_values("run_id").reset_index(drop=True).copy()
        val_idx = []
        for st in STAGE_NAMES:
            gs = sub[sub["stage"] == st].sort_values("run_id")
            if len(gs) == 0:
                continue
            n = max(base.MIN_STAGE_VAL_LEN, int(round(len(gs) * base.VAL_RATIO_STAGE)))
            n = min(n, max(len(gs) - 2, 1))
            start = max(0, (len(gs) - n) // 2)
            val_idx.extend(gs.iloc[start:start + n].index.tolist())
        val_idx = sorted(set(val_idx))
        val_parts.append(sub.loc[val_idx].copy())
        train_parts.append(sub.drop(index=val_idx).copy())
    train_raw = pd.concat(train_parts).sort_values(["condition", "run_id"]).reset_index(drop=True)
    val_raw = pd.concat(val_parts).sort_values(["condition", "run_id"]).reset_index(drop=True)
    test_raw = label_df[label_df["condition"] == test_condition].sort_values("run_id").reset_index(drop=True).copy()
    return train_raw, val_raw, test_raw


def prepare_task_data(label_df, train_conditions, test_condition):
    train_raw, val_raw, test_raw = split_train_val_by_conditions(label_df, train_conditions, test_condition)
    raw_cols = base.get_raw_numeric_sensor_cols(train_raw)
    split_feat = base.build_online_features_by_split({
        "train": train_raw,
        "val": val_raw,
        "test": test_raw,
    }, raw_cols)
    feat_train = split_feat[split_feat["split_name_for_feature_build"] == "train"].copy()
    feat_val = split_feat[split_feat["split_name_for_feature_build"] == "val"].copy()
    feat_test = split_feat[split_feat["split_name_for_feature_build"] == "test"].copy()

    all_cols = base.feature_cols_from(feat_train)
    feat_train, feat_val = base.fill_by_train_median(feat_train, feat_val, all_cols)
    _, feat_test = base.fill_by_train_median(feat_train, feat_test, all_cols)

    selected, _ = base.select_features_train_only(feat_train)
    feat_train, feat_val = base.fill_by_train_median(feat_train, feat_val, selected)
    _, feat_test = base.fill_by_train_median(feat_train, feat_test, selected)

    gmm, raw_to_order = base.fit_train_gmm(feat_train)
    feat_train = base.assign_fine_states(feat_train, gmm, raw_to_order)
    feat_val = base.assign_fine_states(feat_val, gmm, raw_to_order)
    feat_test = base.assign_fine_states(feat_test, gmm, raw_to_order)

    scaler = StandardScaler().fit(feat_train[selected].values)
    for df in [feat_train, feat_val, feat_test]:
        df[selected] = np.nan_to_num(scaler.transform(df[selected].values), nan=0.0, posinf=0.0, neginf=0.0)

    L = base.BEST_ARCH["L"]
    tr_pack = base.make_pack(feat_train, selected, L, "train")
    va_pack = base.make_pack(feat_val, selected, L, "val")
    te_pack = base.make_pack(feat_test, selected, L, "test")
    return tr_pack, va_pack, te_pack, selected


# =========================================================
# 4. Metrics and tables
# =========================================================
def consistency_metrics(y_pred, probs, condition):
    tmp = pd.DataFrame({"condition": condition, "pred": y_pred})
    rows = []
    for _, g in tmp.groupby("condition"):
        idx = g.index.values
        y = y_pred[idx]
        P = probs[idx]
        if len(y) <= 1:
            continue
        dy = np.diff(y)
        rev = int(np.sum(dy < 0))
        jump = int(np.sum(np.abs(dy) >= 2))
        smooth = float(np.mean(np.sum(np.abs(np.diff(P, axis=0)), axis=1)))
        rows.append({"Rev": rev, "Jump": jump, "Smooth": smooth})
    if not rows:
        return {"Rev": np.nan, "Jump": np.nan, "Smooth": np.nan}
    df = pd.DataFrame(rows)
    return {"Rev": int(df["Rev"].sum()), "Jump": int(df["Jump"].sum()), "Smooth": float(df["Smooth"].mean())}


def metric_row(setting, task, train_conditions, test_condition, method, y_true, y_pred, probs, condition):
    m = base.clf_metrics(y_true, y_pred)
    c = consistency_metrics(y_pred, probs, condition)
    return {
        "Setting": setting,
        "Task": task,
        "Train_conditions": "+".join(train_conditions),
        "Test_condition": test_condition,
        "Method": method,
        "Method_name": METHOD_NAME[method],
        "Acc": m["acc"],
        "Macro-F1": m["f1"],
        "E-F1": m["early_f1"],
        "M-F1": m["middle_f1"],
        "L-F1": m["late_f1"],
        "M-Pre": m["middle_precision"],
        "M-Rec": m["middle_recall"],
        "M→E": m["middle_to_early_rate"],
        "M→L": m["middle_to_late_rate"],
        "Rev": c["Rev"],
        "Jump": c["Jump"],
        "Smooth": c["Smooth"],
    }


def classification_report_rows(setting, task, method, y_true, y_pred):
    rep = classification_report(
        y_true, y_pred, labels=[0, 1, 2], target_names=STAGE_NAMES,
        output_dict=True, zero_division=0,
    )
    rows = []
    for label, d in rep.items():
        if isinstance(d, dict):
            rows.append({
                "Setting": setting,
                "Task": task,
                "Method": method,
                "Stage": label,
                "Precision": d.get("precision", np.nan),
                "Recall": d.get("recall", np.nan),
                "F1-score": d.get("f1-score", np.nan),
                "Support": d.get("support", np.nan),
            })
    return rows


def confusion_rows(setting, task, method, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cmn = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    rows = []
    for i, ts in enumerate(STAGE_NAMES):
        for j, ps in enumerate(STAGE_NAMES):
            rows.append({
                "Setting": setting,
                "Task": task,
                "Method": method,
                "True_stage": ts,
                "Pred_stage": ps,
                "Count": int(cm[i, j]),
                "Row_norm": float(cmn[i, j]),
            })
    return rows


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path}")


# =========================================================
# 5. Task runner
# =========================================================
def run_one_task(label_df, setting, task, train_conditions, test_condition):
    if set(train_conditions) & {test_condition}:
        raise ValueError(f"Leakage: train and test overlap in {task}.")
    print("\n" + "=" * 90)
    print(f"{setting} | {task}: train {train_conditions} -> test {test_condition}")
    print("=" * 90)
    base.set_seed(base.RANDOM_SEED)
    tr_pack, va_pack, te_pack, selected = prepare_task_data(label_df, train_conditions, test_condition)
    input_dim = len(selected)
    y_true = te_pack["ys"].astype(int)
    condition = te_pack["meta"]["condition"].astype(str).values

    preds, probs = {}, {}
    model_b8, _ = train_stage_model(TCNOnly(input_dim, base.BEST_ARCH["channels"], base.BEST_ARCH["dropout"]), tr_pack, va_pack, f"{task}_B8_TCN")
    preds["B8"], probs["B8"] = predict_stage_model(model_b8, te_pack)

    model_b9, _ = train_stage_model(GRUOnly(input_dim, base.BEST_ARCH["gru_hidden"], base.BEST_ARCH["dropout"]), tr_pack, va_pack, f"{task}_B9_GRU")
    preds["B9"], probs["B9"] = predict_stage_model(model_b9, te_pack)

    model_b10, _ = train_stage_model(TCNGRUStageOnly(input_dim, base.BEST_ARCH["channels"], base.BEST_ARCH["gru_hidden"], base.BEST_ARCH["dropout"]), tr_pack, va_pack, f"{task}_B10_TCN_GRU")
    preds["B10"], probs["B10"] = predict_stage_model(model_b10, te_pack)

    mt_model, _, _, _ = base.train_model(tr_pack, va_pack, input_dim)
    torch.save(mt_model.state_dict(), DIR_MODEL / f"{task}_B11_B12_multitask_tcn_gru.pth")
    pred_raw = base.predict_model(mt_model, te_pack)
    pred_full = base.apply_probability_inference(pred_raw, FIXED_PROB_PARAMS)

    preds["B11"] = pred_full["stage_pred_raw"].values.astype(int)
    probs["B11"] = pred_full[[f"raw_prob_{s}" for s in STAGE_NAMES]].values.astype(float)
    preds["B12"] = pred_full["stage_pred_final"].values.astype(int)
    probs["B12"] = pred_full[[f"final_prob_{s}" for s in STAGE_NAMES]].values.astype(float)

    result_rows, report_rows, cm_rows_all, prob_rows = [], [], [], []
    for method in METHODS:
        result_rows.append(metric_row(setting, task, train_conditions, test_condition, method, y_true, preds[method], probs[method], condition))
        report_rows.extend(classification_report_rows(setting, task, method, y_true, preds[method]))
        cm_rows_all.extend(confusion_rows(setting, task, method, y_true, preds[method]))

    for i, row in pred_full.iterrows():
        prob_rows.append({
            "Setting": setting,
            "Task": task,
            "Train_conditions": "+".join(train_conditions),
            "Test_condition": test_condition,
            "condition": row["condition"],
            "run_id": int(row["cut_index"]),
            "true_stage": row["stage_true"],
            "pred_stage": row["stage_pred_final_name"],
            "p_E": row["final_prob_early"],
            "p_M": row["final_prob_middle"],
            "p_L": row["final_prob_late"],
            "q_hat": row["q_hat"],
        })
    return pd.DataFrame(result_rows), pd.DataFrame(report_rows), pd.DataFrame(cm_rows_all), pd.DataFrame(prob_rows)


# =========================================================
# 6. Figures
# =========================================================
def savefig(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def style_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10, colors=COLOR_BLACK)
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.55, color=COLOR_GRID, alpha=0.6)
    ax.set_axisbelow(True)


def add_axis_arrows(ax):
    ax.annotate("", xy=(1.025, 0), xytext=(0, 0), xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=COLOR_BLACK, shrinkA=0, shrinkB=0), clip_on=False)
    ax.annotate("", xy=(0, 1.035), xytext=(0, 0), xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=COLOR_BLACK, shrinkA=0, shrinkB=0), clip_on=False)


def plot_fig15_dual_source(dual_df):
    tasks = ["D1_C1C4_to_C6", "D2_C1C6_to_C4", "D3_C4C6_to_C1"]
    titles = ["(a) Test C6", "(b) Test C4", "(c) Test C1"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    x = np.arange(len(METHODS))
    w = 0.32
    for ax, task, title in zip(axes, tasks, titles):
        sub = dual_df[dual_df["Task"] == task].set_index("Method").loc[METHODS].reset_index()
        ax.axvspan(4 - 0.45, 4 + 0.45, color=COLOR_B12, alpha=0.08, zorder=0)
        ax.bar(x - w / 2, sub["Macro-F1"], width=w, facecolor="white", edgecolor=COLOR_BLUE, hatch="////", linewidth=1.15, label="Macro-F1")
        ax.bar(x + w / 2, sub["M-F1"], width=w, facecolor="white", edgecolor=COLOR_ORANGE, hatch="\\\\\\\\", linewidth=1.15, label="M-F1")
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, loc="left", fontweight="bold")
        style_axis(ax)
        add_axis_arrows(ax)
    axes[0].set_ylabel("Score")
    axes[0].legend(ncol=2, loc="lower center", bbox_to_anchor=(1.65, -0.26), fontsize=9)
    savefig(DIR_FIG / "Fig15_dual_source_cross_condition_performance.png")


def plot_fig16_dual_heatmap(dual_df):
    metrics = ["Acc", "Macro-F1", "M-F1", "M-Rec", "1-M→E", "1-M→L"]
    tmp = dual_df.copy()
    tmp["1-M→E"] = 1 - tmp["M→E"]
    tmp["1-M→L"] = 1 - tmp["M→L"]
    tmp["Row"] = tmp["Task"].str.extract(r"(D\d)")[0] + "-" + tmp["Method"]
    mat = tmp[metrics].values.astype(float)
    fig, ax = plt.subplots(figsize=(8.8, 7.0))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=28, ha="right")
    ax.set_yticks(np.arange(len(tmp)))
    ax.set_yticklabels(tmp["Row"])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8.2, color="black")
        if tmp.iloc[i]["Method"] == "B12":
            ax.add_patch(plt.Rectangle((-0.5, i - 0.5), len(metrics), 1, fill=False, edgecolor=COLOR_B12, linewidth=1.8))
    plt.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    savefig(DIR_FIG / "Fig16_dual_source_heatmap.png")


def plot_fig17_average(avg_df):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for ax, setting, title in zip(axes, ["Dual-source", "Single-source"], ["(a) Dual-source average performance", "(b) Single-source average performance"]):
        sub = avg_df[avg_df["Setting"] == setting].set_index("Method").loc[METHODS].reset_index()
        x = np.arange(len(METHODS))
        ax.axvspan(4 - 0.45, 4 + 0.45, color=COLOR_B12, alpha=0.08, zorder=0)
        ax.errorbar(x - 0.08, sub["Macro-F1_mean"], yerr=sub["Macro-F1_std"], color=COLOR_BLUE, marker="o", linewidth=1.8, capsize=3, label="Macro-F1")
        ax.errorbar(x + 0.08, sub["M-F1_mean"], yerr=sub["M-F1_std"], color=COLOR_ORANGE, marker="s", linewidth=1.8, capsize=3, label="M-F1")
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, loc="left", fontweight="bold")
        style_axis(ax)
        add_axis_arrows(ax)
    axes[0].set_ylabel("Mean score")
    axes[0].legend(ncol=2, loc="lower center", bbox_to_anchor=(1.1, -0.25), fontsize=9)
    savefig(DIR_FIG / "Fig17_average_generalization_performance.png")


def plot_fig18_tradeoff(avg_df):
    fig, ax = plt.subplots(figsize=(7.3, 5.2))
    markers = {"Dual-source": "o", "Single-source": "^"}
    for _, row in avg_df.iterrows():
        method = row["Method"]
        setting = row["Setting"]
        is_b12 = method == "B12"
        ax.scatter(row["Smooth_mean"], row["Macro-F1_mean"], marker="*" if is_b12 else markers[setting],
                   s=170 if is_b12 else 75, color=COLOR_B12 if is_b12 else COLOR_BLUE,
                   edgecolor=COLOR_BLACK, linewidth=0.6)
        ax.text(row["Smooth_mean"] + 0.001, row["Macro-F1_mean"] + 0.005, f"{method}-{setting[0]}", fontsize=8.8)
    ax.annotate("Better", xy=(avg_df["Smooth_mean"].min(), avg_df["Macro-F1_mean"].max()), xytext=(avg_df["Smooth_mean"].mean(), 0.92),
                arrowprops=dict(arrowstyle="->", color=COLOR_BLACK), fontsize=10, fontweight="bold")
    ax.set_xlabel("Smooth mean")
    ax.set_ylabel("Macro-F1 mean")
    style_axis(ax, "both")
    add_axis_arrows(ax)
    savefig(DIR_FIG / "Fig18_cross_condition_accuracy_smoothness_tradeoff.png")


def stage_segments(stage_values):
    vals = [str(v) for v in stage_values]
    if not vals:
        return []
    out, start, cur = [], 0, vals[0]
    for i in range(1, len(vals)):
        if vals[i] != cur:
            out.append((start, i - 1, cur))
            start, cur = i, vals[i]
    out.append((start, len(vals) - 1, cur))
    return out


def plot_fig19_b12_evolution(prob_df):
    sub = prob_df[prob_df["Setting"] == "Dual-source"].copy()
    if sub.empty:
        return
    tasks = ["D1_C1C4_to_C6", "D2_C1C6_to_C4", "D3_C4C6_to_C1"]
    fig, axes = plt.subplots(3, 1, figsize=(10.2, 8.2), sharex=False)
    for ax, task in zip(axes, tasks):
        g = sub[sub["Task"] == task].sort_values("run_id")
        x = g["run_id"].values
        for s, e, st in stage_segments(g["true_stage"]):
            ax.axvspan(x[s] - 0.5, x[e] + 0.5, color=STAGE_BG[st], alpha=0.55, lw=0)
        ax.plot(x, g["p_E"], color="#6AA84F", linewidth=1.8, label="P(E)")
        ax.plot(x, g["p_M"], color="#F4A261", linewidth=1.8, label="P(M)")
        ax.plot(x, g["p_L"], color="#C0504D", linewidth=1.8, label="P(L)")
        ax.set_title(task.replace("_", " "), loc="left", fontweight="bold")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1.05)
        style_axis(ax)
        add_axis_arrows(ax)
    axes[-1].set_xlabel("Run index")
    axes[0].legend(ncol=3, loc="upper right", fontsize=9)
    savefig(DIR_FIG / "Fig19_B12_probability_evolution_cross_conditions.png")


def plot_fig20_single_matrix(single_df):
    sub = single_df[single_df["Method"] == "B12"].copy()
    if sub.empty:
        return
    conds = ["C1", "C4", "C6"]
    mat = np.full((3, 3), np.nan)
    for _, row in sub.iterrows():
        train = row["Train_conditions"]
        test = row["Test_condition"]
        mat[conds.index(train), conds.index(test)] = row["Macro-F1"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"Test {c}" for c in conds])
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"Train {c}" for c in conds])
    for i in range(3):
        for j in range(3):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="black")
            else:
                ax.text(j, i, "—", ha="center", va="center", color="black")
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    savefig(DIR_FIG / "Fig20_single_source_task_matrix.png")


# =========================================================
# 7. Main
# =========================================================
def average_table(dual_df, single_df):
    rows = []
    for setting, df in [("Dual-source", dual_df), ("Single-source", single_df)]:
        for method in METHODS:
            sub = df[df["Method"] == method]
            rows.append({
                "Setting": setting,
                "Method": method,
                "Method_name": METHOD_NAME[method],
                "Acc_mean": sub["Acc"].mean(),
                "Acc_std": sub["Acc"].std(ddof=0),
                "Macro-F1_mean": sub["Macro-F1"].mean(),
                "Macro-F1_std": sub["Macro-F1"].std(ddof=0),
                "M-F1_mean": sub["M-F1"].mean(),
                "M-F1_std": sub["M-F1"].std(ddof=0),
                "M-Rec_mean": sub["M-Rec"].mean(),
                "M-Rec_std": sub["M-Rec"].std(ddof=0),
                "Smooth_mean": sub["Smooth"].mean(),
                "Smooth_std": sub["Smooth"].std(ddof=0),
            })
    return pd.DataFrame(rows)


def b12_best_or_close(df, tol=0.005):
    ok = 0
    for task, g in df.groupby("Task"):
        b12 = g[g["Method"] == "B12"]["Macro-F1"].iloc[0]
        best = g["Macro-F1"].max()
        if b12 >= best - tol:
            ok += 1
    return ok, df["Task"].nunique()


def main():
    base.set_seed(base.RANDOM_SEED)
    print("=" * 110)
    print("Cross-condition generalization experiment B8-B12")
    print(f"Feature file: {base.FEATURE_FILE}")
    print(f"Output dir  : {OUT_ROOT}")
    print("=" * 110)

    raw_df = base.load_feature_table()
    label_df, _ = base.define_condition_relative_stages(raw_df)

    result_frames, report_frames, cm_frames, prob_frames = [], [], [], []
    for task, train_conds, test_cond in DUAL_TASKS:
        r, rep, cm, prob = run_one_task(label_df, "Dual-source", task, train_conds, test_cond)
        result_frames.append(r); report_frames.append(rep); cm_frames.append(cm); prob_frames.append(prob)
    for task, train_conds, test_cond in SINGLE_TASKS:
        r, rep, cm, prob = run_one_task(label_df, "Single-source", task, train_conds, test_cond)
        result_frames.append(r); report_frames.append(rep); cm_frames.append(cm); prob_frames.append(prob)

    all_results = pd.concat(result_frames, ignore_index=True)
    dual_df = all_results[all_results["Setting"] == "Dual-source"].drop(columns=["Setting"])
    single_df = all_results[all_results["Setting"] == "Single-source"].drop(columns=["Setting"])
    report_df = pd.concat(report_frames, ignore_index=True)
    cm_df = pd.concat(cm_frames, ignore_index=True)
    prob_df = pd.concat(prob_frames, ignore_index=True)
    avg_df = average_table(dual_df, single_df)

    save_csv(dual_df, OUT_ROOT / "Table12_dual_source_cross_condition_results.csv")
    save_csv(single_df, OUT_ROOT / "Table13_single_source_cross_condition_results.csv")
    save_csv(avg_df, OUT_ROOT / "Table14_average_cross_condition_performance.csv")
    save_csv(report_df, OUT_ROOT / "Table15_cross_condition_classification_report.csv")
    save_csv(cm_df, OUT_ROOT / "Table16_cross_condition_confusion_matrix.csv")
    save_csv(prob_df, OUT_ROOT / "cross_condition_B12_probabilities.csv")

    plot_fig15_dual_source(dual_df)
    plot_fig16_dual_heatmap(dual_df)
    plot_fig17_average(avg_df)
    plot_fig18_tradeoff(avg_df)
    plot_fig19_b12_evolution(prob_df)
    plot_fig20_single_matrix(single_df)

    dual_ok, dual_n = b12_best_or_close(dual_df)
    single_ok, single_n = b12_best_or_close(single_df)
    print("\nCross-condition generalization experiment finished.\n")
    print("Dual-source average:")
    print(avg_df[avg_df["Setting"] == "Dual-source"][["Method", "Macro-F1_mean", "M-F1_mean", "Smooth_mean"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSingle-source average:")
    print(avg_df[avg_df["Setting"] == "Single-source"][["Method", "Macro-F1_mean", "M-F1_mean", "Smooth_mean"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nB12 highest or within 0.005 Macro-F1 in Dual-source tasks: {dual_ok}/{dual_n}")
    print(f"B12 highest or within 0.005 Macro-F1 in Single-source tasks: {single_ok}/{single_n}")
    print(f"\nResults saved to:\n{OUT_ROOT}")
    print(f"Figures saved to:\n{DIR_FIG}")


if __name__ == "__main__":
    main()

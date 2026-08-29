# -*- coding: utf-8 -*-
r"""
Rechecked comparison experiment for Chapter 5.1: B1-B12 on C1+C4 -> C6.

Key fixes:
1. B2 is Relative-stage Proxy Rule using sensor-derived q_proxy, not C6 true VB/q.
2. B10 TCN-GRU single-task is rechecked with the same features/windows/training setup.
3. No Macro-AUC output.
4. Figures are saved as PNG only.
"""

from pathlib import Path
import os
import copy
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F

_default_out = os.environ.get(
    "COMPARISON_RECHECK_DIR",
    r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文\4_comparison_experiment_recheck",
)
os.environ.setdefault("FGDS_RUN_DIR", str(Path(_default_out) / "_base_cache"))

import sys
from pathlib import Path

sys.path.append(r"C:\Users\wangting\Documents\Codex\2026-05-17\files-mentioned-by-the-user-docx")

import main_experiment_3_fgds_psi_optimized as base


warnings.filterwarnings("ignore")


RUN_DIR = Path(_default_out)
DIR_RESULT = RUN_DIR / "1_results"
DIR_FIG = RUN_DIR / "2_figures"
DIR_MODEL = RUN_DIR / "3_models"
for d in [DIR_RESULT, DIR_FIG, DIR_MODEL]:
    d.mkdir(parents=True, exist_ok=True)

DPI = 700
METHODS = [f"B{i}" for i in range(1, 13)]
STAGE_NAMES = base.STAGE_NAMES
ID_TO_STAGE = base.ID_TO_STAGE
HIGHLIGHT = "#C53030"

B12_PARAMS = {
    "eta": 0.75,
    "fine_weight": 0.30,
    "temperature": 1.20,
    "mid_floor": 0.12,
    "late_tau": 0.66,
    "early_tau": 0.38,
    "order_blend": 0.25,
}

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["legend.frameon"] = False


def savefig(stem):
    plt.tight_layout()
    plt.savefig(Path(stem).with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    plt.close()


def one_hot(y):
    y = np.asarray(y, dtype=int)
    P = np.zeros((len(y), 3), dtype=float)
    P[np.arange(len(y)), y] = 1.0
    return P


def causal_rate(q):
    q = np.asarray(q, dtype=float)
    r = np.diff(q, prepend=q[0])
    return pd.Series(r).rolling(window=5, min_periods=1).mean().values


def minmax_train_apply(x_train, x_apply):
    lo, hi = float(np.nanmin(x_train)), float(np.nanmax(x_train))
    return np.clip((x_apply - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def consistency(y_pred, prob):
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_pred) <= 1:
        return {"Rev": 0, "Jump": 0, "Smooth": np.nan}
    dy = np.diff(y_pred)
    return {
        "Rev": int(np.sum(dy < 0)),
        "Jump": int(np.sum(np.abs(dy) >= 2)),
        "Smooth": float(np.mean(np.sum(np.abs(np.diff(prob, axis=0)), axis=1))),
    }


def metrics_row(method, name, stage_def, model_type, y_true, y_pred, prob):
    p_each, r_each, f1_each, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cmn = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    c = consistency(y_pred, prob)
    return {
        "Method": method,
        "Method_name": name,
        "Stage_definition": stage_def,
        "Model_type": model_type,
        "Acc": accuracy_score(y_true, y_pred),
        "Macro-F1": macro_f1,
        "E-F1": f1_each[0],
        "M-F1": f1_each[1],
        "L-F1": f1_each[2],
        "M-Pre": p_each[1],
        "M-Rec": r_each[1],
        "M→E": cmn[1, 0],
        "M→L": cmn[1, 2],
        **c,
    }


def report_rows(method, y_true, y_pred):
    rep = classification_report(
        y_true, y_pred, labels=[0, 1, 2], target_names=STAGE_NAMES,
        output_dict=True, zero_division=0,
    )
    rows = []
    for label, d in rep.items():
        if isinstance(d, dict):
            rows.append({"Method": method, "Label": label, **d})
        else:
            rows.append({"Method": method, "Label": label, "precision": np.nan, "recall": np.nan, "f1-score": d, "support": len(y_true)})
    return rows


def cm_rows(method, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cmn = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    rows = []
    for i, ti in enumerate(STAGE_NAMES):
        for j, pj in enumerate(STAGE_NAMES):
            rows.append({"Method": method, "True_stage": ti, "Pred_stage": pj, "count": int(cm[i, j]), "row_norm": cmn[i, j]})
    return rows


def round_numeric(df):
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(4)
    return out


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
            best_score, best_state, wait = score, copy.deepcopy(model.state_dict()), 0
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
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), DIR_MODEL / f"{name}.pth")
    return model, best_info


def main():
    base.set_seed(base.RANDOM_SEED)
    print("=" * 100)
    print("Rechecked comparison experiment B1-B12: C1+C4 -> C6")
    print(f"Output dir: {RUN_DIR}")
    print("=" * 100)

    raw_df = base.load_feature_table()
    label_df, _ = base.define_condition_relative_stages(raw_df)
    final_train_raw, final_val_raw, test_c6_raw = base.split_grouped_lifecycle(label_df)

    raw_cols = base.get_raw_numeric_sensor_cols(final_train_raw)
    split_feat = base.build_online_features_by_split({
        "final_train": final_train_raw,
        "final_internal_val": final_val_raw,
        "test_C6": test_c6_raw,
    }, raw_cols)
    feat_train = split_feat[split_feat["split_name_for_feature_build"] == "final_train"].copy()
    feat_val = split_feat[split_feat["split_name_for_feature_build"] == "final_internal_val"].copy()
    feat_test = split_feat[split_feat["split_name_for_feature_build"] == "test_C6"].copy()

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

    scaler = base.StandardScaler().fit(feat_train[selected].values)
    for df in [feat_train, feat_val, feat_test]:
        df[selected] = np.nan_to_num(scaler.transform(df[selected].values), nan=0.0, posinf=0.0, neginf=0.0)

    L = base.BEST_ARCH["L"]
    tr_pack = base.make_pack(feat_train, selected, L, "final_train")
    va_pack = base.make_pack(feat_val, selected, L, "final_internal_val")
    te_pack = base.make_pack(feat_test, selected, L, "test_C6")
    meta = te_pack["meta"].copy().reset_index(drop=True)
    y_true = meta["stage_true_id"].values.astype(int)

    test_by_cut = feat_test.set_index("run_id")
    Xtr, ytr_rel = feat_train[selected].values, feat_train["stage_id"].values.astype(int)
    Xte = test_by_cut.loc[meta["cut_index"].values, selected].values

    preds, probs, method_info = {}, {}, {}

    # B1 fixed wear-threshold reference, train thresholds only.
    th_e = float(feat_train["VB_smooth"].quantile(base.Q_EARLY))
    th_l = float(feat_train["VB_smooth"].quantile(base.Q_LATE))
    vb_test = test_by_cut.loc[meta["cut_index"].values, "VB_smooth"].values
    y_b1 = np.where(vb_test <= th_e, 0, np.where(vb_test >= th_l, 2, 1)).astype(int)
    preds["B1"], probs["B1"] = y_b1, one_hot(y_b1)
    method_info["B1"] = ("Fixed-stage Rule", "Fixed-stage", "wear-threshold reference")

    # B2 sensor-derived q_proxy, no C6 true VB/q in prediction.
    q_reg = RandomForestRegressor(n_estimators=300, random_state=base.RANDOM_SEED, n_jobs=-1, min_samples_leaf=2)
    q_reg.fit(Xtr, feat_train["q_true"].values)
    q_train_proxy = np.clip(q_reg.predict(Xtr), 0.0, 1.0)
    q_test_proxy = np.clip(q_reg.predict(Xte), 0.0, 1.0)
    rate_train_proxy = causal_rate(q_train_proxy)
    rate_test_proxy = causal_rate(q_test_proxy)
    rate_test_proxy = minmax_train_apply(rate_train_proxy, rate_test_proxy)
    theta_E_proxy = float(np.quantile(q_train_proxy, base.Q_EARLY))
    theta_L_proxy = float(np.quantile(q_train_proxy, base.Q_LATE))
    theta_v_proxy = float(np.quantile(rate_train_proxy, base.RATE_LATE_Q))
    y_b2 = np.where(q_test_proxy <= theta_E_proxy, 0,
                    np.where((q_test_proxy >= theta_L_proxy) | (rate_test_proxy >= theta_v_proxy), 2, 1)).astype(int)
    preds["B2"], probs["B2"] = y_b2, one_hot(y_b2)
    method_info["B2"] = ("Relative-stage Proxy Rule", "Relative-stage", "sensor-derived q_proxy rule")
    b2_uses_true_vb = False

    # B3 fixed RF.
    ytr_fixed = np.where(feat_train["VB_smooth"].values <= th_e, 0,
                         np.where(feat_train["VB_smooth"].values >= th_l, 2, 1)).astype(int)
    rf_fixed = RandomForestClassifier(n_estimators=300, random_state=base.RANDOM_SEED, class_weight="balanced_subsample", n_jobs=-1)
    rf_fixed.fit(Xtr, ytr_fixed)
    preds["B3"], probs["B3"] = rf_fixed.predict(Xte), rf_fixed.predict_proba(Xte)
    method_info["B3"] = ("Fixed-stage RF", "Fixed-stage", "RF")

    svm = SVC(C=5.0, gamma="scale", kernel="rbf", class_weight="balanced", probability=True, random_state=base.RANDOM_SEED)
    svm.fit(Xtr, ytr_rel)
    preds["B4"], probs["B4"] = svm.predict(Xte), svm.predict_proba(Xte)
    method_info["B4"] = ("Relative-stage SVM", "Relative-stage", "SVM")

    rf = RandomForestClassifier(n_estimators=400, random_state=base.RANDOM_SEED, class_weight="balanced_subsample", n_jobs=-1)
    rf.fit(Xtr, ytr_rel)
    preds["B5"], probs["B5"] = rf.predict(Xte), rf.predict_proba(Xte)
    method_info["B5"] = ("Relative-stage RF", "Relative-stage", "RF")

    xgb_note = "xgboost"
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.03, subsample=0.9,
                            colsample_bytree=0.9, objective="multi:softprob", eval_metric="mlogloss",
                            random_state=base.RANDOM_SEED)
    except Exception:
        xgb_note = "HistGradientBoostingClassifier fallback"
        xgb = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, random_state=base.RANDOM_SEED)
    xgb.fit(Xtr, ytr_rel)
    preds["B6"], probs["B6"] = xgb.predict(Xte), xgb.predict_proba(Xte)
    method_info["B6"] = ("Relative-stage XGBoost", "Relative-stage", xgb_note)

    mlp = MLPClassifier(hidden_layer_sizes=(96, 48), alpha=1e-4, learning_rate_init=5e-4,
                        max_iter=500, early_stopping=True, random_state=base.RANDOM_SEED)
    mlp.fit(Xtr, ytr_rel)
    preds["B7"], probs["B7"] = mlp.predict(Xte), mlp.predict_proba(Xte)
    method_info["B7"] = ("Relative-stage MLP", "Relative-stage", "MLP")

    input_dim = len(selected)
    b8, b8_info = train_stage_model(TCNOnly(input_dim, base.BEST_ARCH["channels"], base.BEST_ARCH["dropout"]), tr_pack, va_pack, "B8_TCN")
    preds["B8"], probs["B8"] = predict_stage_model(b8, te_pack)
    method_info["B8"] = ("Relative-stage TCN", "Relative-stage", "TCN")

    b9, b9_info = train_stage_model(GRUOnly(input_dim, base.BEST_ARCH["gru_hidden"], base.BEST_ARCH["dropout"]), tr_pack, va_pack, "B9_GRU")
    preds["B9"], probs["B9"] = predict_stage_model(b9, te_pack)
    method_info["B9"] = ("Relative-stage GRU", "Relative-stage", "GRU")

    b10, b10_info = train_stage_model(TCNGRUStageOnly(input_dim, base.BEST_ARCH["channels"], base.BEST_ARCH["gru_hidden"], base.BEST_ARCH["dropout"]), tr_pack, va_pack, "B10_TCN_GRU")
    preds["B10"], probs["B10"] = predict_stage_model(b10, te_pack)
    method_info["B10"] = ("Relative-stage TCN-GRU", "Relative-stage", "TCN-GRU")

    mt_model, hist, _, best_epoch_b11 = base.train_model(tr_pack, va_pack, input_dim)
    torch.save(mt_model.state_dict(), DIR_MODEL / "B11_B12_multitask_tcn_gru.pth")
    pred_test_raw = base.predict_model(mt_model, te_pack)
    pred_test = base.apply_probability_inference(pred_test_raw, {
        "eta": 0.75, "fine_weight": 0.30, "temperature": 1.20,
        "mid_floor": 0.12, "late_tau": 0.66, "early_tau": 0.38, "order_blend": 0.25,
    })
    preds["B11"] = pred_test["stage_pred_raw"].values.astype(int)
    probs["B11"] = pred_test[[f"raw_prob_{s}" for s in STAGE_NAMES]].values
    method_info["B11"] = ("Multi-task TCN-GRU", "Relative-stage", "TCN-GRU + auxiliary heads")
    preds["B12"] = pred_test["stage_pred_final"].values.astype(int)
    probs["B12"] = pred_test[[f"final_prob_{s}" for s in STAGE_NAMES]].values
    method_info["B12"] = ("FGDS-PSI", "Relative-stage", "Proposed")

    result_rows, rep_rows, cm_all = [], [], []
    pred_table = meta[["condition", "cut_index", "stage_true"]].rename(columns={"cut_index": "run_id_end", "stage_true": "true_stage"}).copy()
    for m in [f"B{i}" for i in range(1, 13)]:
        name, stdef, model_type = method_info[m]
        result_rows.append(metrics_row(m, name, stdef, model_type, y_true, preds[m], probs[m]))
        rep_rows.extend(report_rows(m, y_true, preds[m]))
        cm_all.extend(cm_rows(m, y_true, preds[m]))
        pred_table[f"pred_{m}"] = [ID_TO_STAGE[int(v)] for v in preds[m]]
        pred_table[f"prob_E_{m}"] = probs[m][:, 0]
        pred_table[f"prob_M_{m}"] = probs[m][:, 1]
        pred_table[f"prob_L_{m}"] = probs[m][:, 2]

    results = pd.DataFrame(result_rows)
    round_numeric(results).to_csv(DIR_RESULT / "FINAL_comparison_results.csv", index=False, encoding="utf-8-sig")
    round_numeric(pd.DataFrame(rep_rows)).to_csv(DIR_RESULT / "FINAL_comparison_classification_reports_long.csv", index=False, encoding="utf-8-sig")
    round_numeric(pd.DataFrame(cm_all)).to_csv(DIR_RESULT / "FINAL_comparison_confusion_matrices_long.csv", index=False, encoding="utf-8-sig")
    round_numeric(pred_table).to_csv(DIR_RESULT / "FINAL_comparison_predictions.csv", index=False, encoding="utf-8-sig")

    b11_val = hist.loc[hist["score"].idxmin()] if "score" in hist.columns else hist.iloc[-1]
    debug_rows = []
    train_cnt = np.bincount(tr_pack["ys"], minlength=3)
    for method, info, pred in [("B10", b10_info, preds["B10"]), ("B11", {
        "best_epoch": int(best_epoch_b11),
        "best_val_acc": float(b11_val["val_acc"]),
        "best_val_macro_f1": float(b11_val["val_macro_f1"]),
        "best_val_MRec": float(b11_val["val_middle_recall"]),
    }, preds["B11"])]:
        mrow = metrics_row(method, method_info[method][0], method_info[method][1], method_info[method][2], y_true, pred, probs[method])
        debug_rows.append({
            "method": method,
            "train_windows": len(tr_pack["ys"]),
            "val_windows": len(va_pack["ys"]),
            "test_windows": len(te_pack["ys"]),
            "input_dim": input_dim,
            "selected_features": len(selected),
            "train_early": int(train_cnt[0]),
            "train_middle": int(train_cnt[1]),
            "train_late": int(train_cnt[2]),
            **info,
            "test_acc": mrow["Acc"],
            "test_macro_f1": mrow["Macro-F1"],
            "test_MF1": mrow["M-F1"],
        })
    debug_df = pd.DataFrame(debug_rows)
    round_numeric(debug_df).to_csv(DIR_RESULT / "DEBUG_B10_vs_B11_check.csv", index=False, encoding="utf-8-sig")

    write_summary(results, xgb_note, b2_uses_true_vb)
    plot_figures(results, pred_test, y_true, preds)

    b10r = results[results["Method"] == "B10"].iloc[0]
    b11r = results[results["Method"] == "B11"].iloc[0]
    b12r = results[results["Method"] == "B12"].iloc[0]
    print("\nFINAL_comparison_results.csv")
    print(round_numeric(results).to_string(index=False))
    print(f"\nB2 check: uses true VB-derived q on C6 = {b2_uses_true_vb}")
    print("\nDEBUG_B10_vs_B11_check.csv")
    print(round_numeric(debug_df).to_string(index=False))
    print("\nB12 vs B11 comparison")
    print(f"Macro-F1 difference: {b12r['Macro-F1'] - b11r['Macro-F1']:.4f}")
    print(f"M-F1 difference    : {b12r['M-F1'] - b11r['M-F1']:.4f}")
    print(f"M→L difference     : {b12r['M→L'] - b11r['M→L']:.4f}")
    print(f"Smooth difference  : {b12r['Smooth'] - b11r['Smooth']:.4f}")
    if b10r["Macro-F1"] < min(results[results["Method"].isin(["B8", "B9"])]["Macro-F1"]) - 0.05:
        print("Note: B10 remains lower than B8/B9 after recheck; see summary for explanation.")


def write_summary(results, xgb_note, b2_uses_true_vb):
    best_macro = results.loc[results["Macro-F1"].idxmax()]
    best_mf1 = results.loc[results["M-F1"].idxmax()]
    best_mlate = results.loc[results["M→L"].idxmin()]
    b10 = results[results["Method"] == "B10"].iloc[0]
    b11 = results[results["Method"] == "B11"].iloc[0]
    b12 = results[results["Method"] == "B12"].iloc[0]
    lines = [
        "FINAL rechecked comparison experiment summary",
        f"B2 uses sensor-derived q_proxy rather than true wear-derived q: {not b2_uses_true_vb}",
        f"XGBoost implementation: {xgb_note}",
        "B10 has been rechecked with the same selected features, scaler, L=12 windows, class weights, optimizer, learning rate, dropout, epochs and early-stopping criterion as the deep baselines.",
        f"Best method by Macro-F1: {best_macro['Method']} ({best_macro['Macro-F1']:.4f})",
        f"Best method by M-F1: {best_mf1['Method']} ({best_mf1['M-F1']:.4f})",
        f"Best method by M→L: {best_mlate['Method']} ({best_mlate['M→L']:.4f})",
        f"B12 vs B10 Macro-F1 change: {b12['Macro-F1'] - b10['Macro-F1']:.4f}; M-F1 change: {b12['M-F1'] - b10['M-F1']:.4f}",
        f"B12 vs B11 Macro-F1 change: {b12['Macro-F1'] - b11['Macro-F1']:.4f}; M-F1 change: {b12['M-F1'] - b11['M-F1']:.4f}",
        f"B12 keeps M→L=0: {abs(b12['M→L']) < 1e-12}",
        f"B12 has lower Smooth than B11: {b12['Smooth'] < b11['Smooth']}",
    ]
    if b10["Macro-F1"] < 0.90:
        lines.append("B10 remains relatively weak after recheck, indicating that the single-task TCN-GRU can be unstable for the middle stage in this split.")
    (DIR_RESULT / "FINAL_comparison_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def emphasize_b12(ax):
    for tick in ax.get_xticklabels():
        if tick.get_text() == "B12":
            tick.set_color(HIGHLIGHT)
            tick.set_fontweight("bold")


def plot_figures(results, pred_test, y_true, preds):
    methods = results["Method"].tolist()
    x = np.arange(len(methods))
    w = 0.36

    plt.figure(figsize=(9.0, 4.2))
    plt.bar(x - w / 2, results["Acc"], width=w, label="Acc", color="#4C78A8")
    plt.bar(x + w / 2, results["Macro-F1"], width=w, label="Macro-F1", color="#F58518")
    plt.xticks(x, methods)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.legend(loc="lower right", ncol=2)
    plt.grid(axis="y", ls="--", lw=0.4, alpha=0.3)
    emphasize_b12(plt.gca())
    savefig(DIR_FIG / "Fig8_overall_comparison_B1_B12")

    plt.figure(figsize=(9.0, 4.2))
    for col, color, marker in [("M-Pre", "#4C78A8", "o"), ("M-Rec", "#54A24B", "s"), ("M-F1", HIGHLIGHT, "^")]:
        plt.plot(x, results[col], marker=marker, lw=1.8, color=color, label=col)
    plt.xticks(x, methods)
    plt.ylim(0, 1.05)
    plt.ylabel("Middle-stage score")
    plt.legend(loc="lower right", ncol=3)
    plt.grid(True, ls="--", lw=0.4, alpha=0.3)
    emphasize_b12(plt.gca())
    savefig(DIR_FIG / "Fig9_middle_stage_performance_B1_B12")

    plt.figure(figsize=(9.0, 4.2))
    plt.bar(x - w / 2, results["M→E"], width=w, label="M→E", color="#8E6C8A")
    plt.bar(x + w / 2, results["M→L"], width=w, label="M→L", color=HIGHLIGHT)
    plt.xticks(x, methods)
    plt.ylabel("Misclassification rate")
    plt.legend(loc="upper right", ncol=2)
    plt.grid(axis="y", ls="--", lw=0.4, alpha=0.3)
    emphasize_b12(plt.gca())
    savefig(DIR_FIG / "Fig10_middle_misclassification_B1_B12")

    plt.figure(figsize=(8.8, 4.0))
    plt.bar(x - w / 2, results["Rev"], width=w, label="Rev", color="#4C78A8")
    plt.bar(x + w / 2, results["Jump"], width=w, label="Jump", color="#F58518")
    plt.xticks(x, methods)
    plt.ylabel("Count")
    plt.legend(loc="upper right", ncol=2)
    plt.grid(axis="y", ls="--", lw=0.4, alpha=0.3)
    emphasize_b12(plt.gca())
    savefig(DIR_FIG / "Fig11a_stage_reversal_jump_B1_B12")

    plt.figure(figsize=(8.8, 4.0))
    plt.bar(x, results["Smooth"], color=["#4C78A8"] * 11 + [HIGHLIGHT])
    plt.xticks(x, methods)
    plt.ylabel("Smooth")
    plt.grid(axis="y", ls="--", lw=0.4, alpha=0.3)
    emphasize_b12(plt.gca())
    savefig(DIR_FIG / "Fig11b_probability_smoothness_B1_B12")

    cm = confusion_matrix(y_true, preds["B12"], labels=[0, 1, 2])
    cmn = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    plt.figure(figsize=(4.6, 4.0))
    im = plt.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(3), ["E", "M", "L"])
    plt.yticks(range(3), ["E", "M", "L"])
    for i in range(3):
        for j in range(3):
            color = "white" if cmn[i, j] > 0.55 else "black"
            plt.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.2f}", ha="center", va="center", color=color, fontsize=10)
    plt.xlabel("Predicted stage")
    plt.ylabel("True stage")
    savefig(DIR_FIG / "Fig12_confusion_matrix_B12")

    g = pred_test.sort_values("cut_index").reset_index(drop=True)
    plt.figure(figsize=(9.4, 4.5))
    bg = {"early": "#DDF5E8", "middle": "#FFF0D6", "late": "#F6C3C3"}
    stage = g["stage_true"].values
    xcut = g["cut_index"].values
    start = 0
    for i in range(1, len(g) + 1):
        if i == len(g) or stage[i] != stage[start]:
            plt.axvspan(xcut[start] - 0.5, xcut[i - 1] + 0.5, color=bg[stage[start]], alpha=0.42, lw=0)
            start = i
    for st, color in [("early", "#2E8B57"), ("middle", "#D97706"), ("late", HIGHLIGHT)]:
        plt.plot(g["cut_index"], g[f"final_prob_{st}"], color=color, lw=1.8, label=st)
    plt.xlabel("Run index")
    plt.ylabel("Stage probability")
    plt.ylim(-0.04, 1.04)
    plt.legend(loc="upper center", ncol=3)
    plt.grid(True, ls="--", lw=0.4, alpha=0.3)
    savefig(DIR_FIG / "Fig13_probability_evolution_B12_C6")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
r"""
Source-domain-only hyperparameter validation for HTT-Net (B13).

Purpose: rule out "HTT-Net's D1 result is low because it is obviously
under-trained / badly configured" without ever looking at C6. C6 is the
final test set for this baseline and must stay completely unseen during
model/hyperparameter selection -- see README.md and PAPER_SPEC.md.

Protocol (bidirectional source-only proxy validation, matching this
project's own existing convention for proxy validation elsewhere in
`代码/1.1阶段分类.py`):
    Fold A: train on C1, validate on C4
    Fold B: train on C4, validate on C1
    Selection score = average(Macro-F1_A, Macro-F1_B)
    Tie-break: average(M-F1_A, M-F1_B), then simpler/smaller config.

Search space (only values PAPER_SPEC.md marks "Missing in paper" and that
plausibly matter at L=12; see PAPER_SPEC.md and FINAL_HTT_NET_REPORT.md for
the justification of what is/isn't included):
    lr        in [1e-4, 3e-4, 5e-4]
    dropout   in [0.10, 0.20]
    embed_dim in [32, 64]
    (num_heads=4 fixed -- divides every stage width for both embed_dim
     choices; window_size=3 fixed -- see "window_size not searched" note
     below)

window_size not searched: window_size=3 evenly divides every pre-merge
stage length at L=12 (12, 6, 3) with no extra padding needed inside
HTTNetStage.forward. window_size=2 would leave stage 3 (L=3, odd) with
3 % 2 != 0, which window_partition() cannot handle without adding padding
logic *inside* HTTNetStage.forward -- a code change, not a hyperparameter
change, and out of scope for this tuning pass (see report).

Each fold's data preparation (feature selection, GMM fine states, scaler --
all fit on that fold's *training* condition only) is done ONCE per fold and
reused across all hyperparameter configs, since data prep does not depend
on model hyperparameters.

Usage:
    python source_only_tuning.py
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import train as T  # reuse base import, HTTNet, class_weights, predict_stage_model, metrics helpers

THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "tuning_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Frozen authoritative feature file -- see FINAL_HTT_NET_REPORT.md.
# Must never be reconstructed_v1 or any other superseded artifact.
AUTHORITATIVE_FEATURE_FILE = THIS_DIR / "data" / "run_level_features_all.csv"
T.base.FEATURE_FILE = AUTHORITATIVE_FEATURE_FILE

SEARCH_SPACE = {
    "lr": [1e-4, 3e-4, 5e-4],
    "dropout": [0.10, 0.20],
    "embed_dim": [32, 64],
}
FIXED = {"num_heads": 4, "window_size": 3, "depths": (2, 2, 2, 2)}
TUNING_SEED = 42
TUNING_EPOCHS = 120
TUNING_PATIENCE = 18


def prepare_source_fold(label_df: pd.DataFrame, train_cond: str, val_cond: str):
    """Single-condition source-only fold: train on train_cond, validate on
    val_cond. Feature selection / GMM / scaler are all fit on train_cond
    only -- val_cond never contributes to any fitted statistic, exactly the
    same no-leakage discipline the existing D1 pipeline already applies to
    C6."""
    base = T.base
    train_raw = label_df[label_df["condition"] == train_cond].copy()
    val_raw = label_df[label_df["condition"] == val_cond].copy()

    raw_cols = base.get_raw_numeric_sensor_cols(train_raw)
    split_feat = base.build_online_features_by_split(
        {"fold_train": train_raw, "fold_val": val_raw}, raw_cols
    )
    feat_train = split_feat[split_feat["split_name_for_feature_build"] == "fold_train"].copy()
    feat_val = split_feat[split_feat["split_name_for_feature_build"] == "fold_val"].copy()

    all_cols = base.feature_cols_from(feat_train)
    feat_train, feat_val = base.fill_by_train_median(feat_train, feat_val, all_cols)
    selected, _ = base.select_features_train_only(feat_train)
    feat_train, feat_val = base.fill_by_train_median(feat_train, feat_val, selected)

    gmm, raw_to_order = base.fit_train_gmm(feat_train)
    feat_train = base.assign_fine_states(feat_train, gmm, raw_to_order)
    feat_val = base.assign_fine_states(feat_val, gmm, raw_to_order)

    scaler = base.StandardScaler().fit(feat_train[selected].values)
    for df in [feat_train, feat_val]:
        df[selected] = np.nan_to_num(scaler.transform(df[selected].values), nan=0.0, posinf=0.0, neginf=0.0)

    L = base.BEST_ARCH["L"]
    tr_pack = base.make_pack(feat_train, selected, L, "fold_train")
    va_pack = base.make_pack(feat_val, selected, L, "fold_val")
    return tr_pack, va_pack, selected


def train_and_eval_one_config(tr_pack, va_pack, input_dim, cfg, tag):
    base = T.base
    base.set_seed(TUNING_SEED)
    model = T.HTTNet(
        input_dim=input_dim,
        num_classes=3,
        embed_dim=cfg["embed_dim"],
        depths=FIXED["depths"],
        num_heads=FIXED["num_heads"],
        window_size=FIXED["window_size"],
        dropout=cfg["dropout"],
    ).to(base.DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=T.TRAIN_CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    w = T.class_weights(tr_pack["ys"])
    best_state, best_score, wait = None, np.inf, 0
    best_epoch, best_val_macro_f1, best_val_m_f1, best_val_acc = 0, np.nan, np.nan, np.nan
    history = []

    for epoch in range(1, TUNING_EPOCHS + 1):
        model.train()
        losses = []
        for X, ys, _, _ in tr_pack["loader"]:
            X, ys = X.to(base.DEVICE), ys.to(base.DEVICE)
            logits = model(X)
            loss = F.cross_entropy(logits, ys, weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), T.TRAIN_CFG["grad_clip"])
            opt.step()
            losses.append(float(loss.detach().cpu()))

        yv, pv = T.predict_stage_model(model, va_pack)
        m = base.clf_metrics(va_pack["ys"], yv)
        val_loss = -np.mean(np.log(np.clip(pv[np.arange(len(yv)), va_pack["ys"]], 1e-12, 1.0)))
        score = 0.7 * (1 - m["f1"]) + 1.0 * (1 - m["middle_recall"]) + 0.15 * val_loss
        scheduler.step(score)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": float(val_loss),
                         "val_acc": m["acc"], "val_macro_f1": m["f1"], "val_m_f1": m["middle_f1"]})
        if score < best_score:
            best_score, best_state, wait = score, {k: v.clone() for k, v in model.state_dict().items()}, 0
            best_epoch, best_val_macro_f1, best_val_m_f1, best_val_acc = epoch, m["f1"], m["middle_f1"], m["acc"]
        else:
            wait += 1
        if wait >= TUNING_PATIENCE:
            break

    pd.DataFrame(history).to_csv(OUT_DIR / f"history_{tag}.csv", index=False, encoding="utf-8-sig")
    return {
        "val_macro_f1": best_val_macro_f1,
        "val_m_f1": best_val_m_f1,
        "val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "n_params": model.num_parameters(),
    }


def main():
    base = T.base
    print("=" * 100)
    print("HTT-Net source-only hyperparameter validation (C6 NEVER used)")
    print(f"Device: {base.DEVICE}")
    print("=" * 100)

    raw_df = base.load_feature_table()
    label_df, _ = base.define_condition_relative_stages(raw_df)

    print("Preparing Fold A (train=C1, val=C4)...")
    trA, vaA, selA = prepare_source_fold(label_df, "C1", "C4")
    print(f"  Fold A: {len(trA['ys'])} train windows, {len(vaA['ys'])} val windows, {len(selA)} features")

    print("Preparing Fold B (train=C4, val=C1)...")
    trB, vaB, selB = prepare_source_fold(label_df, "C4", "C1")
    print(f"  Fold B: {len(trB['ys'])} train windows, {len(vaB['ys'])} val windows, {len(selB)} features")

    configs = []
    for lr, dropout, embed_dim in itertools.product(
        SEARCH_SPACE["lr"], SEARCH_SPACE["dropout"], SEARCH_SPACE["embed_dim"]
    ):
        configs.append({"lr": lr, "dropout": dropout, "embed_dim": embed_dim})

    rows = []
    t0 = time.time()
    for i, cfg in enumerate(configs):
        tag = f"lr{cfg['lr']}_do{cfg['dropout']}_ed{cfg['embed_dim']}"
        print(f"\n[{i+1}/{len(configs)}] {tag}")

        rA = train_and_eval_one_config(trA, vaA, len(selA), cfg, f"A_{tag}")
        rB = train_and_eval_one_config(trB, vaB, len(selB), cfg, f"B_{tag}")

        avg_macro_f1 = (rA["val_macro_f1"] + rB["val_macro_f1"]) / 2
        avg_m_f1 = (rA["val_m_f1"] + rB["val_m_f1"]) / 2
        print(f"  Fold A: Macro-F1={rA['val_macro_f1']:.4f} M-F1={rA['val_m_f1']:.4f} best_epoch={rA['best_epoch']}")
        print(f"  Fold B: Macro-F1={rB['val_macro_f1']:.4f} M-F1={rB['val_m_f1']:.4f} best_epoch={rB['best_epoch']}")
        print(f"  Avg Macro-F1={avg_macro_f1:.4f}  Avg M-F1={avg_m_f1:.4f}")

        rows.append({
            "config_id": tag, **cfg, **FIXED,
            "foldA_macro_f1": rA["val_macro_f1"], "foldA_m_f1": rA["val_m_f1"], "foldA_acc": rA["val_acc"], "foldA_best_epoch": rA["best_epoch"],
            "foldB_macro_f1": rB["val_macro_f1"], "foldB_m_f1": rB["val_m_f1"], "foldB_acc": rB["val_acc"], "foldB_best_epoch": rB["best_epoch"],
            "avg_macro_f1": avg_macro_f1, "avg_m_f1": avg_m_f1, "n_params": rA["n_params"],
        })

    results = pd.DataFrame(rows).sort_values(
        ["avg_macro_f1", "avg_m_f1", "n_params"], ascending=[False, False, True]
    ).reset_index(drop=True)
    results.to_csv(OUT_DIR / "source_only_search.csv", index=False, encoding="utf-8-sig")

    best = results.iloc[0]
    print("\n" + "=" * 100)
    print(f"Search finished in {time.time()-t0:.1f}s, {len(configs)} configs x 2 folds = {2*len(configs)} trainings")
    print("Top 3 configs by avg Macro-F1:")
    print(results.head(3)[["config_id", "avg_macro_f1", "avg_m_f1", "foldA_macro_f1", "foldB_macro_f1"]].to_string(index=False))
    print(f"\nSelected: {best['config_id']}  avg_macro_f1={best['avg_macro_f1']:.4f}")
    print(f"C6 used during this search: NO")

    return best, results


if __name__ == "__main__":
    main()

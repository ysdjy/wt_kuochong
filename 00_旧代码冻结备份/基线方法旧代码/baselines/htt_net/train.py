# -*- coding: utf-8 -*-
r"""
HTT-Net baseline: train + evaluate on this project's unified D1 protocol
(C1+C4 -> C6), reusing the *exact* data pipeline, feature engineering,
windowing, class-weighting, optimizer/scheduler settings and evaluation
metrics already used for every other deep baseline (B8 TCN, B9 GRU,
B10 TCN-GRU, B11 multi-task TCN-GRU) in this project.

This script is READ-ONLY with respect to the existing DC-PSR codebase: it
imports `main_experiment_3_fgds_psi_optimized.py` from `代码/` as a library
(exactly the way `代码/7.4对比实验.py` already does for B8-B12) and never
edits it. HTT-Net itself lives entirely in `model.py` next to this file.

IMPORTANT -- data availability (read before running):
    The feature file this pipeline depends on
    (`run_level_features_all.csv`, referenced by `代码/main_experiment_3_
    fgds_psi_optimized.py` as `ROOT/"PHM实验"/"1run_run_level_features"/
    "02_features"/run_level_features_all.csv` under a `C:\Users\wangting\...`
    path) was NOT found anywhere on this machine as of the writing of this
    baseline. Set the environment variable HTT_FEATURE_FILE to the correct
    local path before running this script, e.g.:

        set HTT_FEATURE_FILE=D:\path\to\run_level_features_all.csv
        python train.py

    Without it, this script prints a clear error and exits rather than
    silently fabricating results. See README.md for details.

Usage:
    python train.py
    python train.py --smoke-test          # synthetic data, verifies the
                                           # pipeline runs end to end without
                                           # requiring the real feature file
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
CODE_DIR = PROJECT_ROOT / "代码"

sys.path.insert(0, str(THIS_DIR))
from model import HTTNet  # noqa: E402

# The base module creates its own output directories at import time and
# writes disposable diagnostic CSVs there as a side effect of some of its
# functions (split_grouped_lifecycle, fit_train_gmm, etc.). None of that is
# a real deliverable of this baseline (our own outputs go to output_root
# below), so it is redirected to the OS temp dir rather than the
# BaiduSyncdisk-synced project folder: writing repeatedly into a folder the
# sync client is actively uploading causes intermittent
# `PermissionError: [Errno 13]` while the client holds a lock on the file.
import tempfile
os.environ.setdefault("FGDS_RUN_DIR", str(Path(tempfile.gettempdir()) / "htt_net_base_import_cache"))

sys.path.insert(0, str(CODE_DIR))
import main_experiment_3_fgds_psi_optimized as base  # noqa: E402

METHOD_ID = "B13"
METHOD_NAME = "HTT-Net"
STAGE_DEF = "Condition-relative stage"
MODEL_TYPE = "HTT-Net (Xue et al., 2023) -- architecture reimplementation"

OUTPUT_ROOT = Path(os.environ.get("HTT_OUTPUT_DIR", str(PROJECT_ROOT / "outputs" / "htt_net" / "D1_C1C4_to_C6")))

# Architecture hyperparameters: see PAPER_SPEC.md for which of these are
# "Missing in paper" implementation choices vs. paper-explicit values.
HTT_ARCH = {
    "embed_dim": 32,
    "depths": (2, 2, 2, 2),
    "num_heads": 4,
    "window_size": 3,
    "dropout": base.BEST_ARCH["dropout"],  # 0.20, reused from the shared unified protocol for fairness
}
TRAIN_CFG = {
    "lr": base.BEST_ARCH["lr"],            # 5e-4, reused (paper's own lr=1e-4/bs=16 is the *original* T1 protocol, not used here -- see PAPER_SPEC.md sec 5)
    "weight_decay": base.WEIGHT_DECAY,     # 1e-5
    "epochs": base.EPOCHS,                 # 120
    "patience": base.PATIENCE,             # 18
    "grad_clip": base.GRAD_CLIP,           # 1.0
    "seed": base.RANDOM_SEED,              # 42
}


def class_weights(y: np.ndarray) -> torch.Tensor:
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


def train_htt_model(model, tr_pack, va_pack, epochs, patience):
    """Identical training protocol to B8/B9/B10 in 代码/7.4对比实验.py
    (same optimizer, scheduler, class weighting, early-stopping score,
    grad clipping), reproduced here rather than imported since those
    helpers are defined inside a script, not a reusable module."""
    model = model.to(base.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN_CFG["lr"], weight_decay=TRAIN_CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=1e-5)
    w = class_weights(tr_pack["ys"])
    best_state, best_score, wait = None, np.inf, 0
    history = []
    best_info = {"best_epoch": 0, "best_val_acc": np.nan, "best_val_macro_f1": np.nan, "best_val_MRec": np.nan}
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for X, ys, _, _ in tr_pack["loader"]:
            X, ys = X.to(base.DEVICE), ys.to(base.DEVICE)
            logits = model(X)
            loss = F.cross_entropy(logits, ys, weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG["grad_clip"])
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))

        yv, pv = predict_stage_model(model, va_pack)
        m = base.clf_metrics(va_pack["ys"], yv)
        val_loss = -np.mean(np.log(np.clip(pv[np.arange(len(yv)), va_pack["ys"]], 1e-12, 1.0)))
        score = 0.7 * (1 - m["f1"]) + 1.0 * (1 - m["middle_recall"]) + 0.15 * val_loss
        scheduler.step(score)

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "val_loss": float(val_loss),
            "val_acc": m["acc"],
            "val_macro_f1": m["f1"],
            "val_middle_recall": m["middle_recall"],
            "score": score,
        })
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
    return model, pd.DataFrame(history), best_info


def consistency(y_pred: np.ndarray, prob: np.ndarray) -> dict:
    """Same definition as 代码/7.4对比实验.py's consistency(): Rev/Jump/Smooth
    computed on predictions already sorted by (condition, cut_index)."""
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_pred) <= 1:
        return {"Rev": 0, "Jump": 0, "Smooth": np.nan}
    dy = np.diff(y_pred)
    return {
        "Rev": int(np.sum(dy < 0)),
        "Jump": int(np.sum(np.abs(dy) >= 2)),
        "Smooth": float(np.mean(np.sum(np.abs(np.diff(prob, axis=0)), axis=1))),
    }


def metrics_row(y_true, y_pred, prob) -> dict:
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
    p_each, r_each, f1_each, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cmn = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    c = consistency(y_pred, prob)
    return {
        "Method": METHOD_ID,
        "Method_name": METHOD_NAME,
        "Stage_definition": STAGE_DEF,
        "Model_type": MODEL_TYPE,
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


def prepare_data(feature_file: Path | None = None):
    """Reproduces the exact data-prep sequence from 代码/7.4对比实验.py's
    main(): load -> condition-relative stage labels -> C1+C4/C6 split ->
    split-safe online features -> train-only feature selection -> GMM fine
    states (needed only because StageDataset expects (X, ys, yf, yq), HTT-Net
    itself never uses yf/yq) -> train-fit StandardScaler -> L=12 windows.
    """
    if feature_file is not None:
        base.FEATURE_FILE = Path(feature_file)

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
    selected, selected_df = base.select_features_train_only(feat_train)
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
    return tr_pack, va_pack, te_pack, selected, scaler


def make_synthetic_feature_csv(path: Path, n_runs: int = 260, n_raw_cols: int = 12, seed: int = 0) -> None:
    """Build a tiny synthetic CSV with the same schema base.load_feature_table()
    expects, purely to smoke-test this script's integration glue (data prep,
    windowing, training loop, saving) without the real PHM2010 feature file.
    This does NOT validate real-world model performance -- see README.md.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cond in ["C1", "C4", "C6"]:
        vb = np.cumsum(rng.uniform(0.05, 0.4, size=n_runs)) + rng.normal(0, 0.05, size=n_runs)
        vb = np.clip(vb, 0.01, None)
        for run_id in range(1, n_runs + 1):
            row = {"condition": cond, "run_id": run_id, "VB": float(vb[run_id - 1])}
            for c in range(n_raw_cols):
                row[f"raw_signal_{c}"] = float(rng.normal(loc=vb[run_id - 1], scale=1.0))
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def run(feature_file: Path | None, smoke_test: bool, epochs_override: int | None = None):
    base.set_seed(TRAIN_CFG["seed"])

    output_root = OUTPUT_ROOT
    if smoke_test:
        # Never write smoke-test output into the real-result directory --
        # a synthetic-data run must not be mistakable for a real one.
        output_root = THIS_DIR / "_smoke_test_output"
        synth_path = THIS_DIR / "_smoke_test_feature_file.csv"
        make_synthetic_feature_csv(synth_path)
        feature_file = synth_path
        print(f"[smoke-test] using synthetic feature file: {synth_path}")
        print(f"[smoke-test] output isolated to: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    if feature_file is None:
        feature_file = Path(os.environ.get("HTT_FEATURE_FILE", "")) if os.environ.get("HTT_FEATURE_FILE") else None

    if feature_file is None and not base.FEATURE_FILE.exists():
        print("=" * 100)
        print("ERROR: real feature file not found and no override was given.")
        print(f"Default (author's original machine) path: {base.FEATURE_FILE}")
        print("Set HTT_FEATURE_FILE=<path to run_level_features_all.csv> or pass --feature-file.")
        print("Run with --smoke-test to validate the pipeline using synthetic data instead.")
        print("=" * 100)
        sys.exit(1)

    print("=" * 100)
    print(f"HTT-Net baseline ({METHOD_ID}): D1 task, C1+C4 -> C6")
    print(f"Device: {base.DEVICE}")
    print(f"Feature file: {feature_file if feature_file else base.FEATURE_FILE}")
    print(f"Output dir: {output_root}")
    print("=" * 100)

    t0 = time.time()
    tr_pack, va_pack, te_pack, selected, scaler = prepare_data(feature_file)
    prep_time = time.time() - t0

    input_dim = len(selected)
    model = HTTNet(input_dim=input_dim, num_classes=3, **HTT_ARCH)
    n_params = model.num_parameters()
    print(f"Selected features: {input_dim}, HTT-Net parameters: {n_params:,}")

    epochs = epochs_override or TRAIN_CFG["epochs"]
    t1 = time.time()
    model, history, best_info = train_htt_model(model, tr_pack, va_pack, epochs, TRAIN_CFG["patience"])
    train_time = time.time() - t1

    torch.save(model.state_dict(), output_root / "checkpoint_best.pth")
    history.to_csv(output_root / "training_log.csv", index=False, encoding="utf-8-sig")

    t2 = time.time()
    y_pred, prob = predict_stage_model(model, te_pack)
    infer_time = (time.time() - t2) / max(len(te_pack["ys"]), 1)

    meta = te_pack["meta"].copy().reset_index(drop=True)
    y_true = meta["stage_true_id"].values.astype(int)

    mrow = metrics_row(y_true, y_pred, prob)
    mrow["n_params"] = n_params
    mrow["train_seconds"] = train_time
    mrow["prep_seconds"] = prep_time
    mrow["inference_seconds_per_sample"] = infer_time
    mrow["best_epoch"] = best_info["best_epoch"]
    mrow["is_smoke_test"] = smoke_test

    pred_table = meta[["condition", "cut_index", "stage_true"]].rename(
        columns={"cut_index": "run_id_end", "stage_true": "true_stage"}
    ).copy()
    id_to_stage = base.ID_TO_STAGE
    pred_table["pred_stage"] = [id_to_stage[int(v)] for v in y_pred]
    pred_table["p_early"] = prob[:, 0]
    pred_table["p_middle"] = prob[:, 1]
    pred_table["p_late"] = prob[:, 2]

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_df = pd.DataFrame(cm, index=["true_early", "true_middle", "true_late"], columns=["pred_early", "pred_middle", "pred_late"])

    pd.DataFrame([mrow]).to_csv(output_root / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    pred_table.to_csv(output_root / "test_predictions.csv", index=False, encoding="utf-8-sig")
    cm_df.to_csv(output_root / "confusion_matrix.csv", encoding="utf-8-sig")

    config_dump = {
        "method_id": METHOD_ID,
        "method_name": METHOD_NAME,
        "arch": HTT_ARCH,
        "train_cfg": TRAIN_CFG,
        "n_selected_features": input_dim,
        "n_params": n_params,
        "smoke_test": smoke_test,
        "feature_file": str(feature_file if feature_file else base.FEATURE_FILE),
    }
    with open(output_root / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dump, f, indent=2, ensure_ascii=False)
    with open(output_root / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in mrow.items()}, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Acc={mrow['Acc']:.4f}  Macro-F1={mrow['Macro-F1']:.4f}  M-F1={mrow['M-F1']:.4f}  "
          f"M→E={mrow['M→E']:.4f}  M→L={mrow['M→L']:.4f}  Rev={mrow['Rev']}  Jump={mrow['Jump']}  Smooth={mrow['Smooth']:.4f}")
    print(f"Params={n_params:,}  best_epoch={best_info['best_epoch']}  train_time={train_time:.1f}s")
    print(f"Outputs saved to: {output_root}")
    return mrow


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", type=str, default=None, help="Path to run_level_features_all.csv")
    parser.add_argument("--smoke-test", action="store_true", help="Use synthetic data to verify the pipeline runs end to end")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch budget (e.g. for a quick smoke run)")
    args = parser.parse_args()
    run(Path(args.feature_file) if args.feature_file else None, args.smoke_test, args.epochs)

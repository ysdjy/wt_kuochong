# -*- coding: utf-8 -*-
r"""
Fixed-preprocessing / training-seed-isolation diagnostic runner.

For a given (method, TRAIN_SEED) pair, this script:
  1. Loads the PREPROCESS_SEED=42 frozen artifacts built by
     build_frozen_preprocess.py (selected features, GMM fine states,
     scaler, split membership, L=12 windows) -- NEVER recomputes them.
  2. Asserts the frozen artifacts' hashes are unchanged (protocol
     integrity check) before touching any model.
  3. Resets all RNGs to TRAIN_SEED, THEN instantiates the model (never
     the other order), so per-method RNG consumption cannot leak into
     the next method's initialization.
  4. Trains using the EXISTING training functions from
     `代码/7.4对比实验.py`, `代码/main_experiment_3_fgds_psi_optimized.py`
     and `baselines/htt_net/train.py` (imported as libraries, never
     copied/modified) -- only TRAIN_SEED changes, no hyperparameter is
     retuned.
  5. Evaluates once on the frozen C6 test universe (304 windows,
     run_id_end 12-315) and writes metrics/predictions/config/DONE.flag.

Usage:
    python run_diagnostic_seed.py --method rf --train_seed 42
    python run_diagnostic_seed.py --method tcn_gru --train_seed 52
    python run_diagnostic_seed.py --method multitask_tcn_gru --train_seed 62
    python run_diagnostic_seed.py --method htt_net --train_seed 72
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import random
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
DIAG_ROOT = THIS_DIR.parent
PROJECT_ROOT = DIAG_ROOT.parent
CODE_DIR = PROJECT_ROOT / "代码"
HTT_DIR = PROJECT_ROOT / "baselines" / "htt_net"
FROZEN_DIR = DIAG_ROOT / "frozen_preprocess"
RESULTS_DIR = DIAG_ROOT / "results"

PREPROCESS_SEED = 42
VALID_METHODS = ["rf", "tcn_gru", "multitask_tcn_gru", "htt_net"]  # dc_psr rides along with multitask_tcn_gru

warnings.filterwarnings("ignore")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_frozen_artifacts_unchanged():
    with open(FROZEN_DIR / "manifest_hashes.json", "r", encoding="utf-8") as f:
        recorded = json.load(f)
    mismatches = []
    for name, expected in recorded["files"].items():
        actual = sha256_file(FROZEN_DIR / name)
        if actual != expected:
            mismatches.append((name, expected, actual))
    if mismatches:
        lines = [f"  {n}: expected {e[:16]}... got {a[:16]}..." for n, e, a in mismatches]
        raise RuntimeError("FROZEN PREPROCESSING ARTIFACTS CHANGED -- protocol violated:\n" + "\n".join(lines))
    return recorded


def set_train_seed(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"[set_train_seed] WARNING: use_deterministic_algorithms unavailable: {e}")


def load_frozen(base):
    with open(FROZEN_DIR / "selected_features_seed42.json", "r", encoding="utf-8") as f:
        sel_info = json.load(f)
    selected = sel_info["selected_features_in_order"]
    feat_train = pd.read_csv(FROZEN_DIR / "feat_train_frozen.csv")
    feat_val = pd.read_csv(FROZEN_DIR / "feat_val_frozen.csv")
    feat_test = pd.read_csv(FROZEN_DIR / "feat_test_frozen.csv")
    return selected, feat_train, feat_val, feat_test


def build_packs(base, selected, feat_train, feat_val, feat_test, L):
    tr_pack = base.make_pack(feat_train, selected, L, "final_train")
    va_pack = base.make_pack(feat_val, selected, L, "final_internal_val")
    te_pack = base.make_pack(feat_test, selected, L, "test_C6")
    return tr_pack, va_pack, te_pack


def import_base():
    os.environ.setdefault("FGDS_RUN_DIR", str(Path(tempfile.gettempdir()) / "protocol_diagnostic_base_side_outputs"))
    sys.path.insert(0, str(CODE_DIR))
    import main_experiment_3_fgds_psi_optimized as base  # noqa
    base.FEATURE_FILE = PROJECT_ROOT / "baselines" / "htt_net" / "data" / "run_level_features_all.csv"
    return base


def import_comparison_script(train_seed: int, out_dir: Path):
    os.environ["COMPARISON_RECHECK_DIR"] = str(out_dir / "_scratch_side_outputs")
    script_path = CODE_DIR / "7.4对比实验.py"
    mod_name = f"protocol_diag_comparison_seed{train_seed}"
    spec = importlib.util.spec_from_file_location(mod_name, script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def import_htt_train(out_dir: Path):
    os.environ["HTT_OUTPUT_DIR"] = str(out_dir / "_scratch_side_outputs")
    sys.path.insert(0, str(HTT_DIR))
    if "train" in sys.modules:
        del sys.modules["train"]
    import train as T  # noqa
    return T


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def save_run(out_dir: Path, method: str, train_seed: int, hashes: dict, config: dict,
             metrics_rows: list, pred_df: pd.DataFrame, t_start: float, t_end: float, best_info: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "method": method,
        "preprocess_seed": PREPROCESS_SEED,
        "train_seed": train_seed,
        "git_commit": git_commit(),
        "feature_hash": hashes["files"]["selected_features_seed42.json"],
        "split_hash": hashes["files"]["split_manifest.csv"],
        "gmm_hash": hashes["files"]["gmm_seed42.pkl"],
        "window_hash": hashes["files"]["window_manifest.csv"],
        "config": config,
        "start_time": t_start,
        "end_time": t_end,
        "training_seconds": t_end - t_start,
        **best_info,
    }
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    if pred_df is not None:
        pred_df.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    (out_dir / "DONE.flag").write_text(f"done at {t_end}\n", encoding="utf-8")
    print(f"[save_run] wrote {out_dir}")


def run_rf(train_seed: int, hashes: dict):
    base = import_base()
    selected, feat_train, feat_val, feat_test = load_frozen(base)
    L = base.BEST_ARCH["L"]
    out_dir = RESULTS_DIR / "rf" / f"seed{train_seed}"
    mod = import_comparison_script(train_seed, out_dir)

    set_train_seed(train_seed)
    tr_pack, va_pack, te_pack = build_packs(base, selected, feat_train, feat_val, feat_test, L)
    meta = te_pack["meta"].copy().reset_index(drop=True)
    y_true = meta["stage_true_id"].values.astype(int)
    test_by_cut = feat_test.set_index("run_id")
    Xtr = feat_train[selected].values
    ytr_rel = feat_train["stage_id"].values.astype(int)
    Xte = test_by_cut.loc[meta["cut_index"].values, selected].values

    from sklearn.ensemble import RandomForestClassifier
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=400, random_state=train_seed, class_weight="balanced_subsample", n_jobs=-1)
    rf.fit(Xtr, ytr_rel)
    t1 = time.time()
    y_pred = rf.predict(Xte)
    prob = rf.predict_proba(Xte)

    row = mod.metrics_row("RF", "Relative-stage RF (diagnostic)", "Relative-stage", "RF", y_true, y_pred, prob)
    pred_df = meta[["condition", "cut_index", "stage_true"]].copy()
    pred_df["pred"] = [base.ID_TO_STAGE[int(v)] for v in y_pred]
    pred_df["prob_E"], pred_df["prob_M"], pred_df["prob_L"] = prob[:, 0], prob[:, 1], prob[:, 2]

    save_run(out_dir, "RF", train_seed, hashes,
             {"n_estimators": 400, "class_weight": "balanced_subsample", "random_state": train_seed},
             [row], pred_df, t0, t1, {"best_epoch": None})


def run_tcn_gru(train_seed: int, hashes: dict):
    base = import_base()
    selected, feat_train, feat_val, feat_test = load_frozen(base)
    L = base.BEST_ARCH["L"]
    out_dir = RESULTS_DIR / "tcn_gru" / f"seed{train_seed}"
    mod = import_comparison_script(train_seed, out_dir)
    # mod.train_stage_model saves a .pth into mod.DIR_MODEL -- redirect it into our own results dir.
    mod.DIR_MODEL = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    set_train_seed(train_seed)
    tr_pack, va_pack, te_pack = build_packs(base, selected, feat_train, feat_val, feat_test, L)
    meta = te_pack["meta"].copy().reset_index(drop=True)
    y_true = meta["stage_true_id"].values.astype(int)
    input_dim = len(selected)

    t0 = time.time()
    model = mod.TCNGRUStageOnly(input_dim, base.BEST_ARCH["channels"], base.BEST_ARCH["gru_hidden"], base.BEST_ARCH["dropout"])
    model, info = mod.train_stage_model(model, tr_pack, va_pack, f"TCN_GRU_diag_seed{train_seed}",
                                         epochs=base.EPOCHS, patience=base.PATIENCE)
    t1 = time.time()
    y_pred, prob = mod.predict_stage_model(model, te_pack)

    row = mod.metrics_row("TCN-GRU", "Relative-stage TCN-GRU (diagnostic)", "Relative-stage", "TCN-GRU", y_true, y_pred, prob)
    pred_df = meta[["condition", "cut_index", "stage_true"]].copy()
    pred_df["pred"] = [base.ID_TO_STAGE[int(v)] for v in y_pred]
    pred_df["prob_E"], pred_df["prob_M"], pred_df["prob_L"] = prob[:, 0], prob[:, 1], prob[:, 2]

    save_run(out_dir, "TCN-GRU", train_seed, hashes,
             {"channels": base.BEST_ARCH["channels"], "gru_hidden": base.BEST_ARCH["gru_hidden"],
              "dropout": base.BEST_ARCH["dropout"], "lr": base.BEST_ARCH["lr"],
              "epochs": base.EPOCHS, "patience": base.PATIENCE},
             [row], pred_df, t0, t1, info)


def run_multitask_tcn_gru(train_seed: int, hashes: dict):
    base = import_base()
    selected, feat_train, feat_val, feat_test = load_frozen(base)
    L = base.BEST_ARCH["L"]
    out_dir_b11 = RESULTS_DIR / "multitask_tcn_gru" / f"seed{train_seed}"
    out_dir_b12 = RESULTS_DIR / "dc_psr" / f"seed{train_seed}"
    mod = import_comparison_script(train_seed, out_dir_b11)

    set_train_seed(train_seed)
    base.RANDOM_SEED = train_seed  # base.train_model() internally re-calls base.set_seed(base.RANDOM_SEED); keep consistent
    tr_pack, va_pack, te_pack = build_packs(base, selected, feat_train, feat_val, feat_test, L)
    meta = te_pack["meta"].copy().reset_index(drop=True)
    y_true = meta["stage_true_id"].values.astype(int)
    input_dim = len(selected)

    t0 = time.time()
    mt_model, hist, best_score, best_epoch = base.train_model(tr_pack, va_pack, input_dim)
    t1 = time.time()
    out_dir_b11.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(mt_model.state_dict(), out_dir_b11 / f"multitask_tcn_gru_seed{train_seed}.pth")

    pred_test_raw = base.predict_model(mt_model, te_pack)
    b12_params = mod.B12_PARAMS
    pred_test = base.apply_probability_inference(pred_test_raw, b12_params)

    y_pred_b11 = pred_test["stage_pred_raw"].values.astype(int)
    prob_b11 = pred_test[[f"raw_prob_{s}" for s in base.STAGE_NAMES]].values
    y_pred_b12 = pred_test["stage_pred_final"].values.astype(int)
    prob_b12 = pred_test[[f"final_prob_{s}" for s in base.STAGE_NAMES]].values

    row_b11 = mod.metrics_row("B11", "Multi-task TCN-GRU (diagnostic)", "Relative-stage", "TCN-GRU + auxiliary heads", y_true, y_pred_b11, prob_b11)
    row_b12 = mod.metrics_row("B12", "DC-PSR / FGDS-PSI (diagnostic)", "Relative-stage", "Proposed", y_true, y_pred_b12, prob_b12)

    b11_val = hist.loc[hist["score"].idxmin()] if "score" in hist.columns else hist.iloc[-1]
    best_info = {
        "best_epoch": int(best_epoch),
        "best_val_acc": float(b11_val["val_acc"]),
        "best_val_macro_f1": float(b11_val["val_macro_f1"]),
        "best_val_MRec": float(b11_val["val_middle_recall"]),
    }

    pred_df = meta[["condition", "cut_index", "stage_true"]].copy()
    pred_df["pred_B11"] = [base.ID_TO_STAGE[int(v)] for v in y_pred_b11]
    pred_df["prob_E_B11"], pred_df["prob_M_B11"], pred_df["prob_L_B11"] = prob_b11[:, 0], prob_b11[:, 1], prob_b11[:, 2]
    pred_df["pred_B12"] = [base.ID_TO_STAGE[int(v)] for v in y_pred_b12]
    pred_df["prob_E_B12"], pred_df["prob_M_B12"], pred_df["prob_L_B12"] = prob_b12[:, 0], prob_b12[:, 1], prob_b12[:, 2]

    save_run(out_dir_b11, "Multi-task TCN-GRU", train_seed, hashes,
             {"channels": base.BEST_ARCH["channels"], "gru_hidden": base.BEST_ARCH["gru_hidden"],
              "dropout": base.BEST_ARCH["dropout"], "lr": base.BEST_ARCH["lr"],
              "epochs": base.EPOCHS, "patience": base.PATIENCE,
              "LAMBDA_STAGE": base.LAMBDA_STAGE, "LAMBDA_FINE": base.LAMBDA_FINE,
              "LAMBDA_Q": base.LAMBDA_Q, "LAMBDA_MONO": base.LAMBDA_MONO},
             [row_b11], pred_df[[c for c in pred_df.columns if "B12" not in c]], t0, t1, best_info)
    save_run(out_dir_b12, "DC-PSR", train_seed, hashes,
             {"B12_PARAMS": b12_params, "checkpoint": "shared with Multi-task TCN-GRU (same training run)"},
             [row_b12], pred_df[[c for c in pred_df.columns if "B11" not in c]], t0, t1, best_info)


def run_htt_net(train_seed: int, hashes: dict):
    base = import_base()
    selected, feat_train, feat_val, feat_test = load_frozen(base)
    L = base.BEST_ARCH["L"]
    out_dir = RESULTS_DIR / "htt_net" / f"seed{train_seed}"
    T = import_htt_train(out_dir)

    # Frozen SOURCE_ONLY_TUNED config -- byte-identical to
    # baselines/htt_net/train_final_tuned.py / best_source_only_config.yaml.
    T.HTT_ARCH = {"embed_dim": 64, "depths": (2, 2, 2, 2), "num_heads": 4, "window_size": 3, "dropout": 0.20}
    T.TRAIN_CFG["lr"] = 5e-4
    T.METHOD_ID, T.METHOD_NAME = "B13", "HTT-Net"
    T.STAGE_DEF, T.MODEL_TYPE = "Relative-stage", "HTT-Net (diagnostic)"

    set_train_seed(train_seed)
    tr_pack, va_pack, te_pack = build_packs(base, selected, feat_train, feat_val, feat_test, L)
    meta = te_pack["meta"].copy().reset_index(drop=True)
    y_true = meta["stage_true_id"].values.astype(int)
    input_dim = len(selected)

    t0 = time.time()
    model = T.HTTNet(input_dim=input_dim, num_classes=3, **T.HTT_ARCH)
    model, hist, best_info = T.train_htt_model(model, tr_pack, va_pack, T.TRAIN_CFG["epochs"], T.TRAIN_CFG["patience"])
    t1 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(model.state_dict(), out_dir / f"htt_net_seed{train_seed}.pth")

    y_pred, prob = T.predict_stage_model(model, te_pack)
    row = T.metrics_row(y_true, y_pred, prob)

    pred_df = meta[["condition", "cut_index", "stage_true"]].copy()
    pred_df["pred"] = [base.ID_TO_STAGE[int(v)] for v in y_pred]
    pred_df["prob_E"], pred_df["prob_M"], pred_df["prob_L"] = prob[:, 0], prob[:, 1], prob[:, 2]

    save_run(out_dir, "HTT-Net", train_seed, hashes,
             {**T.HTT_ARCH, "lr": T.TRAIN_CFG["lr"], "epochs": T.TRAIN_CFG["epochs"], "patience": T.TRAIN_CFG["patience"],
              "n_parameters": model.num_parameters()},
             [row], pred_df, t0, t1, best_info)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=VALID_METHODS)
    parser.add_argument("--train_seed", type=int, required=True)
    args = parser.parse_args()

    hashes = assert_frozen_artifacts_unchanged()
    print(f"[protocol] frozen artifacts verified OK (preprocess_seed={hashes['preprocess_seed']})")
    print(f"[protocol] method={args.method} train_seed={args.train_seed}")

    dispatch = {
        "rf": run_rf,
        "tcn_gru": run_tcn_gru,
        "multitask_tcn_gru": run_multitask_tcn_gru,
        "htt_net": run_htt_net,
    }
    dispatch[args.method](args.train_seed, hashes)
    print(f"[protocol] DONE: method={args.method} train_seed={args.train_seed}")


if __name__ == "__main__":
    main()

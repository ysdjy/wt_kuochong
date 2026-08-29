# -*- coding: utf-8 -*-
r"""
B9 (DC-PHSR, legacy code id DC-PSR/B12) + its B3 (Multi-task TCN-GRU) backbone,
PHM2010 D1 (train=C1+C4, test=C6), strict TRAIN_SEED isolation.

Adapted copy of `protocol_diagnostic_fixed_preprocess/scripts/run_diagnostic_seed.py`
(that original file is left completely untouched). Only change: output
directories are redirected from the old diagnostic results/ tree into this
new project's B3/B9 method folders, and frozen preprocessing is read from
the copied `shared/reproducibility/PHM2010_D1_frozen_preprocess/` (byte-
identical to the original `protocol_diagnostic_fixed_preprocess/frozen_preprocess/`,
verified by sha256 before copy). Training/inference logic is imported live,
read-only, from the original `代码/main_experiment_3_fgds_psi_optimized.py`
and `代码/7.4对比实验.py` -- never copied or modified.

Usage:
    python run_seed_d1.py --train_seed 42
    python run_seed_d1.py --train_seed 52
    ... 62, 72, 82
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent                       # .../B9_DC_PHSR/code
B9_DIR = THIS_DIR.parent                                          # .../B9_DC_PHSR
METHOD_ROOT = B9_DIR.parent                                       # .../01_主对比实验/PHM2010
PROJECT_ROOT = METHOD_ROOT.parents[2]                              # repo root
CODE_DIR = PROJECT_ROOT / "代码"                                   # original, read-only import
FROZEN_DIR = PROJECT_ROOT / "扩充实验代码" / "shared" / "reproducibility" / "PHM2010_D1_frozen_preprocess"
RESULTS_DIR_B3 = METHOD_ROOT / "B3_Multitask_TCN_GRU" / "results"
RESULTS_DIR_B9 = METHOD_ROOT / "B9_DC_PHSR" / "results"

PREPROCESS_SEED = 42

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


def load_frozen():
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
    os.environ.setdefault("FGDS_RUN_DIR", str(Path(tempfile.gettempdir()) / "b9_dcphsr_d1_base_side_outputs"))
    sys.path.insert(0, str(CODE_DIR))
    import main_experiment_3_fgds_psi_optimized as base  # noqa
    base.FEATURE_FILE = PROJECT_ROOT / "baselines" / "htt_net" / "data" / "run_level_features_all.csv"
    return base


def import_comparison_script(train_seed: int, out_dir: Path):
    os.environ["COMPARISON_RECHECK_DIR"] = str(out_dir / "_scratch_side_outputs")
    script_path = CODE_DIR / "7.4对比实验.py"
    mod_name = f"b9_dcphsr_d1_comparison_seed{train_seed}"
    spec = importlib.util.spec_from_file_location(mod_name, script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def save_run(out_dir: Path, method: str, train_seed: int, hashes: dict, config: dict,
             metrics_rows: list, pred_df: pd.DataFrame, t_start: float, t_end: float, best_info: dict,
             training_log: pd.DataFrame | None = None):
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
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_json(out_dir / "metrics.json", orient="records", indent=2, force_ascii=False)
    if pred_df is not None:
        pred_df.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    if training_log is not None:
        training_log.to_csv(out_dir / "training_log.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["epoch", "note"]).to_csv(out_dir / "training_log.csv", index=False, encoding="utf-8-sig")
    (out_dir / "DONE.flag").write_text(f"done at {t_end}\n", encoding="utf-8")
    print(f"[save_run] wrote {out_dir}")


def run_b3_b9(train_seed: int, hashes: dict):
    base = import_base()
    selected, feat_train, feat_val, feat_test = load_frozen()
    L = base.BEST_ARCH["L"]
    out_dir_b3 = RESULTS_DIR_B3 / f"seed{train_seed}"
    out_dir_b9 = RESULTS_DIR_B9 / f"seed{train_seed}"
    mod = import_comparison_script(train_seed, out_dir_b3)

    set_train_seed(train_seed)
    base.RANDOM_SEED = train_seed  # base.train_model() internally re-calls base.set_seed(base.RANDOM_SEED); keep consistent
    tr_pack, va_pack, te_pack = build_packs(base, selected, feat_train, feat_val, feat_test, L)
    meta = te_pack["meta"].copy().reset_index(drop=True)
    y_true = meta["stage_true_id"].values.astype(int)
    input_dim = len(selected)

    t0 = time.time()
    mt_model, hist, best_score, best_epoch = base.train_model(tr_pack, va_pack, input_dim)
    t1 = time.time()
    out_dir_b3.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(mt_model.state_dict(), out_dir_b3 / f"multitask_tcn_gru_seed{train_seed}.pth")

    pred_test_raw = base.predict_model(mt_model, te_pack)
    b12_params = mod.B12_PARAMS  # DC-PHSR / DC-PSR / B12 frozen inference params, read live from 代码/7.4对比实验.py
    pred_test = base.apply_probability_inference(pred_test_raw, b12_params)

    y_pred_b3 = pred_test["stage_pred_raw"].values.astype(int)
    prob_b3 = pred_test[[f"raw_prob_{s}" for s in base.STAGE_NAMES]].values
    y_pred_b9 = pred_test["stage_pred_final"].values.astype(int)
    prob_b9 = pred_test[[f"final_prob_{s}" for s in base.STAGE_NAMES]].values

    row_b3 = mod.metrics_row("B3", "Multi-task TCN-GRU (B9 backbone)", "Relative-stage", "TCN-GRU + auxiliary heads", y_true, y_pred_b3, prob_b3)
    row_b9 = mod.metrics_row("B9", "DC-PHSR (legacy id: DC-PSR/B12/FGDS-PSI)", "Relative-stage", "Proposed", y_true, y_pred_b9, prob_b9)

    b3_val = hist.loc[hist["score"].idxmin()] if "score" in hist.columns else hist.iloc[-1]
    best_info = {
        "best_epoch": int(best_epoch),
        "best_val_acc": float(b3_val["val_acc"]),
        "best_val_macro_f1": float(b3_val["val_macro_f1"]),
        "best_val_MRec": float(b3_val["val_middle_recall"]),
    }

    pred_df_common = meta[["condition", "cut_index", "stage_true"]].copy()
    pred_df_b3 = pred_df_common.copy()
    pred_df_b3["pred"] = [base.ID_TO_STAGE[int(v)] for v in y_pred_b3]
    pred_df_b3["prob_E"], pred_df_b3["prob_M"], pred_df_b3["prob_L"] = prob_b3[:, 0], prob_b3[:, 1], prob_b3[:, 2]

    pred_df_b9 = pred_df_common.copy()
    pred_df_b9["pred"] = [base.ID_TO_STAGE[int(v)] for v in y_pred_b9]
    pred_df_b9["prob_E"], pred_df_b9["prob_M"], pred_df_b9["prob_L"] = prob_b9[:, 0], prob_b9[:, 1], prob_b9[:, 2]
    # carry q (wear-quantity) estimate columns through if base.apply_probability_inference / predict_model produced them
    for qcol in ("q_hat", "q_true", "q_pred", "wear_q_hat", "wear_q_true"):
        if qcol in pred_test.columns:
            pred_df_b9[qcol] = pred_test[qcol].values
            pred_df_b3[qcol] = pred_test[qcol].values if qcol in pred_test.columns else np.nan

    training_log = hist.reset_index(drop=True) if hist is not None else None

    save_run(out_dir_b3, "B3_Multitask_TCN_GRU", train_seed, hashes,
             {"channels": base.BEST_ARCH["channels"], "gru_hidden": base.BEST_ARCH["gru_hidden"],
              "dropout": base.BEST_ARCH["dropout"], "lr": base.BEST_ARCH["lr"],
              "epochs": base.EPOCHS, "patience": base.PATIENCE,
              "LAMBDA_STAGE": base.LAMBDA_STAGE, "LAMBDA_FINE": base.LAMBDA_FINE,
              "LAMBDA_Q": base.LAMBDA_Q, "LAMBDA_MONO": base.LAMBDA_MONO},
             [row_b3], pred_df_b3, t0, t1, best_info, training_log)
    save_run(out_dir_b9, "B9_DC_PHSR", train_seed, hashes,
             {"B12_PARAMS_legacy_name": b12_params, "checkpoint": "shared with B3 Multi-task TCN-GRU (same training run, not independently trained)"},
             [row_b9], pred_df_b9, t0, t1, best_info, training_log)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_seed", type=int, required=True)
    args = parser.parse_args()

    hashes = assert_frozen_artifacts_unchanged()
    print(f"[protocol] frozen artifacts verified OK (preprocess_seed={hashes['preprocess_seed']})")
    print(f"[protocol] method=B3+B9 train_seed={args.train_seed}")

    run_b3_b9(args.train_seed, hashes)
    print(f"[protocol] DONE: method=B3+B9 train_seed={args.train_seed}")


if __name__ == "__main__":
    main()

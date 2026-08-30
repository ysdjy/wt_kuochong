# -*- coding: utf-8 -*-
r"""
B9 (DC-PHSR, legacy id DC-PSR/B12) + its B3 (Multi-task TCN-GRU) backbone,
generalized across PHM2010 D1/D2/D3, strict TRAIN_SEED isolation.

Same core logic as
  05_统计检验/seed_statistics/B9_PHM2010_D1_seed_landscape/code/run_seed_landscape.py
generalized to take --task (selects frozen preprocessing) and --results_root
(so each of B9_PHM2010_D{1,2,3}_seed_landscape/ can call this one shared
script and write into its own results/seed{N}/, matching the established
per-task "seed landscape" folder convention instead of a single combined
results/{task}/seed{N}/ tree).

D2/D3 frozen preprocessing built by
  shared/reproducibility/build_phm_task_frozen_preprocess.py
using the already-validated verbatim reference split logic from
final_statistical_evidence/scripts/methods/common_pipeline.py::split_by_conditions
(copied from 代码/7.7跨工况实验.py). Architecture, hyperparameters, and B9's
inference params (B12_PARAMS) are UNCHANGED across D1/D2/D3.

Usage:
    python run_seed_task.py --task D2 --train_seed 0 \
        --results_root ../B9_PHM2010_D2_seed_landscape/results \
        --backbone_root ../B9_PHM2010_D2_seed_landscape/backbone_checkpoints
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
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None

THIS_DIR = Path(__file__).resolve().parent                          # .../seed_statistics/_shared_code
SEED_STATS_DIR = THIS_DIR.parent                                     # .../05_统计检验/seed_statistics
EXPAND_ROOT = SEED_STATS_DIR.parents[1]                                # .../扩充实验代码
PROJECT_ROOT = EXPAND_ROOT.parent                                       # repo root (论文/)
CODE_DIR = PROJECT_ROOT / "代码"

TASK_CONDITIONS = {
    "D1": {"train": ["C1", "C4"], "test": "C6"},
    "D2": {"train": ["C1", "C6"], "test": "C4"},
    "D3": {"train": ["C4", "C6"], "test": "C1"},
}
STAGE_ORDER = ["E", "M", "L"]
PREPROCESS_SEED = 42

warnings.filterwarnings("ignore")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_dir_for(task: str) -> Path:
    return EXPAND_ROOT / "shared" / "reproducibility" / f"PHM2010_{task}_frozen_preprocess"


def assert_frozen_artifacts_unchanged(frozen_dir: Path):
    with open(frozen_dir / "manifest_hashes.json", "r", encoding="utf-8") as f:
        recorded = json.load(f)
    mismatches = []
    for name, expected in recorded["files"].items():
        actual = sha256_file(frozen_dir / name)
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


def load_frozen(frozen_dir: Path):
    with open(frozen_dir / "selected_features_seed42.json", "r", encoding="utf-8") as f:
        sel_info = json.load(f)
    selected = sel_info["selected_features_in_order"]
    feat_train = pd.read_csv(frozen_dir / "feat_train_frozen.csv")
    feat_val = pd.read_csv(frozen_dir / "feat_val_frozen.csv")
    feat_test = pd.read_csv(frozen_dir / "feat_test_frozen.csv")
    return selected, feat_train, feat_val, feat_test


def build_packs(base, selected, feat_train, feat_val, feat_test, L):
    tr_pack = base.make_pack(feat_train, selected, L, "final_train")
    va_pack = base.make_pack(feat_val, selected, L, "final_internal_val")
    te_pack = base.make_pack(feat_test, selected, L, "test")
    return tr_pack, va_pack, te_pack


def import_base():
    os.environ.setdefault("FGDS_RUN_DIR", str(Path(tempfile.gettempdir()) / "b9_task_runner_base_side_outputs"))
    sys.path.insert(0, str(CODE_DIR))
    import main_experiment_3_fgds_psi_optimized as base  # noqa
    base.FEATURE_FILE = PROJECT_ROOT / "baselines" / "htt_net" / "data" / "run_level_features_all.csv"
    return base


def import_comparison_script(train_seed: int, task: str, out_dir: Path):
    os.environ["COMPARISON_RECHECK_DIR"] = str(out_dir / "_scratch_side_outputs")
    script_path = CODE_DIR / "7.4对比实验.py"
    mod_name = f"b9_task_runner_comparison_{task}_seed{train_seed}"
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


def run_one(task: str, train_seed: int, results_root: Path, backbone_root: Path,
            b3_results_root: Path | None = None):
    if task not in TASK_CONDITIONS:
        raise ValueError(f"unknown task {task}")
    train_conditions = TASK_CONDITIONS[task]["train"]
    test_condition = TASK_CONDITIONS[task]["test"]
    frozen_dir = frozen_dir_for(task)
    hashes = assert_frozen_artifacts_unchanged(frozen_dir)
    print(f"[protocol] frozen artifacts verified OK task={task} preprocess_seed={hashes['preprocess_seed']}")

    out_dir = results_root / f"seed{train_seed}"

    if (out_dir / "DONE.flag").exists():
        try:
            with open(out_dir / "run_meta.json", "r", encoding="utf-8") as f:
                prior = json.load(f)
            if (prior.get("split_hash") == hashes["files"]["split_manifest.csv"]
                    and prior.get("feature_hash") == hashes["files"]["selected_features_seed42.json"]
                    and prior.get("gmm_hash") == hashes["files"]["gmm_seed42.pkl"]
                    and prior.get("window_hash") == hashes["files"]["window_manifest.csv"]
                    and prior.get("train_seed") == train_seed and prior.get("task") == task):
                print(f"[resume] task={task} seed={train_seed} already DONE with matching hashes -- skipping")
                return "skipped"
        except Exception:
            pass

    base = import_base()
    selected, feat_train, feat_val, feat_test = load_frozen(frozen_dir)
    L = base.BEST_ARCH["L"]
    mod = import_comparison_script(train_seed, task, out_dir)

    set_train_seed(train_seed)
    base.RANDOM_SEED = train_seed
    tr_pack, va_pack, te_pack = build_packs(base, selected, feat_train, feat_val, feat_test, L)
    meta = te_pack["meta"].copy().reset_index(drop=True)
    y_true = meta["stage_true_id"].values.astype(int)
    input_dim = len(selected)

    t0 = time.time()
    mt_model, hist, best_score, best_epoch = base.train_model(tr_pack, va_pack, input_dim)
    t1 = time.time()
    backbone_root.mkdir(parents=True, exist_ok=True)
    import torch
    ckpt_path = backbone_root / f"b3_backbone_{task}_seed{train_seed}.pth"
    torch.save(mt_model.state_dict(), ckpt_path)
    backbone_checkpoint_hash = sha256_file(ckpt_path)

    pred_test_raw = base.predict_model(mt_model, te_pack)
    b12_params = mod.B12_PARAMS
    pred_test = base.apply_probability_inference(pred_test_raw, b12_params)

    # B3 = the same backbone's raw output, BEFORE B9's probability post-processing
    # (never trained separately; same checkpoint as B9, see backbone_checkpoint_hash).
    y_pred_b3 = pred_test["stage_pred_raw"].values.astype(int)
    prob_b3 = pred_test[[f"raw_prob_{s}" for s in base.STAGE_NAMES]].values

    y_pred_b9 = pred_test["stage_pred_final"].values.astype(int)
    prob_b9 = pred_test[[f"final_prob_{s}" for s in base.STAGE_NAMES]].values

    row_b9 = mod.metrics_row("B9", f"DC-PHSR (legacy id: DC-PSR/B12/FGDS-PSI) -- {task}", "Relative-stage", "Proposed", y_true, y_pred_b9, prob_b9)
    row = row_b9.iloc[0] if hasattr(row_b9, "iloc") else row_b9
    row_b3_raw = mod.metrics_row("B3", f"Multi-task TCN-GRU (B9 backbone, raw pre-inference) -- {task}", "Relative-stage", "TCN-GRU + auxiliary heads", y_true, y_pred_b3, prob_b3)
    row3 = row_b3_raw.iloc[0] if hasattr(row_b3_raw, "iloc") else row_b3_raw

    q_hat = q_true = None
    if "q_hat" in pred_test.columns and "q_true" in pred_test.columns:
        q_hat = pred_test["q_hat"].values.astype(float)
        q_true = pred_test["q_true"].values.astype(float)
    q_mae = q_rmse = q_r2 = spearman = np.nan
    if q_hat is not None:
        q_mae = mean_absolute_error(q_true, q_hat)
        q_rmse = np.sqrt(mean_squared_error(q_true, q_hat))
        q_r2 = r2_score(q_true, q_hat)
        if spearmanr is not None:
            try:
                spearman = float(spearmanr(q_true, q_hat).correlation)
            except Exception:
                spearman = np.nan

    seed_summary = {
        "Method": "B9_DC_PHSR", "Dataset": "PHM2010", "Task": task, "Seed": train_seed,
        "Acc": row["Acc"], "Macro-F1": row["Macro-F1"], "E-F1": row["E-F1"], "M-F1": row["M-F1"],
        "L-F1": row["L-F1"], "M-Precision": row["M-Pre"], "M-Recall": row["M-Rec"],
        "M_to_E": row["M→E"], "M_to_L": row["M→L"], "Rev": row["Rev"], "Jump": row["Jump"],
        "Smooth": row["Smooth"], "q-MAE": q_mae, "q-RMSE": q_rmse, "q-R2": q_r2, "Spearman": spearman,
        "best_epoch": int(best_epoch), "n_epochs_run": int(len(hist)), "training_seconds": t1 - t0,
    }

    pred_df = meta[["condition", "cut_index", "stage_true"]].copy()
    pred_df["pred"] = [base.ID_TO_STAGE[int(v)] for v in y_pred_b9]
    pred_df["prob_E"], pred_df["prob_M"], pred_df["prob_L"] = prob_b9[:, 0], prob_b9[:, 1], prob_b9[:, 2]
    if q_hat is not None:
        pred_df["q_hat"] = q_hat
        pred_df["q_true"] = q_true

    cm = confusion_matrix(y_true, y_pred_b9, labels=[0, 1, 2])
    cm_df = pd.DataFrame(cm, index=[f"true_{s}" for s in STAGE_ORDER], columns=[f"pred_{s}" for s in STAGE_ORDER])

    out_dir.mkdir(parents=True, exist_ok=True)
    config_resolved = {
        "method": "B9_DC_PHSR", "dataset": "PHM2010", "task": task,
        "train_conditions": train_conditions, "test_condition": test_condition,
        "preprocess_seed": PREPROCESS_SEED, "train_seed": train_seed,
        "architecture": {"channels": base.BEST_ARCH["channels"], "gru_hidden": base.BEST_ARCH["gru_hidden"],
                          "dropout": base.BEST_ARCH["dropout"], "lr": base.BEST_ARCH["lr"], "L": L},
        "training": {"epochs": base.EPOCHS, "patience": base.PATIENCE,
                     "LAMBDA_STAGE": base.LAMBDA_STAGE, "LAMBDA_FINE": base.LAMBDA_FINE,
                     "LAMBDA_Q": base.LAMBDA_Q, "LAMBDA_MONO": base.LAMBDA_MONO},
        "B12_PARAMS_legacy_name": b12_params,
    }
    config_hash = hashlib.sha256(json.dumps(config_resolved, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    with open(out_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        for k, v in config_resolved.items():
            f.write(f"{k}: {json.dumps(v, ensure_ascii=False)}\n")

    pairing = {
        "backbone_checkpoint_hash": backbone_checkpoint_hash,
        "backbone_checkpoint_path": str(ckpt_path),
        "config_hash": config_hash,
        "paired_method": "B9_DC_PHSR<->B3_Multitask_TCN_GRU (same backbone, same seed, no separate B3 training)",
    }
    meta_json = {
        "method": "B9_DC_PHSR", "dataset": "PHM2010", "task": task,
        "train_conditions": train_conditions, "test_condition": test_condition,
        "preprocess_seed": PREPROCESS_SEED, "train_seed": train_seed,
        "git_commit": git_commit(),
        "feature_hash": hashes["files"]["selected_features_seed42.json"],
        "split_hash": hashes["files"]["split_manifest.csv"],
        "gmm_hash": hashes["files"]["gmm_seed42.pkl"],
        "window_hash": hashes["files"]["window_manifest.csv"],
        "B12_PARAMS": b12_params,
        "start_time": t0, "end_time": t1, "training_seconds": t1 - t0, "best_epoch": int(best_epoch),
        "n_test": len(meta), **pairing,
    }
    pd.DataFrame([seed_summary]).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    json_row = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in seed_summary.items()}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump([json_row], f, indent=2, default=str)
    pred_df.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    hist.reset_index(drop=True).to_csv(out_dir / "training_log.csv", index=False, encoding="utf-8-sig")
    hist.reset_index(drop=True).to_csv(out_dir / "validation_history.csv", index=False, encoding="utf-8-sig")
    cm_df.to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2, default=str)
    (out_dir / "DONE.flag").write_text(f"done at {t1}\n", encoding="utf-8")
    print(f"[save_run] task={task} seed={train_seed} Acc={seed_summary['Acc']:.4f} MacroF1={seed_summary['Macro-F1']:.4f} "
          f"MF1={seed_summary['M-F1']:.4f} MRec={seed_summary['M-Recall']:.4f} Smooth={seed_summary['Smooth']:.4f} n_test={len(meta)}")

    # ---- Paired B3 (Multi-task TCN-GRU raw backbone output, same checkpoint/seed/preprocessing) ----
    if b3_results_root is not None:
        b3_out_dir = b3_results_root / f"seed{train_seed}"
        b3_out_dir.mkdir(parents=True, exist_ok=True)
        b3_summary = {
            "Method": "B3_Multitask_TCN_GRU", "Dataset": "PHM2010", "Task": task, "Seed": train_seed,
            "Acc": row3["Acc"], "Macro-F1": row3["Macro-F1"], "E-F1": row3["E-F1"], "M-F1": row3["M-F1"],
            "L-F1": row3["L-F1"], "M-Precision": row3["M-Pre"], "M-Recall": row3["M-Rec"],
            "M_to_E": row3["M→E"], "M_to_L": row3["M→L"], "Rev": row3["Rev"], "Jump": row3["Jump"],
            "Smooth": row3["Smooth"], "q-MAE": q_mae, "q-RMSE": q_rmse, "q-R2": q_r2, "Spearman": spearman,
            "best_epoch": int(best_epoch), "n_epochs_run": int(len(hist)), "training_seconds": t1 - t0,
        }
        pred_df_b3 = meta[["condition", "cut_index", "stage_true"]].copy()
        pred_df_b3["pred"] = [base.ID_TO_STAGE[int(v)] for v in y_pred_b3]
        pred_df_b3["prob_E"], pred_df_b3["prob_M"], pred_df_b3["prob_L"] = prob_b3[:, 0], prob_b3[:, 1], prob_b3[:, 2]
        if q_hat is not None:
            pred_df_b3["q_hat"] = q_hat
            pred_df_b3["q_true"] = q_true
        cm3 = confusion_matrix(y_true, y_pred_b3, labels=[0, 1, 2])
        cm3_df = pd.DataFrame(cm3, index=[f"true_{s}" for s in STAGE_ORDER], columns=[f"pred_{s}" for s in STAGE_ORDER])
        b3_meta_json = {
            "method": "B3_Multitask_TCN_GRU", "dataset": "PHM2010", "task": task,
            "train_conditions": train_conditions, "test_condition": test_condition,
            "preprocess_seed": PREPROCESS_SEED, "train_seed": train_seed, "git_commit": git_commit(),
            "feature_hash": hashes["files"]["selected_features_seed42.json"],
            "split_hash": hashes["files"]["split_manifest.csv"],
            "gmm_hash": hashes["files"]["gmm_seed42.pkl"],
            "window_hash": hashes["files"]["window_manifest.csv"],
            "start_time": t0, "end_time": t1, "training_seconds": t1 - t0, "best_epoch": int(best_epoch),
            "n_test": len(meta), **pairing,
            "note": "raw backbone output, computed from the SAME checkpoint as the paired B9 run in this same process -- never retrained separately",
        }
        pd.DataFrame([b3_summary]).to_csv(b3_out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
        b3_json_row = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in b3_summary.items()}
        with open(b3_out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump([b3_json_row], f, indent=2, default=str)
        pred_df_b3.to_csv(b3_out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
        cm3_df.to_csv(b3_out_dir / "confusion_matrix.csv", encoding="utf-8-sig")
        with open(b3_out_dir / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(b3_meta_json, f, indent=2, default=str)
        (b3_out_dir / "DONE.flag").write_text(f"done at {t1}\n", encoding="utf-8")
        print(f"[save_run] (paired B3) task={task} seed={train_seed} Acc={b3_summary['Acc']:.4f} MacroF1={b3_summary['Macro-F1']:.4f}")

    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_CONDITIONS.keys()))
    ap.add_argument("--train_seed", type=int, required=True)
    ap.add_argument("--results_root", required=True, type=Path)
    ap.add_argument("--backbone_root", required=True, type=Path)
    ap.add_argument("--b3_results_root", required=False, default=None, type=Path)
    args = ap.parse_args()

    out_dir = args.results_root / f"seed{args.train_seed}"
    try:
        status = run_one(args.task, args.train_seed, args.results_root, args.backbone_root, args.b3_results_root)
        print(f"[protocol] DONE({status}): task={args.task} train_seed={args.train_seed}")
    except Exception as e:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "error.log", "w", encoding="utf-8") as f:
            import traceback
            f.write(f"task={args.task} seed={args.train_seed}\n")
            traceback.print_exc(file=f)
        (out_dir / "FAILED.flag").write_text(f"failed: {e}\n", encoding="utf-8")
        print(f"[protocol] FAILED: task={args.task} train_seed={args.train_seed}: {e}")
        raise


if __name__ == "__main__":
    main()

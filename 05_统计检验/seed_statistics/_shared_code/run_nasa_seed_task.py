# -*- coding: utf-8 -*-
r"""
B9 (DC-PHSR, legacy id DC-PSR/B12) + its B11 (Multi-task TCN-GRU) backbone,
NASA Milling N1-N4, strict TRAIN_SEED isolation.

Reuses `代码/9.1nasa数据实验.py`'s model/training/inference functions
(TCNGRUMultiTaskModel, train_model, predict_model, apply_dcpsr_inference,
PROB_PARAMS) via the same in-memory source-patch import used by
shared/reproducibility/build_nasa_task_frozen_preprocess.py (only the two
off-machine hardcoded path literals are patched; the file on disk is never
touched). Metrics computed via the unified, dataset-agnostic
final_statistical_evidence/scripts/methods/common_pipeline.py::full_metrics
(same formulas already used for PHM D1/D2/D3), for cross-dataset consistency.

set_seed: NASA's own train_model() never calls set_seed internally (unlike
PHM's base.train_model()) -- so this runner supplies its own stronger
set_train_seed (cudnn.deterministic=True etc., matching the PHM D1/D2/D3
runner) immediately before model instantiation, per the task's own
instruction to reuse the more complete D1 implementation rather than NASA's
weaker one.

Usage:
    python run_nasa_seed_task.py --task N1 --train_seed 0 \
        --results_root ../B9_NASA_N1_seed_landscape/results \
        --backbone_root ../B9_NASA_N1_seed_landscape/backbone_checkpoints
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
import time
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

THIS_DIR = Path(__file__).resolve().parent                          # .../seed_statistics/_shared_code
SEED_STATS_DIR = THIS_DIR.parent                                     # .../05_统计检验/seed_statistics
EXPAND_ROOT = SEED_STATS_DIR.parents[1]                                # .../扩充实验代码
PROJECT_ROOT = EXPAND_ROOT.parent                                       # repo root (论文/)
CODE_DIR = PROJECT_ROOT / "代码"
NASA_SCRIPT = CODE_DIR / "9.1nasa数据实验.py"
MILL_MAT = PROJECT_ROOT / "mill" / "mill.mat"
COMMON_PIPELINE_PATH = PROJECT_ROOT / "final_statistical_evidence" / "scripts" / "methods" / "common_pipeline.py"

PREPROCESS_SEED = 42
STAGE_ORDER = ["E", "M", "L"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_dir_for(task: str) -> Path:
    return EXPAND_ROOT / "shared" / "reproducibility" / f"NASA_{task}_frozen_preprocess"


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


def import_nasa_module():
    src = NASA_SCRIPT.read_text(encoding="utf-8")
    safe_out_dir = Path(tempfile.gettempdir()) / "nasa_runner_side_outputs"
    old_mat_line = 'MAT_FILE = Path(r"C:\\Users\\wangting\\Desktop\\博士开题\\公开数据\\1NASA\\mill.mat")'
    old_out_line = 'OUT_DIR = Path(r"C:\\Users\\wangting\\Desktop\\博士开题\\公开数据\\1NASA\\nasa_dcpsr_results_stageaware_opt")'
    assert old_mat_line in src, "MAT_FILE line not found -- old script changed unexpectedly, aborting"
    assert old_out_line in src, "OUT_DIR line not found -- old script changed unexpectedly, aborting"
    src = src.replace(old_mat_line, f'MAT_FILE = Path(r"{MILL_MAT}")')
    src = src.replace(old_out_line, f'OUT_DIR = Path(r"{safe_out_dir}")')
    mod = types.ModuleType("nasa_runner_9_1")
    mod.__file__ = str(NASA_SCRIPT)
    sys.modules["nasa_runner_9_1"] = mod
    exec(compile(src, str(NASA_SCRIPT), "exec"), mod.__dict__)
    return mod


def import_common_pipeline():
    spec = importlib.util.spec_from_file_location("common_pipeline_nasa_runner", COMMON_PIPELINE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_frozen(frozen_dir: Path):
    with open(frozen_dir / "selected_features_seed42.json", "r", encoding="utf-8") as f:
        sel_info = json.load(f)
    selected = sel_info["selected_features_in_order"]
    feat_train = pd.read_csv(frozen_dir / "feat_train_frozen.csv")
    feat_val = pd.read_csv(frozen_dir / "feat_val_frozen.csv")
    feat_test = pd.read_csv(frozen_dir / "feat_test_frozen.csv")
    return selected, feat_train, feat_val, feat_test


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def run_one(task: str, train_seed: int, results_root: Path, backbone_root: Path,
            b3_results_root: Path | None = None):
    frozen_dir = frozen_dir_for(task)
    hashes = assert_frozen_artifacts_unchanged(frozen_dir)
    print(f"[protocol] frozen artifacts verified OK task={task} preprocess_seed={hashes['preprocess_seed']}")

    out_dir = results_root / f"seed{train_seed}"
    if (out_dir / "DONE.flag").exists():
        try:
            with open(out_dir / "run_meta.json", "r", encoding="utf-8") as f:
                prior = json.load(f)
            if (prior.get("gmm_hash") == hashes["files"]["gmm_seed42.pkl"]
                    and prior.get("split_hash") == hashes["files"]["split_manifest.csv"]
                    and prior.get("feature_hash") == hashes["files"]["selected_features_seed42.json"]
                    and prior.get("window_hash") == hashes["files"]["window_manifest.csv"]
                    and prior.get("train_seed") == train_seed and prior.get("task") == task):
                print(f"[resume] task={task} seed={train_seed} already DONE with matching hashes -- skipping")
                return "skipped"
        except Exception:
            pass

    nasa = import_nasa_module()
    cp = import_common_pipeline()
    selected, feat_train, feat_val, feat_test = load_frozen(frozen_dir)
    L = hashes["window_length_L"]

    set_train_seed(train_seed)
    tr_pack = nasa.build_sliding_windows(feat_train, selected, L)
    va_pack = nasa.build_sliding_windows(feat_val, selected, L)
    te_pack = nasa.build_sliding_windows(feat_test, selected, L)

    t0 = time.time()
    model = nasa.TCNGRUMultiTaskModel(len(selected))
    model, best_epoch = nasa.train_model(model, tr_pack, va_pack, multitask=True)
    t1 = time.time()
    backbone_root.mkdir(parents=True, exist_ok=True)
    import torch
    ckpt_path = backbone_root / f"nasa_backbone_{task}_seed{train_seed}.pth"
    torch.save(model.state_dict(), ckpt_path)
    backbone_checkpoint_hash = sha256_file(ckpt_path)

    pred_raw = nasa.predict_model(model, te_pack, multitask=True)
    b12_params = nasa.PROB_PARAMS
    pred = nasa.apply_dcpsr_inference(pred_raw, b12_params)

    y_true = pred["stage_id"].values.astype(int)
    # B3 = same backbone's raw output, BEFORE B9/DC-PSR post-processing (never retrained separately)
    prob_b3 = pred[["p_raw_E", "p_raw_M", "p_raw_L"]].values.astype(float)
    y_pred_b3 = prob_b3.argmax(axis=1)
    m3 = cp.full_metrics(y_true, y_pred_b3, prob_b3)

    prob_b9 = pred[["p_final_E", "p_final_M", "p_final_L"]].values.astype(float)
    y_pred_b9 = prob_b9.argmax(axis=1)

    m = cp.full_metrics(y_true, y_pred_b9, prob_b9)
    q_true = pred["q_true"].values.astype(float)
    q_hat = pred["q_hat"].values.astype(float)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    try:
        from scipy.stats import spearmanr
        spearman = float(spearmanr(q_true, q_hat).correlation)
    except Exception:
        spearman = np.nan
    q_mae = mean_absolute_error(q_true, q_hat)
    q_rmse = float(np.sqrt(mean_squared_error(q_true, q_hat)))
    q_r2 = r2_score(q_true, q_hat) if len(set(q_true.round(6))) > 1 else np.nan

    seed_summary = {
        "Method": "B9_DC_PHSR", "Dataset": "NASA_Milling", "Task": task, "Seed": train_seed,
        "Acc": m["Acc"], "Macro-F1": m["MacroF1"], "E-F1": m["E_F1"], "M-F1": m["M_F1"], "L-F1": m["L_F1"],
        "M-Precision": m["M_Precision"], "M-Recall": m["M_Recall"], "M_to_E": m["M_to_E"], "M_to_L": m["M_to_L"],
        "Rev": m["Rev"], "Jump": m["Jump"], "Smooth": m["Smooth"],
        "q-MAE": q_mae, "q-RMSE": q_rmse, "q-R2": q_r2, "Spearman": spearman,
        "best_epoch": int(best_epoch), "n_test": len(pred), "training_seconds": t1 - t0,
    }

    pred_df = pred[["case", "run", "stage_id", "q_true", "q_hat"]].copy()
    pred_df["pred"] = [STAGE_ORDER[i] for i in y_pred_b9]
    pred_df["prob_E"], pred_df["prob_M"], pred_df["prob_L"] = prob_b9[:, 0], prob_b9[:, 1], prob_b9[:, 2]

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred_b9, labels=[0, 1, 2])
    cm_df = pd.DataFrame(cm, index=[f"true_{s}" for s in STAGE_ORDER], columns=[f"pred_{s}" for s in STAGE_ORDER])

    out_dir.mkdir(parents=True, exist_ok=True)
    config_resolved = {
        "method": "B9_DC_PHSR", "dataset": "NASA_Milling", "task": task,
        "train_cases": hashes["train_cases"], "test_cases": hashes["test_cases"],
        "preprocess_seed": PREPROCESS_SEED, "train_seed": train_seed,
        "architecture": {"tcn_channels": nasa.TCN_CHANNELS, "gru_hidden": nasa.GRU_HIDDEN,
                          "dropout": nasa.DROP_OUT, "lr": nasa.LR, "L": L, "batch_size": nasa.BATCH_SIZE},
        "training": {"epochs": nasa.EPOCHS, "patience": nasa.PATIENCE},
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
        "method": "B9_DC_PHSR", "dataset": "NASA_Milling", "task": task,
        "train_cases": hashes["train_cases"], "test_cases": hashes["test_cases"],
        "preprocess_seed": PREPROCESS_SEED, "train_seed": train_seed, "git_commit": git_commit(),
        "feature_hash": hashes["files"]["selected_features_seed42.json"],
        "split_hash": hashes["files"]["split_manifest.csv"],
        "gmm_hash": hashes["files"]["gmm_seed42.pkl"],
        "window_hash": hashes["files"]["window_manifest.csv"],
        "B12_PARAMS": b12_params, "start_time": t0, "end_time": t1,
        "training_seconds": t1 - t0, "best_epoch": int(best_epoch), "n_test": len(pred), **pairing,
    }
    pd.DataFrame([seed_summary]).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    json_row = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in seed_summary.items()}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump([json_row], f, indent=2, default=str)
    pred_df.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"note": "9.1nasa数据实验.py::train_model returns only best_epoch, no per-epoch history"}]).to_csv(
        out_dir / "training_log.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"note": "same limitation as training_log.csv"}]).to_csv(
        out_dir / "validation_history.csv", index=False, encoding="utf-8-sig")
    cm_df.to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2, default=str)
    (out_dir / "DONE.flag").write_text(f"done at {t1}\n", encoding="utf-8")
    print(f"[save_run] task={task} seed={train_seed} Acc={seed_summary['Acc']:.4f} MacroF1={seed_summary['Macro-F1']:.4f} "
          f"MF1={seed_summary['M-F1']:.4f} MRec={seed_summary['M-Recall']:.4f} Smooth={seed_summary['Smooth']:.4f} n_test={len(pred)}")

    # ---- Paired B3 (Multi-task TCN-GRU raw backbone output, same checkpoint/seed/preprocessing) ----
    if b3_results_root is not None:
        b3_out_dir = b3_results_root / f"seed{train_seed}"
        b3_out_dir.mkdir(parents=True, exist_ok=True)
        b3_summary = {
            "Method": "B3_Multitask_TCN_GRU", "Dataset": "NASA_Milling", "Task": task, "Seed": train_seed,
            "Acc": m3["Acc"], "Macro-F1": m3["MacroF1"], "E-F1": m3["E_F1"], "M-F1": m3["M_F1"], "L-F1": m3["L_F1"],
            "M-Precision": m3["M_Precision"], "M-Recall": m3["M_Recall"], "M_to_E": m3["M_to_E"], "M_to_L": m3["M_to_L"],
            "Rev": m3["Rev"], "Jump": m3["Jump"], "Smooth": m3["Smooth"],
            "q-MAE": q_mae, "q-RMSE": q_rmse, "q-R2": q_r2, "Spearman": spearman,
            "best_epoch": int(best_epoch), "n_test": len(pred), "training_seconds": t1 - t0,
        }
        pred_df_b3 = pred[["case", "run", "stage_id", "q_true", "q_hat"]].copy()
        pred_df_b3["pred"] = [STAGE_ORDER[i] for i in y_pred_b3]
        pred_df_b3["prob_E"], pred_df_b3["prob_M"], pred_df_b3["prob_L"] = prob_b3[:, 0], prob_b3[:, 1], prob_b3[:, 2]
        cm3 = confusion_matrix(y_true, y_pred_b3, labels=[0, 1, 2])
        cm3_df = pd.DataFrame(cm3, index=[f"true_{s}" for s in STAGE_ORDER], columns=[f"pred_{s}" for s in STAGE_ORDER])
        b3_meta_json = {
            "method": "B3_Multitask_TCN_GRU", "dataset": "NASA_Milling", "task": task,
            "train_cases": hashes["train_cases"], "test_cases": hashes["test_cases"],
            "preprocess_seed": PREPROCESS_SEED, "train_seed": train_seed, "git_commit": git_commit(),
            "feature_hash": hashes["files"]["selected_features_seed42.json"],
            "split_hash": hashes["files"]["split_manifest.csv"],
            "gmm_hash": hashes["files"]["gmm_seed42.pkl"],
            "window_hash": hashes["files"]["window_manifest.csv"],
            "start_time": t0, "end_time": t1, "training_seconds": t1 - t0, "best_epoch": int(best_epoch),
            "n_test": len(pred), **pairing,
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
    ap.add_argument("--task", required=True, choices=["N1", "N2", "N3", "N4"])
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

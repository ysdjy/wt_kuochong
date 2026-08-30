# -*- coding: utf-8 -*-
r"""
B2 (Relative-stage TCN-GRU, stage-only -- paper's canonical B2; old script's
internal id "B10", see EXPERIMENT_REGISTRY.md), PHM2010 D1/D2/D3, strict
TRAIN_SEED isolation.

Sibling to run_seed_task.py (B9+B3) and run_b1_seed_task.py (B1) -- same
frozen-preprocessing source and output convention. Reuses
`代码/7.4对比实验.py`'s `TCNGRUStageOnly` model class and `train_stage_model`/
`predict_stage_model` functions (loaded read-only via the same
COMPARISON_RECHECK_DIR-redirected import pattern run_seed_task.py already
uses for B12_PARAMS) -- a single-stage-head TCN-GRU classifier, NOT the
multi-task model (that is B3/B9's shared backbone). No probability
post-processing (B9-only step); q-MAE/q-RMSE/q-R2/Spearman are legitimately
NaN (this model has no continuous degradation-index head).

Usage:
    python run_b2_seed_task.py --task D1 --train_seed 0 \
        --results_root ../B2_PHM2010_D1_seed_landscape/results \
        --backbone_root ../B2_PHM2010_D1_seed_landscape/backbone_checkpoints
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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

THIS_DIR = Path(__file__).resolve().parent
SEED_STATS_DIR = THIS_DIR.parent
EXPAND_ROOT = SEED_STATS_DIR.parents[1]
PROJECT_ROOT = EXPAND_ROOT.parent
# Self-contained: vendored copy inside this repo (git-clone portable), NOT the
# outside parent project -- see _shared_code/vendored_legacy/README.md.
CODE_DIR = THIS_DIR / "vendored_legacy"

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


def import_base():
    os.environ.setdefault("FGDS_RUN_DIR", str(Path(tempfile.gettempdir()) / "b2_task_runner_base_side_outputs"))
    sys.path.insert(0, str(CODE_DIR))
    import main_experiment_3_fgds_psi_optimized as base  # noqa
    # Self-contained: the already-committed copy inside this repo (never
    # actually read by this script -- frozen feat_{train,val,test} CSVs are
    # loaded directly -- but kept correct in case anything else touches it).
    base.FEATURE_FILE = EXPAND_ROOT / "data" / "PHM2010" / "features" / "run_level_features_all.csv"
    return base


def import_comparison_script(train_seed: int, task: str, out_dir: Path):
    os.environ["COMPARISON_RECHECK_DIR"] = str(out_dir / "_scratch_side_outputs")
    script_path = CODE_DIR / "7.4对比实验.py"
    mod_name = f"b2_task_runner_comparison_{task}_seed{train_seed}"
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


def build_packs(base, selected, feat_train, feat_val, feat_test, L):
    tr_pack = base.make_pack(feat_train, selected, L, "final_train")
    va_pack = base.make_pack(feat_val, selected, L, "final_internal_val")
    te_pack = base.make_pack(feat_test, selected, L, "test")
    return tr_pack, va_pack, te_pack


def run_one(task: str, train_seed: int, results_root: Path, backbone_root: Path):
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
    tr_pack, va_pack, te_pack = build_packs(base, selected, feat_train, feat_val, feat_test, L)
    meta = te_pack["meta"].copy().reset_index(drop=True)
    y_true = meta["stage_true_id"].values.astype(int)
    input_dim = len(selected)

    t0 = time.time()
    model = mod.TCNGRUStageOnly(input_dim, base.BEST_ARCH["channels"], base.BEST_ARCH["gru_hidden"], base.BEST_ARCH["dropout"])
    model, best_info = mod.train_stage_model(model, tr_pack, va_pack, f"b2_seed{train_seed}_{task}")
    t1 = time.time()
    backbone_root.mkdir(parents=True, exist_ok=True)
    import torch
    ckpt_path = backbone_root / f"b2_backbone_{task}_seed{train_seed}.pth"
    torch.save(model.state_dict(), ckpt_path)
    backbone_checkpoint_hash = sha256_file(ckpt_path)

    y_pred, prob = mod.predict_stage_model(model, te_pack)

    p_each, r_each, f1_each, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cmn = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    dy = np.diff(y_pred.astype(int))
    rev = int(np.sum(dy < 0))
    jump = int(np.sum(np.abs(dy) >= 2))
    smooth = float(np.mean(np.sum(np.abs(np.diff(prob, axis=0)), axis=1))) if len(prob) > 1 else float("nan")

    seed_summary = {
        "Method": "B2_TCN_GRU", "Dataset": "PHM2010", "Task": task, "Seed": train_seed,
        "Acc": float(accuracy_score(y_true, y_pred)), "Macro-F1": float(macro_f1),
        "E-F1": float(f1_each[0]), "M-F1": float(f1_each[1]), "L-F1": float(f1_each[2]),
        "M-Precision": float(p_each[1]), "M-Recall": float(r_each[1]),
        "M_to_E": float(cmn[1, 0]), "M_to_L": float(cmn[1, 2]),
        "Rev": rev, "Jump": jump, "Smooth": smooth,
        "q-MAE": np.nan, "q-RMSE": np.nan, "q-R2": np.nan, "Spearman": np.nan,
        "best_epoch": int(best_info["best_epoch"]), "n_epochs_run": int(best_info["best_epoch"]),
        "training_seconds": t1 - t0,
    }

    pred_df = meta[["condition", "cut_index", "stage_true"]].copy()
    pred_df["pred"] = [base.ID_TO_STAGE[int(v)] for v in y_pred]
    pred_df["prob_E"], pred_df["prob_M"], pred_df["prob_L"] = prob[:, 0], prob[:, 1], prob[:, 2]

    cm_df = pd.DataFrame(cm, index=[f"true_{s}" for s in STAGE_ORDER], columns=[f"pred_{s}" for s in STAGE_ORDER])

    out_dir.mkdir(parents=True, exist_ok=True)
    config_resolved = {
        "method": "B2_TCN_GRU", "dataset": "PHM2010", "task": task,
        "train_conditions": train_conditions, "test_condition": test_condition,
        "preprocess_seed": PREPROCESS_SEED, "train_seed": train_seed,
        "architecture": {"channels": base.BEST_ARCH["channels"], "gru_hidden": base.BEST_ARCH["gru_hidden"],
                          "dropout": base.BEST_ARCH["dropout"], "lr": base.BEST_ARCH["lr"], "L": L},
        "training": {"epochs": 120, "patience": 18},
    }
    config_hash = hashlib.sha256(json.dumps(config_resolved, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    with open(out_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        for k, v in config_resolved.items():
            f.write(f"{k}: {json.dumps(v, ensure_ascii=False)}\n")

    meta_json = {
        "method": "B2_TCN_GRU", "dataset": "PHM2010", "task": task,
        "train_conditions": train_conditions, "test_condition": test_condition,
        "preprocess_seed": PREPROCESS_SEED, "train_seed": train_seed,
        "git_commit": git_commit(),
        "feature_hash": hashes["files"]["selected_features_seed42.json"],
        "split_hash": hashes["files"]["split_manifest.csv"],
        "gmm_hash": hashes["files"]["gmm_seed42.pkl"],
        "window_hash": hashes["files"]["window_manifest.csv"],
        "start_time": t0, "end_time": t1, "training_seconds": t1 - t0,
        "best_epoch": int(best_info["best_epoch"]), "n_test": len(meta),
        "backbone_checkpoint_hash": backbone_checkpoint_hash, "backbone_checkpoint_path": str(ckpt_path),
        "config_hash": config_hash,
    }
    pd.DataFrame([seed_summary]).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    json_row = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in seed_summary.items()}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump([json_row], f, indent=2, default=str)
    pred_df.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([best_info]).to_csv(out_dir / "training_log.csv", index=False, encoding="utf-8-sig")
    cm_df.to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2, default=str)
    (out_dir / "DONE.flag").write_text(f"done at {t1}\n", encoding="utf-8")
    print(f"[save_run] task={task} seed={train_seed} Acc={seed_summary['Acc']:.4f} "
          f"MacroF1={seed_summary['Macro-F1']:.4f} MF1={seed_summary['M-F1']:.4f} "
          f"MRec={seed_summary['M-Recall']:.4f} Smooth={seed_summary['Smooth']:.4f} n_test={len(meta)}")
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_CONDITIONS.keys()))
    ap.add_argument("--train_seed", type=int, required=True)
    ap.add_argument("--results_root", required=True, type=Path)
    ap.add_argument("--backbone_root", required=True, type=Path)
    args = ap.parse_args()

    out_dir = args.results_root / f"seed{args.train_seed}"
    try:
        status = run_one(args.task, args.train_seed, args.results_root, args.backbone_root)
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

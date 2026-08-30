# -*- coding: utf-8 -*-
r"""
Generic seed-task runner for B6 (MTF-AViTK) / B7 (Dynamic GIN+TGP) / B8
(DP2Net) -- PHM2010 D1/D2/D3, strict TRAIN_SEED isolation.

Unlike B1/B2/B4/B5 (each with its own hand-ported vendored runner), this
script reuses `methods/Bx_xxx/adapter.py::ADAPTER_CLASS` DIRECTLY (read-only
import -- `methods/` + `shared/` are already committed to this repo, so this
is just as git-clone-portable as vendoring a copy). Only `prepare()` /
`build_model()` / `train()` / `predict()` are called (the expensive,
correctness-critical, already-validated parts) -- this script does its OWN
resume-check / common-evaluation-universe restriction / metric computation /
output-file writing, matching the exact schema used by
run_b1_seed_task.py/run_b4_seed_task.py/run_b5_seed_task.py/run_seed_task.py
(B9) so `aggregate_dataset.py` works unmodified across all methods.

Deliberately does NOT call `adapter.run()` -- that base-class method scores
metrics on each method's NATIVE run_id coverage (1-315 for B6/B7/B8, raw-
signal methods) rather than the common evaluation universe (run_id 12-315,
n=304) every other method in this repo's comparison table is scored on; its
own metric key names (`MacroF1`, `E_F1`, ...) also differ from this repo's
established schema (`Macro-F1`, `E-F1`, ...) and it never computes Spearman.

Checkpoints are NOT saved by default (B6's alone is ~1.2GB per README "READ
BEFORE RUNNING FOR REAL" -- these are standalone baselines with no B3-style
shared-backbone pairing requirement, so there is nothing to reuse a
checkpoint for).

Usage:
    python run_methods_adapter_seed_task.py --method B7 --task D1 --train_seed 3 \
        --results_root ../B7_PHM2010_D1_seed_landscape/results
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
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

METHOD_DIRS = {"B6": "B6_MTF_AViTK", "B7": "B7_Dynamic_GIN_TGP", "B8": "B8_DP2Net"}
METHOD_FULL_NAMES = {"B6": "B6_MTF_AViTK", "B7": "B7_Dynamic_GIN_TGP", "B8": "B8_DP2Net"}
TASK_CONDITIONS = {
    "D1": {"train": ["C1", "C4"], "test": "C6"},
    "D2": {"train": ["C1", "C6"], "test": "C4"},
    "D3": {"train": ["C4", "C6"], "test": "C1"},
}
STAGE_ORDER = ["E", "M", "L"]
STAGE_TO_ID = {"E": 0, "M": 1, "L": 2}
ID_MAP = {"early": "E", "middle": "M", "late": "L"}
PREPROCESS_SEED = 42
COMMON_UNIVERSE_START, COMMON_UNIVERSE_END, COMMON_UNIVERSE_N = 12, 315, 304

warnings.filterwarnings("ignore")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def import_adapter_class(method: str):
    method_dir = EXPAND_ROOT / "methods" / METHOD_DIRS[method]
    sys.path.insert(0, str(EXPAND_ROOT / "shared"))
    sys.path.insert(0, str(method_dir))
    sys.path.insert(0, str(method_dir / "code"))
    import importlib
    adapter_mod = importlib.import_module("adapter")
    return adapter_mod.ADAPTER_CLASS


def extract_best_epoch(adapter) -> int | None:
    for attr in ("_best_epoch",):
        v = getattr(adapter, attr, None)
        if v is not None:
            return int(v)
    rows = getattr(adapter, "_training_log_rows", [])
    if rows and "best_epoch" in rows[-1]:
        return int(rows[-1]["best_epoch"])
    if rows:
        return len(rows) - 1
    return None


def run_one(method: str, task: str, train_seed: int, results_root: Path):
    if task not in TASK_CONDITIONS:
        raise ValueError(f"unknown task {task}")
    train_conditions = TASK_CONDITIONS[task]["train"]
    test_condition = TASK_CONDITIONS[task]["test"]

    out_dir = results_root / f"seed{train_seed}"
    config_id = {"method": METHOD_FULL_NAMES[method], "dataset": "PHM2010", "task": task,
                 "train_conditions": train_conditions, "test_condition": test_condition, "train_seed": train_seed}
    config_hash = hashlib.sha256(json.dumps(config_id, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    if (out_dir / "DONE.flag").exists():
        try:
            with open(out_dir / "run_meta.json", "r", encoding="utf-8") as f:
                prior = json.load(f)
            if prior.get("config_hash") == config_hash and prior.get("train_seed") == train_seed and prior.get("task") == task:
                print(f"[resume] method={method} task={task} seed={train_seed} already DONE with matching config -- skipping")
                return "skipped"
        except Exception:
            pass

    ADAPTER_CLASS = import_adapter_class(method)
    from utils.seeding import seed_everything

    scratch_out = out_dir / "_adapter_scratch"
    adapter = ADAPTER_CLASS(
        task=task, train_cutters=train_conditions, test_cutter=test_condition,
        seed=train_seed, preprocess_seed=PREPROCESS_SEED, output_dir=scratch_out,
        device="auto", config={},
    )

    t0 = time.time()
    seed_everything(train_seed)
    adapter.prepare()
    seed_everything(train_seed)
    model = adapter.build_model()
    adapter.train(model)
    t1 = time.time()
    full_pred_df = adapter.predict(model)

    mask = (full_pred_df["run_id"] >= COMMON_UNIVERSE_START) & (full_pred_df["run_id"] <= COMMON_UNIVERSE_END)
    pred_df = full_pred_df.loc[mask].sort_values("run_id").reset_index(drop=True)
    if len(pred_df) != COMMON_UNIVERSE_N:
        raise RuntimeError(
            f"PROTOCOL_FAILED: common evaluation universe restriction produced {len(pred_df)} rows, "
            f"expected {COMMON_UNIVERSE_N} (run_id {COMMON_UNIVERSE_START}-{COMMON_UNIVERSE_END})"
        )

    def to_short(s):
        return ID_MAP[s] if s in ID_MAP else s

    y_true = np.array([STAGE_TO_ID[to_short(s)] for s in pred_df["true_stage"]])
    y_pred = np.array([STAGE_TO_ID[to_short(s)] for s in pred_df["pred_stage"]])
    prob_cu = pred_df[["p_early", "p_middle", "p_late"]].values

    p_each, r_each, f1_each, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cmn = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    dy = np.diff(y_pred.astype(int))
    rev = int(np.sum(dy < 0))
    jump = int(np.sum(np.abs(dy) >= 2))
    smooth = float(np.mean(np.sum(np.abs(np.diff(prob_cu, axis=0)), axis=1))) if len(prob_cu) > 1 else float("nan")

    best_epoch = extract_best_epoch(adapter)
    n_epochs_run = len(getattr(adapter, "_training_log_rows", []))

    seed_summary = {
        "Method": METHOD_FULL_NAMES[method], "Dataset": "PHM2010", "Task": task, "Seed": train_seed,
        "Acc": float(accuracy_score(y_true, y_pred)), "Macro-F1": float(macro_f1),
        "E-F1": float(f1_each[0]), "M-F1": float(f1_each[1]), "L-F1": float(f1_each[2]),
        "M-Precision": float(p_each[1]), "M-Recall": float(r_each[1]),
        "M_to_E": float(cmn[1, 0]), "M_to_L": float(cmn[1, 2]),
        "Rev": rev, "Jump": jump, "Smooth": smooth,
        "q-MAE": np.nan, "q-RMSE": np.nan, "q-R2": np.nan, "Spearman": np.nan,
        "best_epoch": best_epoch if best_epoch is not None else np.nan,
        "n_epochs_run": n_epochs_run, "training_seconds": t1 - t0,
    }

    pred_df_out = pred_df[["run_id"]].copy()
    pred_df_out["true_stage"] = [to_short(s) for s in pred_df["true_stage"]]
    pred_df_out["pred"] = [to_short(s) for s in pred_df["pred_stage"]]
    pred_df_out["prob_E"], pred_df_out["prob_M"], pred_df_out["prob_L"] = prob_cu[:, 0], prob_cu[:, 1], prob_cu[:, 2]

    cm_df = pd.DataFrame(cm, index=[f"true_{s}" for s in STAGE_ORDER], columns=[f"pred_{s}" for s in STAGE_ORDER])

    out_dir.mkdir(parents=True, exist_ok=True)
    config_resolved = dict(config_id, preprocess_seed=PREPROCESS_SEED,
                            training={"note": "see methods/" + METHOD_DIRS[method] + "/adapter.py DEFAULT_CFG for hyperparameters"})
    with open(out_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        for k, v in config_resolved.items():
            f.write(f"{k}: {json.dumps(v, ensure_ascii=False)}\n")

    meta_json = {
        **config_id, "git_commit": git_commit(),
        "start_time": t0, "end_time": t1, "training_seconds": t1 - t0,
        "best_epoch": best_epoch, "n_epochs_run": n_epochs_run,
        "n_test": len(pred_df), "n_test_native": len(full_pred_df),
        "config_hash": config_hash,
        "note": "no feature_hash/split_hash/gmm_hash/window_hash -- this method's "
                "prepare() rebuilds a deterministic (non-random) label/split from "
                "raw signal each run, no frozen-artifact file to hash",
    }
    pd.DataFrame([seed_summary]).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    json_row = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in seed_summary.items()}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump([json_row], f, indent=2, default=str)
    pred_df_out.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(getattr(adapter, "_training_log_rows", [])).to_csv(out_dir / "training_log.csv", index=False, encoding="utf-8-sig")
    cm_df.to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2, default=str)
    (out_dir / "DONE.flag").write_text(f"done at {t1}\n", encoding="utf-8")
    print(f"[save_run] method={method} task={task} seed={train_seed} Acc={seed_summary['Acc']:.4f} "
          f"MacroF1={seed_summary['Macro-F1']:.4f} MF1={seed_summary['M-F1']:.4f} "
          f"MRec={seed_summary['M-Recall']:.4f} Smooth={seed_summary['Smooth']:.4f} "
          f"n_test={len(pred_df)} training_seconds={t1 - t0:.1f}")
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHOD_DIRS.keys()))
    ap.add_argument("--task", required=True, choices=list(TASK_CONDITIONS.keys()))
    ap.add_argument("--train_seed", type=int, required=True)
    ap.add_argument("--results_root", required=True, type=Path)
    args = ap.parse_args()

    out_dir = args.results_root / f"seed{args.train_seed}"
    try:
        status = run_one(args.method, args.task, args.train_seed, args.results_root)
        print(f"[protocol] DONE({status}): method={args.method} task={args.task} train_seed={args.train_seed}")
    except Exception as e:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "error.log", "w", encoding="utf-8") as f:
            import traceback
            f.write(f"method={args.method} task={args.task} seed={args.train_seed}\n")
            traceback.print_exc(file=f)
        (out_dir / "FAILED.flag").write_text(f"failed: {e}\n", encoding="utf-8")
        print(f"[protocol] FAILED: method={args.method} task={args.task} train_seed={args.train_seed}: {e}")
        raise


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
r"""
B1 (Relative-stage RF), MTW-CM D1-M/D2-M/D3-M, strict TRAIN_SEED isolation.

Sibling to run_mtw_seed_task.py (B9+B3) -- same frozen-preprocessing source
(`shared/reproducibility/MTW_{task}_frozen_preprocess/`, task string with the
hyphen, e.g. "D1-M") and output convention, same RF logic as
run_b1_seed_task.py (PHM2010) / run_b1_nasa_seed_task.py (NASA). Uses
`experiments_mendeley/code/dcpsr/model.py::build_windows` (read-only import)
only to determine the windowed-feasible test (sequence_id, order_key)
universe -- RF itself trains on all raw feasible train rows unwindowed,
scored only on the windowed-feasible test rows, matching the PHM2010/NASA
convention. No RF implementation exists in dcpsr/baselines.py under this
name (its own "B5" RF is a different, non-frozen-preprocessing pipeline) --
built fresh here against the frozen artifacts instead, for exact seed/task
provenance parity with B9/B3's own frozen-preprocessing runs.

Usage:
    python run_b1_mtw_seed_task.py --task D1-M --train_seed 0 \
        --results_root ../B1_MTW_D1M_seed_landscape/results
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support,
)

warnings.filterwarnings("ignore")

SEED_STATS_DIR = Path(__file__).resolve().parent.parent
EXPAND_ROOT = SEED_STATS_DIR.parents[1]
PROJECT_ROOT = EXPAND_ROOT.parent
# Self-contained: vendored copy inside this repo (git-clone portable), NOT the
# outside parent project -- see _shared_code/vendored_legacy/README.md.
DCPSR_CODE_DIR = SEED_STATS_DIR / "_shared_code" / "vendored_legacy"

STAGE_ORDER = ["E", "M", "L"]
N_ESTIMATORS = 400


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_dir_for(task: str) -> Path:
    return EXPAND_ROOT / "shared" / "reproducibility" / f"MTW_{task}_frozen_preprocess"


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


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def run_one(task: str, train_seed: int, results_root: Path):
    sys.path.insert(0, str(DCPSR_CODE_DIR))
    from dcpsr import config as C
    from dcpsr.model import build_windows

    frozen_dir = frozen_dir_for(task)
    hashes = assert_frozen_artifacts_unchanged(frozen_dir)
    print(f"[protocol] frozen artifacts verified OK task={task} preprocess_seed={hashes['preprocess_seed']}")

    out_dir = results_root / f"seed{train_seed}"
    if (out_dir / "DONE.flag").exists():
        try:
            with open(out_dir / "run_meta.json", "r", encoding="utf-8") as f:
                prior = json.load(f)
            if (prior.get("split_hash") == hashes["files"]["split_info.json"]
                    and prior.get("feature_hash") == hashes["files"]["selected_features_seed42.json"]
                    and prior.get("gmm_hash") == hashes["files"]["gmm_seed42.pkl"]
                    and prior.get("train_seed") == train_seed and prior.get("task") == task):
                print(f"[resume] task={task} seed={train_seed} already DONE with matching hashes -- skipping")
                return "skipped"
        except Exception:
            pass

    with open(frozen_dir / "selected_features_seed42.json", "r", encoding="utf-8") as f:
        selected = json.load(f)["selected_features_in_order"]
    feat_train = pd.read_csv(frozen_dir / "feat_train_frozen.csv")
    feat_test = pd.read_csv(frozen_dir / "feat_test_frozen.csv")
    L = C.ARCH["L"]

    random.seed(train_seed)
    np.random.seed(train_seed)

    _, _, _, _, _, _, meta_te = build_windows(feat_test, selected, L, "test")
    meta = meta_te.reset_index(drop=True)
    y_true = meta["stage_true_id"].values.astype(int)

    Xtr = feat_train[selected].values
    ytr = feat_train["stage_id"].values.astype(int)
    test_by_key = feat_test.set_index(["sequence_id", "order_key"])
    Xte = test_by_key.loc[list(zip(meta["sequence_id"], meta["order_key"])), selected].values

    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=train_seed,
        class_weight="balanced_subsample", n_jobs=-1,
    )
    rf.fit(Xtr, ytr)
    t1 = time.time()

    y_pred = rf.predict(Xte)
    prob = rf.predict_proba(Xte)

    p_each, r_each, f1_each, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cmn = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    dy = np.diff(y_pred.astype(int))
    rev = int(np.sum(dy < 0))
    jump = int(np.sum(np.abs(dy) >= 2))
    smooth = float(np.mean(np.sum(np.abs(np.diff(prob, axis=0)), axis=1))) if len(prob) > 1 else float("nan")

    seed_summary = {
        "Method": "B1_RF", "Dataset": "MTW_CM", "Task": task, "Seed": train_seed,
        "Acc": float(accuracy_score(y_true, y_pred)), "Macro-F1": float(macro_f1),
        "E-F1": float(f1_each[0]), "M-F1": float(f1_each[1]), "L-F1": float(f1_each[2]),
        "M-Precision": float(p_each[1]), "M-Recall": float(r_each[1]),
        "M_to_E": float(cmn[1, 0]), "M_to_L": float(cmn[1, 2]),
        "Rev": rev, "Jump": jump, "Smooth": smooth,
        "q-MAE": np.nan, "q-RMSE": np.nan, "q-R2": np.nan, "Spearman": np.nan,
        "best_epoch": 0, "n_test": len(meta), "training_seconds": t1 - t0,
    }

    pred_df = meta[["sequence_id", "domain_id", "order_key", "stage_true", "q_true"]].copy()
    pred_df["pred"] = [STAGE_ORDER[i] for i in y_pred]
    pred_df["q_hat"] = np.nan  # RF has no continuous degradation-index head
    pred_df["prob_E"], pred_df["prob_M"], pred_df["prob_L"] = prob[:, 0], prob[:, 1], prob[:, 2]

    cm_df = pd.DataFrame(cm, index=[f"true_{s}" for s in STAGE_ORDER], columns=[f"pred_{s}" for s in STAGE_ORDER])

    out_dir.mkdir(parents=True, exist_ok=True)
    config_resolved = {
        "method": "B1_RF", "dataset": "MTW_CM", "task": task,
        "preprocess_seed": hashes["preprocess_seed"], "train_seed": train_seed,
        "model": {"n_estimators": N_ESTIMATORS, "class_weight": "balanced_subsample", "n_jobs": -1},
        "window_length_L": L,
    }
    with open(out_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        for k, v in config_resolved.items():
            f.write(f"{k}: {json.dumps(v, ensure_ascii=False)}\n")

    meta_json = {
        "method": "B1_RF", "dataset": "MTW_CM", "task": task,
        "preprocess_seed": hashes["preprocess_seed"], "train_seed": train_seed,
        "train_sequences": hashes.get("train_sequences"), "validation_sequences": hashes.get("validation_sequences"),
        "test_sequences": hashes.get("test_sequences"), "git_commit": git_commit(),
        "feature_hash": hashes["files"]["selected_features_seed42.json"],
        "split_hash": hashes["files"]["split_info.json"], "gmm_hash": hashes["files"]["gmm_seed42.pkl"],
        "start_time": t0, "end_time": t1, "training_seconds": t1 - t0,
        "n_train": len(Xtr), "n_test": len(meta),
    }
    pd.DataFrame([seed_summary]).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    json_row = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in seed_summary.items()}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump([json_row], f, indent=2, default=str)
    pred_df.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"note": "RF has no per-epoch training curve -- single fit() call, see training_seconds in metrics.csv", "training_seconds": t1 - t0}]).to_csv(
        out_dir / "training_log.csv", index=False, encoding="utf-8-sig"
    )
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
    ap.add_argument("--task", required=True, choices=["D1-M", "D2-M", "D3-M"])
    ap.add_argument("--train_seed", type=int, required=True)
    ap.add_argument("--results_root", required=True, type=Path)
    args = ap.parse_args()

    out_dir = args.results_root / f"seed{args.train_seed}"
    try:
        status = run_one(args.task, args.train_seed, args.results_root)
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

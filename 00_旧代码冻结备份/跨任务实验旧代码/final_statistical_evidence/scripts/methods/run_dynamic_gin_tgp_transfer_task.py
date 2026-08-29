# -*- coding: utf-8 -*-
"""
D2/D3 transfer-task training for Dynamic GIN + TGP.

Reuses baselines/dynamic_gin_tgp/train.py::run_protocol_b(seed=42, ...)
UNMODIFIED, monkeypatching the same D2/D3 condition split
(condition_split.py) and train.OUT_ROOT (scratch path under this stage's
own tree). This is the post-label-leakage-fix code path (per-Dataset row
shuffle at construction, already the only code path in this file) -- no
change made here, just reused. No architecture/hyperparameter (topk, lr,
batch, weight_decay, epochs, patience) change.
"""
from __future__ import annotations

import argparse
import json
import sys
import shutil
import time
from pathlib import Path

import torch

FSE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = FSE_ROOT.parent
METHOD_DIR = REPO_ROOT / "baselines" / "dynamic_gin_tgp"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(METHOD_DIR))
sys.path.insert(0, str(METHOD_DIR / "data"))

import condition_split  # noqa: E402

TASKS = {"D2": (["C1", "C6"], "C4"), "D3": (["C4", "C6"], "C1")}
METHOD_ID = "dynamic_gin_tgp"
TRAIN_SEED = 42


def run(task: str, force: bool = False):
    train_conditions, test_condition = TASKS[task]
    out_dir = FSE_ROOT / "transfer_tasks" / task / METHOD_ID
    if (out_dir / "DONE.flag").exists() and not force:
        print(f"[{METHOD_ID}/{task}] DONE.flag present, skip")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    import train as T  # baselines/dynamic_gin_tgp/train.py
    condition_split.apply_patch(T.dcpsr_base, train_conditions, test_condition)
    scratch_root = out_dir / "_native"
    T.OUT_ROOT = scratch_root

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    metric_row = T.run_protocol_b(device=device, seed=TRAIN_SEED)
    t1 = time.time()

    native_dir = scratch_root / "unified_protocol" / f"seed{TRAIN_SEED}"
    import pandas as pd
    pred_df = pd.read_csv(native_dir / "run_predictions.csv")
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    metrics = condition_split.normalize_metric_row(metric_row)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    ckpt_src = native_dir / "epoch_checkpoint.pth"
    if ckpt_src.exists():
        shutil.copy2(ckpt_src, out_dir / "best.pt")

    (out_dir / "config.yaml").write_text(
        f"method: {METHOD_ID}\ntask: {task}\ntrain_conditions: {train_conditions}\ntest_condition: {test_condition}\n"
        f"PROTO_B_CFG: {T.PROTO_B_CFG}\ntrain_seed: {TRAIN_SEED}\ntrain_seconds: {t1 - t0:.2f}\n",
        encoding="utf-8",
    )
    (out_dir / "status.json").write_text(
        json.dumps({"state": "done", "timestamp": time.time(), "acc": metrics.get("Acc")}, indent=2), encoding="utf-8"
    )
    (out_dir / "DONE.flag").write_text(f"done at {time.time()}\n", encoding="utf-8")
    print(f"[{METHOD_ID}/{task}] DONE acc={metrics.get('Acc'):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASKS.keys()))
    parser.add_argument("--method", required=True, choices=[METHOD_ID])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args.task, force=args.force)


if __name__ == "__main__":
    main()

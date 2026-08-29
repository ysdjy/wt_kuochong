# -*- coding: utf-8 -*-
"""
D2/D3 transfer-task training for MTF-AViTK.

Reuses baselines/mtf_avitk/train.py::run_protocol_b() UNMODIFIED,
monkeypatching the same D2/D3 condition split (condition_split.py) and
train.OUT_ROOT (scratch path under this stage's own tree). No ViT/MTF/
AdaptMLP/KAN architecture or hyperparameter (PROTO_B_CFG) change.

NOTE: this is the largest model in the comparison (309M params, ViT-L/32)
and has historically taken ~30-40 minutes per run on this 8GB laptop GPU,
sometimes longer. Per user instruction, this job is NOT auto-launched by
run_transfer_tasks.py's default `--resume` sweep unless the user explicitly
runs it (see final_statistical_evidence/MTF_AVITK_MANUAL_TUTORIAL.md for
the recommended manual/foreground invocation). It CAN still be invoked
through this same script/CLI, standalone or via
`run_transfer_tasks.py --task D2 --method mtf_avitk --resume`, whenever the
user chooses to run it.
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
METHOD_DIR = REPO_ROOT / "baselines" / "mtf_avitk"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(METHOD_DIR))
sys.path.insert(0, str(METHOD_DIR / "data"))

import condition_split  # noqa: E402

TASKS = {"D2": (["C1", "C6"], "C4"), "D3": (["C4", "C6"], "C1")}
METHOD_ID = "mtf_avitk"


def run(task: str, force: bool = False, grad_checkpoint: bool = False, batch_size: int | None = None):
    train_conditions, test_condition = TASKS[task]
    out_dir = FSE_ROOT / "transfer_tasks" / task / METHOD_ID
    if (out_dir / "DONE.flag").exists() and not force:
        print(f"[{METHOD_ID}/{task}] DONE.flag present, skip")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    import train as T  # baselines/mtf_avitk/train.py
    condition_split.apply_patch(T.dcpsr_base, train_conditions, test_condition)
    scratch_root = out_dir / "_native"
    T.OUT_ROOT = scratch_root

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    metric_row = T.run_protocol_b(device=device, grad_checkpoint=grad_checkpoint, batch_size=batch_size)
    t1 = time.time()

    native_dir = scratch_root / "unified_protocol"
    import pandas as pd
    pred_df = pd.read_csv(native_dir / "test_predictions.csv")
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    metrics = condition_split.normalize_metric_row(metric_row)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    ckpt_src = native_dir / "checkpoint_best.pth"
    if ckpt_src.exists():
        shutil.copy2(ckpt_src, out_dir / "best.pt")

    (out_dir / "config.yaml").write_text(
        f"method: {METHOD_ID}\ntask: {task}\ntrain_conditions: {train_conditions}\ntest_condition: {test_condition}\n"
        f"PROTO_B_CFG: {T.PROTO_B_CFG}\ntrain_seconds: {t1 - t0:.2f}\n",
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
    parser.add_argument("--grad-checkpoint", action="store_true", help="enable if CUDA OOM near the 8GB ceiling")
    parser.add_argument("--batch-size", type=int, default=None, help="override PROTO_B_CFG batch_size if OOM")
    args = parser.parse_args()
    run(args.task, force=args.force, grad_checkpoint=args.grad_checkpoint, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

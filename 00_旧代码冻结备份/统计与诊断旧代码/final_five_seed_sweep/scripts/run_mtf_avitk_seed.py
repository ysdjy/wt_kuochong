# -*- coding: utf-8 -*-
r"""
Runs one additional seed of MTF-AViTK's Protocol B (Unified, C1+C4->C6),
for the final 9-method 5-seed sweep. No MTF/ViT-L/32/AdaptMLP/KAN/training
hyperparameter is changed -- only PROTO_B_CFG['seed'].

Redirects output to a seed-specific directory
(outputs/mtf_avitk/seed_sweep/seed<N>/unified_protocol/) by monkeypatching
the module's OUT_ROOT before calling run_protocol_b, so the existing
official seed=42 output (outputs/mtf_avitk/unified_protocol/) is never
touched.

Usage:
    python run_mtf_avitk_seed.py --seed 52
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
BASELINE_DIR = PROJECT_ROOT / "baselines" / "mtf_avitk"

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--device", default="cuda")
parser.add_argument("--grad-checkpoint", action="store_true")
parser.add_argument("--grad-accum-steps", type=int, default=1)
args = parser.parse_args()

sys.path.insert(0, str(BASELINE_DIR))
import train as T  # noqa: E402

T.OUT_ROOT = PROJECT_ROOT / "outputs" / "mtf_avitk" / "seed_sweep" / f"seed{args.seed}"
T.PROTO_B_CFG = dict(T.PROTO_B_CFG)
T.PROTO_B_CFG["seed"] = args.seed

if __name__ == "__main__":
    print(f"[mtf_avitk seed {args.seed}] PROTO_B_CFG = {T.PROTO_B_CFG}")
    print(f"[mtf_avitk seed {args.seed}] OUT_ROOT    = {T.OUT_ROOT}")
    import inspect
    sig = inspect.signature(T.run_protocol_b)
    kwargs = {"device": args.device}
    if "grad_checkpoint" in sig.parameters:
        kwargs["grad_checkpoint"] = args.grad_checkpoint
    if "grad_accum_steps" in sig.parameters:
        kwargs["grad_accum_steps"] = args.grad_accum_steps
    row = T.run_protocol_b(**kwargs)
    print(f"[mtf_avitk seed {args.seed}] done: {row}")

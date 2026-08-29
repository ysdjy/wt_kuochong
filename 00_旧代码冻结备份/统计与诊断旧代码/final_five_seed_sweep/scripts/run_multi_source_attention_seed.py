# -*- coding: utf-8 -*-
r"""
Runs one additional seed of Multi-source Attention's Protocol B (Unified,
C1+C4->C6), for the final 9-method 5-seed sweep. No CWT/attention/training
hyperparameter is changed -- only PROTO_B_CFG['seed'].

Redirects output to a seed-specific directory
(outputs/multi_source_attention/seed_sweep/seed<N>/unified_protocol/) by
monkeypatching the module's OUT_ROOT before calling run_protocol_b, so the
existing official seed=42 output
(outputs/multi_source_attention/unified_protocol/) is never touched.

Usage:
    python run_multi_source_attention_seed.py --seed 52
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
BASELINE_DIR = PROJECT_ROOT / "baselines" / "multi_source_attention"

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

sys.path.insert(0, str(BASELINE_DIR))
import train as T  # noqa: E402

T.OUT_ROOT = PROJECT_ROOT / "outputs" / "multi_source_attention" / "seed_sweep" / f"seed{args.seed}"
T.PROTO_B_CFG = dict(T.PROTO_B_CFG)
T.PROTO_B_CFG["seed"] = args.seed

if __name__ == "__main__":
    print(f"[multi_source_attention seed {args.seed}] PROTO_B_CFG = {T.PROTO_B_CFG}")
    print(f"[multi_source_attention seed {args.seed}] OUT_ROOT    = {T.OUT_ROOT}")
    row = T.run_protocol_b(args.device)
    print(f"[multi_source_attention seed {args.seed}] done: {row}")

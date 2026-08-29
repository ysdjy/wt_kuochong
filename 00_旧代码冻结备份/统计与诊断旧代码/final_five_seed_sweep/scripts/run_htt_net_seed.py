# -*- coding: utf-8 -*-
r"""
Runs one additional seed of HTT-Net's official SOURCE_ONLY_TUNED config
(baselines/htt_net/best_source_only_config.yaml == train_final_tuned.py's
frozen T.HTT_ARCH / T.TRAIN_CFG['lr']) for the final 9-method 5-seed sweep.

Mirrors baselines/htt_net/train_final_tuned.py exactly, except:
  - TRAIN_CFG['seed'] is overridden to the CLI --seed value (only the seed,
    no other hyperparameter changes).
  - Output goes to a NEW directory (outputs/htt_net/D1_C1C4_to_C6_SOURCE_ONLY_TUNED_seed<N>/),
    so the existing official seed=42 output
    (outputs/htt_net/D1_C1C4_to_C6_SOURCE_ONLY_TUNED/) is never touched.

Usage:
    python run_htt_net_seed.py --seed 52
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
HTT_DIR = PROJECT_ROOT / "baselines" / "htt_net"

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()

os.environ["HTT_OUTPUT_DIR"] = str(
    PROJECT_ROOT / "outputs" / "htt_net" / f"D1_C1C4_to_C6_SOURCE_ONLY_TUNED_seed{args.seed}"
)

import sys
sys.path.insert(0, str(HTT_DIR))
import train as T  # noqa: E402

# Frozen config, byte-identical to train_final_tuned.py -- only seed changes.
T.HTT_ARCH = {
    "embed_dim": 64,
    "depths": (2, 2, 2, 2),
    "num_heads": 4,
    "window_size": 3,
    "dropout": 0.20,
}
T.TRAIN_CFG["lr"] = 5e-4
T.TRAIN_CFG["seed"] = args.seed

AUTHORITATIVE_FEATURE_FILE = HTT_DIR / "data" / "run_level_features_all.csv"

if __name__ == "__main__":
    print(f"[HTT-Net seed {args.seed}] Frozen config (source-only selected, C6 never used for selection):")
    print(f"  HTT_ARCH   = {T.HTT_ARCH}")
    print(f"  lr         = {T.TRAIN_CFG['lr']}")
    print(f"  seed       = {T.TRAIN_CFG['seed']}")
    print(f"  output_dir = {T.OUTPUT_ROOT}")
    T.run(feature_file=AUTHORITATIVE_FEATURE_FILE, smoke_test=False)

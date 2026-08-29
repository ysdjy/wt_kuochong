# -*- coding: utf-8 -*-
r"""
Final, frozen-config HTT-Net (B13) run: trains once on C1+C4, tests once on
C6, using the hyperparameters selected by source_only_tuning.py
(best_source_only_config.yaml). C6 is touched exactly once, after the
config was already frozen from source-only validation -- see
FINAL_HTT_NET_REPORT.md for the leakage statement.

Usage:
    python train_final_tuned.py
"""
from __future__ import annotations

import os
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

# Output goes to a directory distinct from the initial-untuned run, which
# must not be overwritten (outputs/htt_net/D1_C1C4_to_C6_INITIAL_UNTUNED/).
os.environ["HTT_OUTPUT_DIR"] = str(
    THIS_DIR.parents[1] / "outputs" / "htt_net" / "D1_C1C4_to_C6_SOURCE_ONLY_TUNED"
)

import train as T  # noqa: E402  (import after HTT_OUTPUT_DIR is set, since T.OUTPUT_ROOT is computed at import time)

# Apply the frozen, source-only-selected hyperparameters.
T.HTT_ARCH = {
    "embed_dim": 64,
    "depths": (2, 2, 2, 2),
    "num_heads": 4,
    "window_size": 3,
    "dropout": 0.20,
}
T.TRAIN_CFG["lr"] = 5e-4  # unchanged from the initial run -- selected as optimal by the search too

AUTHORITATIVE_FEATURE_FILE = THIS_DIR / "data" / "run_level_features_all.csv"

if __name__ == "__main__":
    print("Frozen config (source-only selected, C6 never used for selection):")
    print(f"  HTT_ARCH   = {T.HTT_ARCH}")
    print(f"  lr         = {T.TRAIN_CFG['lr']}")
    print(f"  output_dir = {T.OUTPUT_ROOT}")
    T.run(feature_file=AUTHORITATIVE_FEATURE_FILE, smoke_test=False)

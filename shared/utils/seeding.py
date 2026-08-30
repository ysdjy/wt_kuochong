"""Unified, isolated RNG reset for every method adapter.

Two independent seeds are used throughout this repo and must never be conflated:

- PREPROCESS_SEED: fixed for a given (dataset, task) — controls feature selection,
  GMM fitting, scaler fitting, and any train/val split constructed from the train
  cutters. Never changes with TRAIN_SEED.
- TRAIN_SEED: varies per run (0..100) — controls only true model-training/
  optimization randomness (weight init, minibatch shuffling, dropout, etc.).

`seed_everything(seed)` must be called immediately before a method's model is
instantiated, not once at the top of a multi-model driver script. Sharing one RNG
stream across sequentially-trained models was a real, confirmed bug in the old
project's `代码/7.4对比实验.py` (its old B10 TCN-GRU seed=42 baseline consumed
RNG state already spent by B8/B9 trained earlier in the same process, making that
"seed=42" run not actually isolated) — see
`扩充实验代码/shared/reproducibility/PHM2010_D1_frozen_preprocess/TCN_GRU_SEED42_DIVERGENCE_NOTE.txt`.
Every adapter in this repo must call seed_everything() right before building its
model, never rely on a seed set earlier in the process by another method or by a
driver loop.
"""
from __future__ import annotations

import os
import random


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Reset Python's, NumPy's, and (if installed/available) PyTorch's RNGs.

    Safe to call even when torch or CUDA is not available/installed — those steps
    are skipped silently rather than raising, so this also works for CPU-only
    methods (e.g. B1 RF) and CPU-only smoke tests.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

# -*- coding: utf-8 -*-
r"""
Vendored verbatim from methods/B5_MultiSource_Attention/code/dataset.py.
On-demand CWT-image cache + torch Dataset for B5. No modifications.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from b5_signal_preprocessing import build_sample

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "_b5_image_cache"
CACHE_DIR = Path(os.environ.get("B5_IMAGE_CACHE_DIR", str(DEFAULT_CACHE_DIR)))


def get_or_build_image_pair(condition: str, run_id: int, cache_dir: Path = CACHE_DIR) -> tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    force_path = cache_dir / f"{condition}_{run_id:03d}_force.npy"
    vib_path = cache_dir / f"{condition}_{run_id:03d}_vib.npy"
    if force_path.exists() and vib_path.exists():
        return np.load(force_path), np.load(vib_path)
    force_img, vib_img = build_sample(condition, run_id)
    np.save(force_path, force_img)
    np.save(vib_path, vib_img)
    return force_img, vib_img


class ImagePairDataset(Dataset):
    """rows: DataFrame with columns condition, run_id, <label_col> (int stage id)."""

    def __init__(self, rows: pd.DataFrame, label_col: str, cache_dir: Path = CACHE_DIR):
        self.rows = rows.reset_index(drop=True)
        self.label_col = label_col
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows.iloc[idx]
        force, vib = get_or_build_image_pair(r.condition, int(r.run_id), self.cache_dir)
        force = torch.from_numpy(force.astype(np.float32) / 255.0).permute(2, 0, 1)
        vib = torch.from_numpy(vib.astype(np.float32) / 255.0).permute(2, 0, 1)
        label = int(getattr(r, self.label_col))
        return force, vib, label

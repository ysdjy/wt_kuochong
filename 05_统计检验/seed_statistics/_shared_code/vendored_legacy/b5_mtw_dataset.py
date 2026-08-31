# -*- coding: utf-8 -*-
r"""
On-demand CWT-image cache + torch Dataset for B5 on MTW-CM. Mirrors
b5_dataset.py (PHM2010), keyed by (sequence_id, order_key) instead of
(condition, run_id).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from b5_mtw_signal_preprocessing import build_sample

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "_b5_mtw_image_cache"
CACHE_DIR = Path(os.environ.get("B5_MTW_IMAGE_CACHE_DIR", str(DEFAULT_CACHE_DIR)))


def get_or_build_image_pair(sequence_id: str, order_key: int, source_file: str, cache_dir: Path = CACHE_DIR) -> tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    a_path = cache_dir / f"{sequence_id}_r{order_key:04d}_a.npy"
    b_path = cache_dir / f"{sequence_id}_r{order_key:04d}_b.npy"
    if a_path.exists() and b_path.exists():
        return np.load(a_path), np.load(b_path)
    img_a, img_b = build_sample(source_file)
    np.save(a_path, img_a)
    np.save(b_path, img_b)
    return img_a, img_b


class ImagePairDataset(Dataset):
    """rows: DataFrame with columns sequence_id, order_key, source_file, <label_col> (int stage id)."""

    def __init__(self, rows: pd.DataFrame, label_col: str, cache_dir: Path = CACHE_DIR):
        self.rows = rows.reset_index(drop=True)
        self.label_col = label_col
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows.iloc[idx]
        img_a, img_b = get_or_build_image_pair(r.sequence_id, int(r.order_key), r.source_file, self.cache_dir)
        a = torch.from_numpy(img_a.astype(np.float32) / 255.0).permute(2, 0, 1)
        b = torch.from_numpy(img_b.astype(np.float32) / 255.0).permute(2, 0, 1)
        label = int(getattr(r, self.label_col))
        return a, b, label

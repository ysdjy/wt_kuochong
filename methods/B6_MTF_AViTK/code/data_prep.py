# -*- coding: utf-8 -*-
"""Metadata construction + on-disk-cached MTF image dataset for MTF-AViTK.

ADAPTATION vs. the old project (allowed per task spec section 35 -- data
location / caching strategy, not architecture): the old project pre-built
ALL 945 runs x 5 sub-windows once via `data/build_dataset.py` into a 2.0GB
`data/images/*.npy` directory that is never committed here (would blow the
git-size budget many times over). This module instead builds MTF images
on-demand (per condition/run, 5 sub-windows at a time, since
`preprocessing.build_main_samples` computes all 5 together from one raw
signal load) and caches them under a configurable directory (default
`data/PHM2010/derived/mtf_avitk_images/`) so repeated seeds over the same
task don't redo the wavelet-transform work. Every image is still generated
by the byte-for-byte-identical `preprocessing.py` pipeline -- only the
storage/caching strategy changed.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import label_utils as L
import preprocessing as P

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "PHM2010" / "derived" / "mtf_avitk_images"


def image_cache_dir() -> Path:
    return Path(os.environ.get("MTF_AVITK_IMAGE_CACHE", str(DEFAULT_CACHE_DIR)))


def build_metadata(train_cutters: list[str], test_cutter: str) -> dict[str, pd.DataFrame]:
    """Returns {"train": df, "val": df, "test": df}, each at SUB-WINDOW
    granularity (one row per of the 5 sub-windows per run), columns:
    condition, run_id, subwindow, stage, stage_id."""
    tr_df, va_df, te_df = L.get_unified_split(train_cutters, test_cutter)
    out = {}
    for name, run_level in [("train", tr_df), ("val", va_df), ("test", te_df)]:
        rows = []
        for r in run_level.itertuples():
            for k in range(P.N_SUBWINDOWS_MAIN):
                rows.append({
                    "condition": r.condition, "run_id": int(r.run_id), "subwindow": k,
                    "stage": r.stage, "stage_id": int(r.stage_id),
                })
        out[name] = pd.DataFrame(rows)
    return out


def get_or_build_images(condition: str, run_id: int, cache_dir: Path) -> list[np.ndarray]:
    """Return the 5 sub-window MTF images for one (condition, run_id),
    building + caching them on first access."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = [cache_dir / f"{condition}_{run_id:03d}_{k}.npy" for k in range(P.N_SUBWINDOWS_MAIN)]
    if all(p.exists() for p in paths):
        return [np.load(p) for p in paths]
    images = P.build_main_samples(condition, run_id)
    for p, img in zip(paths, images):
        np.save(p, img)
    return images


class MTFImageDataset(Dataset):
    """rows: DataFrame with columns condition, run_id, subwindow, stage_id."""

    def __init__(self, rows: pd.DataFrame, cache_dir: Path | None = None):
        self.rows = rows.reset_index(drop=True)
        self.cache_dir = cache_dir or image_cache_dir()
        self._run_cache: dict[tuple[str, int], list[np.ndarray]] = {}

    def __len__(self):
        return len(self.rows)

    def _images_for_run(self, condition: str, run_id: int) -> list[np.ndarray]:
        key = (condition, run_id)
        if key not in self._run_cache:
            self._run_cache[key] = get_or_build_images(condition, run_id, self.cache_dir)
        return self._run_cache[key]

    def __getitem__(self, idx):
        r = self.rows.iloc[idx]
        images = self._images_for_run(r.condition, int(r.run_id))
        img = images[int(r.subwindow)].astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # [3,H,W]
        label = int(r.stage_id)
        return img, label

# -*- coding: utf-8 -*-
r"""
Vendored verbatim from methods/B5_MultiSource_Attention/code/signal_preprocessing.py.
Raw PHM2010 7-channel signal -> CWT RGB scalogram images (force/vibration).
No modifications -- see vendored_legacy/README.md.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pywt
from scipy.signal import resample as scipy_resample

PHM2010_RAW_ROOT = Path(
    os.environ.get("PHM2010_ROOT") or os.environ.get("PHM2010_RAW_ROOT")
    or str(Path(__file__).resolve().parents[4] / "data" / "PHM2010" / "raw")
)

FORCE_COLS = [0, 1, 2]
VIB_COLS = [3, 4, 5]

MIDDLE_REGION_FRACTION = (0.25, 0.75)
CWT_WAVELET = "cmor1.5-1.0"
IMAGE_SIZE = 224
N_SCALES = IMAGE_SIZE


def load_raw_signal(condition: str, run_id: int) -> np.ndarray:
    cond_lower = condition.lower()
    path = PHM2010_RAW_ROOT / cond_lower / cond_lower / f"c_{cond_lower[1:]}_{run_id:03d}.csv"
    sig = np.loadtxt(path, delimiter=",")
    if sig.ndim != 2 or sig.shape[1] != 7:
        raise ValueError(f"{path} has shape {sig.shape}, expected [N,7]")
    return sig


def get_middle_region(signal: np.ndarray, fraction: tuple[float, float] = MIDDLE_REGION_FRACTION) -> np.ndarray:
    n = signal.shape[0]
    start = int(round(n * fraction[0]))
    end = int(round(n * fraction[1]))
    return signal[start:end, :]


def cwt_scalogram(x_1d: np.ndarray, image_size: int = IMAGE_SIZE, n_scales: int = N_SCALES,
                   wavelet: str = CWT_WAVELET) -> np.ndarray:
    xr = scipy_resample(x_1d, image_size).astype(np.float64)
    scales = np.geomspace(1, image_size, n_scales)
    coeffs, _freqs = pywt.cwt(xr, scales, wavelet)
    mag = np.abs(coeffs)
    mmin, mmax = mag.min(), mag.max()
    return (mag - mmin) / (mmax - mmin + 1e-12)


def axes_to_rgb_image(signal_3axis: np.ndarray, image_size: int = IMAGE_SIZE) -> np.ndarray:
    channels = [cwt_scalogram(signal_3axis[:, ax], image_size=image_size, n_scales=image_size) for ax in range(3)]
    img = np.stack(channels, axis=-1)
    return (img * 255.0).astype(np.uint8)


def build_sample(condition: str, run_id: int) -> tuple[np.ndarray, np.ndarray]:
    sig = load_raw_signal(condition, run_id)
    mid = get_middle_region(sig)
    force_img = axes_to_rgb_image(mid[:, FORCE_COLS])
    vib_img = axes_to_rgb_image(mid[:, VIB_COLS])
    return force_img, vib_img

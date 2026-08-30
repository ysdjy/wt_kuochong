# -*- coding: utf-8 -*-
r"""
MTF-AViTK native preprocessing pipeline, vendored from the old project's
`baselines/mtf_avitk/preprocessing.py` (reimplemented there from
`PAPER_SPEC.md`, Dong et al., MSSP 2025, DOI: 10.1016/j.ymssp.2025.112473).

ADAPTATION vs. the original (path routing only, per task spec section 35 --
no architecture/algorithm change): the original hardcoded
`ARCHIVE_DIR = PROJECT_ROOT / "archive"` (old parent project's local dataset
copy). This version resolves the raw PHM2010 directory from the `PHM2010_ROOT`
environment variable, defaulting to `<repo_root>/data/PHM2010/raw` (the
location `scripts/download_phm2010.py` populates) -- see task spec section 11.
Every pipeline function (resultant_force, wavelet_denoise, mtf_encode,
mtf_to_rgb_image, signal_to_mtf_image) is otherwise byte-for-byte identical to
the original, including every "Missing in paper" implementation choice
documented below.

Pipeline (see the old project's PAPER_SPEC.md for the full annotated diagram):

    raw Fx,Fy,Fz  (PHM2010 c_{cond}_{run:03d}.csv)
      -> stable-region slice (per-protocol window, see get_stable_window)
      -> non-overlapping 2000-sample sub-windows (5 per 10,000-sample region)
      -> resultant force F = sqrt(Fx^2+Fy^2+Fz^2)              Eq. 8  [Explicit]
      -> Sym7 wavelet denoise, level 6, soft threshold          [Explicit method;
                                                                  Missing threshold
                                                                  formula -> VisuShrink/MAD]
      -> resample 2000 -> 500 samples                           [Missing/Conflict,
                                                                  see PAPER_SPEC §3.1]
      -> MTF encoding (Q=8 quantile bins) -> [500,500] field     Eq. 1  [Explicit
                                                                  mechanism; Missing Q]
      -> jet colormap -> [500,500,3] RGB image                  [Missing]
      -> bilinear resize -> [384,384,3]                         [Missing]
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pywt
from scipy.signal import resample as scipy_resample

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = Path(os.environ.get("PHM2010_ROOT", str(REPO_ROOT / "data" / "PHM2010" / "raw")))

CHANNEL_NAMES = ["Fx", "Fy", "Fz", "Vx", "Vy", "Vz", "AE"]  # standard PHM2010 7-channel order

# --- Stable-region windows (PAPER_SPEC.md, "Stable cutting region" rows) ---
MAIN_WINDOW = (90_000, 100_000)         # main A/B datasets (Protocol A + used for Protocol B)
BALANCED_POOL_WINDOW = (100_000, 120_000)  # source pool for the class-balanced 700/700/700 set
SUBWINDOW_LEN = 2000
N_SUBWINDOWS_MAIN = (MAIN_WINDOW[1] - MAIN_WINDOW[0]) // SUBWINDOW_LEN  # 5

# --- Wavelet denoising (PAPER_SPEC.md "Wavelet denoising" rows) ---
WAVELET = "sym7"          # Explicit
WAVELET_LEVEL = 6         # Explicit
WAVELET_MODE = "soft"     # Explicit (method); threshold VALUE formula is Missing -> VisuShrink/MAD

# --- MTF (PAPER_SPEC.md "MTF:" rows) ---
MTF_N_QUANTILE_BINS = 8   # Missing in paper -> Wang & Oates default / pyts default
MTF_FIELD_SIZE = 500      # Explicit (§3.2 final image size), reached via 2000->500 resample (Missing/Conflict)
VIT_INPUT_SIZE = 384      # Explicit (Fig. 3)


def load_raw_signal(condition: str, run_id: int) -> np.ndarray:
    """Return the raw 7-channel signal for one cutting pass, shape [N,7]."""
    cond_lower = condition.lower()
    path = RAW_DIR / cond_lower / cond_lower / f"c_{cond_lower[1:]}_{run_id:03d}.csv"
    sig = np.loadtxt(path, delimiter=",")
    if sig.ndim != 2 or sig.shape[1] != 7:
        raise ValueError(f"{path} has shape {sig.shape}, expected [N,7]")
    return sig


def get_stable_window(signal: np.ndarray, window: tuple[int, int]) -> np.ndarray:
    """Slice the force channels (Fx,Fy,Fz = columns 0,1,2) to a stable region."""
    start, end = window
    if signal.shape[0] < end:
        raise ValueError(f"Signal length {signal.shape[0]} < required window end {end}")
    return signal[start:end, 0:3]  # Fx, Fy, Fz only -- paper uses force only, not vibration/AE


def split_subwindows(force_xyz: np.ndarray, length: int = SUBWINDOW_LEN) -> list[np.ndarray]:
    """Non-overlapping length-`length` sub-windows (Explicit, PAPER_SPEC "Sub-window segmentation")."""
    n = force_xyz.shape[0] // length
    return [force_xyz[k * length:(k + 1) * length, :] for k in range(n)]


def resultant_force(force_xyz: np.ndarray) -> np.ndarray:
    """F = sqrt(Fx^2 + Fy^2 + Fz^2), paper Eq. (8). Applied to raw (undenoised) force per Fig. 8."""
    return np.sqrt(np.sum(force_xyz.astype(np.float64) ** 2, axis=1))


def wavelet_denoise(x: np.ndarray, wavelet: str = WAVELET, level: int = WAVELET_LEVEL,
                     mode: str = WAVELET_MODE) -> np.ndarray:
    """Sym7 wavelet, level-6 decomposition, soft thresholding (Explicit).

    Threshold VALUE formula is Missing in the paper -- implementation
    choice: universal (VisuShrink) threshold with MAD-based sigma
    estimated from the finest detail coefficients, applied per-level.
    """
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level)
    detail_finest = coeffs[-1]
    sigma = np.median(np.abs(detail_finest)) / 0.6745
    n = len(x)
    uthresh = sigma * np.sqrt(2 * np.log(max(n, 2)))
    denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, uthresh, mode=mode) for c in coeffs[1:]]
    rec = pywt.waverec(denoised_coeffs, wavelet=wavelet)
    return rec[:n]


def mtf_encode(x: np.ndarray, field_size: int = MTF_FIELD_SIZE, n_bins: int = MTF_N_QUANTILE_BINS) -> np.ndarray:
    """Markov Transition Field, paper Eq. (1).

    1. Resample x (length 2000 after wavelet denoise) down to `field_size` samples.
    2. Normalize resampled series to [0,1].
    3. Quantile-bin into `n_bins` bins.
    4. Build the first-order Markov transition matrix W [n_bins,n_bins].
    5. Expand W into the [field_size,field_size] MTF matrix M: M[i,j] =
       W[bin(x_i), bin(x_j)], the transition probability from the bin of
       sample i to the bin of sample j.

    Returns an [field_size, field_size] float array in [0,1].
    """
    xr = scipy_resample(x, field_size).astype(np.float64)
    xmin, xmax = xr.min(), xr.max()
    xnorm = (xr - xmin) / (xmax - xmin + 1e-12)

    quantiles = np.quantile(xnorm, np.linspace(0, 1, n_bins + 1))
    quantiles[0] -= 1e-9
    quantiles[-1] += 1e-9
    bins = np.clip(np.digitize(xnorm, quantiles) - 1, 0, n_bins - 1)

    W = np.zeros((n_bins, n_bins), dtype=np.float64)
    for t in range(field_size - 1):
        W[bins[t], bins[t + 1]] += 1.0
    row_sums = W.sum(axis=1, keepdims=True)
    W = np.divide(W, row_sums, out=np.zeros_like(W), where=row_sums > 0)

    M = W[bins[:, None], bins[None, :]]
    return M


def mtf_to_rgb_image(mtf_field: np.ndarray, vit_size: int = VIT_INPUT_SIZE) -> np.ndarray:
    """Single-channel MTF field -> jet-colormap RGB -> bilinear resize to
    [vit_size, vit_size, 3], uint8."""
    import matplotlib as mpl
    from PIL import Image

    fmin, fmax = mtf_field.min(), mtf_field.max()
    fnorm = (mtf_field - fmin) / (fmax - fmin + 1e-12)
    rgba = mpl.colormaps["jet"](fnorm)  # [H,W,4] in [0,1]
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB").resize((vit_size, vit_size), Image.BILINEAR)
    return np.array(img)


def signal_to_mtf_image(force_xyz_2000: np.ndarray) -> np.ndarray:
    """Full per-sub-window pipeline: [2000,3] raw force -> [384,384,3] uint8 image."""
    f = resultant_force(force_xyz_2000)
    f_denoised = wavelet_denoise(f)
    mtf_field = mtf_encode(f_denoised)
    return mtf_to_rgb_image(mtf_field)


def build_main_samples(condition: str, run_id: int) -> list[np.ndarray]:
    """Main (unbalanced) A/B-dataset samples for one cutting pass: 5 images
    from the 90,000-100,000 stable region."""
    sig = load_raw_signal(condition, run_id)
    force_xyz = get_stable_window(sig, MAIN_WINDOW)
    images = [signal_to_mtf_image(sw) for sw in split_subwindows(force_xyz)]
    return images

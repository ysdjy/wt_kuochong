# -*- coding: utf-8 -*-
r"""
Multi-source Attention (Multi-Attention-CNN) native preprocessing pipeline,
reimplemented from `baselines/multi_source_attention/PAPER_SPEC.md`
(Wei et al., RCIM 2024, DOI: 10.1016/j.rcim.2024.102741).

Pipeline (see PAPER_SPEC.md §2 for the annotated architecture diagram):

    raw Fx,Fy,Fz,Vx,Vy,Vz  (PHM2010, archive/c{cond}/c{cond}/c_{cond}_{run:03d}.csv)
      -> "middle region" of the cutting pass                    [Missing -> central 50%]
      -> per-axis resample to 224 samples                       [Missing, paired w/ CWT-scale choice]
      -> per-axis Continuous Wavelet Transform (complex Morlet)  [Missing -> cmor1.5-1.0, 224 log-spaced scales]
      -> per-image min-max normalize to [0,1]                   [Missing]
      -> stack 3 axes -> RGB                                     -> force image [224,224,3]
                                                                     vibration image [224,224,3]

Every "Missing in paper" choice below is documented in PAPER_SPEC.md and
repeated here only as a short pointer comment -- PAPER_SPEC.md is the
source of truth.

This module has NO project-shared preprocessing dependency (does not
import from 代码/ or from baselines/mtf_avitk/); fully self-contained per
the task's directory-isolation requirement. It only uses
`baselines/multi_source_attention/data/label_utils.py` for Unified
Protocol B stage labels, never for signal preprocessing. Protocol A
(original-paper sanity reproduction) stage labels are defined in this
file (`ORIGINAL_STAGE_RANGES`), reproducing the paper's own EM-derived
fixed pass-index partition (Table 1).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pywt
from scipy.signal import resample as scipy_resample

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../论文
ARCHIVE_DIR = PROJECT_ROOT / "archive"

CHANNEL_NAMES = ["Fx", "Fy", "Fz", "Vx", "Vy", "Vz", "AE"]  # standard PHM2010 7-channel order
FORCE_COLS = [0, 1, 2]
VIB_COLS = [3, 4, 5]

# --- "Middle region" (PAPER_SPEC.md "Stable-region signal segment" row) ---
MIDDLE_REGION_FRACTION = (0.25, 0.75)  # Missing in paper -> central 50% of the raw pass

# --- CWT (PAPER_SPEC.md "Continuous Wavelet Transform parameters" row) ---
CWT_WAVELET = "cmor1.5-1.0"  # Missing in paper -> standard Morlet CWT convention for this literature
IMAGE_SIZE = 224             # Explicit (paper §2.4): 224x224x3 per source
N_SCALES = IMAGE_SIZE        # Missing in paper -> matches image side length 1:1 (no scale-axis resize needed)


def load_raw_signal(condition: str, run_id: int) -> np.ndarray:
    """Return the raw 7-channel signal for one cutting pass, shape [N,7]."""
    cond_lower = condition.lower()
    path = ARCHIVE_DIR / cond_lower / cond_lower / f"c_{cond_lower[1:]}_{run_id:03d}.csv"
    sig = np.loadtxt(path, delimiter=",")
    if sig.ndim != 2 or sig.shape[1] != 7:
        raise ValueError(f"{path} has shape {sig.shape}, expected [N,7]")
    return sig


def get_middle_region(signal: np.ndarray, fraction: tuple[float, float] = MIDDLE_REGION_FRACTION) -> np.ndarray:
    """Central segment of the raw signal ("middle region", Missing in paper
    -> central 50% by sample-count, see PAPER_SPEC.md)."""
    n = signal.shape[0]
    start = int(round(n * fraction[0]))
    end = int(round(n * fraction[1]))
    return signal[start:end, :]


def cwt_scalogram(x_1d: np.ndarray, image_size: int = IMAGE_SIZE, n_scales: int = N_SCALES,
                   wavelet: str = CWT_WAVELET) -> np.ndarray:
    """One axis, one image side: resample to `image_size` time samples
    (Missing in paper), then complex-Morlet CWT with `n_scales` log-spaced
    scales (Missing in paper) -> [n_scales, image_size] magnitude matrix,
    min-max normalized to [0,1] (Missing in paper)."""
    xr = scipy_resample(x_1d, image_size).astype(np.float64)
    scales = np.geomspace(1, image_size, n_scales)
    coeffs, _freqs = pywt.cwt(xr, scales, wavelet)
    mag = np.abs(coeffs)  # [n_scales, image_size]
    mmin, mmax = mag.min(), mag.max()
    return (mag - mmin) / (mmax - mmin + 1e-12)


def axes_to_rgb_image(signal_3axis: np.ndarray, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """[N,3] (one source's 3 axes) -> [image_size,image_size,3] uint8 image,
    one CWT scalogram per axis mapped onto one RGB channel (Missing in
    paper -> per-axis-to-RGB-channel mapping, see PAPER_SPEC.md "Per-axis
    vs. combined force/vibration signal")."""
    channels = [cwt_scalogram(signal_3axis[:, ax], image_size=image_size, n_scales=image_size) for ax in range(3)]
    img = np.stack(channels, axis=-1)  # [H,W,3] in [0,1]
    return (img * 255.0).astype(np.uint8)


def build_sample(condition: str, run_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Full per-cutting-pass pipeline -> (force_image, vibration_image),
    each [224,224,3] uint8."""
    sig = load_raw_signal(condition, run_id)
    mid = get_middle_region(sig)
    force_img = axes_to_rgb_image(mid[:, FORCE_COLS])
    vib_img = axes_to_rgb_image(mid[:, VIB_COLS])
    return force_img, vib_img


# =========================================================
# Protocol A (original-paper sanity reproduction) stage labels
# =========================================================
# Explicit: paper's own EM-derived fixed pass-index partition, Table 1.
# Applied per-condition, NOT reused for Protocol B (which uses this
# project's condition-relative labels via data/label_utils.py instead).
ORIGINAL_STAGE_RANGES = {
    "C1": {"initial": (1, 47), "normal": (48, 146), "severe": (147, 315)},
    "C4": {"initial": (1, 135), "normal": (136, 204), "severe": (205, 315)},
    "C6": {"initial": (1, 81), "normal": (82, 188), "severe": (189, 315)},
}
ORIGINAL_STAGE_TO_ID = {"initial": 0, "normal": 1, "severe": 2}
ORIGINAL_ID_TO_STAGE = {0: "initial", 1: "normal", 2: "severe"}


def original_stage_of(condition: str, run_id: int) -> str:
    for stage, (lo, hi) in ORIGINAL_STAGE_RANGES[condition].items():
        if lo <= run_id <= hi:
            return stage
    raise ValueError(f"run_id {run_id} out of range for {condition}")


if __name__ == "__main__":
    f_img, v_img = build_sample("C1", 1)
    print(f"C1 run 1: force {f_img.shape} {f_img.dtype}, vibration {v_img.shape} {v_img.dtype}")
    print("stage:", original_stage_of("C1", 1))

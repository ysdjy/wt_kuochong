# -*- coding: utf-8 -*-
r"""
B5 (Multi-source Channel-Spatial Attention) raw-signal preprocessing,
adapted for MTW-CM (Mendeley multi-machine tool wear, raw .h5 files) -- a
NEW adaptation, not a port of an existing baseline.

Channel choice (verified 2026-09 via direct h5py inspection of one file per
machine -- see MTW_TASK_AUDIT.md for the dataset background): MTW-CM's raw
per-run h5 file has TWO groups, `signals_sensor/` (high-rate, ~110k samples)
and `signals_machine/` (control-loop rate, ~2.2k samples). Schema differs
by machine in `signals_machine/` (M1 has torque_axis_x/y/z but no
force_axis_*; M2/M3 have force_axis_x/y/z but no torque_axis_*) -- but
`signals_sensor/force_sensor_{x,y,z}` (a genuine 3-axis force sensor,
directly analogous to PHM2010's Fx/Fy/Fz) and `signals_machine/
tool_position_{x,y,z}` are BOTH present, with the same shape convention,
across all three machines' probed files. These two channel triples are used
for the two B5 branches:
    "branch A" (force):         signals_sensor/force_sensor_{x,y,z}
    "branch B" (tool position):  signals_machine/tool_position_{x,y,z}
No vibration/accelerometer channel exists in this raw dataset (unlike
PHM2010's Vx/Vy/Vz) -- tool_position is the best available second-modality
substitute that is genuinely common to all 3 machines.

CWT math (complex Morlet, 224 log-spaced scales, per-image min-max
normalize) is identical to `b5_signal_preprocessing.py` (PHM2010) --
only the raw-signal source and channel choice differ.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pywt
from scipy.signal import resample as scipy_resample

THIS_DIR = Path(__file__).resolve().parent
EXPAND_ROOT = THIS_DIR.parents[3]
PROJECT_ROOT = EXPAND_ROOT.parent
MTW_RAW_DIR = PROJECT_ROOT / "Multivariate time series data of milling processes with varying tool wear and machine tools"

BRANCH_A_CHANNELS = ["signals_sensor/force_sensor_x", "signals_sensor/force_sensor_y", "signals_sensor/force_sensor_z"]
BRANCH_B_CHANNELS = ["signals_machine/tool_position_x", "signals_machine/tool_position_y", "signals_machine/tool_position_z"]

MIDDLE_REGION_FRACTION = (0.25, 0.75)
CWT_WAVELET = "cmor1.5-1.0"
IMAGE_SIZE = 224
N_SCALES = IMAGE_SIZE

def get_middle_region(signal: np.ndarray, fraction=MIDDLE_REGION_FRACTION) -> np.ndarray:
    n = signal.shape[0]
    start = int(round(n * fraction[0]))
    end = int(round(n * fraction[1]))
    return signal[start:end]


def cwt_scalogram(x_1d: np.ndarray, image_size: int = IMAGE_SIZE, n_scales: int = N_SCALES,
                   wavelet: str = CWT_WAVELET) -> np.ndarray:
    xr = scipy_resample(x_1d, image_size).astype(np.float64)
    scales = np.geomspace(1, image_size, n_scales)
    coeffs, _freqs = pywt.cwt(xr, scales, wavelet)
    mag = np.abs(coeffs)
    mmin, mmax = mag.min(), mag.max()
    return (mag - mmin) / (mmax - mmin + 1e-12)


def axes_to_rgb_image(channels_1d: list[np.ndarray], image_size: int = IMAGE_SIZE) -> np.ndarray:
    channels = [cwt_scalogram(np.asarray(c, dtype=float), image_size=image_size, n_scales=image_size) for c in channels_1d]
    img = np.stack(channels, axis=-1)
    return (img * 255.0).astype(np.uint8)


def build_sample(source_file: str) -> tuple[np.ndarray, np.ndarray]:
    """(branch_a_image, branch_b_image), each [224,224,3] uint8.
    `source_file` comes directly from the frozen-preprocess feat_*_frozen.csv's
    own `source_file` column (see dcpsr.datasets.mendeley.read_filelist) --
    no independent filename lookup/reconstruction needed."""
    path = MTW_RAW_DIR / source_file
    with h5py.File(path, "r") as h:
        a_channels = [get_middle_region(np.asarray(h[c][()], dtype=float).reshape(-1)) for c in BRANCH_A_CHANNELS]
        b_channels = [get_middle_region(np.asarray(h[c][()], dtype=float).reshape(-1)) for c in BRANCH_B_CHANNELS]
    img_a = axes_to_rgb_image(a_channels)
    img_b = axes_to_rgb_image(b_channels)
    return img_a, img_b

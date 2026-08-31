# -*- coding: utf-8 -*-
r"""
B5 (Multi-source Channel-Spatial Attention) raw-signal preprocessing,
adapted for NASA Milling (mill.mat) -- a NEW adaptation, not a port of an
existing baseline (unlike every other B5_* module in this repo, which are
vendored from the old project's `baselines/multi_source_attention/`).

NASA's raw signal has 6 channels (`SIGNAL_FIELDS` in `代码/9.1nasa数据实验.py`):
    smcAC, smcDC   -- spindle motor current (AC/DC)
    vib_table, vib_spindle -- vibration (table-mounted / spindle-mounted)
    AE_table, AE_spindle   -- acoustic emission (table-mounted / spindle-mounted)

PHM2010's B5 needs two 3-channel branches (it has real 3-axis force
Fx,Fy,Fz and 3-axis vibration Vx,Vy,Vz). NASA has no true multi-axis force
channel, so this adaptation groups the 6 channels into two 3-channel
branches by SENSOR LOCATION rather than physical quantity (documented
choice, not a paper/baseline requirement since B5 was never applied to
NASA before):
    "branch A" (table-side):   smcAC,  vib_table,   AE_table
    "branch B" (spindle-side): smcDC,  vib_spindle, AE_spindle

CWT math (complex Morlet, 224 log-spaced scales, per-image min-max
normalize) is otherwise identical to `b5_signal_preprocessing.py`
(PHM2010) -- only the raw-signal source and channel grouping differ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pywt
from scipy.signal import resample as scipy_resample

THIS_DIR = Path(__file__).resolve().parent
EXPAND_ROOT = THIS_DIR.parents[3]
PROJECT_ROOT = EXPAND_ROOT.parent
CODE_DIR = PROJECT_ROOT / "代码"
NASA_SCRIPT = CODE_DIR / "9.1nasa数据实验.py"
MILL_MAT = PROJECT_ROOT / "mill" / "mill.mat"

BRANCH_A_FIELDS = ["smcAC", "vib_table", "AE_table"]
BRANCH_B_FIELDS = ["smcDC", "vib_spindle", "AE_spindle"]

MIDDLE_REGION_FRACTION = (0.25, 0.75)
CWT_WAVELET = "cmor1.5-1.0"
IMAGE_SIZE = 224
N_SCALES = IMAGE_SIZE

_MILL_INDEX = None  # lazily built {(case, run): item}, one process-lifetime load


def _import_nasa_module():
    import tempfile
    import types
    src = NASA_SCRIPT.read_text(encoding="utf-8")
    safe_out_dir = Path(tempfile.gettempdir()) / "nasa_b5_runner_side_outputs"
    old_mat_line = 'MAT_FILE = Path(r"C:\\Users\\wangting\\Desktop\\博士开题\\公开数据\\1NASA\\mill.mat")'
    old_out_line = 'OUT_DIR = Path(r"C:\\Users\\wangting\\Desktop\\博士开题\\公开数据\\1NASA\\nasa_dcpsr_results_stageaware_opt")'
    assert old_mat_line in src, "MAT_FILE line not found -- old script changed unexpectedly, aborting"
    assert old_out_line in src, "OUT_DIR line not found -- old script changed unexpectedly, aborting"
    src = src.replace(old_mat_line, f'MAT_FILE = Path(r"{MILL_MAT}")')
    src = src.replace(old_out_line, f'OUT_DIR = Path(r"{safe_out_dir}")')
    mod = types.ModuleType("nasa_b5_runner_9_1")
    mod.__file__ = str(NASA_SCRIPT)
    sys.modules["nasa_b5_runner_9_1"] = mod
    exec(compile(src, str(NASA_SCRIPT), "exec"), mod.__dict__)
    return mod


def _mill_index() -> dict:
    global _MILL_INDEX
    if _MILL_INDEX is None:
        nasa = _import_nasa_module()
        mill = nasa.load_nasa_mat()
        idx = {}
        for item in mill:
            case = int(item.case)
            run = int(item.run)
            idx[(case, run)] = item
        _MILL_INDEX = idx
    return _MILL_INDEX


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


def build_sample(case: int, run: int) -> tuple[np.ndarray, np.ndarray]:
    """(branch_a_image, branch_b_image), each [224,224,3] uint8."""
    idx = _mill_index()
    key = (int(case), int(run))
    if key not in idx:
        raise KeyError(f"NASA mill.mat has no entry for (case={case}, run={run})")
    item = idx[key]
    a_channels = [get_middle_region(np.asarray(getattr(item, f), dtype=float).reshape(-1)) for f in BRANCH_A_FIELDS]
    b_channels = [get_middle_region(np.asarray(getattr(item, f), dtype=float).reshape(-1)) for f in BRANCH_B_FIELDS]
    img_a = axes_to_rgb_image(a_channels)
    img_b = axes_to_rgb_image(b_channels)
    return img_a, img_b

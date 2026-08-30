#!/usr/bin/env python
"""Repo self-check: verify a fresh clone + environment is ready to run
PHM2010 experiments, WITHOUT running any real training. Checks (in order):

1. Python version.
2. Core third-party imports (numpy, pandas, sklearn, torch, yaml, PyWavelets).
3. Every discoverable method's adapter.py imports cleanly and exposes
   ADAPTER_CLASS with a matching method_id (uses shared/runners/registry.py,
   so this exercises the exact same import-isolation path run_phm2010.py uses).
4. shared/metrics + shared/phm2010 unit tests pass.
5. data/PHM2010/features/run_level_features_all.csv is present with the
   expected sha256 (does NOT require data/PHM2010/raw/ -- that's only needed
   for the 4 raw-signal methods B5-B8; reported as a warning, not a failure,
   if missing, since B1-B4/B9 don't need it).
6. GPU: reports what's detected (informational only, never a failure --
   CPU-only machines are supported).

Exit code 0 = all checks passed (or only non-fatal warnings). Exit code 1 =
at least one fatal check failed.

Run: python scripts/self_check.py
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "shared"))

FEATURE_FILE = REPO_ROOT / "data" / "PHM2010" / "features" / "run_level_features_all.csv"
EXPECTED_FEATURE_SHA256 = "6e8affeb681d0b386e453421a0df7a66932138199eb236403d27b797c11eeb88"

FATAL = []
WARN = []


def check(label: str, fn, fatal: bool = True):
    try:
        detail = fn()
        print(f"  [OK] {label}" + (f" -- {detail}" if detail else ""))
        return True
    except Exception as exc:  # noqa: BLE001
        msg = f"{label}: {exc}"
        (FATAL if fatal else WARN).append(msg)
        tag = "FATAL" if fatal else "WARN"
        print(f"  [{tag}] {label} -- {exc}")
        return False


def check_python_version():
    v = sys.version_info
    if v < (3, 10):
        raise RuntimeError(f"Python {v.major}.{v.minor} too old, need >=3.10")
    return f"{v.major}.{v.minor}.{v.micro}"


def check_core_imports():
    import numpy, pandas, sklearn, yaml  # noqa: F401
    return f"numpy {numpy.__version__}, pandas {pandas.__version__}, sklearn {sklearn.__version__}"


def check_torch():
    import torch
    cuda = torch.cuda.is_available()
    return f"torch {torch.__version__}, cuda_available={cuda}"


def check_pywavelets():
    import pywt
    return f"PyWavelets {pywt.__version__}"


def check_methods_registry():
    from runners.registry import list_methods

    methods = list_methods()
    bad = [m for m in methods if m["method_name"].startswith("<import error")]
    report = ", ".join(f"{m['method_id']}={m['method_name']}" for m in methods)
    if bad:
        raise RuntimeError(f"{len(bad)}/{len(methods)} methods failed to import: {report}")
    return f"{len(methods)} methods OK: {report}"


def check_shared_unit_tests():
    # A fresh TestLoader per discover() call: TestLoader caches top_level_dir
    # from its FIRST discover() call, and reusing one loader for a second,
    # sibling (not nested) test directory makes it compute a relative path
    # containing ".." between the two -- which unittest's own loader asserts
    # against ("Path must be within the project"). One loader per call avoids
    # that entirely.
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().discover(str(REPO_ROOT / "shared" / "metrics" / "tests")))
    suite.addTests(unittest.TestLoader().discover(str(REPO_ROOT / "shared" / "phm2010" / "tests")))
    # io.StringIO instead of the OS null device: some sandboxed execution
    # environments restrict opening device files like nul/dev-null even for
    # writing, so avoid touching the filesystem at all to discard the runner's
    # verbose output.
    import io

    runner = unittest.TextTestRunner(verbosity=0, stream=io.StringIO())
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise RuntimeError(f"{len(result.failures)} failures, {len(result.errors)} errors")
    return f"{result.testsRun} tests passed"


def check_feature_file():
    if not FEATURE_FILE.exists():
        raise RuntimeError(f"missing: {FEATURE_FILE}")
    h = hashlib.sha256()
    with open(FEATURE_FILE, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if digest != EXPECTED_FEATURE_SHA256:
        raise RuntimeError(f"sha256 mismatch: got {digest}, expected {EXPECTED_FEATURE_SHA256}")
    return f"sha256 verified ({FEATURE_FILE.stat().st_size} bytes)"


def check_raw_data():
    raw_root = REPO_ROOT / "data" / "PHM2010" / "raw"
    if not raw_root.exists():
        raise RuntimeError(
            f"not present at {raw_root} -- run scripts/download_phm2010.py if you plan to "
            f"use B5-B8 (raw-signal methods); B1-B4/B9 only need the feature CSV, already verified."
        )
    return str(raw_root)


def check_gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
        return "nvidia-smi present but returned nothing (no GPU?)"
    except FileNotFoundError:
        return "nvidia-smi not found -- CPU-only machine (supported)"


def main() -> int:
    print(f"=== wt_kuochong self-check ({platform.system()} {platform.release()}) ===\n")

    print("1. Python version")
    check("python version", check_python_version)

    print("\n2. Core dependencies")
    check("numpy/pandas/sklearn/yaml", check_core_imports)
    check("torch", check_torch)
    check("PyWavelets (needed by B5/B6)", check_pywavelets, fatal=False)

    print("\n3. Method adapters (registry auto-discovery)")
    check("methods/*/adapter.py import + ADAPTER_CLASS", check_methods_registry)

    print("\n4. shared/ unit tests")
    check("shared/metrics + shared/phm2010 tests", check_shared_unit_tests)

    print("\n5. PHM2010 data")
    check("features/run_level_features_all.csv sha256", check_feature_file)
    check("raw/ (B5-B8 only)", check_raw_data, fatal=False)

    print("\n6. GPU (informational)")
    check("nvidia-smi", check_gpu, fatal=False)

    print(f"\n=== Summary: {len(FATAL)} fatal, {len(WARN)} warning ===")
    if FATAL:
        for m in FATAL:
            print(f"  FATAL: {m}")
        return 1
    if WARN:
        for m in WARN:
            print(f"  WARN: {m}")
    print("Self-check PASSED" + (" (with warnings)" if WARN else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

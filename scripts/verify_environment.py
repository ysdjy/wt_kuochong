#!/usr/bin/env python
"""Lightweight, early-stage environment verification -- run right after
`pip install` by the bootstrap scripts, BEFORE any repo-specific check.
Only verifies that the core third-party packages import and are usable; does
NOT check the method registry, run unit tests, or verify data files (that's
scripts/self_check.py, which bootstrap runs as its final step).

Usage:
    python scripts/verify_environment.py
"""
from __future__ import annotations

import platform
import sys

REQUIRED = [
    "numpy", "pandas", "scipy", "sklearn", "xgboost", "matplotlib",
    "yaml", "pywt", "h5py", "pyarrow", "tqdm", "tabulate", "joblib", "torch",
]


def main() -> int:
    print(f"Python {platform.python_version()} ({platform.system()} {platform.release()})")
    failures = []
    for name in REQUIRED:
        try:
            mod = __import__(name)
            version = getattr(mod, "__version__", "?")
            print(f"  [OK] {name} {version}")
        except ImportError as exc:
            print(f"  [MISSING] {name}: {exc}")
            failures.append(name)

    try:
        import torch
        print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] could not query torch CUDA state: {exc}")

    if failures:
        print(f"\nFAILED: missing packages: {failures}")
        print("Re-run: pip install -r environment/requirements.txt "
              "&& pip install -r environment/requirements_windows.txt (or _ubuntu.txt)")
        return 1
    print("\nAll core dependencies present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

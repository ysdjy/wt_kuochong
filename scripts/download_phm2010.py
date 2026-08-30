#!/usr/bin/env python
"""Downloads and unpacks the raw PHM2010 tool-wear dataset from Kaggle.

Source: Kaggle dataset `tobbyrui/phm2010` -- the only concretely-named public
source of this data anywhere in the old parent project's code/docs
(`baselines/htt_net/data/build_run_level_features.py` docstring and
`baselines/htt_net/README.md`, both verbatim: "standard Kaggle 'tobbyrui/phm2010'
layout"). No other URL/handle for this dataset was found anywhere in that
project -- this script does not invent or substitute a different source.

Only needed for the 4 raw-signal methods (B5 Multi-source Attention, B6
MTF-AViTK, B7 Dynamic GIN+TGP, B8 DP2Net). B1-B4 and B9 only need the already-
committed `data/PHM2010/features/run_level_features_all.csv` and do not
require this script.

One-time setup (see MANUAL_RUN.md for the full walkthrough):
    1. Create a Kaggle account, go to Account -> Create New API Token.
    2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json (Linux/Mac) or
       %USERPROFILE%\\.kaggle\\kaggle.json (Windows). NEVER commit this file
       (already in .gitignore).
    3. pip install kaggle   (already in environment/requirements.txt as an
       optional extra -- see below)

Usage:
    python scripts/download_phm2010.py
    python scripts/download_phm2010.py --dest data/PHM2010/raw --force
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = REPO_ROOT / "data" / "PHM2010" / "raw"
KAGGLE_DATASET = "tobbyrui/phm2010"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", default=str(DEFAULT_DEST), help="Destination directory")
    p.add_argument("--force", action="store_true", help="Re-download even if --dest already looks populated")
    return p.parse_args()


def kaggle_available() -> bool:
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, timeout=15)
        return True
    except FileNotFoundError:
        return False


def main() -> int:
    args = parse_args()
    dest = Path(args.dest)

    if dest.exists() and any(dest.iterdir()) and not args.force:
        print(f"[download_phm2010] {dest} already populated ({sum(1 for _ in dest.rglob('*') if _.is_file())} files) "
              f"-- skipping (use --force to re-download). Run scripts/verify_phm2010.py to check completeness.")
        return 0

    if not kaggle_available():
        print(
            "[download_phm2010] ERROR: the `kaggle` CLI is not installed or not on PATH.\n"
            "  Install it with:  pip install kaggle\n"
            "  Then set up your API token -- see MANUAL_RUN.md 'Kaggle credential setup'.\n"
            f"  Dataset handle: {KAGGLE_DATASET}\n"
            "  (This project does not implement its own Kaggle downloader -- the official "
            "`kaggle` CLI already handles auth, resume, and integrity correctly.)",
            file=sys.stderr,
        )
        return 1

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[download_phm2010] downloading Kaggle dataset {KAGGLE_DATASET} into {dest} ...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(dest), "--unzip"],
        cwd=str(dest),
    )
    if result.returncode != 0:
        print(
            "[download_phm2010] ERROR: `kaggle datasets download` failed (see output above). "
            "Common causes: missing/invalid ~/.kaggle/kaggle.json, not having accepted the "
            "dataset's terms on kaggle.com first. See MANUAL_RUN.md.",
            file=sys.stderr,
        )
        return 1

    # `kaggle ... --unzip` usually unzips automatically; handle a leftover zip defensively.
    for zpath in dest.glob("*.zip"):
        print(f"[download_phm2010] unzipping leftover archive {zpath.name} ...")
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(dest)
        zpath.unlink()

    print(f"[download_phm2010] done. Run scripts/verify_phm2010.py to check completeness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

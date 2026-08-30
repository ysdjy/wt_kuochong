#!/usr/bin/env python
"""Verifies the raw PHM2010 dataset under data/PHM2010/raw/ is complete for
the 3 cutters this project uses (C1, C4, C6 -- the ones PHM2010 ships full
wear-progression labels for; C2/C3/C5 are the original challenge's unlabeled
test cutters and are not used anywhere in this project).

Expected layout (matches the old project's already-organized local copy,
verified directly against `archive/c1/c1/*.csv` + `archive/c1/c1_wear.csv` at
build time -- 315 signal files named `c_{N}_{run:03d}.csv` where N is the
condition's numeric suffix, e.g. c1 -> "c_1_001.csv".."c_1_315.csv", and a
316-line (315 runs + header) wear.csv):

    data/PHM2010/raw/
    └─ c{1,4,6}/
       ├─ c{1,4,6}/
       │  └─ c_{1,4,6}_{run:03d}.csv     (7-channel signal, no header, ~50kHz)
       └─ c{1,4,6}_wear.csv               (columns: cut, flute_1, flute_2, flute_3)

NOTE: this exact nesting was verified against the OLD project's already-
downloaded-and-organized local copy of the Kaggle dataset `tobbyrui/phm2010`,
not against a byte-for-byte fresh `kaggle datasets download` unzip on a new
machine (not independently re-verified in this round). If a fresh download's
top-level folder names differ, reorganize to match the layout above before
running this script -- see MANUAL_RUN.md.

Usage:
    python scripts/verify_phm2010.py
    python scripts/verify_phm2010.py --root data/PHM2010/raw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "data" / "PHM2010" / "raw"

CONDITIONS = ["c1", "c4", "c6"]
EXPECTED_RUNS_PER_CONDITION = 315  # matches the common evaluation universe's full lifecycle length


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    return p.parse_args()


def verify_condition(root: Path, cond: str) -> dict:
    cond_dir = root / cond
    signal_dir = cond_dir / cond
    wear_file = cond_dir / f"{cond}_wear.csv"

    report = {"condition": cond, "ok": True, "problems": []}

    cond_num = cond.lstrip("c")  # "c1" -> "1", filenames are c_{N}_{run:03d}.csv (verified against archive/c1/c1/)

    if not signal_dir.is_dir():
        report["ok"] = False
        report["problems"].append(f"missing signal directory: {signal_dir}")
        report["n_signal_files"] = 0
    else:
        signal_files = sorted(signal_dir.glob(f"c_{cond_num}_*.csv"))
        report["n_signal_files"] = len(signal_files)
        if len(signal_files) != EXPECTED_RUNS_PER_CONDITION:
            report["ok"] = False
            report["problems"].append(
                f"expected {EXPECTED_RUNS_PER_CONDITION} signal files, found {len(signal_files)}"
            )

    if not wear_file.exists():
        report["ok"] = False
        report["problems"].append(f"missing wear file: {wear_file}")
    else:
        n_lines = sum(1 for _ in open(wear_file, encoding="utf-8", errors="replace"))
        report["wear_file_lines"] = n_lines
        if n_lines < EXPECTED_RUNS_PER_CONDITION:  # header + >= runs
            report["ok"] = False
            report["problems"].append(
                f"{wear_file} has only {n_lines} lines, expected >= {EXPECTED_RUNS_PER_CONDITION + 1} (incl. header)"
            )

    return report


def main() -> int:
    args = parse_args()
    root = Path(args.root)

    print(f"=== verify_phm2010: {root} ===")
    if not root.exists():
        print(f"ERROR: {root} does not exist. Run scripts/download_phm2010.py first.", file=sys.stderr)
        return 1

    all_ok = True
    for cond in CONDITIONS:
        r = verify_condition(root, cond)
        status = "OK" if r["ok"] else "FAILED"
        print(f"  [{status}] {cond}: {r.get('n_signal_files', 0)} signal files, "
              f"wear file {r.get('wear_file_lines', 'MISSING')} lines")
        for p in r["problems"]:
            print(f"           - {p}")
        all_ok = all_ok and r["ok"]

    if all_ok:
        print("\nAll 3 conditions (C1, C4, C6) verified complete.")
        return 0
    print(
        "\nVerification FAILED. If you just ran scripts/download_phm2010.py and the Kaggle "
        "archive unzipped to a different folder layout than expected, reorganize it to match "
        "the layout documented at the top of this script (see MANUAL_RUN.md).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

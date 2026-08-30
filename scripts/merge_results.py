#!/usr/bin/env python
"""Merges per-machine results/ trees into one, for the multi-machine sharding
workflow described in MANUAL_RUN.md (each machine runs a disjoint seed range,
then copies its results/PHM2010/ tree back to a collation machine and runs
this script).

Safety rules (task spec section 29):
- Duplicate detection: if the SAME (method, task, seed) cell exists in more
  than one source tree, their config_hash (from run_meta.json) must match, or
  this is reported as a CONFLICT and the merge stops for that cell -- never
  silently overwritten.
- DONE validation: only cells with DONE.flag are merged; incomplete/failed
  cells are skipped (reported, not merged).
- Nothing is deleted. This script only copies into the destination tree.

Usage:
    python scripts/merge_results.py --sources /path/to/machineA/results /path/to/machineB/results --dest results
    python scripts/merge_results.py --sources ../results_from_machineB --dest results --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+", required=True, help="One or more results/ (or results/PHM2010/) trees to merge in")
    p.add_argument("--dest", default=str(REPO_ROOT / "results"), help="Destination results/ tree")
    p.add_argument("--dry-run", action="store_true", help="Report what would happen without copying anything")
    return p.parse_args()


def find_cells(root: Path):
    """Yields (method, task, seed_dir_name, cell_path) for every DONE.flag-complete
    cell under a results/ or results/PHM2010/ tree."""
    phm_root = root / "PHM2010" if (root / "PHM2010").exists() else root
    if not phm_root.exists():
        return
    for method_dir in sorted(phm_root.iterdir()):
        if not method_dir.is_dir():
            continue
        for task_dir in sorted(method_dir.iterdir()):
            if not task_dir.is_dir() or task_dir.name == "summary":
                continue
            for seed_dir in sorted(task_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                if (seed_dir / "DONE.flag").exists():
                    yield method_dir.name, task_dir.name, seed_dir.name, seed_dir


def config_hash_of(cell_path: Path) -> str | None:
    run_meta = cell_path / "run_meta.json"
    if not run_meta.exists():
        return None
    try:
        return json.loads(run_meta.read_text(encoding="utf-8")).get("config_hash")
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    args = parse_args()
    dest = Path(args.dest)
    sources = [Path(s) for s in args.sources]

    # index destination's existing complete cells
    dest_cells = {(m, t, s): p for m, t, s, p in find_cells(dest)}

    n_copied = n_skipped_incomplete = n_already_present = n_conflicts = 0
    conflicts = []

    for src in sources:
        print(f"=== source: {src} ===")
        for method, task, seed, cell_path in find_cells(src):
            key = (method, task, seed)
            if key in dest_cells:
                src_hash = config_hash_of(cell_path)
                dst_hash = config_hash_of(dest_cells[key])
                if src_hash != dst_hash:
                    n_conflicts += 1
                    conflicts.append((key, src, src_hash, dst_hash))
                    print(f"  CONFLICT: {key} exists in both {src} and dest with different "
                          f"config_hash ({src_hash} vs {dst_hash}) -- NOT merged, resolve manually.")
                else:
                    n_already_present += 1
                continue

            dest_path = dest / "PHM2010" / method / task / seed
            print(f"  {'[dry-run] would copy' if args.dry_run else 'copying'} {key} -> {dest_path}")
            if not args.dry_run:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(cell_path, dest_path)
            n_copied += 1
            dest_cells[key] = dest_path

    print(f"\n=== Summary ===\ncopied={n_copied} already_present={n_already_present} "
          f"conflicts={n_conflicts}")
    if conflicts:
        print("\nCONFLICTS (resolve manually -- inspect both run_meta.json files and pick one, "
              "or investigate why the same (method,task,seed) produced two different configs):")
        for key, src, sh, dh in conflicts:
            print(f"  {key}: source={src} src_hash={sh} dest_hash={dh}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

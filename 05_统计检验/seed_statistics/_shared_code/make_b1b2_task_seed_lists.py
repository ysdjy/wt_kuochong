# -*- coding: utf-8 -*-
"""Regenerates the b1_*/b2_*_task_seed_list.txt files (TASK,SEED pairs, one
per line, seeds 0..100 inclusive) consumed by run_one_b1_task_seed.sh /
run_one_b2_task_seed.sh. Re-run this if you ever need to change the seed
range or task set -- the .txt files themselves are what's committed/used,
this script is just how they were generated.

Usage:
    python make_b1b2_task_seed_lists.py
"""
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

TASK_SETS = {
    "phm": ["D1", "D2", "D3"],
    "nasa": ["N1", "N2", "N3", "N4"],
    "mtw": ["D1-M", "D2-M", "D3-M"],
}
SEEDS = list(range(0, 101))  # 0..100 inclusive


def main():
    for method in ["b1", "b2"]:
        for dataset, tasks in TASK_SETS.items():
            lines = [f"{task},{seed}" for task in tasks for seed in SEEDS]
            out_path = THIS_DIR / f"{method}_{dataset}_task_seed_list.txt"
            out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"wrote {out_path.name}: {len(lines)} (task,seed) pairs")


if __name__ == "__main__":
    main()

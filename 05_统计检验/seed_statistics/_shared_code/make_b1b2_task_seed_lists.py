# -*- coding: utf-8 -*-
"""Regenerates the b{1,2,4}_*_task_seed_list.txt files (TASK,SEED pairs, one
per line, seeds 0..100 inclusive) consumed by run_one_b{1,2,4}_*_task_seed.sh.
Re-run this if you ever need to change the seed range or task set -- the
.txt files themselves are what's committed/used, this script is just how
they were generated.

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
# B4 (HTT-Net) currently only has a PHM2010 runner -- NASA/MTW variants not
# built yet (methods/B4_HTT_Net's own methods/ adapter is PHM2010-only too).
METHOD_DATASETS = {
    "b1": ["phm", "nasa", "mtw"],
    "b2": ["phm", "nasa", "mtw"],
    "b4": ["phm"],
}
SEEDS = list(range(0, 101))  # 0..100 inclusive


def main():
    for method, datasets in METHOD_DATASETS.items():
        for dataset in datasets:
            tasks = TASK_SETS[dataset]
            lines = [f"{task},{seed}" for task in tasks for seed in SEEDS]
            out_path = THIS_DIR / f"{method}_{dataset}_task_seed_list.txt"
            out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"wrote {out_path.name}: {len(lines)} (task,seed) pairs")


if __name__ == "__main__":
    main()

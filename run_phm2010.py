#!/usr/bin/env python
"""Unified PHM2010 experiment runner. Dispatches to a method's adapter (see
`shared/runners/registry.py`) for every (task, seed) cell requested; never
knows any method's internals itself.

Examples:
    python run_phm2010.py --method B9 --tasks all --seed-start 0 --seed-end 100 --device auto --workers 1 --resume
    python run_phm2010.py --method B4 --tasks all --seed-start 0 --seed-end 24 --resume      # machine A's shard
    python run_phm2010.py --method B4 --tasks all --seed-start 25 --seed-end 49 --resume     # machine B's shard
    python run_phm2010.py --method B1 --tasks D1 --seed-start 0 --seed-end 0 --smoke-test
    python run_phm2010.py --list-methods
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "shared"))

from phm2010.tasks import TASK_NAMES, resolve_tasks, test_cutter, train_cutters  # noqa: E402
from runners.registry import list_methods, load_adapter_class  # noqa: E402
from runners.gpu_gate import gpu_free_enough  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", help="Method id, e.g. B1..B9. See --list-methods.")
    p.add_argument("--tasks", default="all", help="'all' or comma-separated subset, e.g. D1,D3")
    p.add_argument("--seed-start", type=int, default=0, help="First TRAIN_SEED (inclusive)")
    p.add_argument("--seed-end", type=int, default=100, help="Last TRAIN_SEED (inclusive)")
    p.add_argument("--preprocess-seed", type=int, default=42, help="Fixed PREPROCESS_SEED for every cell")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--workers", type=int, default=1,
                    help="Sequential cells per worker; default 1. Raising this launches "
                         "concurrent GPU training and is NOT safe on an 8GB card without "
                         "knowing what you're doing -- see README.md.")
    p.add_argument("--resume", action="store_true", help="Skip (task, seed) cells with DONE.flag already present")
    p.add_argument("--force", action="store_true", help="Ignore DONE.flag / re-run even if already complete")
    p.add_argument("--smoke-test", action="store_true",
                    help="Plumbing check only: skips train(), writes to tmp/smoke_tests/, "
                         "never results/. Do not use for real results.")
    p.add_argument("--save-checkpoint", choices=["none", "final"], default="none",
                    help="Whether adapters that support it should persist a checkpoint. "
                         "Checkpoints are never committed to git regardless (see .gitignore).")
    p.add_argument("--results-root", default=None,
                    help="Override results/ location (default: <repo>/results)")
    p.add_argument("--list-methods", action="store_true", help="List discoverable methods and exit")
    p.add_argument("--dry-run", action="store_true", help="Print the cell plan without running anything")
    return p.parse_args(argv)


def seed_dir_name(seed: int) -> str:
    return f"seed{seed:03d}"


def build_cells(method_id: str, tasks: list[str], seed_start: int, seed_end: int):
    for task in tasks:
        for seed in range(seed_start, seed_end + 1):
            yield task, seed


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list_methods:
        for m in list_methods():
            print(f"{m['method_id']:>3}  {m['method_name']}")
        return 0

    if not args.method:
        print("error: --method is required (or use --list-methods)", file=sys.stderr)
        return 2

    tasks = resolve_tasks(args.tasks)
    if args.seed_end < args.seed_start:
        print("error: --seed-end must be >= --seed-start", file=sys.stderr)
        return 2

    cells = list(build_cells(args.method, tasks, args.seed_start, args.seed_end))

    if args.dry_run:
        print(f"method={args.method} tasks={tasks} seeds={args.seed_start}..{args.seed_end} "
              f"({len(cells)} cells) device={args.device} workers={args.workers} "
              f"resume={args.resume} smoke_test={args.smoke_test}")
        for task, seed in cells:
            print(f"  {args.method}/{task}/{seed_dir_name(seed)}")
        return 0

    try:
        adapter_cls = load_adapter_class(args.method)
    except Exception as exc:  # noqa: BLE001
        print(f"error: could not load method {args.method!r}: {exc}", file=sys.stderr)
        return 2

    results_root = Path(args.results_root) if args.results_root else REPO_ROOT / "results"
    # Find the method's own directory name (e.g. "B9_DC_PHSR") for the results path.
    from runners.registry import discover_method_dirs
    method_results_dirname = discover_method_dirs()[args.method].name

    if args.workers > 1:
        print(
            f"WARNING: --workers {args.workers} requested. This project defaults to 1 "
            f"deliberately -- concurrent GPU training previously caused CUDA OOM on this "
            f"machine's 8GB card (see 05_统计检验/seed_statistics/.../README.md). Proceeding "
            f"because you explicitly asked for it; if you did not mean to, Ctrl-C now.",
            file=sys.stderr,
        )

    n_done = n_skipped = n_failed = 0
    t_start = time.time()

    print(f"[run_phm2010] method={args.method} ({adapter_cls.method_name}) "
          f"tasks={tasks} seeds={args.seed_start}..{args.seed_end} ({len(cells)} cells) "
          f"device={args.device} smoke_test={args.smoke_test}")

    for i, (task, seed) in enumerate(cells, start=1):
        if args.smoke_test:
            # Smoke-test output must NEVER land in results/ (RESULTS_POLICY.md /
            # task spec section 40) -- route to tmp/smoke_tests/ instead.
            out_dir = REPO_ROOT / "tmp" / "smoke_tests" / args.method / task / seed_dir_name(seed)
        else:
            out_dir = results_root / "PHM2010" / method_results_dirname / task / seed_dir_name(seed)

        if args.device in ("cuda", "auto"):
            # Single-flight GPU gating: never launch a GPU cell while another
            # process already has meaningful GPU memory in use. Mirrors the old
            # project's already-validated run_transfer_tasks.py::gpu_free_enough
            # pattern -- this project defaults to --workers 1 for exactly this
            # reason (see the OOM incident in the seed-landscape sweep's README).
            gpu_free_enough(wait_seconds=600, threshold_mib=2000)

        adapter = adapter_cls(
            task=task,
            train_cutters=train_cutters(task),
            test_cutter=test_cutter(task),
            seed=seed,
            preprocess_seed=args.preprocess_seed,
            output_dir=out_dir,
            device=args.device,
            config={"save_checkpoint": args.save_checkpoint},
        )
        result = adapter.run(resume=(args.resume and not args.force), smoke_test=args.smoke_test)
        status = result["status"]
        if status == "done":
            n_done += 1
        elif status == "skipped":
            n_skipped += 1
        else:
            n_failed += 1
        print(f"  [{i}/{len(cells)}] {args.method}/{task}/{seed_dir_name(seed)}: {status}"
              + (f" ({result.get('error', result.get('reason', ''))})" if status != 'done' else ""))

    elapsed = time.time() - t_start
    print(f"[run_phm2010] done={n_done} skipped={n_skipped} failed={n_failed} "
          f"elapsed={elapsed:.1f}s")
    return 1 if n_failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())

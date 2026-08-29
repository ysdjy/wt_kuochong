# -*- coding: utf-8 -*-
"""
Single resumable entry point for all D2/D3 transfer-task training.

Usage (the ONE command the user needs, from the repo root, `dcpsr` env active):
    python final_statistical_evidence/scripts/run_transfer_tasks.py --resume

Behavior:
    1. Scans every (task, method) cell for a DONE.flag; skips it if present.
    2. Runs remaining cells in a fixed fast->slow method order, D2 before D3.
    3. Checks GPU (nvidia-smi) before each GPU job; if VRAM is heavily used
       by another process, waits and rechecks rather than launching blind.
    4. Updates STATUS.json after every cell.
    5. On completion of all cells, runs aggregate_transfer_results.py
       automatically.
    6. Safe to Ctrl-C and rerun with --resume at any time -- already-DONE
       cells are skipped; in-progress cells restart from scratch for that
       one method (no cross-epoch checkpoint resume inside a single
       training run in this stage -- jobs are short enough, minutes not
       hours, that restart-from-scratch-per-cell is the acceptable
       resume granularity here).

Also supports targeted reruns:
    python final_statistical_evidence/scripts/run_transfer_tasks.py --task D2 --method mtf_avitk --resume
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
METHODS_DIR = SCRIPTS_DIR / "methods"
FSE_ROOT = SCRIPTS_DIR.parent
TRANSFER_ROOT = FSE_ROOT / "transfer_tasks"
LOGS_DIR = FSE_ROOT / "logs"
STATUS_PATH = FSE_ROOT / "STATUS.json"

PYTHON_DCPSR = r"C:\Users\banghai\miniconda3\envs\dcpsr\python.exe"
PYTHON_PUB = r"C:\Users\banghai\miniconda3\envs\pub_baselines\python.exe"

TASKS = ["D2", "D3"]

# (method_id, launcher_script, python_env, extra_args_fn, uses_gpu)
# extra_args_fn(task) -> list[str] appended after --task <task> --method <method_id>
INTERNAL_METHODS = ["rf", "tcn_gru", "multitask_tcn_gru", "htt_net"]  # dc_psr rides along with multitask_tcn_gru

PUBLISHED_METHODS = [
    # method_id, script relative to methods/, python env
    ("multi_source_attention", "run_multi_source_attention_transfer_task.py", PYTHON_PUB),
    ("dp2net_adapted", "run_dp2net_transfer_task.py", PYTHON_PUB),
    ("dynamic_gin_tgp", "run_dynamic_gin_tgp_transfer_task.py", PYTHON_PUB),
    ("mtf_avitk", "run_mtf_avitk_transfer_task.py", PYTHON_PUB),
]

RUN_ORDER = ["rf", "tcn_gru", "multitask_tcn_gru", "htt_net",
             "multi_source_attention", "dp2net_adapted", "dynamic_gin_tgp"]
# mtf_avitk is intentionally excluded from the default --resume sweep --
# largest/slowest model, left for manual execution per user preference
# (see MTF_AVITK_MANUAL_TUTORIAL.md). Still runnable explicitly via
# `--task D2 --method mtf_avitk --resume`.

ALL_METHOD_IDS = ["rf", "tcn_gru", "multitask_tcn_gru", "dc_psr", "htt_net",
                   "multi_source_attention", "mtf_avitk", "dynamic_gin_tgp", "dp2net_adapted"]


def done_flag(task, method):
    return TRANSFER_ROOT / task / method / "DONE.flag"


def load_status():
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    return {"tasks": {t: {m: "PENDING" for m in ALL_METHOD_IDS} for t in TASKS}, "last_update": None}


def save_status(status):
    status["last_update"] = time.time()
    STATUS_PATH.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")


def refresh_status(status):
    for t in TASKS:
        for m in ALL_METHOD_IDS:
            status["tasks"][t][m] = "DONE" if done_flag(t, m).exists() else status["tasks"][t].get(m, "PENDING")
    save_status(status)
    return status


def gpu_free_enough(threshold_mib=2000, wait_s=30, max_wait_s=600) -> bool:
    waited = 0
    while waited < max_wait_s:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                timeout=15,
            ).decode().strip()
            used = int(out.splitlines()[0])
        except Exception as e:
            print(f"[gpu_check] nvidia-smi unavailable ({e}); proceeding without gating")
            return True
        if used < threshold_mib:
            return True
        print(f"[gpu_check] GPU busy ({used} MiB used) -- waiting {wait_s}s (elapsed {waited}s)")
        time.sleep(wait_s)
        waited += wait_s
    print(f"[gpu_check] GPU still busy after {max_wait_s}s -- proceeding anyway (safe exit not possible mid-runner; log and continue)")
    return False


def run_cell(task, method_id, force=False):
    status = refresh_status(load_status())
    if not force and done_flag(task, method_id).exists():
        print(f"[runner] {task}/{method_id}: DONE.flag present, skip")
        status["tasks"][task][method_id] = "DONE"
        save_status(status)
        return True

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{task}_{method_id}.log"
    status["tasks"][task][method_id] = "RUNNING"
    save_status(status)

    if method_id in ("rf",):
        cmd = [PYTHON_DCPSR, str(METHODS_DIR / "run_internal_methods_transfer_task.py"),
               "--task", task, "--method", method_id]
        gpu_free_enough(wait_s=5, max_wait_s=5)  # RF is CPU-only, no real wait needed
    elif method_id in ("tcn_gru", "multitask_tcn_gru", "dc_psr", "htt_net"):
        real_method = "multitask_tcn_gru" if method_id == "dc_psr" else method_id
        cmd = [PYTHON_DCPSR, str(METHODS_DIR / "run_internal_methods_transfer_task.py"),
               "--task", task, "--method", real_method]
        gpu_free_enough()
    else:
        script_map = {m: (s, env) for m, s, env in PUBLISHED_METHODS}
        if method_id not in script_map:
            print(f"[runner] {task}/{method_id}: no launcher registered yet -- SKIP (see METHOD_REGISTRY.yaml notes)")
            status["tasks"][task][method_id] = "NOT_IMPLEMENTED"
            save_status(status)
            return False
        script, env_py = script_map[method_id]
        script_path = METHODS_DIR / script
        if not script_path.exists():
            print(f"[runner] {task}/{method_id}: launcher {script} not found yet -- SKIP")
            status["tasks"][task][method_id] = "NOT_IMPLEMENTED"
            save_status(status)
            return False
        cmd = [env_py, str(script_path), "--task", task, "--method", method_id]
        gpu_free_enough()

    print(f"[runner] launching {task}/{method_id}: {' '.join(cmd)}")
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\n==== {time.ctime()} launching {cmd} ====\n")
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    dt = time.time() - t0

    status = refresh_status(load_status())
    if proc.returncode == 0 and done_flag(task, method_id).exists():
        print(f"[runner] {task}/{method_id}: OK ({dt:.0f}s)")
        status["tasks"][task][method_id] = "DONE"
        ok = True
    else:
        print(f"[runner] {task}/{method_id}: FAILED (exit={proc.returncode}, {dt:.0f}s) -- see {log_path}")
        status["tasks"][task][method_id] = "FAILED"
        ok = False
    save_status(status)
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="skip DONE cells, run everything else in order")
    parser.add_argument("--task", choices=TASKS, default=None, help="restrict to one task")
    parser.add_argument("--method", choices=ALL_METHOD_IDS, default=None, help="restrict to one method")
    parser.add_argument("--force", action="store_true", help="rerun even if DONE.flag present")
    args = parser.parse_args()

    tasks = [args.task] if args.task else TASKS
    methods = [args.method] if args.method else RUN_ORDER

    status = refresh_status(load_status())
    print("=" * 70)
    print("D2/D3 TRANSFER-TASK RUNNER")
    print(f"tasks={tasks} methods={methods} resume={args.resume} force={args.force}")
    print("=" * 70)

    results = {}
    for method_id in methods:
        for task in tasks:
            if method_id == "dc_psr":
                continue  # rides along with multitask_tcn_gru's cell
            ok = run_cell(task, method_id, force=args.force)
            results[(task, method_id)] = ok
            if method_id == "multitask_tcn_gru":
                results[(task, "dc_psr")] = done_flag(task, "dc_psr").exists()

    print("\n" + "=" * 70)
    print("RUN SUMMARY")
    for (task, method_id), ok in results.items():
        print(f"  {task}/{method_id}: {'OK' if ok else 'FAILED/SKIPPED'}")

    status = refresh_status(load_status())
    all_done = all(status["tasks"][t][m] == "DONE" for t in TASKS for m in ALL_METHOD_IDS)
    if all_done:
        print("\n[runner] all cells DONE -- running aggregate_transfer_results.py")
        subprocess.run([PYTHON_DCPSR, str(SCRIPTS_DIR / "aggregate_transfer_results.py")])
    else:
        pending = [(t, m) for t in TASKS for m in ALL_METHOD_IDS if status["tasks"][t][m] != "DONE"]
        print(f"\n[runner] {len(pending)} cells not yet DONE: {pending}")
        print("[runner] rerun with --resume once launchers for remaining published methods exist.")


if __name__ == "__main__":
    main()

# B9 Overnight Run — Status (FINAL for this session)

Start: 2026-08-30 ~02:00. Completed: 2026-08-30 ~06:30.
GPU: RTX 3070 Ti Laptop, 8GB, shared with another concurrent Claude Code session (different scratchpad session id) building a separate English-named `methods/`/`shared/`/`run_phm2010.py` framework in the same repo. This session worked only in `05_统计检验/seed_statistics/`.

## Final tally

```
PHM2010: D1=101/101  D2=101/101  D3=101/101   -> 303/303
NASA:    N1=101/101  N2=101/101  N3=101/101  N4=101/101  -> 404/404
TOTAL B9 runs: 707/707 DONE

Paired B3 (raw pre-B9-inference backbone output, same checkpoint/seed):
PHM2010: D1=101/101  D2=101/101  D3=101/101   -> 303/303
NASA:    N1=101/101  N2=101/101  N3=101/101  N4=101/101  -> 404/404
TOTAL B3 runs: 707/707 DONE (369 native during the batch runs + 338 backfilled
from saved checkpoints via forward-pass-only reconstruction, zero retraining)
```

MTW-CM: audited only (`MTW_TASK_AUDIT.md`), **not started at scale**. 0/303.

## What happened, in order

1. Verified today's earlier PHM2010 D1 5-seed work was safe (separate git repo `wt_kuochong`, different session had already committed an initial snapshot).
2. Detected and coordinated around a second concurrent Claude Code session sharing this GPU/repo (per user instruction: stayed in the Chinese-named `05_统计检验/` tree only).
3. Closed PHM2010 D1 to a full 101/101 (added seed100 to the pre-existing 0-99 diagnostic run).
4. Built fresh, leakage-checked, source-only frozen preprocessing for PHM D2 (train=C1+C6,test=C4) and D3 (train=C4+C6,test=C1), reusing the already-validated `final_statistical_evidence/scripts/methods/common_pipeline.py::split_by_conditions` reference split logic. Verified n_test=304 for both from actual data (not hardcoded), matching D1.
5. Audited NASA Milling: found the user-given N1-N4 case splits **do not match** the old code's own `FIXED_TASKS`, and found (and explicitly excluded) old "case-split-optimization" scripts that chose splits by target performance. Used the user's version (documented in `NASA_TASK_AUDIT.md`). Built frozen preprocessing for all 4 tasks reusing `代码/9.1nasa数据实验.py`'s feature/label/GMM functions (only the case-split logic was replaced).
6. Audited MTW-CM: located data (6419 `.h5` files), confirmed D1-M/D2-M/D3-M machine mapping from an existing validated protocol file (`experiments_mendeley/02_protocols/task_definitions.json`), identified a reusable, well-engineered pipeline (`experiments_mendeley/code/dcpsr/` + `01_run_experiments.py`). Did not get to launching it at scale this session.
7. Launched PHM D2+D3 (198 seeds) and NASA N1-N4 (402 seeds) as two parallel 2-way-concurrency batches (4 GPU processes total).
8. Mid-run, user asked for paired B3 (Multi-task TCN-GRU raw pre-inference output) to be saved alongside every B9 run, using the same checkpoint, no separate training. Patched both live runners; verified hash-paired correctness on both PHM and NASA before letting the batches continue (this auto-applied to every seed dispatched after the patch, since each seed is a fresh process).
9. Both batch **dispatcher loops got killed simultaneously** by an external event with no accompanying user message (possibly the client's "stop" control) — the already-running training processes were unaffected and kept completing normally; no new seeds were dispatched afterward. Asked the user to confirm; they said it wasn't intentional and asked for a restart.
10. While preparing to restart, found the batches had actually gotten within 3 seeds of full completion during the outage window. Manually retried and fixed the last 3 stragglers (PHM D2 seed31, seed65; NASA N2 seed88) instead of relaunching — cheaper and safer than a fresh dispatcher run.
11. Backfilled B3 for the seeds that completed *before* the B3-pairing patch (PHM D2: 32, PHM D3: 2, NASA N1: 41 — all via forward-pass-only reconstruction from already-saved checkpoints, zero retraining, verified `backbone_checkpoint_hash` match against the paired B9 run).

## Known execution-class issues (not protocol failures)

Running two 2-way batches simultaneously produced occasional CUDA OOM (~5 isolated incidents across ~700 seed-runs, all captured cleanly via `FAILED.flag`/`error.log`, all manually retried and fixed). Root cause: shared 8GB GPU with a second, independently-scheduled Claude session; peak-batch-moment overlap pushed usage to 7-7.5GB periodically. No protocol failures (no leakage, no hash drift, no task-definition ambiguity) occurred at any point.

## Next steps (for a follow-up session)

- MTW-CM: build the frozen-preprocessing + runner following the same pattern as NASA (reuse `experiments_mendeley/code/dcpsr/` pipeline; either drive it via its own `01_run_experiments.py --out-root <local> --phase dual --tasks D1-M,D2-M,D3-M --seeds 0..100 --resume`, or port to the same `run_*_seed_task.py` pattern used for PHM/NASA for consistency and native B3 pairing). 303 runs (3 tasks × 101 seeds).
- Aggregation: no `summary/` dirs (seed_level_results.csv, mean±std tables, dataset-level rollups) have been built yet for D2/D3/NASA/the new B3 pairs. All raw per-seed data exists and is complete; this is a pure aggregation pass, same pattern as `analyze_seed_landscape.py`/`rank_top5_seeds.py` already used for PHM D1.
- Git: nothing has been committed/pushed this session (per "no push" instruction from earlier in the day, not explicitly lifted for this overnight task's git-checkpoint asks). Confirm with the user before committing/pushing 707×2 result directories.

## Fatal protocol issues

None.

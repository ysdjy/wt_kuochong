# Results Policy

Authoritative schema for every formal PHM2010 run under `results/PHM2010/`. Any
adapter that does not conform to this document has a bug — fix the adapter, do
not special-case the aggregator around it.

## Directory layout

```
results/
└─ PHM2010/
   └─ B9_DC_PHSR/
      ├─ D1/
      │  ├─ seed000/
      │  ├─ seed001/
      │  ├─ ...
      │  └─ seed100/
      ├─ D2/
      ├─ D3/
      └─ summary/
```

- Seed directories are always zero-padded to 3 digits: `seed000` .. `seed100`
  (101 seeds, 0–100 inclusive).
- `results/` is `.gitignore`d by default except for `summary/` CSVs and small
  aggregate artifacts — see `.gitignore` and section "What gets committed" below.

## Per-run required files

Every `results/PHM2010/Bx_xxx/Dy/seedNNN/` directory, once complete, contains:

| File | Contents |
|---|---|
| `metrics.json` | Full formal metric set (see `shared/metrics/metrics.py`), JSON dict |
| `metrics.csv` | Same metrics, one-row CSV (for easy `pd.concat` aggregation) |
| `predictions.csv` | `run_id,true_stage,pred_stage,p_early,p_middle,p_late[,q_true,q_pred]` |
| `training_log.csv` | Per-epoch (or per-iteration, for non-epoch methods like RF) training curve |
| `run_meta.json` | Full provenance record — see `shared/utils/run_meta.py::build_run_meta()` |
| `config_resolved.yaml` | The fully-resolved config actually used (after CLI overrides) |
| `confusion_matrix.csv` | 3x3 confusion matrix, rows=true, cols=pred, labels=[early,middle,late] |
| `DONE.flag` | Empty marker file. Presence = the run completed successfully. **Written LAST, after every other file above is written**, and only via atomic rename (write to `DONE.flag.tmp` then `os.replace`) — never create it before the run is actually finished, and never leave a `DONE.flag` behind for a run that OOM'd, crashed, or was killed. |

On failure instead of `DONE.flag`:

| File | Contents |
|---|---|
| `FAILED.flag` | Empty marker file |
| `error.log` | Full traceback / stderr |

A directory that exists but has neither `DONE.flag` nor `FAILED.flag` is
**incomplete, not failed** — `--resume` treats it as not-yet-attempted and retries
it from scratch (see MANUAL_RUN.md's resume semantics).

GPU out-of-memory is never recorded as a "bad seed" result — an OOM must produce
`FAILED.flag` + `error.log`, never `DONE.flag` with degraded/fallback metrics.

## `run_meta.json` required keys

```
method, dataset, task, train_cutters, test_cutter, seed, preprocess_seed,
git_commit, machine_alias, hostname, os, python_version, torch_version,
cuda_version, gpu_name, start_time, end_time, runtime_sec,
feature_hash, split_hash, label_hash, evaluation_universe_hash, config_hash
```

Built via `shared/utils/run_meta.py::build_run_meta()` — do not hand-roll this
dict in an adapter.

## Seed conventions

- `PREPROCESS_SEED` is fixed per (dataset, task) and never varies with the CLI
  `--seed-start/--seed-end` range — feature selection, GMM fitting, scaler
  fitting, and any train/val split are frozen once and reused across every
  TRAIN_SEED.
- `TRAIN_SEED` (the CLI seed) controls only true training/optimization
  randomness. Every adapter calls `shared/utils/seeding.py::seed_everything(seed)`
  immediately before instantiating its model — never earlier in a shared driver
  process. See that module's docstring for the concrete historical bug this
  prevents.

## Dataset-level aggregation (`summary/dataset_mean_std.csv`)

For each completed seed N with metrics available on D1, D2, and D3:

```
metric_PHM_seedN = mean(metric_D1_seedN, metric_D2_seedN, metric_D3_seedN)
```

computed independently per metric, per seed (equal-weight average across the
three tasks). Then across every predefined seed that completed all three tasks:

```
mean = numpy.mean(...)
std  = numpy.std(..., ddof=1)   # sample std, uniform convention across the repo
```

This dataset-level mean±std is the basis for the paper's PHM2010 headline
`mean ± std` numbers. D1/D2/D3 per-task mean±std are also kept
(`summary/D1_mean_std.csv` etc.) — the dataset-level number never replaces them.

## `summary/` required files (per method)

```
all_tasks_seed_level.csv   # one row per (task, seed) that has DONE.flag
D1_mean_std.csv
D2_mean_std.csv
D3_mean_std.csv
dataset_seed_level.csv     # one row per seed with all 3 tasks complete
dataset_mean_std.csv
run_status.csv             # every (task, seed) combination x {DONE, FAILED, MISSING}
failed_runs.csv            # subset of run_status.csv with status=FAILED
```

## What gets committed to git

Full per-seed `predictions.csv` / `training_log.csv` / logs are useful locally
but not committed en masse — `9 methods x 3 tasks x 101 seeds = 2727` runs would
bloat the repo. Checkpoints (`.pth`/`.pt`/`.pkl`) are NEVER committed by default
(`--save-checkpoint none`, the default) — see `run_phm2010.py --help`.
`summary/*.csv` (small, aggregated, no large binaries) ARE committed — they are
the audit trail the manuscript's tables are built from. See root `.gitignore`.

## Evaluation universe

All formal comparison-table numbers use the common evaluation universe
`run_id 12-315 inclusive (n=304)`, verified identical across D1/D2/D3 and across
window-based vs. raw-signal method families — see
`shared/phm2010/evaluation_universe.py` for the full derivation and the
`assert_common_universe()` guard every adapter's `evaluate()` step must call.

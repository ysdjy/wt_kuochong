# B1: Random Forest (RF)

## Method

A standard `sklearn.ensemble.RandomForestClassifier` trained on the run-level,
train-only-fit-and-selected feature table (not raw signal, not windowed
sequences for training) — the simplest of the 9 comparison methods, used as a
traditional-ML baseline. Not from a published paper; this project's own
internal baseline.

- **Source paper**: none (internal baseline).
- **Old code source**: `代码/main_experiment_3_fgds_psi_optimized.py` (feature
  pipeline functions) + `代码/7.7跨工况实验.py` (train/test condition split
  logic, generalized via `final_statistical_evidence/scripts/methods/common_pipeline.py`)
  + `final_statistical_evidence/scripts/methods/run_internal_methods_transfer_task.py::run_rf`
  (the historical D1/D2/D3-capable entry point this adapter reproduces).
  Legacy commit `811da096ee47bea4f65db193aa49e793dba6f47d`
  (branch `diagnostic/fixed-preprocess-5seed`). See `source_manifest.json` for
  exact file hashes and the full adaptation-notes list.
- **New repo entry point**: `adapter.py::RFAdapter` (module-level
  `ADAPTER_CLASS = RFAdapter`), dispatched by `run_phm2010.py --method B1`.

## PHM2010 input form

- **Run-level feature table**, NOT raw signal, NOT an image/graph
  representation. Reads `data/PHM2010/features/run_level_features_all.csv`
  (see `扩充实验代码/data/README.md` for provenance).
- Uses the L=12-window-feasible subset of test run_ids for evaluation (same
  common evaluation universe as the other window-based methods:
  `shared/phm2010/evaluation_universe.py`), even though RF's own training
  doesn't need windowing — this exactly matches the historical `run_rf()`
  behavior (RF trains on ALL feasible train rows unwindowed, but is *scored*
  only on the windowed-feasible test run_ids so its test universe is directly
  comparable to the NN-based window methods).

## Preprocessing

Train-cutter-only-fit pipeline (`code/preprocessing.py::prepare_task_data`):
load feature table → per-condition condition-relative stage labeling (VB-based
quantile thresholds) → stage-stratified centered-slice train/val carve (test
cutter held out entirely, never touched) → per-condition online (causal,
expanding-window) feature engineering → train-only median fill → train-only
mutual-information + Spearman + cross-condition-instability feature selection
(45 features, redundancy-pruned) → train-only 5-component GMM fine-state
assignment → train-only `StandardScaler` fit, applied to all splits.
`PREPROCESS_SEED` (fixed per task, not the CLI `--seed`) drives the MI
selection and GMM fitting.

## Model / hyperparameters

`RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
random_state=<CLI --seed>, n_jobs=-1)` — unchanged from the original.

## Task definition

Uses `shared/phm2010/tasks.py`'s authoritative D1/D2/D3 registry — no
independent split logic. See `RESULTS_POLICY.md` for the full per-run output
schema.

## Output files

Per `RESULTS_POLICY.md`: `metrics.json`, `metrics.csv`, `predictions.csv`,
`training_log.csv`, `run_meta.json`, `config_resolved.yaml`,
`confusion_matrix.csv`, `DONE.flag`. RF's `training_log.csv` has a single row
(RF fits in one call, not epochs) recording `train_seconds`/`n_train_rows`.

## Reproduction / adaptation notes

Only task/seed/output routing was changed (train_conditions/test_condition/
preprocess_seed became explicit parameters instead of hardcoded constants);
the RF model itself, the feature-selection formula, and the GMM fine-state
logic are byte-for-byte the same algorithm as the original. See
`source_manifest.json.adaptation_notes` for the full list.

## Smoke test

`tests/smoke_test.py` runs a CPU-only, drastically abbreviated check (a couple
of `n_estimators`, not 400) against real D1 data to verify the full
prepare→build→train→predict plumbing, WITHOUT claiming this is a real/paper-
faithful result. See its own output for the actual run log.

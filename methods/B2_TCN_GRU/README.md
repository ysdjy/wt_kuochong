# B2 — TCN-GRU

Generic deep-learning baseline: a Temporal Convolutional Network (TCN) feature
extractor feeding a GRU, with a single stage-classification head (no auxiliary
fine-state or degradation-index heads — that's B3/B9).

- **Old code source**: `代码/main_experiment_3_fgds_psi_optimized.py` (shared
  data pipeline + `TemporalBlock`) and
  `final_statistical_evidence/scripts/methods/run_internal_methods_transfer_task.py`
  (`TCNGRUStageOnly`, `train_stage_model`, `predict_stage_model`) —
  `legacy_git_commit` `811da096ee47bea4f65db193aa49e793dba6f47d`, branch
  `diagnostic/fixed-preprocess-5seed`. See `source_manifest.json` for hashes.
- **New repo entry point**: `adapter.py::B2TCNGRUAdapter` (dispatched by
  `run_phm2010.py --method B2`); model code in `code/model.py`; shared data
  pipeline in `../_internal_shared/code/pipeline.py`.
- **PHM2010 input form**: run-level engineered feature table (NOT raw signal),
  `L=12` sliding window over consecutive runs within a condition.
- **Run-level feature or raw signal**: run-level feature
  (`data/PHM2010/features/run_level_features_all.csv`).
- **Preprocessing**: per-task train-cutter-only feature selection (mutual info +
  Spearman + redundancy filter, 45 features), StandardScaler fit on train
  cutters only, `PREPROCESS_SEED=42` fixed across all `TRAIN_SEED`s for a given
  task. See `../_internal_shared/code/pipeline.py::prepare_task_data`.
- **Model architecture**: TCN (channels `[32, 64, 64]`, dilations `1,2,4`) → GRU
  (hidden `64`) → shared FC(64) → stage-classification head (3-way). See
  `config.yaml`.
- **Training hyperparameters**: AdamW, `lr=5e-4`, `weight_decay=1e-5`,
  `batch_size=32`, up to 120 epochs, patience 18, `ReduceLROnPlateau`. See
  `config.yaml`.
- **Task definition**: D1/D2/D3 per `shared/phm2010/tasks.py` — no independent
  split logic, this method has none of its own.
- **Output files**: per `RESULTS_POLICY.md` — `predictions.csv` columns
  `run_id, true_stage, pred_stage, p_early, p_middle, p_late` (no `q_true`/
  `q_pred`, this method has no degradation-index head).
- **Reproduction/adaptation notes**: only task/seed/output routing and paths
  changed vs. original — see `source_manifest.json`. Verified 2026-08-30 via a
  CPU-only smoke test (untrained-weight forward pass on real D1 data,
  304-row/6-column prediction table with the expected schema).

## Known limitations

- Not yet trained to convergence in this repo (only smoke-tested). No frozen
  D1 reference number exists specifically for B2 in isolation to cross-check
  against — the historical 5-seed sweep's TCN-GRU numbers
  (`final_five_seed_sweep/`) used the OLD shared-RNG-stream runner and are
  documented as contaminated for this exact method — see
  `shared/reproducibility/PHM2010_D1_frozen_preprocess/TCN_GRU_SEED42_DIVERGENCE_NOTE.txt`.
  A real training run through `run_phm2010.py` is the first trustworthy source
  of this method's numbers in the new repo.

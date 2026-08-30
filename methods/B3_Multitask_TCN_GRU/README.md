# B3 — Multi-task TCN-GRU

Direct backbone of the proposed method (B9, DC-PHSR). Same TCN+GRU trunk as B2
but with three heads: stage classification, fine-state (5-way sub-stage)
classification, and a continuous degradation-index (`q`) regression head,
trained jointly.

- **Old code source**: `代码/main_experiment_3_fgds_psi_optimized.py`
  (`TCNGRUMultiTask`, `train_model`, `predict_model`, `run_epoch`) —
  `legacy_git_commit` `811da096ee47bea4f65db193aa49e793dba6f47d`, branch
  `diagnostic/fixed-preprocess-5seed`. See `source_manifest.json` for hashes.
- **New repo entry point**: `adapter.py::B3MultitaskTCNGRUAdapter` (dispatched
  by `run_phm2010.py --method B3`); shared data/model pipeline in
  `../_internal_shared/code/pipeline.py`.
- **PHM2010 input form**: run-level engineered feature table (NOT raw signal),
  `L=12` sliding window.
- **Preprocessing**: identical to B2 — see B2's README and
  `../_internal_shared/code/pipeline.py::prepare_task_data`. `PREPROCESS_SEED=42`
  fixed.
- **Model architecture**: TCN `[32,64,64]` → GRU(64) → shared FC(64) → three
  heads: stage (3-way), fine-state (5-way), `q` (sigmoid, scalar). See
  `config.yaml`.
- **Training hyperparameters**: AdamW, `lr=5e-4`, multi-task loss
  `1.00*stage_CE + 0.25*fine_CE + 0.30*q_smoothL1 + 0.03*monotonic_penalty`,
  up to 120 epochs, patience 18. See `config.yaml`.
- **Task definition**: D1/D2/D3 per `shared/phm2010/tasks.py`.
- **Output files**: per `RESULTS_POLICY.md` — `predictions.csv` columns
  `run_id, true_stage, pred_stage, p_early, p_middle, p_late, q_true, q_pred`
  (raw softmax output, i.e. B3's own head — NOT B9's post-processed
  probabilities).
- **Reproduction/adaptation notes**: only task/seed/output routing and paths
  changed vs. original — see `source_manifest.json`. **Verified 2026-08-30**:
  `pipeline.prepare_task_data(["C1","C4"], "C6")`'s 45 selected features
  reproduce `shared/reproducibility/PHM2010_D1_frozen_preprocess/
  selected_features_seed42.json` bit-exact, in order — strong evidence this
  vendored port is a faithful copy of the original preprocessing. CPU-only
  smoke test (untrained-weight forward pass on real D1 data) produces a
  304-row/8-column prediction table with the expected schema.

## Relationship to B9

B9 (DC-PHSR) applies a deterministic post-hoc probability-inference step on top
of this exact architecture's output. This round, B9 trains its **own**
independent copy of this backbone rather than reusing a B3 checkpoint — see
`../B9_DC_PHSR/README.md`'s "Known scope limitation" section.

## Known limitations

Same as B2 — not yet trained to convergence in this repo (smoke-tested only).
The pre-existing `01_主对比实验/PHM2010/B3_Multitask_TCN_GRU/results/` (5 seeds,
D1 only) remains the authoritative historical reference until this repo's
`run_phm2010.py` produces a fresh trained run to compare against.

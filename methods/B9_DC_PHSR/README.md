# B9 — DC-PHSR

**Proposed method.** Paper name: **DC-PHSR**. Legacy code identifiers used
throughout the old project: `DC-PSR`, `B12`, `FGDS-PSI` — all refer to this
exact method; this repo does not rename underlying code/variables, only
cross-references the paper name here and in docs.

DC-PHSR = the B3 (Multi-task TCN-GRU) backbone's raw output, followed by a
deterministic, frozen probability-inference step (temperature scaling +
degradation-index prior + fine-state fusion + causal ordered filtering).

- **Source paper**: this manuscript's own proposed method (no external paper).
- **Old code source**: `代码/main_experiment_3_fgds_psi_optimized.py`
  (`apply_probability_inference`, `qhat_prior`, `fine_to_stage_prob`,
  `temperature_scale`, `causal_ordered_filter`), frozen `B12_PARAMS` from
  `final_statistical_evidence/scripts/methods/run_internal_methods_transfer_task.py`
  — `legacy_git_commit` `811da096ee47bea4f65db193aa49e793dba6f47d`, branch
  `diagnostic/fixed-preprocess-5seed`. See `source_manifest.json` for hashes.
- **New repo entry point**: `adapter.py::B9DCPHSRAdapter` (dispatched by
  `run_phm2010.py --method B9`); shared data/model pipeline in
  `../_internal_shared/code/pipeline.py`.
- **PHM2010 input form**: run-level engineered feature table (NOT raw signal),
  `L=12` sliding window — identical to B3.
- **Preprocessing**: identical to B3 — `PREPROCESS_SEED=42` fixed.
- **Model architecture**: B3's exact backbone (TCN `[32,64,64]` → GRU(64) →
  shared FC(64) → stage/fine-state/`q` heads) + a deterministic, non-trained
  probability-inference step on top. See `config.yaml`.
- **Training hyperparameters**: identical to B3 for the backbone (see
  `../B3_Multitask_TCN_GRU/config.yaml`). The inference step has no
  training — its 7 parameters (`eta, fine_weight, temperature, mid_floor,
  late_tau, early_tau, order_blend`) are frozen constants (`B12_PARAMS`),
  never retuned in this repo.
- **Task definition**: D1/D2/D3 per `shared/phm2010/tasks.py`.
- **Output files**: per `RESULTS_POLICY.md` — `predictions.csv` columns
  `run_id, true_stage, pred_stage, p_early, p_middle, p_late, q_true, q_pred`,
  where `pred_stage`/`p_*` are the FINAL post-inference probabilities
  (`stage_pred_final`/`final_prob_*` in the underlying pipeline), not the raw
  backbone output.
- **Reproduction/adaptation notes**: see `source_manifest.json`. Verified
  2026-08-30: CPU-only smoke test (untrained backbone + real
  `apply_probability_inference` call on real D1 data) produces a
  304-row/8-column prediction table whose probability columns sum to 1.0 to
  within `1e-6` for every row.

## Known scope limitation, by deliberate choice (2026-08-30)

EXPERIMENT_REGISTRY.md section 34 says B9 may reuse B3's checkpoint only when
`seed`/`task`/`config_hash`/`preprocess_hash` all match exactly, and explicitly
warns against the failure mode "B9 seed17 错用 B3 seed42 checkpoint" (using the
wrong seed's checkpoint). Building that cross-adapter lookup + hash-verification
plumbing safely is real engineering work; doing it under time pressure risks
exactly that bug. **For this round, B9's adapter trains its own independent
copy of the identical backbone inside its own `(B9, task, seed)` cell**, rather
than reusing a sibling `B3_Multitask_TCN_GRU` run's checkpoint. This means:

- B9 and B3 will NOT produce bit-identical raw-backbone predictions between
  separate runs at the same seed (each trains independently — legitimate
  training-seed-sensitive divergence, not a bug, though it should be small
  since both use the exact same architecture/data/seed).
- Running the full `B1..B9 × D1..D3 × seed0..100` sweep this way roughly
  doubles B9's compute cost vs. an ideal checkpoint-shared version, since it
  duplicates B3's backbone training.
- **Future optimization** (not implemented here): have `run_phm2010.py`'s
  orchestrator train B3 once per `(task, seed)` and hand both B3's and B9's
  adapters the same checkpoint path with a verified hash match, or add a
  `--reuse-b3-checkpoint` flag to `B9DCPHSRAdapter` that checks
  `config_hash`/`preprocess_hash` before loading a sibling checkpoint instead
  of retraining.

## Known limitations

Not yet trained to convergence in this repo (smoke-tested only). The
pre-existing `01_主对比实验/PHM2010/B9_DC_PHSR/results/` (5 seeds, D1 only,
built by the now-superseded non-self-contained `code/run_seed_d1.py`) remains
the authoritative historical reference until this repo's `run_phm2010.py`
produces a fresh trained run to compare against.

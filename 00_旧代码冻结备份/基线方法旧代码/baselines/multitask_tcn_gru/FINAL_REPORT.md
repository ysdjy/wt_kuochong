# Multi-task TCN-GRU — Final 5-Seed Report

Internal ID: B11. Category: generic/internal baseline. DC-PSR's direct predecessor.

## Source

Not a standalone script. One row (`Method == "B11"`) of the shared unified
comparison script:

- `代码/7.4对比实验.py` (`main()`, `base.train_model(...)` — TCN-GRU + auxiliary heads, raw probability output)
- `代码/main_experiment_3_fgds_psi_optimized.py` (architecture, training loop, splits)
- Authoritative feature file: `baselines/htt_net/data/run_level_features_all.csv`
- Trained checkpoint (seed 42): `补充材料/小论文/4_comparison_experiment_recheck/3_models/B11_B12_multitask_tcn_gru.pth`
  (shared with DC-PSR/B12 — B12 reuses this exact network's raw probabilities,
  then applies a separate frozen deterministic post-processing step; only one
  checkpoint file exists for both).

Do not retrain separately from B5/B10/B12 — they share one run of the same script.

## Protocol

C1+C4 (train) -> C6 (test), D1 task. Windowed universe: 304 runs (`L=12`).

## 5-seed result (seeds 42/52/62/72/82, ddof=1)

| Metric | Mean ± Std |
|---|---|
| Acc | 0.8882 ± 0.0901 |
| Macro-F1 | 0.8904 ± 0.0896 |
| E-F1 | 0.9416 ± 0.0579 |
| M-F1 | 0.8345 ± 0.1418 |
| L-F1 | 0.8952 ± 0.0919 |

Full per-metric mean±std: `../../final_five_seed_sweep/results/FINAL_9_METHODS_5SEED.csv` (row `Method == "Multi-task TCN-GRU"`).
Per-seed raw values: `results/seed_level_results.csv`.

## Params

84,009 (state_dict tensor count, counted directly from `B11_B12_multitask_tcn_gru.pth`, architecture fixed across seeds).

## Seed source files

- seed 42: `补充材料/小论文/4_comparison_experiment_recheck/1_results/FINAL_comparison_results.csv`
- seeds 52/62/72/82: `final_five_seed_sweep/results/generic_baselines/seed<N>/1_results/FINAL_comparison_results.csv`

Same frozen architecture/hyperparameters across all 5 seeds; only
`RANDOM_SEED` varies. C6 was never used for tuning. Notably higher
cross-seed variance than the other generic baselines (Macro-F1 std ≈ 0.090) —
recorded as-is, not treated as an error (no crash/NaN/config drift found in
any seed).

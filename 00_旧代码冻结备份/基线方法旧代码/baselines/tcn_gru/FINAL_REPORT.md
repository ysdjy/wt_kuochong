# TCN-GRU (Relative-stage TCN-GRU) — Final 5-Seed Report

Internal ID: B10. Category: generic/internal baseline.

## Source

Not a standalone script. One row (`Method == "B10"`) of the shared unified
comparison script:

- `代码/7.4对比实验.py` (`main()`, `train_stage_model(TCNGRUStageOnly(...))`)
- `代码/main_experiment_3_fgds_psi_optimized.py` (architecture, training loop, splits)
- Authoritative feature file: `baselines/htt_net/data/run_level_features_all.csv`
- Trained checkpoint (seed 42): `补充材料/小论文/4_comparison_experiment_recheck/3_models/B10_TCN_GRU.pth`

Do not retrain TCN-GRU separately from B5/B11/B12 — they share one run of the
same script.

## Protocol

C1+C4 (train) -> C6 (test), D1 task. Windowed universe: 304 runs (`L=12`).

## 5-seed result (seeds 42/52/62/72/82, ddof=1)

| Metric | Mean ± Std |
|---|---|
| Acc | 0.8335 ± 0.0475 |
| Macro-F1 | 0.8364 ± 0.0509 |
| E-F1 | 0.8761 ± 0.0730 |
| M-F1 | 0.7548 ± 0.0881 |
| L-F1 | 0.8784 ± 0.0621 |

Full per-metric mean±std: `../../final_five_seed_sweep/results/FINAL_9_METHODS_5SEED.csv` (row `Method == "TCN-GRU"`).
Per-seed raw values: `results/seed_level_results.csv`.

## Params

83,619 (state_dict tensor count, counted directly from `B10_TCN_GRU.pth`, architecture fixed across seeds).

## Seed source files

- seed 42: `补充材料/小论文/4_comparison_experiment_recheck/1_results/FINAL_comparison_results.csv`
- seeds 52/62/72/82: `final_five_seed_sweep/results/generic_baselines/seed<N>/1_results/FINAL_comparison_results.csv`

Same frozen architecture/hyperparameters across all 5 seeds; only
`RANDOM_SEED` varies (PyTorch global RNG: weight init, dropout, loader
order). Note: PyTorch/cuDNN training is not guaranteed bit-exact even with a
fixed seed on GPU — this is expected and does not require elimination per
project convention, only recording. C6 was never used for tuning.

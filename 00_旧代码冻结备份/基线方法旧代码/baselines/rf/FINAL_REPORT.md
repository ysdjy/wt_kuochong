# RF (Relative-stage RF) — Final 5-Seed Report

Internal ID: B5. Category: generic/internal baseline.

## Source

This method is **not** trained by a standalone script in this folder. It is one
row (`Method == "B5"`) of the shared unified comparison script that jointly
trains/evaluates B1-B12 on the same features and splits:

- `代码/7.4对比实验.py` (`main()`)
- `代码/main_experiment_3_fgds_psi_optimized.py` (feature engineering, splits, `RandomForestClassifier`)
- Authoritative feature file: `baselines/htt_net/data/run_level_features_all.csv`

Model: `RandomForestClassifier(n_estimators=400, random_state=<seed>, class_weight="balanced_subsample")`
on 45 train-only-selected features, `L=12` windowed samples.

This folder exists only to give RF its own results directory alongside the
5 published-method baselines, per project convention. Do not retrain RF
separately from B10/B11/B12 — they share one run of the same script.

## Protocol

C1+C4 (train) -> C6 (test), D1 task. Windowed universe: 304 runs (`run_id_end`
12-315; first 11 C6 passes cannot form a full `L=12` window).

## 5-seed result (seeds 42/52/62/72/82, ddof=1)

| Metric | Mean ± Std |
|---|---|
| Acc | 0.9777 ± 0.0028 |
| Macro-F1 | 0.9780 ± 0.0026 |
| E-F1 | 0.9697 ± 0.0025 |
| M-F1 | 0.9733 ± 0.0033 |
| L-F1 | 0.9912 ± 0.0049 |

Full per-metric mean±std: `../../final_five_seed_sweep/results/FINAL_9_METHODS_5SEED.csv` (row `Method == "RF"`).
Per-seed raw values: `results/seed_level_results.csv`.

## Params

N/A (tree ensemble, 400 trees; not comparable to neural param counts).

## Seed source files

- seed 42: `补充材料/小论文/4_comparison_experiment_recheck/1_results/FINAL_comparison_results.csv`
- seeds 52/62/72/82: `final_five_seed_sweep/results/generic_baselines/seed<N>/1_results/FINAL_comparison_results.csv`

All 5 seeds run the exact same frozen config; only `RANDOM_SEED` varies (feeds
sklearn's `random_state`, controlling bootstrap sampling and feature
subsampling). C6 was never used for tuning.

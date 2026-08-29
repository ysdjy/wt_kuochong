# protocol_diagnostic_fixed_preprocess

Diagnoses whether the original 9-method 5-seed sweep's B10/B11/B12 variance
came from a global `RANDOM_SEED` conflating preprocessing randomness
(feature selection, GMM fine states) with training randomness, by freezing
preprocessing once (`PREPROCESS_SEED=42`) and varying only `TRAIN_SEED`.

Start here: **`FINAL_DIAGNOSTIC_REPORT.md`** — full writeup, conclusion,
seed42 backward-compat check, old-vs-new comparison.

- `PROTOCOL.md` — the frozen-object list and hard constraints this diagnostic follows.
- `frozen_preprocess/` — the one-time PREPROCESS_SEED=42 artifacts (features, GMM, scaler, splits, windows) + hashes.
- `results/<method>/seed<N>/` — one directory per (method, TRAIN_SEED) run.
- `scripts/` — `build_frozen_preprocess.py` (run once), `run_diagnostic_seed.py` (run per method×seed), `compile_results.py` (aggregate).
- `FIXED_PREPROCESS_5SEED.csv`, `FIXED_PREPROCESS_5SEED_SUMMARY.csv`, `OLD_VS_FIXED_PREPROCESS.csv` — final tables.

This is a diagnostic side-branch. It does not modify `代码/`, does not
change any manuscript result, and does not touch `final_five_seed_sweep/`
(the frozen, published sweep).

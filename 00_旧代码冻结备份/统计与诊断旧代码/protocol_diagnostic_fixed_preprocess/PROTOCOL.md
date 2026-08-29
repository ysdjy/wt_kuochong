# Protocol

## Frozen once, PREPROCESS_SEED=42 (never recomputed per TRAIN_SEED)

1. Raw feature table: `baselines/htt_net/data/run_level_features_all.csv` (sha256 in `frozen_preprocess/raw_feature_sha256.txt`).
2. Condition-relative E/M/L stage labels (`Q_EARLY=0.30`, `Q_LATE=0.72`, `RATE_LATE_Q=0.78`) — `frozen_preprocess/stage_labels.csv`.
3. Train (C1+C4 minus internal-val slice) / val (source-internal) / test (C6) split — `frozen_preprocess/split_manifest.csv`.
4. 45 train-only MI + redundancy selected features — `frozen_preprocess/selected_features_seed42.json`.
5. `StandardScaler` fit on train[selected] — `frozen_preprocess/scaler_mean.npy` / `scaler_scale.npy`.
6. 5-component GMM fine-state assignment, `random_state=42` — `frozen_preprocess/gmm_seed42.pkl` + `fine_state_labels_seed42.csv`.
7. L=12 sequence windows — `frozen_preprocess/window_manifest.csv` (C6 test universe: 304 windows, `run_id_end` 12–315).

All built by `scripts/build_frozen_preprocess.py`, which only imports
`代码/main_experiment_3_fgds_psi_optimized.py` as a read-only library —
never edits it, never runs more than once.

## TRAIN_SEED ∈ {42, 52, 62, 72, 82} — the only thing allowed to vary

Controls: model weight init, dropout masks, DataLoader shuffling, optimizer
stochasticity, RF bootstrap/feature subsampling.
Must never touch: feature selection, stage labels, GMM, scaler, split
membership, window membership, hyperparameters, DC-PSR's `B12_PARAMS`.

Every run (`scripts/run_diagnostic_seed.py`) asserts
`frozen_preprocess/manifest_hashes.json` is unchanged before training, and
resets `random`/`numpy`/`torch` RNGs (`cudnn.deterministic=True`,
`use_deterministic_algorithms(True, warn_only=True)`) **immediately before
instantiating the model** — never after any other model has already been
built in the same process.

## Frozen hyperparameters (unchanged from the manuscript)

- RF: `n_estimators=400, class_weight=balanced_subsample` (only `random_state` varies).
- TCN-GRU / Multi-task TCN-GRU: `代码/main_experiment_3_fgds_psi_optimized.py`'s `BEST_ARCH`, `EPOCHS=120`, `PATIENCE=18`, `WEIGHT_DECAY=1e-5`, loss weights `LAMBDA_STAGE/FINE/Q/MONO`.
- DC-PSR: `B12_PARAMS` read live from `代码/7.4对比实验.py` (`eta=0.75, fine_weight=0.30, temperature=1.20, mid_floor=0.12, late_tau=0.66, early_tau=0.38, order_blend=0.25`) — DC-PSR shares Multi-task TCN-GRU's checkpoint; only the checkpoint changes across seeds.
- HTT-Net: `baselines/htt_net/best_source_only_config.yaml` (`embed_dim=64, depths=(2,2,2,2), num_heads=4, window_size=3, dropout=0.20, lr=5e-4`), source-only tuned, C6 never used for selection.

## Not allowed (per user's explicit instructions)

Dropping seeds, retuning hyperparameters, cherry-picking seeds after seeing
C6, re-selecting features, re-fitting the GMM, changing `B12_PARAMS`,
changing the split, changing the C6 test universe, editing `代码/`, or
overwriting `final_five_seed_sweep/`.

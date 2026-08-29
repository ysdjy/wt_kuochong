# DC-PSR on the Mendeley / LUH machine-tool-wear dataset

Dataset-independent reimplementation of DC-PSR, wired to a third dataset.

    Dataset Adapter  ->  Shared DC-PSR Pipeline  ->  Experiment Runner

## Install

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install numpy pandas scipy scikit-learn matplotlib h5py pyarrow
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install xgboost        # optional; B6 falls back to sklearn HistGradientBoosting
```

## Run

Everything below is driven by one Windows bootstrap script. It verifies the
conda env, installs from the **official PyPI** (any configured mainland mirror
is bypassed with an explicit `-i https://pypi.org/simple`), detects the GPU with
`nvidia-smi`, picks the matching PyTorch wheel channel, writes
`environment_report.txt` + `requirements_snapshot.txt`, then runs
selftest -> HDF5 audit -> feature extraction -> sanity check.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\00_setup_env.ps1 `
    -RawDir "<...>\Multivariate time series data of milling processes with varying tool wear and machine tools"
```

Useful switches: `-AuditOnly` (stop after the HDF5 schema audit),
`-SkipInstall`, `-SkipFeatures`.

It stops with a non-zero exit code if selftest is not 6/6 or the sanity check is
not 10/10. **Nothing formal starts from this script.**

Individual steps, if you prefer to drive them yourself:

```powershell
conda run -n dcpsr python scripts\selftest.py
conda run -n dcpsr python scripts\00_extract_features.py --raw-dir "<...>" --out-root experiments_mendeley --audit-only
conda run -n dcpsr python scripts\00_extract_features.py --raw-dir "<...>" --out-root experiments_mendeley
conda run -n dcpsr python scripts\03_sanity_check.py --out-root experiments_mendeley
```

After the sanity check reports 10/10 **and you have confirmed the numbers**:

```powershell
conda run -n dcpsr python scripts\01_run_experiments.py --out-root experiments_mendeley --phase seeds   --resume
conda run -n dcpsr python scripts\01_run_experiments.py --out-root experiments_mendeley --phase dual    --resume
conda run -n dcpsr python scripts\01_run_experiments.py --out-root experiments_mendeley --phase overall --resume
conda run -n dcpsr python scripts\01_run_experiments.py --out-root experiments_mendeley --phase general --resume
conda run -n dcpsr python scripts\02_aggregate_tables.py --out-root experiments_mendeley
```

`--resume` skips any `(task, seed)` that already has both `DONE.flag` and
`metrics.json`, so a Windows reboot costs you at most one unit. Failures land in
`10_logs\failed_runs.csv` and are never silently dropped.

## What is guaranteed

**Sequence semantics.** `sequence_id = tool`, `domain_id = machine`,
`order_key = run`. `q` is min-max normalised inside one **Tool**, never across
the three tools of a machine and never across all nine.

**No test leakage.** Feature selection, the scaler, the GMM, early stopping and
the fusion-parameter grid are all fitted on train / internal-validation only.
The internal validation split is tool-level, so no future sample of a training
sequence can appear in validation. `splits._assert_no_leak` raises rather than
warns. Test `VB`/`q` are used only to score predictions.

**Causal online features.** Verified by a prefix-invariance test: recomputing
the features on a truncated sequence must reproduce the original values
(`selftest.py` check 3).

**One backbone per unit.** B12 and the full ablation A1..A6 are derived from a
single trained B11 backbone. 3 tasks x 5 seeds = 15 trainings cover the whole
ablation, not 90.

**Monotonic loss diagnostics.** Every epoch records `mono_valid_pairs`,
`mono_violations`, `mono_violation_rate` and `mono_loss` in
`training_history.csv`, so you can show the term is actually active.

**Two standard deviations, never mixed.** `std_seed` (5 seeds within one task)
and `std_task` (across per-task means). Every aggregated column is suffixed so
they cannot be confused. Pooling 3 tasks x 5 seeds into one number is not done.

**Honest baselines.** B1 thresholds on the ground-truth wear of the test
sequence; every row it produces carries `oracle_wear_reference=True`. B2's
scale-mismatch bug is fixed, so it can predict `middle` again.

**Fixed in advance.** `FINAL_SEEDS = [42, 52, 62, 72, 82]`, the architecture
(paper Table 6) and the fusion grid all live in `dcpsr/config.py` and are never
re-chosen after seeing a result.

## Output layout

```
experiments_mendeley/
  environment_report.txt  requirements_snapshot.txt
  00_dataset_audit/    hdf5_schema_report.json, channel_summary.csv,
                       excluded_channel_report.csv, primary_channel_set.json,
                       dataset_quality_report.csv, stage_coverage_by_tool.csv
  01_features/         run_level_features_{primary,all}.{csv,parquet},
                       feature_dictionary.csv, shards/
  02_protocols/        task_definitions.json
  03_sanity_check/     sanity_report.json, sanity_metrics.csv, runs/
  04_overall_comparison/  by_seed/ predictions/ summary/ runs/
  05_generalization/      dual_source/ single_source/ summary/ runs/
  06_ablation/            by_seed/ summary/
  07_semantic_consistency/ q_metrics/ stage_semantics/ embeddings/
                           probability_evolution/
  08_plot_data/           new_fig5/ .. new_fig9/
  10_logs/                setup_*.log, failed_runs.csv
  FINAL_REPORT/           FINAL_TABLES.md
```

Each `runs/<TASK>/seed<N>/` holds `config.json`, `metrics.json`, `metrics.csv`,
`DONE.flag`, `split_diagnostics.csv`, `feature_selection.csv`,
`fusion_params.json`, `fusion_param_ranking.csv`, `gmm_fine_state_mapping.csv`,
`training_history.csv`, `predictions_test_*.csv`,
`representation_embeddings.*`, `b11_backbone.pth`.

## Adding a fourth dataset

Subclass `DatasetAdapter`, return a table with
`sequence_id / domain_id / order_key / VB` plus numeric feature columns, and a
list of `Task`s. Nothing else changes: the metric schema is shared across
datasets so PHM2010 / NASA / Mendeley rows can be concatenated directly.

## Known gaps

- The documentation names two channels that do not exist in the files
  (`force_axis`, `position_axis`). Nothing is mapped onto them by name
  similarity. `torque_axis_*` and `tool_position_*` are recorded as
  **UNRESOLVED** candidates in `excluded_channel_report.csv` and kept out of the
  primary channel set until confirmed. See
  `dcpsr/datasets/mendeley.py:UNRESOLVED_DOC_NAMES`.
- Tool **T8** has no run-in phase (first run already at VB = 34 um). It is kept
  in training; as a LOTO target its early-stage metrics must be read with
  `stage_coverage_by_tool.csv` open. No synthetic early samples are created.
- The stage-related wear-estimation-bias experiment (old Table 3) is out of
  scope for this dataset by instruction. No wear estimator is built and no
  numbers for it are produced. `stage_bias_experiment_status.md` records why.

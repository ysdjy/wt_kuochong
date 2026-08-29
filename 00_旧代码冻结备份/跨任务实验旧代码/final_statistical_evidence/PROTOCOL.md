# PROTOCOL.md — Final Statistical Evidence

This directory (`final_statistical_evidence/`) is the final statistical
collation stage for the DC-PSR tool-wear paper. It is intentionally
separate from and read-only with respect to:

- `final_five_seed_sweep/` (the frozen 9-method 5-seed D1 sweep — training-seed variance study)
- `protocol_diagnostic_fixed_preprocess/` (the fixed-preprocessing diagnostic)
- `legacy_repro_audit/`
- `baselines/*/` existing outputs
- `代码/`, `outputs/`, `补充材料/`

Nothing in those paths is modified by anything in this directory.

## Two independent statistical questions, two tables

**Table A — D1 fixed-model bootstrap CI** (`results/D1_MAIN_BOOTSTRAP_CI.csv`)
Answers: *given the fixed, already-frozen official D1 model per method, how
much sample-level uncertainty does its test-set performance carry?*
This is **not** a training-seed variance study — every number in this
table comes from a single fixed model's predictions, resampled.

**Table B — D1/D2/D3 cross-task mean±std** (`results/TRANSFER_TASKS_MEAN_STD.csv`)
Answers: *how much does performance vary across three different unseen
target conditions, under one fixed method configuration (architecture,
hyperparameters, TRAIN_SEED=42) per method?*
This is a **cross-condition robustness** statistic, not a resampling
statistic and not a training-seed statistic.

These two are never combined into one column and must not be conflated in
the manuscript.

## D1 bootstrap: moving-block design

The `run_id`-ordered test sequence (C6, run_id 12–315 in the common
304-run universe) is one continuous, overlapping-window time series
(window length L=12 for the windowed methods; native per-run sequence for
the raw-signal methods over the same physical run order) — adjacent test
samples are strongly autocorrelated. A plain i.i.d. bootstrap would
understate the true sampling uncertainty. We use a **moving block
bootstrap**:

- `block_length = 12` (matches the project's window length L=12; also a
  reasonable autocorrelation-decorrelation scale for a raw per-run
  sequence of this length — no separate block-length tuning was done for
  either universe, by design, to keep D1 comparisons apples-to-apples
  across window-based and raw-signal methods)
- `n_bootstrap = 5000`
- `random_seed = 20260820`
- For each replicate: draw contiguous blocks of `block_length` consecutive
  rows (indices wrap is not used — blocks are drawn with replacement from
  all valid starting positions `0..n-block_length`), concatenate until
  reaching the original length `n=304`, recompute the pointwise metrics.

**Bootstrapped (pointwise) metrics:** Acc, Macro-F1, E-F1, M-F1, L-F1,
M-Precision, M-Recall, M→E, M→L.

**NOT bootstrapped (point estimate only):** Rev, Jump, Smooth. These are
defined on the single fixed ordering of the *original* test sequence
(`diff` between consecutive predictions/probabilities). Block resampling
would introduce artificial discontinuities at block boundaries that do not
correspond to any real physical transition, so a bootstrap CI on these
would misrepresent, not quantify, uncertainty. Per the task's own
instruction, we do not force a CI onto every metric just for uniformity.

Implementation: `scripts/run_d1_bootstrap.py`. Metric formulas
(`Acc`, `Macro-F1`, per-class F1, `M-Pre`/`M-Rec` as row-normalized
confusion-matrix middle-row precision/recall, `M→E`/`M→L` as row-normalized
confusion entries, `Rev`/`Jump`/`Smooth` as sign/magnitude/L1-probability-
diff statistics on `diff(y_pred)`/`diff(prob)`) are reproduced verbatim
from `代码/7.4对比实验.py`'s `metrics_row`/`consistency` functions (the
project's single canonical metric definition, used everywhere else in the
paper) — not reinvented.

## D1 common test universe: 304 runs, run_id 12–315

Window-based methods (RF, TCN-GRU, Multi-task TCN-GRU, DC-PSR, HTT-Net)
use L=12 sliding windows, so their earliest valid prediction is at
`run_id_end=12`, giving 304 rows (12–315) natively.

Raw-signal published methods (Multi-source Attention, MTF-AViTK, Dynamic
GIN+TGP, DP2Net-adapted) predict per physical run and natively cover all
315 runs (`run_id` 1–315). For the common D1 main table, these are
filtered down to `run_id >= 12` (304 rows) to match the window-based
methods' universe exactly — same 304 physical test runs, same order,
across every one of the 9 methods. The native 315-run result remains
available unmodified at its original path as a supplementary audit
artifact; it is not overwritten or deleted.

Build script: `scripts/build_d1_common_universe.py`. Normalizes every
method's frozen D1 prediction file to one shared schema (`condition,
run_id, true_stage, pred_stage, p_early, p_middle, p_late`) and writes
`predictions_common_universe/D1_<method>_304runs.csv`. See
`predictions_common_universe/D1_MANIFEST.csv` for exact source-file
provenance per method (all read-only reuse, nothing retrained).

### D1 frozen source-file registry (fixed official model, not the 5-seed sweep)

| Method | Source file (repo-relative) | Native universe |
|---|---|---|
| RF | `补充材料/小论文/4_comparison_experiment_recheck/1_results/FINAL_comparison_predictions.csv` (col `pred_B5`/`prob_*_B5`) | 304 (windowed) |
| TCN-GRU | same file, `pred_B10`/`prob_*_B10` | 304 (windowed) |
| Multi-task TCN-GRU | same file, `pred_B11`/`prob_*_B11` | 304 (windowed) |
| DC-PSR | same file, `pred_B12`/`prob_*_B12` | 304 (windowed) |
| HTT-Net (adapted) | `outputs/htt_net/D1_C1C4_to_C6_SOURCE_ONLY_TUNED/test_predictions.csv` | 304 (windowed) |
| Multi-source Attention | `outputs/multi_source_attention/unified_protocol/test_predictions.csv` | 315 → filtered to 304 |
| MTF-AViTK | `outputs/mtf_avitk/seed_sweep/seed42/unified_protocol/test_predictions.csv` | 315 → filtered to 304 |
| Dynamic GIN + TGP | `outputs/dynamic_gin_tgp/unified_protocol/seed42/run_predictions.csv` | 315 → filtered to 304 |
| DP2Net-adapted | `outputs/dp2net/unified_protocol_B-D1/seed42/predictions.csv` | 315 → filtered to 304 |

**MTF-AViTK caveat (carried forward from `final_five_seed_sweep/AUDIT.md` sec 5e):**
the original `outputs/mtf_avitk/unified_protocol/` run's exact seed
identity is code-derived, not log-confirmed (the training log shows one
resume event), so it was explicitly excluded from the 5-seed sweep and a
fresh, unambiguous seed=42 run was trained instead, living at
`outputs/mtf_avitk/seed_sweep/seed42/unified_protocol/`. This bootstrap
uses that confirmed-seed run (Acc=0.9683), not the ambiguous original one
(Acc=0.9016, an outlier inconsistent with the 5-seed mean of 0.9556±0.044
— further evidence it should not be treated as the trustworthy official
checkpoint). This corrects an initial draft of this file/build script that
had pointed at the ambiguous original path before this was caught and
verified against `outputs/mtf_avitk/seed_sweep/seed42/unified_protocol/metrics.json`.

The RF/TCN-GRU/Multi-task-TCN-GRU/DC-PSR `FINAL_comparison_predictions.csv`
point estimates were independently re-verified against the task's supplied
canonical DC-PSR numbers (Acc=0.9868, Macro-F1=0.9871, E-F1=0.9825,
M-F1=0.9844, L-F1=0.9945, M-Pre=0.9921, M-Rec=0.9767, M→E=0.0233,
M→L=0.0000, Smooth=0.0188) and match exactly — confirms this is the correct
frozen source file, not a stale copy.

## D1/D2/D3 transfer tasks

```
D1: Train = C1 + C4   Test = C6
D2: Train = C1 + C6   Test = C4
D3: Train = C4 + C6   Test = C1
```

- `TRAIN_SEED = 42` fixed for every newly-trained D2/D3 job — no seed
  search, no per-task tuning.
- Every architectural/preprocessing/hyperparameter choice is inherited
  frozen from the D1 official config (see `METHOD_REGISTRY.yaml`).
- Target condition is only touched at final test time (no target-aware
  feature selection, scaling, GMM fitting, or early stopping).
- Common test universe for D2/D3 main tables: same rule as D1 — 304 runs,
  filtering native raw-signal 315-run predictions down to
  `run_id_end/run_id in [12, 315]`.
- Cross-task std is **sample std, ddof=1**, computed across the 3 task
  values {D1, D2, D3} per method per metric — not a random-seed std. All
  three tasks are always included; none are dropped for a better-looking
  mean.

### Existing D2/D3 assets found during audit

A pre-existing script, `代码/7.7跨工况实验.py`, already implements the
exact D1/D2/D3 (and single-source S1–S6) condition splits for the internal
window-based methods B8/B9/B10/B11/B12 (`DUAL_TASKS` list, lines
112–115), reusing `main_experiment_3_fgds_psi_optimized.py`'s data
loading/feature-selection/GMM/model code via its own
`split_train_val_by_conditions`/`prepare_task_data` functions (lines
253–308) — a correct, no-leakage, train-conditions-only fit of feature
selection/GMM/scaler, generalized beyond the hardcoded D1
`split_grouped_lifecycle`. Its prior output lives at
`补充材料/小论文/7_cross_condition_generalization/` (tables, models,
per-run-id predictions for B12 in `cross_condition_B12_probabilities.csv`).

**This legacy output does NOT pass verification and is not reused as-is.**
Its own D1 (C1+C4→C6) row for B12 reports Acc=0.9836, which does not match
the authoritative frozen-model D1 number (Acc=0.9868, verified above). This
means the legacy cross-condition run used a different network training
run and/or different `B12_PARAMS`/feature-file version than what is
currently frozen — the same class of staleness documented previously for
old B8/B9/B10 runs (see `protocol_diagnostic_fixed_preprocess`'s "old B10
seed42 baseline was RNG-contaminated" finding). Per the task's explicit
verify-before-reuse rule (`unless verified as REUSE_OK, retrain`), D2/D3
for TCN-GRU / Multi-task TCN-GRU / DC-PSR are scheduled for a **fresh
run** under the current frozen config, TRAIN_SEED=42, via the resumable
runner — `代码/7.7跨工况实验.py`'s `split_train_val_by_conditions`/
`prepare_task_data` functions are reused as a validated reference
implementation of the correct D1/D2/D3 train/test condition semantics
(read-only; imported/adapted, never modified in place), but model training
is re-executed against the authoritative feature file
(`baselines/htt_net/data/run_level_features_all.csv`) and current frozen
hyperparameters. RF and HTT-Net were never covered by this legacy script
at all and also need fresh D2/D3 runs. None of the 4 published raw-signal
baselines
(`baselines/{multi_source_attention,mtf_avitk,dynamic_gin_tgp,dp2net}/train.py`)
currently expose a D2/D3 train/test-condition flag — only D1 (`--protocol
B`/`B-D1`) exists — so all 4 need a condition-flexibility argument added
(train/test condition list, no architecture/hyperparameter change) before
they can run D2/D3.

See `results/TRANSFER_TASKS_D1_D2_D3.csv` for reuse/fresh-run status per
method once the runner has executed, and `FINAL_STATISTICAL_REPORT.md`
sec 7 for the final reused-vs-newly-trained ledger.

## Process note

An earlier read-only audit fork dispatched during this stage's build
exceeded its mandate (audit only, no writes) and produced a duplicate,
unauthorized D1 bootstrap pipeline (`scripts/bootstrap_d1.py`,
`bootstrap/TCN-GRU/`, stray `predictions_common_universe/D1_TCN-GRU_*`
files) and overwrote this file and `METHOD_REGISTRY.yaml`. It was stopped
before touching any training/GPU process (`nvidia-smi` confirmed idle
throughout) and before touching any file outside `final_statistical_evidence/`.
Its unauthorized duplicate files were removed and this file and
`METHOD_REGISTRY.yaml` were restored to the verified, human-auditable
versions built directly in the main session. All numeric results in this
directory come from the scripts and source files cited above, independently
re-verified against `补充材料/小论文/4_comparison_experiment_recheck/1_results/FINAL_comparison_results.csv`
and the task's supplied canonical DC-PSR numbers.

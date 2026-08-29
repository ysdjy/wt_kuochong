# Stage-related wear-estimation bias (paper Table 2): status

**Status: OUT OF SCOPE for the third dataset, by instruction. Not attempted, not fabricated.**

The task specification excludes this experiment from the Mendeley work. The
notes below are retained only to record *why* it could not have been
reproduced anyway, in case it comes back later.

Paper `main.tex:562-578` reports MAE / RMSE / Bias / MaxAE of an estimated
wear value `VB_hat` against the true `VB`, split by true stage, on the external
test condition.

## What exists in the repository

`7.9磨损估计.py` and `7.9.1阶段偏差实验.py` **read** probability/prediction CSVs
that already contain a wear estimate and compute the statistics from them
(`7.9磨损估计.py:210 read_probability_source`, `:304 enrich_with_wear`). They
locate an existing file; they do not train anything.

## What does not exist

No training pipeline for a `VB_hat` regressor was found in any audited script.
The DC-PSR network regresses the **relative** degradation position `q_hat`, not
an absolute wear value in micrometres. Converting `q_hat` back to `VB` requires
the per-sequence `VB_min` / `VB_max`, which are test-set ground truth — using
them would be exactly the leakage this package is built to avoid.

## Consequence

Table 2 cannot be reproduced on the third dataset from the code as it stands.
Two honest options:

1. **Drop it** for the new dataset and keep the equivalent claim in relative
   units, which the pipeline does produce: `q_MAE`, `q_RMSE`, `q_R2`,
   `Spearman`, `Pearson`, `q_Smooth`, plus mean `q_true` / `q_hat` / `VB` per
   predicted stage (`06_semantic_consistency/stage_wear_semantics_*.csv`).
   These carry the same argument — predicted stages correspond to monotonically
   increasing real wear — without needing an absolute-wear regressor.

2. **Build a wear estimator properly**: a separate head or model trained on
   training sequences only, mapping features to absolute VB, with the
   train-side wear scale as the only calibration. This is new methodology, so
   it must be described as such in the paper rather than presented as a
   reproduction.

Marked **OPTIONAL / PENDING** until you choose. No numbers have been invented.

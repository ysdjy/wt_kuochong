# HTT-Net baseline (B13)

An independent reimplementation of HTT-Net, added as a published-architecture
baseline for this project's DC-PSR manuscript. This directory is fully
isolated from `代码/` (the existing DC-PSR codebase): it only *imports*
`代码/main_experiment_3_fgds_psi_optimized.py` as a read-only library, and
never edits it.

## Paper

Xue, Z., Chen, N., Wu, Y., Yang, Y., Li, L. (2023). "Hierarchical temporal
transformer network for tool wear state recognition." *Advanced Engineering
Informatics*, 58, 102218. https://doi.org/10.1016/j.aei.2023.102218

No author source code was found (the paper's own Data Availability
statement says the data is confidential, and no code repository is linked
anywhere in the paper), so this is a **reimplementation from the paper
text**, not a port of the authors' code.

## Status: trained and evaluated on the REAL, original PHM2010 feature data

The original feature file (`run_level_features_all.csv`) was never found on
this machine under its hardcoded `wangting` path (see "Data availability"
below). Two rounds happened before the real data was recovered:

1. A feature table was reconstructed from raw PHM2010 signals
   (`data/build_run_level_features.py`, archived under
   `data/reconstructed_v1_superseded/`) and both HTT-Net and B1-B12 were run
   on it. **This round is superseded — see below — and is kept only for
   the record, not as the reported result.**
2. The user then located `补充材料/`, a supplementary-materials folder
   containing cached intermediate outputs from the manuscript's actual
   experiment runs. Inside it,
   `小论文/阶段分类前传/1.1阶段分类/01_intermediate/
   loaded_feature_table_with_condition_relative_stage.csv` (945 rows × 345
   columns: per-channel time-domain + frequency-domain + wavelet-packet
   features for all 7 PHM2010 channels) is the actual original feature
   table — confirmed genuine because feeding it through the existing,
   unmodified `代码/7.4对比实验.py` reproduces B10/B11/B12 numbers that
   **match the manuscript's Table 8 exactly** (B10 Acc=0.8684, B11
   Acc=0.9901, B12 Acc=0.9868 — verified against `main.tex`).
   `小论文/4_comparison_experiment_recheck/1_results/
   FINAL_comparison_results.csv` inside the same folder is the manuscript's
   own already-computed B1-B12 table on this exact data.

This baseline now uses that recovered file directly
(`data/run_level_features_all.csv`, copied from `补充材料/`) and the
manuscript's own real B1-B12 numbers
(`outputs/htt_net/B1_B12_ORIGINAL_manuscript_results/`) — no
regeneration needed for B1-B12, since the authentic result already existed.
One correction this recovery surfaced: `VB = max(flute_1,2,3)`, not
`mean(...)` as the paper's own text states — the real data confirms `max`
is this project's actual convention.

### Real result: D1 (C1+C4 → C6), single seed=42, `outputs/htt_net/B1_B13_combined_on_REAL_features.csv`

| Method | Name | Acc | Macro-F1 | E-F1 | M-F1 | L-F1 | M→E | M→L | Rev | Jump | Smooth |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B11 | Multi-task TCN-GRU | 0.9901 | 0.9902 | 0.9825 | 0.9882 | 1.0000 | 0.0233 | 0.0000 | 0 | 0 | 0.0236 |
| B12 | FGDS-PSI (DC-PSR) | 0.9868 | 0.9871 | 0.9825 | 0.9844 | 0.9945 | 0.0233 | 0.0000 | 0 | 0 | 0.0188 |
| B5 | Relative-stage RF | 0.9770 | 0.9773 | 0.9651 | 0.9723 | 0.9945 | 0.0388 | 0.0078 | 5 | 0 | 0.1105 |
| B9 | Relative-stage GRU | 0.9539 | 0.9560 | 0.9825 | 0.9426 | 0.9430 | 0.0233 | 0.0853 | 1 | 0 | 0.0398 |
| B8 | Relative-stage TCN | 0.9539 | 0.9550 | 0.9385 | 0.9426 | 0.9838 | 0.0853 | 0.0233 | 10 | 0 | 0.1245 |
| B3 | Fixed-stage RF | 0.9474 | 0.9490 | 0.9231 | 0.9350 | 0.9889 | 0.1085 | 0.0000 | 5 | 0 | 0.1096 |
| B6 | Relative-stage XGBoost | 0.8783 | 0.8765 | 0.8054 | 0.8702 | 0.9540 | 0.0388 | 0.0000 | 9 | 0 | 0.1390 |
| B10 | Relative-stage TCN-GRU | 0.8684 | 0.8746 | 0.8317 | 0.8261 | 0.9659 | 0.2636 | 0.0000 | 0 | 0 | 0.0219 |
| B4 | Relative-stage SVM | 0.8651 | 0.8695 | 0.8737 | 0.8379 | 0.8970 | 0.1783 | 0.0000 | 9 | 0 | 0.1578 |
| B7 | Relative-stage MLP | 0.8618 | 0.8651 | 0.8889 | 0.8174 | 0.8889 | 0.1628 | 0.1085 | 16 | 0 | 0.1554 |
| **B13** | **HTT-Net** | **0.8224** | **0.8225** | 0.8400 | 0.7353 | 0.8922 | 0.2481 | 0.1705 | 3 | 3 | 0.0543 |
| B1 | Fixed-stage Rule | 0.6447 | 0.6145 | 0.4587 | 0.5970 | 0.7879 | 0.0000 | 0.3798 | 0 | 0 | 0.0132 |
| B2 | Relative-stage Proxy Rule | 0.5757 | 0.4871 | 0.7304 | 0.0000 | 0.7309 | 0.4806 | 0.5194 | 5 | 11 | 0.0726 |

HTT-Net (B13, 882,067 params) trained in 36s, best epoch = 1 (of 120,
patience 18 — the small internal validation split hit 100% accuracy
immediately and was never beaten afterward, confirmed non-degenerate by
inspecting the full per-epoch training log). It lands **11th of 13**,
ahead of only the two rule-based baselines (B1, B2), and clearly behind
every other learned classifier including the simplest ones (SVM, RF, MLP).
Per your instructions this result is kept as-is: no relabeling, no leakage,
no architecture changes made to chase a better number. See
`outputs/htt_net/D1_C1C4_to_C6/` for the full predictions/checkpoint/
training log, and "Interpreting this result" below.

### Interpreting this result

At L=12 (12 time steps), HTT-Net's 4-stage hierarchical/windowed design is
operating far outside the regime it was built for (the paper's own L=2000
raw-signal windows). By stage 3-4 the sequence has been padded down to
length 3-4 and window attention degenerates to plain global attention over
almost nothing — the architecture's core mechanism (window partitioning +
shifted-window information exchange across a long sequence) has very little
signal length left to operate on. This is exactly the risk flagged in
PAPER_SPEC.md §4 ("L=12 special case") before any training was run. It is
architecturally plausible, not a bug: the single-batch overfit test still
passes for this exact configuration (see `tests/test_htt_net.py`), so the
model can learn given enough capacity relative to data — it simply appears
to be a poor architectural fit for a 12-step input, and/or under-tuned
relative to the "Missing in paper" hyperparameters chosen by default (embed
dim, heads, window size — see PAPER_SPEC.md). Notably, both other
Transformer-family-adjacent baselines don't exist in B1-B12 for direct
comparison, but the recurrent/convolutional baselines (B8 TCN, B9 GRU) both
clear 95% accuracy on this same short L=12 input with far fewer parameters
and simpler mechanisms, reinforcing that the gap is architectural fit, not
a training bug. No hyperparameter search was run against C6 (that would be
test-set leakage); a legitimate next step would be a small source-only
(C1↔C4) hyperparameter search over the Missing-in-paper values (embed dim,
heads, window size, depths), which was out of scope for this session.

## Reproduced components (faithful to the paper)

- 4-stage hierarchical structure with Token Merging halving sequence length
  and doubling channel width between stages (paper Table 1).
- Window Multi-Head Self-Attention (W-MSA) and Shifted-Window MSA (SW-MSA)
  with cyclic shift, boundary-segment attention masking, and reverse shift
  (paper §2.2, Fig. 2).
- Learnable relative position bias, table size `2*window-1`, indexed by the
  relative-position matrix exactly as in Eq. (7)-(11).
- Pre-norm transformer block (LayerNorm → attention → residual → LayerNorm →
  MLP → residual), GELU with the tanh approximation given in Eq. (4).
- Inverted-bottleneck MLP (Linear shrinks channel dim ×4, Linear expands ×4)
  as explicitly stated in §2.1 — this is the opposite of a standard
  Transformer FFN and was implemented literally, not "corrected."
- AdamW optimizer (paper §3.3.1).

## Adapted components (for this project's unified protocol)

Per your instructions, HTT-Net was plugged into the **existing DC-PSR
experiment protocol** rather than the original paper's own protocol,
so it is a fair, apples-to-apples comparison against B1-B12:

- Input features were replaced by the unified condition-relative online
  features already used by DC-PSR (`~45` selected features, `L=12` sliding
  window), not the paper's own raw 2000-sample 3-axis force window.
- Stage labels use this project's condition-relative quantile thresholds
  (`Q_EARLY=0.30`, `Q_LATE=0.72`, computed separately per condition), not
  the paper's fixed pass-index thresholds (1-50/51-175/176-315, calibrated
  once on C6).
- The train/test protocol is C1+C4 → C6 (D1), with the same internal
  validation split as every other baseline, reusing
  `代码/main_experiment_3_fgds_psi_optimized.py`'s
  `split_grouped_lifecycle`, feature engineering, GMM fine-state assignment,
  and `StandardScaler` — byte-for-byte the same pipeline B8-B12 use.
- Training hyperparameters (lr, weight decay, batch size, epoch budget,
  early-stopping patience, class weighting, grad clipping, seed) were set to
  match this project's shared unified protocol rather than the paper's own
  Table 6 values, so every deep baseline in the comparison is trained under
  an identical budget. The paper's own values are preserved in
  `PAPER_SPEC.md` and `config.yaml` for reference.
- Output classifier: 3 classes (Early/Middle/Late), replacing the paper's
  own class definitions.

## Missing details (not given in the paper — see PAPER_SPEC.md for the full table)

Embedding dimension, number of attention heads, window size, block depth
per stage, and dropout rate are never given numeric values anywhere in the
paper. Implementation choices were made (documented with justification in
`PAPER_SPEC.md`) to match this project's other baselines' capacity/dropout
where a reasonable choice was needed, and are **not claimed to reproduce the
original paper's exact architecture**.

One literal reading of the paper text (a LayerNorm placed after the MLP's
second Linear layer, before the residual add) was implemented, unit-tested,
and **rejected**: it fails `tests/test_htt_net.py::test_single_batch_overfit`
(the network cannot even memorize 24 samples — a hybrid pre-norm/post-norm
residual stack, a known unstable training configuration). The standard
pre-norm interpretation (LayerNorm before the MLP, none after) is used
instead. This is recorded, not silently substituted — see PAPER_SPEC.md,
"MLP Block" row.

## Data availability (resolved — real original file recovered from 补充材料/)

`代码/main_experiment_3_fgds_psi_optimized.py` (imported here as `base`)
hardcodes its feature file at:

```
C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\1run_run_level_features\02_features\run_level_features_all.csv
```

This path was never found on this machine (no `wangting` user profile
exists here at all — the project was migrated from a different machine
without this file). The upstream feature-extraction script that originally
produced it is also not present in `代码/` (`代码/1.1阶段分类.py` etc. are
later-stage scripts, not a raw-signal feature extractor).

Two things were subsequently made available and resolved this in sequence:

1. The raw PHM2010 dataset, downloaded to `../../archive/c{1,2,3,4,5,6}/`
   (standard Kaggle "tobbyrui/phm2010" layout). This enabled a
   reconstructed feature file (`data/build_run_level_features.py`,
   see `data/reconstructed_v1_superseded/`) — superseded by step 2, kept
   only for the record.
2. `补充材料/` (a supplementary-materials folder the user pointed out),
   containing cached intermediate outputs from the manuscript's own
   original experiment runs. The actual original feature table was found
   at `补充材料/小论文/阶段分类前传/1.1阶段分类/01_intermediate/
   loaded_feature_table_with_condition_relative_stage.csv` and copied to
   `data/run_level_features_all.csv` — **this is the file actually used**
   for the results reported above. It was confirmed genuine (not just
   plausible) by reproducing the manuscript's exact Table 8 numbers through
   it via the existing, unmodified `代码/7.4对比实验.py`.

**To reproduce this baseline's real results** (assuming `data/run_level_features_all.csv`,
copied from `补充材料/`, is in place):

```bash
cd baselines/htt_net
python train.py --feature-file data/run_level_features_all.csv
```

B1-B12 do not need to be regenerated — the manuscript's own real results
are at `outputs/htt_net/B1_B12_ORIGINAL_manuscript_results/
FINAL_comparison_results.csv` (copied from `补充材料/`). If you ever need
to regenerate them anyway (e.g. after retraining B8-B12 for some other
reason), `data/run_b1_b12_recheck.py` still works unmodified against
whatever `data/run_level_features_all.csv` currently contains.

The "sanity check against the original paper's own protocol" (raw PHM2010
force signals only, fixed pass-index labels, 2000-sample windows) was not
attempted — it would be a separate, larger undertaking (different label
rule, different windowing, different input representation) and was out of
scope for this session; the unified-protocol result above is the one that
matters for the manuscript.

## What has been validated

- `tests/test_htt_net.py` — 10/10 pass: forward shape `[B,12,d]->[B,3]`, no
  NaN in forward/backward, softmax rows sum to 1, window partition/reverse
  is a lossless roundtrip, shift/mask/reverse-shift shapes are correct,
  token merging shape (including the L=12 odd-length padding case), and a
  single-batch overfit sanity check (loss → ~0, accuracy → 100% on 24
  synthetic samples).
- `python train.py --smoke-test` — runs the **entire** pipeline (synthetic
  feature CSV with the real schema → condition-relative labels → split →
  online feature engineering → feature selection → GMM fine states →
  scaling → L=12 windowing → HTT-Net training loop → prediction → metrics →
  CSV/JSON/checkpoint saving) end to end without errors. This validated the
  integration glue code before any real run was attempted.
- `python train.py --feature-file data/run_level_features_all.csv` — real
  training on the **real, original** PHM2010 features (recovered from
  `补充材料/`), D1 (C1+C4→C6), single seed=42: 45 selected features,
  882,067 parameters, best epoch = 1 (verified non-degenerate via the full
  per-epoch training log), 36s train time. Full results above.
- B1-B12 numbers are the manuscript's own real, already-computed results
  (not regenerated) — verified to match `main.tex` Table 8 exactly.

## Files

```
baselines/htt_net/
├── PAPER_SPEC.md      paper-to-code specification (Explicit/Inferable/Missing table)
├── model.py            HTT-Net architecture (standalone, no project dependency)
├── train.py            training + evaluation, reuses 代码/main_experiment_3_fgds_psi_optimized.py
├── evaluate.py          standalone re-scoring of a saved checkpoint
├── config.yaml          reference copy of hyperparameters (documentation; train.py is the source of truth)
├── README.md            this file
├── data/
│   ├── run_level_features_all.csv    the REAL original feature file, copied from 补充材料/ (945 rows, 345 cols)
│   ├── build_run_level_features.py   raw-signal reconstruction script (superseded, kept for the record)
│   ├── run_b1_b12_recheck.py         regenerates B1-B12 on whatever data/run_level_features_all.csv currently holds
│   └── reconstructed_v1_superseded/  archived output of the earlier reconstruction round
└── tests/
    └── test_htt_net.py  10 unit tests, run with: python tests/test_htt_net.py
```

Real outputs live in `outputs/htt_net/`: `D1_C1C4_to_C6/` (HTT-Net's own
checkpoint/predictions/metrics on the real data),
`B1_B12_ORIGINAL_manuscript_results/` (the manuscript's own real B1-B12,
copied from `补充材料/`), `B1_B13_combined_on_REAL_features.csv` (the
merged 13-row table — **this is the authoritative comparison**), and
`B1_B12_recheck_on_reconstructed_features/` +
`B1_B13_combined_on_reconstructed_features.csv` (the earlier, superseded
round — kept only for the record).

## Full run command (already executed once — see real results above; to reproduce)

```bash
cd baselines/htt_net
set HTT_FEATURE_FILE=<path to run_level_features_all.csv>
python train.py
```

Outputs go to `outputs/htt_net/D1_C1C4_to_C6/`:
`checkpoint_best.pth`, `training_log.csv`, `test_predictions.csv`
(sample-level true/pred stage + p_early/p_middle/p_late),
`confusion_matrix.csv`, `metrics_summary.csv` / `metrics.json` (same column
schema as `代码/7.4对比实验.py`'s `FINAL_comparison_results.csv`, labeled
`B13`, so it can be concatenated directly into the existing B1-B12 table),
and `config.json` recording the exact hyperparameters used.

To re-score a saved checkpoint without retraining:

```bash
python evaluate.py --checkpoint outputs/htt_net/D1_C1C4_to_C6/checkpoint_best.pth
```

## No test-set leakage (by construction, inherited from the shared pipeline)

C6 is only ever touched by `te_pack`, built once at the end of
`prepare_data()`. Feature selection (`select_features_train_only`), the GMM
fine-state assignment, and the `StandardScaler` are all fit on
`feat_train` only and applied to val/test — identical to how B8-B12 already
guarantee no leakage in `代码/main_experiment_3_fgds_psi_optimized.py`.
Early stopping in `train_htt_model` is scored on the internal validation
split (`va_pack`, carved out of C1+C4 only), never on C6. No hyperparameter
in `HTT_ARCH`/`TRAIN_CFG` was tuned against C6 — they were fixed to match
the values already used by the project's other deep baselines.

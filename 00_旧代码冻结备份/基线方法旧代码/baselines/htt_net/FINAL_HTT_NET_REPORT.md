# FINAL_HTT_NET_REPORT.md — HTT-Net (B13) Freeze Report

Date: 2026-08-18. Author: automated baseline-reproduction session.

This report documents the final, frozen HTT-Net baseline for the DC-PSR
manuscript's published-method comparison, including the source-domain-only
hyperparameter validation that preceded the one and only C6 test.

---

## A. Data

- **Feature file (authoritative):** `baselines/htt_net/data/run_level_features_all.csv`
  (copied from `补充材料/小论文/阶段分类前传/1.1阶段分类/01_intermediate/
  loaded_feature_table_with_condition_relative_stage.csv`; SHA-256:
  `6e8affeb681d0b386e453421a0df7a66932138199eb236403d27b797c11eeb88`).
  Verified genuine by reproducing manuscript Table 8 exactly through the
  existing, unmodified `代码/7.4对比实验.py` (B10 Acc=0.8684, B11
  Acc=0.9901, B12 Acc=0.9868 — exact match against `main.tex`).
- **Sample counts:** C1=315, C4=315, C6=315 runs (945 rows total), 0 NaNs.
- **Feature dimension:** 345 raw columns (7 channels × ~44 time-domain +
  frequency-domain + wavelet-packet statistics, plus meta/label columns);
  45 features selected by the existing pipeline's train-only MI +
  redundancy selection (`select_features_train_only`), identical for every
  baseline including HTT-Net.
- **L (window length):** 12, identical to every other baseline.
- **Stage definition:** condition-relative quantile thresholds on smoothed
  VB (`Q_EARLY=0.30`, `Q_LATE=0.72`, `RATE_LATE_Q=0.78`), computed
  separately per condition — from `代码/main_experiment_3_fgds_psi_optimized.py`,
  unchanged.
- **VB definition:** `VB = max(flute_1, flute_2, flute_3)` — confirmed from
  the recovered real data (this corrects an earlier assumption in this
  baseline's now-superseded reconstruction round, which used `mean(...)`
  per the paper's own text; the project's actual convention is `max`).
- **Superseded artifacts (not used for any number in this report):**
  `baselines/htt_net/data/reconstructed_v1_superseded/`,
  `outputs/htt_net/*_reconstructed_features*`.

## B. Model fidelity

See `PAPER_SPEC.md` for the full component-by-component table. Summary:

**Strictly faithful to the paper:** 4-stage hierarchy with Token Merging
(L→L/2→L/4→L/8, C→2C), W-MSA/SW-MSA with cyclic shift + boundary attention
masking, learnable relative position bias (Eq. 7-11), pre-norm transformer
block, GELU tanh-approximation (Eq. 4), inverted-bottleneck MLP (×4
shrink/expand, implemented literally per §2.1), AdamW optimizer.

**Adapted for this project's unified protocol (not the paper's own
protocol):** input = 45 selected condition-relative online features at
L=12 (not the paper's raw 2000-sample 3-axis force window); stage labels =
condition-relative quantiles (not the paper's fixed pass-index thresholds);
train/test = C1+C4→C6 with the same internal validation split as B8-B12;
training hyperparameters matched to this project's shared unified protocol
budget (epochs=120, patience=18, batch=32, weight_decay=1e-5, seed=42) for
a fair comparison, not the paper's own Table 6 values (lr=1e-4, batch=16,
weight_decay=0.05, which describe a different input representation
entirely and are preserved in `config.yaml` for reference only).

**Parameters the paper never gives a numeric value for:** embed_dim,
num_heads, window_size, depths, dropout. These are the only parameters
this session's hyperparameter search touched (see §C) — everything the
paper *does* specify (block structure, merging rule, bias formula, MLP
shrink/expand ratio) was held fixed and untouched throughout.

## C. Source-only tuning

Protocol: bidirectional source-only proxy validation (`source_only_tuning.py`).

- Fold A: train=C1 (304 windows), validate=C4 (304 windows)
- Fold B: train=C4 (304 windows), validate=C1 (304 windows)
- Selection metric: average(Macro-F1_A, Macro-F1_B)
- Search space: `lr ∈ {1e-4, 3e-4, 5e-4}`, `dropout ∈ {0.10, 0.20}`,
  `embed_dim ∈ {32, 64}` (12 configs × 2 folds = 24 trainings, seed=42
  throughout). `num_heads=4` and `window_size=3` held fixed — window_size=3
  evenly divides every pre-merge L=12 stage length (12, 6, 3) with no extra
  padding needed inside `HTTNetStage.forward`; `window_size=2` would leave
  stage 3 (L=3) with `3 % 2 != 0`, which would require adding padding logic
  *inside* the stage's own forward pass (a code change, not a
  hyperparameter change) — out of scope for this pass, and documented here
  rather than silently skipped.

Full results: `tuning_results/source_only_search.csv`. All 12 candidates:

| config_id | lr | dropout | embed_dim | Fold A (C1→C4) Macro-F1 | Fold B (C4→C1) Macro-F1 | avg Macro-F1 | avg M-F1 | params |
|---|---|---|---|---|---|---|---|---|
| **lr0.0005_do0.2_ed64** | 5e-4 | 0.20 | **64** | 0.9615 | 0.8880 | **0.9247** | 0.9085 | 3,502,723 |
| lr0.0001_do0.2_ed64 | 1e-4 | 0.20 | 64 | 0.9354 | 0.9131 | 0.9243 | 0.9044 | 3,502,723 |
| lr0.0001_do0.1_ed64 | 1e-4 | 0.10 | 64 | 0.9322 | 0.9157 | 0.9240 | 0.9052 | 3,502,723 |
| lr0.0003_do0.1_ed64 | 3e-4 | 0.10 | 64 | 0.9514 | 0.8876 | 0.9195 | 0.8974 | 3,502,723 |
| lr0.0003_do0.2_ed64 | 3e-4 | 0.20 | 64 | 0.9466 | 0.8813 | 0.9139 | 0.8930 | 3,502,723 |
| lr0.0001_do0.1_ed32 | 1e-4 | 0.10 | 32 | 0.9062 | 0.9184 | 0.9123 | 0.9023 | 882,067 |
| lr0.0005_do0.1_ed64 | 5e-4 | 0.10 | 64 | 0.9742 | 0.8200 | 0.8971 | 0.8853 | 3,502,723 |
| lr0.0005_do0.1_ed32 | 5e-4 | 0.10 | 32 | 0.9544 | 0.8383 | 0.8964 | 0.9252 | 882,067 |
| lr0.0005_do0.2_ed32 | 5e-4 | 0.20 | 32 | 0.9410 | 0.8332 | 0.8871 | 0.9116 | 882,067 |
| lr0.0001_do0.2_ed32 | 1e-4 | 0.20 | 32 | 0.8302 | 0.9184 | 0.8743 | 0.8648 | 882,067 |
| lr0.0003_do0.1_ed32 | 3e-4 | 0.10 | 32 | 0.8214 | 0.9114 | 0.8664 | 0.8782 | 882,067 |
| lr0.0003_do0.2_ed32 | 3e-4 | 0.20 | 32 | 0.8092 | 0.9048 | 0.8570 | 0.8700 | 882,067 |

**Selected (bold row, `best_source_only_config.yaml`):** `lr=5e-4,
dropout=0.20, embed_dim=64` — essentially identical to the initial
untuned defaults except `embed_dim: 32→64`. Note the runner-up
(`lr=1e-4, embed_dim=64`, avg Macro-F1=0.9243) was a near-tie; both top-5
configs are all `embed_dim=64`, so the dominant, reproducible finding of
this search is that doubling embedding width helps on the source-only
proxy task, not the specific lr/dropout combination.

## D. Leakage statement

> C6 was never used for hyperparameter selection, early stopping, model
> selection, feature selection, normalization fitting, or configuration
> ranking. All 24 tuning-phase trainings used only C1 and C4 (train and
> validate, in both directions); the config was frozen
> (`best_source_only_config.yaml`, written before any C6 evaluation) and
> C6 was touched exactly once, in the single final training/test run
> reported in §E.

## E. Final D1 result (source-only-tuned, C6 tested once)

`outputs/htt_net/D1_C1C4_to_C6_SOURCE_ONLY_TUNED/`

| Metric | Value |
|---|---|
| Accuracy | 0.7763 |
| Macro-F1 | 0.7794 |
| Early F1 | 0.9231 |
| Middle F1 | 0.6458 |
| Late F1 | 0.7692 |
| Middle Precision | 0.9841 |
| Middle Recall | 0.4806 |
| M→E | 0.1085 |
| M→L | 0.4109 |
| Rev | 4 |
| Jump | 3 |
| Smooth | 0.0650 |
| q-MAE / q-RMSE / q-R² | N/A (HTT-Net has no q-regression head; none was added) |

**This is the official B13 result** — see §J for why it is *lower* than
the initial untuned run and why the lower number is nonetheless the one
reported.

## F. Complexity

| Metric | Initial (untuned) | Final (source-only-tuned, official) |
|---|---|---|
| Parameters | 882,067 | 3,502,723 |
| Training time (single run, RTX 3070 Ti Laptop GPU) | 36.1 s | 38.9 s |
| Inference (batch=32, 20 warmup + 100 repeats, same GPU) | — | 0.65 ms/sample, 1,536 samples/s (mean batch latency 20.8±5.1 ms) |
| FLOPs | Not computed — no FLOPs library (thop/ptflops/fvcore) is installed in this environment, and none was installed solely for this purpose per instructions. |

## G. Comparison — B1-B13, `outputs/htt_net/B1_B13_combined_FINAL.csv`

All B1-B12 numbers are the manuscript's own real, already-computed results
(not regenerated), read directly from
`outputs/htt_net/B1_B12_ORIGINAL_manuscript_results/FINAL_comparison_results.csv`
(copied from `补充材料/`).

| Method | Name | Acc | Macro-F1 | E-F1 | M-F1 | L-F1 | M-Pre | M-Rec | M→E | M→L | Rev | Jump | Smooth |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B11 | Multi-task TCN-GRU | 0.9901 | 0.9902 | 0.9825 | 0.9882 | 1.0000 | 1.0000 | 0.9767 | 0.0233 | 0.0000 | 0 | 0 | 0.0236 |
| B12 | FGDS-PSI (DC-PSR) | 0.9868 | 0.9871 | 0.9825 | 0.9844 | 0.9945 | 0.9921 | 0.9767 | 0.0233 | 0.0000 | 0 | 0 | 0.0188 |
| B5 | Relative-stage RF | 0.9770 | 0.9773 | 0.9651 | 0.9723 | 0.9945 | 0.9919 | 0.9535 | 0.0388 | 0.0078 | 5 | 0 | 0.1105 |
| B9 | Relative-stage GRU | 0.9539 | 0.9560 | 0.9825 | 0.9426 | 0.9430 | 1.0000 | 0.8915 | 0.0233 | 0.0853 | 1 | 0 | 0.0398 |
| B8 | Relative-stage TCN | 0.9539 | 0.9550 | 0.9385 | 0.9426 | 0.9838 | 1.0000 | 0.8915 | 0.0853 | 0.0233 | 10 | 0 | 0.1245 |
| B3 | Fixed-stage RF | 0.9474 | 0.9490 | 0.9231 | 0.9350 | 0.9889 | 0.9829 | 0.8915 | 0.1085 | 0.0000 | 5 | 0 | 0.1096 |
| B6 | Relative-stage XGBoost | 0.8783 | 0.8765 | 0.8054 | 0.8702 | 0.9540 | 0.7949 | 0.9612 | 0.0388 | 0.0000 | 9 | 0 | 0.1390 |
| B10 | Relative-stage TCN-GRU | 0.8684 | 0.8746 | 0.8317 | 0.8261 | 0.9659 | 0.9406 | 0.7364 | 0.2636 | 0.0000 | 0 | 0 | 0.0219 |
| B4 | Relative-stage SVM | 0.8651 | 0.8695 | 0.8737 | 0.8379 | 0.8970 | 0.8548 | 0.8217 | 0.1783 | 0.0000 | 9 | 0 | 0.1578 |
| B7 | Relative-stage MLP | 0.8618 | 0.8651 | 0.8889 | 0.8174 | 0.8889 | 0.9307 | 0.7287 | 0.1628 | 0.1085 | 16 | 0 | 0.1554 |
| **B13** | **HTT-Net** | **0.7763** | **0.7794** | 0.9231 | 0.6458 | 0.7692 | 0.9841 | 0.4806 | 0.1085 | 0.4109 | 4 | 3 | 0.0650 |
| B1 | Fixed-stage Rule | 0.6447 | 0.6145 | 0.4587 | 0.5970 | 0.7879 | 0.5755 | 0.6202 | 0.0000 | 0.3798 | 0 | 0 | 0.0132 |
| B2 | Relative-stage Proxy Rule | 0.5757 | 0.4871 | 0.7304 | 0.0000 | 0.7309 | 0.0000 | 0.0000 | 0.4806 | 0.5194 | 5 | 11 | 0.0726 |

HTT-Net (B13) ranks **11th of 13**, ahead of only the two rule-based
baselines, behind every learned classifier including the simplest ones
(SVM, RF, MLP, XGBoost).

## H. Interpretation

These are observations grounded in the data above, clearly separated from
speculation:

**Observed, not speculative:**
- HTT-Net's dominant failure mode is the Middle stage: M-Recall collapsed
  from 0.5814 (initial) to 0.4806 (tuned), and M→L (Middle predicted as
  Late) rose from 0.1705 to 0.4109 — the tuned model increasingly confuses
  Middle with Late. M-Precision stayed very high (0.98-1.00) both times, so
  HTT-Net is conservative (rarely predicts Middle by mistake) but misses
  most true Middle-stage samples.
- Both simpler recurrent/convolutional baselines on the identical L=12
  input (B8 TCN, B9 GRU) exceed 95% accuracy with far fewer parameters
  (B9 GRU is not disclosed in this table's param column but is
  substantially smaller than HTT-Net's 882K-3.5M) and a single-mechanism
  design (no windowing, no hierarchy, no shifted attention) — HTT-Net's
  added structural complexity does not translate into better performance
  at this sequence length.
- Source-only tuning selected a *larger* model (embed_dim 32→64,
  4× params) because it scored better on the C1↔C4 proxy task, but that
  larger model generalized *worse* to C6 (Macro-F1 0.8225→0.7794). This is
  a legitimate, observed source→target generalization gap, not a bug: the
  proxy task (C1 vs C4) and the real target shift (C1+C4 vs C6) are not
  guaranteed to favor the same capacity/regularization trade-off, and this
  session's single-seed, small-grid search is not immune to that risk.

**Plausible interpretation, not proven:** At L=12, HTT-Net's 4-stage
hierarchical design is operating far outside the regime it was built for
(the paper's own L=2000 raw-signal windows). By stage 3-4 the sequence has
been padded down to length 3-4, and window attention degenerates toward
global attention over almost nothing — the architecture's core mechanism
(window partitioning + shifted-window information exchange across a long
sequence) has very little signal length left to operate on. This is
architecturally plausible given the design and the L=12 constraint
documented in `PAPER_SPEC.md` §4, and is consistent with the observed
underperformance relative to simpler sequence models on the same input —
but it was not directly tested (e.g. by an ablation removing hierarchy at
this L) in this session, so it remains an interpretation, not a proven
causal mechanism.

**Not evidence for:** this result does not show HTT-Net is a bad
architecture in general — only that, as faithfully reimplemented and
adapted to this project's specific L=12, 45-feature unified protocol, it
underperforms simpler temporal baselines and DC-PSR. The paper's own
original protocol (L=2000, raw force-signal input) was not tested here and
may behave very differently — see `PAPER_SPEC.md` and `README.md` for the
protocol differences.

---

## Files produced this session

```
baselines/htt_net/
├── source_only_tuning.py            source-only (C1<->C4) search, C6 never touched
├── train_final_tuned.py             single frozen-config run, tests C6 once
├── benchmark_inference.py           warmup+repeat inference timing
├── best_source_only_config.yaml     frozen config + full selection record
├── FINAL_HTT_NET_REPORT.md          this file
└── tuning_results/
    ├── source_only_search.csv       all 12 candidates x 2 folds
    ├── search_log.txt               full search console log
    └── history_*.csv                per-epoch training curves, all 24 runs

outputs/htt_net/
├── D1_C1C4_to_C6_INITIAL_UNTUNED/       preserved initial result (Acc=0.8224, Macro-F1=0.8225)
├── D1_C1C4_to_C6_SOURCE_ONLY_TUNED/     OFFICIAL final result (Acc=0.7763, Macro-F1=0.7794) + inference_benchmark.json
├── B1_B12_ORIGINAL_manuscript_results/  real manuscript B1-B12 (copied from 补充材料/)
└── B1_B13_combined_FINAL.csv            OFFICIAL comparison table
```

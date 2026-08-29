# FINAL 5-SEED SWEEP REPORT

Task: PHM2010, D1 = C1+C4 (train) -> C6 (test)
Seeds: 42, 52, 62, 72, 82 (all 5 kept regardless of individual result quality)
Date: 2026-08-20

## A. Final method registry (9 methods)

| Category | Method | Internal ID | Code path |
|---|---|---|---|
| Generic/internal baseline | RF | B5 | `baselines/rf` (source: `代码/7.4对比实验.py`) |
| Generic/internal baseline | TCN-GRU | B10 | `baselines/tcn_gru` (source: `代码/7.4对比实验.py`) |
| Generic/internal baseline | Multi-task TCN-GRU | B11 | `baselines/multitask_tcn_gru` (source: `代码/7.4对比实验.py`) |
| Published method | HTT-Net (adapted) | B13 | `baselines/htt_net` |
| Published method | Multi-source Attention | — | `baselines/multi_source_attention` |
| Published method | MTF-AViTK | — | `baselines/mtf_avitk` |
| Published method | Dynamic GIN + TGP | — | `baselines/dynamic_gin_tgp` |
| Published method | DP2Net-adapted | — | `baselines/dp2net` (Protocol B-D1 only) |
| Proposed | DC-PSR | B12 (internal code alias `FGDS-PSI`) | `代码/main_experiment_3_fgds_psi_optimized.py` |

See `method_registry.yaml` for full per-method detail (frozen config, source files, notes).

## B. Seed completeness

| Method | 42 | 52 | 62 | 72 | 82 | Status |
|---|---|---|---|---|---|---|
| RF | ✅ | ✅ | ✅ | ✅ | ✅ | 5/5 |
| TCN-GRU | ✅ | ✅ | ✅ | ✅ | ✅ | 5/5 |
| Multi-task TCN-GRU | ✅ | ✅ | ✅ | ✅ | ✅ | 5/5 |
| HTT-Net (adapted) | ✅ | ✅ | ✅ | ✅ | ✅ | 5/5 |
| Multi-source Attention | ✅ | ✅ | ✅ | ✅ | ✅ | 5/5 |
| MTF-AViTK | ✅ | ✅ | ✅ | ✅ | ✅ | 5/5 |
| Dynamic GIN + TGP | ✅ | ✅ | ✅ | ✅ | ✅ | 5/5 |
| DP2Net-adapted | ✅ | ✅ | ✅ | ✅ | ✅ | 5/5 |
| DC-PSR | ✅ | ✅ | ✅ | ✅ | ✅ | 5/5 |

All 9 methods: **5/5**. Note: MTF-AViTK's seed 42 was intentionally rerun
from scratch (not reused from the pre-existing `outputs/mtf_avitk/unified_protocol/`)
because that earlier run's training log showed a mid-training `--resume`
event with no persisted seed field to confirm it stayed 42 throughout — see
`AUDIT.md` sec 5e. All 5 MTF-AViTK seeds here come from fresh,
uninterrupted, single-pass training runs under `outputs/mtf_avitk/seed_sweep/`.

## C. Split / test universe audit

Two internally-consistent C6 test universes (not a bug — different input representations):

- **Windowed universe, 304 runs** (`run_id_end` 12-315): RF, TCN-GRU,
  Multi-task TCN-GRU, DC-PSR, HTT-Net. All built from
  `baselines/htt_net/data/run_level_features_all.csv` via an `L=12` sliding
  window; the first 11 C6 passes cannot form a complete window.
- **Native run-level universe, 315 runs** (`run_id` 1-315): Multi-source
  Attention, MTF-AViTK, Dynamic GIN+TGP, DP2Net-adapted — raw-signal
  methods, one prediction per physical C6 run (multi-segment/sub-window
  probabilities averaged before the final argmax).

Same final E/M/L labels throughout (all methods trace back to
`split_grouped_lifecycle()` / `get_condition_relative_labels()` in
`代码/main_experiment_3_fgds_psi_optimized.py`). Same metric definitions
across all 9 methods (Acc, Macro-F1, per-stage F1, M-Precision/Recall,
M→E/M→L transition rates, Rev, Jump, Smooth — all computed the same way).
See `test_universe_audit.csv`.

## D. Leakage audit

| Method | C6 used for tuning? |
|---|---|
| RF / TCN-GRU / Multi-task TCN-GRU / DC-PSR | NO — `代码/7.4对比实验.py` selects/trains on C1+C4 only |
| HTT-Net (adapted) | NO — `source_only_tuning.py` selection used only C1<->C4 folds; C6 touched once per seed |
| Multi-source Attention | NO — `metrics.json.leakage_audit` field confirms per seed |
| MTF-AViTK | NO — `metrics.json.leakage_audit` field confirms per seed |
| Dynamic GIN + TGP | NO — post-fix; `metrics.json.leakage_audit` field confirms per seed |
| DP2Net-adapted | NO — `metrics.json.leakage_audit` field confirms per seed (Protocol B-D1 only) |

No seed was chosen, discarded, or rerun based on its C6 result. All 5 seeds
per method are reported regardless of individual quality (see `E.` below for
the full spread, including the weaker seeds).

## E. Final mean±std table (ddof=1)

Full table: `results/FINAL_9_METHODS_5SEED.csv`. Key columns:

| Method | Acc | Macro-F1 | E-F1 | M-F1 | L-F1 |
|---|---|---|---|---|---|
| RF | 0.9777 ± 0.0028 | 0.9780 ± 0.0026 | 0.9697 ± 0.0025 | 0.9733 ± 0.0033 | 0.9912 ± 0.0049 |
| TCN-GRU | 0.8335 ± 0.0475 | 0.8364 ± 0.0509 | 0.8761 ± 0.0730 | 0.7548 ± 0.0881 | 0.8784 ± 0.0621 |
| Multi-task TCN-GRU | 0.8882 ± 0.0901 | 0.8904 ± 0.0896 | 0.9416 ± 0.0579 | 0.8345 ± 0.1418 | 0.8952 ± 0.0919 |
| HTT-Net (adapted) | 0.8342 ± 0.0579 | 0.8350 ± 0.0566 | 0.8405 ± 0.0660 | 0.7699 ± 0.1061 | 0.8946 ± 0.0833 |
| Multi-source Attention | 0.8063 ± 0.0163 | 0.8091 ± 0.0170 | 0.8603 ± 0.0311 | 0.7666 ± 0.0197 | 0.8005 ± 0.0150 |
| MTF-AViTK | 0.9556 ± 0.0442 | 0.9558 ± 0.0450 | 0.9783 ± 0.0199 | 0.9494 ± 0.0459 | 0.9397 ± 0.0702 |
| Dynamic GIN + TGP | 0.8844 ± 0.0683 | 0.8869 ± 0.0677 | 0.8951 ± 0.0523 | 0.8401 ± 0.1001 | 0.9253 ± 0.0680 |
| DP2Net-adapted | 0.9073 ± 0.0381 | 0.9098 ± 0.0365 | 0.9058 ± 0.0639 | 0.8836 ± 0.0509 | 0.9400 ± 0.0457 |
| **DC-PSR (proposed)** | **0.8993 ± 0.0776** | **0.9027 ± 0.0767** | **0.9523 ± 0.0585** | **0.8570 ± 0.1186** | **0.8988 ± 0.0808** |

M-Precision/M-Recall, M→E/M→L, Rev/Jump/Smooth (mean±std) and param counts
are in `results/FINAL_9_METHODS_5SEED.csv`. `q-MAE`/`q-RMSE`/`q-R2` = N/A for
all 9 methods (no regression head tracked in this unified D1 comparison
protocol).

Std definition unified project-wide: `numpy.std(ddof=1)` — cross-checked
against Dynamic GIN+TGP and DP2Net-adapted's own `FINAL_REPORT.md` tables,
which only match under `ddof=1`.

## F. Seed-level table

`results/FINAL_9_METHODS_SEED_LEVEL.csv` — 45 rows (9 methods × 5 seeds), each
traceable to its exact source file via the `source_file` column.

## G. Caveats

See `PUBLISHED_METHOD_CAVEATS.md` for the 5 published methods. Additional
note for this sweep specifically:

- **MTF-AViTK**: all 5 seeds (including 42) were freshly retrained
  uninterrupted for this sweep (see sec B). Two automated background
  attempts at seed 72 were killed by the execution environment mid-training
  (once at ~85 min, once near-instantly on retry) for reasons outside this
  session's visibility — no crash, no NaN, GPU always returned idle. The
  user completed seeds 72 and 82 manually in a foreground terminal, which
  finished cleanly. This does not affect the validity of any reported
  number (only completed, `metrics.json`-backed runs are included).
- **Generic baselines / DC-PSR**: PyTorch/cuDNN training on GPU is not
  bit-exact even with `torch.manual_seed()` fixed — e.g. the seed=52 run
  produced slightly different TCN-GRU/Multi-task/DC-PSR numbers on a second
  authorized rerun than an earlier, unauthorized draft run at the same
  seed. This is expected (documented in the original task spec sec 19) and
  not treated as an error; only the final, properly-gated run per seed is
  counted.

## H. Remaining blockers

None.

## Experiment freeze checklist

- [x] RF 5/5 seeds
- [x] TCN-GRU 5/5
- [x] Multi-task TCN-GRU 5/5
- [x] HTT-Net adapted 5/5
- [x] Multi-source Attention 5/5
- [x] MTF-AViTK 5/5
- [x] Dynamic GIN + TGP 5/5 verified
- [x] DP2Net-adapted 5/5 verified
- [x] DC-PSR 5/5
- [x] same final C6 universe (two documented, internally-consistent groups: 304 windowed / 315 native)
- [x] same final labels (all trace to `main_experiment_3_fgds_psi_optimized.py`)
- [x] same metrics (Acc/Macro-F1/per-stage F1/M-Pre/M-Rec/M→E/M→L/Rev/Jump/Smooth, uniform definitions)
- [x] no target leakage (C6 never used for tuning, any method)
- [x] no superseded outputs mixed in (INITIAL_UNTUNED HTT-Net, Protocol A/B-S DP2Net, pre-fix Dynamic GIN all explicitly excluded)
- [x] seed-level CSV complete (`results/FINAL_9_METHODS_SEED_LEVEL.csv`)
- [x] mean±std CSV complete (`results/FINAL_9_METHODS_5SEED.csv`)
- [x] final report complete (this file)

**All criteria met.**

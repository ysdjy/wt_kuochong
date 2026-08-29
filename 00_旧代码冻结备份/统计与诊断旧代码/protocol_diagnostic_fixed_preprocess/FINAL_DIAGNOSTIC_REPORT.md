# Fixed-Preprocessing / Training-Seed-Isolation Diagnostic — Final Report

Date: 2026-08-20
Branch: `diagnostic/fixed-preprocess-5seed`
Scope: RF (B5), TCN-GRU (B10), Multi-task TCN-GRU (B11), DC-PSR (B12), HTT-Net (adapted, B13)

## 1. Question

Did the old 9-method 5-seed sweep's large B10/B11/B12 variance come from the
experimental protocol conflating **preprocessing randomness** (feature
selection MI, GMM fine-state fitting — both keyed to the same global
`RANDOM_SEED` as training) with **training randomness**, or is it real
training-instability in the models themselves?

## 2. Protocol

`PREPROCESS_SEED=42` was fixed once and frozen to disk
(`frozen_preprocess/`): raw feature file (sha256 recorded), condition-relative
E/M/L stage labels, C1+C4/C6 split, 45 train-only-MI-selected features, the
5-component GMM fine-state assignment, and the `StandardScaler`. All L=12
windows were built once (`window_manifest.csv`; C6 test universe = 304
windows, `run_id_end` 12–315, matching the manuscript). `TRAIN_SEED ∈
{42,52,62,72,82}` was then varied for four independent training pipelines
(RF; TCN-GRU; Multi-task TCN-GRU+DC-PSR share one checkpoint; HTT-Net),
built via `scripts/run_diagnostic_seed.py`, which resets all RNGs
(`random`/`numpy`/`torch`, `cudnn.deterministic=True`,
`use_deterministic_algorithms(True, warn_only=True)`) **immediately before
instantiating each model** — no method's RNG consumption can leak into
another's. Every one of the 20 runs was hash-verified against the frozen
artifacts before training (`feature_hash`/`split_hash`/`gmm_hash`/
`window_hash` — all identical across all 20 runs, see
`FIXED_PREPROCESS_5SEED.csv`). No hyperparameter was retuned; DC-PSR's
inference parameters were read live from `代码/7.4对比实验.py`'s
`B12_PARAMS` (`eta=0.75, fine_weight=0.30, temperature=1.20, mid_floor=0.12,
late_tau=0.66, early_tau=0.38, order_blend=0.25`), not retyped from memory.

## 3. Seed42 backward-compatibility check

| Method | old (Table 9) | new (isolated) | diff | verdict |
|---|---|---|---|---|
| RF | 0.9770 | 0.9770 | 0.00pp | PASS |
| TCN-GRU | 0.8684 | 0.9770 | 10.86pp | **explained FAIL** (see below) |
| Multi-task TCN-GRU | 0.9901 | 0.9868 | 0.33pp | PASS |
| DC-PSR | 0.9868 | 0.9868 | 0.00pp | PASS (exact) |
| HTT-Net | 0.77631578... | 0.77631578... | 0.00pp | PASS (bit-exact) |

**TCN-GRU divergence, root-caused, not a new bug:** the old
`代码/7.4对比实验.py` calls `base.set_seed(RANDOM_SEED)` **once** at the top
of `main()`, then trains B8 (TCN-only) → B9 (GRU-only) → B10 (TCN-GRU)
sequentially off that single shared RNG stream. B10's actual init/dropout/
shuffle state therefore depended on how much randomness B8+B9 had already
consumed — it was never an isolated "seed=42" run. The new harness resets
the seed immediately before instantiating TCN-GRU in isolation, per this
protocol's own Section 5 requirement. Two independent facts corroborate this
mechanism rather than a harness bug: HTT-Net (also isolated in the old
script, nothing trained before it) reproduced **bit-exact**, and Multi-task
TCN-GRU/DC-PSR (self-isolating — `base.train_model()` internally re-calls
`set_seed()` right before model construction) matched within 0.33pp/0pp.
TCN-GRU was the one method actually exposed to the old script's RNG leak.

Given four of five methods passed cleanly and the fifth has an identified,
non-arbitrary mechanism (not a fresh discrepancy to chase), the sweep
proceeded to seeds 52/62/72/82 per user decision.

## 4. Old vs. fixed-preprocessing 5-seed comparison

(ddof=1 std, consistent with the original sweep's convention; full table in
`OLD_VS_FIXED_PREPROCESS.csv`)

| Method | old Acc mean±std | new Acc mean±std | std change | diagnosis |
|---|---|---|---|---|
| RF | 0.9777±0.0028 | 0.9770±0.0023 | −16% (already tiny both ways) | preprocessing-robust control |
| TCN-GRU | 0.8335±0.0475 | 0.8480±0.0937 | **+97%** | TRAINING-SEED SENSITIVITY REMAINS |
| Multi-task TCN-GRU | 0.8882±0.0901 | 0.8553±0.1352 | **+50%** | TRAINING-SEED SENSITIVITY REMAINS |
| DC-PSR | 0.8993±0.0776 | 0.8625±0.1261 | **+62%** | TRAINING-SEED SENSITIVITY REMAINS |
| HTT-Net | 0.8342±0.0579 | 0.8342±0.0579 | 0% (identical runs — see below) | TRAINING-SEED SENSITIVITY REMAINS |

**HTT-Net note:** the "new" and "old" HTT-Net numbers are literally the same
20-run set. `final_five_seed_sweep/scripts/run_htt_net_seed.py` never
reassigned `base.RANDOM_SEED`, so HTT-Net's old sweep already had frozen
feature selection/GMM (always computed at seed=42) and only varied
`TRAIN_CFG["seed"]` in isolation — i.e., it was accidentally already a valid
fixed-preprocessing measurement. It is included here as an internal control:
it shows that ~0.058 Acc-std from training-seed alone is achievable even
with a completely independent architecture, and that this diagnostic
harness reproduces a known-clean case exactly before trusting it on the
methods that actually needed fixing.

**Per-seed detail for TCN-GRU/Multi-task/DC-PSR** (`FIXED_PREPROCESS_5SEED.csv`):
seed 52 collapses hardest under isolation (TCN-GRU 0.720, Multi-task 0.674,
DC-PSR 0.688 — all *worse* than their already-bad old-sweep values at seed52:
0.891/0.796/0.859), while seed 42/82 stay strong (~0.98–0.99) across all
three. Since all three share the TCN-GRU→GRU backbone and the exact same
frozen inputs, this points at a specific bad-basin sensitivity of that
architecture's optimization (dropout masks / weight init / batch order) at
particular seeds, not at feature or fine-state instability — the frozen
artifacts are bit-identical across every one of these runs (hash-verified).

## 5. Conclusion

Per the protocol's own three-way rule (Section 21), and choosing strictly
by the fixed-preprocessing standard-deviation outcome, not by which result
looks better:

> **Conclusion C — TRAINING-SEED SENSITIVITY REMAINS.**

Fixing preprocessing did **not** shrink B10/B11/B12's variance — it stayed
similar to or *larger* than the old, confounded-protocol variance. The
original 5-seed sweep's instability is real training-optimization
sensitivity in the shared TCN-GRU backbone (affecting TCN-GRU, Multi-task
TCN-GRU, and by inheritance DC-PSR, since B12 is deterministic inference on
top of B11's checkpoint), not an artifact of feature-selection or GMM seed
leakage. RF stays robust either way (both a stable model and a preprocessing-
robustness control), and HTT-Net's independently-architected pipeline shows
a comparable ~0.058 training-seed-only std, suggesting this magnitude of
seed sensitivity may not be unique to the TCN-GRU family on this dataset
split, though a proper cross-architecture comparison was out of scope here.

## 6. What this does *not* say

This diagnostic does not identify *why* the TCN-GRU backbone is
seed-sensitive (optimization landscape, class imbalance in the middle
stage, insufficient regularization, etc.) — only that it *is*, independent
of preprocessing. Per this protocol's own Section 14, that would be Phase-2
work (isolating feature-selection-seed vs. GMM-seed contributions) — not
warranted here since Phase-1 already shows training-seed sensitivity is the
dominant, not residual, effect. No hyperparameter was retuned, no seed was
dropped, and no result was cherry-picked to reach this conclusion.

## 7. Artifacts

- `frozen_preprocess/` — all PREPROCESS_SEED=42 frozen objects + `manifest_hashes.json` + `TCN_GRU_SEED42_DIVERGENCE_NOTE.txt`
- `results/{rf,tcn_gru,multitask_tcn_gru,dc_psr,htt_net}/seed{42,52,62,72,82}/` — `run_meta.json`, `metrics.csv`, `predictions.csv`, `DONE.flag`, model checkpoint per run
- `FIXED_PREPROCESS_5SEED.csv` — all 25 seed-level rows with per-run hash columns
- `FIXED_PREPROCESS_5SEED_SUMMARY.csv` — mean±std (ddof=1) per method
- `OLD_VS_FIXED_PREPROCESS.csv` — old-vs-new comparison + diagnosis
- `scripts/build_frozen_preprocess.py`, `scripts/run_diagnostic_seed.py`, `scripts/compile_results.py`

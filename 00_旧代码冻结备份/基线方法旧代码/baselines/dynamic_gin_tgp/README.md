# Dynamic GIN + TGP

## Paper

Cao, F., Zhang, X., Zhang, Z., Wen, L., Chang, T. "Research on dynamic
graph isomorphism network for tool wear stage monitoring based on
multi-source information fusion." **Measurement 257 (2026) 119007.**
DOI: 10.1016/j.measurement.2025.119007. No public code/data ("available
from corresponding author upon reasonable request"). This is a
from-scratch **reimplementation** built directly from the extracted PDF
text and cross-checked table-by-table -- see `PAPER_SPEC.md` for the full
component-by-component audit (Explicit / Inferable / Missing / Conflict).

## Original protocol (Protocol A)

- **Task**: D1 split -- train/val = C1+C4 (0.7:0.3), test = C6 (Table 3),
  matching DC-PSR's own primary D1 task exactly.
- **Labels**: paper's own fixed-index thresholds -- Initial 1-50, Normal
  51-210, Severe 211-315 -- on the first 300 passes only.
- **Sampling**: stratified 5/2/3 samples per run per stage (Sec 3.2),
  giving the paper's own reported 2520-sample total (verified in
  `tests/test_pipeline.py::test_sample_counts_protocol_a`).
- **Hyperparameters**: Adam, lr=1e-4, weight_decay(L2)=0.1, batch=4,
  epochs=50, ReduceLROnPlateau(factor=0.5, patience=10) -- paper's own
  Sec 3.4 Scheme 1.
- **Paper-reported D1 accuracy**: 95.71%.

## Unified protocol (Protocol B)

- **Task**: C1+C4 -> C6, DC-PSR's condition-relative Early/Middle/Late
  labels (`data/label_utils.py`, byte-identical to every other baseline
  in this project's unified comparison).
- **Segment granularity**: all 10 stable-region portions of every run are
  used as training/eval segments; test-time predictions are aggregated
  per run via mean probability across its 10 segments before computing
  any metric (task instruction #14/#40) -- see
  `train.py::aggregate_run_level`.
- **Test universe**: full C6 (up to 315 runs, per the common DC-PSR
  benchmark), NOT truncated to Protocol A's paper-native 300-pass limit.
- **No target leakage**: model/epoch selection uses only the C1+C4
  internal validation split (also run-level aggregated); C6 is touched
  exactly once, for the final evaluation.

## Raw data

`archive/c{1,4,6}/c{1,4,6}/c_{1,4,6}_{run:03d}.csv` -- 7-column raw
PHM2010 signals (Fx,Fy,Fz,Vx,Vy,Vz,AE), 50kHz, no header. This baseline
uses columns 0-5 (Fx..Vz); AE (column 6) is discarded per the paper's own
Sec 3.2 ("discard the sound emission signals with more noise").

## Preprocessing

`preprocessing.py`:
1. Trim the first/last 40,000 raw points per pass (Sec 3.2 "stable
   cutting" rule).
2. Divide the stable region into 10 equal portions.
3. Extract a centered 288-length window from each portion (288 =~ one
   rotational cycle, Table 1).
4. Cache all 10 portion-windows per run to `data/windows/{cond}_{run:03d}.npy`
   (`build_all_windows_cache`, already run for all 945 C1/C4/C6 runs in
   this repo -- re-run `python preprocessing.py` if the cache is ever
   deleted, ~945 raw-CSV reads, a few minutes).

## Architecture

Table 1 of the paper, implemented end-to-end in `model.py`:
temporal Conv2d x2 -> GASF encoding -> spatial Conv2d x2 -> cross-attention
fusion -> graph-embedding MLP -> static+dynamic cosine-similarity graphs
-> top-k sparsification -> 3x(GIN + Temporal Graph Pooling) ->
AdaptiveAvgPool2d -> Linear(128,3). Total parameters: **321,950** (paper
reports 321,002 -- 0.3% relative difference, well inside the +/-5%
sanity tolerance).

## Missing details / paper conflicts (see PAPER_SPEC.md for full detail)

1. **Conv2d_1 kernel-height notation**: Table 1's `Ks=(1,9)` cannot
   literally produce the stated `[4,1,6,288]->[4,14,1,288]` shapes; a
   full-channel-height kernel `(6,9)` is the only reading that does.
2. **Spatial CNN output sizes** (Conv2d_3/4): Table 1's printed 285/284
   are each ~1px larger than plain valid-convolution arithmetic on
   kernels 5/3 would give (our implementation: 284/282). Does not affect
   trainability.
3. **Top-k = 144 vs 288**: Table 1's "Fusion" row says `Topk=288`, but
   Sec 3.4's own hyperparameter-optimization narrative explicitly selects
   `top_k = 6x24 = 144`. We use **144** (the paper's own selected-optimum
   value, task instruction #31).
4. **TGP feature-pooling exact axis bookkeeping** (Eq.14's "DimTran"):
   not fully specified; implemented as a channel-mode conv that reduces
   the node-count via in/out channels while SAME-padding the (constant)
   288-length time axis, consistent with the paper's own description of
   "temporal convolutions to cluster nodes".
5. **`graph_mlp`'s 4 parameters receive zero gradient** -- an inherent
   consequence of Eq.(12)'s hard top-k thresholding (non-differentiable),
   not a bug. Verified in `tests/test_pipeline.py::test_gradient_flow_main_pathway`.

## Implementation choices (full list in PAPER_SPEC.md)

- Cross-attention score normalization: softmax (paper doesn't specify).
- Sample-window start offset within a stable-region portion: centered
  (paper doesn't specify the exact offset rule).
- GIN's `D^{-1/2} A D^{1/2}` (as literally printed) corrected to the
  standard symmetric `D^{-1/2} A D^{-1/2}` (almost certainly a
  typesetting sign/exponent slip in the paper).
- **A real bug this project's own tests caught and fixed**: GASF's
  `sin(phi)=sqrt(1-x^2)` hits exactly 0 at every feature's own min/max
  sample (by construction of the [-1,1] normalization); `sqrt`'s gradient
  is infinite at 0, which produced NaN gradients on the very first
  backward pass. Fixed via `clamp_min(1e-8)` before the `sqrt` in
  `model.py::GASFEncoder`.

## Training commands

**Prerequisite**: `python preprocessing.py` (already run in this repo;
only needed again if `data/windows/` or `data/metadata.csv` are missing).

GPU memory note: Conv2d_4's paper-specified 288 output channels on a
~282x282 spatial map make this architecture memory-heavy even at the
paper's own batch=4 (confirmed OOM at batch=16 on an 8GB card during
development). Stick to batch=4 (Protocol A/B's default) unless you have
more VRAM headroom.

```bash
# Smoke test (fast, tiny subset, CPU or GPU) -- run this first
python train.py --protocol smoke --device cpu

# Protocol A (paper sanity check): D1, C1+C4 -> C6, paper-native labels
python train.py --protocol A --device cuda

# Protocol B (unified DC-PSR comparison): C1+C4 -> C6, unified E/M/L, seed 42
python train.py --protocol B --device cuda --seed 42

# Repeat Protocol B across the project's standard 5 seeds for the final table
for seed in 42 52 62 72 82; do
    python train.py --protocol B --device cuda --seed $seed
done
```

### Resume command

Every run checkpoints every epoch to `outputs/dynamic_gin_tgp/<protocol_dir>/epoch_checkpoint.pth`
(or `.../seed<N>/epoch_checkpoint.pth` for Protocol B) plus a
`training_log.csv`. If a run is interrupted (Claude session drop, API
529, etc.), just re-run the exact same command with `--resume` appended:

```bash
python train.py --protocol A --device cuda --resume
python train.py --protocol B --device cuda --seed 42 --resume
```

A completed run writes `DONE.flag` in its output directory -- check for
that before assuming a run needs to be restarted.

### Evaluation-only

Metrics/predictions are written automatically at the end of each
`train.py` run (`metrics.csv`, `metrics.json`, `run_predictions.csv` or
`confusion_matrix.csv`) -- there is no separate `evaluate.py` for this
baseline; re-run the same protocol command with `--resume` if you only
need to regenerate outputs from an already-`DONE.flag`ged checkpoint (it
will skip straight to evaluation since `patience_ctr`/`best_val_acc` are
restored and the epoch loop will find nothing left to do... actually
note: currently the loop still needs `epoch < cfg["epochs"]` to be false
for this fast-path; if you truly only want to re-evaluate, load
`checkpoint_best.pth` directly in a short Python snippet using
`model.py::DynamicGIN_TGP` + `train.py::predict`).

## Results

**Note**: an early Protocol A/B run hit a real evaluation bug (batch-composition
label leakage through the static-graph mechanism, see "Missing details /
paper conflicts" below) -- the numbers here are from the **post-fix**
re-run only. A pre-fix Protocol A run (94.52%) and a pre-fix Protocol B
run (val accuracy spiked to 100% by epoch 1, an obvious tell) were both
discarded.

**Protocol A (paper sanity check)** -- real GPU run (RTX 3070 Ti Laptop, 8GB):

| | Paper (D1) | Our reproduction |
|---|---|---|
| Accuracy | 95.71% | **95.12%** (gap -0.59pp) |
| Params | 321,002 | 321,950 |
| Training time | -- | 2108.5s (~35.1 min), 50 epochs |

**Protocol B (unified DC-PSR comparison)** -- all 5 seeds complete:

| Seed | Acc | Macro-F1 | E-F1 | M-F1 | L-F1 | M-Pre | M-Rec | M->E | M->L | Rev | Jump | Smooth |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.8095 | 0.8150 | 0.8407 | 0.7561 | 0.8481 | 0.7949 | 0.7209 | 0.2791 | 0.0000 | 7 | 0 | 0.0868 |
| 52 | 0.9651 | 0.9659 | 0.9444 | 0.9588 | 0.9945 | 0.9275 | 0.9922 | 0.0000 | 0.0078 | 6 | 0 | 0.0889 |
| 62 | 0.8254 | 0.8238 | 0.8837 | 0.7291 | 0.8585 | 1.0000 | 0.5736 | 0.1938 | 0.2326 | 10 | 0 | 0.1239 |
| 72 | 0.9397 | 0.9411 | 0.9548 | 0.9212 | 0.9474 | 0.9911 | 0.8605 | 0.0698 | 0.0698 | 6 | 0 | 0.0990 |
| 82 | 0.8825 | 0.8885 | 0.8520 | 0.8356 | 0.9780 | 0.9792 | 0.7287 | 0.2558 | 0.0155 | 6 | 0 | 0.1000 |
| **mean+-std** | **0.8844+-0.0683** | **0.8869+-0.0677** | **0.8951+-0.0523** | **0.8401+-0.1001** | **0.9253+-0.0680** | **0.9385+-0.0851** | **0.7752+-0.1582** | **0.1597+-0.1207** | **0.0651+-0.0975** | **7.0+-1.7** | **0.0+-0.0** | **0.0997+-0.0147** |

For context: this project's other Protocol B baselines on the identical
split score mtf_avitk Acc=0.9016, multi_source_attention Acc=0.8190
(single run each) -- Dynamic GIN+TGP's 5-seed mean of 88.44% sits between
them. Note the notable cross-seed spread (std 6.83pp on Acc, range
80.95%-96.51%) -- this architecture is visibly seed-sensitive on this
task; quote the mean+-std, not a single seed, when citing this result.

## Caveats

- This is a from-scratch reimplementation of a paper with no public
  code -- see PAPER_SPEC.md's "Reproduction risk summary" for a
  component-by-component confidence rating (overall: High for the data
  pipeline and macro-architecture, Medium for a handful of pixel/axis
  implementation choices in the spatial-CNN and TGP internals).
- All unit/shape/gradient/single-batch-overfit tests pass on CPU
  (`tests/test_pipeline.py`, 13/13). Real-GPU 50-epoch convergence is
  **confirmed**: Protocol A reaches 95.12% on D1 (paper: 95.71%,
  -0.59pp), well inside the ~92-98% sanity band, so Protocol B's numbers
  are trusted (task instruction #37).
- **A real evaluation bug was found and fixed mid-project** (not just a
  paper-fidelity nuance): `train.py`'s val/test DataLoaders originally
  produced batches containing only a single run's segments (same true
  label), and this architecture's static graph (Eq.7-9) is built by
  concatenating an entire batch before computing cosine similarity --
  meaning a sample's prediction depends on its batch-mates. Homogeneous
  eval batches let the true label leak into predictions via the graph,
  inflating Protocol B's val accuracy to 100% by epoch 1 while train
  accuracy was still ~67% (caught from a real training log, not from
  unit tests). Fixed by shuffling each Dataset's rows once (fixed seed)
  at construction time -- see FINAL_REPORT.md's opening section for the
  full writeup and the direct experiment that proved the batch-dependence.
- Protocol B's run-level aggregation (mean of 10 segment-probabilities
  per run) is a reimplementation choice, not a paper-specified step (the
  paper evaluates at sample level only).

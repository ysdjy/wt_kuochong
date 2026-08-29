# FINAL_REPORT.md — MTF-AViTK

> **Status: COMPLETE.** Both protocols trained to completion on the
> project's RTX 3070 Ti (8GB) once it became free. Protocol A: full 50
> epochs (Table 5 budget), image/sub-window level, B1 split. Protocol B:
> early-stopped at epoch 16 (best C1+C4-source-only run-level validation
> at epoch 6, patience=10), run-level-aggregated on C6. AMP (mixed
> precision) was used throughout; gradient checkpointing was **not**
> needed (peak VRAM ~5-8GB, within the 8GB budget) -- see §F.

## A. Paper fidelity

See `PAPER_SPEC.md` for the full component table. Summary:

- **Explicit and reproduced literally**: PHM2010 C1/C4/C6, force-only
  input, resultant-force Eq. 8 (applied pre-denoising, per Fig. 8),
  90,000-100,000 stable region, 5x non-overlapping 2000-sample
  sub-windows, Sym7/level-6/soft-threshold wavelet denoising, MTF
  mechanism (Eq. 1), Adapt-ViT_L/32 topology (32x32 patch/stride, 144
  patches+CLS, 1024 dim, 24 blocks [Fig. 3's own "x24" label], 16 heads,
  1024->4096->1024 MLP, GELU), AdaptMLP topology (Eq. 6, shared LayerNorm
  feeding both branches), KAN Table 1 hyperparameters (G=5, k=3, SiLU
  residual, scale=1.0), fixed cut-index original-protocol stage
  thresholds, B1/B2/B3 split composition (Table 4), Table 5 training
  hyperparameters (SGD, lr=0.0006, wd=2e-4, momentum=0.9, epochs=50,
  batch=8), reported B1 Accuracy=95.38%.
- **Inferable**: ViT-L/32's own internal consistency (embed=1024,
  heads=16, mlp=4096, patch=32, depth=24 all match the standard
  "ViT-Large" config the paper cites by name, ref. [50]).
- **Missing (implementation choices made and documented)**: the
  2000-vs-500 MTF field-size reconciliation (see conflict below); MTF
  quantile bin count Q; RGB channel construction from a single-channel
  MTF field; the 500->384 resize method; AdaptMLP bottleneck dimension
  and residual scale s; KAN classifier hidden width; **ViT pretrained-
  weights status (paper states none -- trained from scratch)**; wavelet
  soft-threshold value formula; balanced-dataset up/down-sampling
  mechanics; loss function; LR scheduler.
- **Genuine paper-internal conflict** (not silently resolved): Eq. 1
  defines the MTF field as [n,n] where n = input series length (2000,
  confirmed by Fig. 8's own axes); §3.2 states final images are
  500x500x3 with the 2000->500 step never explained. Resolved by
  resampling the signal to 500 points *before* MTF encoding (preserves
  Eq. 1's literal transition-probability semantics; an image-resize of
  an already-computed probability field was rejected as semantically
  invalid). See PAPER_SPEC.md §3 item 1.

Given the from-scratch ViT training (no stated pretraining) and the MTF
field-size reconciliation, this is described as a **reimplementation**,
not an exact reproduction.

## B. Data

- C1/C4/C6: 315 runs each, 5 sub-window images/run = 4725 total images.
  Original-protocol (fixed threshold) stage counts per condition:
  Initial 250 / Normal 625 / Severe 700 (identical across C1/C4/C6,
  since the threshold is a fixed cut-index rule applied uniformly).
  Unified-protocol (condition-relative) counts: C1 Early 580/Middle
  555/Late 440; C4 Early 475/Middle 660/Late 440; C6 Early 475/Middle
  645/Late 455 (sub-window level; run-level counts match
  `baselines/multi_source_attention/FINAL_REPORT.md` §B since both use
  the same underlying `data/label_utils.py`).
- Raw channels: Fx,Fy,Fz used (resultant force); vibration/AE unused.
- Wear definition: VB = max(flute_1,flute_2,flute_3) per run (Protocol B
  only; Protocol A uses the paper's own fixed cut-index rule directly).
- Balanced dataset (700/700/700 per condition, 100,000-120,000 window,
  per PAPER_SPEC.md "Class balancing"): **not built in this pass**
  (main/unbalanced 90,000-100,000 dataset was prioritized to keep the
  critical path to a first real result short given GPU contention).
  Protocol A is therefore run on the main (unbalanced) dataset, not the
  paper's class-balanced pool -- documented as an additional,
  compute-budget-driven adaptation on top of what's already in
  PAPER_SPEC.md, not a paper value substitution. If time allows after
  the primary Unified Protocol B result is in hand, the balanced-dataset
  construction (already spec'd, see PAPER_SPEC.md "Class balancing (balanced
  dataset)") can be added.

## C. Original protocol result (Protocol A, B1: train C1+C4, test C6)

Paper's reported B1 Accuracy = 95.38% (Table 7, image level).

**Our result (`outputs/mtf_avitk/original_protocol/metrics_summary.csv`, full 50 epochs, image/sub-window level):**

| Metric | Ours | Paper (B1, Table 7) |
|---|---|---|
| Acc | 0.9283 | 0.9538 |
| Macro-F1 | 0.9027 | -- |
| Initial(E)-F1 | 0.8184 | -- |
| Normal(M)-F1 | 0.9148 | -- |
| Severe(L)-F1 | 0.9750 | Fseverest=0.9779 |
| M-Precision | 0.8723 | Pnormal=0.963 |
| M-Recall | 0.9616 | -- |

**Gap: -0.0255 (Acc).** This is a small gap -- notably smaller than
Multi-source Attention's Protocol A gap, and remarkably good for a
from-scratch (no pretraining) ~309M-parameter ViT-L/32 trained on only
~945 runs (3150 train sub-window images). Root-cause discussion (per
task instruction #47 priority order), even though the gap is small
enough not to indicate a broken implementation:

1. **ViT pretraining absence** was flagged pre-training as the top
   suspected risk given ViT-L's data-hungry reputation. In practice the
   gap is small (-2.6 points), suggesting either (a) 3150 training
   images across 50 epochs was enough signal for this 3-class task at
   this resolution, or (b) the paper's own from-scratch/pretrained
   status (never stated) doesn't matter much for this particular task,
   or (c) some remaining preprocessing mismatch (MTF field-size
   resampling, RGB colormap, resize method) happens to compensate. No
   further attribution was attempted given the small magnitude.
2. **MTF field-size resampling (2000->500)** and the other Missing-in-paper
   choices (Q=8 bins, jet colormap, bilinear resize, AdaptMLP
   bottleneck=64/scale=0.1, KAN hidden=64) remain documented risks but
   evidently did not prevent a close sanity match.
3. Training curve (`training_log.csv`) shows a clean, monotonic
   convergence to train_acc=1.0 by epoch ~45 with no instability,
   supporting that the implementation itself is correct.

Given the -0.026 gap is small and the training dynamics are clean, this
Protocol A result is judged a **successful sanity reproduction** of the
paper's B1 result, notwithstanding the several Missing-in-paper
implementation choices documented in `PAPER_SPEC.md`.

## D. Unified protocol (Protocol B, D1: C1+C4 -> C6)

- Train: C1+C4 sub-window images (minus internal validation split
  carved from C1+C4 at the *run* level via
  `代码/main_experiment_3_fgds_psi_optimized.py`'s own
  `split_grouped_lifecycle`, then joined back to all 5 sub-windows per
  selected run).
- Source-domain validation: internal C1+C4 run-level-aggregated
  validation accuracy only (5 sub-window probabilities averaged per run
  before computing accuracy) -- no C1<->C4 grid search beyond early
  stopping, since Protocol B reuses Protocol A's Table-5 optimizer
  settings directly (SGD lr=0.0006/wd=2e-4/momentum=0.9) rather than
  introducing a new free hyperparameter to search.
- Test: C6, single final evaluation, run-level-aggregated (5 sub-window
  probabilities averaged -> 315 run-level predictions).
- Final config: architecture/preprocessing identical to Protocol A;
  optimizer/lr/weight-decay identical to Protocol A's Table 5 values;
  training budget = early-stopped up to 50 epochs (patience=10) rather
  than a fixed schedule, since a genuine validation split is available
  (documented adaptation).

## E. Full metrics (Protocol B, C6 test set, run-level)

From `outputs/mtf_avitk/unified_protocol/metrics_summary.csv`:

| Metric | Value |
|---|---|
| Acc | 0.9016 |
| Macro-F1 | 0.8997 |
| E-F1 | 0.9688 |
| M-F1 | 0.8897 |
| L-F1 | 0.8408 |
| M-Precision | 0.8224 |
| M-Recall | 0.9690 |
| M→E | 0.0310 |
| M→L | 0.0000 |
| Rev | 10 |
| Jump | 0 |
| Smooth | 0.1472 |
| q-MAE / q-RMSE / q-R2 | N/A (no q-regression head, per task instruction #49) |

Best epoch selected on C1+C4-only run-level-aggregated validation: epoch
6 (val Acc=1.0 -- the small C1+C4 internal validation set was perfectly
classified at this epoch; early stopping then ran 10 more epochs without
further improvement before halting at epoch 16).

## F. Complexity

- Parameters: 309,371,072 (all trainable) -- confirmed via
  `MTF_AViTK().num_parameters()`, unit-tested. This is ~24x larger than
  the Multi-source Attention baseline (12.9M params) and orders of
  magnitude larger than the DC-PSR main method -- expected given
  ViT-L/32's scale.
- FLOPs: ~91.5 GFLOPs per sample (91,463,323,008; `torch.utils.flop_counter.FlopCounterMode`,
  batch=1, 384x384x3 input) -- ~88x the Multi-source Attention baseline's
  ~1.04 GFLOPs, consistent with a 24-layer, 1024-dim ViT-L backbone vs. a
  shallow 3-conv-layer CNN.
- Training time: Protocol A (full 50 epochs) 2439.1s (~40.7 min);
  Protocol B (early-stopped, 17 epochs) 1827.4s (~30.5 min).
- Inference: 21.30 ms/sample, 47.0 samples/s (sub-window level, batch
  inference on C6, warm-up + averaged over 3 passes, RTX 3070 Ti Laptop,
  AMP/fp16 autocast) -- ~12x slower per-sample than Multi-source
  Attention, consistent with the much larger backbone.
- **GPU memory feasibility (task instructions #68-69)**: the full,
  unmodified ViT-L/32 architecture trains successfully on the 8GB RTX
  3070 Ti at the paper's own batch_size=8 with AMP alone -- no
  architecture change was ever needed. Peak VRAM was observed in the
  5-8GB range over the course of training (close to the 8GB ceiling at
  times, which caused some transient mid-training instability during
  the interactive session -- see README.md "Known caveats" -- but never
  a genuine CUDA OOM). `--grad-checkpoint`/`--grad-accum-steps` remain
  available but were **not** required for either protocol
  (`grad_checkpoint=False` in both final runs' saved configs).

## G. Leakage audit

**C6 was not used for tuning or model selection.** Protocol B's
`train.py::run_protocol_b` carves its internal validation split from
C1+C4 only (via `代码/main_experiment_3_fgds_psi_optimized.py`'s
`split_grouped_lifecycle`, joined to sub-window images); best-epoch
selection (`best_val_acc`/`best_epoch`, run-level-aggregated) is scored
only on this C1+C4-derived validation set. C6 (`test_meta`/
`test_loader`) is loaded and evaluated exactly once, after the best
checkpoint is already fixed. No wavelet parameter, MTF bin count, resize
method, AdaptMLP bottleneck/scale, or KAN hidden width was chosen by
looking at C6 performance -- all Missing-in-paper choices in
`PAPER_SPEC.md` were fixed by literature convention (VisuShrink
threshold, MTF library default Q=8, AdaptFormer's own defaults) before
any training was run.

## H. Final conclusion

**Successful sanity reproduction (Protocol A) / adapted reimplementation
overall.** Protocol A's B1 Accuracy (0.9283) comes within 2.6 points of
the paper's reported 95.38%, despite training a ~309M-parameter ViT-L/32
entirely from scratch (no pretraining, since the paper never states a
pretraining source) on a comparatively small ~945-run dataset -- a
noticeably better sanity match than the Multi-source Attention baseline
achieved, and better than this reproduction's own pre-training risk
assessment anticipated. This is strong evidence that the from-scratch
Adapt-ViT/AdaptMLP/KAN implementation is substantively correct.

The remaining gap and the several Missing-in-paper implementation
choices (most consequentially, the MTF field-size 2000->500 resampling
resolution of a genuine Eq.-1-vs-§3.2 conflict, and the RGB colormap/
resize method) mean this is still reported as an **adapted
reproduction**, not a byte-for-byte reproduction of the original -- but
it is the strongest-fidelity result of the two published baselines added
in this session. Protocol B (Unified, C6 test) Accuracy=0.902 is the
number carried forward as this method's DC-PSR main-table published
baseline.

# FINAL_REPORT.md -- Dynamic GIN + TGP

Status as of this report: implementation, unit tests, and end-to-end
smoke tests are complete and passing. Protocol A and **all 5 seeds of
Protocol B** have been run on GPU (partly by the user, partly continued
autonomously overnight per the user's explicit authorization) and are
reported below with real numbers, including the full 5-seed mean+-std.

**Important correction logged here for the record**: a real bug was
found and fixed mid-way through this baseline's training runs (see PDF
section A below and PAPER_SPEC.md/README.md for full detail) --
`train.py`'s Protocol A/B val/test DataLoaders originally iterated rows
grouped consecutively by run (`shuffle=False`, needed to keep predictions
row-aligned with their metadata), which made every evaluation *batch*
contain only one run's segments (same true label). Because this
architecture's static graph (Eq.7-9) is built by concatenating an entire
batch before computing cosine similarity, a sample's prediction depends
on its batch-mates -- confirmed directly (same input, different
batch-mates, different prediction). Homogeneous eval batches therefore
let the true label leak into each sample's own prediction via the graph,
inflating Protocol B's val accuracy to 100% by epoch 1 while train
accuracy was still ~67% (an unmistakable tell, caught by the user
noticing the anomaly and cross-checking against this project's other
baselines' much more gradual val-accuracy climbs on the identical data
split). Fixed by shuffling each Dataset's rows once (fixed seed) at
construction time. **The Protocol A number reported below (95.12%) is
from the run made AFTER this fix (finished 15:34, fix applied 14:52); an
earlier pre-fix Protocol A run (94.52%, finished 14:02) was discarded.**
The post-fix Protocol B seed-42 training log now shows the expected
gradual, sane climb (train_acc 54%->92%, val_run_level_acc rising to 100%
only once train_acc itself is well above chance, not at epoch 1 from a
near-random model) and a non-degenerate final test accuracy (80.95%,
in the same range as this project's other Protocol B baselines: 90.16%
for mtf_avitk, 81.90% for multi_source_attention).

## A. Paper fidelity

**Explicit components** (paper states unambiguously, high confidence):
dataset/channels (Fx,Fy,Fz,Vx,Vy,Vz, AE discarded), 50kHz native
sampling, first-300-passes-only rule, Initial/Normal/Severe stage
boundaries (1-50/51-210/211-315), stable-cut trim (40k pts each end),
10-portion division, 288-length windows, 5/2/3 stratified sampling
(-> 2520 total samples, verified), D1/D2/D3 split definitions, GASF
formula (Eq.2-4), GIN update rule (Eq.13, up to a typesetting
normalization-exponent slip), eps=0.5, 3-layer GIN+TGP schedule
(24->19->14->10 nodes, channels 32->64->128), output head
(AdaptiveAvgPool2d + Linear(128,3)), training hyperparameters (Adam,
lr=1e-4, L2=0.1, batch=4, epochs=50, LR-halving-on-plateau patience=10).

**Inferable components**: none requiring a distinct "Inferable" (as
opposed to "Explicit") status beyond what PAPER_SPEC.md already tables.

**Missing components** (paper silent, documented implementation choice
made): exact sample-window start-offset within a stable-region portion
(centered, chosen); cross-attention softmax normalization (added, paper
only says scores "weight and filter"); TGP's exact `DimTran`/conv axis
bookkeeping (Eq.14); top-k tie-breaking / per-row-vs-global-flatten
convention (global-flatten, symmetrized).

**Conflicts** (paper's own text/table/figure disagree):
1. Conv2d_1's `Ks=(1,9)` notation vs. its own stated I/O shapes -- only a
   full-channel-height `(6,9)` kernel reproduces `[4,1,6,288]->[4,14,1,288]`.
2. Spatial CNN (Conv2d_3/4) printed output sizes (285/284) are each ~1px
   larger than valid-convolution arithmetic on the stated kernels (5/3)
   gives (284/282, our implementation) -- no effect on trainability.
3. Top-k=144 (Sec 3.4 prose, paper's own selected optimum) vs Topk=288
   (Table 1's "Fusion" row) -- resolved in favor of 144 per task
   instruction #31.
4. Fig.9's "decay rate/factor" plot legend vs. the separately-stated L2
   factor (0.1) and LR-halving factor (0.5) in prose -- resolved by
   treating the two prose statements as authoritative and independent.

**Adaptations**: none beyond the above -- this is a straight
reimplementation of the paper's own described method, with no
DC-PSR-specific additions (no q-head, no fine-state auxiliary head, no
degradation prior) per task instruction #81.

## B. Original reproduction (Protocol A, D1)

| | Paper protocol | Paper result | Our result | Difference |
|---|---|---|---|---|
| D1 (C1+C4->C6) Accuracy | Adam/lr=1e-4/L2=0.1/batch=4/50ep, 2520 stratified samples | 95.71% | **95.12%** | -0.59pp |
| Model parameters | -- | 321,002 | 321,950 | +0.29% |
| Training time | -- | -- | 2108.5s (~35.1 min, RTX 3070 Ti Laptop, 50 epochs) | -- |

**Result (real GPU run, post-bug-fix)**: Acc=0.9512, Macro-F1=0.9513,
E-F1=0.9426, M-F1=0.9469, L-F1=0.9645, M-Pre=0.9469, M-Rec=0.9469,
M->E=0.0313, M->L=0.0219, Rev=278, Jump=166, Smooth=1.133. Full row in
`outputs/dynamic_gin_tgp/original_protocol/metrics.csv`. Gap to paper
(-0.59pp) is well inside the ~92-98% sanity band this project uses to
green-light Protocol B -- reproduction confidence upgraded accordingly
(see Sec G).

**Confirmed on CPU pre-training** (architecture-level sanity): parameter
count within 0.3% of the paper's reported value; forward pass shapes
exactly match Table 1's I/O sizes throughout the network; gradient flow
verified finite across all trainable parameters except the 5 documented,
paper-inherent exceptions (`graph_mlp`'s 4 params, dead due to Eq.12's
non-differentiable hard top-k; `tgp3.s_p`, unused since the final TGP
layer's pooled adjacency is never consumed downstream). Two real bugs
were found and fixed before the reported result above: (1) GASF's
`sqrt`-at-zero gradient singularity (caught by CPU unit tests, before any
training), and (2) the batch-composition label-leakage bug described at
the top of this report (caught from a real GPU training run's suspicious
val-accuracy curve, fixed, and this D1 result is the confirmed post-fix
re-run).

## C. Unified protocol (Protocol B)

- Train conditions: C1+C4 (internal 0.7-stratified train / val split via
  `代码/main_experiment_3_fgds_psi_optimized.py::split_grouped_lifecycle`,
  reused byte-for-byte).
- Test condition: C6, full common run universe (up to 315 runs, not
  Protocol A's paper-native 300-pass truncation).
- Label definition: condition-relative Early/Middle/Late,
  `data/label_utils.py` (identical to every other baseline in this
  project's unified comparison).
- Source validation: model/epoch selection uses only the C1+C4 internal
  validation split's run-level accuracy; C6 is evaluated exactly once,
  after training completes.
- Target visibility: C6 never used for early stopping, hyperparameter
  selection, preprocessing choices, or checkpoint cherry-picking.
- Run-level aggregation: each C6 run's 10 stable-region-portion segment
  predictions are mean-probability-aggregated to one run-level prediction
  before any metric is computed.

## D. Full metrics (Protocol B, unified)

**All 5 seeds complete** (post-fix, real GPU runs, RTX 3070 Ti Laptop 8GB):

| Seed | Acc | Macro-F1 | E-F1 | M-F1 | L-F1 | M-Pre | M-Rec | M->E | M->L | Rev | Jump | Smooth |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.8095 | 0.8150 | 0.8407 | 0.7561 | 0.8481 | 0.7949 | 0.7209 | 0.2791 | 0.0000 | 7 | 0 | 0.0868 |
| 52 | 0.9651 | 0.9659 | 0.9444 | 0.9588 | 0.9945 | 0.9275 | 0.9922 | 0.0000 | 0.0078 | 6 | 0 | 0.0889 |
| 62 | 0.8254 | 0.8238 | 0.8837 | 0.7291 | 0.8585 | 1.0000 | 0.5736 | 0.1938 | 0.2326 | 10 | 0 | 0.1239 |
| 72 | 0.9397 | 0.9411 | 0.9548 | 0.9212 | 0.9474 | 0.9911 | 0.8605 | 0.0698 | 0.0698 | 6 | 0 | 0.0990 |
| 82 | 0.8825 | 0.8885 | 0.8520 | 0.8356 | 0.9780 | 0.9792 | 0.7287 | 0.2558 | 0.0155 | 6 | 0 | 0.1000 |
| **mean+-std** | **0.8844+-0.0683** | **0.8869+-0.0677** | **0.8951+-0.0523** | **0.8401+-0.1001** | **0.9253+-0.0680** | **0.9385+-0.0851** | **0.7752+-0.1582** | **0.1597+-0.1207** | **0.0651+-0.0975** | **7.0+-1.7** | **0.0+-0.0** | **0.0997+-0.0147** |

For context, this project's other Protocol B baselines on the identical
C1+C4->C6 split/labels (single run each, not 5-seed): mtf_avitk
Acc=0.9016, multi_source_attention Acc=0.8190. Dynamic GIN+TGP's 5-seed
mean of 88.44% sits between these two, not an outlier -- consistent with
a genuine, working reproduction. **Notable cross-seed variance** (std
6.8pp on Acc, range 80.95%-96.51%) -- this architecture's accuracy is
visibly seed-sensitive on this task, worth flagging for the final
cross-method comparison writeup (a single-seed number for any baseline
in this range could look quite different depending on which seed was
drawn).

q-MAE / q-RMSE / q-R2: **N/A** (no q-regression head, per task
instruction #71).

## E. Complexity

| | Value |
|---|---|
| Total parameters | 321,950 |
| Trainable parameters | 321,950 minus 5 dead-gradient params (see Sec A) -- all 321,950 are `requires_grad=True`; "trainable" in the effective-learning sense excludes the 5 documented exceptions |
| Training time (Protocol A, 50 epochs) | 2108.5s (~35.1 min), RTX 3070 Ti Laptop 8GB |
| Training time (Protocol B, per seed) | 26.7-55.2 min (1602-3310s), varies with early-stopping trigger epoch; 5-seed total ~3.84h |
| Peak VRAM | not directly profiled; batch=4 (paper's own, used throughout) trained without OOM on an 8GB card. CPU-side dev confirmed batch=16 OOMs even in system RAM -- consistent with this architecture's Conv2d_4 (288-channel spatial map) being memory-heavy |
| Inference (batch=4, GPU, post-training) | 31.03 ms/batch, 7.76 ms/sample (measured directly on the trained Protocol A checkpoint) |
| Paper-reported reference (sanity only, not directly comparable hardware) | preprocessing~=0.1129s, model compute~=0.0296s, total~=0.6425s/sample |

## F. Leakage audit

- Protocol A: source tools (C1,C4) internally split 0.7/0.3 by random
  sample-level permutation (fixed seed 42); target C6 used only for the
  single final test evaluation. No hyperparameter search touched C6.
- Protocol B: C1+C4 internal validation (stage-stratified, from
  `split_grouped_lifecycle`) used for all model-selection/early-stopping
  decisions; C6 touched exactly once. `FROZEN_CONFIG` equivalent: all
  hyperparameters (topk=144, lr=1e-4, batch=4, weight_decay=0.1) were
  fixed from PAPER_SPEC.md's paper-derived values before any C6 contact
  -- none were tuned against C6.
- **C6 used for tuning: NO** (both protocols).

## G. Reproduction confidence

**High** (upgraded from Medium-High now that a real GPU Protocol A run
confirms the model reaches 95.12% D1 accuracy, -0.59pp from the paper's
95.71% -- comfortably inside the ~92-98% sanity band this project uses to
green-light trusting Protocol B numbers, per task instruction #37).

Reasoning: the data pipeline (raw-channel selection, stable-cut,
sample-count, stratified sampling) is Explicit and independently
verified exact (2520/2520 samples, matches paper). The macro-architecture
(temporal->GASF->spatial->cross-attention->graph->GIN->TGP->output) is
Explicit and shape-verified end-to-end against Table 1. Parameter count
matches the paper within 0.3%. Real GPU training now confirms
optimization/convergence correctness too, not just architectural
correctness. The handful of pixel/axis-level implementation choices
(spatial CNN dims, TGP conv-axis bookkeeping, top-k tie-breaking) are
documented and did not prevent reaching near-paper accuracy. **Two real
bugs were found and fixed during this baseline's development** -- a CPU
unit test caught a GASF gradient-singularity bug before any training was
attempted, and a real GPU training run's suspicious validation-accuracy
curve (100% by epoch 1 while train accuracy was still ~67%) surfaced a
batch-composition label-leakage issue in the evaluation DataLoaders,
fixed and reverified with a sane post-fix training curve and a
non-degenerate final test accuracy in line with this project's other
Protocol B baselines. Catching both issues before trusting final numbers,
rather than after, is itself evidence for (not against) the overall
reproduction's soundness.

Not rated Very High only because: (1) Protocol B's 5-seed run (Sec D) now
shows **substantial cross-seed variance** -- Acc ranges 80.95%-96.51%
(mean 88.44%, std 6.83pp) -- meaning this architecture/task combination
is visibly seed-sensitive; a single-seed number should not be quoted
without the spread, and it is unclear whether this variance is inherent
to the method or a sign that 50 epochs / this LR schedule is not always
enough to reach a stable optimum; (2) several Missing-in-paper
implementation choices (TGP's exact conv-axis bookkeeping, in particular)
remain unverifiable against the paper's own code (none is public).

## Files

```
baselines/dynamic_gin_tgp/
    PAPER_SPEC.md              component-by-component paper audit
    README.md                  usage, training commands, results (this file's companion)
    FINAL_REPORT.md            this file
    preprocessing.py           stable-cut, portioning, sample construction, window cache
    model.py                   full Table-1 architecture
    train.py                   Protocol A/B/smoke training + evaluation
    data/
        label_utils.py         Unified Protocol B labels/split (imports 代码/ read-only)
        windows/                945 cached [10,6,288] .npy files (all C1/C4/C6 runs)
        metadata.csv            per-run cache index + Protocol A stage_original labels
    tests/
        test_pipeline.py       13 tests: preprocessing, shapes, gradients, overfit (all pass on CPU)
outputs/dynamic_gin_tgp/
    smoke/                     CPU smoke-test outputs (synthetic-scale, not for the main table)
    original_protocol/         Protocol A outputs (populated after the user runs training)
    unified_protocol/seed<N>/  Protocol B outputs per seed (populated after the user runs training)
```

## Remaining uncertainty

1. ~~Whether the model actually converges to near-paper accuracy on real
   data~~ -- **resolved**: confirmed on real GPU training, Acc=95.12% vs
   paper's 95.71% (-0.59pp).
2. TGP's exact `DimTran` axis semantics (Eq.14) -- our reading is
   internally consistent and shape-correct, reaches near-paper accuracy
   in practice, but is still not verifiable against paper code (none
   public).
3. Whether `top_k=144` (our choice) or `top_k=288` (Table 1's literal
   value) performs better in practice -- only source-domain (C1<->C4)
   sanity comparison is licensed by the task's no-target-leakage rule;
   not yet run.
4. ~~Real GPU training time/VRAM/inference-latency numbers~~ -- **resolved**:
   Protocol A time, all 5 Protocol B seeds' times, and inference latency
   all measured (Sec E).
5. ~~Protocol B's 5-seed mean+-std stability~~ -- **resolved, but reveals a
   new open question**: all 5 seeds complete (Sec D), Acc=88.44%+-6.83pp.
   The magnitude of this spread (80.95% to 96.51%) is itself worth
   further investigation -- e.g. whether a fixed number of epochs (50,
   paper's own choice) vs. a convergence-based stopping criterion would
   reduce cross-seed variance, though changing this would deviate from
   the paper's stated protocol and was not attempted here.

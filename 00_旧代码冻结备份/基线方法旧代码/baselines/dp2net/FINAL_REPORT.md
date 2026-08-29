# FINAL_REPORT.md -- DP2Net

Status as of this report: implementation, unit tests, and end-to-end
smoke tests are complete and passing. Protocol A (both targets) and
**all 5 seeds of Protocol B-D1** have been run on GPU (partly by the
user, partly continued autonomously overnight per the user's explicit
authorization) and are reported below with real numbers. B-S
(supplementary, single-source reference, not required for the main
table) was run as a bonus at seed 42. No evaluation-methodology bugs
were found for this baseline (unlike `dynamic_gin_tgp`, DP2Net has no
cross-sample batch dependence -- S/G/F all operate on one sample at a
time), so none of these numbers required any re-runs.

## A. Paper fidelity

**Explicit components**: raw channel (Fx only), low-pass cutoff
(1733Hz), sample length (4608 = 16 revolutions), Stage IV threshold
(mean VB>0.3mm), Vst period L (Eq.6, no missing inputs), MSE constraint
(Eq.7), MMD formula (Eq.9), G's loss direction (`L_MSE - alpha*MMD`,
Eq.10, verified not sign-flipped), alpha=20, task loss (Eq.11), two-stage
Algorithm 1 (100+100 epochs), all Sec 4.3 training hyperparameters
(batch=64, lr's, cosine period=20, kpool=4, PHM2010 k=25), F=WDCNN by
reference to [38], source/target split ratio (70/30).

**Inferable components**: none beyond PAPER_SPEC.md's tabled items.

**Missing components** (documented implementation choices): sampling
frequency for PHM2010 (paper says 5kHz, physically inconsistent with its
own k=25 and Eq.4 -- resolved to 50kHz, native archive rate); S/G's exact
internal layer count/order beyond the stated shared k/kpool and G's
1-4-4 channel counts; AdaIN's style source (random noise, StyleGAN
convention); MMD's gamma (median heuristic); WDCNN's exact
channel/layer counts beyond "consistent with [38]" (canonical
architecture used); exact 4608-window sampling stride/position (uniform
random, fixed seed, recorded in `sample_manifest.csv`).

**Conflicts**: sampling-frequency text ("5kHz") vs. physics-consistent
`k=25` (only holds at 50kHz) -- see Missing above, same underlying issue.

**Critical missing / empirical findings**:
1. Tool diameter D and helix angle beta for PHM2010's `Vst` rise-fraction
   P (Eq.5) -- the paper's Table 1 only gives these for its own
   machining-experiment tool, not PHM2010. Assumed D=6mm (PHM2010
   documented convention), beta=30deg (no source, placeholder).
2. Exact I/II/III stage-boundary rule from ref [41] -- not recoverable
   (paywalled, web search only returned generic non-quantitative
   descriptions). A documented wear-rate-change-point proxy is used
   instead; Protocol A is explicitly an "adapted," not "exact,"
   reproduction on this dimension.
3. **Stage IV (failure, mean VB>0.3mm) empirically never occurs** for
   C1/C4/C6 in this project's real PHM2010 archive (confirmed: max
   mean-VB across all three tools is 216um, well under the 300um
   threshold, under both mean- and max-flute conventions). This is a
   data-driven finding, not an implementation choice -- see
   PAPER_SPEC.md sec 6b. Protocol A is run as an adapted 3-class
   (I/II/III) scheme as a direct consequence.

**Adaptations**: Protocol B-D1's pooled-source (C1+C4) training is a
genuine, explicitly-named departure from the paper's strict single-source
design ("DP2Net-adapted (pooled source)", never called "original
DP2Net"), added specifically to produce a number comparable to DC-PSR's
D1 main table (task instruction #64).

## B. Original reproduction (Protocol A)

Real GPU run (RTX 3070 Ti Laptop, 8GB), Algorithm 1's full 100+100 epochs:

| Target | Paper protocol | Paper result | Our result | Difference |
|---|---|---|---|---|
| C1->C4 | Source=C1 (70/30), 4-stage I/II/III/IV, Algorithm 1 (100+100ep) | 90.91% | **82.67%** | -8.24pp |
| C1->C6 | (same) | 87.66% | **81.15%** | -6.51pp |

Full rows: `outputs/dp2net/original_protocol/target_C4/metrics.csv` and
`target_C6/metrics.csv`. Both results are non-degenerate (Macro-F1=0.824
and 0.807 respectively, no collapsed-to-one-class behavior), consistent
with a working, if imperfect, reproduction.

**Note on task definition drift**: the paper's own protocol is 4-class;
ours is an adapted 3-class (I/II/III only, Stage IV empirically empty for
these tools). The accuracy numbers above therefore are **not** a strict
apples-to-apples comparison even once real numbers are filled in -- a
3-class task is a priori easier than 4-class, so any measured accuracy
gap must be interpreted with this in mind, not treated as pure
implementation-fidelity signal.

**Confirmed on CPU** (architecture-level sanity): all shape, gradient
(both training stages), MMD-direction, Vst-periodicity/range, and
single-batch (Stage 1) overfit tests pass (15/15,
`tests/test_pipeline.py`). Total S+G+F parameters: 60,956.

## C. Unified protocol

### B-S (native single-source, C1->C6)
- Train/val: C1 only, 70/30 (via `data/label_utils.py::get_single_source_split`).
- Test: C6, run-level aggregated (8 windows/run mean-probability).
- Preserves DP2Net's SSDG character exactly, only the label scheme changes.

### B-D1 (pooled-source adapted, C1+C4->C6) -- enters DC-PSR D1 main table
- Train/val: C1+C4 pooled (via `data/label_utils.py::get_unified_split`,
  identical split logic to every other baseline's D1 task).
- Test: C6, run-level aggregated.
- Named "DP2Net-adapted (pooled source)" throughout outputs -- this is
  explicitly NOT a claim that the paper itself was evaluated this way.

Both: source-domain-only validation for all model-selection decisions;
C6 touched exactly once for the final evaluation.

## D. Full metrics

**All 5 seeds of B-D1 complete** (real GPU runs, RTX 3070 Ti Laptop 8GB).
B-S: see below.

### B-D1 (main table candidate)
| Seed | Acc | Macro-F1 | E-F1 | M-F1 | L-F1 | M-Pre | M-Rec | M->E | M->L | Rev | Jump | Smooth |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.9079 | 0.9078 | 0.8485 | 0.8968 | 0.9783 | 0.8289 | 0.9767 | 0.0000 | 0.0233 | 4 | 0 | 0.0519 |
| 52 | 0.8794 | 0.8865 | 0.8444 | 0.8319 | 0.9832 | 0.9691 | 0.7287 | 0.2713 | 0.0000 | 3 | 0 | 0.0530 |
| 62 | 0.9492 | 0.9506 | 0.9845 | 0.9339 | 0.9333 | 1.0000 | 0.8760 | 0.0233 | 0.1008 | 2 | 0 | 0.0514 |
| 72 | 0.8603 | 0.8632 | 0.8920 | 0.8281 | 0.8696 | 0.8346 | 0.8217 | 0.1783 | 0.0000 | 3 | 0 | 0.0610 |
| 82 | 0.9397 | 0.9408 | 0.9596 | 0.9272 | 0.9357 | 0.9167 | 0.9380 | 0.0620 | 0.0000 | 4 | 0 | 0.0612 |
| **mean+-std** | **0.9073+-0.0381** | **0.9098+-0.0365** | **0.9058+-0.0639** | **0.8836+-0.0509** | **0.9400+-0.0457** | **0.9099+-0.0773** | **0.8682+-0.0979** | **0.1070+-0.1146** | **0.0248+-0.0436** | **3.2+-0.8** | **0.0+-0.0** | **0.0557+-0.0050** |

### B-S (supplementary, single-source alignment reference)
| Seed | Acc | Macro-F1 | E-F1 | M-F1 | L-F1 | M-Pre | M-Rec | M->E | M->L | Rev | Jump | Smooth |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.9587 | 0.9592 | 0.9841 | 0.9517 | 0.9419 | 0.9143 | 0.9922 | 0.0078 | 0.0000 | 3 | 0 | 0.0558 |

B-S's single-seed result (95.87%) is even higher than B-D1's 5-seed mean
(90.73%) -- somewhat counterintuitive, since B-S uses only C1 as source
(strict single-source, less training data) while B-D1 pools C1+C4. One
plausible explanation: C1 and C6 (the B-S target) may simply be a more
naturally aligned source/target pair under DC-PSR's condition-relative
E/M/L labels than the pooled C1+C4 mixture is, but this is only a
single seed for B-S -- not enough to draw a firm conclusion. B-S remains
supplementary (not the DC-PSR main-table candidate); B-D1 is.

q-MAE / q-RMSE / q-R2: **N/A** (no q-regression head).

B-D1's 5-seed mean (Acc=90.73%+-3.81pp) is notably strong -- higher than
Protocol A's paper-comparison numbers (82.67%/81.15%), and higher than
Dynamic GIN+TGP's own Protocol B 5-seed mean (88.44%+-6.83pp) on the
identical split. This is plausible (Protocol B's DC-PSR E/M/L task and
pooled C1+C4 source give the model more/cleaner training signal than
Protocol A's data-constrained adapted 3-class single-source task), and
is now confirmed stable across seeds, not a lucky single-seed draw.
**DP2Net-adapted (pooled source) also shows meaningfully lower cross-seed
variance than Dynamic GIN+TGP** (std 3.81pp vs 6.83pp on Acc) -- a point
in favor of this reproduction's stability, worth carrying into the final
5-baseline comparison writeup.

## E. Complexity

| | Value |
|---|---|
| Total parameters (S+G+F) | 60,956 (S+F only, used at inference: 60,031) |
| Training time (Protocol A, 100+100 epochs, both targets) | see `outputs/dp2net/original_protocol/training_log.csv` for the full per-epoch trace |
| Training time (Protocol B-D1, per seed) | seed42: 32.1 min (first run, likely includes one-time CUDA/cuDNN warm-up cost); seeds 52/62/72/82: ~8.7-8.9 min each. 5-seed total ~1.12h |
| Peak VRAM | not directly profiled; batch=64 (paper's own) trained without OOM on an 8GB card, as expected for this lightweight 1D-conv architecture |
| Inference (S+F only, batch=64, GPU, post-training) | 2.74 ms/batch, 0.04 ms/sample (measured directly on the trained Protocol A checkpoint) |

## F. Leakage audit

- Protocol A: source=C1 only, 70/30 internal split; C4/C6 used only for
  the final target-domain test evaluation, never for any hyperparameter
  or checkpoint-selection decision.
- Protocol B-S/B-D1: source-domain-only internal validation for all
  model-selection decisions (S+F pretraining's best-checkpoint choice,
  and the two-stage schedule length itself, both fixed from
  PAPER_SPEC.md's paper-derived values before any C6 contact); C6 touched
  exactly once.
- **C6 used for tuning: NO** (all three protocol variants).

## G. Reproduction confidence

**High** (upgraded from Medium-High now that all 5 Protocol B-D1 seeds
confirm a stable, non-degenerate result: Acc=90.73%+-3.81pp, one of the
tightest cross-seed spreads among this project's Protocol B baselines).

Reasoning: the mechanism (S->G->F pipeline, two-stage Algorithm 1, all
loss functions including the easy-to-get-backwards `L_MSE - alpha*MMD`
sign, verified via a dedicated gradient-direction test) is reproduced
with high confidence, and all training hyperparameters are Explicit in
the paper. Real training confirms the model actually learns and
generalizes (Protocol A: 82.67%/81.15%, non-degenerate Macro-F1;
Protocol B-D1: 90.73%+-3.81pp across 5 seeds, notably more stable than
Dynamic GIN+TGP's 88.44%+-6.83pp on the identical split). This baseline
still carries three genuinely unresolved physical/definitional gaps that
directly affect Protocol A's paper-comparison validity specifically:
(1) PHM2010 tool diameter/helix angle for `Vst`'s shape (assumed, not
sourced), (2) ref [41]'s exact I/II/III boundary rule (proxied, not
reproduced), and (3) the empirical non-occurrence of Stage IV for these
specific tools (a real data constraint, not a choice, but one that
changes the task from 4-class to 3-class) -- these explain a meaningful
share of Protocol A's 6-8pp gap to the paper, on top of ordinary
optimization variance. None of these affect Protocol B (which uses
DC-PSR's own E/M/L labels throughout, sidestepping items 2 and 3
entirely), so **Protocol B-D1's number is trusted with the highest
confidence of any result in this report** -- confirmed stable across 5
independent seeds, not a lucky draw. Not rated higher only because
Protocol A's paper-comparison gap (6-8pp) remains larger than
`dynamic_gin_tgp`'s (-0.59pp), and items (1)-(2) above remain genuinely
unverifiable without the original paper's code or ref [41]'s full text.

## Files

```
baselines/dp2net/
    PAPER_SPEC.md              component-by-component paper audit (incl. sec 6b empirical finding)
    README.md                  usage, training commands, results (this file's companion)
    FINAL_REPORT.md            this file
    preprocessing.py           low-pass filter, Vst construction, 4-stage proxy labels, window caches
    model.py                   S, G (+AdaIN), F=WDCNN, MMD
    train.py                   Protocol A/B-S/B-D1/smoke, two-stage Algorithm-1 training
    data/
        label_utils.py         Unified Protocol B-S/B-D1 labels/splits (imports 代码/ read-only)
        windows/                Protocol A cache (sample_manifest.csv-indexed)
        windows_unified/        Protocol B cache (unified_manifest.csv-indexed, 8/run)
    tests/
        test_pipeline.py       15 tests: preprocessing, S/G/F shapes, MMD, both stages' gradients (all pass on CPU)
outputs/dp2net/
    smoke/                              CPU smoke-test outputs
    original_protocol/target_{C4,C6}/   Protocol A outputs (populated after the user runs training)
    unified_protocol_B-S/seed<N>/       Protocol B-S outputs
    unified_protocol_B-D1/seed<N>/      Protocol B-D1 outputs (the DC-PSR D1 main-table candidate)
```

## Remaining uncertainty

1. ~~Whether the model converges to near-paper accuracy on real data~~ --
   **resolved**: confirmed on real GPU training, non-degenerate results
   on both Protocol A (82.67%/81.15%) and Protocol B-D1 (90.79%), though
   Protocol A's gap to the paper (6-8pp) is larger than Dynamic
   GIN+TGP's (-0.59pp), consistent with this baseline carrying more
   Missing/adapted items (see item 3 below and Sec A).
2. True PHM2010 tool diameter/helix angle -- affects Vst's shape (not its
   period), and hence the physical-constraint term's exact target;
   unverifiable without the original PHM2010 CAD/tool documentation.
3. Whether Protocol A's adapted 3-class task is "easier" than the paper's
   4-class task in a way that materially inflates our accuracy numbers
   relative to the paper's own 90.91%/87.66% -- flagged explicitly, not
   resolved (would require reproducing failure-stage data the archive
   doesn't contain). Note the *actual* measured numbers came in LOWER
   than the paper's, not inflated -- so if anything this effect (which
   would push our numbers up) is being outweighed by other gaps (D/beta
   assumption, I/II/III proxy, generic optimization variance).
4. ~~Real GPU training time/VRAM/inference-latency numbers~~ -- **resolved**:
   Protocol A, all 5 B-D1 seeds' training times, and inference latency
   all measured (Sec E).
5. ~~Protocol B-D1's 5-seed mean+-std stability~~ -- **resolved**: all 5
   seeds complete, Acc=90.73%+-3.81pp, a tight and reassuring spread
   (tighter than Dynamic GIN+TGP's 88.44%+-6.83pp on the identical
   split). B-S (supplementary, not required for the main table) is
   optional and may not have been run in every session -- check
   `outputs/dp2net/unified_protocol_B-S/seed42/` for its status.

# DP2Net

## Paper

Lai, X., Zhang, K., Zheng, Q., Zhao, M., Ding, G., Tang, B., Li, Z.
"DP2Net: A discontinuous physical property-constrained single-source
domain generalization network for tool wear state recognition."
**Mechanical Systems and Signal Processing 215 (2024) 111421.** DOI:
10.1016/j.ymssp.2024.111421. No public code/data ("The authors do not
have permission to share data"). From-scratch **reimplementation** --
see `PAPER_SPEC.md` for the full component-by-component audit.

## Original protocol (Protocol A)

- **Task**: SSDG -- Source = C1 (70/30 train/val), Target test = C4 and
  C6 separately (paper's own Table 2, Task 1/Task 2).
- **Labels**: paper-native 4-stage (I/II/III/IV). **Empirical finding**:
  Stage IV (mean VB > 0.3mm) never occurs for C1/C4/C6 in this project's
  real PHM2010 archive (max mean-VB across all three tools: 216um, well
  under the 300um/0.3mm threshold) -- see PAPER_SPEC.md sec 6b. Protocol
  A is therefore run as an **adapted 3-class (I/II/III)** scheme, not the
  paper's claimed 4-class scheme.
- **Raw channel**: Fx only, low-pass filtered (Butterworth, order 4,
  cutoff 1733Hz).
- **Sample length**: 4608 (16 spindle revolutions at 50kHz/10400rpm).
- **Hyperparameters** (Sec 4.3): Adam lr=1e-3 for S and F (F's lr
  additionally cosine-annealed, period 20 epochs), Adam lr=1e-5 for G,
  batch=64, alpha=20, kpool=4, k=25 (PHM2010). Algorithm 1: 100 epochs
  pretrain(S+F) + 100 epochs generalize(G+F, S frozen).
- **Paper-reported accuracy**: Task1 (C1->C4) = 90.91%, Task2 (C1->C6) = 87.66%.

## Unified protocol

Two variants, per task instructions #63-67:

- **Protocol B-S** (native single-source, preserves DP2Net's SSDG
  character): source=C1 only, target/test=C6, DC-PSR condition-relative
  E/M/L labels.
- **Protocol B-D1** (pooled-source, enters the DC-PSR D1 main table):
  source=C1+C4 pooled, target/test=C6, DC-PSR E/M/L labels. Named
  **"DP2Net-adapted (pooled source)"** in all outputs -- NOT "original
  DP2Net", since pooling two source domains is a genuine protocol
  deviation from the paper's own strict single-source design (task
  instruction #64).

Both variants: 8 windows/run (`data/unified_manifest.csv`), run-level
mean-probability aggregation at test time, no target (C6) leakage into
model selection.

## Raw data

`archive/c{1,4,6}/c{1,4,6}/c_{1,4,6}_{run:03d}.csv` column 0 (Fx) only.
`archive/c{1,4,6}/c{1,4,6}_wear.csv` for flute wear (**in micrometers**,
not millimeters -- see Missing/conflicts below).

## Preprocessing

`preprocessing.py`:
1. Load Fx only.
2. Butterworth low-pass, cutoff=1733Hz, order=4, zero-phase (`filtfilt`).
3. Protocol A: paper-native 4-stage labeling (Stage IV: mean VB>300um,
   explicit; Stages I/II/III: documented wear-rate-change-point proxy,
   see below), then 2000 windows/class (I/II/III; IV empirically empty)
   via fixed-seed uniform-random 4608-length window sampling, recorded
   in `data/sample_manifest.csv` (condition, run_id, class, start_idx,
   end_idx).
4. Protocol B: 8 windows/run spread evenly across each run's signal,
   `data/unified_manifest.csv`, labels attached at train time via
   `data/label_utils.py`.

Run `python preprocessing.py` (full 2000/class build) and
`python -c "import preprocessing as P; P.build_unified_windows_cache()"`
to (re)build both caches -- both are already built in this repo.

## Architecture

`model.py`, per PAPER_SPEC.md sec 3-5:
- **S** (spatial attention): `AvgPool1d(kpool=4) -> Conv1d(k=25,SAME) ->
  BN -> ReLU -> Sigmoid` gate, upsampled back to full length, multiplies
  the raw filtered input -> `Fa`.
- **G** (generator): same pooling/kernel as S, 3 conv layers
  (channels 1->1->4->4, paper-explicit), AdaIN (style from random noise)
  after the last two convs, `ConvTranspose1d` back to 4608, `tanh` output
  (bounded to [-1,1], matching `Vst`'s range) -> `Wg`.
- **`Vst`** (standard trend vector): periodic -1->1 ramp (fraction `P` of
  each period `L`) + constant-0 rest. `L=96.15` samples/tooth-period
  (Eq.6, paper-exact, no missing inputs). `P=0.055` (Eq.5, depends on
  tool diameter D and helix angle beta, **both Missing for PHM2010 in
  this paper** -- assumed D=6mm/beta=30deg, see below).
- **F** (WDCNN): canonical wide-first-kernel deep CNN (Zhang et al. 2018,
  [38]), input length adapted from the original 2048/6000pt window to
  this task's 4608pt window.
- **MMD**: Gaussian kernel, median-heuristic gamma (Missing in paper).
- Total S+G+F parameters: **60,956**.

## Missing details / paper conflicts (full detail in PAPER_SPEC.md)

1. **Sampling frequency**: paper text says PHM2010 is "sampled at 5kHz",
   but the real archive is 50kHz and the paper's own `k=25` estimate is
   only physically consistent at 50kHz. We use 50kHz (native) + k=25.
2. **I/II/III stage boundaries** (ref [41]): not recoverable from this
   paper's own text or from a public-web search of ref [41]'s abstract
   (paywalled, full text not accessible). Protocol A uses a documented
   wear-rate-change-point proxy -- **not** an exact reproduction of
   ref [41]'s own criterion.
3. **Stage IV empirically never occurs** for C1/C4/C6 in this project's
   real archive (units: micrometers, not millimeters -- confirmed by
   cross-checking against `archive/c{1,4,6}/c{1,4,6}_wear.csv`'s value
   range of ~30-220). Protocol A is an adapted 3-class scheme as a
   result -- see PAPER_SPEC.md sec 6b for the full investigation.
4. **Tool diameter D and helix angle beta for PHM2010** (needed for
   `Vst`'s `P` fraction): the paper's own Table 1 only gives these for
   its *own* machining-experiment tool (16mm/35deg), not for PHM2010.
   Assumed D=6mm (PHM2010-documented ball-nose cutter diameter) and
   beta=30deg (placeholder, no PHM2010-specific source located). This
   only affects `Vst`'s rise-fraction shape, not its period.
5. **WDCNN** is a well-known public architecture ([38]) reused by
   reference, not independently re-derived from this paper's own text --
   flagged as an external dependency per task instruction #58.

## Training commands

**Prerequisite**: `python preprocessing.py` and
`python -c "import preprocessing as P; P.build_unified_windows_cache()"`
(both already run in this repo).

```bash
# Smoke test (fast, tiny subset, CPU or GPU) -- run this first
python train.py --protocol smoke --device cpu

# Protocol A (paper sanity check): source C1, target C4/C6 separately
python train.py --protocol A --device cuda

# Protocol B-S (unified, native single-source): C1 -> C6
python train.py --protocol B-S --device cuda --seed 42

# Protocol B-D1 (unified, pooled-source adapted, enters DC-PSR D1 main table): C1+C4 -> C6
python train.py --protocol B-D1 --device cuda --seed 42

# Repeat B-D1 (and optionally B-S) across the project's standard 5 seeds
for seed in 42 52 62 72 82; do
    python train.py --protocol B-D1 --device cuda --seed $seed
done
```

### Resume command

Each protocol checkpoints both training stages separately
(`stage1_checkpoint.pth`, `stage2_checkpoint.pth`) plus a
`training_log.csv`, in `outputs/dp2net/<protocol_dir>/` (or
`.../seed<N>/` for B-S/B-D1). If interrupted, re-run the exact same
command with `--resume`:

```bash
python train.py --protocol A --device cuda --resume
python train.py --protocol B-D1 --device cuda --seed 42 --resume
```

A completed run writes `DONE.flag`.

## Results

**Protocol A (paper sanity check)** -- real GPU run (RTX 3070 Ti Laptop, 8GB):

| Target | Paper | Our reproduction | Note |
|---|---|---|---|
| C4 | 90.91% | **82.67%** (gap -8.24pp) | 4-stage in paper, adapted 3-stage here (Stage IV empty) |
| C6 | 87.66% | **81.15%** (gap -6.51pp) | 4-stage in paper, adapted 3-stage here (Stage IV empty) |

Both results are non-degenerate (Macro-F1 0.82/0.81), consistent with a
working reproduction; the gap vs. the paper is larger than for
`dynamic_gin_tgp` (-0.59pp), plausibly reflecting this baseline's larger
number of Missing/adapted items (Vst's D/beta assumption, the I/II/III
boundary proxy) on top of ordinary optimization variance.

**Protocol B-D1 (unified, main table)** -- all 5 seeds complete;
**B-S** (supplementary, single-source) running:

| Variant/Seed | Acc | Macro-F1 | E-F1 | M-F1 | L-F1 | M-Pre | M-Rec | M->E | M->L | Rev | Jump | Smooth |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B-D1, seed 42 | 0.9079 | 0.9078 | 0.8485 | 0.8968 | 0.9783 | 0.8289 | 0.9767 | 0.0000 | 0.0233 | 4 | 0 | 0.0519 |
| B-D1, seed 52 | 0.8794 | 0.8865 | 0.8444 | 0.8319 | 0.9832 | 0.9691 | 0.7287 | 0.2713 | 0.0000 | 3 | 0 | 0.0530 |
| B-D1, seed 62 | 0.9492 | 0.9506 | 0.9845 | 0.9339 | 0.9333 | 1.0000 | 0.8760 | 0.0233 | 0.1008 | 2 | 0 | 0.0514 |
| B-D1, seed 72 | 0.8603 | 0.8632 | 0.8920 | 0.8281 | 0.8696 | 0.8346 | 0.8217 | 0.1783 | 0.0000 | 3 | 0 | 0.0610 |
| B-D1, seed 82 | 0.9397 | 0.9408 | 0.9596 | 0.9272 | 0.9357 | 0.9167 | 0.9380 | 0.0620 | 0.0000 | 4 | 0 | 0.0612 |
| **B-D1, mean+-std** | **0.9073+-0.0381** | **0.9098+-0.0365** | **0.9058+-0.0639** | **0.8836+-0.0509** | **0.9400+-0.0457** | **0.9099+-0.0773** | **0.8682+-0.0979** | **0.1070+-0.1146** | **0.0248+-0.0436** | **3.2+-0.8** | **0.0+-0.0** | **0.0557+-0.0050** |
| B-S, seed 42 | 0.9587 | 0.9592 | 0.9841 | 0.9517 | 0.9419 | 0.9143 | 0.9922 | 0.0078 | 0.0000 | 3 | 0 | 0.0558 |

B-D1's 5-seed mean (90.73%+-3.81pp) is notably strong and, importantly,
**more stable across seeds than `dynamic_gin_tgp`'s Protocol B result**
(std 3.81pp vs 6.83pp on Acc) on the identical split.

B-D1's seed-42 accuracy (90.79%) is notably higher than Protocol A's
paper-comparison numbers -- plausible given Protocol B uses DC-PSR's
cleaner E/M/L labels and pooled C1+C4 source data, but treat as
preliminary until the remaining 4 seeds confirm it's not a lucky draw.

## Caveats

- This baseline carries the most Missing/adapted items of the two
  baselines in this project (see PAPER_SPEC.md's risk summary: overall
  **Medium** fidelity) -- specifically the I/II/III boundary proxy, the
  D/beta assumption for `Vst`, and the empirical Stage-IV-never-occurs
  finding. None of these are silently patched; all are documented and
  the resulting Protocol A comparison is explicitly labeled "adapted
  reproduction," not "exact reproduction."
- All unit/shape/gradient/single-batch-overfit tests pass on CPU
  (`tests/test_pipeline.py`, 15/15), including both training stages'
  gradient flow and the MMD-maximization direction check (Eq.10's
  `-alpha*MMD` sign, verified not accidentally flipped).
- Full 100+100-epoch convergence is **confirmed** on real GPU training
  (Results above). Unlike `dynamic_gin_tgp`, no cross-sample batch
  dependence exists in this architecture (S/G/F all process one sample
  at a time), so no evaluation-methodology bugs were found here.

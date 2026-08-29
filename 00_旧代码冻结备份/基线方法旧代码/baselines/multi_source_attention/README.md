# Multi-source Attention (Multi-Attention-CNN) baseline

An independent reimplementation of the proposed method from Wei et al.,
added as a published-architecture baseline for this project's DC-PSR
manuscript. This directory is fully isolated from `代码/`,
`experiments_mendeley/`, `baselines/htt_net/`, and `补充材料/` (read-only
imports of `代码/main_experiment_3_fgds_psi_optimized.py` for shared
labels/metrics only -- never edits it).

## Paper

Wei, P., Li, R., Liu, X., Gao, H., Dai, M., Zhang, Y., Zhao, W., Liu, E.
(2024). "Research on tool wear state identification method driven by
multi-source information fusion and multi-dimension attention mechanism."
*Robotics and Computer-Integrated Manufacturing*, 88, 102741.
https://doi.org/10.1016/j.rcim.2024.102741

No author source code was found (Data availability: "Data will be made
available on request"; no code link in the paper) -- this is a
reimplementation from the paper text, not a port of the authors' code.
See `PAPER_SPEC.md` for the full Explicit/Inferable/Missing component
table (68 rows) and the "Open questions / conflicts" section (6 flagged
items, most consequential: CWT parameters are entirely unstated, and the
Fig. 3 vs. Table 2 fusion-convolution mismatch).

## Data

PHM2010, conditions C1/C4/C6 (315 cutting passes each, 945 total), from
`archive/c{1,4,6}/`. Raw channels: 7-column CSV, no header, order
Fx,Fy,Fz,Vx,Vy,Vz,AE (standard PHM2010 layout). This method uses **force +
vibration only** (6 of 7 channels); AE is loaded but never used, matching
the paper's own 2-source framing (§2.1).

Wear label: VB = max(flute_1, flute_2, flute_3) per run, from
`archive/c{cond}/c{cond}_wear.csv` -- this project's confirmed real-data
convention (see `baselines/htt_net/README.md`). Used only for Protocol B
labels (via `data/label_utils.py`); Protocol A uses the paper's own
EM-derived pass-index partition instead (see below).

## Native preprocessing (kept for both protocols, per task instructions #10-11)

```
raw Fx,Fy,Fz,Vx,Vy,Vz (7-channel PHM2010 CSV)
  -> "middle region" of the cutting pass         [Missing in paper -> central 50%]
  -> per-axis resample to 224 samples             [Missing in paper]
  -> per-axis Continuous Wavelet Transform         [Missing in paper -> complex Morlet
     (complex Morlet, 224 log-spaced scales)        cmor1.5-1.0, 224 scales = image side]
  -> per-image min-max normalize to [0,1]         [Missing in paper]
  -> stack 3 axes -> RGB                           -> force image [224,224,3]
                                                       vibration image [224,224,3]
  -> dual-branch Conv2D(16,k3,ReLU) each -> concat [B,32,224,224]
  -> Channel+Spatial Attention (Eqs 1-18, r=16)    [only insertion point, per paper]
  -> MaxPool -> Conv2D(64) -> MaxPool -> Conv2D(128) -> MaxPool -> FC(128) -> Dropout(0.5) -> FC(3)
```

This is **not** reduced to engineered scalar features at any point --
unlike HTT-Net (B13), which explicitly *was* re-plugged into the shared
L=12 engineered-feature pipeline. The CWT + dual-branch CNN + attention
*is* the method; collapsing it to a feature vector would no longer be
this paper's architecture (task instruction #10).

CWT parameters (wavelet family, scale count, frequency range) are
**entirely unstated in the paper** -- the single largest fidelity risk for
Protocol A. See `PAPER_SPEC.md` rows "Continuous Wavelet Transform
parameters" and "Open questions #2".

## Reproduced components (faithful to the paper)

- Dual-branch CNN: independent `Conv2D(16,k3,s1,same,ReLU)` per source,
  concatenated to `[B,32,224,224]` (Table 2 Layers 1-2, Fig. 3).
- Channel attention: GAP -> FC1(reduce by r) -> ReLU -> FC2(restore) ->
  Sigmoid (Eqs. 1-8), literally implemented.
- Spatial attention: 1x1 conv channel compression -> {AvgPool,MaxPool}
  over channel dim -> concat -> 1x1 conv -> Sigmoid (Eqs. 9-15).
- Parallel (non-cascaded) combination `X~ = X * s * Ms`, both computed
  from the *same* pre-attention `X` (Eq. 18) -- explicitly **not** a
  sequential-recompute CBAM, per the paper's own Fig. 6.
- Attention applied **only** at the first (post-concatenation) layer, not
  at every conv block -- literal reading of §2.3.3's own stated reasoning
  (deeper feature maps too small, extra params hurt anti-interference).
- CNN backbone layer-for-layer per Table 2 (11 layers, filters/kernel/
  stride/padding/activation/output-shape all Explicit).
- Optimizer Adam, CrossEntropyLoss, L2 weight decay (coefficient itself
  Missing -> 1e-4 default).
- Protocol A: paper's own "Scheme 5" (Fig. 9): 100 epochs, batch=128,
  lr=0.001, single step decay x0.1 at epoch 30.
- Protocol A: paper's own EM-derived fixed pass-index stage partition
  (Table 1: C1 Initial 1-47/Normal 48-146/Severe 147-315; C4 Initial
  1-135/Normal 136-204/Severe 205-315; C6 Initial 1-81/Normal 82-188/
  Severe 189-315) and 70/30 stratified split pooled across C1+C4+C6.

## Adaptations (documented, not silent)

- **CWT parameters** (wavelet family = complex Morlet `cmor1.5-1.0`, 224
  log-spaced scales, per-axis-to-RGB-channel mapping, min-max image
  normalization): all **Missing in paper**, chosen as the standard
  convention for this literature niche. This is the largest reimplementation
  risk -- see PAPER_SPEC.md.
- **"Middle region" of the signal**: Missing in paper -> central 50% of
  each pass's raw signal (indices 25%-75%).
- **Channel-attention reduction ratio r=16**: Missing in paper -> SE-Net
  default (the paper explicitly cites SE-Net as its channel-attention
  basis, ref. [38]).
- **Spatial-attention channel-compression ratio**: reuses the same r=16
  for consistency (paper gives no separate number).
- **L2 weight-decay coefficient = 1e-4**: Missing in paper -> conventional
  Adam-CNN default.
- **Repeated-training/averaging strategy**: paper does 12 runs, drops
  best/worst, averages 10. This reproduction uses **N_SEEDS=5** by
  default (`--n-seeds` overridable), no best/worst dropping -- a
  documented compute-budget adaptation, not the paper's literal
  procedure. `train.py --protocol A --n-seeds 12` reproduces the paper's
  exact count if run.
- **Table 2 vs. Fig. 3 fusion-convolution mismatch**: Table 2 (used as
  ground truth) has no separate fusion-conv layer between Concatenation
  and MaxPooling; Fig. 3 shows one. `Layer4`'s 64-filter conv (after the
  first MaxPool) is treated as the closest literal match to Fig. 3's role.
  Recorded as a genuine paper-internal inconsistency, not resolved by
  guessing which is "more correct."
- Framework/hardware: PyTorch on an RTX 3070 Ti Laptop 8GB, vs. the
  paper's MATLAB on an RTX 3090 -- not treated as an algorithmic fidelity
  gap.

Because of the CWT-parameter and middle-region gaps above, this
reproduction is described throughout as a **reimplementation** /
**adapted implementation**, not an "exact reproduction," per task
instruction #7.

## Two protocols (never mixed into one number)

**Protocol A (original-paper sanity reproduction)**
Paper's own EM stage labels, 70/30 stratified split (pooled C1+C4+C6),
Scheme 5 hyperparameters. Purpose: verify the implementation itself is
credible. **Not used as the DC-PSR main-table result.**
Output: `outputs/multi_source_attention/original_protocol/`

**Protocol B (Unified DC-PSR comparison, D1 = C1+C4 -> C6)**
This project's condition-relative Early/Middle/Late labels (reused
byte-for-byte from `代码/main_experiment_3_fgds_psi_optimized.py` via
`data/label_utils.py`), train/val carved from C1+C4 only, C6 held out
entirely for the single final evaluation. Native CWT + dual-branch +
attention preprocessing unchanged. This **is** the DC-PSR main-table
result.
Output: `outputs/multi_source_attention/unified_protocol/`

Model/epoch selection for Protocol B uses **only** the C1+C4 internal
validation split -- C6 is never used for preprocessing tuning, learning
rate, epoch count, or architecture choice (task instruction #27/#58).

## Files

```
baselines/multi_source_attention/
├── PAPER_SPEC.md        paper-to-code specification (Explicit/Inferable/Missing table)
├── preprocessing.py      middle-region extraction, CWT, RGB image construction, original-protocol stage labels
├── model.py              dual-branch CNN + channel/spatial attention (MultiAttentionCNN)
├── train.py               Protocol A / Protocol B / smoke-test training + evaluation
├── README.md              this file
├── FINAL_REPORT.md        full fidelity/results/leakage-audit report
├── data/
│   ├── label_utils.py     read-only reuse of 代码/'s condition-relative labels + C1+C4/C6 split
│   ├── build_dataset.py   builds+caches the 945-run force/vibration image dataset
│   ├── images/            cached [224,224,3] uint8 .npy force/vib images (945 runs x 2)
│   └── metadata.csv       condition, run_id, stage_original(_id), VB, stage_unified(_id)
└── tests/
    └── test_multi_source_attention.py   15 unit tests
```

Outputs live in `outputs/multi_source_attention/`: `smoke/` (pipeline
sanity only), `original_protocol/` (Protocol A), `unified_protocol/`
(Protocol B, the DC-PSR main-table result).

## Commands

```bash
# Unit tests
python tests/test_multi_source_attention.py

# Build the cached image dataset (once; ~15 min on CPU)
python data/build_dataset.py

# Pipeline smoke test (CPU-safe, no GPU needed)
python train.py --protocol smoke --device cpu

# Protocol A (original-paper sanity reproduction)
python train.py --protocol A --device cuda --n-seeds 5

# Protocol B (Unified DC-PSR comparison)
python train.py --protocol B --device cuda
```

## What has been validated so far

- `tests/test_multi_source_attention.py` -- 15/15 pass: middle-region
  length, CWT scalogram shape/range, RGB image construction, real-data
  build_sample() on C1 run 1, original-stage-range coverage (no gap/
  overlap across 1..315), channel/spatial/combined attention shapes and
  gradient flow, dual-branch fusion shape, forward/softmax/NaN checks,
  and single-batch overfit (16 synthetic samples, loss drops >70%,
  accuracy >=90%).
- `train.py --protocol smoke --device cpu` -- full pipeline (cached-image
  loading -> training loop -> checkpoint -> DC-PSR metric functions ->
  CSV/JSON output) runs end to end without errors.
- Full 945-run image cache built (`data/metadata.csv`, verified stage
  counts for both label schemes sum correctly per condition).

## Known caveats

- **Results are in — see `FINAL_REPORT.md`.** Protocol A: 5-seed mean
  Acc=0.8254 (paper reports 0.982, gap=-0.157), attributed primarily to
  the unstated CWT parameters (the entire preprocessing backbone).
  Protocol B (Unified, DC-PSR main-table result): C6 test Acc=0.819,
  Macro-F1=0.824.
- Protocol A used 5 seeds (not the paper's 12-with-drop-best/worst),
  documented above as a compute-budget adaptation.
- 12.9M params, ~1.04 GFLOPs/sample, 1.81 ms/sample inference (RTX 3070
  Ti Laptop, fp32).

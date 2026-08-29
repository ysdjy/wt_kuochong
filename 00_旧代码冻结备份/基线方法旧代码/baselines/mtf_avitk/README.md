# MTF-AViTK baseline

An independent reimplementation of the proposed method from Dong et al.,
added as a published-architecture baseline for this project's DC-PSR
manuscript. This directory is fully isolated from `代码/`,
`experiments_mendeley/`, `baselines/htt_net/`, `baselines/multi_source_attention/`
and `补充材料/` (read-only imports of
`代码/main_experiment_3_fgds_psi_optimized.py` for shared labels/metrics
only -- never edits it).

## Paper

Dong, S., Meng, Y., Yin, S., Liu, X. (2025). "Tool wear state recognition
study based on an MTF and a vision transformer with a Kolmogorov-Arnold
network." *Mechanical Systems and Signal Processing*, 228, 112473.
https://doi.org/10.1016/j.ymssp.2025.112473

No author source code was found (Data Availability: "Data will be made
available on request"; no code link anywhere in the paper) -- this is a
reimplementation from the paper text. See `PAPER_SPEC.md` for the full
Explicit/Inferable/Missing component table and the "Open questions /
conflicts" section (9 flagged items; most consequential: the Eq.1-vs-§3.2
2000-vs-500 MTF field-size conflict, and the completely unstated
ViT-pretraining status).

## Data

PHM2010, conditions C1/C4/C6 (315 cutting passes each, 945 total), from
`archive/c{1,4,6}/`. This method uses **force only** (Fx,Fy,Fz -> resultant
force F=sqrt(Fx^2+Fy^2+Fz^2), paper's own Eq. 8) -- vibration and AE are
never used, matching the paper's own explicit statement (§5 Conclusions:
"restricted by time and cost issues", only force was used).

Wear label: VB = max(flute_1, flute_2, flute_3) per run (this project's
confirmed real-data convention, see `baselines/htt_net/README.md`). Used
only for Protocol B labels (`data/label_utils.py`); Protocol A uses the
paper's own fixed cut-index thresholds instead.

## Native preprocessing (kept for both protocols, per task instructions #10-11)

```
raw Fx,Fy,Fz
  -> stable-region slice (90,000-100,000, main dataset)          [Explicit]
  -> non-overlapping 2000-sample sub-windows (5 per pass)         [Explicit]
  -> resultant force F = sqrt(Fx^2+Fy^2+Fz^2)                     [Explicit, Eq. 8]
  -> Sym7 wavelet, level-6 decomposition, soft threshold          [Explicit method;
                                                                     Missing threshold formula
                                                                     -> VisuShrink/MAD]
  -> resample 2000 -> 500 samples                                 [Missing/Conflict resolution]
  -> MTF encoding (Q=8 quantile bins) -> [500,500] field           [Explicit mechanism, Eq.1;
                                                                     Missing Q]
  -> jet colormap -> [500,500,3] RGB image                        [Missing]
  -> bilinear resize -> [384,384,3]                                [Missing]
  -> Adapt-ViT_L/32 (24 blocks, 16 heads, 1024 dim, AdaptMLP)      [Explicit topology;
                                                                     Missing adapter d_hat/scale;
                                                                     trained from scratch, no
                                                                     pretraining stated in paper]
  -> 2-layer KAN (G=5, k=3, SiLU, scale=1.0) -> 3 classes          [Explicit hyperparams;
                                                                     Missing hidden width]
```

This is **not** reduced to engineered scalar features -- the resultant
force + wavelet-denoise + MTF encoding + ViT + AdaptMLP + KAN pipeline
*is* the method (task instruction #10).

## Reproduced components (faithful to the paper)

- Resultant force Eq. 8, applied to raw (undenoised) force before
  wavelet denoising, matching Fig. 8's pipeline order.
- Sym7 wavelet, decomposition level 6, soft thresholding (all Explicit).
- MTF mechanism (Eq. 1): normalize to [0,1], quantile-bin, first-order
  Markov transition matrix, expand to the [n,n] field.
- Adapt-ViT_L/32: 32x32 patch conv (stride 32) -> 144 patches + CLS ->
  145x1024, 24 encoder blocks, 16-head MHSA, 1024->4096->1024 MLP, GELU
  -- all Explicit (patch/embed/heads from text, depth=24 from Fig. 3's
  own "Encoder Block x24" label).
- AdaptMLP topology (Eq. 6): shared LayerNorm feeding both the original
  MLP branch and a parallel down-project -> ReLU -> up-project adapter
  branch, scaled and added to the MLP output.
- KAN classifier: 2-layer, from-scratch B-spline `KANLinear`
  implementation (no third-party KAN package used, per task instruction
  #42), with the paper's own Table 1 hyperparameters (G=5, k=3, SiLU
  residual, scale=1.0) implemented exactly.
- Training hyperparameters (Protocol A): paper's own Table 5 -- SGD,
  lr=0.0006, weight_decay=2e-4, momentum=0.9, epochs=50, batch_size=8.
- Protocol A: paper's own fixed cut-index stage thresholds (Initial
  1-50/Normal 51-175/Severe 176-315) and B1 split (train C1+C4, test C6)
  -- B1 is the paper's own direct analogue of this project's D1 task.

## Adaptations (documented, not silent)

- **MTF field-size resolution (2000 -> 500)**: Eq. 1 literally defines
  the MTF field as [n,n] with n = input series length (2000, per the
  paper's own 2000-sample sub-windows); §3.2 states final images are
  500x500x3 with no explained step between the two. Implementation
  choice: resample the denoised 2000-sample signal down to 500 points
  *before* MTF encoding (keeps a literal Eq.-1 application, avoids
  post-hoc image-interpolation corrupting the transition-probability
  semantics of each cell). Flagged as the single highest-fidelity-risk
  gap in the whole reproduction.
- **MTF quantile bin count Q=8**: Missing in paper -> Wang & Oates'
  original MTF default / `pyts` library default.
- **RGB channel construction**: an MTF field is inherently single-channel;
  paper never explains the "x3". Implementation choice: `jet` colormap
  (visually matches Fig. 9's rendered images).
- **500x500 -> 384x384 resize method**: Missing in paper -> bilinear
  (standard ViT preprocessing default).
- **AdaptMLP bottleneck dim=64, scale=0.1**: Missing in paper -> reused
  directly from AdaptFormer (ref. [56]), the module's own cited origin.
- **KAN hidden width=64**: Missing in paper (only G/k/residual-fn/scale
  are given, no layer width).
- **ViT pretrained weights: NOT loaded (trained from scratch)**. The
  paper never mentions a pretraining source (no "ImageNet", no
  checkpoint, no transfer-learning discussion anywhere in 18 pages).
  This is the **second-highest fidelity risk**: ViT-L/32 is a ~300M-param,
  data-hungry architecture, and PHM2010 (945 runs, ~4725 sub-window
  images) is small by ViT-pretraining standards. If Protocol A accuracy
  comes in far below the paper's reported 95.38% (B1), this is the first
  hypothesis to test.
- **Wavelet soft-threshold value formula**: Missing in paper (only the
  "soft thresholding" *method* is stated) -> universal/VisuShrink
  threshold with MAD-based sigma.
- **Sub-window-to-run-level aggregation (Protocol B only)**: the paper
  evaluates at the image (sub-window) level; this project's other
  baselines are run-level. Protocol B trains at sub-window level but
  aggregates each run's 5 sub-window probability vectors (mean) into one
  run-level prediction before computing metrics -- a reimplementation
  choice for comparability, not a paper-specified step. Protocol A
  reports image-level accuracy (matching how the paper itself evaluates
  B1/B2/B3).
- **GPU memory adaptations** (task instruction #68, only if actually
  needed on the 8GB RTX 3070 Ti): AMP (mixed precision) is on by default;
  `--grad-checkpoint` and `--grad-accum-steps` are available as pure
  compute/memory trade-offs. The model architecture itself is never
  changed (no ViT-L -> ViT-B substitution). See FINAL_REPORT.md for
  which, if any, were actually required.
- Framework/hardware: this project uses an RTX 3070 Ti Laptop 8GB vs. the
  paper's RTX 4080 Super -- not treated as an algorithmic fidelity gap.

Because of the MTF field-size conflict and pretraining-status gaps above,
this reproduction is described throughout as a **reimplementation** /
**adapted implementation**, not an "exact reproduction," per task
instruction #7.

## Two protocols (never mixed into one number)

**Protocol A (original-paper sanity reproduction, B1 split)**
Paper's own fixed-threshold stage labels, B1 = train{C1,C4}/test{C6},
image (sub-window) level, Table 5 hyperparameters. Purpose: verify the
implementation is credible against the paper's reported B1=95.38%. **Not
used as the DC-PSR main-table result.**
Output: `outputs/mtf_avitk/original_protocol/`

**Protocol B (Unified DC-PSR comparison, D1 = C1+C4 -> C6)**
This project's condition-relative Early/Middle/Late labels (reused
byte-for-byte from `代码/main_experiment_3_fgds_psi_optimized.py` via
`data/label_utils.py`), train/val carved from C1+C4 only, C6 held out
entirely, run-level-aggregated evaluation. Native MTF + Adapt-ViT +
AdaptMLP + KAN preprocessing/architecture unchanged. This **is** the
DC-PSR main-table result.
Output: `outputs/mtf_avitk/unified_protocol/`

Model/epoch selection for Protocol B uses **only** the C1+C4 internal
(run-level-aggregated) validation accuracy -- C6 is never used for
preprocessing tuning, learning rate, epoch count, or architecture choice.

## Files

```
baselines/mtf_avitk/
├── PAPER_SPEC.md        paper-to-code specification (Explicit/Inferable/Missing table)
├── preprocessing.py      resultant force, wavelet denoise, MTF encoding, RGB/resize
├── kan.py                 from-scratch KANLinear / KANClassifier (no third-party KAN package)
├── model.py               Adapt-ViT_L/32 + AdaptMLP + KAN (MTF_AViTK)
├── train.py                Protocol A / Protocol B / smoke-test training + evaluation
├── README.md               this file
├── FINAL_REPORT.md         full fidelity/results/leakage-audit report
├── data/
│   ├── label_utils.py      read-only reuse of 代码/'s condition-relative labels + C1+C4/C6 split
│   ├── build_dataset.py    builds+caches the main (unbalanced) 5-image-per-run dataset
│   ├── images/              cached [384,384,3] uint8 .npy MTF images (945 runs x 5)
│   └── metadata.csv         condition, run_id, subwindow, stage_original(_id), VB, stage_unified(_id)
└── tests/
    └── test_mtf_avitk.py    20 unit tests
```

Outputs live in `outputs/mtf_avitk/`: `smoke/` (pipeline sanity only),
`original_protocol/` (Protocol A, B1), `unified_protocol/` (Protocol B,
the DC-PSR main-table result).

## Commands

```bash
# Unit tests
python tests/test_mtf_avitk.py

# Build the cached image dataset (once; ~9 min on CPU)
python data/build_dataset.py

# Pipeline smoke test (CPU-safe but slow -- ViT-L on CPU; GPU preferred)
python train.py --protocol smoke --device cpu

# Protocol A (original-paper sanity reproduction, B1)
python train.py --protocol A --device cuda

# Protocol B (Unified DC-PSR comparison)
python train.py --protocol B --device cuda

# If 8GB VRAM is insufficient at the paper's batch_size=8:
python train.py --protocol A --device cuda --grad-checkpoint --grad-accum-steps 2 --batch-size 4
```

## What has been validated so far

- `tests/test_mtf_avitk.py` -- 20/20 pass: resultant-force correctness,
  wavelet denoising shape/finite/noise-reduction, MTF field shape/range,
  RGB image construction, real-data `build_main_samples()` on C1 run 1,
  `KANLinear`/`KANClassifier` shape/gradient/no-NaN, patch embedding,
  MHSA, AdaptMLP (incl. verifying its zero-initialized adapter branch
  contributes exactly 0 at init), small-ViT forward, full-size
  (ViT-L/32-scale, ~309M param) forward/backward/NaN checks, softmax-
  sums-to-1, and single-batch overfit (16 synthetic samples, on a
  shrunk-down ViT config for test speed).
- `train.py --protocol smoke --device cpu` -- full pipeline (cached-image
  loading -> training loop -> sub-window-to-run-level aggregation ->
  DC-PSR metric functions -> CSV/JSON output) runs end to end without
  errors, using the **real, full-size** 309M-param model.
- Full main-dataset image cache built (`data/metadata.csv`, 4725
  sub-window rows, verified stage counts for both label schemes).

## Known caveats

- **Results are in — see `FINAL_REPORT.md`.** Protocol A (B1, full 50
  epochs, image level): Acc=0.9283 (paper reports 0.9538, gap=-0.026) —
  a close sanity match despite from-scratch (no pretraining) training.
  Protocol B (Unified, DC-PSR main-table result, run-level): C6 test
  Acc=0.902, Macro-F1=0.900.
- The full, unmodified ViT-L/32 trained successfully on the 8GB RTX 3070
  Ti at the paper's own batch_size=8 with AMP alone —
  `--grad-checkpoint`/`--grad-accum-steps` were **not** needed for either
  protocol.
- 309M params, ~91.5 GFLOPs/sample, 21.3 ms/sample inference (sub-window
  level, AMP, RTX 3070 Ti Laptop).
- **MTF field-size resampling (2000->500) remains a reimplementation
  choice**, not a paper value — documented as the top remaining fidelity
  risk despite the small overall Acc gap.

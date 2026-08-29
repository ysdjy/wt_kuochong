# PAPER_SPEC.md — Multi-source Attention (Multi-Attention-CNN) Paper-to-Code Specification

Source paper: Peining Wei, Rongyi Li, Xianli Liu, Haining Gao, Mingqiu Dai, Yuhan
Zhang, Wenkai Zhao, Erliang Liu (2024). "Research on tool wear state identification
method driven by multi-source information fusion and multi-dimension attention
mechanism." *Robotics and Computer-Integrated Manufacturing*, 88, 102741.
https://doi.org/10.1016/j.rcim.2024.102741

Extracted by reading the full text (15 pages, all sections, all tables, all
figures rendered and visually inspected) of
`Research on tool wear state identification method driven by multi-source
information fusion and multi-dimension attention mechanism.pdf` shipped in the
project root. No author source code was found (Data availability statement:
"Data will be made available on request"; no code repository link anywhere in
the paper).

> Note on the paper's own naming: the paper compares five models trained under
> identical settings — `Attention-CNN` (single force signal + channel-spatial
> attention), `Multi-CNN` (force+vibration fusion, no attention), `Multi-Resnet`
> (fusion + ResNet backbone, no attention), `Multi-Attention-CNN` (fusion +
> attention, **the paper's own proposed method**), `Multi-Attention-Resnet`
> (fusion + attention + ResNet backbone). This project reproduces
> **Multi-Attention-CNN** specifically — the model the paper's abstract and
> conclusion both call out as its contribution (Acc=0.982, F1
> 0.977/0.968/0.993).

Legend for the "Paper status" column: `Explicit` (paper states the
value/formula directly), `Inferable` (not stated as a number, but uniquely
determined by a formula, shape constraint, or an unambiguous figure elsewhere
in the paper), `Missing` (paper gives no way to derive a unique value; an
implementation choice was made and is justified in the same cell).

## 1. Component table

| Component | Paper status | Paper definition/value | Page/Fig/Table/Eq | Implementation decision |
|---|---|---|---|---|
| Dataset | Explicit | PHM2010, conditions C1, C4, C6 only (the only three of the six PHM2010 tools with recorded flank-wear labels); 315 cutting passes per condition | §3.1, §3.2 | Reused as-is from `archive/c{1,4,6}/` |
| Raw channels acquired | Explicit | 7 channels: cutting force (Fx,Fy,Fz), vibration (Vx,Vy,Vz), acoustic emission (AE); Kistler dynamometer + accelerometer + AE sensor, NI DAQ, 50 kHz sample rate | §3.1, p.7-8 | Matches `archive/c*/c*/c_*_*.csv` 7-column layout exactly (col order Fx,Fy,Fz,Vx,Vy,Vz,AE, no header) |
| Raw channels actually used by the model | Explicit | Only **force** and **vibration** signals are used as the two fusion sources ("Force signals and vibration signals of cutting are utilized to identify states", §2.1); AE is acquired but never mentioned again as model input | §2.1, §2.2 | AE channel loaded but excluded from the model; only the 3-axis force and 3-axis vibration channels are used, matching the paper's own 2-source framing |
| Per-axis vs. combined force/vibration signal | Missing | Text always says "the force signal" / "the vibration signal" (singular), never states whether the 3 axes are combined into 1 signal or each axis feeds a separate channel before CWT | §2.1, §2.2 | **Missing in paper.** Implementation choice: map the CWT scalogram of each of the 3 axes (x,y,z) onto the R,G,B channels of one 224×224×3 image per source (one image for force, one for vibration). Reason: the paper states the neural-network input is "224×224×3... an RGB image" (§2.4) for each source branch, and cutting-force/vibration are inherently 3-axis; per-axis→RGB-channel mapping is the standard way this literature (e.g. refs [35] in the paper) turns a 3-axis mechanical signal into one CWT-derived RGB image without inventing a signal-combination formula the paper never gives. |
| Stable-region signal segment ("middle region") | Missing | "The middle region of the processed signal is used as the training dataset... reduces the influence of the cutting process and the cutter in and out process" (§2.1) — no numeric start/end sample index or window length given anywhere | §2.1 | **Missing in paper.** Implementation choice: take the central 50% of each cutting pass's raw signal (indices spanning 25%–75% of the recorded sample count for that pass), which operationalizes "middle region... reduces cutter in/out effects" without inventing a specific paper-cited number. Documented as an adaptation, not a paper value. |
| Continuous Wavelet Transform parameters | Missing | Only "Continuous wavelet transforms time-domain signals into time-frequency pictures" is stated (§2.1); no wavelet family, no scale range, no number of scales, no frequency range is given anywhere in text, figures, or tables | §2.1, §1 (contributions) | **Missing in paper.** Implementation choice: complex Morlet wavelet (`cmor1.5-1.0`, PyWavelets default-style parameterization), 224 log-spaced scales spanning the segment's Nyquist-limited frequency range (50 kHz sample rate). Reason: complex Morlet is the de facto standard CWT mother wavelet for vibration/force time-frequency imaging in this exact literature niche (tool-wear CWT-CNN papers overwhelmingly use Morlet); 224 scales chosen to match the 224×224 image side length one-to-one (no separate resize step needed on the scale axis). |
| Time-frequency image size / channel meaning | Explicit | 224 × 224 × 3, "224×224 represent the actual dimension of the image, 3 means... RGB image" | §2.4, p.7 | `224×224×3` per source-branch image |
| Image normalization / resize method | Missing | Not stated | — | **Missing in paper.** Implementation choice: per-image min-max normalization of the CWT magnitude to [0,1] before RGB-channel stacking, then no further resize needed since the scale count is fixed at 224 and the time axis is resampled/truncated to exactly 224 samples per segment. |
| Dual-branch (multi-source) fusion architecture | Explicit | Parallel 2D-CNN structure: force CWT image → CNN branch, vibration CWT image → CNN branch, independently; feature maps concatenated; followed by a fusion convolution producing the "multi-source feature" | §2.2, Fig. 3 | Two independent `Conv2D(filters=16, k=3, s=1, same, ReLU)` branches (one per source), each producing 224×224×16, concatenated → 224×224×32 |
| CNN backbone (Table 2, "Basic structure of neural network") | Explicit | 11-layer table, see architecture summary below | Table 2, p.9 | Implemented layer-for-layer, see §2 below |
| Table 2 vs. Fig. 3 discrepancy | — (see Open questions) | Fig. 3 shows an explicit "Convolution" fusion layer strictly between "Concatenation" and the "Multi-source feature" output; Table 2 lists only `Layer2: Concatenation (output 224×224×32)` then jumps straight to `Layer3: Max Pooling`, with no separate fusion-conv layer named between them | Fig. 3 vs. Table 2 | Table 2 treated as authoritative for the reproducible layer-by-layer spec (it is the paper's own numbered hyperparameter table); `Layer4`'s `Conv2D(64, k=3)` is treated as absorbing the "fusion convolution" role depicted qualitatively in Fig. 3. Recorded as a conflict, not silently resolved — see §3. |
| Channel attention module | Explicit | GAP → FC1 (dim reduce by r) → ReLU → FC2 (restore dim) → Sigmoid → channel-wise multiply. Eqs. 1–8 | §2.3.1, Fig. 4, Eqs. (1)-(8) | Implemented exactly: `Xtrain ∈ R^(H×W×M)` → GAP → `F_sq ∈ R^(1×1×M)` → `FC1 = W1·F_sq` → ReLU → `FC2 = W2·ReLU(FC1)` → `S = Sigmoid(FC2)` → `X̂ = Xtrain × S` |
| Channel attention reduction ratio `r` | Missing | `W1, W2 ∈ R^(M/r × M)`; `r` named but never given a numeric value anywhere in text/tables | Eq. (6) context, p.5 | **Missing in paper.** Implementation choice: `r=16`, the standard SE-Net (Hu et al. 2018, ref [38] of this very paper) default reduction ratio, since the paper explicitly cites SE-Net as its channel-attention basis and gives no reason to deviate. |
| Spatial attention module | Explicit | 1×1 conv channel compression → {AvgPool, MaxPool} over channel dim → concat → 1×1 conv → Sigmoid → spatial multiply. Eqs. 9–15 | §2.3.2, Fig. 5, Eqs. (9)-(15) | Implemented exactly as given |
| Channel+spatial combination order | Explicit | Sequential: `X̃train = Xtrain × s × Ms` (channel attention coefficient `s` applied, then spatial attention coefficient `Ms` applied, both derived from the same original `Xtrain`, then both multiplied onto it together — not a cascaded/sequential-recompute CBAM) | §2.3.3, Fig. 6, Eqs. (16)-(18) | `X̃ = X × s × Ms` — both attention coefficients computed from the same pre-attention feature map `Xtrain` (channel and spatial branches run in parallel on the same input, not one feeding the other), matching Eq. (18) literally (this differs from the standard CBAM sequential-recompute design, and is recorded as such since the paper's own Fig. 6/Eq.18 is unambiguous on this point) |
| Attention placement | Explicit | "The attention mechanism is only utilized to load the network's first layer... isn't added to each convolutional layer", because deeper feature maps are too small and full-layer attention hurts anti-interference capability and adds parameters | §2.3.3, p.6 | CBAM (channel+spatial) applied once, immediately after `Layer2: Concatenation` (224×224×32) and before `Layer3: Max Pooling`. Not applied to Layer4 or Layer6 convolutions. |
| CNN classifier head | Explicit | Two fully-connected layers (128, 3 neurons) + Dropout(0.5) + Softmax | Table 2 (Layers 8-11), §2.4 | `Flatten → FC(128) → Dropout(0.5) → FC(3) → Softmax` |
| Activation function | Explicit | ReLU for all conv layers; Eq. (21) | Eq. (19)-(21) | `nn.ReLU()` |
| Loss function | Explicit | Cross-entropy | §3.3(4) | `nn.CrossEntropyLoss()` |
| Regularization | Explicit | L2 regularization ("weight decay") | §3.3(4) | Applied via optimizer `weight_decay` |
| L2 / weight-decay coefficient | Missing | Named but no numeric coefficient given | §3.3(4) | **Missing in paper.** Implementation choice: `weight_decay=1e-4`, a conventional default for Adam-trained CNNs of this depth; documented as a guess, not a paper value. |
| Optimizer | Explicit | "Adams optimizer" (paper's typo for Adam) | §3.3(3) | `torch.optim.Adam` |
| Stage/label definition (original-protocol) | Explicit | EM (Expectation-Maximization) clustering on the per-condition mean-of-3-flutes VB curve; resulting pass-index partition given in Table 1: **C1**: Initial 1–47, Normal 48–146, Severe 147–315. **C4**: Initial 1–135, Normal 136–204, Severe 205–315. **C6**: Initial 1–81, Normal 82–188, Severe 189–315 | Table 1, §3.2, Fig. 8 | Reused exactly for Protocol A (original-paper sanity reproduction) only |
| Train/test split (original protocol) | Explicit | 7:3 ratio, split independently per wear stage ("each stage's data is separated into the training set and test set in a 7:3 ratio") | §3.2 | Stratified 70/30 split within each of the 3 stage labels, pooled across C1+C4+C6 (945 total passes) |
| Repeated-training / result-averaging strategy | Explicit | 12 full training runs per model; drop the single best and single worst result; report the mean of the remaining 10 | §4, p.9 ("For each model in the article, 12 model trainings are performed to remove the most and the worst results... average results... after 10 trainings") | Protocol A: reproduce with **5 seeds**, drop none (12-run drop-best/worst is reproduced only if compute budget allows after the primary 5-seed run completes — see README for status). Documented as an adaptation if 12 runs are not ultimately run; if run, reproduced exactly (12 runs, drop best/worst, average 10). |
| Epochs | Explicit | 100 ("The amount of iterations for the training process is determined as 100") | §3.3(2), p.8-9 | `epochs=100` |
| Training scheme selection ("Scheme 5") | Explicit | Fig. 9 legend, scheme 5 (highlighted, final choice): **Iterations=100, Batch size=128, Learning rate=0.001, Decay rate=0.1, Decay times=30** | Fig. 9, §3.3 | `batch_size=128`, `lr=0.001`, single step-decay ×0.1 applied after epoch 30 (the figure's "Decay times: 30" field is read as "decay occurs at iteration 30", consistent with the body text: "after 30 iterations, the learning rate is configured to decrease to 1/10 of its initial value" — a single decay event, not 30 repeated decay steps; see Open questions) |
| Software / hardware (original protocol, informational only) | Explicit | MATLAB 2021a, NVIDIA RTX 3090 | §3.3 | Not reproduced (this project uses PyTorch on an RTX 3070 Ti Laptop GPU); framework/hardware difference documented, not treated as a fidelity gap in the algorithm itself |
| Evaluation metrics | Explicit | Confusion Matrix, Accuracy, Precision, Recall, F1-score (Eqs. 22-25) | §2.5 | `sklearn` accuracy/precision/recall/F1, macro and per-class |
| Reported sanity result — Multi-Attention-CNN | Explicit | Accuracy=0.982; per-stage (Precision/Recall/F1): Initial 0.98/0.975/0.977, Normal 0.965/0.972/0.968, Severe 0.994/0.993/0.993; Average P/R/F1 = 0.98/0.98/0.979 | Table 3, p.10 | Target for Protocol A sanity check |
| Ablation-style comparison (informational) | Explicit | Multi-CNN (fusion, no attention) Acc=0.962; Attention-CNN (single force signal + attention) Acc=0.965; Multi-Resnet Acc=0.952; Multi-Attention-Resnet Acc=0.978; Multi-Attention-CNN (proposed) Acc=0.982 — paper's own component-contribution narrative: attention helps most in Initial/Normal stages (reduces the fuzzy inter-stage boundary), multi-source fusion helps most in the Severe stage | Table 3, §4 | Used only as narrative context; not reproduced as a formal ablation in this project unless requested |

## 2. Architecture summary (as implemented, Table 2 literal)

```
Force branch:  CWT image [B,3,224,224]  -> Conv2D(16,k3,s1,same,ReLU) -> [B,16,224,224]
Vibration branch: CWT image [B,3,224,224] -> Conv2D(16,k3,s1,same,ReLU) -> [B,16,224,224]
  -> Concatenate (channel dim)                      -> [B,32,224,224]   (Table 2 Layer1+2)
  -> Channel-Spatial Attention (CBAM, Eqs 1-18)      -> [B,32,224,224]   (only insertion point)
  -> MaxPool(k3,s2,same)                             -> [B,32,112,112]   (Layer3)
  -> Conv2D(64,k3,s1,same,ReLU)                      -> [B,64,112,112]   (Layer4; also plays Fig.3's "fusion convolution" role)
  -> MaxPool(k3,s2,same)                             -> [B,64,56,56]     (Layer5)
  -> Conv2D(128,k3,s1,same,ReLU)                     -> [B,128,56,56]    (Layer6)
  -> MaxPool(k3,s2,same)                             -> [B,128,28,28]    (Layer7)
  -> Flatten -> FC(128)                              -> [B,128]          (Layer8)
  -> Dropout(0.5)                                    -> [B,128]          (Layer9)
  -> FC(3)                                           -> [B,3]            (Layer10)
  -> Softmax                                                             (Layer11)
```

## 3. Open questions / conflicts

1. **Fig. 3 vs. Table 2 fusion-convolution mismatch.** Fig. 3 ("Fusion of multi-source signals") visually depicts `Convolution (branch 1) -> Concatenation -> Convolution (fusion) -> Multi-source feature` — i.e. a convolution *after* concatenation, before what it calls the "multi-source feature." Table 2 ("Basic parameters of the neural network model") lists `Layer1: Conv2D(16)`, `Layer2: Concatenation (output 224×224×32)`, then jumps straight to `Layer3: MaxPooling` — no separate conv layer is named between concatenation and pooling. We resolve this by treating Table 2 as authoritative (it is the paper's literal hyperparameter table, used by the paper itself as "the basic structural parameters" reference for reproducibility) and note that `Layer4`'s 64-filter conv, immediately following the post-concatenation max-pool, is the closest literal match to Fig. 3's "fusion convolution." This is a genuine paper-internal inconsistency, not a choice we introduced.
2. **CWT parameters are completely unstated.** No wavelet family, scale count, or frequency range is given anywhere — a highly consequential gap since CWT is the entire preprocessing backbone of this method. Documented at length in the component table above; this is the single largest fidelity risk for Protocol A sanity reproduction.
3. **"Middle region" of the signal has no numeric definition.** Same category of gap as CWT parameters — directly affects how much of each 315-cut life-cycle signal is used per image.
4. **Fig. 9 "Decay times: 30" field is ambiguous** between "one decay event triggered at iteration 30" (matches the body text exactly) and "30 successive decay steps" (which would not make sense with only 100 total iterations and a single stated 1/10 drop). We use the body-text-consistent reading: one decay event at epoch 30.
5. **Whether the 3-axis force/vibration signals are combined (e.g., resultant magnitude) before CWT, or each axis is CWT'd separately and stacked as RGB, is never stated.** We chose the per-axis→RGB-channel mapping (see table row above) because it is consistent with the paper's own "224×224×3, RGB image" framing without requiring an unstated combination formula; a resultant-magnitude approach (single-channel CWT tiled ×3) was also considered and rejected because it would make the stated "RGB image" framing vacuous (three identical channels).
6. **Confusion-matrix sample count (283) vs. described "training set."** §4 says "the model Multi-Attention-CNN has four sample identification errors in 283 samples of the training set," but 283 ≈ 30% of 945 (C1+C4+C6 pooled), matching the *test* set size under the stated 7:3 split, not a training set. We treat this as a translation/wording slip in the paper (should read "test set") and do not attempt to reconcile it further; Fig. 10's confusion matrix is treated as the test-set confusion matrix for Multi-Attention-CNN.

# PAPER_SPEC.md -- Dynamic GIN + TGP

Source: Cao, F., Zhang, X., Zhang, Z., Wen, L., Chang, T. "Research on dynamic
graph isomorphism network for tool wear stage monitoring based on
multi-source information fusion." **Measurement 257 (2026) 119007.**
DOI: 10.1016/j.measurement.2025.119007. Code/data: not public ("available
from corresponding author upon reasonable request" -- Code availability
section). This is therefore a from-scratch **reimplementation**, verified
against the full extracted PDF text (`../../DynGIN_text.txt`, 23 pages,
password-stripped with pikepdf from the project-root PDF).

Status legend: `Explicit` = paper states the value/definition unambiguously.
`Inferable` = not stated directly but derivable from adjacent explicit facts.
`Missing` = paper gives no usable value; an implementation choice was made.
`Conflict` = paper's own text/table/figure disagree with each other.

## 1. Dataset / raw channels

| Component | Status | Paper definition/value | Location | Implementation decision |
|---|---|---|---|---|
| Raw dataset | Explicit | PHM2010, C1/C4/C6 only (only 3 tools have VB labels) | Sec 3.1 | `archive/c{1,4,6}/` |
| Raw channels used | Explicit | Force (Fx,Fy,Fz) + vibration (Vx,Vy,Vz), 6 channels; AE **discarded** ("discard the sound emission signals with more noise") | Sec 3.2 | Use archive raw CSV cols 0-5 (Fx,Fy,Fz,Vx,Vy,Vz); col 6 (AE) dropped |
| Sampling rate | Explicit | 50 kHz (NI DAQ card) | Sec 3.1 | matches archive raw data natively, no resampling |
| Pass universe (Protocol A) | Explicit | first 300 passes only ("the first 300 walking tool data of each data set are selected... discard" later/noisier passes) | Sec 3.2 | Protocol A uses runs 1-300 only |
| Stage boundaries (Protocol A, paper's own) | Explicit | Initial 1-50, Normal 51-210, Severe 211-315 (confirmed twice: Sec 3.1 prose and again exactly in Sec 3.2 text) | Sec 3.1 p.7-8 ("the (1-50)th passes...(51-210)th...(211-315)th") | Used verbatim in Protocol A; Severe therefore only has runs 211-300 available since data is truncated to 300 |

## 2. Preprocessing / sample construction

| Component | Status | Paper value | Location | Implementation decision |
|---|---|---|---|---|
| Stable-cut trim | Explicit | remove first 40,000 and last 40,000 raw points per pass, keep only "stable cutting" region | Sec 3.2 | implemented verbatim in `preprocessing.py::stable_cut_region` |
| Segmentation | Explicit | divide stable region into 10 equal portions | Sec 3.2 | implemented |
| Sample length | Explicit | 288 points (~= 1 rotational cycle at 10,400 rpm: 50000*60/10400/3 teeth ~ 96, but paper states 288 directly and ties it to "one rotational cycle" -- taken as given, not re-derived) | Sec 3.2, Table 1 (`[4,1,6,288]`) | 288 fixed |
| Stratified sampling per stage | Explicit | Initial: 5 samples/run, Normal: 2/run, Severe: 3/run | Sec 3.2 | implemented; sample start offsets = first N of the 10 portions (paper's own worked example: "collect 288-length sample points from each of the first five segments" for a 50-pass initial stage) |
| Sample-position-within-portion rule | Missing | Paper says samples are drawn "from each of the first N segments" (N=5/2/3) but never states *where inside* each 10th-portion the 288-length window starts (portion length varies per pass; typically thousands of points) | Sec 3.2 | **Implementation choice**: start each 288-sample at the temporal center of its portion. **Reason**: minimizes edge effects near portion boundaries, is a natural symmetric default. **Effect**: paper-exact resulting *samples* cannot be guaranteed byte-identical, but sample *count* (2520 total) and per-stage stratification are reproduced exactly. |
| Total sample count (Protocol A sanity target) | Explicit | Initial 50x5=250, Normal 160x2=320, Severe 90x3=270 -> 840/tool x 3 tools = **2520** | Sec 3.2 | must be verified in a unit test (`test_sample_counts`) |
| Dataset variants A/B/D/E | Explicit | A=force only, B=vibration only, D=force+vibration (both 6ch), E=force+vibration one-source generalization | Table 2 | Only **D** is reproduced (paper's own primary multi-source-fusion result; A/B/E out of scope per task instructions #21) |
| D1/D2/D3 splits | Explicit | D1: train/val={C1,C4}, test=C6; D2: train/val={C4,C6}, test=C1; D3: train/val={C6,C1}, test=C4; source tools split 0.7:0.3 train:val | Table 3, Sec 3.2 | Protocol A implements D1 (matches DC-PSR's own D1 exactly: C1+C4->C6) |

## 3. Temporal feature extraction (network)

| Component | Status | Paper value | Location | Implementation decision |
|---|---|---|---|---|
| Conv2d_1 | Explicit | Fs=14, Ks=(1,9), Pd=(0,4), ReLU, BatchNorm2d. `[4,1,6,288] -> [4,14,1,288]` | Table 1, Eq.(1) | implemented; kernel (1,9) with padding (0,4) on a [6,288] input collapses the 6-channel axis to 1 via the kernel spanning all 6 rows (kernel height must equal 6, not 1 -- Table 1's `Ks=(1,9)` notation is `(H,W)`-inconsistent with the stated output shape; see Conflict #1 below) |
| Conv2d_2 | Explicit | Fs=24, Ks=(1,5), Pd=(0,2), ReLU, BatchNorm2d, Dropout=0.5. `[4,14,1,288] -> [4,1,24,288]` (after the stated `Transpose(1,2)`) | Table 1 | implemented; produces the `Nsf=24` temporal-feature "nodes" x 288 timesteps that feed everything downstream |
| **Conflict #1: Conv2d_1 kernel-height notation** | Conflict | Table 1 lists `Ks=(1,9)` for Conv2d_1 but the *input* is `[4,1,6,288]` (channel dim=1, H=6, W=288) and the *output* is `[4,14,1,288]` (H collapses 6->1). A literal `(1,9)` kernel with `Pd=(0,4)` cannot change H from 6 to 1; the kernel's H-dim must be 6 (a "valid" conv over the full channel axis, output H = 6-6+1 = 1), while the *W*-dim kernel/padding is `9`/`4` (SAME conv on the 288 time axis). The paper table's `(1,9)` almost certainly means `(K_H, K_W)` printed for the *time* dimension only, omitting the always-full-height channel kernel that is implicit from the stated I/O shapes. | Table 1 row "Conv2d_1" | **Implementation choice**: `nn.Conv2d(1, 14, kernel_size=(6,9), padding=(0,4))`, i.e. kernel height = Nf = 6 (full-height, "valid" in H), kernel width = 9 with padding 4 (SAME in W). This exactly reproduces the stated I/O shapes `[4,1,6,288]->[4,14,1,288]`. **Reason**: shape-driven, only internally-consistent reading. **Effect**: none on downstream shapes; this is the only interpretation that satisfies Table 1's own stated output size. |

## 4. Spatial feature extraction (GASF + CNN) + cross-attention

| Component | Status | Paper value | Location | Implementation decision |
|---|---|---|---|---|
| GASF normalization | Explicit | Eq.(2): x~ = (2x-max-min)/(max-min), per-feature, into [-1,1] | Eq.(2) | implemented per-feature (per one of the 24 temporal-feature rows), over its own 288 samples |
| GASF polar mapping | Explicit | Eq.(3): r=[1/N..1], phi=arccos(x~) | Eq.(3) | implemented |
| GASF matrix | Explicit | Eq.(4): GASF(i,j)=cos(phi_i+phi_j), size NΔt x NΔt = 288x288, one image per of the 24 features -> `[4,24,288,288]` | Eq.(4), Table 1 | implemented, vectorized via `cos(phi_i+phi_j) = cos(phi_i)cos(phi_j) - sin(phi_i)sin(phi_j)` (standard GASF identity; numerically identical to the arccos/cos formula, avoids an explicit N x N python loop) |
| Conv2d_3 (spatial) | Explicit | Fs=64, Ks=5, ReLU, BatchNorm2d. Table 1 states `[4,24,288,288] -> [4,64,285,285]` | Table 1 | See **Conflict #2** below |
| Conv2d_4 (spatial) | Explicit | Fs=288, Ks=3, ReLU, BatchNorm2d, Dropout=0.5. Table 1 states `[4,64,285,285] -> [4,288,284,284]` | Table 1 | See **Conflict #2** |
| **Conflict #2: spatial CNN output sizes off by one from stated kernels** | Conflict | For a "valid" (no-padding, stride-1) `Conv2d` with kernel K on an HxW input, `out = in - K + 1`. Conv2d_3: `288 - 5 + 1 = 284`, but Table 1 states output `285`. Conv2d_4: if Conv2d_3's *true* output were 285 (table's own number), then `285 - 3 + 1 = 283`, but Table 1 states `284`. Both rows are internally short by exactly 1 vs. plain valid-convolution arithmetic; Fig.2/Fig.3/Eq.(5)-(6) give no additional padding/stride information to explain the discrepancy. | Table 1 rows "Conv2d_3", "Conv2d_4" | **Implementation choice**: use standard valid convolution (`padding=0, stride=1`) exactly as the paper's own kernel sizes state (`Ks=5`, `Ks=3`, no `Pd` listed for either row -- unlike Conv2d_1/2 which *do* list `Pd`). This yields `288->284->282`, i.e. one pixel smaller at each stage than Table 1's stated numbers. **Reason**: the *kernel sizes* (5, 3) are stated with high confidence (used again implicitly in the flatten size `284x284`≈ `80656` the paper itself lists for Attention-score, which is closer to `282x282=79524` than either 284² or 285²... see note below) -- since none of Table1's own downstream numbers (`80656`) are self-consistent with *any* of 284²=80656 exactly matches 284×284=80656! Table 1's attention-score row explicitly states `[4,288,284,284]-> [4,288,80656]`, and `284*284=80656` **exactly**. This means Table 1's own flatten step is self-consistent with the *284x284* spatial size (post Conv2d_4), which only follows algebraically if Conv2d_3's output is **285x285** (since `285-3+1=283`, not 284 -- still off by one) OR if Conv2d_3 uses `Ks=5` with input `289x289` not `288x288`. Given the flatten-size arithmetic (80656=284²) is the most load-bearing, verified downstream number in the whole table, the implementation prioritizes **matching 284x284 after Conv2d_4** by using `padding=1` on Conv2d_3 only (`288+2-5+1=286`... still not clean) -- no single (kernel,padding,stride) tuple combination reproduces both 285 and 284 from valid-conv arithmetic with the stated Ks=5/Ks=3. **Final decision**: implement literal valid convolutions (`Ks=5,Pd=0` then `Ks=3,Pd=0`), giving actual shapes `288->284->282` (flatten size `282²=79524`, close to but not identical to the paper's `80656`). The attention/fusion math (Eq. 5-6) does not depend on the *exact* numeric value of `H'xW'`, only on `Xsp1` and `Xsf3` being reshape-compatible for the dot product, which holds for any valid `H'xW'` — so this discrepancy has **no effect on model behavior or trainability**, only on the reported intermediate shape not matching Table 1's printed numbers by 2 pixels. Documented per task instruction #27 ("let cross-attention dimensions be self-consistent with Eq 5-6, keep 24x288 fused representation-- record if a minimal choice is needed"). |
| Cross-attention (Eq. 5) | Explicit | `alpha_si = Xsf3_si * Reshape(Xsp1_si,(NΔt,HxW))`; `Xsp2_si = alpha_si * Reshape(Xsp1_si,(NΔt,HxW))^T` | Eq.(5), Table 1 "Attention score"/"Spatial feature" rows | implemented: `Xsf3` reshaped to `[B,24,288]`, `Xsp1` (post Conv2d_4, `[B,288,H',W']`) reshaped to `[B,288,H'*W']`; `alpha = Xsf3 @ Xsp1_flat -> [B,24,H'*W']` (matmul, not elementwise `*`, since paper's own shapes `[4,24,288]x[4,288,80656]->[4,24,80656]` are matrix-multiply-shaped despite the `*` symbol in Eq.5); `Xsp2 = alpha @ Xsp1_flat^T -> [B,24,288]` |
| Fusion (Eq. 6) | Explicit | `XF_si = Xsf3_si + Xsp2_si`, elementwise add, shape stays `[B,24,288]`(reshaped back to `[4,1,24,288]` per Table 1) | Eq.(6) | implemented |

## 5. Static / dynamic graph generation and fusion

| Component | Status | Paper value | Location | Implementation decision |
|---|---|---|---|---|
| Graph-embedding MLP | Explicit | Hid=256, Hout=64, ReLU. `[4,1,24,288]->[4,1,24,64]` (applied to `Xsf3`, i.e. the pre-fusion temporal feature per Eq.7, not `XF`) | Table 1, Eq.(7) | implemented as `Linear(288,256)->ReLU->Linear(256,64)` applied per-node (24 nodes, each an independent 288-vector) |
| Static graph | Explicit | Eq.(8)-(9): concatenate all `Bs` samples' node embeddings along time dim before cosine similarity -> one shared `[24,24]` adjacency per batch | Sec 2.3.1, Table 1 ("Concatenate in time dimension", `[4,1,24,64]->[1,1,24,24]`) | implemented: reshape `[B,24,64]->[24,B*64]`, pairwise cosine similarity over the `B*64` axis |
| Dynamic graph | Explicit | Eq.(10)-(11): per-sample cosine similarity, no cross-batch concatenation -> `[B,24,24]` | Sec 2.3.2, Table 1 | implemented: per-sample pairwise cosine similarity over the 64-dim embedding |
| Fusion | Explicit | `A = A_static + A_dynamic` (static broadcast over batch), Eq. area between (11)-(12) | Sec 2.3.3, Table 1 "Fusion" row | implemented: `A_static[1,24,24]` broadcast-added to `A_dynamic[B,24,24]` |
| **Conflict #3: Top-k value** | Conflict | Table 1 "Fusion" row states `Topk = 288`. Sec 3.4 (hyperparameter optimization, prose) explicitly states: "the impact of different top_k values (multiples of 24) on model performance was tested... top_k was set to **6 x 24 = 144**". A `top_k=288` on a 24x24=576-entry adjacency matrix would retain exactly *half* the matrix (dense, barely-sparsified graph); `top_k=144` retains 25% (`6` edges/node average) and is explicitly the value the paper's own optimization sweep selected. | Table 1 "Fusion" row vs. Sec 3.4 prose | **Implementation choice**: `top_k=144`, since Sec 3.4 is the paper's own explicit optimization-selection narrative (not just a static spec table entry) and is far more specific/justified ("balancing efficiency and performance"). **Reason**: task instruction #31 explicitly directs this choice. **Effect**: sparser graph (75% of entries zeroed vs Table-1's implied 50%); documented, and a low-cost `top_k=288` **source-only** (C1<->C4, never touching C6) sanity variant is left as a config flag `--topk-variant` in `model.py` for optional comparison, per task instruction #31's "if compute is cheap, also run a source-validation sanity check" -- not run by default. |
| Top-k sparsification mechanics | Explicit | Eq.(12): find topk indices, set those to 1, rest to 0 (binarizes the *summed* adjacency, discards magnitude) | Eq.(12) | implemented literally: `torch.topk` per-row... **Missing**: paper doesn't say if top-k is per-row, per-matrix-flattened, or symmetric-preserving. **Implementation choice**: flatten each `[24,24]` matrix (per batch item) and take the global top-`k=144` entries (out of 576) set to 1, symmetrized by `A = max(A, A^T)` after thresholding to guarantee a valid undirected graph for GIN's symmetric-normalization step. **Reason**: per-row top-k would need a separate row-level k and isn't what "top_k values (multiples of 24)" in Sec 3.4 implies (24 itself suggests a *global* k expressed as multiples of the node count, not per-row). **Effect**: minor -- affects exact edge selection, not overall sparsity level or model capacity. |

## 6. GIN + TGP

| Component | Status | Paper value | Location | Implementation decision |
|---|---|---|---|---|
| GIN update rule | Explicit | Eq.(13): `H^l = MLP^l((1+eps^l)*H^(l-1) + Ã^l H^(l-1))`, `Ã = D^-1/2 A D^1/2` (symmetric normalization, note paper writes `D^-1/2 ... D^1/2` not `D^-1/2 A D^-1/2` -- see note) | Eq.(13) | **Note (not a blocking conflict)**: the paper's own text renders the normalization as `D_g^{-1/2} A_g^{(l)} D_g^{1/2}` (asymmetric power on the two D terms), which is *not* a valid symmetric Laplacian normalization (standard GCN/GIN practice, and the only form that preserves an undirected graph's spectral properties, is `D^{-1/2} A D^{-1/2}`). This is very likely a typesetting artifact (missing minus sign on the second D). **Implementation decision**: use the standard `D^{-1/2} A D^{-1/2}` symmetric normalization. **Reason**: this is universal GIN/GCN convention and the only numerically well-behaved reading; using the literal `D^{1/2}` would blow up features for high-degree nodes with no justification in the text. |
| eps | Explicit | eps=0.5, fixed (not learnable per Table 1's plain numeric listing, no "learnable" qualifier given) | Table 1 | fixed scalar 0.5 per layer, not a learnable `nn.Parameter` (paper gives no indication of learnability; GIN's original formulation supports either, paper is silent) |
| GIN layer 1 | Explicit | Fs=32, Ks=1, bias=True. `[4,1,24,24]+[4,1,24,288] -> [4,32,24,288]` | Table 1 | implemented as `Conv2d(1,32,kernel_size=1,bias=True)` applied per-node after the GIN aggregation (the "MLP" in Eq.13 is realized as a 1x1 Conv2d across the node/feature axis, consistent with Table1's `Ks=1` framing) |
| GIN layer 2 / 3 | Explicit | Fs=64/128, Ks=1, bias=True, eps=0.5 each | Table 1 | implemented identically, channels 32->64->128 |
| TGP layer 1 (feature pooling) | Explicit | 24->19 nodes, Fs=19, Ks=(1,18), Sd=1, Pd=(0,8) | Table 1 | `Conv2d(32,19,kernel_size=(1,18),stride=1,padding=(0,8))` -- wait, this pools the *node* dimension (24->19), so kernel operates on the node axis; implemented with the node axis as the conv's spatial dim being reduced (`H`), channel count stays at the GIN layer's output channels (32) as `Cin`, output channel count 19 acts as the *new node count* per Eq.(14)'s `DimTran` framing -- see `model.py` docstring for the exact axis bookkeeping |
| TGP layer 1 (adjacency pooling) | Explicit | Eq.(15)-(16): `S_p = W_p . Z_p` (learnable `Z_p` in R^{Skn x 1}), `A_next = S_p A S_p^T` | Eq.(15)-(16), Table 1 (`[4,1,24,24]->[4,1,19,19]`) | `S_p` implemented as a learnable `nn.Parameter` of shape `[N_pos, N_pre]` (19x24) directly (subsumes `W_p . Z_p` factorization -- the paper's `W_p` (fixed, from the pooling kernel) times learnable `Z_p` is one way to parameterize `S_p`; since `W_p`'s exact form is not given (**Missing**), we directly learn the assignment matrix `S_p` as a single parameter block, which is a strict superset of the paper's factorized form and preserves the Eq.16 pooling equation exactly) |
| TGP layer 2 | Explicit | 19->14 nodes, Fs=14, Ks=(1,9), Sd=1, Pd=(0,4) | Table 1 | same pattern |
| TGP layer 3 | Explicit | 14->10 nodes, Fs=10, Ks=(1,5), Sd=1, Pd=(0,2) | Table 1 | same pattern |
| Pooling rate | Explicit | ~0.2 selected as optimal in Sec 3.4 hyperparameter sweep (Fig 8c) | Sec 3.4 | the fixed 24->19->14->10 schedule from Table 1 is used directly (this schedule is itself already the paper's chosen "pooling rate ~=0.2" operating point, no further tuning needed) |
| TGP feature-pooling exact mechanics | Missing | Eq.(14) states `H^(l+1) = Conv2d(H^l_GIN)` after two `DimTran` (dimension transpose) calls, but doesn't specify which axes are transposed, i.e. whether the conv operates on (node, time) as (H,W) directly or after a transpose swapping them | Eq.(14) | **Implementation choice**: transpose so the conv's kernel (e.g. `Ks=(1,18)`) acts along the **node axis** (to shrink 24->19 nodes) while channel dim carries the GIN's feature channels (32/64/128) and the temporal axis (288, constant throughout all 3 TGP layers per Table 1's `288` column staying fixed) is folded into the conv's "channel-like" batch dimension via reshape, consistent with the fact that Table1 never changes 288 across TGP layers 1-3, only the node count shrinks. Documented in `model.py::TGPLayer` docstring. |

## 7. Output head

| Component | Status | Paper value | Location | Implementation decision |
|---|---|---|---|---|
| Output pooling | Explicit | `AdaptiveAvgPool2d(output_size=1)`. `[4,128,10,288]->[4,128,1,1]` | Table 1, Eq.(17) | implemented |
| Output MLP | Explicit | Hin=128, Hout=3 | Table 1 | `Linear(128,3)` |
| Loss | Explicit | Cross entropy | Sec 3.4 | `nn.CrossEntropyLoss` |

## 8. Training hyperparameters (Protocol A, Scheme 1 -- paper's chosen optimum)

| Component | Status | Paper value | Location | Implementation decision |
|---|---|---|---|---|
| Epochs | Explicit | 50 | Sec 3.4, Fig.9 Scheme 1 | 50 |
| Batch size | Explicit | 4 | Sec 3.4, Fig.9 Scheme 1 (also matches Table 1's `[4,...]` shapes throughout, confirming batch=4 was literally the shape used to print Table 1) | 4 |
| LR | Explicit | 0.0001 | Sec 3.4, Fig.9 Scheme 1 | 1e-4 |
| Optimizer | Explicit | Adam | Sec 3.4 prose | `torch.optim.Adam` |
| L2 regularization factor | Explicit | 0.1 (prose: "the regularization factor is set to 0.1") | Sec 3.4 prose | `weight_decay=0.1` in Adam |
| LR plateau schedule | Explicit | "if there is no improvement in the index after 10 iterations, the learning rate is reduced by half" | Sec 3.4 prose | `ReduceLROnPlateau(mode='max', factor=0.5, patience=10)` on validation accuracy |
| **Terminology ambiguity: Fig.9 "Decay rate/factor=0.1" vs prose "L2 factor=0.1" vs "reduced by half"** | Conflict (documented, not blocking) | Fig.9's four hyperparameter-scheme comparison plot has an axis/legend labeled with a "decay"-type term whose exact relation to the L2 factor (0.1, stated in prose) and the LR-halving rule (also prose, unrelated numeric factor of 0.5) is not disambiguated by the figure caption alone | Fig.9 caption + Sec 3.4 prose | **Implementation decision**: treat "L2 regularization factor = 0.1" (prose, explicit) as `weight_decay=0.1`, and "reduced by half" (prose, explicit) as the *separate* `ReduceLROnPlateau(factor=0.5)` mechanism. The two are **not** conflated into a single 0.1 applied twice. **Reason**: prose is unambiguous for both individually; only Fig.9's plot legend is ambiguous, and prose takes precedence per task instruction #36. |
| Random seed | Missing | not stated | -- | 42 for Protocol A sanity run (task instruction #70 default); Protocol B unified uses the shared project seed set {42,52,62,72,82} |

## 9. Reported reference numbers (paper_reported, sanity targets only -- never copied into our_reproduction)

| Metric | Value | Location |
|---|---|---|
| D1 Accuracy | 95.71% | Table 6 / Sec 4.4 prose |
| D2 Accuracy | 96.07% | Table 6 |
| D3 Accuracy | 97.74% | Table 6 |
| D avg / std | 96.51% / 0.86 | Table 6 |
| Total parameters | 321,002 | Sec 3.4 prose |
| Model memory | 246.74 MB | Sec 3.4 prose |
| Preprocessing time / model compute time / total latency (per sample) | 0.1129 s / 0.0296 s / 0.6425 s (incl. 0.5s "data packet acquisition") | Sec 3.4 prose |
| GIN layers | 3 (optimal, Fig.8b sweep) | Sec 3.4 |
| Pooling rate | ~0.2 (optimal, Fig.8c sweep) | Sec 3.4 |

## 10. Reproduction risk summary

- **Low risk**: dataset/channels/pass-universe/stage-boundary/preprocessing/sample-count (all Explicit, directly checkable via `test_sample_counts`).
- **Low risk**: temporal conv, GASF, GIN update rule, output head (all Explicit with only cosmetic notational issues).
- **Medium risk**: spatial CNN exact pixel dims (Conflict #2) -- does not affect trainability, only reported intermediate shapes differ by ~2px from Table 1.
- **Medium risk**: TGP exact axis-transpose mechanics (Missing) -- functionally sound (produces correct node-count schedule 24->19->14->10 and pools both H and A per Eq.14-16), but exact conv-kernel-orientation choice is an implementation decision, not verified against paper code (paper has no public code).
- **Medium risk**: Top-k=144 vs 288 (Conflict #3) -- resolved in favor of the paper's own explicit optimization narrative; a 288 variant is available as a flag for optional comparison.
- Overall paper-fidelity: **High** for the data pipeline and macro-architecture; **Medium** for a handful of pixel/axis-level implementation choices in the spatial-CNN and TGP internals, all individually documented and non-blocking for correct end-to-end training.

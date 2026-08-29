# PAPER_SPEC.md -- DP2Net

Source: Lai, X., Zhang, K., Zheng, Q., Zhao, M., Ding, G., Tang, B., Li, Z.
"DP2Net: A discontinuous physical property-constrained single-source domain
generalization network for tool wear state recognition." **Mechanical
Systems and Signal Processing 215 (2024) 111421.** DOI:
10.1016/j.ymssp.2024.111421. Data availability: "The authors do not have
permission to share data" -- no public code or PHM2010-specific artifacts.
This is a from-scratch **reimplementation**, verified against the full
extracted PDF text (`../../DP2Net_text.txt`, 14 pages, password-stripped
with pikepdf).

Status legend: `Explicit` / `Inferable` / `Missing` / `Conflict` (see
`../dynamic_gin_tgp/PAPER_SPEC.md` for definitions -- identical convention).

## 1. Dataset / task definition

| Component | Status | Paper value | Location | Implementation decision |
|---|---|---|---|---|
| Raw dataset (this baseline's scope) | Explicit | PHM2010; C1=source domain, C4 and C6=target domains | Sec 4.1(1), 4.2 | `archive/c{1,4,6}/` (machining-experiment dataset M1-M5 is out of scope per task instructions #2/#41-67, which only ask for PHM2010) |
| Raw channel | Explicit | "the force signal in the x-direction is taken as the monitoring data" -- **Fx only** | Sec 4.2 | archive raw CSV column 0 (Fx) only; Fy/Fz/Vx/Vy/Vz/AE unused |
| **Conflict #1: sampling frequency** | Conflict | Sec 4.1(1) states PHM2010 "sampled at 5 kHz". This project's actual PHM2010 raw archive is 50 kHz (confirmed: `archive/c1/c1/c_1_001.csv` has ~5.5M rows for a ~110s pass, i.e. ~50kHz; also independently confirmed by every other baseline in this project and by the Dynamic-GIN paper's own explicit "50 kHz" statement for the same dataset). Eq.(4)'s own worked numbers (`k=25` for PHM2010, `Nspeed=10400 rpm`, `n=3` teeth, `kpool=4`) only reproduce ~25 under `fs=50000`: `k = (50000*60/10400)/3/4 = (288.46)/3/4 = 96.15/4 = 24.04 ~= 24-25`. Under the paper's own stated `fs=5000`: `k = (5000*60/10400)/3/4 = 28.85/3/4 = 9.6/4 = 2.4`, nowhere near the paper's own reported `k=25`. | Sec 4.1(1) prose vs. Eq.(4) + Sec 4.3 prose ("The k of the PHM 2010 dataset was estimated to be 25") | **Implementation decision**: use the real 50 kHz archive data (native, no downsampling) and the paper's own explicitly reported `k=25`. **Reason**: task instruction #44 directs this; the physics arithmetic is only self-consistent at 50kHz, and 50kHz is independently confirmed by three other sources (this project's raw files, the Dynamic-GIN paper on the identical dataset, and the original PHM2010 challenge documentation convention). "5 kHz" in Sec 4.1(1) is treated as a typo/error in the paper's own text. |
| Sample length | Explicit | 4608 points, "contains 16 cycles of data" | Sec 4.2 | 4608, taken from native 50kHz signal (not resampled) |
| **4-stage original definition (I/II/III/IV)** | Explicit for IV, Missing for I/II/III boundary | Stage IV ("failure"): mean VB > 0.3 mm (GB/T 16460-2016, explicit numeric threshold). Stages I (initial)/II (steady)/III (accelerated) wear: "according to [41]" -- **no threshold or rule given in this paper's own text**, only a reference to Zhang, Zhu, Duan, Li, "Tool wear estimation and life prognostics in milling: Model extension and generalization," MSSP 155 (2021) 107617 | Sec 4.2 | See "Protocol A stage-boundary handling" below -- **Critical missing item**, task instruction #48 |
| Samples per class | Explicit | "Each class contains 2000 samples" (4 classes x 2000 = 8000 samples/tool for PHM2010) | Sec 4.2 | see sampling-construction row below |
| Sample-extraction rule (stride/overlap/exact positions) | Missing | paper gives no stride, overlap, or start-index rule for slicing 4608-point windows out of each run's stable-cutting signal to hit exactly 2000/class | Sec 4.2 | **Implementation choice**: uniform random sampling (fixed seed) of 4608-length windows from each run's post-low-pass-filter, post-stable-cut signal, drawn from runs belonging to that class, until 2000 samples/class are collected; recorded in `sample_manifest.csv` (condition, run_id, class, start_idx, end_idx) per task instruction #49. **Reason**: paper is silent; random uniform sampling is the least-assumption default and is fully reproducible via the fixed seed + manifest. |
| Train/val/test split | Explicit | Source (C1): 70% train / 30% validation. Targets (C4, C6): 100% test, never seen in training | Sec 4.2 | implemented; C4/C6 held out entirely for Protocol A |
| **PHM2010 Table 1 physical parameters** | Explicit (partial) | `Nspeed=10400 rpm, Ap=0.2mm, Ae=0.125mm, f=1555 mm/min` for C1/C4/C6 (Case I row) | Table 1 | used directly for `k` (Eq.4) and `P`/`L` (Eq.5-6) where applicable |
| **Critical missing physical parameters for exact Vst (PHM2010)** | Missing | Eq.(5)'s `P = (Ap/tanβ) / ((D*pi)/n)` requires **tool diameter D** and **helix angle β**. Table 1 gives these only for the authors' own machining-experiment tool ("three-tooth integral end mill... diameter: 16 mm, helix angle: 35 deg", Sec 4.1(2)) -- **not for the PHM2010 tools**. The PHM2010 challenge's official documentation (Li, Lim, Zhou, Huang, Phua, Shaw, Er, ref [40]) specifies **6mm-diameter, 3-flute ball-nose tungsten-carbide** cutters (this is the standard, widely-cited PHM2010 tool spec used consistently across the tool-wear-monitoring literature, and matches the Dynamic-GIN paper's own PHM2010 description of "ball-end tungsten carbide cutters"); **helix angle is not published anywhere in the PHM2010 documentation.** | Table 1 (only gives machining-exp tool geometry, not PHM2010 tool geometry); Sec 4.1(2) | **Implementation choice**: `D = 6 mm` (PHM2010's documented/consensus ball-nose cutter diameter, used project-wide, cf. `代码/` comments and other baselines' PAPER_SPEC docs referencing the same PHM2010 tool spec). `β (helix angle)`: **no PHM2010-specific value exists in any source** we could locate; the paper's own machining-experiment tool uses 35 deg. **Decision: β = 30 deg**, a standard/typical helix angle for ball-nose carbide end mills in this diameter class (documented as an assumption, not a paper value). **Effect**: `P` (the monotonic-rise fraction of `Vst`'s period) is therefore an approximation, not a paper-exact value; `L` (the period, Eq.6) does **not** depend on D/β and is paper-exact. This is flagged in FINAL_REPORT.md as directly affecting `Vst`'s shape (hence the `L_MSE` physical constraint) and is the single largest source of Protocol-A "adapted, not exact" uncertainty for this method. |

## 2. Preprocessing

| Component | Status | Paper value | Location | Implementation decision |
|---|---|---|---|---|
| Low-pass filter, PHM2010 | Explicit | cutoff = 1733 Hz ("spindle frequency less than six times [7]") | Sec 4.2 | Butterworth low-pass, order 4 (order not stated -- **Missing**, standard choice, documented), cutoff=1733Hz, applied to Fx before windowing |
| Low-pass filter rationale/order | Missing | filter order/type not specified | Sec 4.2 | 4th-order Butterworth (zero-phase, `scipy.signal.filtfilt`) -- standard default, documented as an implementation choice |
| Stage-IV threshold | Explicit | mean(flute_1,flute_2,flute_3) VB > 0.3mm -> Failure. **Note**: this is a `mean`, explicitly stated for DP2Net's own 4-class scheme -- in contrast to this project's own real-data-confirmed convention of `VB=max(...)` used everywhere else in DC-PSR's unified labels (see `baselines/mtf_avitk/data/label_utils.py` docstring). Protocol A uses DP2Net's own `mean` convention (paper-native); Protocol B uses this project's `max` convention (per task instruction #13, "do not invent a new E/M/L scheme -- reuse the current DC-PSR authoritative one") | Sec 4.2 | Protocol A: `VB = mean(flute_1,2,3)`, threshold 0.3mm for stage IV. Protocol B: reuses `max(...)` via `data/label_utils.py`, condition-relative E/M/L (no 4th "failure" class) |

### Protocol A stage-boundary handling (I / II / III)

Per task instruction #48, since [41]'s exact numeric wear-rate/percentage
thresholds for the I/II/III boundary are not reproducible from this paper's
own text (a websearch for ref [41]'s abstract/content only returned the
generic, non-quantitative "bathtub curve" description common to all
tool-wear literature -- rapid initial wear, slow steady wear, then
accelerating wear near end-of-life -- with no numeric threshold recoverable
from public abstract/citation text; the full paper is paywalled and was not
accessible), this implementation:

- **Does NOT claim exact reproduction of ref [41]'s I/II/III boundaries.**
- Uses a documented, reproducible proxy for Protocol A only: **wear-rate
  change-point detection** on the smoothed `mean(flute_1,2,3)` curve --
  Stage I ends at the first local minimum of the *rate* (`dVB/dpass`) after
  the initial transient (i.e., where wear-rate first drops below its
  post-transient median, marking the initial-wear-to-steady-wear
  transition), Stage III begins where the *smoothed* wear-rate exceeds
  1.5x the steady-state (Stage II) median rate (accelerated-wear onset),
  Stage IV (failure) uses the paper's own explicit `mean VB > 0.3mm` rule
  and takes priority over III. This is implemented in
  `preprocessing.py::assign_paper_native_4stage` and is explicitly labeled
  `"Protocol-A stage boundaries: reproduced proxy, NOT verified against
  ref [41]'s exact criterion -- see PAPER_SPEC.md"` in code comments and
  in all Protocol-A output files.
- Because of this, **Protocol A for DP2Net is an "adapted reproduction"
  for the stage-boundary dimension specifically**, even though every other
  component (S, G, Vst period L, WDCNN, MMD, low-pass, sample length,
  train/val split, hyperparameters) follows the paper exactly. This is
  called out explicitly in FINAL_REPORT.md's confidence rating.

## 3. S module (discontinuous-physical-property-guided spatial attention)

| Component | Status | Paper value | Location | Implementation decision |
|---|---|---|---|---|
| Receptive-field kernel size `k` | Explicit | Eq.(4): `k = (fs*60/Nspeed)/n / kpool`. PHM2010: `k~=25` (paper's own reported estimate) | Eq.(4), Sec 4.3 | `k=25` fixed (matches our fs=50000 computation of 24.04, rounded to paper's stated 25) |
| `kpool` | Explicit | 4 (first avg-pooling layer kernel/stride, "for high computational efficiency") | Sec 4.3 | `AvgPool1d(kernel_size=4, stride=4)` as S's first layer |
| S internal structure | Missing (mechanism explicit, exact layer count/order not tabulated) | Fig.4 + prose: spatial attention "constrained by discontinuous physical properties," receptive field = k (one-tooth signal segment post-downsampling), includes BN + ReLU (per Fig.4 caption) | Sec 3.1, Fig.4 | **Implementation choice**: `S = AvgPool1d(kpool=4) -> Conv1d(1,1,kernel_size=k=25,padding=k//2) -> BN -> ReLU -> Sigmoid` producing a per-timestep attention weight `Wa` over the pooled signal, then upsampled (nearest) back to full length and elementwise-multiplied with the raw filtered input to give `Fa = Wa * input`. **Reason**: this is the minimal structure consistent with (a) Fig.4's stated BN+ReLU, (b) the k-sized receptive field being the attention conv's kernel (not a separate feature-extraction conv), (c) Fig.11's visualization showing `Wa` as a per-timestep weight map with the same temporal length as the input. Sigmoid (to bound attention in [0,1]) is an implementation choice (Missing in paper) -- standard for spatial-attention gating. |
| S output | Explicit | `Fa` (weighted features), used as input to G | Sec 3.1 | `Fa` shape = same as input `[B,1,4608]` |

## 4. G module (generation)

| Component | Status | Paper value | Location | Implementation decision |
|---|---|---|---|---|
| G structure | Explicit | "one pooling layer, three convolution layers, and a transposed convolution layer" | Sec 3.2, Fig.5 | implemented as: `AvgPool1d(kpool=4) -> Conv1d(1,1,k) -> Conv1d(1,4,k) -> Conv1d(4,4,k) -> ConvTranspose1d(4,1,k,stride=kpool)` (upsamples back to 4608) |
| G conv channel counts | Explicit | "the number of channels of the convolutional layer (1, 4, and 4, respectively)" | Sec 3.2 | 1 -> 4 -> 4 channels for the 3 conv layers, as stated |
| G other params (pooling/conv kernel/stride) | Explicit | "consistent with those in S", i.e. same `kpool=4`, same `k=25` kernel | Sec 3.2 | reused from S's config |
| AdaIN | Explicit (mechanism), Missing (exact placement/style-source) | "AdaIN further increases the diversity in the generation process" | Sec 3.2 | **Implementation choice**: AdaIN applied after each conv layer in G, with style statistics (mean/std) drawn from a per-sample random Gaussian noise vector projected through a small learnable affine layer (standard StyleGAN-style AdaIN noise-injection, since the paper cites AdaIN's origin as [37]=Karras et al. StyleGAN but gives no PHM2010-specific style-source). **Reason**: paper explicitly cites AdaIN's role as "increasing diversity," which requires *some* external stochastic style source; random noise is the standard, minimal-assumption choice consistent with StyleGAN's own usage. |
| Vst (standard trend vector) | Explicit | periodic: monotonic rise [-1,1] for a fraction `P` of each period, then constant 0 for the rest; period `L` | Eq.(5)-(6), Sec 3.2 | implemented in `preprocessing.py::build_vst`; **P depends on the missing D/beta** (see Sec 1 Critical-missing row above) |
| L (period length) | Explicit | Eq.(6): `L = fs*(60/Nspeed)/n` (paper-exact, no missing params) | Eq.(6) | `L = 50000*(60/10400)/3 = 96.15` samples/tooth-period, matches `k~=24` (i.e. `L/kpool~=k`), internally consistent with Eq.(4) |
| `L_MSE` (Wg <-> Vst constraint) | Explicit | Eq.(7): `L_MSE = (1/nS) * sum((Wg_i - Vst)^2)` | Eq.(7) | implemented, `nn.MSELoss()(Wg, Vst)` |

## 5. Training / optimization

| Component | Status | Paper value | Location | Implementation decision |
|---|---|---|---|---|
| MMD | Explicit | Eq.(9): Gaussian-kernel MMD between SD and GD features (pre-classification-layer) | Eq.(9) | implemented, `k(x,y)=exp(-\|\|x-y\|\|^2/(2*gamma^2))` |
| `gamma` (MMD kernel width) | Missing | not given a numeric value | Sec 3.3 | **Implementation choice**: median heuristic (`gamma` = median pairwise distance among the batch's SD+GD features) -- standard, parameter-free MMD default when a paper doesn't specify `gamma` explicitly. Documented. |
| G loss direction | Explicit | Eq.(10): `L_G = L_MSE - alpha*L_MMD` (**minus** sign -- G *maximizes* diversity/MMD while honoring the physical MSE constraint) | Eq.(10) | implemented literally as `L_G = L_MSE - alpha * L_MMD`; **task instruction #57 explicitly warns not to flip this sign** -- verified against the paper text directly (line: "LG = LMSE -alpha*LMMD") |
| alpha | Explicit | 20 ("experimentally tried") | Sec 4.3 | 20 |
| F = WDCNN | Explicit (delegated) | "structure and parameters adopted are consistent with those in [38]" = Zhang, Li, Peng, Chen, Zhang, "A deep convolutional neural network with new training methods for bearing fault diagnosis under noisy environment and different working load," MSSP 2018 | Sec 3.3 | **Canonical WDCNN** implemented per [38]'s well-known, widely-reproduced public architecture: wide first-layer kernel (64@1x64, stride 16) -> BN -> ReLU -> MaxPool -> 4x(Conv 3x3, 16-64 channels, BN, ReLU, MaxPool) -> AdaptiveAvgPool -> FC(100) -> FC(num_classes). Flagged in code as "External dependency / inferred component -- canonical WDCNN [38], not independently re-derived from this paper's own text" per task instruction #58. Input length adapted from [38]'s original 2048/6000-point vibration window to this task's 4608-point Fx window (first-layer kernel/stride re-scaled proportionally so the final feature-map length before global pooling stays in a sane range; documented in `model.py`). |
| Task loss | Explicit | Eq.(11): `L_task = mean(CE(Y_S,y_S)) + mean(CE(Y_G,y_G))` (source + generated, equal weight) | Eq.(11) | implemented |
| Two-stage training (Algorithm 1) | Explicit | Stage 1: train S+F on source only, CE loss, 100 epochs. Stage 2: freeze trained S, train G (Eq.10) + F (Eq.11) jointly, 100 epochs | Algorithm 1 | implemented exactly: `pretrain_S_F()` (100 epochs) then `train_G_F()` (100 epochs, S frozen) |
| S optimizer/LR | Explicit | Adam, lr=0.001 | Sec 4.3 | Adam, lr=1e-3 |
| F optimizer/LR schedule | Explicit | Adam, lr=0.001, "cosine attenuation strategy... attenuation period was 20 epochs" | Sec 4.3 | Adam base lr=1e-3, `CosineAnnealingLR(T_max=20)` |
| G optimizer/LR | Explicit | Adam, lr=0.00001 (constant, "always") | Sec 4.3 | Adam, lr=1e-5, no schedule |
| Batch size | Explicit | 64 | Sec 4.3 | 64 |

## 6. Reported reference numbers (paper_reported, sanity targets only)

| Metric | Value | Location |
|---|---|---|
| Task 1 (C1->C4) Accuracy | 90.91% | Table 2 |
| Task 2 (C1->C6) Accuracy | 87.66% | Table 2 |
| PHM2010 std across tasks | 1.63 | Sec 4.3 / Fig.8 |
| Improvement vs WDCNN(raw) benchmark, PHM2010 avg | +12.69% | Sec 5 Conclusion |
| WDCNN(raw) Task1/Task2 | 82.21% / 71.59% | Table 2, for sanity cross-check of our WDCNN(raw) ablation baseline |

## 6b. Empirical finding: Stage IV (failure) never occurs in this project's real PHM2010 archive for C1/C4/C6

Discovered while building `preprocessing.py::assign_paper_native_4stage` and
confirmed directly against `archive/c{1,4,6}/c{1,4,6}_wear.csv`: this
project's archive stores flute wear in **micrometers** (values range
~30-220 across all three tools, the standard PHM2010 flank-wear unit used
throughout the literature). The paper's own explicit Stage-IV rule ("mean
VB > 0.3mm", i.e. 300 um) is **never reached by any of C1, C4, or C6**
within their recorded 315 passes, under either the `mean(flute_1,2,3)`
convention the paper states for its own 4-class scheme, or the
`max(flute_1,2,3)` convention this project uses elsewhere (max mean-VB:
C1=165um, C4=203um, C6=216um; max of the single worst flute: C1=173um,
C4=211um, C6=235um -- all well below 300um).

This means the paper's own claim of achieving 4 balanced classes x 2000
samples each for PHM2010 (Sec 4.2) **cannot be independently reproduced
from this project's real archive data for these three specific tools** --
not because of a preprocessing bug, but because C1/C4/C6 are well-documented
in the wider PHM2010 literature as never reaching full "failure" (they are
commonly used specifically to study run-in/steady/accelerated wear without
catastrophic failure). Either the paper used a different VB
reading/convention we could not identify, or a different set of PHM2010
tools/runs than C1/C4/C6, or this is an inconsistency in the paper's own
text.

**Implementation decision**: Protocol A for this baseline is therefore run
as an **adapted 3-class (I/II/III) scheme** for C1/C4/C6 -- Stage IV's
`assign_paper_native_4stage` code path is kept exactly as specified (so it
would trigger correctly on real failure data), but empirically produces 0
samples for all three tools, so the sample manifest only builds
`SAMPLES_PER_CLASS` windows for I/II/III (2000/class would be requested for
IV too, but `build_sample_manifest` silently produces an empty class rather
than erroring, and this is surfaced explicitly in `README.md` and
`FINAL_REPORT.md` rather than hidden). This is called out as a further,
data-driven (not paper-text) source of "adapted reproduction, not exact
reproduction" status for Protocol A, on top of the I/II/III boundary-proxy
issue in Sec 1.

## 7. Reproduction risk summary

- **Low risk**: low-pass filter cutoff, sample length, k=25/kpool=4, Vst period L, MMD sign/direction, alpha, two-stage Algorithm 1, all training hyperparameters, WDCNN as F (canonical, well-documented architecture).
- **Medium risk**: S/G internal layer-by-layer structure beyond the stated channel counts and shared k/kpool (Missing details, minimal-assumption implementation chosen).
- **Medium-high risk / Critical missing**: PHM2010 tool diameter D and helix angle beta for `Vst`'s `P` fraction -- assumed (D=6mm PHM2010-consensus, beta=30deg placeholder), directly affects the physical-constraint term's exact shape (not its period).
- **Medium-high risk / Critical missing**: exact I/II/III stage-boundary rule from ref [41] -- Protocol A uses a documented wear-rate-change-point proxy, explicitly NOT claimed to be an exact reproduction of ref [41].
- **High risk / empirical finding (sec 6b)**: Stage IV (failure, mean VB>0.3mm) never occurs for C1/C4/C6 in this project's real PHM2010 archive under either mean- or max-flute convention -- Protocol A is therefore an adapted **3-class (I/II/III)** scheme for PHM2010, not the paper's claimed 4-class scheme. This is a data-driven finding, not a preprocessing choice.
- Overall paper-fidelity: **Medium**. The mechanism (S -> G -> F, physical constraints, two-stage training, loss functions) is reproduced with high confidence; two *physical-parameter* and *stage-boundary* inputs are irreducibly missing from the paper's own text and are handled as documented assumptions, not silently guessed defaults; and Stage IV is empirically unreachable for this project's PHM2010 tools regardless of implementation choice.

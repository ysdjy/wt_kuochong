# PAPER_SPEC.md — HTT-Net Paper-to-Code Specification

Source paper: Xue, Z., Chen, N., Wu, Y., Yang, Y., Li, L. (2023). "Hierarchical
temporal transformer network for tool wear state recognition." *Advanced
Engineering Informatics*, 58, 102218. https://doi.org/10.1016/j.aei.2023.102218

Extracted by reading the full text of
`Hierarchical temporal transformer network for tool wear state recognition.pdf`
(14 pages) shipped in the project root. No author source code was found
(paper's Data Availability statement says "The data that has been used is
confidential"; no GitHub/code link is given anywhere in the paper).

> **Real results now available.** HTT-Net has been trained and evaluated on
> the REAL, original PHM2010 feature data (recovered from a supplementary-
> materials folder the user pointed to — `补充材料/`; verified genuine by
> reproducing the manuscript's exact Table 8 numbers) and compared against
> the manuscript's own real B1-B12 results. See `README.md` for the full
> results table, the recovery story, and the `VB = max(...)` convention
> correction it surfaced.

Legend for the "Status" column:
- **A (Explicit)** — paper states the value/formula directly.
- **B (Inferable)** — not stated as a number, but uniquely determined by a
  formula, a referenced base architecture (Swin Transformer), or an
  unambiguous shape constraint elsewhere in the paper.
- **C (Missing)** — paper gives no way to derive a unique value. An
  implementation choice was made and is justified below the table.

## 1. Component table

| Component | Paper explicitly specifies? | Paper value / definition | Source in paper | Implementation decision |
|---|---|---|---|---|
| Input (original paper protocol) | A | Raw 3-axis force signal, non-overlapping window of length 2000, shape 2000×3 | §3.2, "small samples of size 2000 × 3" | Not used for our unified experiment (see §3 below); implemented only as an optional shape config in `model.py` |
| Input (our unified DC-PSR protocol) | N/A (paper doesn't define this) | — | — | X ∈ ℝ^(B×12×d): L=12 sliding window over condition-relative-online engineered features, d = number of selected features (~45). Reused byte-for-byte from `代码/main_experiment_3_fgds_psi_optimized.py` |
| Linear Embedding | A | Maps channel dim C0 → C1 (higher dim), applied once before Stage 1 | §2.1, Table 1 | `nn.Linear(in_dim, embed_dim)` |
| Number of stages | A | 4 stages (Stage 1..4), each stage operating at L, L/2, L/4, L/8 with channel C1, 2C1, 4C1, 8C1 | Table 1 | 4 stages, matching Table 1 exactly |
| Token Merging placement | A | After Stage 1, 2, 3 (not after Stage 4); halves sequence length, doubles channel count | Table 1, §2.1 | Merge module placed between stages i and i+1 for i=1,2,3 |
| Token Merging formula | B | Not given as an equation, only as a shape transform (L→L/2, C→2C). Paper explicitly frames Token Merging as an adaptation of Swin Transformer's Patch Merging layer for 1D data | §1.3(2), §2.1, Table 1 | Concatenate each pair of adjacent tokens along the channel axis (C→2C, matching Swin's PatchMerging pattern of concatenating neighbors then linearly projecting), then `LayerNorm(2C) → Linear(2C→2C)`. This is the direct 1D analogue of Swin's 2D PatchMerging (which concatenates 4 neighbors→4C, projects to 2C); for 1D, 2 neighbors→2C is the natural equivalent, so channel count matches Table 1 without needing a channel-reducing projection. |
| 1D Temporal Transformer Block internals | A | LayerNorm → (W-MSA or SW-MSA) → residual → LayerNorm → MLP → residual | §2.1, Fig. 1 | Standard pre-norm transformer block, implemented exactly |
| W-MSA | A | Multi-head dot-product self-attention computed independently inside each non-overlapping window of length T | §2.2, Eq. (2), Eq. (6) | `WindowAttention1D` module, causal-free, operates per window |
| SW-MSA | A | Same as W-MSA but windows are shifted by ⌊T/2⌋ before partition (cyclic shift), with the wrapped boundary window built by concatenating the two non-adjacent end segments; a reverse shift restores original order after attention | §2.2, Fig. 2 | Cyclic `torch.roll` shift by −T/2, window partition, **attention mask** to prevent the wrapped (non-adjacent) segments inside a merged boundary window from attending to each other (standard Swin masking, needed because the paper's own text says naively attending across the concatenated segments "prevents the model from extracting reasonable features"), then reverse roll |
| Attention mask (SW-MSA) | B | Paper states the *problem* (concatenated window mixes non-adjacent time segments) and that shifting resolves it, but does not give the masking formula itself | §2.2 | Standard Swin-style additive mask: window pairs whose members come from different pre-shift segments get −100 (soft -inf) added to attention logits before softmax |
| Relative position bias | A | Learnable table of size (2L_m−1) indexed by a relative-position index matrix; added to attention logits before softmax | §2.3, Eq. (7)-(11) | `nn.Parameter` bias table of shape `(2*window_size-1, num_heads)`, gathered per-window via a precomputed relative-index buffer; added to QKᵀ/√d before softmax, exactly as Eq. (7) |
| LayerNorm | A | Standard formula, ε ≈ 1e-5 | Eq. (1) | `nn.LayerNorm(C, eps=1e-5)` |
| MLP Block | A / B (mixed) | "MLP Block consists of: Linear, GELU, Dropout, and Layer Normalisation"; separately, "Linear1 shrinks channel dim by x4 ... Linear2 magnifies by x4" | §2.1 | Channel shrink/expand ×4 implemented literally (explicit, **A**): `Linear(C, C/4) → GELU → Dropout → Linear(C/4, C)`. This is an **inverted bottleneck** (shrink-then-expand), the opposite of the standard Transformer FFN — implemented literally since that part of the text is unambiguous. The "Layer Normalisation" ingredient's *placement* is ambiguous (**B**, inferable, not explicit): a literal post-Linear2-pre-residual placement was implemented and unit-tested first, and it **failed** `test_single_batch_overfit` (loss plateaus at ~0.8, 71% train accuracy, vanishing gradients after ~150 steps) — a hybrid pre-norm/post-norm residual stack, a known unstable configuration. It was replaced with the standard interpretation: the paper's "Layer Normalisation" = the pre-MLP `norm2` already required by a standard (pre-norm) Swin-style block, i.e. LN→MLP→+residual with no extra norm after Linear2. This is recorded here per the "no silent common-sense substitution" rule: the substitution was empirically forced by a failing test, not applied blindly. |
| GELU | A | tanh approximation given explicitly | Eq. (4) | `nn.GELU(approximate='tanh')` |
| Residual connections | A | Present around both the attention block and the MLP block (implied by "Block" terminology and standard transformer diagram in Fig. 1) | §2.1, Fig. 1 | Standard pre-norm residual: `x = x + Attn(LN(x))`, `x = x + MLP(LN(x))` |
| Classifier head | B | Table 1: "Output: Class Number × 1" following Stage 4 (which is L/8 × 8·C1) — paper does not state how the L/8 sequence dimension is collapsed to a single vector before the FC layer | Table 1, §2.1 | `LayerNorm(8·C1) → mean-pool over the (padded-aware) sequence dimension → Linear(8·C1, num_classes)`. Mean pooling is the standard Swin Transformer classification head choice and the simplest operation consistent with "Class Number × 1" output. |
| Loss function | B | Not named explicitly anywhere in the paper | — | Standard multi-class cross-entropy (only established convention consistent with an FC→Class-Number head trained via confusion-matrix/accuracy/precision/recall/AUC evaluation) |
| Optimizer | A | AdamW | §3.3.1 | `torch.optim.AdamW` |
| Learning rate | A | 0.0001 | Table 6 | `lr=1e-4` (original-protocol default; **not** used for the unified-protocol run, see §4) |
| Batch size | A | 16 | Table 6 | `batch_size=16` (original-protocol default) |
| Weight decay | A | 0.05 | Table 6 | `weight_decay=0.05` (original-protocol default) |
| LR scheduler | C | Not mentioned | — | **Missing in paper.** Implementation choice: none (constant LR), consistent with "no scheduler mentioned" being the null hypothesis. Documented, not silently assumed. |
| Epochs | C | Only "the model converged before Epoch 5" is stated qualitatively (Fig. 8); no total epoch count or stopping rule is given | §3.3.2 | **Missing in paper.** Implementation choice: reuse the existing manuscript's own protocol for all deep baselines — `EPOCHS=120`, `PATIENCE=18` early stopping on a validation score (see `代码/main_experiment_3_fgds_psi_optimized.py`), so HTT-Net is trained under the *same* budget as every other baseline in the unified comparison. This is a deliberate fairness adaptation, not a guess at the original paper's setting. |
| Embedding dimension (C1) | C | Never given a numeric value | — | **Missing in paper.** Implementation choice: `embed_dim=32`, matching the smallest channel width already used by the existing TCN-GRU baselines in this project (`BEST_ARCH["channels"][0] = 32`) for a comparable parameter budget at L=12. |
| Number of attention heads | C | Never given for HTT-Net itself (only the *comparison* Transformer/Informer baselines in §3.4.1 are stated to use 3 heads, which describes a different model, not HTT-Net) | §3.4.1 (comparison models only) | **Missing in paper.** Implementation choice: `num_heads=4` per stage (must evenly divide embed dim at every stage: 32,64,128,256 all divisible by 4). |
| Window size (T) | C | Only referenced symbolically in Eq. (5)-(6) and Fig. 2; no numeric value given | §2.2 | **Missing in paper**, and additionally constrained by our L=12 unified protocol (see §4 for the L=12 adaptation). Implementation choice documented in §4. |
| Shift size | B | Stated as a formula: shift = ⌊T/2⌋ | §2.2 | `shift_size = window_size // 2` |
| Number of blocks per stage (depth) | C | Never stated; Table 1 only fixes the *shape* per stage, not how many W-MSA/SW-MSA block pairs are stacked inside each stage | Table 1 | **Missing in paper.** Implementation choice: `depths=(2,2,2,2)` — one W-MSA block + one SW-MSA block per stage, the minimal configuration consistent with the paper's description that each stage contains "W-MSA, SW-MSA" (§2.1 lists both once per Block description, and Fig.1's block diagram shows one W-MSA/SW-MSA pair as "the" 1D Temporal Transformer Block per stage). |
| Dropout rate | C | Mentioned only qualitatively ("Dropout randomly inactivates the neurons... to prevent overfitting"), no numeric rate | §2.1 | **Missing in paper.** Implementation choice: `dropout=0.20`, matching the dropout already used across every other backbone in this project's unified protocol (`BEST_ARCH["dropout"]=0.20`), for a fair comparison. |
| Weight initialization | B | Not stated; paper explicitly frames HTT-Net as an improvement of the Swin Transformer backbone | §1.3(2), §2 | Swin Transformer's own published convention: `trunc_normal_(std=0.02)` for Linear weights and the relative-position-bias table, zero-init for biases and LayerNorm biases, ones-init for LayerNorm weights. |
| Random seed | C | Not stated | — | **Missing in paper.** Implementation choice: reuse this project's fixed seed (`RANDOM_SEED = 42`), and reuse the same seed count/protocol as every other baseline (see §5). |
| Class definition (original PHM2010 protocol) | A | Fixed pass-index thresholds, calibrated on C6: passes 1-50 = initial wear, 51-175 = normal wear, 176-315 = severe wear — identical absolute cut points applied to C1, C4 and C6 | §3.2 | **Not reused.** Our unified protocol uses condition-relative quantile thresholds per condition (see §3 conflict note below), which is a materially different label definition from the original paper. |
| Signals used (original protocol) | A | 3-axis force signal only (of the 7 available channels: 3 force + 3 vibration + 1 AE) | §3.2 | Not reused; our unified protocol's engineered feature set already spans all available signal channels (see `代码` feature engineering) |
| Train/test protocol closest to ours | A | "T1" dataset: train on C1+C4, test on C6 | Table 5 | This is the paper's own analogue of our D1 task (C1+C4 → C6); reused only as a naming/sanity reference, not as the literal protocol (see §3) |
| Evaluation metrics (original protocol) | A | Accuracy, per-class Precision/Recall/Specificity, AUC (via ROC) | §3.3.3, Tables 7-9 | Not reused; our unified protocol instead reuses this project's own metric set (Accuracy, Macro-F1, per-stage F1, M→E/M→L confusion rates, Rev/Jump/Smooth) for a fair head-to-head against the other B-series baselines (see §5) |

## 2. Architecture summary (as implemented)

```
X: [B, L=12, d]                      (d = number of selected online features)
  -> Linear Embedding: [B, L, C1]     C1 = 32 (Missing-in-paper choice)
  -> Stage 1: 2x{LN -> (W-MSA|SW-MSA) -> +res -> LN -> MLP(C1->C1/4->C1) -> +res}   at length L
  -> Token Merging: [B, L/2, 2*C1]
  -> Stage 2: 2x block                at length L/2, channels 2*C1
  -> Token Merging: [B, L/4, 4*C1]
  -> Stage 3: 2x block                at length L/4 (padded), channels 4*C1
  -> Token Merging: [B, L/8, 8*C1]
  -> Stage 4: 2x block                at length L/8 (padded), channels 8*C1
  -> LayerNorm -> masked mean-pool over valid (non-padded) positions -> Linear(8*C1, 3)
  -> logits: [B, 3]
```

## 3. Known conflict between the original paper protocol and our unified protocol

The original paper and this project's existing DC-PSR experiment protocol
(`代码/7.3主实验.py`, `代码/7.4对比实验.py`) disagree on multiple points that
are *not* free implementation choices — they are different scientific
choices made by two different papers:

| Aspect | HTT-Net paper (original) | This project's unified protocol (reused for HTT-Net-as-baseline) |
|---|---|---|
| Input | Raw 2000-sample force-signal window, 3 channels | L=12 sliding window over ~45 selected condition-relative *engineered* features |
| Stage labels | Fixed pass-index thresholds (1-50/51-175/176-315), calibrated once on C6 and applied identically to all conditions | Condition-relative quantile thresholds (`Q_EARLY=0.30`, `Q_LATE=0.72` on smoothed VB, computed *separately per condition*) |
| Class balancing | Upsample minority class, downsample majority classes to 10,000/class | No resampling; class-weighted loss instead |
| Train/val/test | No internal validation split; T1 = train {C1,C4} full, test {C6} full | Train {C1,C4} minus a 20%-per-stage internal validation carve-out, test {C6} held out entirely |

Per task instructions, this project's own current protocol takes priority
over the original paper's protocol whenever they conflict, and the
divergence is recorded here rather than silently resolved. Consequently,
this baseline is a **reimplementation / adaptation** of HTT-Net's
*architecture* only — it is not an exact reproduction of the paper's
original numerical results, and must never be reported as one.

## 4. L=12 special case

This project's unified protocol fixes the input sequence length at L=12,
far shorter than the paper's native L=2000. This forces window-size and
token-merging choices that the paper never had to make:

- **Token merging across 4 stages** needs L, L/2, L/4, L/8 = 12, 6, 3, 1.5.
  12→6→3 is exact; 3→1.5 is not an integer. **Implementation choice:**
  right-pad the sequence by repeating the last token before each merge step
  whose input length is odd, and track a per-sample valid-length mask through
  every stage so that (a) attention never attends *from* a padded position
  into the loss, and (b) the final classifier pooling step averages only over
  real (non-padded) positions. Concretely: stage lengths become
  12 → 6 → 3 → **4 (padded from 3)** → 2, giving 4 stages exactly matching
  Table 1's structure. This is recorded here, not silently patched.
- **Window size** must divide each stage's (possibly padded) sequence length
  for the implementation to avoid further padding inside W-MSA itself.
  Implementation choice: `window_size=3` at stage 1 (L=12, 4 windows),
  and the model auto-shrinks the window to the full (padded) sequence length
  whenever a later stage's length is `<= window_size` (i.e. stage 4 with
  padded length 4 attends globally in one window — equivalent to plain MSA
  at that depth, which is a reasonable degenerate case given how short the
  sequence has become by then). This auto-shrink is implemented and unit
  tested (`tests/test_htt_net.py::test_shifted_window_shapes`).
- Because L=12 is tiny, the "hierarchical, linear-complexity" motivation
  from the original paper (built for L=2000 raw signals) is largely moot at
  this scale — HTT-Net is included here purely as a **published architecture
  baseline** for fair comparison, not because window attention's efficiency
  benefit matters at L=12.

## 5. Random seeds and validation protocol

The existing unified experiment (`代码/7.4对比实验.py`) trains every deep
baseline (B8 TCN, B9 GRU, B10 TCN-GRU, B11 multi-task TCN-GRU) with a single
fixed seed (`RANDOM_SEED = 42`) and a single train/internal-val/test split —
there is no multi-seed mean±std protocol at the B-series level (that only
appears later, at the cross-condition Table 10 level, aggregating across the
4 *cross-condition tasks*, not across seeds). HTT-Net-as-baseline reuses this
exact single-seed protocol for the main D1 (C1+C4→C6) comparison, so it is
directly comparable to B1-B12 in `FINAL_comparison_results.csv`.

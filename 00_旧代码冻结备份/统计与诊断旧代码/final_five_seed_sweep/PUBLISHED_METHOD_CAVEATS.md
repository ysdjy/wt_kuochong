# PUBLISHED_METHOD_CAVEATS.md

Reproduction caveats for the 5 published methods in the final 9-method
main table, PHM2010 D1 (C1+C4 -> C6). Quoted/paraphrased directly from
each method's own FINAL_REPORT.md; see that file for full detail.

**HTT-Net (adapted)** -- adapted from the paper's original long-sequence
raw-force-window regime (L~=2000) to this project's unified run-level
protocol (45 selected condition-relative online features, L=12). Ranks
11th of 13 total methods (B1-B13) in the manuscript's own comparison,
ahead of only the two rule-based baselines. Dominant failure mode:
confuses Middle with Late (M-Recall 0.48, M-to-L 0.41).

**Multi-source Attention** -- the paper omits the CWT preprocessing
parameters entirely (wavelet family, scale count, frequency range), which
is the *entire* preprocessing backbone for this method; this
reimplementation uses a literature-standard choice (complex Morlet, 224
log-spaced scales). Protocol A sanity gap vs. the paper (-15.7pp) is
attributed primarily to this.

**MTF-AViTK** -- close original-protocol reproduction: Protocol A sanity
gap only -2.6pp vs. the paper, despite training a from-scratch
~309M-parameter ViT-L/32 (paper never states a pretraining source) on a
comparatively small dataset. Unified protocol adapted to this project's
authoritative condition-relative E/M/L labels and its unbalanced (not
class-balanced) dataset variant.

**Dynamic GIN + TGP** -- high-fidelity reproduction; original-protocol D1
accuracy (95.12%) closely matches the paper's own 95.71% (-0.59pp), well
inside this project's 92-98% sanity band. Notable cross-seed variance on
the unified protocol (std 6.83pp on Acc) -- this architecture/task
combination is genuinely seed-sensitive, not a reproduction defect. An
older pre-fix version had an evaluation batch-composition label-leakage
bug; all 5 seeds used in the final table are confirmed post-fix.

**DP2Net-adapted** -- pooled-source adaptation for D1
("DP2Net-adapted (pooled source)"); the original paper is a strict
single-source domain-generalization (SSDG) method. This adaptation was
added specifically to produce a number comparable to DC-PSR's D1 setup.
Original-protocol gaps vs. the paper (-6.5 to -8.2pp) are driven by three
unresolved physical/definitional gaps specific to PHM2010 (assumed tool
diameter/helix angle, a proxied stage-boundary rule, and Stage IV/failure
empirically never occurring for these tools) -- none of which affect the
D1 (unified protocol) result, which uses DC-PSR's own E/M/L labels
throughout and is this project's highest-confidence published-method
number (std 3.81pp, the tightest cross-seed spread of the 4 already-5-seed
methods).

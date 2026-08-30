# B8: DP2Net-adapted (pooled source)

**Full name**: Discontinuous Physical-Property-constrained single-source
Domain Generalization network. Adapted here as a pooled-source
("DP2Net-adapted") variant for this repo's unified comparison — see "Pinned
variant" below.

**Source paper**: Lai et al., *Mechanical Systems and Signal Processing*
(MSSP), 2024. DOI: 10.1016/j.ymssp.2024.111421. Reimplemented from
`baselines/dp2net/PAPER_SPEC.md` sec 3-5.

**Old code source**: `baselines/dp2net/` at commit
`811da096ee47bea4f65db193aa49e793dba6f47d` (branch
`diagnostic/fixed-preprocess-5seed`) — see `source_manifest.json`.

**New repo entry point**: `adapter.py::Dp2NetAdapter` (module-level
`ADAPTER_CLASS`), dispatched by `run_phm2010.py --method B8`.

**Pinned variant**: old code offered 3 protocols (A = paper-native
single-source sanity check; B-S = native single-source DC-PSR-label
comparison; B-D1 = pooled-source, C1+C4→C6-style, explicitly named
"DP2Net-adapted (pooled source)" in the old code and stated as "what enters
the DC-PSR D1 main comparison table"). **Only B-D1 is ported here** — it is
the official B8 for this repo. Protocol A / B-S are documented in
`source_manifest.json` as not-copied, not silently dropped.

## PHM2010 input form

Raw signal, Fx channel ONLY (not the full 6/7-channel signal — see paper
Sec 4.2). Butterworth low-pass filtered (cutoff 1733Hz @ native 50kHz), then
8 windows/run of length 4608 sampled evenly across the run.

## Preprocessing

`code/preprocessing.py::build_unified_windows_cache()` — caches
`data/PHM2010/cache/dp2net/windows_unified/` (regenerable, not committed).
Labels via `code/stage_labels.py` — identical formula to every other method
(task spec section 20).

## Model architecture

Two-stage Algorithm 1: **S** (discontinuous physical-property-guided spatial
attention, pooling+conv+BN+ReLU+sigmoid gate), **G** (physically-constrained
generator: pool→3×(conv+AdaIN)→transposed-conv, style noise via AdaIN,
output bounded to [-1,1] to match `Vst`, the physically-derived periodic
trend vector from Eq.5-6), **F** (WDCNN classifier, Zhang et al. 2018
canonical architecture). MMD loss (Eq.9, median-heuristic gamma) drives G's
adversarial-generation objective. Unmodified from the validated reproduction
(`code/model.py`, verbatim copy).

## Training hyperparameters (frozen, `config.yaml`)

**Stage 1** (pretrain S+F, classification): Adam lr=1e-3 for both S and F,
F's lr cosine-annealed (period=20 epochs), 100 epochs, batch=64.
**Stage 2** (train G+F, S frozen): Adam lr_g=1e-5, lr_f=1e-3, alpha=20.0
(`L_G = L_MSE - alpha·L_MMD`), 100 epochs, batch=64. Model selection uses
only the source-domain (train-cutter) internal validation split — the test
cutter is never touched until the single final evaluation.

**Inference**: trained S+F only (G is training-time-only, per the paper's own
Algorithm 1 "Inference: Model = Trained S and F").

## Task definition

Uses this repo's single authoritative `shared/phm2010/tasks.py` D1/D2/D3
registry via the `MethodAdapter` base class — this method defines no split
of its own; `stage_labels.py::get_task_split()` generalizes the pooled-source
split to any (train_cutters, test_cutter) pair.

## Output files

Per `RESULTS_POLICY.md`: `metrics.json`, `metrics.csv`, `predictions.csv`
(run-level, aggregated by mean over each run's 8 window-level probability
vectors), `training_log.csv` (both stages, `stage` column distinguishes
pretrain vs. generalize), `run_meta.json`, `config_resolved.yaml`,
`confusion_matrix.csv`, `DONE.flag`.

## Reproduction / adaptation notes

Architecture and training hyperparameters unchanged from the validated
reproduction. Only task/seed/output/path routing and variant pinning changed
(task spec section 35). Seed handling: old `train.py::run_protocol_b`
already accepted `seed` as a proper kwarg for variant B-D1 (verified against
the live current file, not assumed).

**Known limitation** (see `source_manifest.json`): this adapter's `train()`
does not implement the old code's intra-stage checkpoint/resume (Stage 1 /
Stage 2 mid-epoch resume) — a killed mid-training run restarts from epoch 0
of Stage 1 on retry. Only whole-cell resume (via `DONE.flag`) works. With
pretrain_epochs=gen_epochs=100, batch=64, this is a real cost for a future
full sweep; flagged for the parent rather than silently addressed, since
generic intra-cell resume belongs in `shared/runners/method_adapter.py`
(out of this fork's scope), not duplicated per-method.

## Status this round

Self-contained (no live imports from the old parent project). CPU-only smoke
test passed with a tiny debug run (`debug_max_runs`, 0 real training epochs)
— see this fork's report for the exact command/output. No real training was
run.

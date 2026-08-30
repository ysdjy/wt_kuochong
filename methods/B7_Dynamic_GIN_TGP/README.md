# B7: Dynamic GIN + TGP

**Full name**: Dynamic Graph Isomorphism Network + Temporal Graph Pooling.

**Source paper**: Cao et al., *Measurement*, 2026. DOI:
10.1016/j.measurement.2025.119007. Reimplemented from the paper's Table 1 /
Eq.(1)-(17) (old project: `baselines/dynamic_gin_tgp/PAPER_SPEC.md` documents
every "explicit vs. missing-in-paper -> documented choice" decision).

**Old code source**: `baselines/dynamic_gin_tgp/` at commit
`811da096ee47bea4f65db193aa49e793dba6f47d` (branch
`diagnostic/fixed-preprocess-5seed`) — see `source_manifest.json` for the
exact per-file diff.

**New repo entry point**: `adapter.py::DynamicGinTgpAdapter` (module-level
`ADAPTER_CLASS`), dispatched by `run_phm2010.py --method B7`.

## PHM2010 input form

Raw 6-channel signal (Fx,Fy,Fz,Vx,Vy,Vz — AE channel dropped per the paper's
own Sec 3.2), NOT run-level tabular features. Every run's stable-cut region
(first/last 40,000 points dropped) is split into 10 equal portions, each
contributing one centered 288-sample window — so one physical run = 10
model-input segments.

## Preprocessing

`code/preprocessing.py::build_windows_cache()` extracts and caches
`[10,6,288]` window arrays per run to
`data/PHM2010/cache/dynamic_gin_tgp/windows/` (regenerable, not committed).
`code/stage_labels.py` computes condition-relative Early/Middle/Late labels
(VB = max(flute_1,2,3), smoothed/normalized degradation index `q`, quantile
thresholds) — **identical formula to every other method** (task spec section
20) — and the task-parameterized train/val/test split (stage-stratified,
centered-slice validation carved from the train cutters only; test cutter
never touched until final evaluation).

## Model architecture

GASF (Gramian Angular Summation Field) spatial encoding fused with a temporal
CNN branch via cross-attention, feeding a 3-layer GIN+TGP graph-pooling
classifier. Unmodified from the validated reproduction (`code/model.py`,
verbatim copy) — see the file's own docstring for documented "missing in
paper -> implementation choice" decisions (softmax attention normalization,
TGP pooling axis bookkeeping, top-k=144 graph sparsification).

**Known paper-inherent property** (documented in `code/model.py`'s
`GraphEmbeddingMLP` docstring, not a bug introduced here): its parameters
receive no gradient from the classification loss, because Eq.(12)'s
hard-top-k adjacency construction is non-differentiable and the paper gives
no relaxation.

## Training hyperparameters (frozen, `config.yaml`)

Adam, lr=1e-4, weight_decay=0.1, batch_size=4, max_epochs=50, early stopping
patience=15 on run-level validation accuracy, ReduceLROnPlateau(factor=0.5,
patience=10). Model selection uses ONLY the run-level-aggregated validation
accuracy on the train cutters — the test cutter is never touched until the
single final evaluation.

**Batch-shuffle requirement (do not remove)**: the model's static graph
(Eq.7-9) concatenates every sample in a batch before computing cosine
similarity, so a batch's composition affects each sample's own prediction. A
data loader whose batches are homogeneous by run (all 10 portions of one run
together) would leak that run's true label into predictions via its
batch-mates. `adapter.py::DynamicGinTgpDataset` shuffles rows once at
construction (fixed `shuffle_seed=12345`, independent of `TRAIN_SEED`) to
break this — see the class docstring for the real training run this was
caught from (val accuracy hit 100% by epoch 1).

## Task definition

Uses this repo's single authoritative `shared/phm2010/tasks.py` D1/D2/D3
registry via the `MethodAdapter` base class's `train_cutters`/`test_cutter`
constructor args — this method defines no split of its own.

## Output files

Per `RESULTS_POLICY.md`: `metrics.json`, `metrics.csv`, `predictions.csv`
(`run_id, true_stage, pred_stage, p_early, p_middle, p_late` — run-level,
aggregated by mean over each run's 10 portion-level probability vectors),
`training_log.csv`, `run_meta.json`, `config_resolved.yaml`,
`confusion_matrix.csv`, `DONE.flag`.

## Reproduction / adaptation notes

Only task/seed/output/path routing changed vs. the original validated
reproduction — architecture, optimizer, learning rate, and loss are
unchanged (task spec section 35). Seed handling: the old `train.py` already
took `seed` as a proper function kwarg (verified against the live file),
so no monkeypatch was needed here (contrast B5/B6, which needed one).

**Time-planning note** (from prior project memory, not from this round's
testing): one historical full training run of this method was observed at
~185s/epoch, ~1hr total — noticeably slower than the other GPU baselines on
this project's hardware. Budget accordingly for any future real 9×3×101
sweep; this round only ran a tiny CPU smoke test (a handful of runs, 0
epochs), which is not representative of full-run wall-clock time.

## Status this round

Self-contained (no live imports from the old parent project). CPU-only smoke
test passed — see this fork's report for the exact command/output. No real
training was run.

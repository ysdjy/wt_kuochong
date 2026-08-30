# B5: Multi-source Channel-Spatial Attention (Multi-Attention-CNN)

- **Full method name**: Multi-source Channel-Spatial Attention CNN
- **Source paper**: Wei et al., *Robotics and Computer-Integrated Manufacturing* (RCIM) 2024, DOI: 10.1016/j.rcim.2024.102741
- **Old code source**: `baselines/multi_source_attention/` at legacy_git_commit
  `811da096ee47bea4f65db193aa49e793dba6f47d` (branch `diagnostic/fixed-preprocess-5seed`)
  — see `source_manifest.json` for exact per-file provenance/hashes.
- **New repo entry point**: `adapter.py::MultiSourceAttentionAdapter` (via
  `run_phm2010.py --method B5 ...`)

## PHM2010 input form

- **Raw signal**, not run-level features. Two independent CNN branches
  (force, vibration), each fed a 224x224x3 CWT scalogram image built from 3
  of the 7 raw channels (force: Fx,Fy,Fz; vibration: Vx,Vy,Vz).
- Not directly windowed — one image pair per physical cutting-pass run.
- **Evaluation universe**: raw-signal method, native coverage run_id 1-315;
  reduced to the common universe run_id 12-315 (n=304) before formal
  metrics, per `shared/phm2010/evaluation_universe.py`.

## Preprocessing (`code/signal_preprocessing.py`, `code/dcpsr_labels.py`)

1. Load raw 7-channel signal for a run (`PHM2010_RAW_ROOT/c{n}/c{n}/c_{n}_{run:03d}.csv`).
2. Central 50% ("middle region") of the pass.
3. Resample each of the 3 force / 3 vibration axes to 224 samples.
4. Complex-Morlet CWT (`cmor1.5-1.0`, 224 log-spaced scales) per axis -> magnitude scalogram.
5. Per-image min-max normalize to [0,1], stack 3 axes -> RGB uint8 image.
6. Stage labels: condition-relative Early/Middle/Late from VB=max(flute_1,2,3)
   (identical thresholds/logic to every other method in this repo — see
   `code/dcpsr_labels.py`). Train/val split: stage-stratified centered slice
   carved from train cutters ONLY (`VAL_RATIO_STAGE=0.20`,
   `MIN_STAGE_VAL_LEN=8`); test cutter held out entirely until final eval.

Images are built and cached on demand to `_cache/images/` (gitignored,
regenerable) the first time each (condition, run_id) pair is needed — no
275MB pre-built cache is shipped or required.

## Model architecture (`code/model.py`, unchanged from original)

Two parallel `Conv2d(3->16,k3)+ReLU` branches (force, vibration) ->
channel-concat [B,32,224,224] -> Channel-Spatial Attention (SE-style channel
gate + spatial gate, both computed from the same pre-attention input, then
combined by element-wise product, Eq. 18 in the paper) -> 3x
`{Conv2d+ReLU, MaxPool(k3,s2)}` (32->64->128 channels, 224->28 spatial) ->
Flatten -> FC(128) -> Dropout(0.5) -> FC(3) logits.

## Training hyperparameters (`config.yaml`, unchanged from original)

`max_epochs=100, patience=15 (early stop on C1+C4-only val acc), batch_size=64,
lr=0.001, Adam, weight_decay=1e-4, CrossEntropyLoss`. `seed` is NOT a fixed
config value — it comes from the CLI `--seed-start/--seed-end` range
(`self.seed`), replacing the original's hardcoded `PROTO_B_CFG['seed']=42`.

## Task definition

D1/D2/D3 per `shared/phm2010/tasks.py` — train/val split constructed only
from the task's train cutters; test cutter never touched until the single
final evaluation.

## Output files

Standard schema per `RESULTS_POLICY.md`: `metrics.json`, `metrics.csv`,
`predictions.csv` (`run_id,true_stage,pred_stage,p_early,p_middle,p_late` —
no `q_true`/`q_pred`, this architecture has no regression head, left absent
rather than faked), `training_log.csv` (per-epoch train/val loss+acc),
`run_meta.json`, `config_resolved.yaml`, `confusion_matrix.csv`, `DONE.flag`.

## Reproduction / adaptation notes

- Only task/seed/output routing and cross-platform paths were changed vs.
  the original (task spec section 35). Architecture, CWT math, and
  hyperparameters are byte-for-byte unchanged.
- The old `data/label_utils.py` live-imported `代码/main_experiment_3_fgds_psi_optimized.py`
  from outside this repo — fixed here by vendoring the needed functions into
  `code/dcpsr_labels.py` (self-contained, no dependency on the old parent
  project at runtime).
- `split_grouped_lifecycle` was hardcoded to C1+C4->C6; generalized to
  accept `train_conditions`/`test_condition`, matching the logic already
  validated for D2/D3 in the old project's `condition_split.py`.
- Metric computation uses `shared/metrics/metrics.py` (verified bit-exact
  against the old project's authoritative formula on real D1 predictions).
- Protocol A (original-paper 70/30 sanity-check split) was NOT ported — out
  of scope, this repo only needs the Unified Protocol B (D1/D2/D3)
  comparison.
- **Not yet run for real training this round** — task spec section 45
  explicitly excludes launching the full 9x3x101 sweep this round. Only a
  CPU smoke test (`tests/test_smoke.py`) has been run — see its output for
  confirmation the full prepare -> build -> forward -> write-outputs
  pipeline executes end to end.

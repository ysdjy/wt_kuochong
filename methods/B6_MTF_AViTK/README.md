# B6 — MTF-AViTK

**Full method name**: Adapt-ViT-L/32 + AdaptMLP + KAN classifier ("MTF-AViTK"),
input = Markov Transition Field (MTF) image of the resultant cutting-force
signal.

**Source paper**: Dong et al., *Mechanical Systems and Signal Processing*
(MSSP), 2025. DOI: 10.1016/j.ymssp.2025.112473. See the old project's
`baselines/mtf_avitk/PAPER_SPEC.md` for the full Explicit/Inferable/Missing
determination this reimplementation was built from (every architectural
choice not explicitly stated in the paper — adapter bottleneck dim, KAN
hidden width, RGB colormap, resize method, MTF quantile-bin count, wavelet
threshold formula — is documented there, not re-derived here).

**Old code source**: `baselines/mtf_avitk/{model.py,kan.py,preprocessing.py,train.py}`
+ `baselines/mtf_avitk/data/label_utils.py` + `代码/main_experiment_3_fgds_psi_optimized.py`
(labeling constants only) + `final_statistical_evidence/scripts/methods/condition_split.py`
(D2/D3 split generalization pattern), all at commit `811da096ee47bea4f65db193aa49e793dba6f47d`
(branch `diagnostic/fixed-preprocess-5seed`). Full file-by-file mapping and
sha256 hashes: `source_manifest.json`.

**New repo entry point**: `adapter.py::MTFAViTKAdapter` (module-level
`ADAPTER_CLASS`), dispatched by `run_phm2010.py --method B6`.

## PHM2010 input form

**Image-like, from raw signal** — NOT run-level tabular features. Pipeline
(see `code/preprocessing.py` for the exact, byte-for-byte-ported formulas):

```
raw 7-channel signal (Fx,Fy,Fz,Vx,Vy,Vz,AE), one CSV per cutting pass
  -> take Fx,Fy,Fz, stable region [90000:100000]
  -> 5 non-overlapping 2000-sample sub-windows
  -> resultant force F = sqrt(Fx^2+Fy^2+Fz^2)
  -> Sym7 wavelet denoise (level 6, soft threshold, VisuShrink/MAD)
  -> resample 2000 -> 500 samples
  -> Markov Transition Field encoding (8 quantile bins) -> [500,500] field
  -> jet colormap -> [500,500,3] RGB -> bilinear resize -> [384,384,3] uint8
```

Each physical run therefore yields **5 images** (one per sub-window); the
model is trained/evaluated at sub-window granularity, then the 5
per-run softmax outputs are **mean-aggregated** back to one run-level
prediction before scoring (a documented reimplementation choice — the
original paper evaluates at image level, not run level; every other method
in this comparison reports one prediction per physical run, so this
aggregation makes B6 directly comparable).

Raw signal source: `PHM2010_ROOT` env var (default `data/PHM2010/raw/`,
populated by `scripts/download_phm2010.py`) — **not** used here directly:
the old project's precomputed 2.0GB `data/images/*.npy` cache is not
committed anywhere in this repo; `code/data_prep.py` regenerates images
on-demand from raw signal and caches them locally under
`data/PHM2010/derived/mtf_avitk_images/`.

## Model architecture (unchanged from the original — do not edit without a CHANGELOG entry)

- **Backbone**: Adapt-ViT-L/32 — 384×384×3 image → 32×32 patch conv embed
  (144 patches) → +CLS token +learned position embedding → 24 pre-norm
  transformer blocks (16-head MHSA + AdaptMLP: a standard 4096-hidden MLP
  branch plus a parallel bottleneck-64 adapter branch, scale 0.1, adapter
  `up` weights zero-initialized) → final LayerNorm → CLS token, 1024-dim.
- **Classifier**: 2-layer Kolmogorov-Arnold Network (`code/kan.py`, a
  from-scratch B-spline KAN, no third-party `kan`/`pykan` package —
  `kan.py` is vendored verbatim), grid_size=5, spline_order=3, SiLU base
  activation, hidden width 64 → 3 classes.
- **~309M parameters**, no pretrained-weight loading anywhere (trained from
  scratch — the paper never states a pretraining source).

## Training hyperparameters (unchanged — `config.yaml`)

SGD, lr=0.0006, weight_decay=2e-4, momentum=0.9, batch_size=8, max_epochs=50,
early-stopping patience=10 on run-level validation accuracy (validation
carved from the train cutters only, per this project's unified split —
`code/label_utils.py::split_grouped_lifecycle_generic`). AMP (mixed
precision) on by default when CUDA is available; `grad_checkpoint` /
`batch_size` are exposed OOM-avoidance knobs (pure memory/compute
trade-offs, not architecture changes — see `config.yaml`).

## Task definition

Uses the repo-wide authoritative `shared/phm2010/tasks.py` D1/D2/D3
definitions via `self.train_cutters` / `self.test_cutter` (no independently
hardcoded split). Validation is carved from the train cutters only —
the test cutter is never touched before the single final evaluation.

## Output files

Per `RESULTS_POLICY.md`: `metrics.json`, `metrics.csv`, `predictions.csv`
(`run_id,true_stage,pred_stage,p_early,p_middle,p_late` — no `q_true`/`q_pred`,
this method has no continuous degradation-index head), `training_log.csv`
(one row per epoch: `train_loss,train_acc,val_run_level_acc,best_val_acc,
best_epoch,time_s`), `run_meta.json`, `config_resolved.yaml`,
`confusion_matrix.csv`, `DONE.flag`.

## Compute cost — READ BEFORE RUNNING FOR REAL

- **~309M parameters, `checkpoint_best.pth` ~1.2GB** (never committed —
  `--save-checkpoint none` is the default; `.gitignore` excludes `*.pth`
  repo-wide regardless).
- **~30-40 minutes per (task, seed) on an 8GB laptop GPU** historically,
  sometimes longer — by far the slowest of the 9 methods.
- Per this project's established training-execution policy, **this method is
  never launched unattended** — a human runs it directly (or the automated
  runner is explicitly authorized for it, one task/seed range at a time,
  never `--workers > 1` for this method on an 8GB card alongside anything
  else GPU-bound).

## This round's scope — what was actually done

This build task **vendored the code and ran a CPU-only plumbing smoke test
only — no real training was executed**. See `tests/smoke_test.py` for what
the smoke test actually checks (single real image, one forward pass) and its
recorded output below.

### Smoke test result (2026-08-30, CPU-only, `CUDA_VISIBLE_DEVICES=""`)

`python tests/smoke_test.py` (conda env `pub_baselines`, has PyWavelets;
`dcpsr` env is missing it — use `pub_baselines` or add PyWavelets, see
"Blockers found" below). All 4 levels passed:

```
[1/4] import all vendored modules ......................... OK (6.8s)
[2/4] instantiate MTF_AViTK on CPU ......................... OK (2.8s), n_params=309,371,072 (309.4M)
[3/4] single REAL image forward pass (C1 run 1, real raw
      signal -> real wavelet-denoise -> real MTF encode ->
      real 384x384x3 image -> real model forward) ........... OK, image built in 0.7s, forward in 0.95s,
                                                                 output shape (1,3) as expected
[4/4] full adapter path: MTFAViTKAdapter(...).run(
      task=D1, seed=0, device=cpu, smoke_test=True) ........ OK (4.0s), status=done, all expected
                                                                 output files written under tmp/smoke_tests/
                                                                 (predictions.csv had 1 real row:
                                                                 run_id=1,true_stage=early,pred_stage=early,
                                                                 p=[0.363,0.285,0.353])
```

**No real/full training loop was run at any point** — `train()` is only
called by `MethodAdapter.run()` when `smoke_test=False`, and this smoke test
never passes that. Total wall time for all 4 levels: well under 2 minutes.

### Blockers found (not fixed, out of this task's scope)

1. **conda env**: the `dcpsr` env (Python 3.11, torch 2.7.1+cu118) does NOT
   have `PyWavelets` installed, even though `environment/requirements.txt`
   already lists it — only the `pub_baselines` env has it on this machine.
   Use `pub_baselines`, or `pip install PyWavelets` into whichever env
   `bootstrap_windows.ps1`/`bootstrap_ubuntu.sh` ultimately creates.
2. **Shared-code bug, not B6-specific**: `shared/utils/run_meta.py::_torch_cuda_gpu_info()`
   calls `torch.cuda.get_device_name(0)` whenever `torch.cuda.is_available()`
   is True, but on this dev machine (and presumably any machine run with
   `CUDA_VISIBLE_DEVICES=""`) `is_available()` returns `True` even when
   `torch.cuda.device_count() == 0`, so `get_device_name(0)` raises
   `AssertionError: Invalid device id`. This will hit **every** method's
   `run()`, not just B6, whenever CPU-only execution is forced via an empty
   `CUDA_VISIBLE_DEVICES`. Fix belongs in `shared/utils/run_meta.py`
   (`if torch.cuda.is_available() and torch.cuda.device_count() > 0:`) —
   left unfixed here since this fork's scope is `methods/B6_MTF_AViTK/`
   only; `tests/smoke_test.py` works around it locally (monkeypatches
   `torch.cuda.get_device_name` inside the test process only) so this
   method's own plumbing could still be verified end-to-end.

## Reproduction / adaptation notes

All numeric formulas (wavelet params, MTF bin count, condition-relative
stage thresholds Q_EARLY=0.30/Q_LATE=0.72/RATE_LATE_Q=0.78, SGD
hyperparameters, KAN spline config) are copied verbatim from the old,
already-validated implementation — see `source_manifest.json` for the
file-by-file diff summary. The only real changes are: (1) raw-data and
image-cache path routing (env-var based, cross-platform, no hardcoded old
paths), (2) `split_grouped_lifecycle` generalized from hardcoded
train={C1,C4}/test={C6} to accept any `(train_cutters, test_cutter)` pair
(needed for D2/D3, not just D1), (3) seed handling moved from an internal
`set_seed(42)` call to the shared adapter-level `seed_everything(self.seed)`
contract, (4) image storage moved from a 2.0GB precomputed cache to
on-demand generation + local caching. Per task spec section 34's B3/B9
"same backbone" rule: B6 is fully independent of B3/B9, no checkpoint
sharing applies here.

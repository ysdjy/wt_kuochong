# B4: HTT-Net (adapted)

**Full method name**: Hierarchical Temporal Transformer Network (HTT-Net)

**Source paper**: Xue, Z., Chen, N., Wu, Y., Yang, Y., Li, L. (2023).
"Hierarchical temporal transformer network for tool wear state recognition."
*Advanced Engineering Informatics*, 58, 102218.

This is a **reimplementation/adaptation**, not an exact reproduction of the
original paper — several architecture hyperparameters (embed dim, heads,
window size, depths, dropout) are not given numeric values in the paper and
were chosen to match this project's shared unified protocol for a fair
cross-method comparison (see `code/model.py`'s module docstring and the old
project's `baselines/htt_net/PAPER_SPEC.md` for the full paper-to-code mapping
of every such choice).

## Old code source (read-only, commit `811da096ee47bea4f65db193aa49e793dba6f47d`, branch `diagnostic/fixed-preprocess-5seed`)

- `baselines/htt_net/model.py` — HTTNet architecture (vendored verbatim into `code/model.py`, sha256-verified identical)
- `baselines/htt_net/train.py` — original D1-only training script (reimplemented as `adapter.py`)
- `代码/main_experiment_3_fgds_psi_optimized.py` — shared window-based preprocessing pipeline this baseline reuses (trimmed + vendored into `code/pipeline.py`)

See `source_manifest.json` for exact per-file provenance and every modification made.

## New repo entry point

```python
from methods.B4_HTT_Net.adapter import HTTNetAdapter  # or import ADAPTER_CLASS from the module
```

Driven by `run_phm2010.py --method B4 --tasks all --seed-start 0 --seed-end 100 --resume`.

## PHM2010 input form

- **Run-level feature table** (not raw signal, not image-like, not graph):
  `data/PHM2010/features/run_level_features_all.csv` (`PHM2010_FEATURES` env
  var override).
- Sliding windows of `L=12` consecutive runs' condition-relative online
  features, per condition (cutter) — same windowing convention as B1/B2/B3/B9.
- Input tensor shape: `[batch, L=12, d≈45 selected features]`.

## Preprocessing

Identical sequence to B1/B2/B3/B9 (see `code/pipeline.py`):
1. Load feature table, normalize condition names, coerce run_id/VB.
2. Condition-relative stage labeling (`Q_EARLY=0.30`, `Q_LATE=0.72` quantile
   thresholds on smoothed, per-condition-normalized VB).
3. Train/val/test split: `split_grouped_lifecycle(df, train_cutters, test_cutter)`
   — a per-stage, centered-slice validation carve out of the TRAIN cutters only
   (test cutter never contributes to validation). Generalized from the
   original's hardcoded D1-only split to accept any (train, test) pair — see
   `source_manifest.json` adaptation #1.
4. Split-safe "online" feature engineering (expanding-window z-score, slope,
   online rank) computed independently per split (no test-time leakage into
   train statistics).
5. Train-only feature selection (mutual information + Spearman + cross-condition
   stability score, redundancy-pruned to 45 features), fit only on
   `final_train`, applied to val/test.
6. Train-only 5-component GMM over `(q_true, rate_norm)` for a fine-grained
   degradation state (not used by HTT-Net's loss itself, computed only because
   the shared pipeline produces it — HTT-Net's `predict()` never reads it).
7. Train-only `StandardScaler`.
8. `L=12` windowing per condition (`build_windows` in `code/pipeline.py`).

`PREPROCESS_SEED` controls steps 5-6 (mutual-info `random_state`, GMM
`random_state`); it is fixed per task and never varies with `TRAIN_SEED`.

## Model architecture

Swin-Transformer-style 1D hierarchical window-attention network, 4 stages
(`embed_dim=32`, `depths=(2,2,2,2)`, `num_heads=4`, `window_size=3`,
`dropout=0.20`), token merging between stages (`12→6→3→2` after
right-padding), relative position bias, shifted-window attention. See
`code/model.py` for the full implementation and its extensive inline notes on
the L=12-specific padding/masking design and one documented paper-ambiguity
resolution (post-MLP LayerNorm placement — resolved to standard pre-norm via a
unit-test-verified overfit check, see the comment above
`TemporalTransformerBlock.__init__`). Verbatim copy — architecture unchanged
from the original baseline.

## Training hyperparameters

| | |
|---|---|
| Optimizer | AdamW |
| LR | 5e-4 (unified-protocol value; paper's own T1 protocol uses 1e-4 — not used here, see `PAPER_SPEC.md` in the old baseline) |
| Weight decay | 1e-5 |
| Batch size | 32 |
| Epochs | 120 (budget; early-stopped) |
| Patience | 18 |
| Grad clip | 1.0 (L2 norm) |
| Early-stop score | `0.7*(1-val_macro_f1) + 1.0*(1-val_middle_recall) + 0.15*val_loss`, minimized |

Unchanged from the original baseline — see `config.yaml`.

## Task definition

Uses `shared/phm2010/tasks.py` (D1/D2/D3) — no independent split logic. Test
cutter's data never touches feature selection/GMM/scaler/validation
(enforced structurally: `split_grouped_lifecycle` only carves validation out
of `train_conditions`).

## Output files

Per `RESULTS_POLICY.md`: `metrics.json`, `metrics.csv`, `predictions.csv`
(`run_id, true_stage, pred_stage, p_early, p_middle, p_late` — no `q_true`/`q_pred`,
HTT-Net has no continuous degradation-index head), `training_log.csv`,
`run_meta.json`, `config_resolved.yaml`, `confusion_matrix.csv`, `DONE.flag`.

## Reproduction / adaptation notes

- Common evaluation universe: window-based (`run_id` 12-315, n=304) — see
  `shared/phm2010/evaluation_universe.py`. No manual restriction needed in
  `predict()`, this method's native coverage already equals the common universe.
- No pretrained weights, no external assets — trained from scratch every run.
- CPU-safe: `tests/test_smoke.py` runs `prepare()` on the REAL D1 feature file,
  builds the model, and does a real forward pass (no training loop) —
  verified passing 2026-08-30, CPU, 304/304 predictions, 45 selected features.
- Not yet run with real multi-epoch training in this new repo (per master task
  spec section 45, this round is packaging + smoke test only, not the full
  9x3x101 sweep).

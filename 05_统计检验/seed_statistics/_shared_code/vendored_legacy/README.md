# vendored_legacy/

Read-only, verbatim copies of the old parent project's source files that the
`run_b1_*_seed_task.py` / `run_b2_*_seed_task.py` scripts in the parent
`_shared_code/` directory need at import time, vendored into this repo so
those scripts work from a plain `git clone` of `wt_kuochong` alone — no
dependency on the outside `论文/` project folder.

| File | Copied from (legacy source, outside this repo) |
|---|---|
| `main_experiment_3_fgds_psi_optimized.py` | `代码/main_experiment_3_fgds_psi_optimized.py` |
| `7.4对比实验.py` | `代码/7.4对比实验.py` |
| `9.1nasa数据实验.py` | `代码/9.1nasa数据实验.py` |
| `dcpsr/` (full package) | `experiments_mendeley/code/dcpsr/` |

Legacy git commit at vendoring time (2026-08-30): `811da096ee47bea4f65db193aa49e793dba6f47d`
(branch `diagnostic/fixed-preprocess-5seed` of the parent `论文` project).

**Not modified** except that the six `run_b*_*_seed_task.py` scripts point
their own `CODE_DIR`/`DCPSR_CODE_DIR` constants at this directory instead of
the outside project (see each script's header comment) — the vendored files
themselves are byte-identical to their legacy source at copy time.

**Note on raw data**: `9.1nasa数据实验.py`'s `MAT_FILE` constant is patched at
import time by the NASA runner scripts, but is never actually read by them —
they only call `build_sliding_windows`/`TCNGRUStageModel`/
`TCNGRUMultiTaskModel`/`train_model`/`predict_model` against the already-
frozen `shared/reproducibility/NASA_N*_frozen_preprocess/` CSVs, never the
raw-signal loader. Same for `dcpsr`'s raw `.h5` loaders and MTW-CM's frozen
preprocessing. No raw dataset file is required to run any of the six scripts
in this directory's parent — only the frozen-preprocess artifacts already
committed under `shared/reproducibility/`.

**Existing B9/B3 scripts** (`run_seed_task.py`, `run_nasa_seed_task.py`,
`run_mtw_seed_task.py`, in the parent `_shared_code/`, built by a different
Claude Code session) still import from the outside `论文/` project directly
and are NOT changed by this vendoring — they were left as-is, per this
repo's convention of not modifying another session's files. Only the new
B1/B2 scripts use this vendored copy.

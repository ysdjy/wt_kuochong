# MANUAL_RUN.md

The complete guide to go from a fresh `git clone` on a new Windows or Ubuntu machine
to running any of the 9 PHM2010 comparison methods, without touching the old parent
project or any machine-specific path. If you only read one document in this repo,
read this one.

## 0. Platform verification status

```
Windows: TESTED       (dev machine, RTX 3070 Ti Laptop 8GB, this round)
Ubuntu:  SCRIPT PREPARED, PENDING PHYSICAL-MACHINE VERIFICATION
```

`scripts/bootstrap_ubuntu.sh` was written carefully (mirrors every step of the
tested Windows script) but has not been run on a physical/VM Ubuntu machine this
round. If you're the first to run it, please report back anything that needed
fixing.

## 1. Clone

```bash
git clone https://github.com/ysdjy/wt_kuochong.git
cd wt_kuochong/扩充实验代码
```

(The repo's outer directory name may differ depending on how you cloned it — this
guide assumes you're inside `扩充实验代码/`, the repo root referenced everywhere
below as `<repo>`.)

## 2. One-command environment install

### Windows

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows.ps1
```

### Ubuntu

```bash
bash scripts/bootstrap_ubuntu.sh
```

Both scripts: check conda is installed → create (or reuse) a single conda env
named `wt_kuochong` (Python 3.11.15) → install `environment/requirements.txt` →
detect an NVIDIA GPU via `nvidia-smi` and install the matching torch build
(CUDA 11.8 wheel if a GPU is found, CPU wheel otherwise) → verify CUDA →
`scripts/verify_environment.py` → `scripts/self_check.py`. They never touch the
`base` conda environment.

If bootstrap fails partway, each step is safe to re-run individually — see the
step list inside the script.

### Environment self-check (also run automatically by bootstrap)

```bash
conda run -n wt_kuochong python scripts/self_check.py
```

Checks: Python version, core dependency imports, all 9 methods' `adapter.py`
import cleanly through the same registry `run_phm2010.py` uses (catches the
cross-method import-collision class of bug — see `shared/runners/registry.py`),
`shared/metrics` + `shared/phm2010` unit tests pass, the committed feature CSV's
sha256 matches, and GPU info (informational, never fails on a CPU-only machine).

## 3. PHM2010 data

### Already committed, no action needed

`data/PHM2010/features/run_level_features_all.csv` (5.4MB) — the authoritative
run-level feature table used by B1 (RF), B2 (TCN-GRU), B3 (Multi-task TCN-GRU),
B4 (HTT-Net), and B9 (DC-PHSR). See `data/README.md` for provenance/sha256.
**These 5 methods need nothing further from this section.**

### Needed only for B5, B6, B7, B8 (raw-signal methods)

```bash
python scripts/download_phm2010.py
python scripts/verify_phm2010.py
```

**Source**: Kaggle dataset `tobbyrui/phm2010` — the only concretely-named public
source of this data found anywhere in the old parent project's code/docs (see
`data/README.md`).

**One-time Kaggle credential setup** (before running `download_phm2010.py`):

1. Create a free Kaggle account if you don't have one.
2. Go to your Kaggle account settings → "Create New API Token". This downloads
   a `kaggle.json` file.
3. Place it at:
   - Windows: `%USERPROFILE%\.kaggle\kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json`
4. `pip install kaggle` (not in `environment/requirements.txt` by default since
   only B5-B8 need it — install it manually, or add it yourself).
5. On kaggle.com, open the `tobbyrui/phm2010` dataset page once and accept its
   terms if prompted (required before the API can download it).
6. **Never commit `kaggle.json`** — already excluded via `.gitignore`.

`download_phm2010.py` unzips into `data/PHM2010/raw/`. `verify_phm2010.py` checks
all 3 conditions (C1, C4, C6 — the ones PHM2010 ships full wear labels for) have
315 signal files + a complete wear.csv each. If the Kaggle archive's internal
folder layout differs from what's expected (not independently re-verified this
round — see `scripts/verify_phm2010.py`'s docstring), reorganize to match before
re-running verify.

## 4. Formal run commands — all 9 methods

```bash
conda run -n wt_kuochong python run_phm2010.py --method B1 --tasks all --seed-start 0 --seed-end 100 --resume
conda run -n wt_kuochong python run_phm2010.py --method B2 --tasks all --seed-start 0 --seed-end 100 --resume
conda run -n wt_kuochong python run_phm2010.py --method B3 --tasks all --seed-start 0 --seed-end 100 --resume
conda run -n wt_kuochong python run_phm2010.py --method B4 --tasks all --seed-start 0 --seed-end 100 --resume
conda run -n wt_kuochong python run_phm2010.py --method B5 --tasks all --seed-start 0 --seed-end 100 --resume
conda run -n wt_kuochong python run_phm2010.py --method B6 --tasks all --seed-start 0 --seed-end 100 --resume
conda run -n wt_kuochong python run_phm2010.py --method B7 --tasks all --seed-start 0 --seed-end 100 --resume
conda run -n wt_kuochong python run_phm2010.py --method B8 --tasks all --seed-start 0 --seed-end 100 --resume
conda run -n wt_kuochong python run_phm2010.py --method B9 --tasks all --seed-start 0 --seed-end 100 --resume
```

`--tasks all` = D1+D2+D3; `--seed-start`/`--seed-end` are inclusive (0..100 = 101
seeds). `--resume` skips any `(task, seed)` cell that already has `DONE.flag`.

### Per-method notes (recommended device, extra deps, tested/untested, known risk)

| ID | Method | Recommended device | Extra deps | Tested this round | Known VRAM risk |
|---|---|---|---|---|---|
| B1 | RF | CPU (fast, seconds/seed) | none | **Yes** — smoke test + full D1/D2/D3 real run at seed 0, matches frozen numbers | none (CPU-only) |
| B2 | TCN-GRU | GPU (few-min/seed) | none | Yes — smoke test only | low |
| B3 | Multi-task TCN-GRU | GPU (few-min/seed) | none | Yes — smoke test only | low |
| B4 | HTT-Net | GPU (few-min/seed) | none | Yes — smoke test only (D1 AND D2/D3 routing both verified) | low |
| B5 | Multi-source Attention | GPU (few-min/seed) | PyWavelets (in requirements.txt) | Yes — smoke test only | low-medium (~50MB checkpoint) |
| B6 | MTF-AViTK | GPU, **but see warning below** | PyWavelets; vendored `kan.py` (not a pip package) | Yes — smoke test only, 4 levels (import/instantiate/1 real forward/1 full adapter cycle) | **HIGH** — 309M params, ~1.2GB checkpoint, historically ~30-40min/seed on an 8GB card, sometimes hours |
| B7 | Dynamic GIN + TGP | GPU (historically slower than expected, ~185s/epoch observed once) | none (hand-rolled GIN, no torch_geometric) | Yes — smoke test only | low |
| B8 | DP2Net-adapted | GPU (fast, minutes) | none | Yes — smoke test only. Pinned variant: B-D1 (pooled-source) | low |
| B9 | DC-PHSR | GPU (few-min/seed) | none | Yes — smoke test only. Trains its own B3-identical backbone copy per cell this round (no cross-adapter checkpoint reuse yet — see `methods/B9_DC_PHSR/README.md`) | low |

**Do not run times were fabricated** — every "few-min/seed" above is this round's
smoke-test-scale observation or the old project's own historical notes
(`ENVIRONMENT.md`), never invented.

**⚠️ B6 (MTF-AViTK) warning**: this is the one method this project's established
policy says should NOT be launched unattended/by an automated agent — write a
manual run, run it yourself, watch the GPU. On an 8GB card, run it ALONE
(`--workers 1`, nothing else using the GPU) and consider `--save-checkpoint none`
unless you specifically need the 1.2GB checkpoint (never committed to git either
way).

## 5. Multi-machine sharding

Split a method's 101 seeds across N machines by giving each one a disjoint
`--seed-start`/`--seed-end` range — output directories never collide since they're
keyed by `(method, task, seed)`:

```bash
# Machine A
python run_phm2010.py --method B4 --tasks all --seed-start 0  --seed-end 24 --resume
# Machine B
python run_phm2010.py --method B4 --tasks all --seed-start 25 --seed-end 49 --resume
# Machine C
python run_phm2010.py --method B4 --tasks all --seed-start 50 --seed-end 74 --resume
# Machine D
python run_phm2010.py --method B4 --tasks all --seed-start 75 --seed-end 100 --resume
```

### Getting results back to one place

Recommended: each machine copies its own `results/PHM2010/` tree back to a
collation machine (e.g. via `rsync`/`scp`/a shared drive), then on the collation
machine:

```bash
python scripts/merge_results.py --sources /path/to/machineA/results /path/to/machineB/results /path/to/machineC/results /path/to/machineD/results --dest results
```

`merge_results.py` only ADDS cells that don't already exist in `--dest`; if the
SAME `(method, task, seed)` exists in two sources with a different
`config_hash` (from `run_meta.json`), it's reported as a **CONFLICT** and neither
copy is merged automatically — inspect both `run_meta.json` files and resolve by
hand. Nothing is ever silently overwritten. `--dry-run` previews without copying.

(A git-branch-per-machine strategy also works if you prefer, but since per-seed
result directories are gitignored by default — see `RESULTS_POLICY.md` — the
file-copy + `merge_results.py` route is simpler for this project.)

### Aggregation (mean±std tables)

```bash
python scripts/aggregate_results.py --method B9      # one method
python scripts/aggregate_results.py --all            # every discoverable method
```

Produces `results/PHM2010/Bx_xxx/summary/*.csv` — see `RESULTS_POLICY.md` for the
exact file list and the dataset-level mean±std formula (equal-weight average of
D1/D2/D3 per seed, then `numpy.std(ddof=1)` across seeds that completed all 3
tasks).

## 6. Smoke test (do this before any large run)

```bash
python run_phm2010.py --method B4 --tasks D1 --seed-start 0 --seed-end 0 --smoke-test
```

`--smoke-test` skips the training loop, uses a tiny/abbreviated model where
needed (e.g. B1's RF uses 5 trees instead of 400), and writes ONLY to
`tmp/smoke_tests/<method>/<task>/seedNNN/` — never `results/`. Verifies:
import → data prepare → task routing → model instantiate → one forward pass →
metric computation → output-file writing, all without a real training cost.

```bash
python scripts/self_check.py    # broader: checks every method + shared/ unit tests
```

## 7. Result schema

See `RESULTS_POLICY.md` for the complete, authoritative spec: directory layout
(`results/PHM2010/Bx_xxx/{D1,D2,D3}/seedNNN/`), the 8 required per-run files
(`metrics.json`, `metrics.csv`, `predictions.csv`, `training_log.csv`,
`run_meta.json`, `config_resolved.yaml`, `confusion_matrix.csv`, `DONE.flag`),
resume semantics, and what gets committed to git (only small `summary/*.csv`
files — never per-seed predictions/logs, never checkpoints).

## 8. Fresh-clone verification (what "self-contained" was checked against)

```
Self-contained: YES — grepped every methods/*/*.py and shared/*/*.py for any
  reference to the old parent project's paths (C:\Users\banghai\..., 代码/,
  baselines/ imports); only pre-existing files under 00_旧代码冻结备份/,
  01_主对比实验/, and shared/reproducibility/build_phm_task_frozen_preprocess.py
  (a pre-existing build-time regeneration utility, not imported by any adapter
  at runtime) still reference it -- those are historical/audit assets outside
  the new framework, explicitly left untouched per this project's read-only
  rule for prior rounds' work. Test scripts for B5/B6 default their raw-data
  env var to the old project's archive/ ONLY as an explicitly-allowed smoke-
  test convenience before download_phm2010.py has been run -- production code
  paths default to data/PHM2010/raw/.
Dependency on old parent project: NO (for methods/, shared/, scripts/, run_phm2010.py)
Windows actually tested: YES (this dev machine)
Ubuntu actually tested: NO (script prepared, not run on physical/VM hardware)
```

A literal `git clone` to a new directory + a full independent registry/import
check (`python -c "...runners.registry.list_methods()..."`) was run this round
and confirmed all 9 methods import cleanly with zero references to the old
parent project's absolute paths.

## 9. Known limitations (read before relying on this for real results)

- **No 9×3×101 full sweep has been run.** Every method's real-data verification
  this round is a smoke test (tiny/abbreviated) plus, for B1 only, one small
  real (non-smoke) validation run at seed 0 across D1/D2/D3. Running the full
  sweep for real papers-grade numbers is future work requiring explicit
  authorization (per this round's task scope).
- **Preprocessing is currently re-run from scratch on every single cell.** For
  the internal window-based methods (B1-B4, B9), `prepare()` re-executes the
  full ~90-140s feature-engineering pipeline for every seed, even though
  `PREPROCESS_SEED` (and therefore the entire preprocessed dataset) is IDENTICAL
  across all 101 seeds for a given task. At full scale (101 seeds × 3 tasks ×
  5 such methods) this wastes a very large amount of redundant compute. A
  per-(method,task,preprocess_seed) preprocessing cache (similar in spirit to
  the existing `shared/reproducibility/PHM2010_D1_frozen_preprocess/` for D1)
  would be a high-value follow-up before attempting the full sweep.
- **B6 (MTF-AViTK) real training was never executed this round** — only a
  plumbing/shape smoke test, per this project's established "write a tutorial,
  don't auto-execute" policy for this specific method (see `ENVIRONMENT.md`).
- **Ubuntu bootstrap is unverified on physical hardware.**
- **Kaggle download folder layout is unverified on a truly fresh download** —
  `scripts/verify_phm2010.py`'s expected layout was confirmed against the old
  project's already-organized local copy of the same Kaggle dataset, not
  against a byte-for-byte fresh `kaggle datasets download --unzip` run.
- **No Kaggle credential is bundled** (by design — never should be). B5-B8 need
  one-time manual setup per machine, see section 3.
- **High-VRAM methods**: B6 (1.2GB checkpoint, 8GB-card risk) is the standout;
  B5 (~50MB checkpoint) is comparatively low-risk but still GPU-bound.
- **B9 does not yet share B3's checkpoint** (trains its own identical-config
  backbone copy per cell) — a documented, deliberate scope decision to avoid
  building unsafe cross-adapter checkpoint-identity matching under time
  pressure (see `methods/B9_DC_PHSR/README.md` and task spec §34's explicit
  warning against a seed-mismatched reuse bug).
- **Method-internal env var naming is not perfectly uniform**: B5 accepts both
  `PHM2010_ROOT` and `PHM2010_RAW_ROOT`; B6/B7/B8 use `PHM2010_ROOT` only. Set
  `PHM2010_ROOT` and you're covered for all four.

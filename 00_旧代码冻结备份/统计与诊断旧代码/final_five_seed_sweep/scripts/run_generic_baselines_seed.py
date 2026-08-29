# -*- coding: utf-8 -*-
r"""
Runs the existing, UNMODIFIED `代码/7.4对比实验.py` (B1-B12 comparison
experiment -- includes RF=B5, TCN-GRU=B10, Multi-task TCN-GRU=B11,
DC-PSR/FGDS-PSI=B12) against the authoritative feature file
(baselines/htt_net/data/run_level_features_all.csv), for ONE additional
random seed, as part of the final 9-method 5-seed sweep.

Mirrors baselines/htt_net/data/run_b1_b12_recheck.py's existing,
already-validated monkey-patch pattern:
  1. Import main_experiment_3_fgds_psi_optimized ourselves first and
     override its FEATURE_FILE (authoritative CSV) and RANDOM_SEED
     (CLI --seed) attributes -- 7.4对比实验.py's own
     `import ... as base` line reuses this already-patched module from
     sys.modules rather than re-executing/resetting it.
  2. Set COMPARISON_RECHECK_DIR to a fresh, seed-specific output directory
     under final_five_seed_sweep/results/ so the original manuscript
     output (补充材料/.../4_comparison_experiment_recheck/, the seed=42
     run) is never touched or overwritten.
  3. Load 7.4对比实验.py via importlib (filename is not a valid Python
     identifier) and call its main() explicitly.

No hyperparameter is changed -- only RANDOM_SEED.

Usage:
    python run_generic_baselines_seed.py --seed 52
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
CODE_DIR = PROJECT_ROOT / "代码"
FEATURE_FILE = PROJECT_ROOT / "baselines" / "htt_net" / "data" / "run_level_features_all.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()

OUTPUT_DIR = PROJECT_ROOT / "final_five_seed_sweep" / "results" / "generic_baselines" / f"seed{args.seed}"

if not FEATURE_FILE.exists():
    raise FileNotFoundError(f"Authoritative feature file missing: {FEATURE_FILE}")

os.environ.setdefault("FGDS_RUN_DIR", str(Path(tempfile.gettempdir()) / "generic_baselines_5seed_base_import_cache"))
os.environ["COMPARISON_RECHECK_DIR"] = str(OUTPUT_DIR)

sys.path.insert(0, str(CODE_DIR))
import main_experiment_3_fgds_psi_optimized as base  # noqa: E402

base.FEATURE_FILE = FEATURE_FILE
base.RANDOM_SEED = args.seed
print(f"[generic-baselines seed {args.seed}] base.FEATURE_FILE = {base.FEATURE_FILE}")
print(f"[generic-baselines seed {args.seed}] base.RANDOM_SEED  = {base.RANDOM_SEED}")
print(f"[generic-baselines seed {args.seed}] output dir        = {OUTPUT_DIR}")

script_path = CODE_DIR / "7.4对比实验.py"
mod_name = f"comparison_recheck_script_seed{args.seed}"
spec = importlib.util.spec_from_file_location(mod_name, script_path)
mod = importlib.util.module_from_spec(spec)
sys.modules[mod_name] = mod
spec.loader.exec_module(mod)

mod.main()

#!/usr/bin/env bash
# One-shot environment bootstrap for Ubuntu (and other Linux).
#
# Usage:
#   bash scripts/bootstrap_ubuntu.sh
#
# Steps: check conda/mamba -> create/update the `wt_kuochong` env -> install
# core deps -> install torch (GPU build if an NVIDIA GPU is detected, else
# CPU) -> verify CUDA -> verify imports -> repo self-check. Does NOT download
# the raw PHM2010 dataset (run scripts/download_phm2010.py separately -- it
# needs one-time Kaggle credential setup, see MANUAL_RUN.md).
#
# STATUS: script prepared, not yet verified on a physical Ubuntu machine --
# see MANUAL_RUN.md "Platform verification status".
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== wt_kuochong bootstrap (Ubuntu/Linux) ==="

# 1. conda / mamba check
CONDA_BIN="$(command -v mamba || command -v conda || true)"
if [ -z "$CONDA_BIN" ]; then
    echo "ERROR: neither mamba nor conda found on PATH. Install Miniconda first:" >&2
    echo "  https://docs.conda.io/en/latest/miniconda.html" >&2
    exit 1
fi
echo "[1/7] found: $CONDA_BIN"

# 2. Create or reuse the wt_kuochong env (never touches base)
if "$CONDA_BIN" env list | grep -qE '^\s*wt_kuochong\s'; then
    echo "[2/7] conda env 'wt_kuochong' already exists -- reusing it."
else
    echo "[2/7] creating conda env 'wt_kuochong' from environment/environment.yml ..."
    "$CONDA_BIN" env create -f environment/environment.yml
fi

ENV_PATH="$("$CONDA_BIN" env list | grep -E '^\s*wt_kuochong\s' | awk '{print $NF}')"
PY_EXE="$ENV_PATH/bin/python"
if [ ! -x "$PY_EXE" ]; then
    echo "ERROR: could not locate wt_kuochong env's python at $PY_EXE." >&2
    echo "Activate it manually (conda activate wt_kuochong) and rerun the remaining steps -- see MANUAL_RUN.md." >&2
    exit 1
fi
echo "    using interpreter: $PY_EXE"

# 3. Core dependencies
echo "[3/7] installing core dependencies ..."
"$PY_EXE" -m pip install --upgrade pip --quiet
"$PY_EXE" -m pip install -r environment/requirements.txt --quiet

# 4. torch: GPU build if nvidia-smi is present and responds, else CPU build
echo "[4/7] detecting GPU for torch install ..."
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
    echo "    NVIDIA GPU detected -- installing torch (CUDA 11.8 build)"
    "$PY_EXE" -m pip install torch==2.7.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 --quiet
else
    echo "    no NVIDIA GPU detected -- installing CPU-only torch"
    "$PY_EXE" -m pip install torch==2.7.1 --extra-index-url https://download.pytorch.org/whl/cpu --quiet
fi

# 5. Verify CUDA / imports
echo "[5/7] verifying torch + CUDA ..."
"$PY_EXE" -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"

# 6. Verify environment end-to-end
echo "[6/7] running scripts/verify_environment.py ..."
"$PY_EXE" scripts/verify_environment.py

# 7. Repo self-check
echo "[7/7] running scripts/self_check.py ..."
set +e
"$PY_EXE" scripts/self_check.py
SELF_CHECK_EXIT=$?
set -e

echo ""
echo "=== Bootstrap complete ==="
echo "Next steps:"
echo "  1. (raw-signal methods B5-B8 only) python scripts/download_phm2010.py"
echo "  2. conda run -n wt_kuochong python run_phm2010.py --method B9 --tasks all --seed-start 0 --seed-end 100 --device auto --workers 1 --resume"
echo "See MANUAL_RUN.md for the full per-method command list."

exit $SELF_CHECK_EXIT

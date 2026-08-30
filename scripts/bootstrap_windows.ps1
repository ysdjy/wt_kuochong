# One-shot environment bootstrap for Windows.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows.ps1
#
# Steps: check conda -> create/update the `wt_kuochong` env -> install core deps
# -> install torch (GPU build if an NVIDIA GPU is detected, else CPU) ->
# verify CUDA -> verify imports -> feature-file sha256 check -> repo self-check.
# Does NOT download the raw PHM2010 dataset (run scripts/download_phm2010.py
# separately -- it needs one-time Kaggle credential setup, see MANUAL_RUN.md).

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "=== wt_kuochong bootstrap (Windows) ===" -ForegroundColor Cyan

# 1. conda / mamba check
$condaCmd = Get-Command conda -ErrorAction SilentlyContinue
if (-not $condaCmd) {
    Write-Error "conda not found on PATH. Install Miniconda/Anaconda first: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
}
Write-Host "[1/7] conda found: $($condaCmd.Source)"

# 2. Create or reuse the wt_kuochong env (never touches base)
$envExists = (conda env list) -match "^\s*wt_kuochong\s"
if ($envExists) {
    Write-Host "[2/7] conda env 'wt_kuochong' already exists -- reusing it."
} else {
    Write-Host "[2/7] creating conda env 'wt_kuochong' from environment/environment.yml ..."
    conda env create -f environment\environment.yml
    if ($LASTEXITCODE -ne 0) { Write-Error "conda env create failed"; exit 1 }
}

$PyExe = "$env:USERPROFILE\miniconda3\envs\wt_kuochong\python.exe"
if (-not (Test-Path $PyExe)) {
    # Fall back to asking conda where the env actually lives (Anaconda vs Miniconda vs custom install path).
    $envInfo = conda env list | Select-String "wt_kuochong"
    $envPath = ($envInfo -split "\s+")[-1]
    $PyExe = Join-Path $envPath "python.exe"
}
if (-not (Test-Path $PyExe)) {
    Write-Error "Could not locate wt_kuochong env's python.exe. Activate it manually and rerun this script's remaining steps -- see MANUAL_RUN.md."
    exit 1
}
Write-Host "    using interpreter: $PyExe"

# 3. Core dependencies
Write-Host "[3/7] installing core dependencies ..."
& $PyExe -m pip install --upgrade pip --quiet
& $PyExe -m pip install -r environment\requirements.txt --quiet
if ($LASTEXITCODE -ne 0) { Write-Error "pip install -r environment/requirements.txt failed"; exit 1 }

# 4. torch: GPU build if nvidia-smi is present and responds, else CPU build
Write-Host "[4/7] detecting GPU for torch install ..."
$hasGpu = $false
try {
    $null = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0) { $hasGpu = $true }
} catch { $hasGpu = $false }

if ($hasGpu) {
    Write-Host "    NVIDIA GPU detected -- installing torch (CUDA 11.8 build)"
    & $PyExe -m pip install torch==2.7.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 --quiet
} else {
    Write-Host "    no NVIDIA GPU detected -- installing CPU-only torch"
    & $PyExe -m pip install torch==2.7.1 --extra-index-url https://download.pytorch.org/whl/cpu --quiet
}
if ($LASTEXITCODE -ne 0) { Write-Error "torch install failed"; exit 1 }

# 5. Verify CUDA / imports
Write-Host "[5/7] verifying torch + CUDA ..."
& $PyExe -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"

# 6. Verify environment end-to-end (imports, method registry, unit tests, feature-file hash)
Write-Host "[6/7] running scripts/verify_environment.py ..."
& $PyExe scripts\verify_environment.py

# 7. Repo self-check
Write-Host "[7/7] running scripts/self_check.py ..."
& $PyExe scripts\self_check.py
$selfCheckExit = $LASTEXITCODE

Write-Host ""
Write-Host "=== Bootstrap complete ===" -ForegroundColor Cyan
Write-Host "Next steps:"
Write-Host "  1. (raw-signal methods B5-B8 only) python scripts\download_phm2010.py"
Write-Host "  2. conda run -n wt_kuochong python run_phm2010.py --method B9 --tasks all --seed-start 0 --seed-end 100 --device auto --workers 1 --resume"
Write-Host "See MANUAL_RUN.md for the full per-method command list."

exit $selfCheckExit

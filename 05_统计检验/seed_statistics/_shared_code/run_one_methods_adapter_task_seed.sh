#!/bin/bash
# Generic B6/B7/B8 (methods/ adapter reuse), PHM2010. Usage: run_one_methods_adapter_task_seed.sh "B7,D2,17"
# Set PYTHON_EXE / PHM2010_ROOT env vars to override interpreter/raw-data location.
triplet="$1"
method="${triplet%%,*}"
rest="${triplet#*,}"
task="${rest%%,*}"
seed="${rest##*,}"
SHARED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANDSCAPE_DIR="$SHARED_DIR/../${method}_PHM2010_${task}_seed_landscape"
LOG_DIR="$LANDSCAPE_DIR/logs"
mkdir -p "$LOG_DIR"
"${PYTHON_EXE:-python}" "$SHARED_DIR/run_methods_adapter_seed_task.py" \
  --method "$method" --task "$task" --train_seed "$seed" \
  --results_root "$LANDSCAPE_DIR/results" \
  > "$LOG_DIR/seed${seed}.log" 2>&1
echo "method=$method task=$task seed=$seed exit=$?"

#!/bin/bash
# B4 (HTT-Net), NASA. Usage: run_one_b4_nasa_task_seed.sh "N2,17"
pair="$1"
task="${pair%%,*}"
seed="${pair##*,}"
SHARED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANDSCAPE_DIR="$SHARED_DIR/../B4_NASA_${task}_seed_landscape"
LOG_DIR="$LANDSCAPE_DIR/logs"
mkdir -p "$LOG_DIR"
"${PYTHON_EXE:-python}" "$SHARED_DIR/run_b4_nasa_seed_task.py" \
  --task "$task" --train_seed "$seed" \
  --results_root "$LANDSCAPE_DIR/results" \
  --backbone_root "$LANDSCAPE_DIR/backbone_checkpoints" \
  > "$LOG_DIR/seed${seed}.log" 2>&1
echo "task=$task seed=$seed exit=$?"

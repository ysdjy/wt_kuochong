#!/bin/bash
# Usage: run_one_task_seed.sh "D2,17"
pair="$1"
task="${pair%%,*}"
seed="${pair##*,}"
SHARED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANDSCAPE_DIR="$SHARED_DIR/../B9_PHM2010_${task}_seed_landscape"
B3_LANDSCAPE_DIR="$SHARED_DIR/../B3_PHM2010_${task}_seed_landscape"
LOG_DIR="$LANDSCAPE_DIR/logs"
mkdir -p "$LOG_DIR" "$B3_LANDSCAPE_DIR/results"
"C:\Users\banghai\miniconda3\envs\dcpsr\python.exe" "$SHARED_DIR/run_seed_task.py" \
  --task "$task" --train_seed "$seed" \
  --results_root "$LANDSCAPE_DIR/results" \
  --backbone_root "$LANDSCAPE_DIR/backbone_checkpoints" \
  --b3_results_root "$B3_LANDSCAPE_DIR/results" \
  > "$LOG_DIR/seed${seed}.log" 2>&1
echo "task=$task seed=$seed exit=$?"

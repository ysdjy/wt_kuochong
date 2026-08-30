#!/bin/bash
# Usage: run_one_mtw_task_seed.sh "D1-M,17"
pair="$1"
task="${pair%%,*}"
seed="${pair##*,}"
task_dir_tag="${task//-/}"   # D1-M -> D1M (matches folder naming used elsewhere)
SHARED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANDSCAPE_DIR="$SHARED_DIR/../B9_MTW_${task_dir_tag}_seed_landscape"
B3_LANDSCAPE_DIR="$SHARED_DIR/../B3_MTW_${task_dir_tag}_seed_landscape"
LOG_DIR="$LANDSCAPE_DIR/logs"
mkdir -p "$LOG_DIR" "$B3_LANDSCAPE_DIR/results"
"C:\Users\banghai\miniconda3\envs\dcpsr\python.exe" "$SHARED_DIR/run_mtw_seed_task.py" \
  --task "$task" --train_seed "$seed" \
  --results_root "$LANDSCAPE_DIR/results" \
  --backbone_root "$LANDSCAPE_DIR/backbone_checkpoints" \
  --b3_results_root "$B3_LANDSCAPE_DIR/results" \
  > "$LOG_DIR/seed${seed}.log" 2>&1
echo "task=$task seed=$seed exit=$?"

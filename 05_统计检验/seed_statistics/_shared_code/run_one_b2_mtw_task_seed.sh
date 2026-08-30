#!/bin/bash
# B2 (TCN-GRU), MTW-CM. Usage: run_one_b2_mtw_task_seed.sh "D2-M,17"
# Set PYTHON_EXE env var to override the interpreter (default: python.exe on PATH).
pair="$1"
task="${pair%%,*}"
seed="${pair##*,}"
task_dirname="${task//-/}"   # "D1-M" -> "D1M" (matches the existing B9/B3 landscape dir naming)
SHARED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANDSCAPE_DIR="$SHARED_DIR/../B2_MTW_${task_dirname}_seed_landscape"
LOG_DIR="$LANDSCAPE_DIR/logs"
mkdir -p "$LOG_DIR"
"${PYTHON_EXE:-python}" "$SHARED_DIR/run_b2_mtw_seed_task.py" \
  --task "$task" --train_seed "$seed" \
  --results_root "$LANDSCAPE_DIR/results" \
  --backbone_root "$LANDSCAPE_DIR/backbone_checkpoints" \
  > "$LOG_DIR/seed${seed}.log" 2>&1
echo "task=$task seed=$seed exit=$?"

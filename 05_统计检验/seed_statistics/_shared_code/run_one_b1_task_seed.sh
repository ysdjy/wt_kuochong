#!/bin/bash
# B1 (RF), PHM2010. Usage: run_one_b1_task_seed.sh "D2,17"
# Set PYTHON_EXE env var to override the interpreter (default: python.exe on PATH,
# i.e. whatever conda env you've activated -- see ../../../MANUAL_RUN.md or the
# B1B2_CROSS_MACHINE_TUTORIAL.md in this directory's parent for setup).
pair="$1"
task="${pair%%,*}"
seed="${pair##*,}"
SHARED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANDSCAPE_DIR="$SHARED_DIR/../B1_PHM2010_${task}_seed_landscape"
LOG_DIR="$LANDSCAPE_DIR/logs"
mkdir -p "$LOG_DIR"
"${PYTHON_EXE:-python}" "$SHARED_DIR/run_b1_seed_task.py" \
  --task "$task" --train_seed "$seed" \
  --results_root "$LANDSCAPE_DIR/results" \
  > "$LOG_DIR/seed${seed}.log" 2>&1
echo "task=$task seed=$seed exit=$?"

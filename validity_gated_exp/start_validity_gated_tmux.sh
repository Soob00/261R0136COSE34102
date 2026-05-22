#!/bin/bash
set -euo pipefail

SESSION="validity_gated"
EXP_DIR="/home/ubuntu/hi/validity_gated_exp"
VENV_PY="/home/ubuntu/hi/.venv/bin/python"
LOG_DIR="$EXP_DIR/logs"
RUN_LOG="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
CKPT_LOG="$EXP_DIR/checkpoint.log"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION"
    echo "attach with: tmux attach -t $SESSION"
    exit 0
fi

tmux new-session -d -s "$SESSION" -n train \
    "bash -lc 'cd \"$EXP_DIR\" && env PYTHONUNBUFFERED=1 \"$VENV_PY\" run_exp.py 2>&1 | tee -a \"$RUN_LOG\"'"

tmux split-window -v -t "$SESSION":train \
    "bash -lc 'cd \"$EXP_DIR\" && exec bash ./auto_checkpoint.sh'"

tmux select-pane -t "$SESSION":train.0
tmux split-window -h -t "$SESSION":train.0 \
    "bash -lc 'while true; do clear; echo \"=== validity_gated progress ===\"; echo \"time: \$(date +%Y-%m-%d\\ %H:%M:%S)\"; echo; echo \"-- latest experiment log --\"; tail -n 18 \"$RUN_LOG\" 2>/dev/null || echo \"waiting for experiment log...\"; echo; echo \"-- latest checkpoint log --\"; tail -n 12 \"$CKPT_LOG\" 2>/dev/null || echo \"waiting for checkpoint log...\"; echo; echo \"-- checkpoints --\"; ls -1t \"$EXP_DIR\"/checkpoints/*.pt 2>/dev/null | head -n 5 | sed 's|.*/||' || true; sleep 30; done'"

tmux split-window -v -t "$SESSION":train.1 \
    "bash -lc 'cd \"$EXP_DIR\" && exec bash ./auto_push_validity.sh'"

tmux select-layout -t "$SESSION":train tiled

echo "tmux session started: $SESSION"
echo "attach with: tmux attach -t $SESSION"
echo "run log: $RUN_LOG"
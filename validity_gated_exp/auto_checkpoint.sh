#!/bin/bash
# 체크포인트 및 결과 자동 저장 (매 30초마다)

RESULT_FILE="/home/ubuntu/hi/validity_gated_exp/results.json"
BACKUP_DIR="/home/ubuntu/hi/validity_gated_exp/results_backup"
CKPT_DIR="/home/ubuntu/hi/validity_gated_exp/checkpoints"
LOG_FILE="/home/ubuntu/hi/validity_gated_exp/checkpoint.log"

mkdir -p "$BACKUP_DIR"
touch "$LOG_FILE"

PUSH_INTERVAL=20  # 20 × 30초 = 10분마다 push
push_counter=0

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    if [ -f "$RESULT_FILE" ]; then
        ts=$(date '+%Y%m%d_%H%M%S')
        backup_file="$BACKUP_DIR/results_${ts}.json"
        cp "$RESULT_FILE" "$backup_file"

        ls -t "$BACKUP_DIR"/results_*.json 2>/dev/null | tail -n +6 | xargs -r rm

        echo "[$timestamp] ✓ 백업 저장: $backup_file" >> "$LOG_FILE"
    fi

    if [ -d "$CKPT_DIR" ]; then
        ckpt_count=$(find "$CKPT_DIR" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l)
        latest_ckpt=$(find "$CKPT_DIR" -maxdepth 1 -type f -name '*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)

        if [ -n "$latest_ckpt" ]; then
            echo "[$timestamp] 체크포인트: ${ckpt_count}개 | 최신: $(basename "$latest_ckpt")" >> "$LOG_FILE"
        else
            echo "[$timestamp] 체크포인트: ${ckpt_count}개" >> "$LOG_FILE"
        fi
    fi

    /home/ubuntu/hi/.venv/bin/python /home/ubuntu/hi/validity_gated_exp/update_progress.py 2>/dev/null

    push_counter=$((push_counter + 1))
    if [ $push_counter -ge $PUSH_INTERVAL ]; then
        push_counter=0
        cd /home/ubuntu/hi
        git add validity_gated_exp/status.json \
                validity_gated_exp/results.json \
                validity_gated_exp/checkpoints/*.pt \
                validity_gated_exp/PROGRESS.md 2>/dev/null
        if ! git diff --cached --quiet; then
            git commit -m "auto: checkpoint update $(date '+%Y-%m-%d %H:%M:%S')"
            git push origin master >> "$LOG_FILE" 2>&1 \
                && echo "[$timestamp] ✅ git push 완료" >> "$LOG_FILE" \
                || echo "[$timestamp] ❌ git push 실패" >> "$LOG_FILE"
        fi
    fi

    sleep 30
done
#!/bin/bash
# validity_gated 실험 완료 시 자동 git push

EXP_DIR="/home/ubuntu/hi/validity_gated_exp"
STATUS_FILE="$EXP_DIR/status.json"
RESULT_FILE="$EXP_DIR/results.json"
MAX_WAIT=259200  # 3일
ELAPSED=0
INTERVAL=60

echo "[$(date '+%Y-%m-%d %H:%M:%S')] git push 감시 시작 (status: completed 대기 중...)"

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if [ -f "$STATUS_FILE" ]; then
        STATUS=$(python3 -c "import json; d=json.load(open('$STATUS_FILE')); print(d.get('status',''))" 2>/dev/null)
        if [ "$STATUS" = "completed" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ 실험 완료 감지! git push 시작..."

            cd /home/ubuntu/hi
            git add validity_gated_exp/results.json validity_gated_exp/status.json
            git commit -m "Add validity_gated experiment results - $(date '+%Y-%m-%d %H:%M:%S')"

            if git push origin master; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ GitHub push 완료!"
            else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ push 실패. 수동으로 확인 필요."
            fi
            exit 0
        fi
    fi

    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ 타임아웃 (6시간 경과)"

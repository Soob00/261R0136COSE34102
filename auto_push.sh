#!/bin/bash

# 결과 파일 감시 후 자동 push

RESULT_FILE="/home/ubuntu/hi/results_consistency_reg.json"
MAX_WAIT=14400  # 4시간
ELAPSED=0

echo "📊 결과 파일 감시 시작..."
echo "위치: $RESULT_FILE"
echo "최대 대기: ${MAX_WAIT}초 (약 4시간)"

while [ $ELAPSED -lt $MAX_WAIT ]; do
    # 파일 크기 확인 (크기가 크면 결과가 저장된 것)
    if [ -f "$RESULT_FILE" ]; then
        SIZE=$(stat -f%z "$RESULT_FILE" 2>/dev/null || stat -c%s "$RESULT_FILE" 2>/dev/null)

        # 크기가 5KB 이상이면 (빈 파일이 아님)
        if [ "$SIZE" -gt 5000 ]; then
            echo ""
            echo "✅ 결과 파일 감지! (크기: $SIZE bytes)"
            echo "🔄 Git에 커밋/푸시 중..."

            cd /home/ubuntu/hi
            git add results_consistency_reg.json
            git commit -m "Add counterfactual consistency experiment results - $(date '+%Y-%m-%d %H:%M:%S')"

            if git push origin master; then
                echo "✅ GitHub에 성공적으로 push됨!"
                echo "   https://github.com/Soob00/hi"
                exit 0
            else
                echo "❌ Push 실패"
                exit 1
            fi
        fi
    fi

    sleep 30  # 30초마다 확인
    ELAPSED=$((ELAPSED + 30))

    # 진행 상황 출력
    if [ $((ELAPSED % 300)) -eq 0 ]; then
        echo "⏳ 대기 중... (${ELAPSED}초 경과)"
    fi
done

echo "❌ 타임아웃 (4시간 경과)"
exit 1

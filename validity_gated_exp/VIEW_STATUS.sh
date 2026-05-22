#!/bin/bash
# 실시간 상태 모니터링 - 3초마다 업데이트

clear
echo "📊 Validity-Gated Experiment Monitor"
echo "=================================="
echo ""

while true; do
    if [ -f status.json ]; then
        clear
        echo "📊 Validity-Gated Experiment Monitor"
        echo "=================================="
        echo ""

        # JSON 파싱해서 주요 정보 표시
        python3 << 'EOF'
import json
from pathlib import Path

with open('status.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Status: {data.get('status', 'unknown').upper()}")
print(f"Started: {data.get('start_time', 'N/A')}")
print(f"Last Update: {data.get('last_update', 'N/A')}")
print(f"Current Exp: {data.get('current_exp', 'N/A')}")
print(f"Progress: {data.get('progress', 'N/A')}")
print("")
print("=" * 60)
print("📝 Recent Logs:")
print("=" * 60)
logs = data.get('logs', [])
for log in logs[-15:]:
    print(log)
EOF

        echo ""
        echo "=================================="
        echo "(갱신중... Ctrl+C로 중지)"
    else
        echo "⏳ 대기 중... (status.json 아직 생성되지 않음)"
    fi

    sleep 3
done

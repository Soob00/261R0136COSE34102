#!/usr/bin/env bash
# Simple monitor that appends periodic snapshots of train.log to monitor.log
LOG_FILE="$(dirname "$0")/../train.log"
OUT_FILE="$(dirname "$0")/../monitor.log"
INTERVAL=1800  # seconds (30 minutes)

mkdir -p "$(dirname "$OUT_FILE")"
echo "Starting monitor: logging $LOG_FILE -> $OUT_FILE every ${INTERVAL}s" >> "$OUT_FILE"
while true; do
  echo "=== $(date --iso-8601=seconds) ===" >> "$OUT_FILE"
  echo "---- tail ${LOG_FILE} ----" >> "$OUT_FILE"
  if [ -f "$LOG_FILE" ]; then
    tail -n 20 "$LOG_FILE" >> "$OUT_FILE" 2>&1
  else
    echo "(no log yet)" >> "$OUT_FILE"
  fi
  echo "" >> "$OUT_FILE"
  sleep $INTERVAL
done

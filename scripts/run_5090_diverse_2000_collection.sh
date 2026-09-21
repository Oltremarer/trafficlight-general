#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_ROOT JINAN_DATA_DIR HANGZHOU_DATA_DIR" >&2
  exit 2
fi

RUN_ROOT=$(realpath -m "$1")
JINAN_DATA_DIR=$(realpath "$2")
HANGZHOU_DATA_DIR=$(realpath "$3")
SOURCE_DIR=$(pwd)
case "$RUN_ROOT" in
  /mnt/pan/*) ;;
  *) echo "RUN_ROOT must be under /mnt/pan" >&2; exit 2 ;;
esac
if [[ -e "$RUN_ROOT" ]]; then
  echo "RUN_ROOT already exists: $RUN_ROOT" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT/logs"
printf '%s\n' "formal_2000_per_city_collection" > "$RUN_ROOT/stage.txt"
git rev-parse HEAD > "$RUN_ROOT/source_commit.txt"

nohup bash "$SOURCE_DIR/scripts/run_5090_diverse_city_collection.sh" \
  "$RUN_ROOT/jinan" Jinan "$JINAN_DATA_DIR" \
  > "$RUN_ROOT/logs/jinan_launcher.log" 2>&1 &
JINAN_PID=$!
nohup bash "$SOURCE_DIR/scripts/run_5090_diverse_city_collection.sh" \
  "$RUN_ROOT/hangzhou" Hangzhou "$HANGZHOU_DATA_DIR" \
  > "$RUN_ROOT/logs/hangzhou_launcher.log" 2>&1 &
HANGZHOU_PID=$!
printf '%s\n' "$JINAN_PID" > "$RUN_ROOT/jinan_launcher.pid"
printf '%s\n' "$HANGZHOU_PID" > "$RUN_ROOT/hangzhou_launcher.pid"

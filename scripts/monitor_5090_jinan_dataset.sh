#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 2
fi

RUN_DIR=$(realpath "$1")
case "$RUN_DIR" in
  /mnt/pan/*) ;;
  *) echo "RUN_DIR must be under /mnt/pan" >&2; exit 2 ;;
esac

echo "run_dir=$RUN_DIR"
echo "stage=$(cat "$RUN_DIR/stage.txt" 2>/dev/null || echo unavailable)"
echo "trajectory_manifests=$(find "$RUN_DIR" -type f -name trajectory.manifest.json 2>/dev/null | wc -l)"
if [[ -f "$RUN_DIR/trajectories_rules/rules_progress.json" ]]; then
  echo "rules_progress=$(tr -d '\n' < "$RUN_DIR/trajectories_rules/rules_progress.json")"
fi
if [[ -f "$RUN_DIR/trajectories_dqn/dqn_progress.json" ]]; then
  echo "dqn_progress=$(tr -d '\n' < "$RUN_DIR/trajectories_dqn/dqn_progress.json")"
fi
for name in rules dqn; do
  pid_file="$RUN_DIR/$name.pid"
  if [[ -f "$pid_file" ]]; then
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      echo "$name=running(pid=$pid)"
    else
      echo "$name=not_running(pid=$pid)"
    fi
  fi
done
echo "gpu=$(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader)"
echo "disk=$(df -h /mnt/pan | tail -1)"
echo "rules_log_tail:"
tail -5 "$RUN_DIR/logs/rules.log" 2>/dev/null || true
echo "dqn_log_tail:"
tail -5 "$RUN_DIR/logs/dqn.log" 2>/dev/null || true

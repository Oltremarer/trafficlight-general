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
for progress in \
  "$RUN_DIR/trajectories_rules/rules_progress.json" \
  "$RUN_DIR/trajectories_dqn/dqn_progress.json"; do
  if [[ -f "$progress" ]]; then
    echo "$(basename "$progress")=$(tr -d '\n' < "$progress")"
  fi
done
for name in launcher rules dqn finetune_direct finetune_latent; do
  if [[ -f "$RUN_DIR/$name.pid" ]]; then
    pid=$(cat "$RUN_DIR/$name.pid")
    if kill -0 "$pid" 2>/dev/null; then
      echo "$name=running(pid=$pid)"
    else
      echo "$name=not_running(pid=$pid)"
    fi
  fi
done
echo "gpu=$(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader)"
for log in rules dqn finetune_direct finetune_latent finetune_comparison; do
  path="$RUN_DIR/logs/$log.log"
  if [[ -f "$path" ]]; then
    echo "$log.log tail:"
    tail -3 "$path" || true
  fi
done

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

PYTHON="$RUN_DIR/venv/bin/python"
TDMPC2_CHECKPOINT="$RUN_DIR/models/tdmpc2-mt80-5M.pt"
for path in \
  "$PYTHON" \
  "$TDMPC2_CHECKPOINT" \
  "$RUN_DIR/collection_random" \
  "$RUN_DIR/collection_max_pressure" \
  "$RUN_DIR/collection_shared_dqn"; do
  if [[ ! -e "$path" ]]; then
    echo "required input does not exist: $path" >&2
    exit 2
  fi
done

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/tmp" "$RUN_DIR/python_cache"
export TMPDIR="$RUN_DIR/tmp"
export PYTHONPYCACHEPREFIX="$RUN_DIR/python_cache"

DATA_ROOTS=(
  --manifest-root "$RUN_DIR/collection_random"
  --manifest-root "$RUN_DIR/collection_max_pressure"
  --manifest-root "$RUN_DIR/collection_shared_dqn"
)
TRAIN_COMMON=(
  "${DATA_ROOTS[@]}"
  --device cuda
  --history-length 3
  --rollout-horizon 5
  --validation-fraction 0.2
  --seed 73
  --encoder-hidden-dim 128
  --epochs 30
  --batch-size 128
  --adapter-learning-rate 3e-4
  --pretrained-learning-rate 3e-5
  --frozen-pretrained-epochs 5
  --num-workers 2
)

printf '%s\n' "parallel_model_training_metrics_v2" > "$RUN_DIR/stage.txt"
"$PYTHON" -m cityflow_tsc.train_world_model_prediction train \
  "${TRAIN_COMMON[@]}" --output "$RUN_DIR/direct_model_v2" --model direct \
  > "$RUN_DIR/logs/direct_model_v2.log" 2>&1 &
PID_DIRECT=$!

"$PYTHON" -m cityflow_tsc.train_world_model_prediction train \
  "${TRAIN_COMMON[@]}" --output "$RUN_DIR/latent_model_v2" --model latent \
  --tdmpc2-checkpoint "$TDMPC2_CHECKPOINT" \
  > "$RUN_DIR/logs/latent_model_v2.log" 2>&1 &
PID_LATENT=$!

wait "$PID_DIRECT"
wait "$PID_LATENT"

printf '%s\n' "comparison_metrics_v2" > "$RUN_DIR/stage.txt"
"$PYTHON" -m cityflow_tsc.train_world_model_prediction compare \
  --direct-summary "$RUN_DIR/direct_model_v2/prediction_summary.json" \
  --latent-summary "$RUN_DIR/latent_model_v2/prediction_summary.json" \
  --output "$RUN_DIR/comparison_v2" \
  > "$RUN_DIR/logs/comparison_v2.log" 2>&1

printf '%s\n' "metrics_v2_complete" > "$RUN_DIR/stage.txt"
printf '%s\n' "0" > "$RUN_DIR/metrics_v2_exit_code.txt"

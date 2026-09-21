#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_DIR DATASET_INDEX TDMPC2_SOURCE_CHECKPOINT" >&2
  exit 2
fi

RUN_DIR=$(realpath "$1")
DATASET_INDEX=$(realpath "$2")
TDMPC2_SOURCE=$(realpath "$3")
SOURCE_DIR=$(pwd)
case "$RUN_DIR" in
  /mnt/pan/*) ;;
  *) echo "RUN_DIR must be under /mnt/pan" >&2; exit 2 ;;
esac
if [[ ! -w /mnt/pan || ! -f "$DATASET_INDEX" || ! -s "$TDMPC2_SOURCE" ]]; then
  echo "finetune inputs are unavailable" >&2
  exit 2
fi

PYTHON="$RUN_DIR/venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "run venv is unavailable: $PYTHON" >&2
  exit 2
fi
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/models" "$RUN_DIR/tmp" "$RUN_DIR/python_cache"
# PyTorch DataLoader workers create AF_UNIX sockets beneath TMPDIR.  RUN_DIR is
# deliberately descriptive and can exceed the kernel's socket-path limit, so
# keep this run-specific temporary directory short while retaining it on the
# required /mnt/pan volume.
RUN_TMP="/mnt/pan/.world_model_tmp/$(printf '%s' "$RUN_DIR" | sha256sum | cut -c1-12)"
mkdir -p "$RUN_TMP"
export TMPDIR="$RUN_TMP"
export PYTHONPYCACHEPREFIX="$RUN_DIR/python_cache"
"$PYTHON" -m pip install --quiet -e "${SOURCE_DIR}[rl]"

TDMPC2_CHECKPOINT="$RUN_DIR/models/tdmpc2-mt80-5M.pt"
if [[ ! -s "$TDMPC2_CHECKPOINT" ]]; then
  cp "$TDMPC2_SOURCE" "$TDMPC2_CHECKPOINT"
fi
sha256sum "$TDMPC2_CHECKPOINT" > "$TDMPC2_CHECKPOINT.sha256"

FINETUNE_ROOT="$RUN_DIR/finetune"
if [[ -e "$FINETUNE_ROOT" ]]; then
  if [[ -f "$FINETUNE_ROOT/direct_model/prediction_summary.json" || \
        -f "$FINETUNE_ROOT/latent_model/prediction_summary.json" ]]; then
    echo "refusing to overwrite completed or partially completed finetune output: $FINETUNE_ROOT" >&2
    exit 2
  fi
  FAILED_ROOT="$RUN_DIR/finetune_failed_$(date -u +%Y%m%dT%H%M%SZ)"
  mv "$FINETUNE_ROOT" "$FAILED_ROOT"
  printf '%s\n' "$FAILED_ROOT" > "$RUN_DIR/finetune_previous_failed_dir.txt"
fi
mkdir -p "$FINETUNE_ROOT"
COMMON=(
  --dataset-index "$DATASET_INDEX"
  --device cuda
  --history-length 3
  --rollout-horizon 5
  --seed 73
  --encoder-hidden-dim 128
  --movement-latent-dim 64
  --epochs 30
  --batch-size 128
  --adapter-learning-rate 3e-4
  --pretrained-learning-rate 3e-5
  --frozen-pretrained-epochs 5
  --reward-loss-weight 0.1
  --consistency-loss-weight 2.0
  --movement-consistency-loss-weight 2.0
  --reconstruction-loss-weight 0.5
  --num-workers 2
)

printf '%s\n' "parallel_direct_and_latent_finetune" > "$RUN_DIR/stage.txt"
"$PYTHON" -m cityflow_tsc.train_world_model_prediction train \
  "${COMMON[@]}" \
  --output "$FINETUNE_ROOT/direct_model" \
  --model direct \
  > "$RUN_DIR/logs/finetune_direct.log" 2>&1 &
PID_DIRECT=$!
"$PYTHON" -m cityflow_tsc.train_world_model_prediction train \
  "${COMMON[@]}" \
  --output "$FINETUNE_ROOT/latent_model" \
  --model latent \
  --tdmpc2-checkpoint "$TDMPC2_CHECKPOINT" \
  > "$RUN_DIR/logs/finetune_latent.log" 2>&1 &
PID_LATENT=$!
printf '%s\n' "$PID_DIRECT" > "$RUN_DIR/finetune_direct.pid"
printf '%s\n' "$PID_LATENT" > "$RUN_DIR/finetune_latent.pid"

set +e
wait "$PID_DIRECT"
DIRECT_EXIT=$?
wait "$PID_LATENT"
LATENT_EXIT=$?
set -e
printf '%s\n' "$DIRECT_EXIT" > "$RUN_DIR/finetune_direct.exit_code.txt"
printf '%s\n' "$LATENT_EXIT" > "$RUN_DIR/finetune_latent.exit_code.txt"
if [[ "$DIRECT_EXIT" -ne 0 || "$LATENT_EXIT" -ne 0 ]]; then
  printf '%s\n' "finetune_failed" > "$RUN_DIR/stage.txt"
  printf '%s\n' "1" > "$RUN_DIR/finetune.exit_code.txt"
  exit 1
fi

printf '%s\n' "finetune_comparison" > "$RUN_DIR/stage.txt"
"$PYTHON" -m cityflow_tsc.train_world_model_prediction compare \
  --direct-summary "$FINETUNE_ROOT/direct_model/prediction_summary.json" \
  --latent-summary "$FINETUNE_ROOT/latent_model/prediction_summary.json" \
  --output "$FINETUNE_ROOT/comparison" \
  > "$RUN_DIR/logs/finetune_comparison.log" 2>&1
printf '%s\n' "complete" > "$RUN_DIR/stage.txt"
printf '%s\n' "0" > "$RUN_DIR/finetune.exit_code.txt"

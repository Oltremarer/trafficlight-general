#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_DIR ROADNET FLOW" >&2
  exit 2
fi

RUN_DIR=$(realpath "$1")
ROADNET=$(realpath "$2")
FLOW=$(realpath "$3")
SOURCE_DIR=$(pwd)

case "$RUN_DIR" in
  /mnt/pan/*) ;;
  *) echo "RUN_DIR must be under /mnt/pan" >&2; exit 2 ;;
esac

if [[ ! -w /mnt/pan ]]; then
  echo "/mnt/pan is unavailable or not writable" >&2
  exit 2
fi
if [[ ! -f "$ROADNET" || ! -f "$FLOW" ]]; then
  echo "roadnet or flow file does not exist" >&2
  exit 2
fi

mkdir -p "$RUN_DIR" "$RUN_DIR/logs" "$RUN_DIR/models" "$RUN_DIR/tmp" \
  "$RUN_DIR/python_cache" "$RUN_DIR/pip_cache"
export TMPDIR="$RUN_DIR/tmp"
export PYTHONPYCACHEPREFIX="$RUN_DIR/python_cache"
export PIP_CACHE_DIR="$RUN_DIR/pip_cache"

BASE_PYTHON=${BASE_PYTHON:-/home/chenyuyang/miniconda3/envs/c2t/bin/python}
if [[ ! -x "$BASE_PYTHON" ]]; then
  echo "base Python is not executable: $BASE_PYTHON" >&2
  exit 2
fi

printf '%s\n' "environment_setup" > "$RUN_DIR/stage.txt"
"$BASE_PYTHON" -m venv --system-site-packages "$RUN_DIR/venv"
PYTHON="$RUN_DIR/venv/bin/python"
"$PYTHON" -m pip install --quiet -e "${SOURCE_DIR}[rl]"

"$PYTHON" - <<PY > "$RUN_DIR/environment.json"
import json
import platform
import torch
import cityflow
print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "cityflow": cityflow.__file__,
    "source_dir": "$SOURCE_DIR",
    "source_commit": "$(git rev-parse HEAD)",
    "roadnet": "$ROADNET",
    "flow": "$FLOW",
}, indent=2, sort_keys=True))
PY
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu \
  --format=csv,noheader > "$RUN_DIR/gpu_before.txt"

TDMPC2_CHECKPOINT="$RUN_DIR/models/tdmpc2-mt80-5M.pt"
if [[ ! -s "$TDMPC2_CHECKPOINT" ]]; then
  wget --tries=3 --timeout=30 \
    "https://huggingface.co/nicklashansen/tdmpc2/resolve/main/multitask/mt80-5M.pt?download=true" \
    -O "$TDMPC2_CHECKPOINT"
fi
sha256sum "$TDMPC2_CHECKPOINT" > "$TDMPC2_CHECKPOINT.sha256"

COMMON=(
  --roadnet "$ROADNET"
  --flow "$FLOW"
  --duration 3600
  --decision-interval 30
  --simulator-step 1
  --yellow-time 5
  --green-phases 1,2,3,4
  --thread-num 1
)

printf '%s\n' "trajectory_collection" > "$RUN_DIR/stage.txt"
"$PYTHON" -m cityflow_tsc.train_world_model_prediction collect \
  "${COMMON[@]}" --output "$RUN_DIR/collection_random" --policy random \
  --episodes 12 --seed 0 --device cpu \
  > "$RUN_DIR/logs/collection_random.log" 2>&1 &
PID_RANDOM=$!

"$PYTHON" -m cityflow_tsc.train_world_model_prediction collect \
  "${COMMON[@]}" --output "$RUN_DIR/collection_max_pressure" --policy max_pressure \
  --episodes 4 --seed 1000 --device cpu \
  > "$RUN_DIR/logs/collection_max_pressure.log" 2>&1 &
PID_MAX_PRESSURE=$!

"$PYTHON" -m cityflow_tsc.train_world_model_prediction collect \
  "${COMMON[@]}" --output "$RUN_DIR/collection_shared_dqn" --policy shared_dqn \
  --episodes 12 --seed 2000 --device cuda \
  > "$RUN_DIR/logs/collection_shared_dqn.log" 2>&1 &
PID_DQN=$!

wait "$PID_RANDOM"
wait "$PID_MAX_PRESSURE"
wait "$PID_DQN"

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

printf '%s\n' "parallel_model_training" > "$RUN_DIR/stage.txt"
"$PYTHON" -m cityflow_tsc.train_world_model_prediction train \
  "${TRAIN_COMMON[@]}" --output "$RUN_DIR/direct_model" --model direct \
  > "$RUN_DIR/logs/direct_model.log" 2>&1 &
PID_DIRECT=$!

"$PYTHON" -m cityflow_tsc.train_world_model_prediction train \
  "${TRAIN_COMMON[@]}" --output "$RUN_DIR/latent_model" --model latent \
  --tdmpc2-checkpoint "$TDMPC2_CHECKPOINT" \
  > "$RUN_DIR/logs/latent_model.log" 2>&1 &
PID_LATENT=$!

wait "$PID_DIRECT"
wait "$PID_LATENT"

printf '%s\n' "comparison" > "$RUN_DIR/stage.txt"
"$PYTHON" -m cityflow_tsc.train_world_model_prediction compare \
  --direct-summary "$RUN_DIR/direct_model/prediction_summary.json" \
  --latent-summary "$RUN_DIR/latent_model/prediction_summary.json" \
  --output "$RUN_DIR/comparison" \
  > "$RUN_DIR/logs/comparison.log" 2>&1

nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu \
  --format=csv,noheader > "$RUN_DIR/gpu_after.txt"
printf '%s\n' "complete" > "$RUN_DIR/stage.txt"
printf '%s\n' "0" > "$RUN_DIR/exit_code.txt"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_DIR JINAN_DATA_DIR" >&2
  exit 2
fi

RUN_DIR=$(realpath "$1")
JINAN_DATA_DIR=$(realpath "$2")
SOURCE_DIR=$(pwd)

case "$RUN_DIR" in
  /mnt/pan/*) ;;
  *) echo "RUN_DIR must be under /mnt/pan" >&2; exit 2 ;;
esac
if [[ ! -w /mnt/pan ]]; then
  echo "/mnt/pan is unavailable or not writable" >&2
  exit 2
fi

ROADNET="$JINAN_DATA_DIR/roadnet_3_4.json"
REAL_FLOW="$JINAN_DATA_DIR/anon_3_4_jinan_real.json"
REAL_2000_FLOW="$JINAN_DATA_DIR/anon_3_4_jinan_real_2000.json"
REAL_2500_FLOW="$JINAN_DATA_DIR/anon_3_4_jinan_real_2500.json"
SYNTHETIC_60MIN_FLOW="$JINAN_DATA_DIR/anon_3_4_jinan_synthetic_24000_60min.json"
SYNTHETIC_24H_FLOW="$JINAN_DATA_DIR/anon_3_4_jinan_synthetic_24h_6000.json"
for path in \
  "$ROADNET" \
  "$REAL_FLOW" \
  "$REAL_2000_FLOW" \
  "$REAL_2500_FLOW" \
  "$SYNTHETIC_60MIN_FLOW" \
  "$SYNTHETIC_24H_FLOW"; do
  if [[ ! -f "$path" ]]; then
    echo "required Jinan input does not exist: $path" >&2
    exit 2
  fi
done

mkdir -p \
  "$RUN_DIR/logs" \
  "$RUN_DIR/tmp" \
  "$RUN_DIR/python_cache" \
  "$RUN_DIR/pip_cache"
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
    "jinan_data_dir": "$JINAN_DATA_DIR",
}, indent=2, sort_keys=True))
PY
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu \
  --format=csv,noheader > "$RUN_DIR/gpu_before.txt"

printf '%s\n' "prepare_jinan_flows" > "$RUN_DIR/stage.txt"
"$PYTHON" -m cityflow_tsc.collect_jinan_world_model_data prepare \
  --output "$RUN_DIR/dataset" \
  --roadnet "$ROADNET" \
  --real-flow "$REAL_FLOW" \
  --real-2000-flow "$REAL_2000_FLOW" \
  --real-2500-flow "$REAL_2500_FLOW" \
  --synthetic-60min-flow "$SYNTHETIC_60MIN_FLOW" \
  --synthetic-24h-flow "$SYNTHETIC_24H_FLOW" \
  > "$RUN_DIR/logs/prepare.log" 2>&1

PLAN="$RUN_DIR/dataset/dataset_plan.json"
printf '%s\n' "parallel_trajectory_collection" > "$RUN_DIR/stage.txt"
"$PYTHON" -m cityflow_tsc.collect_jinan_world_model_data collect-rules \
  --plan "$PLAN" \
  --output "$RUN_DIR/trajectories_rules" \
  --workers 8 \
  --thread-num 1 \
  > "$RUN_DIR/logs/rules.log" 2>&1 &
PID_RULES=$!

"$PYTHON" -m cityflow_tsc.collect_jinan_world_model_data collect-dqn \
  --plan "$PLAN" \
  --output "$RUN_DIR/trajectories_dqn" \
  --device cuda \
  --thread-num 1 \
  --training-rounds 10 \
  > "$RUN_DIR/logs/dqn.log" 2>&1 &
PID_DQN=$!

printf '%s\n' "$PID_RULES" > "$RUN_DIR/rules.pid"
printf '%s\n' "$PID_DQN" > "$RUN_DIR/dqn.pid"
set +e
wait "$PID_RULES"
RULES_EXIT=$?
wait "$PID_DQN"
DQN_EXIT=$?
set -e
printf '%s\n' "$RULES_EXIT" > "$RUN_DIR/rules.exit_code.txt"
printf '%s\n' "$DQN_EXIT" > "$RUN_DIR/dqn.exit_code.txt"
if [[ "$RULES_EXIT" -ne 0 || "$DQN_EXIT" -ne 0 ]]; then
  printf '%s\n' "collection_failed" > "$RUN_DIR/stage.txt"
  printf '%s\n' "1" > "$RUN_DIR/exit_code.txt"
  exit 1
fi

printf '%s\n' "validate_and_finalize" > "$RUN_DIR/stage.txt"
"$PYTHON" -m cityflow_tsc.collect_jinan_world_model_data finalize \
  --plan "$PLAN" \
  --rules-index "$RUN_DIR/trajectories_rules/rules_index.json" \
  --dqn-index "$RUN_DIR/trajectories_dqn/dqn_index.json" \
  --output "$RUN_DIR/jinan_dataset.index.json" \
  > "$RUN_DIR/logs/finalize.log" 2>&1

nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu \
  --format=csv,noheader > "$RUN_DIR/gpu_after.txt"
printf '%s\n' "complete" > "$RUN_DIR/stage.txt"
printf '%s\n' "0" > "$RUN_DIR/exit_code.txt"

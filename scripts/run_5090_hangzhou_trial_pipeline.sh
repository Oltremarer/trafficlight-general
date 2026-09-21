#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_DIR HANGZHOU_DATA_DIR TDMPC2_SOURCE_CHECKPOINT" >&2
  exit 2
fi

RUN_DIR=$(realpath "$1")
HANGZHOU_DATA_DIR=$(realpath "$2")
TDMPC2_SOURCE=$(realpath "$3")
SOURCE_DIR=$(pwd)
case "$RUN_DIR" in
  /mnt/pan/*) ;;
  *) echo "RUN_DIR must be under /mnt/pan" >&2; exit 2 ;;
esac
if [[ ! -w /mnt/pan ]]; then
  echo "/mnt/pan is unavailable or not writable" >&2
  exit 2
fi

ROADNET="$HANGZHOU_DATA_DIR/roadnet_4_4.json"
REAL_FLOW="$HANGZHOU_DATA_DIR/anon_4_4_hangzhou_real.json"
REAL_5816_FLOW="$HANGZHOU_DATA_DIR/anon_4_4_hangzhou_real_5816.json"
SYNTHETIC_FLOW="$HANGZHOU_DATA_DIR/anon_4_4_hangzhou_synthetic_24000_60min.json"
for path in "$ROADNET" "$REAL_FLOW" "$REAL_5816_FLOW" "$SYNTHETIC_FLOW" "$TDMPC2_SOURCE"; do
  if [[ ! -f "$path" ]]; then
    echo "required Hangzhou input does not exist: $path" >&2
    exit 2
  fi
done

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/tmp" "$RUN_DIR/python_cache" "$RUN_DIR/pip_cache"
export TMPDIR="$RUN_DIR/tmp"
export PYTHONPYCACHEPREFIX="$RUN_DIR/python_cache"
export PIP_CACHE_DIR="$RUN_DIR/pip_cache"
BASE_PYTHON=${BASE_PYTHON:-/home/chenyuyang/miniconda3/envs/c2t/bin/python}
"$BASE_PYTHON" -m venv --system-site-packages "$RUN_DIR/venv"
PYTHON="$RUN_DIR/venv/bin/python"
"$PYTHON" -m pip install --quiet -e "${SOURCE_DIR}[rl]"
"$PYTHON" - <<PY > "$RUN_DIR/environment.json"
import json, platform, torch, cityflow
print(json.dumps({
  "python": platform.python_version(),
  "torch": torch.__version__,
  "cuda_available": torch.cuda.is_available(),
  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
  "cityflow": cityflow.__file__,
  "source_dir": "$SOURCE_DIR",
  "source_commit": "$(git rev-parse HEAD)",
  "hangzhou_data_dir": "$HANGZHOU_DATA_DIR",
}, indent=2, sort_keys=True))
PY

printf '%s\n' "prepare_hangzhou_flows" > "$RUN_DIR/stage.txt"
"$PYTHON" -m cityflow_tsc.collect_jinan_world_model_data prepare-hangzhou \
  --output "$RUN_DIR/dataset" \
  --roadnet "$ROADNET" \
  --real-flow "$REAL_FLOW" \
  --real-5816-flow "$REAL_5816_FLOW" \
  --synthetic-60min-flow "$SYNTHETIC_FLOW" \
  > "$RUN_DIR/logs/prepare.log" 2>&1
PLAN="$RUN_DIR/dataset/dataset_plan.json"

printf '%s\n' "parallel_trajectory_collection" > "$RUN_DIR/stage.txt"
"$PYTHON" -m cityflow_tsc.collect_jinan_world_model_data collect-rules \
  --plan "$PLAN" --output "$RUN_DIR/trajectories_rules" --workers 3 \
  > "$RUN_DIR/logs/rules.log" 2>&1 &
PID_RULES=$!
"$PYTHON" -m cityflow_tsc.collect_jinan_world_model_data collect-dqn \
  --plan "$PLAN" --output "$RUN_DIR/trajectories_dqn" --device cuda \
  --training-rounds 10 \
  > "$RUN_DIR/logs/dqn.log" 2>&1 &
PID_DQN=$!
printf '%s\n' "$PID_RULES" > "$RUN_DIR/rules.pid"
printf '%s\n' "$PID_DQN" > "$RUN_DIR/dqn.pid"
set +e
wait "$PID_RULES"; RULES_EXIT=$?
wait "$PID_DQN"; DQN_EXIT=$?
set -e
printf '%s\n' "$RULES_EXIT" > "$RUN_DIR/rules.exit_code.txt"
printf '%s\n' "$DQN_EXIT" > "$RUN_DIR/dqn.exit_code.txt"
if [[ "$RULES_EXIT" -ne 0 || "$DQN_EXIT" -ne 0 ]]; then
  printf '%s\n' "collection_failed" > "$RUN_DIR/stage.txt"
  exit 1
fi

printf '%s\n' "validate_and_finalize" > "$RUN_DIR/stage.txt"
"$PYTHON" -m cityflow_tsc.collect_jinan_world_model_data finalize \
  --plan "$PLAN" \
  --rules-index "$RUN_DIR/trajectories_rules/rules_index.json" \
  --dqn-index "$RUN_DIR/trajectories_dqn/dqn_index.json" \
  --output "$RUN_DIR/hangzhou_dataset.index.json" \
  > "$RUN_DIR/logs/finalize.log" 2>&1

bash "$SOURCE_DIR/scripts/run_5090_dataset_finetune.sh" \
  "$RUN_DIR" "$RUN_DIR/hangzhou_dataset.index.json" "$TDMPC2_SOURCE"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_DIR CITY DATA_DIR" >&2
  exit 2
fi

RUN_DIR=$(realpath -m "$1")
CITY=$2
DATA_DIR=$(realpath "$3")
SOURCE_DIR=$(pwd)
RULE_WORKERS=${RULE_WORKERS:-2}
case "$RUN_DIR" in
  /mnt/pan/*) ;;
  *) echo "RUN_DIR must be under /mnt/pan" >&2; exit 2 ;;
esac
if [[ ! -w /mnt/pan ]]; then
  echo "/mnt/pan is unavailable or not writable" >&2
  exit 2
fi
if ! [[ "$RULE_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "RULE_WORKERS must be a positive integer" >&2
  exit 2
fi

case "$CITY" in
  Jinan)
    ROADNET="$DATA_DIR/roadnet_3_4.json"
    FLOW_ARGS=(
      --flow "$DATA_DIR/anon_3_4_jinan_real.json"
      --flow "$DATA_DIR/anon_3_4_jinan_real_2000.json"
      --flow "$DATA_DIR/anon_3_4_jinan_real_2500.json"
      --flow "$DATA_DIR/anon_3_4_jinan_synthetic_24000_60min.json"
      --source-24h-flow "$DATA_DIR/anon_3_4_jinan_synthetic_24h_6000.json"
    )
    INDEX_NAME="jinan_dataset.index.json"
    ;;
  Hangzhou)
    ROADNET="$DATA_DIR/roadnet_4_4.json"
    FLOW_ARGS=(
      --flow "$DATA_DIR/anon_4_4_hangzhou_real.json"
      --flow "$DATA_DIR/anon_4_4_hangzhou_real_5816.json"
      --flow "$DATA_DIR/anon_4_4_hangzhou_synthetic_24000_60min.json"
    )
    INDEX_NAME="hangzhou_dataset.index.json"
    ;;
  NewYork)
    ROADNET="$DATA_DIR/roadnet_28_7.json"
    FLOW_ARGS=(
      --flow "$DATA_DIR/anon_28_7_newyork_real_double.json"
      --flow "$DATA_DIR/anon_28_7_newyork_real_triple.json"
    )
    INDEX_NAME="newyork_dataset.index.json"
    ;;
  *) echo "CITY must be Jinan, Hangzhou, or NewYork" >&2; exit 2 ;;
esac
for PATH_ITEM in "$ROADNET" "${FLOW_ARGS[@]:1}"; do
  if [[ "$PATH_ITEM" == --* ]]; then
    continue
  fi
  if [[ ! -f "$PATH_ITEM" ]]; then
    echo "required input does not exist: $PATH_ITEM" >&2
    exit 2
  fi
done

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/tmp" "$RUN_DIR/python_cache" "$RUN_DIR/pip_cache"
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
import json, platform, torch, cityflow
print(json.dumps({
    "city": "$CITY",
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "cityflow": cityflow.__file__,
    "source_dir": "$SOURCE_DIR",
    "source_commit": "$(git rev-parse HEAD)",
    "data_dir": "$DATA_DIR",
}, indent=2, sort_keys=True))
PY

printf '%s\n' "prepare_136_diverse_flows" > "$RUN_DIR/stage.txt"
"$PYTHON" -m cityflow_tsc.collect_jinan_world_model_data prepare-diverse \
  --output "$RUN_DIR/dataset" \
  --city "$CITY" \
  --roadnet "$ROADNET" \
"${FLOW_ARGS[@]}" \
  > "$RUN_DIR/logs/prepare.log" 2>&1
PLAN="$RUN_DIR/dataset/dataset_plan.json"

printf '%s\n' "parallel_2000_trajectory_collection" > "$RUN_DIR/stage.txt"
"$PYTHON" -m cityflow_tsc.collect_jinan_world_model_data collect-rules \
  --plan "$PLAN" --output "$RUN_DIR/trajectories_rules" --workers "$RULE_WORKERS" --thread-num 1 \
  > "$RUN_DIR/logs/rules.log" 2>&1 &
RULES_PID=$!
"$PYTHON" -m cityflow_tsc.collect_jinan_world_model_data collect-dqn \
  --plan "$PLAN" --output "$RUN_DIR/trajectories_dqn" --device cuda --thread-num 1 \
  --training-rounds 8 > "$RUN_DIR/logs/dqn.log" 2>&1 &
DQN_PID=$!
printf '%s\n' "$RULES_PID" > "$RUN_DIR/rules.pid"
printf '%s\n' "$DQN_PID" > "$RUN_DIR/dqn.pid"

set +e
wait "$RULES_PID"; RULES_EXIT=$?
wait "$DQN_PID"; DQN_EXIT=$?
set -e
printf '%s\n' "$RULES_EXIT" > "$RUN_DIR/rules.exit_code.txt"
printf '%s\n' "$DQN_EXIT" > "$RUN_DIR/dqn.exit_code.txt"
if [[ "$RULES_EXIT" -ne 0 || "$DQN_EXIT" -ne 0 ]]; then
  printf '%s\n' "collection_failed" > "$RUN_DIR/stage.txt"
  exit 1
fi

printf '%s\n' "validate_and_finalize_collection" > "$RUN_DIR/stage.txt"
"$PYTHON" -m cityflow_tsc.collect_jinan_world_model_data finalize \
  --plan "$PLAN" \
  --rules-index "$RUN_DIR/trajectories_rules/rules_index.json" \
  --dqn-index "$RUN_DIR/trajectories_dqn/dqn_index.json" \
  --output "$RUN_DIR/$INDEX_NAME" \
  > "$RUN_DIR/logs/finalize.log" 2>&1
printf '%s\n' "collection_complete_no_finetune" > "$RUN_DIR/stage.txt"

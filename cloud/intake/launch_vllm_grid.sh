#!/usr/bin/env bash
# Launch 8 vLLM instances — one per GPU — same FP8 Qwen3-VL-8B model.
# Ports 8011..8018 map to GPUs 0..7. Each instance is tp=1, independent.
#
# Usage: bash launch_vllm_grid.sh [--dry-run]
#
# Writes log/pid to /tmp/vllm_grid/<port>.log / .pid. Exits as soon as the
# launcher scripts are fired (does NOT wait for the instances to be ready —
# check /v1/models separately, boot is ~2-3 min each).

set -euo pipefail

ROOT_DIR="/home/mambauser/tangxuan/online_vllm"
START_SCRIPT="$ROOT_DIR/test/start_online_prefill_server.sh"
MODEL="${MODEL:-/home/mambauser/tangxuan/models/Qwen3-VL-8B-Instruct-FP8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
BASE_PORT="${BASE_PORT:-8011}"
BATCH_SIZE="${BATCH_SIZE:-4}"
BATCH_DELAY_S="${BATCH_DELAY_S:-30}"

DRY=""
if [[ "${1:-}" == "--dry-run" ]]; then DRY=1; fi

mkdir -p /tmp/vllm_grid

for gpu in 0 1 2 3 4 5 6 7; do
  port=$((BASE_PORT + gpu))
  log="/tmp/vllm_grid/${port}.log"
  pid_file="/tmp/vllm_grid/${port}.pid"
  echo "[launch] GPU ${gpu} -> port ${port}  log=${log}"
  if [[ -n "$DRY" ]]; then continue; fi
  rm -f "$log" "$pid_file"
  (
    cd "$ROOT_DIR"
    setsid env \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      MODEL="${MODEL}" \
      HOST="127.0.0.1" \
      PORT="${port}" \
      TP_SIZE="1" \
      GPU_MEM_UTIL="${GPU_MEM_UTIL}" \
      MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
      bash "${START_SCRIPT}" \
        --no-enable-prefix-caching \
        > "${log}" 2>&1 </dev/null &
    echo $! > "${pid_file}"
  )
  if (( (gpu + 1) % BATCH_SIZE == 0 && gpu < 7 )); then
    echo "[launch] batch boundary reached, sleeping ${BATCH_DELAY_S}s"
    sleep "${BATCH_DELAY_S}"
  fi
done

echo "[launch] 8 instances fired. Watch /tmp/vllm_grid/<port>.log for progress."

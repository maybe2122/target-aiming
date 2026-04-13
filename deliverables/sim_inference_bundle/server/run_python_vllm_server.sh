#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
MODEL_PATH="${1:-outputs/eyevla_v8_r2_merged_fp16}"
PORT="${2:-8097}"
HOST="${HOST:-127.0.0.1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
VLLM_BIN="${VLLM_BIN:-vllm}"

cd "$ROOT_DIR"

ARGS=(
  serve
  "$MODEL_PATH"
  --host "$HOST"
  --port "$PORT"
  --trust-remote-code
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
)

if [[ -f "$MODEL_PATH/chat_template.jinja" ]]; then
  ARGS+=(--chat-template "$MODEL_PATH/chat_template.jinja")
fi

"$VLLM_BIN" "${ARGS[@]}"

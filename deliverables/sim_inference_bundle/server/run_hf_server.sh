#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
MODEL_PATH="${1:-outputs/eyevla_v8_merged_fp16_r2}"
PORT="${2:-8000}"

cd "$ROOT_DIR"
uv run python deliverables/sim_inference_bundle/server/serve.py \
  --backend hf \
  --model "$MODEL_PATH" \
  --port "$PORT"

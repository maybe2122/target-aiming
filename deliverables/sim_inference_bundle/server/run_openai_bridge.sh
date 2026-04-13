#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
OPENAI_BASE_URL="${1:-http://127.0.0.1:8097/v1}"
PORT="${2:-8000}"

cd "$ROOT_DIR"
uv run python deliverables/sim_inference_bundle/server/serve.py \
  --backend openai \
  --openai-base-url "$OPENAI_BASE_URL" \
  --port "$PORT"

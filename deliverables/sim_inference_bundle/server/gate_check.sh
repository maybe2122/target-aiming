#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
URL="${1:-http://127.0.0.1:8000/predict}"
IMAGE="${2:-}"
TYPE="${3:-grounding_action}"

if [ -z "$IMAGE" ]; then
  echo "usage: bash deliverables/sim_inference_bundle/server/gate_check.sh <url> <image> [type]" >&2
  exit 1
fi

cd "$ROOT_DIR"
python scripts/gate_action_tokens.py \
  --url "$URL" \
  --image "$IMAGE" \
  --type "$TYPE" \
  -n 20

#!/usr/bin/env bash
# Start the model server for one Qwen-Instruct size.
#
# Usage:   ./start_serving.sh {7b|14b|32b}
# Ports:   7b→8000, 14b→8001, 32b→8002
#
# Uses the transformers-based OpenAI shim that's already on disk
# (langgraph_eval/transformers_openai_server.py) and the AstroNet venv
# where it was installed. Drop-in replacement for the old base-model
# launch — same ports, same served-name, but now pointing at the
# -Instruct weights so the chat template + instruction tuning are
# actually applied and tool calling works.
set -euo pipefail

case "${1:-}" in
  7b)  SIZE=7b;  PORT=8000;;
  14b) SIZE=14b; PORT=8001;;
  32b) SIZE=32b; PORT=8002;;
  *) echo "usage: $0 {7b|14b|32b}"; exit 2;;
esac

MODEL_DIR="/media/alexander/PortableSSD/AstroNet/models/qwen2.5-${SIZE}-instruct"
[ -d "$MODEL_DIR" ] || { echo "model dir missing: $MODEL_DIR"; exit 1; }
[ -f "$MODEL_DIR/tokenizer_config.json" ] || {
  echo "tokenizer_config.json missing in $MODEL_DIR (download incomplete?)"
  exit 1
}

REPO=/home/alexander/Schreibtisch/open-fem-agent
PYBIN=/home/alexander/Schreibtisch/AstroNet/venv/bin/python
export HF_HOME="${HF_HOME:-/media/alexander/PortableSSD/.cache/huggingface}"

exec "$PYBIN" "$REPO/langgraph_eval/transformers_openai_server.py" \
    --model "$MODEL_DIR" \
    --served-name "qwen2.5-${SIZE}" \
    --port "$PORT"

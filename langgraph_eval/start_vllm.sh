#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server for one Qwen2.5 size.
#
# Usage:   ./start_vllm.sh {7b|14b|32b}
# Ports:   7b→8000, 14b→8001, 32b→8002
set -euo pipefail

case "${1:-}" in
  7b)  SIZE=7b;  PORT=8000; EXTRA="";;
  14b) SIZE=14b; PORT=8001; EXTRA="";;
  32b) SIZE=32b; PORT=8002; EXTRA="--quantization awq --gpu-memory-utilization 0.92";;
  *) echo "usage: $0 {7b|14b|32b}"; exit 2;;
esac

MODEL="/media/alexander/PortableSSD/AstroNet/models/qwen2.5-${SIZE}"
[ -d "$MODEL" ] || { echo "model dir missing: $MODEL"; exit 1; }

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "qwen2.5-${SIZE}" \
    --port "${PORT}" \
    --max-model-len 16384 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    ${EXTRA}

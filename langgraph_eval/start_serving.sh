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

# Machine-local paths — override via env; defaults match the reference machine.
MODELS_ROOT="${MODELS_ROOT:-/media/$USER/PortableSSD/AstroNet/models}"
MODEL_DIR="${MODEL_DIR:-$MODELS_ROOT/qwen2.5-${SIZE}-instruct}"
[ -d "$MODEL_DIR" ] || { echo "model dir missing: $MODEL_DIR"; exit 1; }
[ -f "$MODEL_DIR/tokenizer_config.json" ] || {
  echo "tokenizer_config.json missing in $MODEL_DIR (download incomplete?)"
  exit 1
}

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYBIN="${PYBIN:-$HOME/Schreibtisch/AstroNet/venv/bin/python}"
export HF_HOME="${HF_HOME:-/media/$USER/PortableSSD/.cache/huggingface}"

# nvidia-smi index 0 is a 2 GB GT 1030; indices 1 and 2 are 24 GB TITAN
# RTX cards. WITHOUT CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA reorders by
# performance internally and CUDA_VISIBLE_DEVICES selects from THAT
# list — so 'CUDA_VISIBLE_DEVICES=2' silently picked the slowest card.
# Forcing PCI_BUS_ID order makes the indices match nvidia-smi.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
case "$SIZE" in
  7b|14b) export CUDA_VISIBLE_DEVICES=1 ;;   # single 24 GB TITAN
  32b)    export CUDA_VISIBLE_DEVICES=1,2 ;; # both 24 GB TITANs
esac

exec "$PYBIN" "$REPO/langgraph_eval/transformers_openai_server.py" \
    --model "$MODEL_DIR" \
    --served-name "qwen2.5-${SIZE}" \
    --port "$PORT"

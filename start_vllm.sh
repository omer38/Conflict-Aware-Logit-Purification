#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$ROOT/vllm.log"
SERVER_DIR="$ROOT/vllm-server"

cd "$SERVER_DIR"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

echo "Syncing vLLM server dependencies (Python 3.12)..."
uv sync --python 3.12 --quiet
uv run --python 3.12 python patch_mistral_pixtral.py

VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.5-27B}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_LOGPROBS="${VLLM_MAX_LOGPROBS:-262144}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"

args=(
  vllm serve "$VLLM_MODEL"
  --max-model-len "$VLLM_MAX_MODEL_LEN"
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$VLLM_MAX_NUM_SEQS"
  --max-logprobs "$VLLM_MAX_LOGPROBS"
  --host 0.0.0.0 --port 8000
  --api-key "${VLLM_API_KEY:-mysecretkey}"
)

if [[ "${VLLM_ENABLE_AUTO_TOOL_CHOICE:-0}" == "1" ]]; then
  args+=(--enable-auto-tool-choice)
fi
if [[ -n "${VLLM_REASONING_PARSER:-}" ]]; then
  args+=(--reasoning-parser "$VLLM_REASONING_PARSER")
fi
if [[ -n "${VLLM_TOOL_CALL_PARSER:-}" ]]; then
  args+=(--tool-call-parser "$VLLM_TOOL_CALL_PARSER")
fi
if [[ "$VLLM_MODEL" == mistralai/* ]]; then
  args+=(
    --tokenizer-mode mistral
    --config-format mistral
    --load-format mistral
    --limit-mm-per-prompt '{"image":10}'
  )
  if [[ -z "${VLLM_TOOL_CALL_PARSER:-}" ]]; then
    args+=(--tool-call-parser mistral --enable-auto-tool-choice)
  fi
fi

echo "Serving model: $VLLM_MODEL"
nohup uv run --python 3.12 "${args[@]}" > "$LOG" 2>&1 &

echo "vLLM started (PID $!). Logs: $LOG"
echo "Follow with: tail -f $LOG"

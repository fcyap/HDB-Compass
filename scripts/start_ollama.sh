#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime="$project_root/data/local/ollama-runtime/bin/ollama"
if [[ ! -x "$runtime" ]]; then
  runtime="$(command -v ollama)" || {
    echo "Install Ollama before starting the local model service." >&2
    exit 1
  }
fi

export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS="$project_root/data/local/ollama-models"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-16384}"
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NO_CLOUD=1
export OLLAMA_GO_TEMPLATE=0

exec "$runtime" serve

#!/usr/bin/env bash
#SBATCH --job-name=hdb-vllm
#SBATCH --partition=student
#SBATCH --account=is469
#SBATCH --qos=studentqos
#SBATCH --gres=gpu:1                                # studentqos max: 1 GPU per job
#SBATCH --constraint=rtx_pro_6000_blackwell_max     # a 24 GB slice: each 96 GB card is split in four
#SBATCH --cpus-per-task=4                           # studentqos max
#SBATCH --mem=32G                                   # studentqos max
#SBATCH --time=06:00:00                             # partition max is 1 day; cancel when done
#SBATCH --output=%x-%j.out
# ---------------------------------------------------------------------------
# Serve the benchmark model with vLLM (OpenAI-compatible) on one GPU. Member 4.
#   sbatch ~/hdb-cluster/serve_vllm.sbatch          then on the laptop: bash cluster/tunnel.sh
#   scancel <jobid>                                  when the run is finished (frees the GPU)
# Default model fits the 24 GB slice with the full 40,960-token context (16.9 GB weights).
# When a 48 GB A40/L40S is free, the dense 32B in 4-bit is the stronger option:
#   sbatch --constraint='a40|l40s' --export=ALL,MODEL_REPO=Qwen/Qwen3-32B-AWQ ~/hdb-cluster/serve_vllm.sbatch
#   (download that repo first with setup_env.sbatch and the same MODEL_REPO)
# Qwen3-32B in bf16 (62 GB) needs an 80 GB+ GPU, which the student QoS cannot request.
# Security: the port is reachable by other cluster users, so every request needs
# the API key in ~/.hdb_vllm_key (passed by env, never on the command line).
# ---------------------------------------------------------------------------
set -euo pipefail
SCRATCH="/common/scratch/users/${USER:0:1}/$USER"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-30B-A3B-GPTQ-Int4}"
MODEL_DIR="${MODEL_DIR:-$SCRATCH/models/${MODEL_REPO##*/}}"
PARSER="${PARSER:-hermes}"             # Qwen3 tool calls -> hermes parser
MAX_LEN="${MAX_LEN:-40960}"            # Qwen3 native context (32k + 8k generation)
export HF_HOME="$SCRATCH/hf" HF_HUB_OFFLINE=1
# The nodes have no nvcc, and FlashInfer's top-k/top-p sampler JIT-compiles on first use, which
# kills start-up. Our requests are greedy (temperature 0), so vLLM's PyTorch sampler is equivalent.
export VLLM_USE_FLASHINFER_SAMPLER=0

KEY_FILE="$HOME/.hdb_vllm_key"
if [ ! -s "$KEY_FILE" ]; then (umask 077; python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > "$KEY_FILE"); fi
export VLLM_API_KEY="$(cat "$KEY_FILE")"

# pick a free port derived from the job id (other users share this node)
PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
while (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; do PORT=$((PORT + 1)); done
echo "$(hostname) $PORT $SLURM_JOB_ID" > "$HOME/vllm_endpoint.txt"

module purge
module load Python/3.12.8-GCCcore-13.3.0
source "$HOME/hdb-venv/bin/activate"
echo "== node $(hostname)  port $PORT  model $MODEL_REPO from $MODEL_DIR"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import torch, vllm; print('torch', torch.__version__, '| torch CUDA', torch.version.cuda, '| vllm', vllm.__version__, '| GPU visible', torch.cuda.is_available())"

exec vllm serve "$MODEL_DIR" \
  --served-model-name "$MODEL_REPO" \
  --host 0.0.0.0 --port "$PORT" \
  --seed 42 \
  --max-model-len "$MAX_LEN" \
  --enable-auto-tool-choice --tool-call-parser "$PARSER" \
  --generation-config vllm \
  --gpu-memory-utilization 0.92

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
PARSER="${PARSER:-hermes}"             # Qwen3 tool calls -> hermes parser
MAX_LEN="${MAX_LEN:-32768}"            # 40960 needs 3.75 GiB of KV cache; the 24 GB slice has 3.3 (vLLM 0.33)
GPU_UTIL="${GPU_UTIL:-0.95}"
REASONING_PARSER="${REASONING_PARSER:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# MODEL_PRESET selects the flags used for the reported Experiment 1 runs (README "Setup used for these runs"); with no
# preset the defaults above (Qwen3-30B-A3B on the 24 GB slice) apply. Submit with the GPU named in the comment, e.g.
#   sbatch --constraint='a40|l40s' --export=ALL,MODEL_PRESET=qwen38 cluster/serve_vllm.sbatch
# (download the model first with setup_env.sbatch and the same MODEL_REPO). A preset sets every variable it names, so to
# change a single flag, submit without MODEL_PRESET and pass MODEL_REPO, PARSER, MAX_LEN, ... yourself.
case "${MODEL_PRESET:-}" in
  "") ;;
  qwen3-30b) ;;                                                    # defaults; --constraint=rtx_pro_6000_blackwell_max
  qwen38)    MODEL_REPO=Qwen/Qwen3.8-27B-FP8; PARSER=qwen3_coder; REASONING_PARSER=qwen3; MAX_LEN=65536
             GPU_UTIL=0.92; EXTRA_ARGS="--max-num-seqs 32" ;;      # --constraint='a40|l40s'
  gptoss)    MODEL_REPO=openai/gpt-oss-20b; PARSER=openai; REASONING_PARSER=openai_gptoss; MAX_LEN=65536
             GPU_UTIL=0.92; EXTRA_ARGS="--attention-backend TRITON_ATTN" ;;   # --constraint=rtx_pro_6000_blackwell_max
  gemma4)    MODEL_REPO=nvidia/Gemma-4-26B-A4B-NVFP4; PARSER=gemma4; REASONING_PARSER=gemma4; MAX_LEN=65536
             GPU_UTIL=0.92; EXTRA_ARGS="--max-num-seqs 16" ;;      # --constraint='a40|l40s' (the 24 GB slice is too small)
  *) echo "unknown MODEL_PRESET=$MODEL_PRESET (qwen3-30b, qwen38, gptoss, gemma4)" >&2; exit 2 ;;
esac
MODEL_DIR="${MODEL_DIR:-$SCRATCH/models/${MODEL_REPO##*/}}"
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
  ${REASONING_PARSER:+--reasoning-parser "$REASONING_PARSER"} \
  --generation-config vllm \
  --gpu-memory-utilization "$GPU_UTIL" $EXTRA_ARGS

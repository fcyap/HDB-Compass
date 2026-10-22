#!/usr/bin/env bash
#SBATCH --job-name=hdb-setup
#SBATCH --partition=student
#SBATCH --account=is469
#SBATCH --qos=studentqos
#SBATCH --cpus-per-task=4          # studentqos max
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# ---------------------------------------------------------------------------
# One-off CPU-only setup on the SMU Origami cluster (no GPU held). Member 4.
#   - venv with vLLM in $HOME/hdb-venv  (~8 GB, home quota is 120 GB)
#   - model weights in scratch (default Qwen3-30B-A3B GPTQ-Int4, 16.9 GB; scratch purges files older than 30 days)
# Submit from the login node:   sbatch ~/hdb-cluster/setup_env.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail
export SCRATCH="/common/scratch/users/${USER:0:1}/$USER"
export HF_HOME="$SCRATCH/hf"
export MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-30B-A3B-GPTQ-Int4}"
export MODEL_DIR="${MODEL_DIR:-$SCRATCH/models/${MODEL_REPO##*/}}"

module purge
module load Python/3.12.8-GCCcore-13.3.0
[ -d "$HOME/hdb-venv" ] || python3 -m venv "$HOME/hdb-venv"
source "$HOME/hdb-venv/bin/activate"
pip install --no-cache-dir -q -U pip
pip install --no-cache-dir -q vllm huggingface_hub
python -c "import torch, vllm; print('torch', torch.__version__, '| torch CUDA', torch.version.cuda, '| vllm', vllm.__version__)"

mkdir -p "$MODEL_DIR"
python - <<'PY'
import os, time
from huggingface_hub import snapshot_download
t0 = time.time()
path = snapshot_download(os.environ["MODEL_REPO"], local_dir=os.environ["MODEL_DIR"],
                         allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "merges.txt", "vocab.json"])
print(f"downloaded to {path} in {time.time() - t0:.0f}s")
PY
du -sh "$MODEL_DIR"
du -sh "$HOME/hdb-venv"
echo SETUP_DONE

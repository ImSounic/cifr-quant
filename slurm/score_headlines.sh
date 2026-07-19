#!/bin/bash
#SBATCH --job-name=score-news
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:lovelace:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/score_news_%j.log

# Phase 3.3 headline scoring on an L40S. Prereqs (one-time, HEAD node — it has
# internet, compute nodes do not):
#   1. user-space ollama:  curl -L https://ollama.com/download/ollama-linux-amd64.tgz \
#        -o ~/ollama.tgz && mkdir -p ~/ollama && tar -xzf ~/ollama.tgz -C ~/ollama
#   2. cache the model (server briefly on the head node, download-only):
#        OLLAMA_HOST=127.0.0.1:11500 ~/ollama/bin/ollama serve >/dev/null 2>&1 &
#        sleep 8
#        OLLAMA_HOST=127.0.0.1:11500 ~/ollama/bin/ollama pull qwen2.5:7b
#        kill %1
#   3. corpus present at data/raw/news/ (scp from laptop)
# Submit:  sbatch slurm/score_headlines.sh

set -e
cd ~/cifr-quant

export OLLAMA_HOST=127.0.0.1:11500
~/ollama/bin/ollama serve > slurm/logs/ollama_serve_${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!
sleep 12

source $(conda info --base)/etc/profile.d/conda.sh && conda activate trade
OLLAMA_URL=http://127.0.0.1:11500/api/chat python scripts/score_headlines.py

kill $OLLAMA_PID || true

#!/bin/bash
#SBATCH --job-name=cifr-finetune
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/finetune_%j.out
#SBATCH --error=slurm/logs/finetune_%j.err

# Kronos finetuning — requires 1 GPU (L40S preferred)
# Usage: sbatch slurm/finetune.sh

cd ~/cifr-quant
source ~/.conda/envs/trade/bin/activate trade

export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/Kronos:$(pwd)/Kronos/finetune_csv"

echo "============================================"
echo "CIFR-QUANT Finetuning on $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

nvidia-smi

# ─── BTC/USDT ───
echo ""
echo ">>> [1/3] Finetuning BTC (tokenizer + predictor)..."
cd ~/cifr-quant/Kronos/finetune_csv
python train_sequential.py --config ../../finetune/config_btc.yaml
cd ~/cifr-quant

# ─── EUR/USD ───
echo ""
echo ">>> [2/3] Finetuning EUR (tokenizer + predictor)..."
cd ~/cifr-quant/Kronos/finetune_csv
python train_sequential.py --config ../../finetune/config_eur.yaml
cd ~/cifr-quant

# ─── XAU/USD ───
echo ""
echo ">>> [3/3] Finetuning XAU (tokenizer + predictor)..."
cd ~/cifr-quant/Kronos/finetune_csv
python train_sequential.py --config ../../finetune/config_xau.yaml
cd ~/cifr-quant

echo ""
echo "============================================"
echo "All finetuning complete!"
echo "Checkpoints saved to:"
ls -la checkpoints/*/tokenizer/best_model/ 2>/dev/null
ls -la checkpoints/*/predictor/best_model/ 2>/dev/null
echo "============================================"

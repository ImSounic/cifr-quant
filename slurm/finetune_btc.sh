#!/bin/bash
#SBATCH --job-name=cifr-btc
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/finetune_btc_%j.out
#SBATCH --error=slurm/logs/finetune_btc_%j.err

cd ~/cifr-quant
source ~/.conda/envs/trade/bin/activate trade
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/Kronos:$(pwd)/Kronos/finetune_csv"

echo "=== BTC/USDT Finetuning on $(hostname) ==="
nvidia-smi

cd Kronos/finetune_csv
python train_sequential.py --config ../../finetune/config_btc.yaml

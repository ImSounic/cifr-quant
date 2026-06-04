#!/bin/bash
#SBATCH --job-name=cifr-xau
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/finetune_xau_%j.out
#SBATCH --error=slurm/logs/finetune_xau_%j.err

cd ~/cifr-quant
source ~/.conda/envs/trade/bin/activate trade
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/Kronos:$(pwd)/Kronos/finetune_csv"

echo "=== XAU/USD Finetuning on $(hostname) ==="
nvidia-smi

cd Kronos/finetune_csv
python train_sequential.py --config ../../finetune/config_xau.yaml

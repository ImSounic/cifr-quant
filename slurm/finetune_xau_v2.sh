#!/bin/bash
#SBATCH --job-name=cifr-xau-v2
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:lovelace:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/finetune_xau_v2_%j.out
#SBATCH --error=slurm/logs/finetune_xau_v2_%j.err

cd ~/cifr-quant
source $(conda info --base)/etc/profile.d/conda.sh && conda activate trade
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/Kronos:$(pwd)/Kronos/finetune_csv"
export HF_HUB_OFFLINE=1

export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

echo "=== XAU/USD v2 Finetuning (anti-overfitting) on $(hostname) ==="
nvidia-smi

cd Kronos/finetune_csv
python train_sequential.py --config ../../finetune/config_xau_v2.yaml

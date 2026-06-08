#!/bin/bash
#SBATCH --job-name=cifr-cqr
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:lovelace:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=/home/s3702111/cifr-quant/slurm/logs/cqr_calibrate_%j.out
#SBATCH --error=/home/s3702111/cifr-quant/slurm/logs/cqr_calibrate_%j.err

cd /home/s3702111/cifr-quant
source $(conda info --base)/etc/profile.d/conda.sh && conda activate trade
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/Kronos:$(pwd)/Kronos/finetune_csv"
export HF_HUB_OFFLINE=1

# Maximize GPU throughput
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export OMP_NUM_THREADS=32

# Force unbuffered output — critical for live tailing
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

echo "=== Multi-Asset CQR Calibration on $(hostname) ==="
nvidia-smi

# 50 MC paths for better quantile estimates, L40S can handle it
# stdout and stderr both go to .out for easy tailing
python -u scripts/calibrate_cqr_multi.py \
    --market all \
    --n-paths 50 \
    --coverage 0.90 \
    2>&1

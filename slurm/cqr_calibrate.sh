#!/bin/bash
#SBATCH --job-name=cifr-cqr
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/s3702111/cifr-quant/slurm/logs/cqr_calibrate_%j.out
#SBATCH --error=/home/s3702111/cifr-quant/slurm/logs/cqr_calibrate_%j.err

cd /home/s3702111/cifr-quant
source $(conda info --base)/etc/profile.d/conda.sh && conda activate trade
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/Kronos:$(pwd)/Kronos/finetune_csv"
export HF_HUB_OFFLINE=1

echo "=== Multi-Asset CQR Calibration on $(hostname) ==="
nvidia-smi

# Unbuffered output so we can tail logs
python -u scripts/calibrate_cqr_multi.py \
    --market all \
    --n-paths 30 \
    --coverage 0.90

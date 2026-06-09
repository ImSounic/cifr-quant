#!/bin/bash
#SBATCH --job-name=cifr-fcache
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:lovelace:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Build the Kronos forecast cache (GPU). Runs the same ensemble CQR was
# calibrated on over the walk-forward grid and dumps one row per
# (symbol, rebalance_t) to results/forecasts/. After this, strategy A/B runs
# CPU-only on the laptop with no GPU and no queue wait.

mkdir -p /home/s3702111/cifr-quant/slurm/logs
LOGFILE="/home/s3702111/cifr-quant/slurm/logs/build_forecast_cache_${SLURM_JOB_ID}.log"

cd /home/s3702111/cifr-quant
source $(conda info --base)/etc/profile.d/conda.sh && conda activate trade
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/Kronos:$(pwd)/Kronos/finetune_csv"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export OMP_NUM_THREADS=8

{
    echo "=== Build Forecast Cache on $(hostname) ==="
    echo "Job ID: $SLURM_JOB_ID"
    echo "Started: $(date)"
    echo ""
    nvidia-smi
    echo ""

    python -u scripts/build_forecast_cache.py \
        --market all \
        --n-paths 30

    echo ""
    echo "Finished: $(date)"
} 2>&1 | tee "$LOGFILE"

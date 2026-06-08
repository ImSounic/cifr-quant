#!/bin/bash
#SBATCH --job-name=cifr-cqr-crypto
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:lovelace:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Crypto CQR is heavy: pred_len=48, 3-month cal window ~8640 candles.
# Mitigations vs the original timeout (job 510442):
#   - n_paths 50 -> 30 (still fine for q05/q95 bands)
#   - step-size 48 -> 96 (~89 non-overlapping windows/asset; plenty for CQR)
#   - results saved incrementally per asset; re-submitting RESUMES (skips
#     assets already in results/cqr/cqr_calibrations.json). If this hits the
#     12h wall before finishing all 15 assets, just submit it again.

mkdir -p /home/s3702111/cifr-quant/slurm/logs
LOGFILE="/home/s3702111/cifr-quant/slurm/logs/cqr_crypto_${SLURM_JOB_ID}.log"

cd /home/s3702111/cifr-quant
source $(conda info --base)/etc/profile.d/conda.sh && conda activate trade
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/Kronos:$(pwd)/Kronos/finetune_csv"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export OMP_NUM_THREADS=32

{
    echo "=== Crypto CQR Calibration on $(hostname) ==="
    echo "Job ID: $SLURM_JOB_ID"
    echo "Started: $(date)"
    echo ""
    nvidia-smi
    echo ""

    python -u scripts/calibrate_cqr_multi.py \
        --market crypto \
        --n-paths 30 \
        --step-size 96 \
        --coverage 0.90

    echo ""
    echo "Finished: $(date)"
} 2>&1 | tee "$LOGFILE"

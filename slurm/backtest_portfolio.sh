#!/bin/bash
#SBATCH --job-name=cifr-backtest
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:lovelace:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Multi-asset dual-market walk-forward portfolio backtest.
# Uses the SAME ensemble CQR was calibrated on (zero-shot + 2 finetunes/market)
# and the per-asset CQR corrections from results/cqr/cqr_calibrations.json.
# Joint risk-parity per market on the last 90 days, long & short, combined
# into a single daily top-line. Batched MC paths keep this well inside wall.

mkdir -p /home/s3702111/cifr-quant/slurm/logs
LOGFILE="/home/s3702111/cifr-quant/slurm/logs/backtest_portfolio_${SLURM_JOB_ID}.log"

cd /home/s3702111/cifr-quant
source $(conda info --base)/etc/profile.d/conda.sh && conda activate trade
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/Kronos:$(pwd)/Kronos/finetune_csv"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export OMP_NUM_THREADS=8

{
    echo "=== Portfolio Backtest on $(hostname) ==="
    echo "Job ID: $SLURM_JOB_ID"
    echo "Started: $(date)"
    echo ""
    nvidia-smi
    echo ""

    python -u scripts/backtest_portfolio.py \
        --market all \
        --n-paths 30 \
        --capital 100000

    echo ""
    echo "Finished: $(date)"
} 2>&1 | tee "$LOGFILE"

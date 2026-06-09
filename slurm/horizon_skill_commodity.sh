#!/bin/bash
#SBATCH --job-name=cifr-hskc
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:lovelace:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Commodity horizon skill with a PROPER sample (stride 6 on 4h bars -> ~64
# windows/asset, n~300+ per horizon). The earlier all-market run used stride 96
# which gave only n=24 for commodity (useless). GPU because CPU was too slow.

mkdir -p /home/s3702111/cifr-quant/slurm/logs
LOGFILE="/home/s3702111/cifr-quant/slurm/logs/horizon_skill_commodity_${SLURM_JOB_ID}.log"

cd /home/s3702111/cifr-quant
source $(conda info --base)/etc/profile.d/conda.sh && conda activate trade
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/Kronos:$(pwd)/Kronos/finetune_csv"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

{
    echo "=== Commodity Horizon Skill on $(hostname) ==="
    echo "Job ID: $SLURM_JOB_ID  Started: $(date)"
    nvidia-smi; echo ""

    python -u scripts/horizon_skill.py --market commodity --stride 6 --n-paths 20

    echo ""; echo "Finished: $(date)"
} 2>&1 | tee "$LOGFILE"

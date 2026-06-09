#!/bin/bash
#SBATCH --job-name=cifr-hskill
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:lovelace:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Forecast skill vs HORIZON. Re-runs the ensemble over a thinned grid and reports
# IC + directional hit rate at every horizon step. Decides whether the model has
# extractable short-horizon edge that the 48-bar final-step view was hiding.

mkdir -p /home/s3702111/cifr-quant/slurm/logs
LOGFILE="/home/s3702111/cifr-quant/slurm/logs/horizon_skill_${SLURM_JOB_ID}.log"

cd /home/s3702111/cifr-quant
source $(conda info --base)/etc/profile.d/conda.sh && conda activate trade
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/Kronos:$(pwd)/Kronos/finetune_csv"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

{
    echo "=== Horizon Skill on $(hostname) ==="
    echo "Job ID: $SLURM_JOB_ID  Started: $(date)"
    nvidia-smi; echo ""

    python -u scripts/horizon_skill.py --market all --n-paths 20 --stride 96

    echo ""; echo "Finished: $(date)"
} 2>&1 | tee "$LOGFILE"

#!/bin/bash
#SBATCH --job-name=cifr-eval
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/eval_ensemble_%j.out
#SBATCH --error=slurm/logs/eval_ensemble_%j.err

cd ~/cifr-quant
source $(conda info --base)/etc/profile.d/conda.sh && conda activate trade
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/Kronos:$(pwd)/Kronos/finetune_csv"
export HF_HUB_OFFLINE=1

echo "=== Ensemble Evaluation on $(hostname) ==="
echo "Checkpoints available:"
ls -d checkpoints/cifr-*/predictor/best_model 2>/dev/null || echo "None found via symlink"
ls -d Kronos/checkpoints/cifr-*/predictor/best_model 2>/dev/null || echo "None found in Kronos/"
nvidia-smi

python scripts/eval_ensemble.py --markets eur xau --n-windows 50 --n-paths 10

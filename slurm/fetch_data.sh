#!/bin/bash
#SBATCH --job-name=cifr-fetch-data
#SBATCH --partition=main-cpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm/logs/fetch_data_%j.out
#SBATCH --error=slurm/logs/fetch_data_%j.err

# Data fetching — CPU only, needs internet access
# Usage: sbatch slurm/fetch_data.sh

cd ~/cifr-quant
source ~/.conda/envs/trade/bin/activate trade

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "=== Fetching BTC/USDT 15m ==="
python -m src.data.binance_client

echo "=== Fetching EUR/USD 1h ==="
python -m src.data.forex_client

echo "=== Fetching XAU/USD 4h ==="
# Set your TwelveData key here or export before submitting
export TWELVEDATA_API_KEY="${TWELVEDATA_API_KEY}"
python -m src.data.gold_client

echo "=== Preparing data splits ==="
python finetune/prepare_data.py --market btc --input data/raw/btc/btc_usdt_15m.csv
python finetune/prepare_data.py --market eur --input data/raw/eur/eurusd_1h.csv
python finetune/prepare_data.py --market xau --input data/raw/xau/gold_4h.csv

echo "=== Done ==="

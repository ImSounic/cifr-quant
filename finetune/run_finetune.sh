#!/bin/bash
# Run all finetuning jobs sequentially using Kronos's train_sequential.py
# Execute from project root on uni GPU machine.
#
# Usage:
#   bash finetune/run_finetune.sh [NUM_GPUS]
#
# Prerequisites:
#   1. Data prepared (all 3 markets)
#   2. Kronos repo cloned in project root
#   3. Dependencies installed: pip install -r requirements.txt

NUM_GPUS=${1:-1}

echo "============================================"
echo "CIFR-QUANT Finetuning Pipeline"
echo "GPUs: $NUM_GPUS"
echo "============================================"

# Ensure Kronos is accessible
if [ ! -d "Kronos" ]; then
    echo "ERROR: Kronos not found. Clone it first:"
    echo "  git clone https://github.com/shiyu-coder/Kronos.git"
    exit 1
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/Kronos:$(pwd)/Kronos/finetune_csv"

# ─── BTC/USDT ───
echo ""
echo ">>> [1/3] Finetuning BTC (tokenizer + predictor)..."
cd Kronos/finetune_csv
if [ "$NUM_GPUS" -gt 1 ]; then
    DIST_BACKEND=nccl torchrun --standalone --nproc_per_node=$NUM_GPUS \
        train_sequential.py --config ../../finetune/config_btc.yaml
else
    python train_sequential.py --config ../../finetune/config_btc.yaml
fi
cd ../..

# ─── EUR/USD ───
echo ""
echo ">>> [2/3] Finetuning EUR (tokenizer + predictor)..."
cd Kronos/finetune_csv
if [ "$NUM_GPUS" -gt 1 ]; then
    DIST_BACKEND=nccl torchrun --standalone --nproc_per_node=$NUM_GPUS \
        train_sequential.py --config ../../finetune/config_eur.yaml
else
    python train_sequential.py --config ../../finetune/config_eur.yaml
fi
cd ../..

# ─── XAU/USD ───
echo ""
echo ">>> [3/3] Finetuning XAU (tokenizer + predictor)..."
cd Kronos/finetune_csv
if [ "$NUM_GPUS" -gt 1 ]; then
    DIST_BACKEND=nccl torchrun --standalone --nproc_per_node=$NUM_GPUS \
        train_sequential.py --config ../../finetune/config_xau.yaml
else
    python train_sequential.py --config ../../finetune/config_xau.yaml
fi
cd ../..

echo ""
echo "============================================"
echo "All finetuning complete!"
echo "Checkpoints saved to:"
echo "  checkpoints/cifr-btc/tokenizer/best_model/"
echo "  checkpoints/cifr-btc/predictor/best_model/"
echo "  checkpoints/cifr-eur/tokenizer/best_model/"
echo "  checkpoints/cifr-eur/predictor/best_model/"
echo "  checkpoints/cifr-xau/tokenizer/best_model/"
echo "  checkpoints/cifr-xau/predictor/best_model/"
echo "============================================"

"""Finetuning configuration for XAU/USD (Gold) checkpoint."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "xau"
SAVE_PATH = PROJECT_ROOT / "checkpoints" / "ckpt-xau"

PRETRAINED_TOKENIZER_PATH = "NeoQuasar/Kronos-Tokenizer-base"
PRETRAINED_PREDICTOR_PATH = "NeoQuasar/Kronos-base"

TRAIN_DATA_PATH = DATASET_PATH / "train.csv"
VAL_DATA_PATH = DATASET_PATH / "validation.csv"
TOKENIZER_TRAIN_PATH = DATASET_PATH / "tokenizer_train.csv"

TOKENIZER_SAVE_PATH = SAVE_PATH / "tokenizer"
TOKENIZER_EPOCHS = 50        # Most epochs: smallest dataset
TOKENIZER_BATCH_SIZE = 16
TOKENIZER_LR = 5e-5          # Conservative LR

PREDICTOR_SAVE_PATH = SAVE_PATH / "predictor"
PREDICTOR_EPOCHS = 20
PREDICTOR_BATCH_SIZE = 4      # Very small batch for small dataset
PREDICTOR_LR = 2e-5           # Very conservative LR
PREDICTOR_WARMUP_STEPS = 50
MAX_CONTEXT = 512

USE_WANDB = False
WANDB_PROJECT = "cifr-quant-xau"

INSTRUMENT = "XAU/USD"
TIMEFRAME = "4h"

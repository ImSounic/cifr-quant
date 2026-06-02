"""Finetuning configuration for EUR/USD checkpoint."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "eur"
SAVE_PATH = PROJECT_ROOT / "checkpoints" / "ckpt-eur"

PRETRAINED_TOKENIZER_PATH = "NeoQuasar/Kronos-Tokenizer-base"
PRETRAINED_PREDICTOR_PATH = "NeoQuasar/Kronos-base"

TRAIN_DATA_PATH = DATASET_PATH / "train.csv"
VAL_DATA_PATH = DATASET_PATH / "validation.csv"
TOKENIZER_TRAIN_PATH = DATASET_PATH / "tokenizer_train.csv"

TOKENIZER_SAVE_PATH = SAVE_PATH / "tokenizer"
TOKENIZER_EPOCHS = 30       # More epochs: smaller dataset
TOKENIZER_BATCH_SIZE = 32
TOKENIZER_LR = 1e-4

PREDICTOR_SAVE_PATH = SAVE_PATH / "predictor"
PREDICTOR_EPOCHS = 15        # More epochs: smaller dataset
PREDICTOR_BATCH_SIZE = 8     # Smaller batch: less data
PREDICTOR_LR = 3e-5          # Lower LR: prevent overfitting on small data
PREDICTOR_WARMUP_STEPS = 100
MAX_CONTEXT = 512

USE_WANDB = False
WANDB_PROJECT = "cifr-quant-eur"

INSTRUMENT = "EUR/USD"
TIMEFRAME = "1h"

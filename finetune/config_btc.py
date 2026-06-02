"""Finetuning configuration for BTC/USDT checkpoint."""

from pathlib import Path

# ─── Paths ───
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "btc"
SAVE_PATH = PROJECT_ROOT / "checkpoints" / "ckpt-btc"

# ─── Pre-trained model paths (HuggingFace or local) ───
PRETRAINED_TOKENIZER_PATH = "NeoQuasar/Kronos-Tokenizer-base"
PRETRAINED_PREDICTOR_PATH = "NeoQuasar/Kronos-base"

# ─── Data files ───
TRAIN_DATA_PATH = DATASET_PATH / "train.csv"
VAL_DATA_PATH = DATASET_PATH / "validation.csv"
TOKENIZER_TRAIN_PATH = DATASET_PATH / "tokenizer_train.csv"

# ─── Tokenizer finetuning ───
TOKENIZER_SAVE_PATH = SAVE_PATH / "tokenizer"
TOKENIZER_EPOCHS = 20
TOKENIZER_BATCH_SIZE = 64
TOKENIZER_LR = 1e-4

# ─── Predictor finetuning ───
PREDICTOR_SAVE_PATH = SAVE_PATH / "predictor"
PREDICTOR_EPOCHS = 10
PREDICTOR_BATCH_SIZE = 16
PREDICTOR_LR = 5e-5
PREDICTOR_WARMUP_STEPS = 200
MAX_CONTEXT = 512

# ─── Experiment tracking ───
USE_WANDB = False
WANDB_PROJECT = "cifr-quant-btc"

# ─── Market-specific ───
INSTRUMENT = "BTC/USDT"
TIMEFRAME = "15m"

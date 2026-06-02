"""Prepare market data for Kronos CSV-based finetuning.

Kronos finetune_csv expects CSV files with columns:
    timestamps, open, high, low, close, volume, amount

This script converts our processed data into the format
expected by Kronos's finetune_csv pipeline.
"""

import pandas as pd
from pathlib import Path
import sys
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.base_config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.data.preprocessor import preprocess
from src.data.splitter import temporal_split, save_split


def prepare_for_finetuning(
    market: str,
    raw_csv_path: Path,
    output_dir: Path | None = None,
    test_months: int = 3,
    val_months: int = 3,
    cal_months: int = 3,
):
    """
    Full pipeline: load raw CSV → preprocess → split → save.

    Args:
        market: 'btc', 'eur', or 'xau'
        raw_csv_path: Path to raw OHLCV CSV
        output_dir: Where to save processed splits
        test_months: Months for test set
        val_months: Months for validation
        cal_months: Months for CQR calibration
    """
    if output_dir is None:
        output_dir = DATA_PROCESSED_DIR / market

    print(f"=== Preparing {market.upper()} data for finetuning ===")

    # Load raw data
    print(f"Loading {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path, parse_dates=["timestamps"])
    print(f"Loaded {len(df)} candles")

    # Preprocess
    fill_volume = market in ("eur", "xau")  # Forex and gold may lack volume
    df = preprocess(df, z_threshold=5.0, fill_volume=fill_volume)

    # Split
    split = temporal_split(
        df,
        test_months=test_months,
        val_months=val_months,
        cal_months=cal_months,
    )

    # Save
    save_split(split, output_dir)

    # Also save the full train+cal set for tokenizer finetuning
    # (tokenizer can use all non-test data)
    tokenizer_data = pd.concat([split.train, split.calibration]).sort_values("timestamps")
    tok_path = output_dir / "tokenizer_train.csv"
    tokenizer_data.to_csv(tok_path, index=False)
    print(f"Saved tokenizer training data → {tok_path} ({len(tokenizer_data)} rows)")

    print(f"\n=== {market.upper()} data preparation complete ===\n")
    return split


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for Kronos finetuning")
    parser.add_argument("--market", required=True, choices=["btc", "eur", "xau"])
    parser.add_argument("--input", required=True, help="Path to raw OHLCV CSV")
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()

    output = Path(args.output) if args.output else None
    prepare_for_finetuning(args.market, Path(args.input), output)

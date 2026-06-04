"""Data preprocessing: cleaning, validation, and Z-score clipping for Kronos."""

import pandas as pd
import numpy as np
from pathlib import Path


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate OHLCV data integrity.

    Checks:
    - Required columns exist
    - High >= Low for every candle
    - Open and Close are between Low and High
    - No negative prices
    - No zero-price candles (except volume/amount)
    """
    required = ["timestamps", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    n_before = len(df)

    # Drop rows with NaN in price columns
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Ensure high >= low
    invalid_hl = df["high"] < df["low"]
    if invalid_hl.any():
        print(f"  Removing {invalid_hl.sum()} candles where high < low")
        df = df[~invalid_hl]

    # Ensure no negative prices
    neg_mask = (df[["open", "high", "low", "close"]] < 0).any(axis=1)
    if neg_mask.any():
        print(f"  Removing {neg_mask.sum()} candles with negative prices")
        df = df[~neg_mask]

    # Ensure no zero prices
    zero_mask = (df[["open", "high", "low", "close"]] == 0).any(axis=1)
    if zero_mask.any():
        print(f"  Removing {zero_mask.sum()} candles with zero prices")
        df = df[~zero_mask]

    n_after = len(df)
    if n_before != n_after:
        print(f"  Validation removed {n_before - n_after} rows ({n_before} → {n_after})")

    return df.reset_index(drop=True)


def clip_zscore(df: pd.DataFrame, z_threshold: float = 5.0) -> pd.DataFrame:
    """
    Clip extreme values using Z-score, matching Kronos training data preprocessing.

    Kronos training pipeline clips values outside Z-score [-5, 5].
    We apply this to returns rather than raw prices to handle non-stationarity.
    """
    price_cols = ["open", "high", "low", "close"]

    for col in price_cols:
        returns = df[col].pct_change()
        mean = returns.mean()
        std = returns.std()

        if std == 0 or pd.isna(std):
            continue

        z_scores = (returns - mean) / std
        extreme = z_scores.abs() > z_threshold

        if extreme.any():
            print(f"  Clipping {extreme.sum()} extreme values in {col} (Z > {z_threshold})")
            # Replace extreme returns with clipped values, then reconstruct prices
            clipped_returns = returns.clip(
                lower=mean - z_threshold * std,
                upper=mean + z_threshold * std
            )
            # Reconstruct prices from clipped returns
            df.loc[extreme.index[extreme], col] = (
                df[col].shift(1) * (1 + clipped_returns)
            ).loc[extreme.index[extreme]]

    return df


def preprocess(
    df: pd.DataFrame,
    z_threshold: float = 5.0,
    fill_volume: bool = False,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Args:
        df: Raw OHLCV DataFrame
        z_threshold: Z-score clipping threshold (Kronos uses 5.0)
        fill_volume: If True, fill missing volume/amount with 0

    Returns:
        Cleaned DataFrame ready for Kronos
    """
    print(f"Preprocessing {len(df)} candles...")

    # Ensure timestamps are datetime
    df["timestamps"] = pd.to_datetime(df["timestamps"])

    # Sort by time
    df = df.sort_values("timestamps").reset_index(drop=True)

    # Add volume/amount if missing
    if "volume" not in df.columns or fill_volume:
        df["volume"] = 0.0
    if "amount" not in df.columns or fill_volume:
        df["amount"] = 0.0

    # Validate
    df = validate_ohlcv(df)

    # Z-score clip
    df = clip_zscore(df, z_threshold)

    # Final cleanup
    df = df.dropna().reset_index(drop=True)

    print(f"Preprocessing complete: {len(df)} candles remaining")

    return df

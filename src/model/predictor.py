"""Wrapper around KronosPredictor for standardized inference."""

import pandas as pd
import torch
from typing import Optional

from src.model.loader import setup_kronos_path


class CifrPredictor:
    """
    Wrapper around Kronos KronosPredictor for cifr-quant pipeline.

    Handles:
    - Model/tokenizer loading
    - Single predictions
    - Batch predictions across instruments
    - Device management
    """

    def __init__(self, model, tokenizer, max_context: int = 512, device: str = "cpu"):
        setup_kronos_path()
        from model import KronosPredictor

        self.predictor = KronosPredictor(model, tokenizer, max_context=max_context)
        self.max_context = max_context
        self.device = device

    def predict_single(
        self,
        df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        pred_len: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> pd.DataFrame:
        """
        Generate a single deterministic-ish forecast (averaged from 1 sample).

        Args:
            df: Historical OHLCV DataFrame (must have open, high, low, close columns)
            x_timestamp: Timestamps for historical data
            y_timestamp: Timestamps for prediction period
            pred_len: Number of candles to predict
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold

        Returns:
            DataFrame with predicted OHLCV values
        """
        return self.predictor.predict(
            df=df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=temperature,
            top_p=top_p,
            sample_count=1,
        )

    def predict_batch(
        self,
        df_list: list[pd.DataFrame],
        x_timestamp_list: list[pd.Series],
        y_timestamp_list: list[pd.Series],
        pred_len: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> list[pd.DataFrame]:
        """
        Batch prediction across multiple instruments.

        All series must have same lookback and prediction length.
        Leverages GPU parallelism.
        """
        return self.predictor.predict_batch(
            df_list=df_list,
            x_timestamp_list=x_timestamp_list,
            y_timestamp_list=y_timestamp_list,
            pred_len=pred_len,
            T=temperature,
            top_p=top_p,
            sample_count=1,
            verbose=True,
        )

"""Shared walk-forward grid helpers.

Both the forecast-cache builder (GPU) and the backtest engine (CPU) must use the
IDENTICAL rebalance grid and context-location logic, otherwise cached forecasts
won't line up with the engine's rebalance points. This module is the single
source of truth for both.
"""

from __future__ import annotations

from datetime import timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_test_window(asset_dfs: dict, test_days: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Common test window ending at the latest timestamp across all assets."""
    test_end = max(df["timestamps"].max() for df in asset_dfs.values())
    test_start = test_end - timedelta(days=test_days)
    return pd.Timestamp(test_start), pd.Timestamp(test_end)


def compute_rebalance_grid(asset_dfs: dict, pred_len: int, step: int,
                           test_start: pd.Timestamp, test_end: pd.Timestamp
                           ) -> Tuple[List[pd.Timestamp], str]:
    """Reference clock = asset with the most candles in the window; rebalance
    every `step` candles, leaving room for one `pred_len` horizon at the end."""
    symbols = list(asset_dfs.keys())

    def _n_in_window(df):
        return int(((df["timestamps"] >= test_start) & (df["timestamps"] <= test_end)).sum())

    ref_sym = max(symbols, key=lambda s: _n_in_window(asset_dfs[s]))
    ref_df = asset_dfs[ref_sym]
    ref_win = ref_df[(ref_df["timestamps"] >= test_start) &
                     (ref_df["timestamps"] <= test_end)].reset_index(drop=True)
    rebalance_times = [pd.Timestamp(ref_win["timestamps"].iloc[i])
                       for i in range(0, len(ref_win) - pred_len, step)]
    return rebalance_times, ref_sym


def locate_context(df: pd.DataFrame, t: pd.Timestamp, lookback: int, pred_len: int):
    """Return (context_df, future_df) for rebalance time `t`, or (None, None) if
    the asset lacks a full lookback+pred_len window around `t`."""
    ts = df["timestamps"].values
    idx = int(np.searchsorted(ts, np.datetime64(t), side="right") - 1)
    if idx < 0:
        return None, None
    ctx_start = idx - lookback + 1
    if ctx_start < 0 or idx + pred_len >= len(df):
        return None, None
    context = df.iloc[ctx_start:idx + 1]
    future = df.iloc[idx + 1:idx + 1 + pred_len]
    if len(context) < lookback or len(future) < pred_len:
        return None, None
    return context, future

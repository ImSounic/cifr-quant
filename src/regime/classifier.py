"""Hurst + ADX composite regime classifier (signed-off June 9 2026).

Composite rule (v1):
  TREND if Hurst > hurst_trend AND ADX > adx_trend
  RANGE if Hurst < hurst_range OR  ADX < adx_range
  else  NEUTRAL (neither playbook strongly favoured)

Vol state from the short-vs-long vol percentile: top quartile => HIGH.

Trend direction (UP/DOWN) is taken from the sign of the lookback drift so that
trend strategies can pick a side; it is NOT a forecast (uses context only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.strategy_api import RegimeLabel
from src.regime.indicators import adx, atr, hurst_exponent, realized_vol, vol_percentile


class RegimeClassifier:
    def __init__(self,
                 hurst_trend: float = 0.55, hurst_range: float = 0.45,
                 adx_trend: float = 25.0, adx_range: float = 20.0,
                 adx_period: int = 14, atr_period: int = 14,
                 hurst_max_lag: int = 64,
                 vol_short: int = 96, vol_long: int = 512,
                 high_vol_pct: float = 0.75):
        self.hurst_trend = hurst_trend
        self.hurst_range = hurst_range
        self.adx_trend = adx_trend
        self.adx_range = adx_range
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.hurst_max_lag = hurst_max_lag
        self.vol_short = vol_short
        self.vol_long = vol_long
        self.high_vol_pct = high_vol_pct

    def classify(self, context_df: pd.DataFrame) -> RegimeLabel:
        close = context_df["close"].to_numpy(dtype=float)

        h = hurst_exponent(close, max_lag=self.hurst_max_lag)
        a = adx(context_df, period=self.adx_period)
        atr_v = atr(context_df, period=self.atr_period)
        rv = realized_vol(close, window=self.vol_short)
        vpct = vol_percentile(close, self.vol_short, self.vol_long)

        # Trend vs range.
        is_trend = (h > self.hurst_trend) and (a > self.adx_trend)
        is_range = (h < self.hurst_range) or (a < self.adx_range)

        if is_trend:
            # Direction from lookback drift sign.
            drift = close[-1] - close[max(0, len(close) - self.vol_short)]
            trend_state = "TREND_UP" if drift >= 0 else "TREND_DOWN"
        elif is_range:
            trend_state = "RANGE"
        else:
            trend_state = "NEUTRAL"

        vol_state = "HIGH" if vpct >= self.high_vol_pct else (
            "LOW" if vpct <= (1.0 - self.high_vol_pct) else "NORMAL")

        return RegimeLabel(
            trend_state=trend_state, vol_state=vol_state,
            features={"hurst": h, "adx": a, "atr": atr_v,
                      "realized_vol": rv, "vol_pct": vpct},
        )

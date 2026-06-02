"""Empirical quantile extraction from Monte Carlo forecast paths."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class QuantileLevels:
    """Risk levels derived from forecast path quantiles."""
    stop_loss: float            # Price level for stop loss
    take_profit: float          # Price level for take profit
    confidence_lo: float        # Lower confidence band
    confidence_hi: float        # Upper confidence band
    point_forecast: float       # Median forecast
    directional_confidence: float  # % of paths agreeing on direction
    entry_price: float          # Current price for reference

    @property
    def risk_reward_ratio(self) -> float:
        """Reward-to-risk ratio. > 1 means more upside than downside."""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else 0.0

    @property
    def interval_width(self) -> float:
        """Width of the SL-TP interval as fraction of entry price."""
        return (self.take_profit - self.stop_loss) / self.entry_price

    def __repr__(self) -> str:
        direction = "LONG" if self.point_forecast > self.entry_price else "SHORT"
        return (
            f"QuantileLevels({direction}):\n"
            f"  Entry:      {self.entry_price:.4f}\n"
            f"  Stop Loss:  {self.stop_loss:.4f} ({(self.stop_loss/self.entry_price - 1)*100:+.2f}%)\n"
            f"  Take Profit:{self.take_profit:.4f} ({(self.take_profit/self.entry_price - 1)*100:+.2f}%)\n"
            f"  R:R Ratio:  {self.risk_reward_ratio:.2f}\n"
            f"  Direction:  {self.directional_confidence:.0%} confidence\n"
            f"  Band Width: {self.interval_width:.2%}"
        )


def extract_quantiles(
    forecast_paths,  # ForecastPaths from sampler.py
    entry_price: float,
    sl_quantile: float = 0.05,
    tp_quantile: float = 0.95,
    band_lo: float = 0.25,
    band_hi: float = 0.75,
    horizon_idx: int = -1,
) -> QuantileLevels:
    """
    Extract risk levels from Monte Carlo forecast paths.

    Uses the distribution across N paths at a specific prediction
    horizon to compute percentile-based SL/TP levels.

    Args:
        forecast_paths: ForecastPaths with N trajectories
        entry_price: Current market price
        sl_quantile: Percentile for stop loss (default 5th)
        tp_quantile: Percentile for take profit (default 95th)
        band_lo: Lower confidence band percentile
        band_hi: Upper confidence band percentile
        horizon_idx: Which prediction timestep to use (-1 = final)

    Returns:
        QuantileLevels with all derived risk levels
    """
    # Use lows for stop loss (worst case), highs for take profit (best case)
    lows = forecast_paths.lows[:, horizon_idx]
    highs = forecast_paths.highs[:, horizon_idx]
    closes = forecast_paths.closes[:, horizon_idx]

    return QuantileLevels(
        stop_loss=float(np.quantile(lows, sl_quantile)),
        take_profit=float(np.quantile(highs, tp_quantile)),
        confidence_lo=float(np.quantile(closes, band_lo)),
        confidence_hi=float(np.quantile(closes, band_hi)),
        point_forecast=float(np.median(closes)),
        directional_confidence=forecast_paths.directional_confidence(entry_price),
        entry_price=entry_price,
    )


def extract_quantile_series(
    forecast_paths,
    entry_price: float,
    sl_quantile: float = 0.05,
    tp_quantile: float = 0.95,
) -> pd.DataFrame:
    """
    Extract quantile levels at every prediction timestep.

    Returns DataFrame with columns: timestamp, sl, tp, median, q25, q75
    """
    pred_len = forecast_paths.closes.shape[1]

    records = []
    for t in range(pred_len):
        lows_t = forecast_paths.lows[:, t]
        highs_t = forecast_paths.highs[:, t]
        closes_t = forecast_paths.closes[:, t]

        records.append({
            "step": t,
            "sl": float(np.quantile(lows_t, sl_quantile)),
            "tp": float(np.quantile(highs_t, tp_quantile)),
            "median": float(np.median(closes_t)),
            "q25": float(np.quantile(closes_t, 0.25)),
            "q75": float(np.quantile(closes_t, 0.75)),
            "mean": float(np.mean(closes_t)),
        })

    return pd.DataFrame(records)

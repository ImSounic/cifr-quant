"""Regime indicators computed from a price context window (no lookahead).

All functions take an OHLCV DataFrame (columns open/high/low/close/...) ordered
oldest->newest and return a scalar from the lookback only. Pure NumPy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hurst_exponent(close: np.ndarray, min_lag: int = 2, max_lag: int = 64) -> float:
    """Hurst exponent via the rescaled-range / variance-of-lagged-differences
    method. H > 0.5 trending (persistent), H < 0.5 mean-reverting, ~0.5 random.

    Uses the log-log slope of the std of lagged differences vs lag — robust and
    cheap. Returns 0.5 (random walk) if the series is too short or degenerate.
    """
    close = np.asarray(close, dtype=float)
    n = len(close)
    if n < min_lag + 2:
        return 0.5
    max_lag = int(min(max_lag, n // 2))
    if max_lag <= min_lag:
        return 0.5
    lags = np.arange(min_lag, max_lag)
    tau = []
    for lag in lags:
        diff = close[lag:] - close[:-lag]
        sd = np.std(diff)
        tau.append(sd if sd > 0 else 1e-12)
    tau = np.asarray(tau)
    # slope of log(tau) vs log(lag) ~ H
    try:
        slope = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    except Exception:
        return 0.5
    if not np.isfinite(slope):
        return 0.5
    return float(np.clip(slope, 0.0, 1.0))


def adx(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder's Average Directional Index on the context window. Returns the
    final ADX value. >25 trending, <20 ranging. 0 if insufficient data."""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    if n < 2 * period + 1:
        return 0.0

    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1]),
    ])

    def _wilder_smooth(x, p):
        out = np.zeros_like(x, dtype=float)
        if len(x) < p:
            return out
        out[p - 1] = x[:p].sum()
        for i in range(p, len(x)):
            out[i] = out[i - 1] - out[i - 1] / p + x[i]
        return out

    atr = _wilder_smooth(tr, period)
    plus_sm = _wilder_smooth(plus_dm, period)
    minus_sm = _wilder_smooth(minus_dm, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * np.where(atr > 0, plus_sm / atr, 0.0)
        minus_di = 100.0 * np.where(atr > 0, minus_sm / atr, 0.0)
        denom = plus_di + minus_di
        dx = 100.0 * np.where(denom > 0, np.abs(plus_di - minus_di) / denom, 0.0)

    # ADX = Wilder-smoothed DX over `period`, take the final value.
    valid = dx[period - 1:]
    if len(valid) < period:
        return float(dx[-1]) if len(dx) else 0.0
    adx_val = valid[:period].mean()
    for i in range(period, len(valid)):
        adx_val = (adx_val * (period - 1) + valid[i]) / period
    return float(adx_val)


def atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range (final value) — absolute price units."""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    if len(close) < 2:
        return 0.0
    tr = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1]),
    ])
    p = min(period, len(tr))
    if p <= 0:
        return 0.0
    return float(pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1])


def realized_vol(close: np.ndarray, window: int = 96) -> float:
    """Std of log-returns over the most recent `window` bars (per-bar vol)."""
    close = np.asarray(close, dtype=float)
    if len(close) < 3:
        return 0.0
    rets = np.diff(np.log(np.clip(close, 1e-12, None)))
    w = min(window, len(rets))
    return float(np.std(rets[-w:]))


def vol_percentile(close: np.ndarray, short_window: int = 96, long_window: int = 512) -> float:
    """Percentile rank (0..1) of current short-window vol within the trailing
    distribution of rolling vols over the long window. >0.75 => elevated."""
    close = np.asarray(close, dtype=float)
    if len(close) < short_window + 5:
        return 0.5
    rets = np.diff(np.log(np.clip(close, 1e-12, None)))
    s = pd.Series(rets)
    rolling = s.rolling(short_window).std().dropna()
    if rolling.empty:
        return 0.5
    tail = rolling.iloc[-long_window:]
    current = rolling.iloc[-1]
    return float((tail <= current).mean())

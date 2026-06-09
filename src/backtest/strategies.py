"""Concrete portfolio strategies.

Phase A ships the baseline `DirectionalMomentum` which must reproduce the exact
logic of the original `run_market_backtest` loop (so the refactor is provably
faithful). Mean-reversion / regime-gated / cross-sectional land in later phases.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.backtest.strategy_api import AssetDecision, Strategy, TradeIntent
from src.regime.indicators import atr as _atr


class DirectionalMomentum(Strategy):
    """Original strategy: trade the ensemble's majority direction, gated by
    confidence, with SL/TP taken from the CQR-widened q05/q95 band.

    Reproduces the old engine exactly:
      - gate: confidence >= min_confidence and direction in {long, short}
      - long  -> stop=lower(q05-corr), target=upper(q95+corr)
      - short -> stop=upper,           target=lower
      - conviction = 1 / width, width = (upper-lower)/entry (clamped at min_width)
    """
    name = "directional_momentum"

    def __init__(self, min_confidence: float = 0.55, min_width: float = 1e-4):
        self.min_confidence = min_confidence
        self.min_width = min_width

    def _decide_one(self, d: AssetDecision) -> Optional[TradeIntent]:
        f = d.forecast
        entry = float(f.entry_price)
        if entry <= 0:
            return None
        lower = f.lower          # q05 - cqr_correction
        upper = f.upper          # q95 + cqr_correction
        width = (upper - lower) / entry if entry > 0 else np.inf
        if f.confidence < self.min_confidence or f.direction not in ("long", "short"):
            return None

        if f.direction == "long":
            stop, target = lower, upper
        else:
            stop, target = upper, lower

        return TradeIntent(
            symbol=d.symbol,
            direction=f.direction,
            entry_ref=entry,
            stop=stop,
            target=target,
            conviction=1.0 / max(width, self.min_width),
            meta={"strategy": self.name, "confidence": f.confidence, "width": width},
        )


class RegimeGatedTrend(DirectionalMomentum):
    """DirectionalMomentum that only fires in a TREND regime (and, by default,
    only when the forecast direction agrees with the trend's direction). Stands
    aside in RANGE / NEUTRAL and (optionally) in HIGH-vol windows.

    This is the direct test of "does regime gating fix crypto?" — same signal,
    same SL/TP, just refused outside trends.
    """
    name = "regime_gated_trend"

    def __init__(self, min_confidence: float = 0.55, min_width: float = 1e-4,
                 require_trend_alignment: bool = True, avoid_high_vol: bool = True):
        super().__init__(min_confidence=min_confidence, min_width=min_width)
        self.require_trend_alignment = require_trend_alignment
        self.avoid_high_vol = avoid_high_vol

    def _decide_one(self, d: AssetDecision) -> Optional[TradeIntent]:
        reg = d.regime
        if reg is None or not reg.is_trend:
            return None
        if self.avoid_high_vol and reg.vol_state == "HIGH":
            return None
        intent = super()._decide_one(d)
        if intent is None:
            return None
        if self.require_trend_alignment:
            want = "long" if reg.trend_state == "TREND_UP" else "short"
            if intent.direction != want:
                return None
        intent.meta["regime"] = reg.trend_state
        intent.meta["strategy"] = self.name
        return intent


class MeanReversion(Strategy):
    """Fade extension from the lookback mean; revert toward it. Designed for
    RANGE regimes (the opposite hypothesis to momentum).

    Entry when |z| = |(price - MA) / std| exceeds `entry_z`:
      z > 0 (extended up)  -> short, target = MA
      z < 0 (extended down)-> long,  target = MA
    Stop = entry +/- `stop_atr_mult` * ATR (against the trade). Conviction = |z|.
    """
    name = "mean_reversion"

    def __init__(self, ma_window: int = 96, entry_z: float = 1.5,
                 stop_atr_mult: float = 2.0, atr_period: int = 14,
                 only_in_range: bool = True, avoid_high_vol: bool = True):
        self.ma_window = ma_window
        self.entry_z = entry_z
        self.stop_atr_mult = stop_atr_mult
        self.atr_period = atr_period
        self.only_in_range = only_in_range
        self.avoid_high_vol = avoid_high_vol

    def _decide_one(self, d: AssetDecision) -> Optional[TradeIntent]:
        reg = d.regime
        if self.only_in_range and (reg is None or not reg.is_range):
            return None
        if self.avoid_high_vol and reg is not None and reg.vol_state == "HIGH":
            return None

        ctx = d.context_df
        close = ctx["close"].to_numpy(dtype=float)
        if len(close) < self.ma_window + 2:
            return None
        window = close[-self.ma_window:]
        ma = float(window.mean())
        sd = float(window.std())
        entry = float(d.forecast.entry_price)
        if sd <= 0 or entry <= 0:
            return None

        z = (entry - ma) / sd
        if abs(z) < self.entry_z:
            return None

        atr_v = _atr(ctx, period=self.atr_period)
        if atr_v <= 0:
            return None

        if z > 0:  # extended above mean -> short, revert down to MA
            direction, target = "short", ma
            stop = entry + self.stop_atr_mult * atr_v
        else:      # extended below mean -> long, revert up to MA
            direction, target = "long", ma
            stop = entry - self.stop_atr_mult * atr_v

        return TradeIntent(
            symbol=d.symbol, direction=direction, entry_ref=entry,
            stop=stop, target=target, conviction=abs(z),
            meta={"strategy": self.name, "z": z, "atr": atr_v,
                  "regime": (reg.trend_state if reg else "NA")},
        )

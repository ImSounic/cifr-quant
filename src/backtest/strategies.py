"""Concrete portfolio strategies.

Phase A ships the baseline `DirectionalMomentum` which must reproduce the exact
logic of the original `run_market_backtest` loop (so the refactor is provably
faithful). Mean-reversion / regime-gated / cross-sectional land in later phases.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.backtest.strategy_api import AssetDecision, Strategy, TradeIntent


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

"""Position sizers for the portfolio backtest.

Phase A ships `InverseWidthRiskParity`, which reproduces the original engine's
sizing exactly: weight each intent by its conviction (= 1/width), normalise,
cap at max_position_pct, and DO NOT renormalise after capping (so when caps
bind the book is intentionally under-deployed — matching the old behaviour).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.backtest.strategy_api import Sizer, TradeIntent


class InverseWidthRiskParity(Sizer):
    name = "inverse_width_risk_parity"

    def __init__(self, max_position_pct: float = 0.10):
        self.max_position_pct = max_position_pct

    def size(self, intents: List[TradeIntent], capital: float) -> Dict[str, float]:
        if not intents:
            return {}
        conv = np.array([it.conviction for it in intents], dtype=float)
        total = conv.sum()
        if total <= 0:
            return {it.symbol: 0.0 for it in intents}
        raw_w = conv / total
        weights = np.minimum(raw_w, self.max_position_pct)
        return {it.symbol: float(w) * capital for it, w in zip(intents, weights)}


class EqualWeight(Sizer):
    name = "equal_weight"

    def __init__(self, max_position_pct: float = 0.10):
        self.max_position_pct = max_position_pct

    def size(self, intents: List[TradeIntent], capital: float) -> Dict[str, float]:
        if not intents:
            return {}
        w = min(1.0 / len(intents), self.max_position_pct)
        return {it.symbol: w * capital for it in intents}

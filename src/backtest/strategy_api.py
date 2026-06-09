"""Portfolio-native strategy interface for the multi-asset backtest.

Separates the four fused concerns of the old engine: FORECAST (cached Kronos
output), REGIME (classification), STRATEGY (decide trade), SIZER (allocate
capital). The expensive forecast is computed once and cached, so strategies are
swapped and A/B-tested CPU-only.

This is the PORTFOLIO interface (shared capital, cross-sectional ranking). It is
deliberately separate from the single-asset `src/strategy/` LLM layer so neither
breaks the other; names/semantics are mirrored so they can merge later.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  Data contracts
# --------------------------------------------------------------------------- #
@dataclass
class ForecastBundle:
    """One Kronos forecast for (symbol, rebalance_t). Final-step horizon values.

    `cqr_correction` is attached at run time (not baked into the cache) so a
    strategy may choose raw or CQR-widened bands.
    """
    symbol: str
    timestamp: pd.Timestamp
    entry_price: float
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float
    mean: float
    direction: str            # raw majority-vote direction ('long'/'short')
    confidence: float         # raw directional_confidence in [0.5, 1.0]
    pred_len: int
    cqr_correction: float = 0.0

    def with_correction(self, corr: float) -> "ForecastBundle":
        return replace(self, cqr_correction=corr)

    @property
    def lower(self) -> float:
        """CQR-widened lower band."""
        return self.q05 - self.cqr_correction

    @property
    def upper(self) -> float:
        """CQR-widened upper band."""
        return self.q95 + self.cqr_correction


@dataclass
class RegimeLabel:
    trend_state: str = "UNKNOWN"   # TREND_UP | TREND_DOWN | RANGE | UNKNOWN
    vol_state: str = "NORMAL"      # LOW | NORMAL | HIGH
    features: dict = field(default_factory=dict)

    @property
    def is_trend(self) -> bool:
        return self.trend_state in ("TREND_UP", "TREND_DOWN")

    @property
    def is_range(self) -> bool:
        return self.trend_state == "RANGE"


@dataclass
class AssetDecision:
    """Everything a strategy may look at for one asset at one rebalance.

    `future_df` is intentionally NOT part of this object — strategies cannot see
    the future. The engine holds futures separately and passes them only to the
    simulator after intents are produced.
    """
    symbol: str
    context_df: pd.DataFrame
    forecast: ForecastBundle
    regime: Optional[RegimeLabel] = None


@dataclass
class TradeIntent:
    symbol: str
    direction: str            # 'long' | 'short'
    entry_ref: float
    stop: float
    target: float
    conviction: float         # fed to the sizer; higher = bigger desired weight
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Base classes
# --------------------------------------------------------------------------- #
class Strategy:
    """Maps a list of AssetDecisions to a list of TradeIntents.

    Per-asset strategies implement `_decide_one`. Cross-sectional strategies
    (which need all assets at once) override `generate_intents`.
    """
    name = "base"

    def generate_intents(self, decisions: List[AssetDecision]) -> List[TradeIntent]:
        out: List[TradeIntent] = []
        for d in decisions:
            intent = self._decide_one(d)
            if intent is not None:
                out.append(intent)
        return out

    def _decide_one(self, decision: AssetDecision) -> Optional[TradeIntent]:
        raise NotImplementedError


class Sizer:
    """Allocates capital across intents → {symbol: position_usd}."""
    name = "base"

    def size(self, intents: List[TradeIntent], capital: float) -> Dict[str, float]:
        raise NotImplementedError

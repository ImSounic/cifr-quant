"""Transaction cost models for realistic backtesting."""

from dataclasses import dataclass


@dataclass
class CostModel:
    """Transaction cost model for a specific market."""
    spread_pct: float = 0.0      # Bid-ask spread as fraction
    slippage_pct: float = 0.0    # Market impact / slippage
    commission_pct: float = 0.0  # Broker commission

    @property
    def total_one_way(self) -> float:
        """Total cost for a single entry or exit."""
        return self.spread_pct / 2 + self.slippage_pct + self.commission_pct

    @property
    def total_round_trip(self) -> float:
        """Total cost for entry + exit."""
        return self.spread_pct + 2 * self.slippage_pct + 2 * self.commission_pct

    def apply_entry(self, price: float, direction: str) -> float:
        """Apply entry costs to get effective fill price."""
        cost = self.total_one_way
        if direction == "long":
            return price * (1 + cost)  # Pay more to buy
        else:
            return price * (1 - cost)  # Receive less to sell

    def apply_exit(self, price: float, direction: str) -> float:
        """Apply exit costs to get effective fill price."""
        cost = self.total_one_way
        if direction == "long":
            return price * (1 - cost)  # Receive less when selling
        else:
            return price * (1 + cost)  # Pay more to cover


# Pre-built cost models
BTC_COSTS = CostModel(spread_pct=0.0001, slippage_pct=0.0002, commission_pct=0.0004)
EUR_COSTS = CostModel(spread_pct=0.00008, slippage_pct=0.00002, commission_pct=0.0)
XAU_COSTS = CostModel(spread_pct=0.00015, slippage_pct=0.00005, commission_pct=0.0001)

# Market-level presets for the multi-asset portfolio backtest.
# Crypto: Binance taker fee 0.04% + spread/slippage (liquid majors+large caps).
# Commodity: tighter spreads, small slippage, modest commission (4h bars, futures).
CRYPTO_COSTS = CostModel(spread_pct=0.0002, slippage_pct=0.0003, commission_pct=0.0004)
COMMODITY_COSTS = CostModel(spread_pct=0.00015, slippage_pct=0.00010, commission_pct=0.0001)

COST_MODELS = {
    "btc": BTC_COSTS,
    "eur": EUR_COSTS,
    "xau": XAU_COSTS,
    "crypto": CRYPTO_COSTS,
    "commodity": COMMODITY_COSTS,
}

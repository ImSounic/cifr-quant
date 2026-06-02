"""EUR/USD configuration — 1-hour candles via TwelveData."""

from configs.base_config import MarketConfig


EUR_CONFIG = MarketConfig(
    instrument="EUR/USD",
    timeframe="1h",
    candles_per_year=6_240,        # ~24 * 260 trading days
    pred_len=24,                   # 24 candles = 1 day
    data_years=5,

    # Cost model (institutional forex)
    spread_pct=0.00008,            # ~0.8 pips on EUR/USD
    slippage_pct=0.00002,          # ~0.2 pips
    commission_pct=0.0,            # No commission on spread-based

    has_volume=False,              # Forex spot has no real volume
    data_source="twelvedata",
)

"""BTC/USDT configuration — 15-minute candles via Binance."""

from configs.base_config import MarketConfig


BTC_CONFIG = MarketConfig(
    instrument="BTC/USDT",
    timeframe="15m",
    candles_per_year=70_080,       # 4 * 24 * 365
    pred_len=48,                   # 48 candles = 12 hours
    data_years=2,

    # Cost model (Binance spot, maker)
    spread_pct=0.0001,             # ~0.01%
    slippage_pct=0.0002,           # ~0.02%
    commission_pct=0.0004,         # 0.04% maker

    has_volume=True,
    data_source="binance",
)

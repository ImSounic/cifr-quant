"""XAU/USD (Gold) configuration — 4-hour candles via yfinance."""

from configs.base_config import MarketConfig


XAU_CONFIG = MarketConfig(
    instrument="XAU/USD",
    timeframe="4h",
    candles_per_year=1_560,        # 6 * 260 trading days
    pred_len=6,                    # 6 candles = 1 day
    data_years=5,                  # Need 5+ years (small candle count per year)

    # Cost model (gold CFD/futures)
    spread_pct=0.00015,            # ~$0.30 on ~$2000 = 0.015%
    slippage_pct=0.00005,          # ~$0.10
    commission_pct=0.0001,         # varies by broker

    has_volume=False,              # Gold spot volume unreliable from free sources
    data_source="yfinance",        # GC=F (gold futures) has real volume
)

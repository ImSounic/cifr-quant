"""Crypto universe — all tradeable assets on Binance USDT pairs.

Assets selected by: liquidity (top 30 by volume), data availability (2+ years),
and correlation structure (want diversity, not 30 BTC clones).
"""

from configs.base_config import MarketConfig


# Shared cost model for Binance spot (maker fees)
CRYPTO_COSTS = {
    "spread_pct": 0.0001,       # ~0.01% (tight for majors, wider for alts)
    "slippage_pct": 0.0002,     # ~0.02%
    "commission_pct": 0.0004,   # 0.04% maker (VIP0)
}


# Tier 1: Majors — highest liquidity, most data, lowest costs
TIER1_ASSETS = {
    "BTC/USDT":  {"pred_len": 48, "notes": "Reference asset, finetuned checkpoint"},
    "ETH/USDT":  {"pred_len": 48, "notes": "Second most liquid, high correlation with BTC"},
    "BNB/USDT":  {"pred_len": 48, "notes": "Exchange token, partially decorrelated"},
    "SOL/USDT":  {"pred_len": 48, "notes": "High vol, fast-moving"},
    "XRP/USDT":  {"pred_len": 48, "notes": "Low correlation with BTC during range periods"},
}

# Tier 2: Large caps — good liquidity, moderate correlation
TIER2_ASSETS = {
    "ADA/USDT":  {"pred_len": 48},
    "AVAX/USDT": {"pred_len": 48},
    "DOGE/USDT": {"pred_len": 48},
    "DOT/USDT":  {"pred_len": 48},
    "LINK/USDT": {"pred_len": 48},
    "MATIC/USDT": {"pred_len": 48},
    "UNI/USDT":  {"pred_len": 48},
    "ATOM/USDT": {"pred_len": 48},
    "LTC/USDT":  {"pred_len": 48},
    "NEAR/USDT": {"pred_len": 48},
}

# Tier 3: Mid caps — higher alpha potential, higher risk, wider spreads
TIER3_ASSETS = {
    "APT/USDT":  {"pred_len": 48},
    "ARB/USDT":  {"pred_len": 48},
    "FIL/USDT":  {"pred_len": 48},
    "INJ/USDT":  {"pred_len": 48},
    "OP/USDT":   {"pred_len": 48},
    "SUI/USDT":  {"pred_len": 48},
    "TIA/USDT":  {"pred_len": 48},
    "SEI/USDT":  {"pred_len": 48},
    "AAVE/USDT": {"pred_len": 48},
    "MKR/USDT":  {"pred_len": 48},
}


def get_crypto_configs(tiers=(1, 2), timeframe="15m", lookback=512):
    """
    Build MarketConfig for each crypto asset in the selected tiers.

    Args:
        tiers: Which tiers to include (1=majors, 2=large, 3=mid)
        timeframe: Candle interval
        lookback: Context window for Kronos

    Returns:
        Dict of {symbol: MarketConfig}
    """
    assets = {}
    if 1 in tiers:
        assets.update(TIER1_ASSETS)
    if 2 in tiers:
        assets.update(TIER2_ASSETS)
    if 3 in tiers:
        assets.update(TIER3_ASSETS)

    configs = {}
    for symbol, info in assets.items():
        configs[symbol] = MarketConfig(
            instrument=symbol,
            timeframe=timeframe,
            candles_per_year=70_080,   # 4 * 24 * 365 (24/7 market)
            pred_len=info["pred_len"],
            data_years=2,
            spread_pct=CRYPTO_COSTS["spread_pct"],
            slippage_pct=CRYPTO_COSTS["slippage_pct"],
            commission_pct=CRYPTO_COSTS["commission_pct"],
            has_volume=True,
            data_source="binance",
        )

    return configs


# Quick access
CRYPTO_UNIVERSE = get_crypto_configs(tiers=(1, 2))
CRYPTO_UNIVERSE_FULL = get_crypto_configs(tiers=(1, 2, 3))

# The finetuned checkpoint asset — all other crypto uses this + zero-shot ensemble
CRYPTO_FINETUNE_ASSET = "BTC/USDT"

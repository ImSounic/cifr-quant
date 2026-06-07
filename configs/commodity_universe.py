"""Commodity universe — tradeable via TwelveData or futures exchanges.

Commodities trade on different schedules and have varying liquidity.
All use 4h timeframe to capture macro trend structure.
Volume is unreliable for spot pairs — use futures symbols where possible.
"""

from configs.base_config import MarketConfig


# Shared cost model for commodity CFDs/futures
COMMODITY_COSTS = {
    "spread_pct": 0.00015,      # Varies by instrument
    "slippage_pct": 0.00005,
    "commission_pct": 0.0001,
}


# Precious metals — strong trending, safe haven dynamics
PRECIOUS_METALS = {
    "XAU/USD": {
        "pred_len": 6,
        "notes": "Gold — reference asset, finetuned checkpoint. Macro-driven, central bank flows.",
        "twelvedata_symbol": "XAU/USD",
        "futures_symbol": "GC=F",
        "has_volume": False,  # spot has no real volume
    },
    "XAG/USD": {
        "pred_len": 6,
        "notes": "Silver — higher vol than gold, industrial + precious demand.",
        "twelvedata_symbol": "XAG/USD",
        "futures_symbol": "SI=F",
        "has_volume": False,
    },
    "XPT/USD": {
        "pred_len": 6,
        "notes": "Platinum — industrial metal, auto catalysts.",
        "twelvedata_symbol": "XPT/USD",
        "futures_symbol": "PL=F",
        "has_volume": False,
    },
}

# Energy — supply/demand driven, geopolitical risk
ENERGY = {
    "CRUDE_OIL": {
        "pred_len": 6,
        "notes": "WTI Crude — most liquid commodity globally. OPEC driven.",
        "twelvedata_symbol": "WTI/USD",
        "futures_symbol": "CL=F",
        "has_volume": True,  # futures have real volume
    },
    "BRENT_OIL": {
        "pred_len": 6,
        "notes": "Brent Crude — international benchmark. Correlated with WTI.",
        "twelvedata_symbol": "BRENT/USD",
        "futures_symbol": "BZ=F",
        "has_volume": True,
    },
    "NATURAL_GAS": {
        "pred_len": 6,
        "notes": "Henry Hub — seasonal, weather-driven, very volatile.",
        "twelvedata_symbol": "NG/USD",
        "futures_symbol": "NG=F",
        "has_volume": True,
    },
}

# Industrial metals — China demand, construction, manufacturing
INDUSTRIAL_METALS = {
    "COPPER": {
        "pred_len": 6,
        "notes": "Dr. Copper — economic bellwether. China demand driven.",
        "twelvedata_symbol": "XCU/USD",
        "futures_symbol": "HG=F",
        "has_volume": True,
    },
}


def get_commodity_configs(
    categories=("precious", "energy"),
    timeframe="4h",
    lookback=200,
):
    """
    Build MarketConfig for each commodity asset.

    Args:
        categories: Which categories to include
        timeframe: Candle interval (4h recommended)
        lookback: Context window

    Returns:
        Dict of {symbol_key: MarketConfig}
    """
    assets = {}
    if "precious" in categories:
        assets.update(PRECIOUS_METALS)
    if "energy" in categories:
        assets.update(ENERGY)
    if "industrial" in categories:
        assets.update(INDUSTRIAL_METALS)

    configs = {}
    for key, info in assets.items():
        symbol = info.get("twelvedata_symbol", key)
        configs[key] = MarketConfig(
            instrument=symbol,
            timeframe=timeframe,
            candles_per_year=1_560,   # 6 candles/day * 260 trading days
            pred_len=info["pred_len"],
            data_years=5,             # Commodities need longer history
            spread_pct=COMMODITY_COSTS["spread_pct"],
            slippage_pct=COMMODITY_COSTS["slippage_pct"],
            commission_pct=COMMODITY_COSTS["commission_pct"],
            has_volume=info.get("has_volume", False),
            data_source="twelvedata",
        )

    return configs


# Quick access
COMMODITY_UNIVERSE = get_commodity_configs(categories=("precious", "energy"))
COMMODITY_UNIVERSE_FULL = get_commodity_configs(categories=("precious", "energy", "industrial"))

# The finetuned checkpoint asset
COMMODITY_FINETUNE_ASSET = "XAU/USD"

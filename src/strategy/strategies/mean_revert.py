"""Mean reversion strategy — enters when price is at confidence band extremes."""


def mean_revert_signal(data, capital, position, context=None, params=None):
    """
    Mean reversion signal generator.

    Entry: price near confidence band edges AND high volatility regime
    Direction: fade the extreme (buy at lower band, sell at upper band)

    Params:
        min_vol_percentile: float (default 0.70)
        band_threshold: float (default 0.9, how close to band edge)
        max_risk_pct: float (default 0.015)
    """
    if context is None or position is not None:
        return None

    p = params or {}
    min_vol = p.get("min_vol_percentile", 0.70)
    band_thresh = p.get("band_threshold", 0.90)
    max_risk_pct = p.get("max_risk_pct", 0.015)

    if context.volatility_percentile < min_vol:
        return None

    price = context.current_price
    lo = context.confidence_lo
    hi = context.confidence_hi
    band_width = hi - lo

    if band_width <= 0:
        return None

    # How far is price from band center?
    center = (lo + hi) / 2
    distance_from_center = abs(price - center) / (band_width / 2)

    if distance_from_center < band_thresh:
        return None  # Price not extreme enough

    # Fade the extreme
    risk_per_unit = abs(price - context.stop_loss)
    if risk_per_unit <= 0:
        return None

    max_risk = capital * max_risk_pct
    units = max_risk / risk_per_unit

    is_at_lower_band = price < center

    return {
        "action": "buy" if is_at_lower_band else "sell",
        "units": units,
        "stop_loss": context.stop_loss,
        "take_profit": context.point_forecast,  # Revert to median
    }

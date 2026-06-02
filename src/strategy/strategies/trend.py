"""Trend following strategy — enters when directional confidence is high."""


def trend_signal(data, capital, position, context=None, params=None):
    """
    Trend following signal generator.

    Entry: directional_confidence > threshold AND trend_strength > threshold
    Exit: via SL/TP (handled by backtest engine)

    Params:
        min_confidence: float (default 0.70)
        min_trend_strength: float (default 0.60)
        min_rr_ratio: float (default 1.5)
    """
    if context is None or position is not None:
        return None

    p = params or {}
    min_conf = p.get("min_confidence", 0.70)
    min_trend = p.get("min_trend_strength", 0.60)
    min_rr = p.get("min_rr_ratio", 1.5)
    max_risk_pct = p.get("max_risk_pct", 0.02)

    dc = context.directional_confidence
    ts = context.trend_strength
    rr = context.risk_reward_ratio

    if dc < min_conf or ts < min_trend or rr < min_rr:
        return None

    # Position sizing: risk max_risk_pct of capital
    risk_per_unit = abs(context.current_price - context.stop_loss)
    if risk_per_unit <= 0:
        return None

    max_risk = capital * max_risk_pct
    units = max_risk / risk_per_unit

    # Direction
    is_long = context.point_forecast > context.current_price

    return {
        "action": "buy" if is_long else "sell",
        "units": units,
        "stop_loss": context.stop_loss,
        "take_profit": context.take_profit,
    }

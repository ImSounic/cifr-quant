"""Breakout strategy — enters on low-volatility compression then expansion."""


def breakout_signal(data, capital, position, context=None, params=None):
    """
    Breakout signal generator.

    Entry: low path dispersion (compression) + strong directional confidence
    The idea: when forecast paths agree tightly, a breakout from that
    consensus is likely to continue.

    Params:
        max_interval_width: float (default 0.02, tight bands = compression)
        min_confidence: float (default 0.65)
        max_risk_pct: float (default 0.02)
    """
    if context is None or position is not None:
        return None

    p = params or {}
    max_width = p.get("max_interval_width", 0.02)
    min_conf = p.get("min_confidence", 0.65)
    max_risk_pct = p.get("max_risk_pct", 0.02)

    # Compression: narrow interval width
    if context.interval_width > max_width:
        return None

    # Must have directional conviction
    if context.directional_confidence < min_conf:
        return None

    risk_per_unit = abs(context.current_price - context.stop_loss)
    if risk_per_unit <= 0:
        return None

    max_risk = capital * max_risk_pct
    units = max_risk / risk_per_unit

    is_long = context.point_forecast > context.current_price

    return {
        "action": "buy" if is_long else "sell",
        "units": units,
        "stop_loss": context.stop_loss,
        "take_profit": context.take_profit,
    }

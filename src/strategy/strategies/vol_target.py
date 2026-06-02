"""Volatility targeting strategy — sizes positions to target constant portfolio vol."""


def vol_target_signal(data, capital, position, context=None, params=None):
    """
    Volatility targeting signal generator.

    Always enters if directional confidence exceeds minimum.
    Position size scales inversely with current volatility
    to maintain constant portfolio risk.

    Params:
        target_vol: float (default 0.15, annualized target vol)
        min_confidence: float (default 0.60)
        lookback: int (default 20, periods for realized vol)
    """
    if context is None or position is not None:
        return None

    p = params or {}
    target_vol = p.get("target_vol", 0.15)
    min_conf = p.get("min_confidence", 0.60)
    lookback = p.get("lookback", 20)

    if context.directional_confidence < min_conf:
        return None

    # Estimate realized vol from recent data
    if len(data) < lookback + 1:
        return None

    recent_closes = data["close"].iloc[-lookback:].values
    returns = (recent_closes[1:] / recent_closes[:-1]) - 1
    realized_vol = float(returns.std())

    if realized_vol <= 0:
        return None

    # Scale position: target_vol / realized_vol
    vol_scalar = target_vol / (realized_vol * (252 ** 0.5))  # Annualize
    vol_scalar = max(0.1, min(vol_scalar, 3.0))  # Clamp

    risk_per_unit = abs(context.current_price - context.stop_loss)
    if risk_per_unit <= 0:
        return None

    base_risk = capital * 0.02  # 2% base risk
    units = (base_risk * vol_scalar) / risk_per_unit

    is_long = context.point_forecast > context.current_price

    return {
        "action": "buy" if is_long else "sell",
        "units": units,
        "stop_loss": context.stop_loss,
        "take_profit": context.take_profit,
    }

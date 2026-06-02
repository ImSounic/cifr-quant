"""Adaptive position sizing based on quantile interval width and confidence."""

from dataclasses import dataclass


@dataclass
class PositionSizing:
    """Computed position size and rationale."""
    position_size: float        # Fraction of capital to allocate (0 to max_position_pct)
    direction: str              # 'long', 'short', or 'flat'
    reason: str                 # Human-readable explanation
    risk_per_unit: float        # Dollar risk per unit
    units: float                # Number of units to trade

    def __repr__(self) -> str:
        return (
            f"Position: {self.direction.upper()} {self.position_size:.1%} of capital\n"
            f"  Units: {self.units:.4f}\n"
            f"  Risk/unit: ${self.risk_per_unit:.2f}\n"
            f"  Reason: {self.reason}"
        )


def compute_position(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    directional_confidence: float,
    capital: float,
    max_position_pct: float = 0.10,
    min_confidence: float = 0.60,
    max_risk_pct: float = 0.02,
) -> PositionSizing:
    """
    Compute position size using fixed-fractional risk model.

    Position size is determined by:
    1. Maximum risk per trade (default 2% of capital)
    2. Distance to stop loss (determines units)
    3. Directional confidence (scales position down if uncertain)
    4. Maximum position cap (default 10% of capital)

    Args:
        entry_price: Current price
        stop_loss: Stop loss level (CQR-adjusted)
        take_profit: Take profit level (CQR-adjusted)
        directional_confidence: Fraction of paths agreeing on direction
        capital: Available capital
        max_position_pct: Maximum fraction of capital per trade
        min_confidence: Minimum confidence to enter (below = flat)
        max_risk_pct: Maximum capital at risk per trade

    Returns:
        PositionSizing with computed values
    """
    # Determine direction
    if take_profit > entry_price and directional_confidence >= min_confidence:
        direction = "long"
        risk_per_unit = abs(entry_price - stop_loss)
    elif take_profit < entry_price and (1 - directional_confidence) >= min_confidence:
        direction = "short"
        risk_per_unit = abs(stop_loss - entry_price)
    else:
        return PositionSizing(
            position_size=0.0,
            direction="flat",
            reason=f"Confidence {directional_confidence:.0%} below threshold {min_confidence:.0%}",
            risk_per_unit=0.0,
            units=0.0,
        )

    if risk_per_unit <= 0:
        return PositionSizing(
            position_size=0.0,
            direction="flat",
            reason="Stop loss at or beyond entry price",
            risk_per_unit=0.0,
            units=0.0,
        )

    # Fixed-fractional: risk max_risk_pct of capital per trade
    max_risk_dollars = capital * max_risk_pct
    units = max_risk_dollars / risk_per_unit

    # Scale by confidence (higher confidence = closer to full size)
    confidence_scalar = (directional_confidence - min_confidence) / (1.0 - min_confidence)
    confidence_scalar = max(0.0, min(1.0, confidence_scalar))
    units *= confidence_scalar

    # Cap at max position size
    position_value = units * entry_price
    max_value = capital * max_position_pct
    if position_value > max_value:
        units = max_value / entry_price
        position_value = max_value

    position_size = position_value / capital

    return PositionSizing(
        position_size=position_size,
        direction=direction,
        reason=f"Confidence {directional_confidence:.0%}, risk/unit ${risk_per_unit:.2f}",
        risk_per_unit=risk_per_unit,
        units=units,
    )

"""Strategy executor — runs a named strategy with given parameters."""

from typing import Callable, Optional

from src.strategy.strategies.trend import trend_signal
from src.strategy.strategies.mean_revert import mean_revert_signal
from src.strategy.strategies.breakout import breakout_signal
from src.strategy.strategies.vol_target import vol_target_signal


STRATEGIES = {
    "trend": trend_signal,
    "mean_revert": mean_revert_signal,
    "breakout": breakout_signal,
    "vol_target": vol_target_signal,
}


def get_strategy(name: str) -> Callable:
    """Get a strategy function by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]


def make_signal_fn(
    strategy_name: str,
    context,
    params: Optional[dict] = None,
) -> Callable:
    """
    Create a signal function compatible with the backtest engine.

    The backtest engine calls signal_fn(data, capital, position).
    This wraps a strategy to inject the market context and params.

    Args:
        strategy_name: Name of strategy ('trend', 'mean_revert', etc.)
        context: MarketContext from context_builder
        params: Strategy-specific parameters

    Returns:
        Callable compatible with backtest engine
    """
    strategy_fn = get_strategy(strategy_name)

    def signal_fn(data, capital, position):
        return strategy_fn(data, capital, position, context=context, params=params)

    return signal_fn

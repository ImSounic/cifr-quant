"""Build structured context packs for LLM strategy generation."""

import json
from dataclasses import dataclass, asdict
from typing import Optional

from src.risk.quantile import QuantileLevels


@dataclass
class MarketContext:
    """Complete market context for LLM strategy generation."""

    # Instrument info
    instrument: str
    timeframe: str
    current_price: float

    # Kronos forecast
    point_forecast: float
    forecast_horizon_candles: int
    directional_confidence: float  # 0-1

    # Risk levels (CQR-adjusted)
    stop_loss: float
    take_profit: float
    confidence_lo: float
    confidence_hi: float
    risk_reward_ratio: float
    interval_width: float

    # Regime signals
    volatility_percentile: float   # 0-1, current vol vs historical
    path_dispersion: float         # Variance of path endpoints
    trend_strength: float          # 0-1, directional consistency

    # Recent performance (last N trades)
    recent_win_rate: Optional[float] = None
    recent_avg_pnl: Optional[float] = None
    recent_n_trades: int = 0

    def to_prompt(self) -> str:
        """Format as structured text for LLM input."""
        direction = "BULLISH" if self.point_forecast > self.current_price else "BEARISH"
        forecast_return = (self.point_forecast / self.current_price - 1) * 100

        return f"""## Market Analysis: {self.instrument} ({self.timeframe})

**Current Price**: {self.current_price:.4f}
**Forecast**: {self.point_forecast:.4f} ({forecast_return:+.2f}% over {self.forecast_horizon_candles} candles)
**Direction**: {direction} — {self.directional_confidence:.0%} of Monte Carlo paths agree

### Risk Levels (CQR-calibrated, 90% coverage)
- Stop Loss: {self.stop_loss:.4f} ({(self.stop_loss/self.current_price - 1)*100:+.2f}%)
- Take Profit: {self.take_profit:.4f} ({(self.take_profit/self.current_price - 1)*100:+.2f}%)
- Risk:Reward Ratio: {self.risk_reward_ratio:.2f}
- Confidence Band: [{self.confidence_lo:.4f}, {self.confidence_hi:.4f}]
- Interval Width: {self.interval_width:.2%}

### Regime Indicators
- Volatility Percentile: {self.volatility_percentile:.0%} (vs 1-year history)
- Path Dispersion: {self.path_dispersion:.6f} (low = consensus, high = uncertainty)
- Trend Strength: {self.trend_strength:.0%} (% of paths trending same direction)

### Recent Performance
- Last {self.recent_n_trades} trades: {self.recent_win_rate:.0%} win rate, avg P&L {self.recent_avg_pnl:+.2%}
"""

    def to_dict(self) -> dict:
        return asdict(self)


def build_context(
    instrument: str,
    timeframe: str,
    current_price: float,
    quantile_levels: QuantileLevels,
    forecast_paths,   # ForecastPaths from sampler
    pred_len: int,
    recent_trades: Optional[list] = None,
    historical_vol: Optional[float] = None,
    current_vol: Optional[float] = None,
) -> MarketContext:
    """
    Construct a MarketContext from forecast and risk components.

    Args:
        instrument: e.g., "BTC/USDT"
        timeframe: e.g., "15m"
        current_price: Latest market price
        quantile_levels: CQR-adjusted risk levels
        forecast_paths: Monte Carlo paths from sampler
        pred_len: Prediction horizon in candles
        recent_trades: List of recent Trade objects for performance stats
        historical_vol: Historical volatility (annualized or rolling)
        current_vol: Current realized volatility

    Returns:
        MarketContext ready for LLM prompt
    """
    # Recent performance
    win_rate = 0.0
    avg_pnl = 0.0
    n_trades = 0
    if recent_trades:
        n_trades = len(recent_trades)
        pnls = [t.pnl_pct for t in recent_trades]
        win_rate = sum(1 for p in pnls if p > 0) / n_trades if n_trades > 0 else 0
        avg_pnl = sum(pnls) / n_trades if n_trades > 0 else 0

    # Volatility percentile
    vol_pct = 0.5  # default
    if historical_vol and current_vol and historical_vol > 0:
        vol_pct = min(current_vol / historical_vol, 1.0)

    return MarketContext(
        instrument=instrument,
        timeframe=timeframe,
        current_price=current_price,
        point_forecast=quantile_levels.point_forecast,
        forecast_horizon_candles=pred_len,
        directional_confidence=quantile_levels.directional_confidence,
        stop_loss=quantile_levels.stop_loss,
        take_profit=quantile_levels.take_profit,
        confidence_lo=quantile_levels.confidence_lo,
        confidence_hi=quantile_levels.confidence_hi,
        risk_reward_ratio=quantile_levels.risk_reward_ratio,
        interval_width=quantile_levels.interval_width,
        volatility_percentile=vol_pct,
        path_dispersion=forecast_paths.path_dispersion(),
        trend_strength=forecast_paths.trend_strength(),
        recent_win_rate=win_rate,
        recent_avg_pnl=avg_pnl,
        recent_n_trades=n_trades,
    )

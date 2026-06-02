"""Strategy performance metrics for evaluation."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class PerformanceReport:
    """Complete performance report for a strategy."""
    # Returns
    total_return: float
    annualized_return: float

    # Risk
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # Trade stats
    n_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float

    # Other
    tail_ratio: float

    def __repr__(self) -> str:
        return (
            f"Performance Report:\n"
            f"  Total Return:     {self.total_return:+.2%}\n"
            f"  Ann. Return:      {self.annualized_return:+.2%}\n"
            f"  Sharpe Ratio:     {self.sharpe_ratio:.2f}\n"
            f"  Max Drawdown:     {self.max_drawdown:.2%}\n"
            f"  Calmar Ratio:     {self.calmar_ratio:.2f}\n"
            f"  Trades:           {self.n_trades}\n"
            f"  Win Rate:         {self.win_rate:.1%}\n"
            f"  Profit Factor:    {self.profit_factor:.2f}\n"
            f"  Avg Win:          {self.avg_win:+.2%}\n"
            f"  Avg Loss:         {self.avg_loss:+.2%}\n"
            f"  Tail Ratio:       {self.tail_ratio:.2f}"
        )


def compute_sharpe(returns: np.ndarray, periods_per_year: int = 252, risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio."""
    excess = returns - risk_free / periods_per_year
    if np.std(excess) == 0:
        return 0.0
    return float(np.mean(excess) / np.std(excess) * np.sqrt(periods_per_year))


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum drawdown from peak."""
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return float(np.min(drawdown))


def compute_profit_factor(trade_pnls: np.ndarray) -> float:
    """Gross profit / gross loss."""
    wins = trade_pnls[trade_pnls > 0].sum()
    losses = abs(trade_pnls[trade_pnls < 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def compute_tail_ratio(returns: np.ndarray) -> float:
    """95th percentile gain / abs(5th percentile loss)."""
    p95 = np.percentile(returns, 95)
    p5 = abs(np.percentile(returns, 5))
    if p5 == 0:
        return float("inf") if p95 > 0 else 0.0
    return float(p95 / p5)


def compute_calmar(annualized_return: float, max_drawdown: float) -> float:
    """Calmar ratio: annualized return / max drawdown."""
    if max_drawdown == 0:
        return 0.0
    return float(annualized_return / abs(max_drawdown))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_returns: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado).

    Adjusts for multiple testing: given N strategy trials,
    what's the probability the best Sharpe is due to luck?

    Args:
        observed_sharpe: Best observed Sharpe ratio
        n_trials: Number of strategies tested
        n_returns: Number of return observations
        skewness: Return distribution skewness
        kurtosis: Return distribution kurtosis

    Returns:
        p-value. Low = likely genuine. High = likely luck.
    """
    from scipy import stats

    # Expected maximum Sharpe under null (all strategies have Sharpe=0)
    e_max_sharpe = stats.norm.ppf(1 - 1 / n_trials) if n_trials > 1 else 0

    # Standard error of Sharpe estimate
    se = np.sqrt(
        (1 + 0.5 * observed_sharpe**2 - skewness * observed_sharpe
         + ((kurtosis - 3) / 4) * observed_sharpe**2) / n_returns
    )

    if se == 0:
        return 1.0

    # Test statistic
    z = (observed_sharpe - e_max_sharpe) / se

    # One-sided p-value
    return float(1 - stats.norm.cdf(z))


def evaluate_trades(
    trade_pnls: np.ndarray,
    equity_curve: np.ndarray,
    periods_per_year: int = 252,
    n_trials: int = 1,
) -> PerformanceReport:
    """
    Compute full performance report from trade P&Ls and equity curve.

    Args:
        trade_pnls: Array of per-trade P&L as fractions (0.05 = 5% gain)
        equity_curve: Cumulative equity curve (starting at initial capital)
        periods_per_year: For annualization
        n_trials: Number of strategies tested (for deflated Sharpe)

    Returns:
        PerformanceReport
    """
    if len(trade_pnls) == 0:
        return PerformanceReport(
            total_return=0.0, annualized_return=0.0,
            sharpe_ratio=0.0, max_drawdown=0.0, calmar_ratio=0.0,
            n_trades=0, win_rate=0.0, profit_factor=0.0,
            avg_win=0.0, avg_loss=0.0, tail_ratio=0.0,
        )

    returns = np.diff(equity_curve) / equity_curve[:-1]

    total_return = (equity_curve[-1] / equity_curve[0]) - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    sharpe = compute_sharpe(returns, periods_per_year)
    max_dd = compute_max_drawdown(equity_curve)

    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls < 0]

    return PerformanceReport(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        calmar_ratio=compute_calmar(annualized_return, max_dd),
        n_trades=len(trade_pnls),
        win_rate=float(len(wins) / len(trade_pnls)) if len(trade_pnls) > 0 else 0,
        profit_factor=compute_profit_factor(trade_pnls),
        avg_win=float(np.mean(wins)) if len(wins) > 0 else 0,
        avg_loss=float(np.mean(losses)) if len(losses) > 0 else 0,
        tail_ratio=compute_tail_ratio(returns),
    )

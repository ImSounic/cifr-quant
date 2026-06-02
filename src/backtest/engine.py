"""Walk-forward backtest engine."""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Optional
from tqdm import tqdm

from src.backtest.costs import CostModel
from src.backtest.metrics import evaluate_trades


@dataclass
class Trade:
    """Record of a single trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str              # 'long' or 'short'
    entry_price: float
    exit_price: float
    units: float
    pnl: float                  # Absolute P&L
    pnl_pct: float              # P&L as fraction
    exit_reason: str            # 'tp', 'sl', 'timeout', 'signal'


@dataclass
class BacktestResult:
    """Complete backtest results."""
    trades: list[Trade]
    equity_curve: np.ndarray
    timestamps: list[pd.Timestamp]
    initial_capital: float
    final_capital: float

    @property
    def trade_pnls(self) -> np.ndarray:
        return np.array([t.pnl_pct for t in self.trades])

    def report(self, periods_per_year: int = 252, n_trials: int = 1):
        """Generate performance report."""
        return evaluate_trades(
            self.trade_pnls,
            self.equity_curve,
            periods_per_year=periods_per_year,
            n_trials=n_trials,
        )


def run_backtest(
    data: pd.DataFrame,
    signal_fn: Callable,
    cost_model: CostModel,
    initial_capital: float = 100_000.0,
    max_drawdown_halt: float = 0.15,
    verbose: bool = True,
) -> BacktestResult:
    """
    Walk-forward backtest engine.

    At each timestep, calls signal_fn(data_so_far, capital, position)
    which returns a dict with keys:
        - 'action': 'buy', 'sell', 'close', or 'hold'
        - 'units': number of units (for buy/sell)
        - 'stop_loss': SL price
        - 'take_profit': TP price

    Args:
        data: Full OHLCV DataFrame with timestamps
        signal_fn: Strategy function that produces signals
        cost_model: Transaction cost model
        initial_capital: Starting capital
        max_drawdown_halt: Stop trading if drawdown exceeds this
        verbose: Show progress

    Returns:
        BacktestResult with trades and equity curve
    """
    capital = initial_capital
    position = None  # None or dict with direction, units, entry_price, sl, tp
    trades = []
    equity = [capital]
    timestamps = [data["timestamps"].iloc[0]]
    halted = False

    iterator = range(1, len(data))
    if verbose:
        iterator = tqdm(iterator, desc="Backtesting")

    for i in iterator:
        row = data.iloc[i]
        current_price = row["close"]
        current_time = row["timestamps"]

        # Check drawdown halt
        peak_equity = max(equity)
        current_dd = (peak_equity - capital) / peak_equity
        if current_dd > max_drawdown_halt and not halted:
            halted = True
            if verbose:
                print(f"\nHALTED: Drawdown {current_dd:.1%} exceeded {max_drawdown_halt:.1%}")

        # Check stop loss / take profit on existing position
        if position is not None:
            hit_sl = False
            hit_tp = False

            if position["direction"] == "long":
                if row["low"] <= position["sl"]:
                    hit_sl = True
                    exit_price = cost_model.apply_exit(position["sl"], "long")
                elif row["high"] >= position["tp"]:
                    hit_tp = True
                    exit_price = cost_model.apply_exit(position["tp"], "long")
            else:  # short
                if row["high"] >= position["sl"]:
                    hit_sl = True
                    exit_price = cost_model.apply_exit(position["sl"], "short")
                elif row["low"] <= position["tp"]:
                    hit_tp = True
                    exit_price = cost_model.apply_exit(position["tp"], "short")

            if hit_sl or hit_tp:
                # Close position
                if position["direction"] == "long":
                    pnl = (exit_price - position["entry_price"]) * position["units"]
                else:
                    pnl = (position["entry_price"] - exit_price) * position["units"]

                pnl_pct = pnl / capital
                capital += pnl

                trades.append(Trade(
                    entry_time=position["entry_time"],
                    exit_time=current_time,
                    direction=position["direction"],
                    entry_price=position["entry_price"],
                    exit_price=exit_price,
                    units=position["units"],
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    exit_reason="sl" if hit_sl else "tp",
                ))
                position = None

        # Get signal (only if no position and not halted)
        if position is None and not halted:
            data_so_far = data.iloc[:i + 1]
            signal = signal_fn(data_so_far, capital, position)

            if signal and signal.get("action") in ("buy", "sell"):
                direction = "long" if signal["action"] == "buy" else "short"
                entry_price = cost_model.apply_entry(current_price, direction)

                position = {
                    "direction": direction,
                    "units": signal.get("units", 0),
                    "entry_price": entry_price,
                    "entry_time": current_time,
                    "sl": signal.get("stop_loss", 0),
                    "tp": signal.get("take_profit", 0),
                }

        # Track equity (mark-to-market)
        mtm = capital
        if position is not None:
            if position["direction"] == "long":
                mtm += (current_price - position["entry_price"]) * position["units"]
            else:
                mtm += (position["entry_price"] - current_price) * position["units"]

        equity.append(mtm)
        timestamps.append(current_time)

    # Close any remaining position at end
    if position is not None:
        final_price = cost_model.apply_exit(data["close"].iloc[-1], position["direction"])
        if position["direction"] == "long":
            pnl = (final_price - position["entry_price"]) * position["units"]
        else:
            pnl = (position["entry_price"] - final_price) * position["units"]

        capital += pnl
        trades.append(Trade(
            entry_time=position["entry_time"],
            exit_time=data["timestamps"].iloc[-1],
            direction=position["direction"],
            entry_price=position["entry_price"],
            exit_price=final_price,
            units=position["units"],
            pnl=pnl,
            pnl_pct=pnl / capital,
            exit_reason="timeout",
        ))
        equity[-1] = capital

    return BacktestResult(
        trades=trades,
        equity_curve=np.array(equity),
        timestamps=timestamps,
        initial_capital=initial_capital,
        final_capital=capital,
    )

"""Joint multi-asset walk-forward portfolio backtest (pluggable strategy).

For one market, all assets share a single capital pool. At each rebalance point
the engine:

  1. locates a `lookback` context + `pred_len` future window per asset,
  2. looks up the asset's CACHED Kronos forecast for that rebalance time,
  3. (optionally) classifies the regime from the context,
  4. asks the STRATEGY for trade intents (direction, stop, target, conviction),
  5. asks the SIZER to allocate capital across intents,
  6. simulates each position over the future window (intrabar SL/TP, else
     timeout) applying transaction costs,
  7. realises P&L, updates capital, applies the drawdown halt, records equity.

The forecast layer is decoupled (see `src/model/forecast_cache.py`) so strategy
A/B is CPU-only. STRATEGY and SIZER are injected — the baseline
`DirectionalMomentum` + `InverseWidthRiskParity` reproduces the original loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest.costs import CostModel
from src.backtest.grid import compute_rebalance_grid, compute_test_window, locate_context
from src.backtest.strategy_api import (AssetDecision, ForecastBundle, Sizer,
                                       Strategy, TradeIntent)


@dataclass
class PortfolioTrade:
    rebalance_time: pd.Timestamp
    symbol: str
    direction: str
    confidence: float
    entry_price: float        # cost-adjusted fill
    exit_price: float         # cost-adjusted fill
    position_usd: float
    weight: float
    pnl: float
    pnl_pct: float
    exit_reason: str          # 'tp' | 'sl' | 'timeout'


@dataclass
class MarketBacktestResult:
    market: str
    equity: pd.Series
    trades: List[PortfolioTrade]
    initial_capital: float
    final_capital: float
    n_rebalances: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    strategy_name: str = ""
    sizer_name: str = ""
    per_asset: Dict[str, dict] = field(default_factory=dict)


def _simulate_position(direction, entry_raw, sl, tp, future, cost_model, position_usd):
    """Simulate one position over the future window. SL is checked before TP
    within each bar (conservative). Returns (pnl, pnl_pct, exit_price, reason)."""
    entry = cost_model.apply_entry(entry_raw, direction)
    units = position_usd / entry_raw if entry_raw > 0 else 0.0

    exit_price = None
    reason = "timeout"
    for _, fc in future.iterrows():
        if direction == "long":
            if fc["low"] <= sl:
                exit_price = cost_model.apply_exit(sl, "long"); reason = "sl"; break
            if fc["high"] >= tp:
                exit_price = cost_model.apply_exit(tp, "long"); reason = "tp"; break
        else:  # short
            if fc["high"] >= sl:
                exit_price = cost_model.apply_exit(sl, "short"); reason = "sl"; break
            if fc["low"] <= tp:
                exit_price = cost_model.apply_exit(tp, "short"); reason = "tp"; break

    if exit_price is None:
        exit_price = cost_model.apply_exit(future["close"].iloc[-1], direction)
        reason = "timeout"

    if direction == "long":
        pnl = (exit_price - entry) * units
    else:
        pnl = (entry - exit_price) * units
    pnl_pct = pnl / position_usd if position_usd > 0 else 0.0
    return pnl, pnl_pct, exit_price, reason


def run_market_backtest(
    market: str,
    asset_dfs: Dict[str, pd.DataFrame],
    asset_pred_len: Dict[str, int],
    cost_model: CostModel,
    *,
    forecasts: Dict[Tuple[str, pd.Timestamp], ForecastBundle],
    corrections: Dict[str, float],
    strategy: Strategy,
    sizer: Sizer,
    regime_classifier=None,
    lookback: int = 512,
    step_size: Optional[int] = None,
    test_days: int = 90,
    initial_capital: float = 100_000.0,
    max_drawdown_halt: float = 0.25,
    verbose: bool = True,
) -> MarketBacktestResult:
    """Run the joint walk-forward backtest for one market off cached forecasts."""
    symbols = list(asset_dfs.keys())
    pred_len = max(asset_pred_len[s] for s in symbols)
    step = step_size or pred_len

    test_start, test_end = compute_test_window(asset_dfs, test_days)
    rebalance_times, ref_sym = compute_rebalance_grid(
        asset_dfs, pred_len, step, test_start, test_end)

    if verbose:
        print(f"\n[{market}] test {test_start.date()}..{test_end.date()}  ref={ref_sym}  "
              f"rebalances={len(rebalance_times)}  assets={len(symbols)}  "
              f"pred_len={pred_len} step={step}  strat={strategy.name} sizer={sizer.name}",
              flush=True)

    capital = initial_capital
    peak = capital
    halted = False
    equity_times, equity_vals = [], []
    trades: List[PortfolioTrade] = []
    per_asset = {s: {"trades": 0, "pnl": 0.0, "wins": 0} for s in symbols}

    for r, t in enumerate(rebalance_times):
        if halted:
            equity_times.append(t); equity_vals.append(capital)
            continue

        # 1-3. Build per-asset decisions from cached forecasts (+ optional regime).
        decisions: List[AssetDecision] = []
        futures: Dict[str, pd.DataFrame] = {}
        for s in symbols:
            pl = asset_pred_len[s]
            context, future = locate_context(asset_dfs[s], t, lookback, pl)
            if context is None:
                continue
            fb = forecasts.get((s, pd.Timestamp(t)))
            if fb is None:
                continue
            fb = fb.with_correction(corrections.get(s, 0.0))
            regime = regime_classifier.classify(context) if regime_classifier else None
            decisions.append(AssetDecision(symbol=s, context_df=context,
                                           forecast=fb, regime=regime))
            futures[s] = future

        # 4. Strategy -> intents.
        intents = strategy.generate_intents(decisions)
        if not intents:
            equity_times.append(t); equity_vals.append(capital)
            continue

        # 5. Sizer -> capital allocation.
        sizes = sizer.size(intents, capital)

        # 6-7. Simulate & realise.
        window_pnl = 0.0
        for it in intents:
            pos_usd = float(sizes.get(it.symbol, 0.0))
            if pos_usd <= 0:
                continue
            pnl, pnl_pct, exit_price, reason = _simulate_position(
                it.direction, it.entry_ref, it.stop, it.target,
                futures[it.symbol], cost_model, pos_usd)
            window_pnl += pnl
            weight = pos_usd / capital if capital > 0 else 0.0
            trades.append(PortfolioTrade(
                rebalance_time=t, symbol=it.symbol, direction=it.direction,
                confidence=float(it.meta.get("confidence", 0.0)),
                entry_price=cost_model.apply_entry(it.entry_ref, it.direction),
                exit_price=exit_price, position_usd=pos_usd, weight=weight,
                pnl=pnl, pnl_pct=pnl_pct, exit_reason=reason))
            pa = per_asset[it.symbol]
            pa["trades"] += 1; pa["pnl"] += pnl; pa["wins"] += int(pnl > 0)

        capital += window_pnl
        peak = max(peak, capital)
        dd = (peak - capital) / peak if peak > 0 else 0.0
        if dd > max_drawdown_halt:
            halted = True
            if verbose:
                print(f"    HALTED at {t}: drawdown {dd:.1%} > {max_drawdown_halt:.0%}",
                      flush=True)

        equity_times.append(t); equity_vals.append(capital)
        if verbose and (r % 10 == 0 or r == len(rebalance_times) - 1):
            print(f"    [{r+1}/{len(rebalance_times)}] {t}  open={len(intents)}  "
                  f"cap=${capital:,.0f}", flush=True)

    equity = pd.Series(equity_vals, index=pd.DatetimeIndex(equity_times))
    return MarketBacktestResult(
        market=market, equity=equity, trades=trades,
        initial_capital=initial_capital, final_capital=capital,
        n_rebalances=len(rebalance_times),
        test_start=test_start, test_end=test_end,
        strategy_name=strategy.name, sizer_name=sizer.name, per_asset=per_asset,
    )

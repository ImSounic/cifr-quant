"""Joint multi-asset walk-forward portfolio backtest.

For one market (crypto or commodity), all assets share a single capital pool.
At each rebalance point (non-overlapping `pred_len` windows) the engine:

  1. builds a `lookback`-candle context ending at the rebalance time for every
     asset that has data there,
  2. runs the market ensemble to get CQR-widened q05/q95 close bands plus a
     directional signal + confidence,
  3. gates assets by `min_confidence`,
  4. sizes the survivors by inverse-interval-width risk parity (cap
     `max_position_pct`), sharing the market's current capital,
  5. simulates each position over the next `pred_len` candles with intrabar
     SL/TP checks (timeout-close otherwise), applying transaction costs,
  6. realizes P&L, updates capital, records the equity point.

Long & short are both supported. SL/TP are derived from the CQR-widened close
band per direction (long: TP=upper, SL=lower; short: flipped).

The combined cross-market top-line is assembled by the driver script, which
reindexes each market's per-rebalance equity onto a common daily calendar and
sums them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional

from src.backtest.costs import CostModel


@dataclass
class PortfolioTrade:
    rebalance_time: pd.Timestamp
    symbol: str
    direction: str            # 'long' or 'short'
    confidence: float
    entry_price: float        # cost-adjusted fill
    exit_price: float         # cost-adjusted fill
    position_usd: float
    weight: float
    pnl: float                # absolute P&L (USD)
    pnl_pct: float            # P&L as fraction of the position's notional
    exit_reason: str          # 'tp', 'sl', 'timeout'


@dataclass
class MarketBacktestResult:
    market: str
    equity: pd.Series                 # indexed by rebalance timestamp
    trades: List[PortfolioTrade]
    initial_capital: float
    final_capital: float
    n_rebalances: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    per_asset: Dict[str, dict] = field(default_factory=dict)


def _locate_context(df: pd.DataFrame, t: pd.Timestamp, lookback: int, pred_len: int):
    """Return (context_df, future_df) for rebalance time `t`, or (None, None) if
    the asset lacks a full lookback+pred_len window around `t`."""
    ts = df["timestamps"].values
    # index of last candle with timestamp <= t
    idx = int(np.searchsorted(ts, np.datetime64(t), side="right") - 1)
    if idx < 0:
        return None, None
    # require the located candle to actually be at (or essentially at) t
    ctx_start = idx - lookback + 1
    if ctx_start < 0 or idx + pred_len >= len(df):
        return None, None
    context = df.iloc[ctx_start:idx + 1]
    future = df.iloc[idx + 1:idx + 1 + pred_len]
    if len(context) < lookback or len(future) < pred_len:
        return None, None
    return context, future


def _simulate_position(direction, entry_raw, sl, tp, future, cost_model, position_usd):
    """Simulate one position over the future window. Returns (pnl, pnl_pct,
    exit_price, exit_reason)."""
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
    ensemble,
    asset_dfs: Dict[str, pd.DataFrame],
    asset_pred_len: Dict[str, int],
    corrections: Dict[str, float],
    cost_model: CostModel,
    *,
    lookback: int = 512,
    n_paths: int = 30,
    step_size: Optional[int] = None,
    test_days: int = 90,
    min_confidence: float = 0.55,
    max_position_pct: float = 0.10,
    initial_capital: float = 100_000.0,
    max_drawdown_halt: float = 0.25,
    verbose: bool = True,
) -> MarketBacktestResult:
    """Run the joint walk-forward backtest for one market."""
    symbols = list(asset_dfs.keys())
    # All assets in a market share pred_len in our universes; use the max to be safe.
    pred_len = max(asset_pred_len[s] for s in symbols)
    step = step_size or pred_len

    # Common test window: ends at the latest timestamp across assets.
    test_end = max(df["timestamps"].max() for df in asset_dfs.values())
    test_start = test_end - timedelta(days=test_days)

    # Reference clock = asset with the most candles inside the window.
    def _n_in_window(df):
        return int(((df["timestamps"] >= test_start) & (df["timestamps"] <= test_end)).sum())
    ref_sym = max(symbols, key=lambda s: _n_in_window(asset_dfs[s]))
    ref_df = asset_dfs[ref_sym]
    ref_win = ref_df[(ref_df["timestamps"] >= test_start) &
                     (ref_df["timestamps"] <= test_end)].reset_index(drop=True)
    rebalance_times = [ref_win["timestamps"].iloc[i]
                       for i in range(0, len(ref_win) - pred_len, step)]

    if verbose:
        print(f"\n[{market}] test {test_start.date()}..{test_end.date()}  "
              f"ref={ref_sym}  rebalances={len(rebalance_times)}  "
              f"assets={len(symbols)}  pred_len={pred_len} step={step}", flush=True)

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

        # 1-3. Gather gated signals across assets.
        candidates = []  # dicts: symbol, direction, conf, lower, upper, entry, width, future
        for s in symbols:
            pl = asset_pred_len[s]
            context, future = _locate_context(asset_dfs[s], t, lookback, pl)
            if context is None:
                continue
            try:
                res = ensemble.predict_with_quantiles(
                    df=context[["open", "high", "low", "close", "volume", "amount"]],
                    x_timestamp=context["timestamps"],
                    y_timestamp=future["timestamps"],
                    pred_len=pl, n_paths=n_paths,
                )
            except Exception as e:
                if verbose:
                    print(f"    {t} {s} predict failed: {e}", flush=True)
                continue

            corr = corrections.get(s, 0.0)
            entry = float(res["entry_price"])
            lower = float(res["close_q05"][-1]) - corr   # CQR-widened
            upper = float(res["close_q95"][-1]) + corr
            width = (upper - lower) / entry if entry > 0 else np.inf
            conf = float(res["directional_confidence"])
            direction = res["direction"]
            if conf < min_confidence or direction not in ("long", "short"):
                continue
            candidates.append(dict(symbol=s, direction=direction, conf=conf,
                                   lower=lower, upper=upper, entry=entry,
                                   width=max(width, 1e-4), future=future))

        if not candidates:
            equity_times.append(t); equity_vals.append(capital)
            continue

        # 4. Risk-parity weights (inverse interval width), capped.
        inv = np.array([1.0 / c["width"] for c in candidates])
        raw_w = inv / inv.sum()
        weights = np.minimum(raw_w, max_position_pct)

        # 5-6. Simulate & realize P&L for the window.
        window_pnl = 0.0
        for c, w in zip(candidates, weights):
            pos_usd = float(w) * capital
            if pos_usd <= 0:
                continue
            if c["direction"] == "long":
                sl, tp = c["lower"], c["upper"]
            else:
                sl, tp = c["upper"], c["lower"]
            pnl, pnl_pct, exit_price, reason = _simulate_position(
                c["direction"], c["entry"], sl, tp, c["future"], cost_model, pos_usd)
            window_pnl += pnl
            trades.append(PortfolioTrade(
                rebalance_time=t, symbol=c["symbol"], direction=c["direction"],
                confidence=c["conf"], entry_price=cost_model.apply_entry(c["entry"], c["direction"]),
                exit_price=exit_price, position_usd=pos_usd, weight=float(w),
                pnl=pnl, pnl_pct=pnl_pct, exit_reason=reason))
            pa = per_asset[c["symbol"]]
            pa["trades"] += 1; pa["pnl"] += pnl; pa["wins"] += int(pnl > 0)

        capital += window_pnl
        peak = max(peak, capital)
        dd = (peak - capital) / peak if peak > 0 else 0.0
        if dd > max_drawdown_halt:
            halted = True
            if verbose:
                print(f"    HALTED at {t}: drawdown {dd:.1%} > {max_drawdown_halt:.0%}", flush=True)

        equity_times.append(t); equity_vals.append(capital)
        if verbose and (r % 10 == 0 or r == len(rebalance_times) - 1):
            print(f"    [{r+1}/{len(rebalance_times)}] {t}  "
                  f"open={len(candidates)}  cap=${capital:,.0f}", flush=True)

    equity = pd.Series(equity_vals, index=pd.DatetimeIndex(equity_times))
    return MarketBacktestResult(
        market=market, equity=equity, trades=trades,
        initial_capital=initial_capital, final_capital=capital,
        n_rebalances=len(rebalance_times),
        test_start=test_start, test_end=test_end, per_asset=per_asset,
    )

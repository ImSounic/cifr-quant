"""Fetch crypto derivatives data from Binance USDT-M futures via ccxt.

Three datasets per asset (all public, no auth):
  - Funding rate history  (8h interval, full history back to listing)
  - Perp 1h OHLCV         (full history — gives correct perp returns + basis vs spot)
  - Open interest history (Binance LIMITATION: only the last ~30 days are served;
                           we fetch what exists and accumulate going forward)

Liquidation history is NOT freely available (the allForceOrders endpoint was
retired; aggregated history is paywalled e.g. Coinglass) — out of scope for now.
"""

import time
from pathlib import Path

import ccxt
import pandas as pd
from tqdm import tqdm


def _exchange(exchange_id: str = "binanceusdm"):
    return getattr(ccxt, exchange_id)({"enableRateLimit": True})


def _perp_symbol(spot_symbol: str) -> str:
    """'BTC/USDT' -> ccxt unified swap symbol 'BTC/USDT:USDT'."""
    return f"{spot_symbol}:USDT"


def fetch_funding_history(
    spot_symbol: str,
    start_date: str = "2022-01-01",
    output_path: Path | None = None,
    exchange_id: str = "binanceusdm",
) -> pd.DataFrame:
    """Funding-rate history (one row per 8h funding event). INCREMENTAL: if the
    output file exists, fetch only events after its last timestamp and append
    (dedup). This matters on OKX, whose API serves only ~3 months — periodic
    re-runs accumulate history that would otherwise be lost."""
    ex = _exchange(exchange_id)
    symbol = _perp_symbol(spot_symbol)
    since = ex.parse8601(f"{start_date}T00:00:00Z")
    now = ex.milliseconds()

    existing = None
    if output_path is not None and output_path.exists():
        existing = pd.read_csv(output_path)
        existing["timestamps"] = pd.to_datetime(existing["timestamps"])
        if not existing.empty:
            since = max(since, int(existing["timestamps"].max().timestamp() * 1000) + 1)

    rows = []
    with tqdm(desc=f"funding {spot_symbol}") as pbar:
        while since < now:
            batch = ex.fetch_funding_rate_history(symbol, since=since, limit=1000)
            if not batch:
                break
            rows.extend(batch)
            new_since = batch[-1]["timestamp"] + 1
            if new_since <= since:
                break
            since = new_since
            pbar.update(len(batch))
            time.sleep(ex.rateLimit / 1000)

    df = pd.DataFrame([{
        "timestamps": pd.to_datetime(r["timestamp"], unit="ms"),
        "funding_rate": float(r["fundingRate"]),
    } for r in rows])
    if existing is not None and not existing.empty:
        df = pd.concat([existing, df], ignore_index=True)
    if not df.empty:
        df = df.drop_duplicates(subset="timestamps").sort_values("timestamps").reset_index(drop=True)
    if output_path and not df.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


def fetch_perp_ohlcv(
    spot_symbol: str,
    timeframe: str = "1h",
    start_date: str = "2022-01-01",
    output_path: Path | None = None,
    exchange_id: str = "binanceusdm",
) -> pd.DataFrame:
    """Perp klines (same schema as spot fetcher: timestamps OHLCV + amount)."""
    ex = _exchange(exchange_id)
    symbol = _perp_symbol(spot_symbol)
    since = ex.parse8601(f"{start_date}T00:00:00Z")
    now = ex.milliseconds()

    candles = []
    with tqdm(desc=f"perp {timeframe} {spot_symbol}") as pbar:
        while since < now:
            batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not batch:
                break
            candles.extend(batch)
            since = batch[-1][0] + 1
            pbar.update(len(batch))
            time.sleep(ex.rateLimit / 1000)

    df = pd.DataFrame(candles, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    if not df.empty:
        df["timestamps"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
        df["amount"] = df["close"] * df["volume"]
        df = df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
        df = df.drop_duplicates(subset="timestamps").sort_values("timestamps").reset_index(drop=True)
    if output_path and not df.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


def fetch_oi_history(
    spot_symbol: str,
    timeframe: str = "1h",
    output_path: Path | None = None,
    exchange_id: str = "binanceusdm",
) -> pd.DataFrame:
    """Open-interest history. Binance only serves ~30 days — fetch all of it.
    Re-running APPENDS to an existing file (dedup on timestamp) so history
    accumulates if we run this periodically."""
    ex = _exchange(exchange_id)
    symbol = _perp_symbol(spot_symbol)
    since = ex.milliseconds() - 30 * 24 * 3600 * 1000  # 30d back (API limit)

    rows = []
    with tqdm(desc=f"OI {spot_symbol}") as pbar:
        while True:
            batch = ex.fetch_open_interest_history(symbol, timeframe, since=since, limit=500)
            if not batch:
                break
            rows.extend(batch)
            new_since = batch[-1]["timestamp"] + 1
            if new_since <= since:
                break
            since = new_since
            pbar.update(len(batch))
            time.sleep(ex.rateLimit / 1000)
            if len(batch) < 500:
                break

    df = pd.DataFrame([{
        "timestamps": pd.to_datetime(r["timestamp"], unit="ms"),
        "open_interest": float(r.get("openInterestAmount") or r.get("openInterestValue") or 0.0),
        "open_interest_value": float(r.get("openInterestValue") or 0.0),
    } for r in rows])

    if output_path:
        if output_path.exists() and not df.empty:
            old = pd.read_csv(output_path)
            old["timestamps"] = pd.to_datetime(old["timestamps"])
            df = pd.concat([old, df], ignore_index=True)
        if not df.empty:
            df = df.drop_duplicates(subset="timestamps").sort_values("timestamps").reset_index(drop=True)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
    return df

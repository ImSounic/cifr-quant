"""Fetch BTC/USDT historical OHLCV data from Binance via ccxt."""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import time


def fetch_btc_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "15m",
    start_date: str = "2022-01-01",
    end_date: str | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Binance.

    Binance public API does not require authentication for historical klines.
    Rate limit: 1200 requests/min. We fetch 1000 candles per request.

    Args:
        symbol: Trading pair (default BTC/USDT)
        timeframe: Candle interval (default 15m)
        start_date: Start date as YYYY-MM-DD
        end_date: End date as YYYY-MM-DD (default: now)
        output_path: Optional path to save CSV

    Returns:
        DataFrame with columns: timestamps, open, high, low, close, volume, amount
    """
    exchange = ccxt.binance({"enableRateLimit": True})

    since = exchange.parse8601(f"{start_date}T00:00:00Z")
    end_ts = exchange.parse8601(f"{end_date}T00:00:00Z") if end_date else exchange.milliseconds()

    all_candles = []
    current = since

    print(f"Fetching {symbol} {timeframe} from {start_date} to {end_date or 'now'}...")

    with tqdm(desc="Downloading") as pbar:
        while current < end_ts:
            try:
                candles = exchange.fetch_ohlcv(
                    symbol, timeframe, since=current, limit=1000
                )
                if not candles:
                    break

                all_candles.extend(candles)
                current = candles[-1][0] + 1  # Next ms after last candle
                pbar.update(len(candles))

                # Respect rate limits
                time.sleep(exchange.rateLimit / 1000)

            except ccxt.RateLimitExceeded:
                print("Rate limited, waiting 60s...")
                time.sleep(60)
            except Exception as e:
                print(f"Error: {e}, retrying in 10s...")
                time.sleep(10)

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp_ms", "open", "high", "low", "close", "volume"]
    )

    # Convert timestamp
    df["timestamps"] = pd.to_datetime(df["timestamp_ms"], unit="ms")

    # Compute amount (trade value = close * volume, approximate)
    df["amount"] = df["close"] * df["volume"]

    # Select and order columns for Kronos compatibility
    df = df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]

    # Remove duplicates
    df = df.drop_duplicates(subset="timestamps").sort_values("timestamps").reset_index(drop=True)

    print(f"Downloaded {len(df)} candles from {df['timestamps'].iloc[0]} to {df['timestamps'].iloc[-1]}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    from configs.base_config import DATA_RAW_DIR

    fetch_btc_ohlcv(
        start_date="2022-01-01",
        output_path=DATA_RAW_DIR / "btc" / "btc_usdt_15m.csv",
    )

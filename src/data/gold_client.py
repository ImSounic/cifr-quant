"""Fetch XAU/USD (Gold) historical OHLCV data.

Two methods:
1. TwelveData API (preferred) — supports 4h interval natively, 5+ years history
2. yfinance fallback — only ~730 days of intraday data
"""

import os
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
import time


def fetch_gold_twelvedata(
    symbol: str = "XAU/USD",
    interval: str = "4h",
    start_date: str = "2019-01-01",
    end_date: str | None = None,
    output_path: Path | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch gold OHLCV data from TwelveData API.

    TwelveData supports 4h interval natively and provides 5+ years
    of historical data. Free tier: 800 requests/day, 5000 points/request.

    Args:
        symbol: TwelveData symbol (XAU/USD for gold spot)
        interval: Candle interval (4h supported natively)
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD (default: now)
        output_path: Optional path to save CSV
        api_key: TwelveData API key (defaults to TWELVEDATA_API_KEY env var)

    Returns:
        DataFrame with Kronos-compatible columns
    """
    key = api_key or os.environ.get("TWELVEDATA_API_KEY")
    if not key:
        # Try loading from .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
            key = os.environ.get("TWELVEDATA_API_KEY")
        except ImportError:
            pass

    if not key:
        raise ValueError(
            "TwelveData API key required. Set TWELVEDATA_API_KEY in .env or pass api_key=..."
        )

    end = end_date or datetime.now().strftime("%Y-%m-%d")
    base_url = "https://api.twelvedata.com/time_series"

    all_data = []
    current_end = end

    print(f"Fetching {symbol} {interval} from {start_date} to {end} via TwelveData...")

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "start_date": start_date,
            "end_date": current_end,
            "outputsize": 5000,
            "apikey": key,
            "format": "JSON",
        }

        response = requests.get(base_url, params=params)
        data = response.json()

        if "code" in data and data["code"] != 200:
            print(f"  API error: {data.get('message', 'Unknown error')}")
            break

        if "values" not in data or not data["values"]:
            print("  No more data available")
            break

        values = data["values"]
        all_data.extend(values)
        print(f"  Fetched {len(values)} candles (oldest: {values[-1]['datetime']})")

        if len(values) < 5000:
            break  # No more data

        # Next batch: end before the oldest candle we just got
        oldest = pd.Timestamp(values[-1]["datetime"])
        current_end = (oldest - pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")

        if pd.Timestamp(current_end) < pd.Timestamp(start_date):
            break

        # Rate limit: free tier is 8 req/min
        time.sleep(8)

    if not all_data:
        raise ValueError(f"No data fetched for {symbol}")

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    df["timestamps"] = pd.to_datetime(df["datetime"])
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)

    # XAU/USD spot has no real volume — set to 0
    df["volume"] = 0.0
    df["amount"] = 0.0

    df = df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
    df = df.drop_duplicates(subset="timestamps").sort_values("timestamps").reset_index(drop=True)

    # Remove weekends
    df = df[df["timestamps"].dt.dayofweek < 5]
    df = df.reset_index(drop=True)

    print(f"Total: {len(df)} candles from {df['timestamps'].iloc[0]} to {df['timestamps'].iloc[-1]}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    return df


def fetch_gold_yfinance(
    symbol: str = "GC=F",
    start_date: str = "2019-01-01",
    end_date: str | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Fallback: Fetch gold data from yfinance (limited to ~730 days intraday).

    Fetches 1h data and resamples to 4h. Uses GC=F (Gold Futures)
    which has real volume data.
    """
    import yfinance as yf

    print(f"Fetching {symbol} 1h from {start_date} via yfinance (fallback)...")
    print("  WARNING: yfinance limits intraday to ~730 days")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()

    chunks = []
    current_start = start
    chunk_days = 59

    while current_start < end:
        current_end = min(current_start + pd.Timedelta(days=chunk_days), end)
        try:
            ticker = yf.Ticker(symbol)
            chunk = ticker.history(
                start=current_start.strftime("%Y-%m-%d"),
                end=current_end.strftime("%Y-%m-%d"),
                interval="1h",
            )
            if not chunk.empty:
                chunks.append(chunk)
                print(f"  Fetched {len(chunk)} candles: {chunk.index[0]} to {chunk.index[-1]}")
        except Exception as e:
            print(f"  Error: {e}")
        current_start = current_end

    if not chunks:
        raise ValueError(f"No data fetched for {symbol}")

    df = pd.concat(chunks).reset_index()
    df = df.rename(columns={
        "Date": "timestamps", "Datetime": "timestamps",
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    if "timestamps" not in df.columns:
        df["timestamps"] = df.index
    df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)

    # Resample 1h → 4h
    df = df.set_index("timestamps")
    resampled = df.resample("4h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    resampled["amount"] = resampled["close"] * resampled["volume"]
    resampled = resampled.reset_index()
    resampled = resampled[resampled["timestamps"].dt.dayofweek < 5]
    resampled = resampled[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
    resampled = resampled.drop_duplicates(subset="timestamps").sort_values("timestamps").reset_index(drop=True)

    print(f"Total after resample: {len(resampled)} candles")
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resampled.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
    return resampled


if __name__ == "__main__":
    from configs.base_config import DATA_RAW_DIR

    # Try TwelveData first (5+ years), fall back to yfinance
    try:
        fetch_gold_twelvedata(
            start_date="2019-01-01",
            output_path=DATA_RAW_DIR / "xau" / "gold_4h.csv",
        )
    except (ValueError, Exception) as e:
        print(f"\nTwelveData failed: {e}")
        print("Falling back to yfinance...\n")
        fetch_gold_yfinance(
            start_date="2019-01-01",
            output_path=DATA_RAW_DIR / "xau" / "gold_4h.csv",
        )

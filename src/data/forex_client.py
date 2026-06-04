"""Fetch EUR/USD historical OHLCV data from yfinance (free, no API key)."""

import yfinance as yf
import pandas as pd
from pathlib import Path


def fetch_eurusd_ohlcv(
    symbol: str = "EURUSD=X",
    interval: str = "1h",
    start_date: str = "2021-01-01",
    end_date: str | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Fetch EUR/USD historical OHLCV data from yfinance.

    yfinance limitations for intraday data:
    - 1h data: max ~730 days per request
    - We fetch in chunks and concatenate

    Note: Forex spot has no real volume. Volume column from yfinance
    is unreliable for forex. We set volume=0 and amount=0 for Kronos
    (Kronos fills missing volume with zeros).

    Args:
        symbol: yfinance ticker (EURUSD=X for EUR/USD spot)
        interval: Candle interval
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD (default: now)
        output_path: Optional path to save CSV

    Returns:
        DataFrame with Kronos-compatible columns
    """
    print(f"Fetching {symbol} {interval} from {start_date} to {end_date or 'now'}...")

    # yfinance limits intraday to ~730 days, fetch in chunks
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()

    chunks = []
    current_start = start
    chunk_days = 59  # yfinance 1h limit per request is ~60 days

    while current_start < end:
        current_end = min(current_start + pd.Timedelta(days=chunk_days), end)

        try:
            ticker = yf.Ticker(symbol)
            chunk = ticker.history(
                start=current_start.strftime("%Y-%m-%d"),
                end=current_end.strftime("%Y-%m-%d"),
                interval=interval,
            )
            if not chunk.empty:
                chunks.append(chunk)
                print(f"  Fetched {len(chunk)} candles: {chunk.index[0]} to {chunk.index[-1]}")
        except Exception as e:
            print(f"  Error fetching {current_start} to {current_end}: {e}")

        current_start = current_end

    if not chunks:
        raise ValueError(f"No data fetched for {symbol}")

    df = pd.concat(chunks)

    # Standardize columns for Kronos
    df = df.reset_index()
    df = df.rename(columns={
        "Date": "timestamps",
        "Datetime": "timestamps",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
    })

    # Handle timezone-aware index
    if "timestamps" not in df.columns:
        df["timestamps"] = df.index

    df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)

    # Forex has no real volume — set to 0 for Kronos
    df["volume"] = 0.0
    df["amount"] = 0.0

    df = df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
    df = df.drop_duplicates(subset="timestamps").sort_values("timestamps").reset_index(drop=True)

    # Remove weekend gaps (Saturday/Sunday candles if any leak through)
    df = df[df["timestamps"].dt.dayofweek < 5]

    print(f"Total: {len(df)} candles from {df['timestamps'].iloc[0]} to {df['timestamps'].iloc[-1]}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    from configs.base_config import DATA_RAW_DIR

    fetch_eurusd_ohlcv(
        start_date="2021-01-01",
        output_path=DATA_RAW_DIR / "eur" / "eurusd_1h.csv",
    )

"""Fetch commodity OHLCV data via yfinance futures symbols.

TwelveData free tier only supports XAU/USD. For all other commodities,
we use yfinance futures symbols which have real volume data.

yfinance intraday limit: ~730 days. We fetch 1h and resample to 4h.
This gives ~1000+ 4h candles — sufficient for inference and backtesting
(we don't finetune on these; the XAU checkpoint transfers).

Usage:
    python scripts/fetch_commodities_yf.py
    python scripts/fetch_commodities_yf.py --symbols SI=F CL=F
"""

import argparse
import pandas as pd
import yfinance as yf
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "commodity"

# yfinance futures symbols → our naming convention
COMMODITY_FUTURES = {
    "SI=F":  {"name": "xag_usd", "desc": "Silver futures"},
    "PL=F":  {"name": "xpt_usd", "desc": "Platinum futures"},
    "CL=F":  {"name": "wti_usd", "desc": "WTI Crude Oil futures"},
    "BZ=F":  {"name": "brent_usd", "desc": "Brent Crude Oil futures"},
    "NG=F":  {"name": "ng_usd", "desc": "Natural Gas futures"},
    "HG=F":  {"name": "copper_usd", "desc": "Copper futures"},
}


def fetch_commodity_yf(symbol: str, name: str, output_dir: Path) -> bool:
    """Fetch 1h data from yfinance and resample to 4h."""
    output_path = output_dir / f"{name}_4h.csv"

    if output_path.exists():
        print(f"  SKIP {symbol} ({name}) — already exists")
        return True

    print(f"  Fetching {symbol} ({name}) 1h via yfinance...")

    try:
        # yfinance max intraday period is 730 days
        # Fetch in 59-day chunks to avoid issues
        ticker = yf.Ticker(symbol)
        end = pd.Timestamp.now()
        start = end - pd.Timedelta(days=729)

        chunks = []
        current_start = start
        chunk_days = 59

        while current_start < end:
            current_end = min(current_start + pd.Timedelta(days=chunk_days), end)
            try:
                chunk = ticker.history(
                    start=current_start.strftime("%Y-%m-%d"),
                    end=current_end.strftime("%Y-%m-%d"),
                    interval="1h",
                )
                if not chunk.empty:
                    chunks.append(chunk)
            except Exception as e:
                print(f"    Chunk error: {e}")
            current_start = current_end
            time.sleep(0.5)

        if not chunks:
            print(f"  ✗ {symbol} — no data returned")
            return False

        df = pd.concat(chunks)
        df = df.reset_index()

        # Normalize column names
        col_map = {}
        for col in df.columns:
            lower = col.lower()
            if lower in ("date", "datetime"):
                col_map[col] = "timestamps"
            elif lower in ("open", "high", "low", "close", "volume"):
                col_map[col] = lower
        df = df.rename(columns=col_map)

        if "timestamps" not in df.columns:
            df["timestamps"] = df.index

        df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)

        # Resample 1h → 4h
        df = df.set_index("timestamps")
        resampled = df.resample("4h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        resampled["amount"] = resampled["close"] * resampled["volume"]
        resampled = resampled.reset_index()

        # Remove weekends
        resampled = resampled[resampled["timestamps"].dt.dayofweek < 5]
        resampled = resampled[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
        resampled = resampled.drop_duplicates(subset="timestamps").sort_values("timestamps").reset_index(drop=True)

        output_dir.mkdir(parents=True, exist_ok=True)
        resampled.to_csv(output_path, index=False)
        print(f"  ✓ {symbol} → {len(resampled)} candles ({resampled['timestamps'].iloc[0].date()} to {resampled['timestamps'].iloc[-1].date()})")
        return True

    except Exception as e:
        print(f"  ✗ {symbol} FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fetch commodity futures via yfinance")
    parser.add_argument("--symbols", nargs="+", default=list(COMMODITY_FUTURES.keys()),
                        help="yfinance symbols to fetch (default: all)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Fetching {len(args.symbols)} commodities via yfinance")
    print(f"{'='*60}\n")

    success = 0
    for symbol in args.symbols:
        if symbol not in COMMODITY_FUTURES:
            print(f"  Unknown symbol: {symbol}")
            continue
        info = COMMODITY_FUTURES[symbol]
        if fetch_commodity_yf(symbol, info["name"], DATA_DIR):
            success += 1
        time.sleep(1)

    print(f"\nDone: {success}/{len(args.symbols)} fetched successfully")


if __name__ == "__main__":
    main()

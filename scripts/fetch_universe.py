"""Fetch OHLCV data for the entire asset universe.

Downloads data for all crypto and commodity assets in parallel-ish fashion,
respecting API rate limits. Saves to data/raw/{market}/{symbol}.csv

Usage:
    python scripts/fetch_universe.py --market crypto --tiers 1 2
    python scripts/fetch_universe.py --market commodity --categories precious energy
    python scripts/fetch_universe.py --market all
"""

import sys
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import DATA_RAW_DIR
from src.data.binance_client import fetch_btc_ohlcv  # Works for any Binance symbol


def fetch_crypto_universe(tiers=(1, 2), start_date="2022-01-01"):
    """Fetch all crypto assets from Binance."""
    from configs.crypto_universe import get_crypto_configs

    configs = get_crypto_configs(tiers=tiers)
    print(f"\n{'='*60}")
    print(f"  Fetching {len(configs)} crypto assets from Binance")
    print(f"{'='*60}\n")

    for symbol, config in configs.items():
        safe_name = symbol.replace("/", "_").lower()
        output_path = DATA_RAW_DIR / "crypto" / f"{safe_name}_15m.csv"

        if output_path.exists():
            print(f"  SKIP {symbol} — already exists at {output_path}")
            continue

        try:
            fetch_btc_ohlcv(
                symbol=symbol,
                timeframe="15m",
                start_date=start_date,
                output_path=output_path,
            )
            print(f"  ✓ {symbol} saved to {output_path}\n")
        except Exception as e:
            print(f"  ✗ {symbol} FAILED: {e}\n")

        time.sleep(2)  # Be nice to the API


def fetch_commodity_universe(categories=("precious", "energy"), start_date="2019-01-01"):
    """Fetch all commodity assets from TwelveData."""
    from configs.commodity_universe import get_commodity_configs
    from src.data.gold_client import fetch_gold_twelvedata

    configs = get_commodity_configs(categories=categories)
    print(f"\n{'='*60}")
    print(f"  Fetching {len(configs)} commodity assets from TwelveData")
    print(f"{'='*60}\n")

    for key, config in configs.items():
        safe_name = config.instrument.replace("/", "_").lower()
        output_path = DATA_RAW_DIR / "commodity" / f"{safe_name}_4h.csv"

        if output_path.exists():
            print(f"  SKIP {config.instrument} — already exists at {output_path}")
            continue

        try:
            fetch_gold_twelvedata(
                symbol=config.instrument,
                interval="4h",
                start_date=start_date,
                output_path=output_path,
            )
            print(f"  ✓ {config.instrument} saved to {output_path}\n")
        except Exception as e:
            print(f"  ✗ {config.instrument} FAILED: {e}\n")

        time.sleep(10)  # TwelveData free tier: 8 req/min


def main():
    parser = argparse.ArgumentParser(description="Fetch universe OHLCV data")
    parser.add_argument("--market", choices=["crypto", "commodity", "all"], default="all")
    parser.add_argument("--tiers", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--categories", nargs="+", default=["precious", "energy"])
    parser.add_argument("--crypto-start", default="2022-01-01")
    parser.add_argument("--commodity-start", default="2019-01-01")
    args = parser.parse_args()

    if args.market in ("crypto", "all"):
        fetch_crypto_universe(tiers=tuple(args.tiers), start_date=args.crypto_start)

    if args.market in ("commodity", "all"):
        fetch_commodity_universe(categories=tuple(args.categories), start_date=args.commodity_start)

    print("\nDone!")


if __name__ == "__main__":
    main()

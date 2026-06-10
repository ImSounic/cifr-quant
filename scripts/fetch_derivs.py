"""Fetch derivatives data (funding, perp klines, OI) for the crypto universe.

Phase 2A of the v2 plan: information-bearing inputs. All Binance USDT-M public
endpoints, no auth. Network I/O only — run directly on the HPC head node (no
sbatch needed), or anywhere with internet.

Saves to data/raw/derivs/:
    {sym}_funding.csv   — 8h funding events (full history)
    {sym}_perp_1h.csv   — perp 1h OHLCV (full history; perp returns + basis)
    {sym}_oi_1h.csv     — open interest (Binance serves last ~30d; re-runs append)

Usage:
    python scripts/fetch_derivs.py                  # tier 1+2, everything
    python scripts/fetch_derivs.py --what funding   # just funding
    python scripts/fetch_derivs.py --symbols BTC/USDT ETH/USDT
"""

import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import DATA_RAW_DIR
from src.data.derivs_client import fetch_funding_history, fetch_perp_ohlcv, fetch_oi_history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--symbols", nargs="+", default=None,
                    help="Explicit symbols (default: full tier selection)")
    ap.add_argument("--what", choices=["all", "funding", "perp", "oi"], default="all")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--exchange", choices=["binanceusdm", "okx"], default="binanceusdm",
                    help="Venue to fetch from; non-Binance saves to data/raw/derivs_<exchange>/")
    args = ap.parse_args()

    if args.symbols:
        symbols = args.symbols
    else:
        from configs.crypto_universe import get_crypto_configs
        symbols = list(get_crypto_configs(tiers=tuple(args.tiers)).keys())

    out_dir = (DATA_RAW_DIR / "derivs" if args.exchange == "binanceusdm"
               else DATA_RAW_DIR / f"derivs_{args.exchange}")
    print(f"Fetching derivs data ({args.exchange}) for {len(symbols)} symbols -> {out_dir}\n",
          flush=True)

    failures = []
    for sym in symbols:
        safe = sym.replace("/", "_").lower()
        print(f"=== {sym} ===", flush=True)

        jobs = []
        if args.what in ("all", "funding"):
            jobs.append(("funding", lambda: fetch_funding_history(
                sym, start_date=args.start, output_path=out_dir / f"{safe}_funding.csv",
                exchange_id=args.exchange),
                out_dir / f"{safe}_funding.csv", False))
        if args.what in ("all", "perp"):
            jobs.append(("perp_1h", lambda: fetch_perp_ohlcv(
                sym, timeframe="1h", start_date=args.start,
                output_path=out_dir / f"{safe}_perp_1h.csv", exchange_id=args.exchange),
                out_dir / f"{safe}_perp_1h.csv", False))
        if args.what in ("all", "oi"):
            # OI always refetches (appends new history; Binance only keeps 30d)
            jobs.append(("oi_1h", lambda: fetch_oi_history(
                sym, timeframe="1h", output_path=out_dir / f"{safe}_oi_1h.csv",
                exchange_id=args.exchange),
                out_dir / f"{safe}_oi_1h.csv", True))

        for name, fn, path, always in jobs:
            if path.exists() and not always:
                print(f"  SKIP {name} — exists ({path.name})", flush=True)
                continue
            try:
                df = fn()
                if df is None or df.empty:
                    print(f"  ! {name}: empty result", flush=True)
                    failures.append((sym, name, "empty"))
                else:
                    print(f"  OK {name}: {len(df)} rows "
                          f"({df['timestamps'].iloc[0]} .. {df['timestamps'].iloc[-1]})", flush=True)
            except Exception as e:
                print(f"  FAIL {name}: {e}", flush=True)
                failures.append((sym, name, str(e)[:120]))
            time.sleep(1)
        print(flush=True)

    if failures:
        print("Failures (delisted symbols like MATIC are expected):", flush=True)
        for sym, name, err in failures:
            print(f"  {sym} {name}: {err}", flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

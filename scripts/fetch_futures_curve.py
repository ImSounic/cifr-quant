"""Fetch commodity futures TERM-STRUCTURE data (contract 1..4 panels).

Two sources, two eras (probed July 18, 2026):

1. EIA HISTORY (default): daily contract-1..4 settlement prices for six energy
   commodities, 1980s..April 2024 — EIA stopped republishing CME settlements
   then, so this is a STATIC 40-year validation panel. Free, no key, .xls files.
   yfinance CANNOT substitute: Yahoo deletes expired contracts (probed: 404 on
   every expired contract; only active ones serve history), so the historical
   front/second curve is unrecoverable there.

2. --snapshot: today's live curve (front 4 contracts by expiry) from yfinance
   ACTIVE contracts, appended to *_chain_snapshots.csv. Run daily via cron to
   accumulate the post-2024 out-of-sample panel going forward — same
   accumulate-it-ourselves pattern as the OKX funding archive.

Saves to data/raw/futures_curve/:
    {key}_c{1..4}.csv           — EIA history (timestamps, price)
    {key}_chain_snapshots.csv   — accumulating live curve (one row/day/contract)

Usage:
    python scripts/fetch_futures_curve.py               # EIA 40y history (rerun-safe)
    python scripts/fetch_futures_curve.py --snapshot    # today's live curve (cron)

Requires: xlrd (pip install xlrd) for the EIA .xls files.
"""

import io
import sys
import time
import argparse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import DATA_RAW_DIR

OUT_DIR = DATA_RAW_DIR / "futures_curve"

# EIA dnav series: {key: (url_dir, series_prefix_fmt, description)}
# xls URL = https://www.eia.gov/dnav/{dir}/hist_xls/{series}d.xls
EIA_UNIVERSE = {
    "wti":      ("pet", "RCLC{n}",                    "WTI crude (NYMEX CL), 1983-"),
    "natgas":   ("ng",  "RNGC{n}",                    "Henry Hub natgas (NYMEX NG), 1994-"),
    "heatoil":  ("pet", "EER_EPD2F_PE{n}_Y35NY_DPG",  "NY Harbor No.2 heating oil (HO), 1980-"),
    "rbob":     ("pet", "EER_EPMRR_PE{n}_Y35NY_DPG",  "RBOB gasoline (RB), 2005-"),
    "gasoline": ("pet", "EER_EPMR_PE{n}_Y35NY_DPG",   "Conv. gasoline (pre-RBOB), 1984-2006"),
    "propane":  ("pet", "EER_EPLLPA_PE{n}_Y44MB_DPG", "Mont Belvieu propane, 1993-"),
}

# yfinance live-chain universe: root, exchange suffix, listed delivery months
# (month codes F..Z = Jan..Dec)
YF_CHAINS = {
    "wti":     ("CL", "NYM", "FGHJKMNQUVXZ"),
    "natgas":  ("NG", "NYM", "FGHJKMNQUVXZ"),
    "heatoil": ("HO", "NYM", "FGHJKMNQUVXZ"),
    "rbob":    ("RB", "NYM", "FGHJKMNQUVXZ"),
    "brent":   ("BZ", "NYM", "FGHJKMNQUVXZ"),
    "gold":    ("GC", "CMX", "GJMQVZ"),
    "silver":  ("SI", "CMX", "FHKNUZ"),
}
MONTH_CODE = {c: i + 1 for i, c in enumerate("FGHJKMNQUVXZ")}


def fetch_eia(key, url_dir, series_fmt):
    ok = 0
    for n in range(1, 5):
        series = series_fmt.format(n=n)
        url = f"https://www.eia.gov/dnav/{url_dir}/hist_xls/{series}d.xls"
        out = OUT_DIR / f"{key}_c{n}.csv"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=60).read()
            df = pd.read_excel(io.BytesIO(raw), sheet_name="Data 1", skiprows=2)
            df.columns = ["timestamps", "price"]
            df["timestamps"] = pd.to_datetime(df["timestamps"])
            df = df.dropna().sort_values("timestamps").reset_index(drop=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            print(f"  OK c{n}: {len(df)} rows ({df['timestamps'].iloc[0].date()} .. "
                  f"{df['timestamps'].iloc[-1].date()})", flush=True)
            ok += 1
        except Exception as e:
            print(f"  FAIL c{n} ({series}): {str(e)[:100]}", flush=True)
        time.sleep(1)
    return ok


def snapshot_chain(key, root, exch, months):
    """Probe active contracts around now, keep the front 4 by expiry, return rows."""
    import yfinance as yf
    now = datetime.now(timezone.utc)
    candidates = []
    y, m = now.year, now.month
    for k in range(0, 10):                       # this month + 9 forward
        mm = (m - 1 + k) % 12 + 1
        yy = y + (m - 1 + k) // 12
        code = "FGHJKMNQUVXZ"[mm - 1]
        if code in months:
            candidates.append((yy, mm, f"{root}{code}{str(yy)[2:]}.{exch}"))

    rows = []
    for yy, mm, tkr in candidates:
        if len(rows) >= 4:
            break
        try:
            h = yf.Ticker(tkr).history(period="5d", interval="1d", auto_adjust=False)
            if h is None or h.empty:
                continue
            last = h.dropna(subset=["Close"]).iloc[-1]
            ts = h.dropna(subset=["Close"]).index[-1]
            # stale quote = contract effectively dead/expired; skip
            if (now - ts.to_pydatetime().astimezone(timezone.utc)).days > 5:
                continue
            rows.append({"snapshot_utc": now.date().isoformat(), "contract": tkr,
                         "delivery": f"{yy}-{mm:02d}", "rank": len(rows) + 1,
                         "close": float(last["Close"]),
                         "volume": float(last.get("Volume") or 0),
                         "quote_date": ts.date().isoformat()})
        except Exception:
            continue
        time.sleep(0.5)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true",
                    help="Append today's live curve from yfinance active contracts")
    ap.add_argument("--keys", nargs="+", default=None)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.snapshot:
        for key, (root, exch, months) in YF_CHAINS.items():
            if args.keys and key not in args.keys:
                continue
            rows = snapshot_chain(key, root, exch, months)
            if not rows:
                print(f"{key}: no live contracts found", flush=True)
                continue
            out = OUT_DIR / f"{key}_chain_snapshots.csv"
            df = pd.DataFrame(rows)
            hdr = not out.exists()
            df.to_csv(out, mode="a", header=hdr, index=False)
            slope = (rows[0]["close"] / rows[1]["close"] - 1) if len(rows) > 1 else float("nan")
            print(f"{key}: {len(rows)} contracts, front {rows[0]['contract']} "
                  f"{rows[0]['close']:.2f}, c1/c2 slope {slope:+.2%}", flush=True)
        return

    for key, (url_dir, series_fmt, desc) in EIA_UNIVERSE.items():
        if args.keys and key not in args.keys:
            continue
        print(f"=== {key} ({desc}) ===", flush=True)
        fetch_eia(key, url_dir, series_fmt)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

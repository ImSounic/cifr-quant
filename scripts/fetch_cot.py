"""Fetch CFTC COT (Commitments of Traders) positioning + daily futures prices.

In plain terms: every week the US regulator publishes who is positioned how in
every futures market — speculators (hedge funds/CTAs) vs commercials (producers
and consumers hedging real goods). Decades of free, official positioning data.
The hypothesis it feeds (cot_skill.py): when speculators are EXTREMELY crowded
on one side, forward returns tend to go the other way — the same fade-the-
crowd mechanism as our validated funding-carry brick, in a different market.

Two datasets per commodity:
  1. COT legacy futures-only via the CFTC Socrata API (publicreporting.cftc.gov,
     dataset 6dca-aqww) — weekly since ~1986, fetched by contract code.
  2. Daily front-futures closes via yfinance (GC=F etc.) — decades of history,
     needed because our 4h price files only cover ~2 years.

Saves to data/raw/cot/{key}_cot.csv and {key}_px_1d.csv. Incremental: re-runs
fetch only new COT rows. Run on HPC head node (network only).

Usage:
    python scripts/fetch_cot.py
"""

import sys
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import DATA_RAW_DIR

COT_DIR = DATA_RAW_DIR / "cot"
API = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"

# key -> (CFTC contract code(s) — older history sometimes sits under a sibling
#         code; we fetch all listed and concat, yfinance front-future ticker)
UNIVERSE = {
    "gold":     (["088691"], "GC=F"),
    "silver":   (["084691"], "SI=F"),
    "platinum": (["076651"], "PL=F"),
    "wti":      (["067651"], "CL=F"),
    "brent":    (["06765T"], "BZ=F"),
    "natgas":   (["023651"], "NG=F"),
}

FIELDS = ["report_date_as_yyyy_mm_dd", "open_interest_all",
          "noncomm_positions_long_all", "noncomm_positions_short_all",
          "comm_positions_long_all", "comm_positions_short_all"]


def fetch_cot(codes, since_date=None):
    rows = []
    for code in codes:
        offset = 0
        while True:
            where = f"cftc_contract_market_code='{code}'"
            if since_date:
                where += f" AND report_date_as_yyyy_mm_dd>'{since_date}'"
            q = (f"{API}?$select={','.join(FIELDS)}&$where={urllib.parse.quote(where)}"
                 f"&$order=report_date_as_yyyy_mm_dd&$limit=10000&$offset={offset}")
            with urllib.request.urlopen(q, timeout=60) as r:
                batch = json.loads(r.read())
            rows.extend(batch)
            if len(batch) < 10000:
                break
            offset += 10000
            time.sleep(0.5)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    for c in FIELDS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (df[["report_date"] + FIELDS[1:]]
          .dropna(subset=["open_interest_all"])
          .drop_duplicates(subset="report_date")
          .sort_values("report_date").reset_index(drop=True))
    return df


def fetch_prices(ticker):
    import yfinance as yf
    px = yf.download(ticker, period="max", interval="1d",
                     progress=False, auto_adjust=False)
    if px is None or px.empty:
        return pd.DataFrame()
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(0)
    out = px[["Close"]].reset_index()
    out.columns = ["timestamps", "close"]
    return out.dropna().sort_values("timestamps").reset_index(drop=True)


def main():
    import urllib.parse  # noqa — used inside fetch_cot via module ref
    COT_DIR.mkdir(parents=True, exist_ok=True)
    for key, (codes, ticker) in UNIVERSE.items():
        cot_path = COT_DIR / f"{key}_cot.csv"
        since = None
        existing = None
        if cot_path.exists():
            existing = pd.read_csv(cot_path)
            existing["report_date"] = pd.to_datetime(existing["report_date"])
            if not existing.empty:
                since = existing["report_date"].max().strftime("%Y-%m-%dT%H:%M:%S")
        df = fetch_cot(codes, since)
        if existing is not None and not existing.empty:
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset="report_date").sort_values("report_date")
        if not df.empty:
            df.to_csv(cot_path, index=False)
            print(f"  OK {key} COT: {len(df)} weeks "
                  f"({df['report_date'].iloc[0].date()} .. {df['report_date'].iloc[-1].date()})",
                  flush=True)
        else:
            print(f"  ! {key} COT: empty", flush=True)

        px_path = COT_DIR / f"{key}_px_1d.csv"
        try:
            px = fetch_prices(ticker)
            if not px.empty:
                px.to_csv(px_path, index=False)
                print(f"  OK {key} px: {len(px)} days "
                      f"({px['timestamps'].iloc[0]} .. {px['timestamps'].iloc[-1]})", flush=True)
            else:
                print(f"  ! {key} px ({ticker}): empty", flush=True)
        except Exception as e:
            print(f"  FAIL {key} px: {e}", flush=True)
        time.sleep(1)
    print("Done.", flush=True)


import urllib.parse  # module-level for fetch_cot

if __name__ == "__main__":
    main()

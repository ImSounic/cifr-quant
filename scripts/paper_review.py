"""Graduation-review tool: reads the live measurement CSVs and prints the
day-14 / day-30 checklist verdicts automatically.

In plain terms: the shadow + demo books write a row every 8 hours; this script
compares those rows against what the frozen backtest PROMISED and says
PASS / WATCH / FAIL per checklist item — so the review is mechanical, not vibes.

Checks (and why each exists):
  1. Coverage/staleness    — did every 8h cycle actually run? (ops integrity)
  2. Funding collection    — live funding PnL/event vs backtest's +0.0070%/event.
                             The low-noise, fast-verifiable half of the edge.
  3. Turnover              — vs backtest's 0.11 gross/event. If live churns more,
                             the cost assumptions break.
  4. Synthetic fill rate   — lower bound on maker fills from public data.
  5. OKX real fill rate    — actual post-only fills + reject count.
  6. Equity vs noise band  — live cumulative net vs backtest mean ± 2σ band.
                             NOT a pass/fail on sign (30d of Sharpe 0.5 is noise);
                             only flags statistically surprising deviation.
  7. Realized beta to BTC  — regress live net returns on BTC 8h returns
                             (dollar-neutral should be ~0; material beta means
                             the hedge isn't doing its job).

Usage (HPC head node):
    python scripts/paper_review.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import RESULTS_DIR

PAPER_DIR = RESULTS_DIR / "paper"

# carry_history.csv schema grew mid-run: the header was written with 11 columns,
# then the synthetic-fill columns were inserted before longs/shorts, so the file
# mixes 11- and 13-field rows under an 11-column header. Map each row by its
# field count instead of trusting the header (naive NaN-padding would misalign
# longs/shorts on the early rows).
CARRY_COLS_V1 = ["run_at", "event", "n_assets", "price_pnl", "funding_pnl",
                 "cost", "net", "equity", "turnover", "longs", "shorts"]
CARRY_COLS_V2 = CARRY_COLS_V1[:9] + ["synth_orders", "synth_filled"] + CARRY_COLS_V1[9:]
CARRY_NUM_COLS = ["n_assets", "price_pnl", "funding_pnl", "cost", "net",
                  "equity", "turnover", "synth_orders", "synth_filled"]


def load_carry_history(path):
    recs = []
    for line in path.read_text().splitlines():
        f = line.split(",")
        if not line.strip() or f[0] == "run_at":
            continue
        if len(f) == len(CARRY_COLS_V1):
            cols = CARRY_COLS_V1
        elif len(f) == len(CARRY_COLS_V2):
            cols = CARRY_COLS_V2
        else:
            print(f"  (skipping malformed history line: {line[:60]}...)", flush=True)
            continue
        recs.append(dict(zip(cols, f)))
    h = pd.DataFrame(recs)
    for c in CARRY_NUM_COLS:
        if c in h.columns:
            h[c] = pd.to_numeric(h[c], errors="coerce")
    return h

# ---- Backtest expectations (frozen v2_maker_8h, results/backtest/carry_backtest_v2_maker_8h.json)
BT_NET_MEAN = 0.477 / 4938          # ~+0.0079%/event  (total +47.7% over 4938 events)
BT_NET_STD = 0.0050                 # per-event std implied by Sharpe 0.52
BT_FUND_MEAN = 0.3467 / 4938        # ~+0.0070%/event  (funding leg +34.67%)
BT_TURNOVER = 0.11                  # gross per event


def _verdict(ok, watch=False):
    return "PASS" if ok else ("WATCH" if watch else "FAIL")


def fetch_btc_8h_returns(events):
    """BTC perp 8h returns aligned to the live events (for the beta check)."""
    try:
        import ccxt
        ex = ccxt.binanceusdm({"enableRateLimit": True})
        since = int(pd.Timestamp(events[0]).timestamp() * 1000) - 3600_000
        candles = []
        cur = since
        while True:
            batch = ex.fetch_ohlcv("BTC/USDT:USDT", "1h", since=cur, limit=1000)
            if not batch:
                break
            candles.extend(batch)
            if len(batch) < 1000:
                break
            cur = batch[-1][0] + 1
        px = pd.Series({pd.Timestamp(c[0], unit="ms"): float(c[4]) for c in candles}).sort_index()
        marks = px.reindex(pd.DatetimeIndex(events), method="ffill")
        return marks.pct_change()
    except Exception as e:
        print(f"  (beta check skipped: {e})", flush=True)
        return None


def main():
    hist_path = PAPER_DIR / "carry_history.csv"
    if not hist_path.exists():
        print("No shadow history yet — nothing to review.")
        return
    h = load_carry_history(hist_path)
    h["event"] = pd.to_datetime(h["event"])
    h = h.sort_values("event").reset_index(drop=True)
    n = len(h)
    days = (h["event"].iloc[-1] - h["event"].iloc[0]).total_seconds() / 86400 if n > 1 else 0

    print(f"{'='*64}\n  SHADOW BOOK REVIEW  ({n} cycles, {days:.1f} days)\n{'='*64}", flush=True)

    # 1. Coverage
    expected = int(days * 3) + 1
    cov = n / max(expected, 1)
    print(f"\n  1. Coverage: {n}/{expected} expected cycles ({cov:.0%})  "
          f"[{_verdict(cov >= 0.95, cov >= 0.85)}]", flush=True)

    # 2. Funding collection
    fund = h["funding_pnl"].astype(float)
    fm, fse = fund.mean(), fund.std(ddof=1) / np.sqrt(max(n - 1, 1))
    z = (fm - BT_FUND_MEAN) / fse if fse > 0 else float("nan")
    print(f"  2. Funding/event: {fm:+.4%} vs backtest {BT_FUND_MEAN:+.4%} "
          f"(z={z:+.1f})  [{_verdict(abs(z) < 2 if z == z else False, abs(z) < 3 if z == z else False)}]",
          flush=True)

    # 3. Turnover (skip the opening deployment row)
    to = h["turnover"].astype(float).iloc[1:] if n > 1 else h["turnover"].astype(float)
    print(f"  3. Turnover/event: {to.mean():.3f} vs backtest {BT_TURNOVER:.2f}  "
          f"[{_verdict(to.mean() < 2 * BT_TURNOVER, to.mean() < 3 * BT_TURNOVER)}]", flush=True)

    # 4. Synthetic fills
    if "synth_orders" in h.columns and h["synth_orders"].sum() > 0:
        rate = h["synth_filled"].sum() / h["synth_orders"].sum()
        print(f"  4. Synthetic fill LOWER BOUND: {rate:.0%} "
              f"({int(h['synth_filled'].sum())}/{int(h['synth_orders'].sum())})  "
              f"[{_verdict(rate >= 0.8, rate >= 0.6)}]", flush=True)
    else:
        print("  4. Synthetic fills: no orders recorded yet", flush=True)

    # 5. OKX real fills
    ok_path = PAPER_DIR / "okx_demo_history.csv"
    if ok_path.exists():
        o = pd.read_csv(ok_path)
        placed = o["placed_usd"].astype(float).sum()
        filled = o["filled_usd"].astype(float).sum()
        rej = int(o["postonly_rejects"].astype(float).sum())
        if placed > 0:
            rate = filled / placed
            print(f"  5. OKX REAL fill rate: {rate:.0%} (${filled:,.0f}/${placed:,.0f}, "
                  f"{rej} rejects)  [{_verdict(rate >= 0.85, rate >= 0.7)}]", flush=True)
        else:
            print(f"  5. OKX fills: no reconciled orders yet ({len(o)} cycles)", flush=True)
    else:
        print("  5. OKX demo history not found", flush=True)

    # 6. Equity vs backtest noise band
    net = h["net"].astype(float)
    cum = net.sum()
    exp = BT_NET_MEAN * n
    band = 2 * BT_NET_STD * np.sqrt(n)
    inside = abs(cum - exp) <= band
    print(f"  6. Cumulative net: {cum:+.2%} vs expected {exp:+.2%} ± {band:.2%} "
          f"(2σ band)  [{_verdict(inside, abs(cum - exp) <= 1.5 * band)}]", flush=True)
    print(f"     (reminder: sign of {days:.0f}-day PnL is NOT the test — the band is)",
          flush=True)

    # 7. Beta to BTC
    if n >= 10:
        btc = fetch_btc_8h_returns(list(h["event"]))
        if btc is not None:
            m = (~btc.isna()).to_numpy() & np.isfinite(net.to_numpy())
            x, y = btc.to_numpy()[m], net.to_numpy()[m]
            if len(x) >= 10 and x.std() > 0:
                beta = float(np.cov(x, y)[0, 1] / np.var(x))
                print(f"  7. Realized beta to BTC: {beta:+.3f}  "
                      f"[{_verdict(abs(beta) < 0.15, abs(beta) < 0.3)}]  "
                      f"(dollar-neutral should be ~0)", flush=True)
    else:
        print("  7. Beta to BTC: needs ≥10 cycles", flush=True)

    print(f"\n  Graduation requires: 1,2,3 PASS + 4 or 5 PASS + 6 not FAIL + 7 not FAIL.",
          flush=True)


if __name__ == "__main__":
    main()

"""Shadow paper-trader for the frozen carry brick (Phase 4, v1).

Runs once per 8h funding event (via cron on the HPC head node — network I/O
only). v1 is SHADOW trading: no exchange account, no orders. It measures
signal-live consistency — does the live book, marked at real prices with real
funding accruals, track what the backtest promised? (Maker fill risk needs real
testnet orders — that's v2.)

Each run:
  1. Skips if the current funding event was already processed (idempotent).
  2. Fetches the last 10 funding events + current mark per asset (ccxt, public).
  3. Accrues REAL funding on the held book; marks price PnL against last marks.
  4. Recomputes the FROZEN-config target book (K=3, smooth=9 events,
     exit_band=2, gross 1.0 dollar-neutral) and simulates the rebalance at
     mark ± 0.03% on turnover.
  5. Appends to results/paper/carry_state.json + carry_history.csv.

FROZEN CONFIG (June 10 2026 — do not tune): K=3, SMOOTH=9, EXIT_BAND=2,
COST_PER_SIDE=0.0003. Any change = a new strategy that must re-earn its gate.

Cron (resolve <python> with `conda activate trade && which python`):
  5 0,8,16 * * * cd /home/s3702111/cifr-quant && <python> scripts/paper_trade_carry.py >> slurm/logs/paper_carry.log 2>&1

Manual first run:
  python scripts/paper_trade_carry.py
"""

import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import RESULTS_DIR

# ----- FROZEN CONFIG (brick #1) -----
K = 3
SMOOTH = 9
EXIT_BAND = 2
COST_PER_SIDE = 0.0003
# ------------------------------------

PAPER_DIR = RESULTS_DIR / "paper"
STATE_PATH = PAPER_DIR / "carry_state.json"
HIST_PATH = PAPER_DIR / "carry_history.csv"


def get_symbols():
    from configs.crypto_universe import get_crypto_configs
    return list(get_crypto_configs(tiers=(1, 2)).keys())


def fetch_live(symbols):
    """Per asset: trailing-SMOOTH mean funding (signal), the latest event's rate
    (accrual), the latest event timestamp, and the current mark price."""
    ex = ccxt.binanceusdm({"enableRateLimit": True})
    rows = {}
    latest_event = None
    for sym in symbols:
        m = f"{sym}:USDT"
        try:
            hist = ex.fetch_funding_rate_history(m, limit=SMOOTH + 1)
            if len(hist) < SMOOTH:
                continue
            rates = [float(h["fundingRate"]) for h in hist]
            ev_ts = pd.Timestamp(hist[-1]["timestamp"], unit="ms").round("h")
            ticker = ex.fetch_ticker(m)
            mark = float(ticker["last"])
            rows[sym] = {"signal": float(pd.Series(rates[-SMOOTH:]).mean()),
                         "last_rate": rates[-1], "mark": mark}
            latest_event = max(latest_event, ev_ts) if latest_event else ev_ts
            time.sleep(ex.rateLimit / 1000)
        except Exception as e:
            print(f"  skip {sym}: {e}", flush=True)
    return rows, latest_event


def target_book(rows, prev_w):
    """Frozen-config ranking with band hysteresis. Returns {sym: weight}."""
    order = sorted(rows, key=lambda s: rows[s]["signal"])   # low funding first
    rank_of = {a: r for r, a in enumerate(order)}
    n = len(order)

    prev_longs = [a for a, w in prev_w.items() if w > 0 and a in rank_of]
    prev_shorts = [a for a, w in prev_w.items() if w < 0 and a in rank_of]
    keep_l = sorted([a for a in prev_longs if rank_of[a] < K + EXIT_BAND],
                    key=lambda a: rank_of[a])[:K]
    keep_s = sorted([a for a in prev_shorts if rank_of[a] >= n - K - EXIT_BAND],
                    key=lambda a: -rank_of[a])[:K]

    longs = list(keep_l)
    for a in order:
        if len(longs) >= K:
            break
        if a not in longs and a not in keep_s:
            longs.append(a)
    shorts = list(keep_s)
    for a in reversed(order):
        if len(shorts) >= K:
            break
        if a not in shorts and a not in longs:
            shorts.append(a)

    w = {}
    for a in longs:
        w[a] = 0.5 / K
    for a in shorts:
        w[a] = -0.5 / K
    return w


def main():
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())
    else:
        state = {"started": now, "last_event": None, "equity": 1.0,
                 "positions": {}, "marks": {}}

    symbols = get_symbols()
    rows, event = fetch_live(symbols)
    if not rows or event is None:
        print(f"[{now}] no live data, aborting", flush=True)
        return
    event_iso = event.isoformat()
    if state["last_event"] == event_iso:
        print(f"[{now}] event {event_iso} already processed, exiting", flush=True)
        return

    prev_w = {k: float(v) for k, v in state["positions"].items()}
    prev_marks = {k: float(v) for k, v in state["marks"].items()}

    # 1. Mark the held book: price PnL since last marks + REAL funding accrual.
    price_pnl = 0.0
    for a, w in prev_w.items():
        if a in rows and a in prev_marks and prev_marks[a] > 0:
            price_pnl += w * (rows[a]["mark"] / prev_marks[a] - 1.0)
    fund_pnl = sum(-w * rows[a]["last_rate"] for a, w in prev_w.items() if a in rows)

    # 2. Rebalance to the frozen-config target; cost on turnover.
    new_w = target_book(rows, prev_w)
    all_syms = set(prev_w) | set(new_w)
    turnover = sum(abs(new_w.get(a, 0.0) - prev_w.get(a, 0.0)) for a in all_syms)
    cost = turnover * COST_PER_SIDE

    net = price_pnl + fund_pnl - cost
    state["equity"] = state["equity"] * (1.0 + net)
    state["last_event"] = event_iso
    state["positions"] = new_w
    state["marks"] = {a: rows[a]["mark"] for a in rows}

    STATE_PATH.write_text(json.dumps(state, indent=2))

    row = {"run_at": now, "event": event_iso, "n_assets": len(rows),
           "price_pnl": price_pnl, "funding_pnl": fund_pnl, "cost": cost,
           "net": net, "equity": state["equity"], "turnover": turnover,
           "longs": "|".join(a for a, w in new_w.items() if w > 0),
           "shorts": "|".join(a for a, w in new_w.items() if w < 0)}
    hdr = not HIST_PATH.exists()
    pd.DataFrame([row]).to_csv(HIST_PATH, mode="a", header=hdr, index=False)

    print(f"[{now}] event={event_iso}  net={net:+.4%}  equity={state['equity']:.4f}  "
          f"turnover={turnover:.2f}  longs={row['longs']}  shorts={row['shorts']}",
          flush=True)


if __name__ == "__main__":
    main()

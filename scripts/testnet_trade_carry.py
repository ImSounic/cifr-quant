"""Testnet trader for the frozen carry brick (Phase 4, v2).

Same FROZEN signal as the shadow trader (imported from paper_trade_carry — one
source of truth), but with REAL post-only limit orders on the Binance USDT-M
FUTURES TESTNET (fake money). Its single job is to measure what shadow cannot:
**maker fill risk** — do our passive orders actually fill at the prices the
backtest assumed?

Each 8h cycle:
  1. Idempotency check against its own state file.
  2. RECONCILE last cycle: fetch each order placed 8h ago -> filled / partial /
     unfilled; cancel leftovers; log the fill rate (the metric this exists for).
  3. Read CURRENT positions + account equity from the exchange (the exchange is
     the source of truth, not our state).
  4. Recompute the frozen target book; place POST-ONLY limit orders for the
     deltas: buys at best bid, sells at best ask. Post-only orders that would
     cross are REJECTED by the exchange — we log the reject and let the next
     cycle retry (that friction is exactly the fill risk we're measuring).
  5. Append to results/paper/testnet_history.csv.

Setup (one time) — NOTE June 2026: the old testnet.binancefuture.com portal is
DOWN for Binance's revamp; the replacement is Binance DEMO TRADING:
  1. Log into (or create) a regular Binance account — no KYC needed for demo.
  2. Open https://demo.binance.com/futures -> [Start Demo Trading]
     (also reachable via More -> Demo Trading on the main site/app).
  3. Inside the DEMO environment, open API Management and create an API
     key/secret (demo keys, fake balance, control no real money).
  4. Add them to the cron line as env vars. The script targets the demo API
     base (demo-fapi.binance.com) automatically; set BINANCE_USE_OLD_TESTNET=1
     to fall back to the legacy testnet endpoints if you have old keys.

Cron (10 min after each funding event, after the shadow trader's slot;
head node is CEST = UTC+2):
  10 2,10,18 * * * cd /home/s3702111/cifr-quant && BINANCE_TESTNET_KEY=... BINANCE_TESTNET_SECRET=... /home/s3702111/.conda/envs/trade/bin/python scripts/testnet_trade_carry.py >> slurm/logs/testnet_carry.log 2>&1
"""

import os
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
# Single source of truth for the frozen signal:
from scripts.paper_trade_carry import fetch_live, get_symbols, target_book, COST_PER_SIDE

PAPER_DIR = RESULTS_DIR / "paper"
STATE_PATH = PAPER_DIR / "testnet_state.json"
HIST_PATH = PAPER_DIR / "testnet_history.csv"

MIN_DELTA_USD = 50.0      # ignore dust rebalances
LEVERAGE = 2              # margin headroom for the short leg; gross stays 1.0


def _to_demo_urls(node):
    """Rewrite ccxt's sandbox URLs (legacy testnet.binancefuture.com) to the
    new Binance Demo Trading API base (demo-fapi.binance.com)."""
    if isinstance(node, dict):
        return {k: _to_demo_urls(v) for k, v in node.items()}
    if isinstance(node, str):
        return node.replace("testnet.binancefuture.com", "demo-fapi.binance.com")
    return node


def get_exchange():
    key = os.environ.get("BINANCE_TESTNET_KEY", "").strip()
    sec = os.environ.get("BINANCE_TESTNET_SECRET", "").strip()
    if not key or not sec:
        print("BINANCE_TESTNET_KEY / BINANCE_TESTNET_SECRET not set — aborting", flush=True)
        sys.exit(1)
    ex = ccxt.binanceusdm({"apiKey": key, "secret": sec, "enableRateLimit": True})
    ex.set_sandbox_mode(True)
    if os.environ.get("BINANCE_USE_OLD_TESTNET", "") != "1":
        ex.urls = _to_demo_urls(ex.urls)
    ex.load_markets()
    return ex


def account_equity(ex) -> float:
    bal = ex.fetch_balance()
    return float(bal["info"]["totalMarginBalance"])


def current_positions(ex, symbols) -> dict:
    """{spot_symbol: signed notional USD} from the exchange."""
    out = {}
    try:
        positions = ex.fetch_positions([f"{s}:USDT" for s in symbols])
    except Exception:
        positions = ex.fetch_positions()
    for p in positions:
        try:
            amt = float(p["info"].get("positionAmt", 0.0))
            mark = float(p["info"].get("markPrice") or p.get("markPrice") or 0.0)
            if amt != 0.0 and mark > 0:
                spot = p["symbol"].replace(":USDT", "")
                out[spot] = amt * mark
        except Exception:
            continue
    return out


def reconcile_last_cycle(ex, state) -> dict:
    """Fetch each order placed last cycle, cancel leftovers, compute fill stats."""
    placed = state.get("open_orders", [])
    stats = {"n_orders": len(placed), "placed_usd": 0.0, "filled_usd": 0.0}
    for o in placed:
        try:
            od = ex.fetch_order(o["id"], o["symbol"])
            price = float(od.get("price") or o["price"])
            stats["placed_usd"] += float(od.get("amount") or 0.0) * price
            stats["filled_usd"] += float(od.get("filled") or 0.0) * price
            if od.get("status") in ("open", "partially_filled"):
                try:
                    ex.cancel_order(o["id"], o["symbol"])
                except Exception:
                    pass
            time.sleep(ex.rateLimit / 1000)
        except Exception as e:
            print(f"  reconcile {o.get('symbol')}: {e}", flush=True)
    stats["fill_rate"] = (stats["filled_usd"] / stats["placed_usd"]
                          if stats["placed_usd"] > 0 else float("nan"))
    return stats


def place_rebalance(ex, symbols, targets_w, equity, positions):
    """Post-only limit orders toward the target book. Returns (orders, rejects)."""
    orders, rejects = [], 0
    for sym in set(list(targets_w.keys()) + list(positions.keys())):
        tgt = targets_w.get(sym, 0.0) * equity
        cur = positions.get(sym, 0.0)
        delta = tgt - cur
        if abs(delta) < MIN_DELTA_USD:
            continue
        m = f"{sym}:USDT"
        try:
            try:
                ex.set_leverage(LEVERAGE, m)
            except Exception:
                pass
            ob = ex.fetch_order_book(m, limit=5)
            side = "buy" if delta > 0 else "sell"
            price = ob["bids"][0][0] if side == "buy" else ob["asks"][0][0]
            amount = float(ex.amount_to_precision(m, abs(delta) / price))
            if amount <= 0:
                continue
            od = ex.create_order(m, "limit", side, amount,
                                 float(ex.price_to_precision(m, price)),
                                 params={"postOnly": True})
            orders.append({"id": od["id"], "symbol": m, "side": side,
                           "price": price, "amount": amount})
            print(f"  order {side} {amount} {m} @ {price}", flush=True)
        except Exception as e:
            msg = str(e)
            if "would immediately match" in msg or "-5022" in msg or "GTX" in msg:
                rejects += 1
                print(f"  post-only REJECT {sym} ({side} would cross)", flush=True)
            else:
                print(f"  order failed {sym}: {msg[:120]}", flush=True)
        time.sleep(ex.rateLimit / 1000)
    return orders, rejects


def main():
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {
        "last_event": None, "open_orders": []}

    symbols = get_symbols()
    rows, event = fetch_live(symbols)          # public data; frozen signal inputs
    if not rows or event is None:
        print(f"[{now}] no live data, aborting", flush=True)
        return
    event_iso = event.isoformat()
    if state["last_event"] == event_iso:
        print(f"[{now}] event {event_iso} already processed, exiting", flush=True)
        return

    ex = get_exchange()

    fill_stats = reconcile_last_cycle(ex, state)
    eq = account_equity(ex)
    positions = current_positions(ex, [s for s in symbols if s in rows])
    prev_w = {s: n / eq for s, n in positions.items() if eq > 0}

    targets = target_book(rows, prev_w)
    orders, rejects = place_rebalance(ex, symbols, targets, eq, positions)

    state["last_event"] = event_iso
    state["open_orders"] = orders
    STATE_PATH.write_text(json.dumps(state, indent=2))

    row = {"run_at": now, "event": event_iso, "equity_usdt": eq,
           "n_positions": len(positions),
           "last_cycle_orders": fill_stats["n_orders"],
           "last_cycle_fill_rate": fill_stats["fill_rate"],
           "placed_usd": fill_stats["placed_usd"],
           "filled_usd": fill_stats["filled_usd"],
           "new_orders": len(orders), "postonly_rejects": rejects,
           "longs": "|".join(s for s, w in targets.items() if w > 0),
           "shorts": "|".join(s for s, w in targets.items() if w < 0)}
    hdr = not HIST_PATH.exists()
    pd.DataFrame([row]).to_csv(HIST_PATH, mode="a", header=hdr, index=False)

    print(f"[{now}] event={event_iso}  equity={eq:,.0f} USDT  "
          f"fill_rate(last)={fill_stats['fill_rate']:.0%}  "
          f"new_orders={len(orders)}  rejects={rejects}"
          if fill_stats["placed_usd"] > 0 else
          f"[{now}] event={event_iso}  equity={eq:,.0f} USDT  "
          f"new_orders={len(orders)}  rejects={rejects}", flush=True)


if __name__ == "__main__":
    main()

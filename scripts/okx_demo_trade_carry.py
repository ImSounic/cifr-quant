"""OKX demo-trading executor for the frozen carry brick (Phase 4 v2).

Replaces the unreachable Binance testnet (Binance is geo-blocked in NL). OKX is
MiCA-licensed (legal in NL) and ccxt drives its demo environment via the
x-simulated-trading header (set_sandbox_mode).

DESIGN: the SIGNAL is unchanged — Binance funding rates via the frozen logic
imported from paper_trade_carry (the data the strategy was validated on). OKX
demo is the EXECUTION lab only: real post-only limit orders on OKX's simulated
swap books, measuring maker fill rates. (OKX's own funding rates differ from
Binance's — re-validating carry on OKX funding data is part of the Phase 4.5
venue decision, before any live capital.)

Each 8h cycle: reconcile last cycle's orders (fill rate = THE metric) ->
read positions/equity from OKX -> recompute frozen targets -> post-only limit
orders for the deltas (buys at bid, sells at ask; rejects logged).

OKX specifics handled here:
  - swap orders are sized in CONTRACTS (contractSize conversion);
  - tdMode='cross' passed explicitly;
  - account must be in ONE-WAY (net) position mode: demo Settings ->
    Trade settings -> Position mode -> One-way. (Hedge mode errors mention
    'posSide' — that's the tell.)
  - BTC contract granularity is coarse (~1 contract = 0.01 BTC); deltas that
    round to 0 contracts are skipped and logged.

Setup (one time):
  1. On OKX (logged in): switch to Demo trading (profile menu -> Demo trading).
  2. While IN demo mode: API keys page -> create a DEMO API key with Trade
     permission. You get key + secret + PASSPHRASE (OKX needs all three).
  3. Smoke test:
     OKX_API_KEY=.. OKX_SECRET=.. OKX_PASSPHRASE=.. python scripts/okx_demo_trade_carry.py

Cron (10 min after each funding event; head node is CEST = UTC+2):
  10 2,10,18 * * * cd /home/s3702111/cifr-quant && OKX_API_KEY=.. OKX_SECRET=.. OKX_PASSPHRASE=.. /home/s3702111/.conda/envs/trade/bin/python scripts/okx_demo_trade_carry.py >> slurm/logs/okx_demo_carry.log 2>&1
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
from scripts.paper_trade_carry import fetch_live, get_symbols, target_book

PAPER_DIR = RESULTS_DIR / "paper"
STATE_PATH = PAPER_DIR / "okx_demo_state.json"
HIST_PATH = PAPER_DIR / "okx_demo_history.csv"

MIN_DELTA_USD = 50.0


def get_exchange():
    key = os.environ.get("OKX_API_KEY", "").strip()
    sec = os.environ.get("OKX_SECRET", "").strip()
    pwd = os.environ.get("OKX_PASSPHRASE", "").strip()
    if not (key and sec and pwd):
        print("OKX_API_KEY / OKX_SECRET / OKX_PASSPHRASE not set — aborting", flush=True)
        sys.exit(1)
    ex = ccxt.okx({"apiKey": key, "secret": sec, "password": pwd,
                   "enableRateLimit": True})
    ex.set_sandbox_mode(True)          # x-simulated-trading: 1 (demo env)
    ex.load_markets()
    return ex


def account_equity(ex) -> float:
    bal = ex.fetch_balance()
    try:
        return float(bal["info"]["data"][0]["totalEq"])
    except Exception:
        return float(bal.get("USDT", {}).get("total") or 0.0)


def current_positions(ex) -> dict:
    """{spot_symbol: signed notional USD}."""
    out = {}
    for p in ex.fetch_positions():
        try:
            contracts = float(p.get("contracts") or 0.0)
            if contracts == 0.0:
                continue
            csize = float(p.get("contractSize") or 1.0)
            mark = float(p.get("markPrice") or 0.0)
            if mark <= 0:
                continue
            sign = 1.0 if p.get("side") == "long" else -1.0
            spot = p["symbol"].replace(":USDT", "")
            out[spot] = sign * contracts * csize * mark
        except Exception:
            continue
    return out


def reconcile_last_cycle(ex, state) -> dict:
    placed = state.get("open_orders", [])
    stats = {"n_orders": len(placed), "placed_usd": 0.0, "filled_usd": 0.0}
    for o in placed:
        try:
            od = ex.fetch_order(o["id"], o["symbol"])
            csize = o.get("contract_size", 1.0)
            price = float(od.get("price") or o["price"])
            stats["placed_usd"] += float(od.get("amount") or 0.0) * csize * price
            stats["filled_usd"] += float(od.get("filled") or 0.0) * csize * price
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


def place_rebalance(ex, targets_w, equity, positions):
    orders, rejects, granularity_skips = [], 0, 0
    for sym in set(list(targets_w.keys()) + list(positions.keys())):
        tgt = targets_w.get(sym, 0.0) * equity
        cur = positions.get(sym, 0.0)
        delta = tgt - cur
        if abs(delta) < MIN_DELTA_USD:
            continue
        m = f"{sym}:USDT"
        if m not in ex.markets:
            print(f"  no OKX swap for {sym}, skipping", flush=True)
            continue
        try:
            market = ex.market(m)
            csize = float(market.get("contractSize") or 1.0)
            ob = ex.fetch_order_book(m, limit=5)
            side = "buy" if delta > 0 else "sell"
            price = ob["bids"][0][0] if side == "buy" else ob["asks"][0][0]
            contracts = abs(delta) / (price * csize)
            amount = float(ex.amount_to_precision(m, contracts))
            if amount <= 0:
                granularity_skips += 1
                print(f"  skip {sym}: ${abs(delta):.0f} delta < 1 contract "
                      f"(~${price * csize:.0f})", flush=True)
                continue
            od = ex.create_order(m, "limit", side, amount,
                                 float(ex.price_to_precision(m, price)),
                                 params={"postOnly": True, "tdMode": "cross"})
            orders.append({"id": od["id"], "symbol": m, "side": side,
                           "price": price, "amount": amount,
                           "contract_size": csize})
            print(f"  order {side} {amount} contracts {m} @ {price}", flush=True)
        except Exception as e:
            msg = str(e)
            if "post" in msg.lower() and ("match" in msg.lower() or "taker" in msg.lower()):
                rejects += 1
                print(f"  post-only REJECT {sym}", flush=True)
            elif "posSide" in msg:
                print(f"  ORDER FAILED {sym}: account is in HEDGE mode — switch demo "
                      f"account to One-way position mode (see docstring).", flush=True)
            else:
                print(f"  order failed {sym}: {msg[:400]}", flush=True)
        time.sleep(ex.rateLimit / 1000)
    return orders, rejects, granularity_skips


def main():
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {
        "last_event": None, "open_orders": []}

    symbols = get_symbols()
    rows, event = fetch_live(symbols)          # Binance funding = frozen signal
    if not rows or event is None:
        print(f"[{now}] no live signal data, aborting", flush=True)
        return
    event_iso = event.isoformat()
    if state["last_event"] == event_iso:
        print(f"[{now}] event {event_iso} already processed, exiting", flush=True)
        return

    ex = get_exchange()
    fill_stats = reconcile_last_cycle(ex, state)
    eq = account_equity(ex)
    if eq <= 0:
        print(f"[{now}] OKX demo equity is 0 — top up demo funds in the demo UI",
              flush=True)
        return
    positions = current_positions(ex)
    prev_w = {s: n / eq for s, n in positions.items()}

    targets = target_book(rows, prev_w)
    orders, rejects, gran_skips = place_rebalance(ex, targets, eq, positions)

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
           "granularity_skips": gran_skips,
           "longs": "|".join(s for s, w in targets.items() if w > 0),
           "shorts": "|".join(s for s, w in targets.items() if w < 0)}
    hdr = not HIST_PATH.exists()
    pd.DataFrame([row]).to_csv(HIST_PATH, mode="a", header=hdr, index=False)

    fr = fill_stats["fill_rate"]
    fr_txt = f"{fr:.0%}" if fr == fr else "n/a"
    print(f"[{now}] event={event_iso}  equity={eq:,.0f}  fill_rate(last)={fr_txt}  "
          f"new_orders={len(orders)}  rejects={rejects}  gran_skips={gran_skips}",
          flush=True)


if __name__ == "__main__":
    main()

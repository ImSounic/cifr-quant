"""OKX demo account diagnostic — find out WHY orders fail with 51008 (margin).

Hypothesis being tested: ZOMBIE OPEN ORDERS. The executor only cancels orders
recorded in okx_demo_state.json's last cycle; any cycle that crashed mid-run
(e.g. the 50013 "Systems busy" errors) leaks its live orders. In cross margin,
leaked orders reserve margin (ordFrozen) forever, so placement eventually fails
with 51008 even though equity is fine — which freezes the book on stale
positions (the XRP position that is in neither target leg is the tell).

Read-only by default. Prints:
  1. Account equity + available equity + margin frozen by open orders
  2. All open positions (symbol, side, notional, margin)
  3. ALL open orders with age; flags orders older than 9h as ZOMBIES
     (anything from a previous cycle should have been reconciled/cancelled)

--cancel-zombies : cancel open orders older than 9h (state-changing; run the
                   read-only pass first and eyeball the list).

Usage (creds are the same demo keys as the cron line):
    OKX_API_KEY=.. OKX_SECRET=.. OKX_PASSPHRASE=.. python scripts/okx_demo_diag.py
    OKX_API_KEY=.. OKX_SECRET=.. OKX_PASSPHRASE=.. python scripts/okx_demo_diag.py --cancel-zombies
"""

import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.okx_demo_trade_carry import get_exchange  # noqa: E402

ZOMBIE_AGE_H = 9.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancel-zombies", action="store_true",
                    help="Cancel open orders older than 9h (state-changing)")
    args = ap.parse_args()

    ex = get_exchange()
    now_ms = ex.milliseconds()

    # 1. Balance: equity vs what's actually available vs frozen by orders
    bal = ex.fetch_balance()
    try:
        acct = bal["info"]["data"][0]
        print(f"\n{'='*64}\n  ACCOUNT\n{'='*64}", flush=True)
        print(f"  totalEq (USD):     {float(acct.get('totalEq') or 0):>12,.0f}", flush=True)
        print(f"  adjEq   (USD):     {float(acct.get('adjEq') or 0):>12,.0f}", flush=True)
        for d in acct.get("details", []):
            ccy = d.get("ccy")
            print(f"  [{ccy}] eq={float(d.get('eq') or 0):,.0f}  "
                  f"availEq={float(d.get('availEq') or 0):,.0f}  "
                  f"ordFrozen={float(d.get('ordFrozen') or 0):,.0f}  "
                  f"frozenBal={float(d.get('frozenBal') or 0):,.0f}  "
                  f"imr={float(d.get('imr') or 0):,.0f}  "
                  f"mmr={float(d.get('mmr') or 0):,.0f}", flush=True)
    except Exception as e:
        print(f"  (raw balance details unavailable: {e})", flush=True)

    # 2. Positions
    print(f"\n{'='*64}\n  POSITIONS\n{'='*64}", flush=True)
    gross = 0.0
    net = 0.0
    for p in ex.fetch_positions():
        contracts = float(p.get("contracts") or 0.0)
        if contracts == 0.0:
            continue
        csize = float(p.get("contractSize") or 1.0)
        mark = float(p.get("markPrice") or 0.0)
        sign = 1.0 if p.get("side") == "long" else -1.0
        notional = sign * contracts * csize * mark
        gross += abs(notional)
        net += notional
        print(f"  {p['symbol']:22s} {p.get('side'):5s} {contracts:12,.2f} contracts  "
              f"notional ${notional:+12,.0f}  margin ${float(p.get('initialMargin') or 0):,.0f}  "
              f"lev {p.get('leverage')}", flush=True)
    print(f"  gross ${gross:,.0f}   net ${net:+,.0f}", flush=True)

    # 3. Open orders — the zombie hunt
    print(f"\n{'='*64}\n  OPEN ORDERS\n{'='*64}", flush=True)
    try:
        orders = ex.fetch_open_orders()
    except Exception as e:
        print(f"  fetch_open_orders without symbol failed ({e}); trying per-position…",
              flush=True)
        orders = []
        for p in ex.fetch_positions():
            try:
                orders.extend(ex.fetch_open_orders(p["symbol"]))
                time.sleep(ex.rateLimit / 1000)
            except Exception:
                pass

    zombies = []
    total_reserved = 0.0
    for o in orders:
        age_h = (now_ms - (o.get("timestamp") or now_ms)) / 3_600_000
        px = float(o.get("price") or 0.0)
        amt = float(o.get("remaining") or o.get("amount") or 0.0)
        m = ex.market(o["symbol"]) if o.get("symbol") in ex.markets else {}
        csize = float(m.get("contractSize") or 1.0)
        notional = amt * csize * px
        total_reserved += notional
        tag = "ZOMBIE" if age_h > ZOMBIE_AGE_H else "current"
        if age_h > ZOMBIE_AGE_H:
            zombies.append(o)
        ts = datetime.fromtimestamp((o.get("timestamp") or 0) / 1000, tz=timezone.utc)
        print(f"  [{tag:7s}] {o['symbol']:22s} {o.get('side'):4s} {amt:12,.2f} @ {px:<12g} "
              f"~${notional:10,.0f}  age {age_h:6.1f}h  placed {ts:%Y-%m-%d %H:%M}  id={o.get('id')}",
              flush=True)
    print(f"\n  {len(orders)} open orders, {len(zombies)} zombies (>{ZOMBIE_AGE_H:.0f}h), "
          f"~${total_reserved:,.0f} notional resting", flush=True)

    if args.cancel_zombies and zombies:
        print(f"\n  Cancelling {len(zombies)} zombie orders…", flush=True)
        ok = 0
        for o in zombies:
            try:
                ex.cancel_order(o["id"], o["symbol"])
                ok += 1
                print(f"  cancelled {o['symbol']} {o.get('id')}", flush=True)
            except Exception as e:
                print(f"  cancel FAILED {o['symbol']} {o.get('id')}: {str(e)[:100]}", flush=True)
            time.sleep(ex.rateLimit / 1000)
        print(f"  {ok}/{len(zombies)} cancelled — margin should free up; the next cron "
              f"cycle can rebalance the stale book (incl. closing XRP).", flush=True)
    elif zombies:
        print(f"\n  Re-run with --cancel-zombies to cancel them and free the margin.",
              flush=True)


if __name__ == "__main__":
    main()

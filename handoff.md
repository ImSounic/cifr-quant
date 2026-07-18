# CIFR-QUANT — Handoff

> Generated June 10, 2026. Companion docs: `PROJECT_STATE.md` (authoritative
> running state + glossary), `PROJECT_PLAN.md` (phased plan with reasoning),
> `docs/OPERATIONS.md` (cron schedule + check routine), `docs/STRATEGY_DESIGN.md`.

## Project Goal

CIFR-QUANT is a systematic crypto/commodities trading research system, built by Raj
(UTwente student, HPC user `s3702111`) toward a one-person quant firm. It exists to
**find statistically real, capacity-constrained trading edges, validate them
ruthlessly, and trade them automatically** — the kind of small edges large funds
step over.

The project has two eras:
- **v1 (retired):** trade a financial foundation model (**Kronos**, 102M params) that
  forecasts OHLCV candles. **Rigorously falsified** June 6–10, 2026 — the model has no
  extractable predictive edge (directional, cross-sectional, or volatility-beyond-
  persistence) on crypto 15m or commodity 4h data. Full evidence chain in
  `PROJECT_STATE.md` → "Phase 1 Post-Mortem".
- **v2 (current): the "Signal Factory."** Reuse the validated research pipeline
  (statistical diagnostics → cost-honest backtest → live shadow trading) but point it
  at *information-bearing* data (perp funding rates, open interest, CFTC positioning)
  instead of price candles. **One validated strategy ("brick #1") exists and is live
  on paper money:** cross-sectional funding-rate carry, +9%/yr backtested, Sharpe 0.52,
  profitable every year 2022–2026, dollar-neutral.

Core methodology (the "constitution," each rule learned from a specific failure):
diagnostics before backtests (|t|>2 or no backtest); count every trial and deflate;
multi-year stability never one window; test what the portfolio actually trades
(construction-matched); model costs from day one; freeze what passes.

## Tech Stack

- **Language:** Python 3.11 (HPC `trade` conda env) / 3.12 (laptop `mlenv` venv).
- **Core libs:** `pandas`, `numpy`, `scipy` (all statistics/backtests are hand-rolled
  NumPy/pandas — no ML framework in the v2 live path).
- **Exchange/data:** `ccxt` (Binance USDT-M, OKX, Hyperliquid funding/perp/OI),
  `yfinance` (commodity futures daily), CFTC Socrata API via `urllib` (COT positioning),
  `twelvedata` (legacy commodity spot).
- **v1 model stack (dormant but intact):** `torch`, `transformers`, `huggingface_hub`,
  `einops`; **Kronos** vendored at `Kronos/` (gitlink, commit `67b630e`, **no
  `.gitmodules`** — a fresh clone leaves it empty; restore manually).
- **Execution venue:** **OKX** (MiCA-licensed, legal in NL). Live orders go through
  OKX demo (sandbox) via ccxt. **Binance is geo-blocked from the Netherlands** (withdrew
  2023) — used for public data only, never account trading.
- **Alerting:** `ntfy.sh` (push notifications, topic `cifr-carry-rj76x2`) via stdlib `urllib`.
- **Compute:** UTwente SLURM HPC (head node `hpc-head2.ewi.utwente.nl`, conda env
  `trade`, L40S/L40 GPUs for the dormant Kronos jobs). **All v2 work is CPU + network
  I/O**; the live crons run on the HPC **head node** (it has internet; compute nodes
  don't). No GPU is used by anything currently running.
- **`requirements.txt` is v1-era and stale** — it lists `vectorbt`/`mapie` which are
  NOT used (custom backtest engine + custom CQR instead) and must NOT be installed on
  the laptop (they downgrade numpy/pandas). v2 only needs:
  `pandas numpy scipy ccxt yfinance tqdm`.

## How to Run

There is no single "app" — it's a research repo plus a 4-job cron stack. Two contexts:

### A. Laptop (dev / small checks only — `W:\cifr-quant`)
```powershell
# venv 'mlenv' is pre-built at C:\Work Drive\envs\mlenv; activate by typing:
mlenv
# deps already present; for a bare machine: pip install pandas numpy scipy ccxt yfinance tqdm
# run a diagnostic (needs data fetched first; see below):
python scripts/carry_skill.py
```

### B. HPC head node (all real workloads — `~/cifr-quant`)
```bash
ssh s3702111@hpc-head2.ewi.utwente.nl
cd ~/cifr-quant
source $(conda info --base)/etc/profile.d/conda.sh && conda activate trade

# 1) fetch derivatives data (network only, ~15-30 min, head node):
python scripts/fetch_derivs.py                          # Binance funding/perp/OI, 14 assets
python scripts/fetch_derivs.py --exchange okx --what funding   # OKX (incremental)

# 2) run the validated diagnostic (CPU, seconds):
python scripts/carry_skill.py                           # 8h XS t=-4.83 expected

# 3) run the earned backtest (CPU, seconds):
python scripts/backtest_carry.py --smooth 9 --exit-band 2 --cost-per-side 0.0003 --tag v2_maker_8h

# 4) the LIVE stack runs via crontab (already installed). Manual single cycle:
python scripts/paper_trade_carry.py                     # shadow book, no creds needed
OKX_API_KEY=.. OKX_SECRET=.. OKX_PASSPHRASE=.. python scripts/okx_demo_trade_carry.py
```

### Environment variables (only the live OKX executor needs them)
```bash
OKX_API_KEY="<demo key from OKX demo-trading API page>"
OKX_SECRET="<demo secret>"
OKX_PASSPHRASE="<the passphrase you chose when creating the key>"
# Heartbeat:
NTFY_TOPIC="cifr-carry-rj76x2"
# v1 only (HPC): HF_HUB_OFFLINE=1 (compute nodes are offline)
```
There is **no `.env` file in use** for v2; keys are passed inline in the crontab lines
(demo/sandbox keys only — they control no real money).

### The live cron stack (installed on hpc-head2; times are CEST = UTC+2)
```
5  2,10,18 * * *  paper_trade_carry.py        >> slurm/logs/paper_carry.log
10 2,10,18 * * *  okx_demo_trade_carry.py     >> slurm/logs/okx_demo_carry.log   (OKX creds inline)
30 *       * * *  heartbeat_carry.py          >> slurm/logs/heartbeat.log        (NTFY_TOPIC inline)
0  3       * * 0  fetch_derivs.py --exchange okx --what funding >> slurm/logs/fetch_okx_funding.log
```

## Architecture Overview

```
data/raw/{derivs,derivs_okx,derivs_hyperliquid,cot}/   ← funding/perp/OI/positioning CSVs (gitignored, on HPC)
        │  (fetchers: src/data/derivs_client.py, scripts/fetch_derivs.py, scripts/fetch_cot.py)
        ▼
scripts/*_skill.py   THE GAUNTLET — pure-statistics diagnostics (IC / t-stat / multi-year /
        │            cross-sectional / construction-matched tail spread). RULE: a signal earns a
        │            backtest only by passing here (|t|>2). Most candidates die here, cheaply.
        ▼
scripts/backtest_*.py   Cost-honest backtests for survivors (funding cashflows, turnover costs,
        │               smoothing, hysteresis, per-year decomposition). Writes results/backtest/*.json
        ▼
scripts/paper_trade_carry.py  + okx_demo_trade_carry.py   LIVE 8h cron loop running the FROZEN config:
        │  shadow (imaginary money, real prices, synthetic fill check) + OKX demo (real sandbox orders).
        ▼
scripts/heartbeat_carry.py + scripts/paper_review.py   Watchdog (ntfy alerts) + automated graduation checklist.
        Outputs: results/paper/{carry,okx_demo}_history.csv  ← the live track record (gitignored, on HPC).
```

**Key v2 files (active):**
- `src/data/derivs_client.py` — venue-agnostic ccxt fetchers (funding/perp/OI). Funding
  fetch is **incremental/appending** (critical for OKX/HL, whose APIs serve short windows).
  `_perp_symbol()` handles per-venue symbol formats (Binance `BTC/USDT:USDT`, HL `BTC/USDC:USDC`).
- `scripts/fetch_derivs.py` — driver: `--exchange {binanceusdm,okx,hyperliquid}`, `--what {all,funding,perp,oi}`.
- `scripts/fetch_cot.py` — CFTC COT (weekly, 1986–present) + yfinance daily futures, 6 commodities.
- `scripts/carry_skill.py` — **the validated diagnostic.** Per-asset + cross-sectional IC,
  t-stats on non-overlapping samples, per-year stability, funding-harvest economics. `--exchange` flag.
- `scripts/backtest_carry.py` — **the frozen strategy.** Cross-sectional long-low/short-high
  funding, dollar-neutral; funding cashflows in PnL; costs on turnover; `--smooth`, `--exit-band`,
  `--cost-per-side`, `--rebalance-every`.
- `scripts/paper_trade_carry.py` — **shadow trader + FROZEN CONFIG (the constants K=3, SMOOTH=9,
  EXIT_BAND=2, COST_PER_SIDE=0.0003 live in its header).** `target_book()` and `fetch_live()` here
  are the single source of truth imported by the OKX executor.
- `scripts/okx_demo_trade_carry.py` — OKX demo executor; imports the frozen signal, places real
  post-only limit orders (sized in OKX *contracts*), reconciles fills, logs fill rate.
- `scripts/heartbeat_carry.py` — staleness (>9h) + drawdown (>10%) watchdog → ntfy push.
- `scripts/paper_review.py` — reads both history CSVs, prints PASS/WATCH/FAIL graduation checklist
  (coverage, funding collection vs backtest, turnover, fill rates, equity-vs-noise-band, BTC beta).
- Other diagnostics (all ran, all verdicts recorded): `forecast_skill.py`, `horizon_skill.py`,
  `vol_skill.py` (the Kronos falsification trio); `momo_skill.py`, `tsmom_skill.py` (failed candidates);
  `oi_skill.py`, `cot_skill.py` (data-gated / failed candidates); `backtest_reversal.py` (failed).
- `configs/base_config.py` — paths (`DATA_RAW_DIR`, `RESULTS_DIR`) + dataclasses.
- `configs/crypto_universe.py` — the 14-asset tier-1/2 universe (`get_crypto_configs`).

**v1 infrastructure (intact, dormant — reusable patterns):**
- `src/backtest/` — pluggable walk-forward engine (`portfolio_engine.py`, `strategy_api.py`,
  `strategies.py`, `sizing.py`, `grid.py`, `costs.py`, `metrics.py`). Strategy/Sizer injectable.
- `src/model/` — Kronos stack: `build.py` (ensemble builder), `batched_inference.py` (fixes Kronos's
  path-averaging bug), `forecast_cache.py` (run GPU once, test strategies CPU-only), `ensemble.py`.
- `src/regime/` — `indicators.py` (hurst/adx/atr/realized_vol — reusable for vol sizing), `classifier.py`
  (Hurst+ADX gate, empirically net-negative, retired).
- `src/risk/cqr.py` — conformal calibration (works). `src/portfolio/orchestrator.py`, `src/strategy/`
  (LLM strategy-picker, cancelled by design), `src/backtest/engine.py` (single-asset) — legacy.
- `Kronos/` — vendored model (gitlink @ 67b630e). `checkpoints/` (gitignored) — finetuned weights on HPC.

## Current State

**Fully working & live:**
- Funding-carry diagnostic + backtest → **brick #1 frozen** (`v2_maker_8h`: K=3, smooth 9,
  exit-band 2, 8h cadence, maker 0.03%/side; +9.0%/yr, Sharpe 0.52, net-positive all 5 years).
- 4-job cron stack running unattended since June 10 (shadow + OKX demo + heartbeat + weekly OKX archive).
  OKX demo has placed real post-only orders successfully (first batch: 5 orders, 0 rejects, 4/5 filled <1h).
- Venue validated three ways: carry confirmed on Binance (t=-4.83 / 5yr) and OKX (t=-2.68 / 3mo,
  13/13 sign agreement); **Phase 4.5 decision = OKX**.
- Full diagnostic gauntlet, all v1 backtest/model infra, data fetchers for 4 venues + CFTC.

**Partially implemented / in progress:**
- **30-day live shadow/demo window** (June 10 → ~July 10). Day-14 review ~June 24, graduation ~July 10.
  Judged by checklist (`paper_review.py`), NOT by PnL sign (30 days of Sharpe-0.5 is statistical noise).
- `oi_skill.py` — built, **data-gated**: refuses to conclude below 21 usable days of OI history
  (accumulating since June 10; ripe ~early July). Open-interest/squeeze is the next live candidate.
- Synthetic maker-fill measurement — newly added to the shadow trader; columns `synth_orders`/
  `synth_filled` will populate from the second-onward cron cycles.

**Killed (verdicts recorded, infra kept):** XS momentum, short-term reversal, tail momentum, TSMOM
(crypto price signals); COT positioning (40 years, no edge); Hyperliquid carry (wrong sign on its window).
One survivor from eight candidates — a normal hit rate.

**Not started:** commodity **term-structure carry** (the declared next commodity candidate — needs
yfinance contract-chain stitching, ~1 day of data engineering); **Phase 3** LLM-as-data-source (news/
sentiment → features → gauntlet); **Phase 2D** portfolio layer with vol-targeted sizing (needs ≥2 bricks);
**Phase 5** platform (only alerting built so far); **VPS migration** (required before real capital — the
HPC account is study-tied and shouldn't host a live firm); **automated data backups** (manual `scp` for now).

## Active Work

The most recent meaningful change was **bringing commodities back into the candidate pipeline**
(`scripts/fetch_cot.py` + `scripts/cot_skill.py`, committed `7408a43`). The COT positioning signal
("fade crowded speculators") was fetched (1986–present, 6 commodities) and run through the gauntlet —
and **failed cleanly** (mean IC ≈ 0, sign-flipping across 3-year buckets for four decades, the one
tempting 4w cell fails multiplicity + construction-matched + stability checks). It matches the modern
literature (the simple COT fade is arbitraged flat). The fetcher infrastructure is kept for reuse.

**Intended next step:** with COT killed, the declared next commodity candidate is **term-structure
carry** (futures-curve backwardation/contango) — a deliberate ~1-day data-engineering project (stitch
yfinance contract chains), explicitly scheduled for *within* the shadow window, not bolted onto today.
In parallel, the **scheduled reviews are the real near-term work**: run `python scripts/paper_review.py`
at ~June 24 and ~July 10 and act on the checklist.

## Known Issues / TODOs

- **`requirements.txt` is stale (v1-era).** Lists `vectorbt`/`mapie` (unused — custom engine + custom
  CQR) and omits the v2 data libs as a clean set. Do NOT `pip install -r` it on the laptop (numpy/pandas
  downgrade risk). TODO: split into `requirements-v1.txt` / `requirements-v2.txt`.
- **OKX demo book runs occasionally net-long, not dollar-neutral.** `okx_demo_trade_carry.py`: OKX demo
  lists **no UNI perp**, so when UNI is in the short leg the book is 3-long/2-short (~17% net exposure).
  Acceptable for fill measurement; before live, exclude UNI from the *signal* universe on OKX so legs stay balanced.
- **Live track record + accumulated OKX/HL funding archives exist ONLY on the HPC** (gitignored:
  `results/paper/`, `data/raw/derivs_okx/`). No automated backup. TODO: weekly `scp` to laptop
  `W:\backups\cifr-quant\` (commands in `docs/OPERATIONS.md`); proper fix is the VPS migration.
- **Cron timezone gotcha:** head node is CEST; funding events are UTC. Cron lines use local time
  (2,10,18). When NL shifts to CET in winter, the offset-from-event grows ~1h (acceptable; idempotency
  guards prevent double-processing).
- **Hyperliquid price history is API-capped at ~5,000 bars (~7 months);** any future HL research must
  accumulate forward. HL also pays funding *hourly* — `carry_skill.load_asset()` resamples to 8h.
- **`paper_review.py` backtest-expectation constants** (`BT_NET_MEAN`, `BT_FUND_MEAN`, `BT_TURNOVER`,
  ~lines 38–42) are hardcoded from `carry_backtest_v2_maker_8h.json`; if the frozen config ever changes
  (it shouldn't), update them.
- **OKX one-way position mode required.** If `okx_demo_trade_carry.py` errors mentioning `posSide`, the
  demo account is in hedge mode — switch to One-way in demo trade settings (the script prints this hint).
- **Kronos `Kronos/` is a gitlink with no `.gitmodules`** — fresh clones get an empty dir; restore with
  `git clone https://github.com/shiyu-coder/Kronos.git Kronos && cd Kronos && git checkout 67b630e`.
  Edits inside `Kronos/` do NOT propagate via git — all project code lives in tracked `src/`.

## Context for Next Session

You are continuing **CIFR-QUANT v2**, a solo-quant "signal factory." The original thesis (trade the
Kronos foundation model on price candles) was **rigorously falsified** and is retired — do not revive it;
read `PROJECT_STATE.md` "Phase 1 Post-Mortem" if tempted. The firm currently has **exactly one validated
strategy** ("brick #1"): cross-sectional funding-rate carry (short the highest-funding crypto perps / long
the lowest, dollar-neutral, 8h rebalance, maker execution), **FROZEN** with constants in
`scripts/paper_trade_carry.py` — any change is a new strategy that must re-earn the |t|>2 gate, so do not
"improve" it. It has been live on paper money + OKX demo via cron since June 10, 2026; the job now is a
30-day measurement window with reviews via `python scripts/paper_review.py` at ~June 24 and ~July 10
(graduation is decided by the **checklist, never the 30-day PnL sign**). All v2 work is CPU + network I/O
on the HPC head node (no GPU). The non-negotiable methodology: **diagnostics before backtests, |t|>2,
multi-year stability, count every trial, construction-matched tests, model costs, freeze winners** — every
rule traces to a specific past failure. Open threads, in priority order: run the scheduled reviews; run
`oi_skill.py` when OI history ripens (~early July) for candidate brick #2; build commodity term-structure
carry (declared next commodity candidate, ~1 day); before any real capital, migrate the cron stack off the
study-tied HPC to a ~€5/mo VPS and set up data backups. Live venue is decided: **OKX** (Binance is
geo-blocked from the Netherlands). Read `PROJECT_STATE.md` first for the authoritative state and glossary;
`docs/OPERATIONS.md` for the cron schedule and check commands.

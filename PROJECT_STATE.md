# CIFR-QUANT: Project State

> **Purpose**: Cross-session memory for Claude AND a plain-language record for Raj.
> Each section has the technical facts plus **In plain terms** (what it means) and
> **Why we did it** (the reasoning). Read this file at the start of every session.
> **Last updated**: June 10, 2026

---

## Glossary (plain English for the terms used everywhere below)

- **Signal**: any piece of data that might predict where prices go (e.g. "funding rates predict returns").
- **IC (Information Coefficient)**: a score from −1 to +1 for how well a signal's predictions line up with what actually happened. 0 = useless. Real, tradeable signals are tiny — 0.02–0.05.
- **t-stat**: "how sure are we this isn't luck?" Roughly: |t| > 2 means less than ~5% chance it's a fluke. Our rule: no t-stat, no trust.
- **Sharpe ratio**: return earned per unit of risk taken. 0.5 = decent, 1.0–1.5 = excellent, 4+ in a backtest = probably fooling yourself.
- **Backtest**: replaying a strategy on historical data to see what it *would* have made. Dangerous because it's easy to accidentally cheat.
- **Diagnostic (vs backtest)**: a pure statistics test of whether a signal predicts anything at all — run BEFORE any backtest, so we never waste time (or fool ourselves) backtesting noise.
- **Funding rate**: every 8 hours, perpetual-futures exchanges transfer money between long and short traders to keep the perp price near spot. Crowded longs → longs pay shorts.
- **Dollar-neutral**: equal money long and short, so the market crashing or pumping roughly cancels out — we bet on the *gap between* coins, not the direction of crypto.
- **Maker / taker**: maker = patient limit order that waits to be filled (cheap fees); taker = order that executes instantly by crossing the spread (expensive). Our strategy only works with maker execution.
- **Drawdown**: how far the account has fallen from its peak. Our tripwire: −10% → halt and investigate.
- **Shadow trading**: running the strategy against live market data with imaginary money — the only test that can't be accidentally cheated, because the future hadn't happened when we wrote the code.

---

## Quick Summary

CIFR-QUANT is a systematic-trading research project building toward a quant firm. Raj (s3702111), UTwente student. HP Windows laptop = dev; UTwente HPC = all heavy compute.

**In plain terms**: we set out to make an AI model predict crypto/commodity prices and trade on it. We tested that idea to destruction — it doesn't work — and in the process built something more valuable: a fast, honest *testing machine* for trading ideas. We then pointed that machine at better data (funding rates), found one real, statistically proven strategy, and as of June 10 it's running live on fake money, fully automated, while we hunt for the next one.

**The two eras**:
1. **v1 (June 2025 – June 10, 2026)**: Kronos foundation model → directional trading. **Falsified** — the model has no predictive edge of any kind on price candles. Full evidence chain below.
2. **v2 (June 10, 2026 →)**: the **Signal Factory** — test many simple trading ideas against hard statistics; trade only what survives. First survivor: **funding-rate carry** (+9%/yr backtested, profitable every year since 2022, market-neutral), now in a 30-day live shadow run.

---

## Day-38 Live Review (July 18, 2026) — findings

**In plain terms**: the 30-day window ran long (reviews were 8 days overdue) but the
verdict is good where it matters: the strategy is being executed live *exactly* as
the backtest promised — the market is just paying less rent right now. Separately,
the OKX practice book turned out to be broken in an instructive way, and two of our
own monitoring scripts had bugs (all fixed).

**Shadow book (the strategy test): 6/7 checklist PASS.** Coverage 100% (116/116
cycles), turnover 0.099 vs 0.11 promised, synthetic maker-fill lower bound 100%
(68/68), OKX real post-only fill rate 96% on $785k placed (0 post-only rejects) —
**the maker-execution assumption, the strategy's biggest unprovable, is now
validated with real orders**. Beta to BTC +0.055 (neutral as designed). Equity
−4.59% is inside the 2σ noise band.

**The one FAIL — funding collection (+0.0043%/event vs the +0.0070% constant,
z=−12) — attributed via `shadow_vs_backtest.py`**: the frozen backtest run over the
SAME 38-day window also collects +0.0042%/event; live-vs-backtest funding corr is
**0.98**, paired-t 0.4. → The implementation is faithful; the shortfall is the 2026
low-funding REGIME (already noted June 10). Lesson recorded: checklist constants
for regime-dependent quantities must be window-matched, not unconditional 5-year
means — `shadow_vs_backtest.py` is now the standing tool for that.

**OKX demo book post-mortem (root cause found July 18)**: the demo account is
seeded with 1 BTC + 1 ETH + 100 OKB + ~$1.5k USDT (≈$75k total), but in
single-currency margin mode only USDT counts as margin for USDT-swaps. The
executor sizes off totalEq (~$75k) → every position wants ~$12.5k → only ATOM ever
fully filled, consuming nearly all usable margin → 224× error 51008, book frozen
for weeks as a ~$11.5k naked ATOM long + dust. NOT a strategy failure — an
execution-lab account-setup failure. The fill-rate measurements remain valid.
*Fix path*: convert the demo BTC/ETH/OKB to USDT (or enable multi-currency margin),
flatten the stray book, restart the loop. Zombie-order hypothesis tested and
rejected (0 open orders, ordFrozen=0; `okx_demo_diag.py` is the tool).

**Ops repairs shipped (July 18, laptop → git → HPC)**: `paper_review.py` and
`heartbeat_carry.py` fixed for the mixed 11/13-column history CSV (schema grew
mid-run); heartbeat had also silently VANISHED from the HPC (it was never
committed — now tracked; watchdog was down the whole window and it only monitors
the shadow book, not the OKX book — extension TODO); perp fetcher made incremental
(was skip-if-exists, silently serving June-10 data); MATIC delisted from Binance
futures mid-window (universe now 13 — handled gracefully, recorded here).

**Graduation decision: PENDING** (options: extend shadow in thin regime while
hunting brick #2 — OI data is now ripe — vs. graduate to small live capital with
expectations reset to the regime, ~+4–5%/yr).

---

## THE LIVE MACHINE (what is running right now, unattended)

| When (CEST) | Job | What it does |
|---|---|---|
| 02:05 / 10:05 / 18:05 | **Shadow book** (`paper_trade_carry.py`) | Trades the frozen strategy with imaginary money against real prices/funding; also measures whether our patient orders *would* have filled (synthetic fill check) |
| 02:10 / 10:10 / 18:10 | **OKX demo book** (`okx_demo_trade_carry.py`) | Places REAL limit orders on OKX's fake-money exchange — measures actual fill behavior on a venue we could legally go live on |
| :30 every hour | **Heartbeat** (`heartbeat_carry.py`) | Pings Raj's phone (ntfy topic `cifr-carry-rj76x2`) if a book stops updating or drops 10% from peak |
| Sun 03:00 | **OKX funding accumulator** | Saves OKX funding history weekly (their API only keeps ~3 months — we must build our own archive) |

**In plain terms**: two robots trade the same strategy every 8 hours — one on paper, one with real orders on a practice exchange — while a watchdog guards them. No human action needed.

**Why this setup**: a backtest can be accidentally cheated (you already know the future when you write it). The only un-cheatable test is running the strategy *forward* on data that didn't exist yet. The 30-day window verifies: (a) the strategy behaves live like the backtest promised, (b) our patient limit orders actually fill (the one cost assumption a backtest can't prove), (c) the operations don't break. **Important**: 30 days of profit/loss is statistically meaningless for a Sharpe-0.5 strategy — we judge the checklist, never the PnL sign.

**Checkpoints**: ~June 24 (two-week read of the CSVs), ~July 10 (day-30 graduation review → live-capital conversation).
Logs: `results/paper/{carry,okx_demo}_history.csv`, `slurm/logs/*.log`.

---

## BRICK #1: Funding-Rate Carry (the firm's first validated strategy — FROZEN)

**The strategy in one sentence**: every 8 hours, rank 14 crypto perpetuals by their average funding over the last 3 days; bet AGAINST the 3 with the most expensive funding (short them) and ON the 3 with the cheapest (long them), equal money each side.

**In plain terms — why this makes money** (two engines, both proven in the data):
1. **Collection**: shorting expensive-funding coins means the crowd of leveraged longs literally pays us cash every 8 hours (and longs on negative-funding coins get paid too). Like collecting rent. (+35% of backtest PnL)
2. **Prediction**: coins with expensive funding are crowded trades, and crowded trades underperform over the next hours. We're short exactly those. (+31% of backtest PnL)
The counterparty losing money is the leveraged trader knowingly paying for leverage — an edge with an identifiable, willing payer, which is why it persists.

**The numbers**:
- Diagnostic: 8h cross-sectional IC −0.020, **t = −4.83**, negative in ALL 5 years (2022–26), strongest in 2026 (not decaying). 93% of assets agree on sign.
- Backtest (frozen config `v2_maker_8h`: K=3, 3-day smoothing, exit band 2, 8h cadence, maker fees 0.03%/side): **+9.0%/yr, Sharpe 0.52, NET POSITIVE ALL 5 YEARS** (2022 +14.9 / 2023 +22.6 / 2024 +1.8 / 2025 +4.1 / 2026 +5.5), max drawdown −30%.
- Cross-venue check: the same effect exists on OKX's own data (t = −2.68 on the available 3 months, 13/13 assets agree) → it's a market-wide phenomenon, not a Binance quirk.

**Why each design choice** (every number was measured, not guessed):
- **8h cadence**: funding only updates every 8h (no new info sooner), the predictive edge dies after ~1 day (no point slower), and 8h beat daily in A/B (Sharpe 0.52 vs 0.43).
- **3-day smoothing + exit band**: the naive version flipped half the book every cycle and lost **−91% to fees despite +124% gross profit**. Smoothing the ranking and letting incumbents keep their seats cut trading 8× — that single fix is the difference between the strategy existing and not.
- **Maker (patient) orders**: fee math only works at ~0.03%/side; instant (taker) execution at 0.08% eats the edge. Hence the fill-rate measurement obsession.
- **No stop-losses per position**: these aren't directional bets — the book is hedged as a whole (dollar-neutral), and a per-position stop would *un-hedge* it at the worst moment. Risk control = neutrality + position caps + the −10% portfolio tripwire + the 8h rebalance itself.
- **FROZEN**: 6 configurations were tried (counted!); the best-of-6 Sharpe is optimistically biased, and every further tweak risks fitting noise. Any change = a new strategy that must re-earn its gate.

**Honest weaknesses**: equity-curve evidence alone is weak (t≈1.1 — the credibility comes from the diagnostic + the mechanical funding leg + 5/5-year consistency); maxDD −30% vs +9%/yr is poor (fix = portfolio-level vol targeting once brick #2 exists, not carry tweaks); maker fill risk unproven until the live measurements land; single-name blowup risk (a short pumping 50% costs ~8%).

---

## Phase 1 Post-Mortem: The Kronos Thesis, Tested and Falsified (June 6–10, 2026)

**In plain terms**: the original idea was "a big AI model trained on millions of price charts can predict the next candles; trade those predictions, with an LLM on top adapting strategies to the market." We built all of it, then measured the predictions themselves — and they turned out to be statistically indistinguishable from coin flips, every way we sliced it. The strategies lost money not because they were badly designed, but because there was nothing upstream to trade on.

### What was built (works, reusable)
Finetuned checkpoints (BTC v1/v2, XAU v1/v2); batched MC path generation (`src/model/batched_inference.py`, ~20x speedup — Kronos's own API silently *averages* paths); CQR calibration, 21 assets, 90% target → 91-92.5% achieved; **forecast cache** (run the GPU model once, test strategies CPU-only in seconds — the key enabler of fast iteration); **pluggable strategy engine** (swap strategies/sizers without touching the engine; proven equivalent to the original by regression test); **skill diagnostics** (the statistical gauntlet — became the heart of v2); Hurst+ADX regime classifier (tested, net-negative).

### The evidence chain (each step closed one escape hatch)
1. **First backtest**: crypto −23.4%, commodity **+16.0%** (Sharpe 4.36!). *Why we didn't celebrate*: same engine, opposite results = something structural, and Sharpe 4 on one window smells like luck. Investigate, don't ship.
2. **Strategy A/B sweep** (regime gates, mean-reversion): every variant lost. *Telltale*: crypto had **0 take-profits in 2,341 trades** — the CQR bands were too wide to ever trigger; trades just rode 12h to timeout. The "strategy" was a pure direction bet.
3. **ATR exits** (working stops/targets): crypto *still* lost at every risk/reward setting. *The smoking gun*: with symmetric stops (a fair coin would break even), win rate was **48.9%** — the direction call itself was sub-coinflip.
4. **Direct skill measurement** (decouple model from strategy): crypto hit rate 50.8% (noise), IC +0.004 (noise). Model "confidence" meaningless (82% of forecasts pinned at max confidence, which hit 50.7%). Cross-sectional ranking (can it pick relative winners?): t = +0.76 — nothing. **And commodity's +16% exposed**: coinflip direction + a strongly trending quarter = the profit was the market's trend (beta), not skill (alpha).
5. **Every horizon tested** (h=1..48): IC ≈ 0 everywhere. No "it works at short horizons" escape. (Commodity's exciting small-sample blip, IC +0.29 on n=24, collapsed to +0.05 at n=384 — the canonical small-sample trap.)
6. **Volatility forecasting** (last hypothesis): the model's uncertainty bands DO correlate with future volatility (+0.34) — but a free, two-line "tomorrow's vol ≈ recent vol" rule scores **+0.61**. The model is a worse version of a moving average.

### Verdict
| Capability | Result |
|---|---|
| Price direction (any horizon, both markets) | none |
| Relative ranking of assets | none |
| Volatility forecasting | worse than naive persistence |
| Uncertainty calibration (CQR) | works — but only because vol persists |

### Why it failed — the three lessons that define v2
1. **Edge lives in the inputs, not the model.** Price candles are the most-analyzed dataset on earth; whatever was predictable in them was arbitraged away years ago. Kronos trained on 12B candles and learned nothing tradeable — a bigger in-house model on the same data would learn the same nothing at 1000× the cost. *Consequence*: change the data, never the model size.
2. **Intelligence downstream of a coin flip is still a coin flip.** The planned LLM strategy-picker consumed Kronos's output; no reasoning can add information that isn't in its input. (We ran that layer's job manually via the strategy sweep — every variant lost.) *Consequence*: LLMs re-enter UPSTREAM, turning text (news/sentiment) into new input data — never as a strategy chooser.
3. **The +16% trap.** A gorgeous single-window backtest that was pure market trend. Without the statistics-first discipline we'd have shipped it and been wrecked in the next flat quarter. *Consequence*: the methodology rules below are non-negotiable.

---

## Phase 2: The Signal Factory (June 10) — candidates tested, one survivor

**In plain terms**: instead of one big model, we now test simple trading ideas one at a time against hard statistics. Most die — that's the point and that's normal (industry hit rates are low). Each test costs minutes and pennies. The survivors get backtested with realistic fees, and only then traded.

**The scoreboard (5 candidates in one day)**:
| Candidate | Idea in plain terms | Verdict & why |
|---|---|---|
| **Funding carry** | bet against coins whose leveraged longs are paying heavily | ✅ **Brick #1** — t=−4.83, stable 5/5 years |
| XS momentum | recent winner coins keep winning vs losers | ❌ statistics incoherent — no there there |
| Short-term reversal | last week's losers bounce back tomorrow | ❌ passed stats marginally, but the *portfolio* lost every year — see lesson below |
| Tail momentum | the mirror trade of reversal | ❌ was real in 2022, decayed to zero by 2026 |
| Trend-timing (TSMOM) | each coin: long when rising, short when falling | ❌ 2025 negative in every variant; can't beat noise |

**The reversal lesson (a gauntlet upgrade we paid −105% in backtest ink to learn)**: the statistic said "coins mildly mean-revert" — true *on average across all 14*. But a real portfolio only trades the *extremes*, and in crypto the extremes do the opposite (crashing coins keep crashing — LUNA-style; pumping coins keep pumping). **Rule added: every diagnostic must also test exactly what the portfolio would trade (bottom-K minus top-K), not just the average relationship.**

**Why simple statistics and no ML here**: the carry relationship is nearly linear — a rolling average and a sort capture it fully; a neural net would add overfitting risk and remove interpretability. **Complexity must be earned**: ML returns when there are several validated signals to *combine* (a genuine ML problem); LLMs return when we want features from *text* (a genuinely-LLM problem).

**❌ COT positioning (June 10)** — commodities' cheap candidate, killed on 40 years
of data: spec-crowding percentile mean IC ≈ 0, sign coinflip across 6 commodities,
the one tempting cell (4w XS t=−2.30) fails multiplicity AND its construction-
matched spread AND 3-year-bucket stability (sign flips every bucket since 1998).
Matches modern literature: the simple COT fade was arbitraged flat decades ago.
*Infrastructure keeps*: `fetch_cot.py` (CFTC API, weekly since 1986, incremental)
+ daily futures prices to 2000 — reusable as features/data for later candidates.

**Remaining candidate queue**: open-interest/squeeze signals (OI history
accumulating since June 10 — ripe ~end of June); **commodity TERM-STRUCTURE carry**
(the classic backwardation/contango signal — now the declared next commodity
candidate; needs stitching yfinance contract chains, ~a day of data engineering —
deliberate project for the shadow window, not a quick run); Phase 3 text features.

---

## Phase 4 events (June 10): going live on fake money — and what we hit on the way

**In plain terms**: we put the strategy on autopilot with imaginary money, tried to also test it on Binance's practice exchange, discovered Binance is closed to the Netherlands, pivoted to OKX (which is legal here), verified the strategy's edge exists on OKX's own data too, and left a four-job autonomous machine running.

- **Shadow book live** (8h cron). *Why first*: cheapest un-cheatable test; measures signal-vs-backtest consistency.
- **Synthetic fill measurement** added into the shadow book: record the live bid/ask at decision time; 8h later check if price traded through it. *Why*: gives a pessimistic lower bound on "do our patient orders fill?" using only public data — no account, no geo-block can take it away.
- **Binance testnet dead end**: old testnet down for revamp; new demo geo-blocked (**Binance withdrew from NL in 2023**). *Strategic discovery*: the live venue can never be Binance — candidates are **OKX** (centralized, MiCA-licensed, legal in NL) or **Hyperliquid** (permissionless DEX). `testnet_trade_carry.py` kept but unusable from NL.
- **OKX demo live** (04:31 UTC): the firm's first real orders — 5 post-only limits, 0 rejects, all contract-size math verified. 4 of 5 filled within the hour. Quirk: OKX demo lists no UNI perp → book occasionally 3-long/2-short (~17% net exposure) — flagged for the venue decision.
- **OKX validation** (`carry_skill --exchange okx`): the carry edge on OKX's own funding/prices: t=−2.68 over the 3 months their API serves, **13/13 assets agree on sign**, same strength as Binance over the identical window. *Why this mattered*: we trade OKX's funding if we go live there, not Binance's — Raj spotted this gap and the test closed it. OKX = provisionally validated live venue. *Data caveat handled*: OKX's API only serves ~3 months of funding history → fetcher made incremental/appending + weekly cron now archives it ourselves.
- Current regime note: funding is LOW everywhere right now (Binance 2026 harvest is negative). The cross-sectional design is insensitive to the overall level by construction — this is that design choice earning its keep.
- **Phase 4.5 DECIDED (June 10): the live venue is OKX.** Hyperliquid validation
  (carry_skill --exchange hyperliquid) FAILED: on the available window (HL's API
  caps price history at ~5,000 candles → only Nov 2025–Jun 2026 evaluable),
  per-asset 8h IC is +0.037 — the WRONG SIGN, positive in 87% of assets — while
  Binance (t=−2.79) and OKX (t=−2.68) both confirm the edge over the SAME window.
  Genuine venue difference (likely HL's hourly funding mechanism reverting
  intra-window + different crowd), not a period effect. *In plain terms*: the
  same bet that works on Binance/OKX has been backwards on Hyperliquid lately —
  so we don't trade it there. The HL "momentum" pattern is parked as a
  scan-discovered observation (zero attention until it earns its own gauntlet).
  HL data limitation recorded: past price history unretrievable beyond ~5k bars.

---

## Methodology rules (the constitution — each learned the hard way)

1. **Diagnostics before backtests** (|t| > 2 or no backtest). *Why*: backtests of noise produce beautiful accidents; statistics first makes lying to ourselves hard.
2. **Count every trial; deflate accordingly.** *Why*: try 6 configs and the best one's Sharpe is inflated by selection — we must discount for how much we searched.
3. **Multi-year stability, never one window.** *Why*: the +16% commodity trap — one good quarter proves nothing; 5-for-5 years is hard to fake.
4. **Construction-matched tests** (test what the portfolio actually trades). *Why*: the reversal corpse — averages and extremes can disagree violently.
5. **Costs modeled from day one; fee assumptions verified live.** *Why*: carry was a −91% strategy at naive costs and a +9% one executed patiently — implementation IS the strategy.
6. **Freeze what passes; changes re-earn the gate.** *Why*: endless tuning = fitting noise; the frozen config is the testable claim.
7. **Final test window touched once; live shadow = the real out-of-sample.** *Why*: data you've peeked at can't surprise you, and only surprises are evidence.

---

## What's Next (in order, with reasoning)

1. 🔄 **30-day shadow/demo window** (now → ~July 10). Reviews: day-14 (~June 24) read of funding collection + turnover + fill rates vs backtest; day-30 graduation checklist. *Why the wait*: the live measurements are the evidence the go-live decision will rest on; nothing useful can shortcut it.
2. ⏳ **OI/squeeze diagnostic** (~end June, when OI history is thick enough). *Why next*: data already accumulating, structurally different from carry (positioning vs flows) → a candidate second brick.
3. ⏳ **Phase 2D — portfolio layer** once ≥2 bricks: combine validated strategies with **vol-targeted sizing** (vol persistence IC 0.61 is free and proven — it's the risk layer, not the alpha). *Why*: fixes carry's −30% maxDD properly, and uncorrelated bricks raise portfolio Sharpe more than any single-strategy tuning could.
4. ⏳ **Phase 3 — LLM as a data source**: news/events/sentiment → numerical features → the same gauntlet. *Why upstream not downstream*: lesson #2 of the post-mortem; text is the one input where LLMs extract information nothing else can.
5. ✅/⏳ **Phase 4.5 — venue decided (OKX)**; remaining pre-live items:
   - **Infrastructure migration**: the live stack needs UPTIME, not compute — the
     HPC head node serves as a free always-on server today, but the account is
     tied to Raj's studies and a live firm shouldn't run on university infra.
     Before real capital: move the cron stack to a small owned VPS (~€5/mo,
     one afternoon — everything is already cron + git).
   - **Backups**: the accumulating OKX/HL funding archives + paper-trading track
     record live ONLY on the HPC (gitignored). Weekly rsync to the laptop
     (`results/paper/`, `data/raw/derivs_okx/`) until the VPS exists.
6. ⏳ **Phase 5 — platform** in thin slices as pain appears (alerting ✅ done; dashboard at graduation; research console + automated trial ledger when signal #2 enters; execution console + kill switch only when real capital). *Why thin slices*: building a pretty platform around one Sharpe-0.5 book is the classic small-fund failure mode.

### Closed items (and why)
- ~~Kronos in the signal path~~ — falsified (post-mortem above); checkpoints kept on disk.
- ~~Regime gating~~ — tested, killed the only profitable book; indicators kept for vol sizing.
- ~~Meta-labeling~~ — there's nothing upstream to filter.
- ~~LLM strategy-picker~~ — information can't be created downstream; reborn as Phase 3.
- ~~Final Kronos test-set eval~~ — moot at this level of significance.

---

## HPC Environment

- **Cluster**: UTwente SLURM, head nodes `hpc-head1/2.ewi.utwente.nl`
- **Best GPUs in main-gpu**: Lovelace (L40S/L40, 48GB) on hpc-node01-18, `--gres=gpu:lovelace:1`
- **Conda env**: `trade`. Activate: `source $(conda info --base)/etc/profile.d/conda.sh && conda activate trade` (direct path doesn't work)
- **HuggingFace**: cached on head node; jobs use `HF_HUB_OFFLINE=1` (compute nodes offline — also why all live-data crons run on the HEAD node, which has internet)
- **SLURM sizing**: 8 CPU / 32G backfills fast; 32 CPU / 128G blocks for days
- **Raj runs all SSH/sbatch/git/cron commands himself** — provide commands, he executes & pastes output
- **Commits**: Raj's credentials only, NEVER any Claude/AI attribution
- **Crontab**: the 4-job stack (see THE LIVE MACHINE) — `crontab -l` to inspect

---

## Development Environment

### HP Windows 11 laptop (dev machine; small checks/tests ONLY — all real workloads on HPC)
- **Repo**: `W:\cifr-quant` (W: = `subst` of `C:\Work Drive`)
- **Python env**: venv `mlenv` at `C:\Work Drive\envs\mlenv`; activate by typing `mlenv` in PowerShell. Python 3.12.10
- **GPU**: RTX PRO 1000 (Blackwell), torch cu128, CUDA works
- **Deps added**: `ccxt yfinance twelvedata anthropic python-dotenv` (do NOT install vectorbt/mapie — would downgrade shared numpy/pandas)
- **Kronos**: gitlink (commit `67b630e`), **no `.gitmodules`** → fresh clone leaves `Kronos/` empty. Restore: `git clone https://github.com/shiyu-coder/Kronos.git Kronos && cd Kronos && git checkout 67b630e`. **Edits inside `Kronos/` do NOT propagate via git — all new code goes in tracked `src/`.**
- **Not present locally** (gitignored, live on HPC): `data/`, `checkpoints/`, `results/`
- `PYTHONPATH` = repo root + `Kronos` when running locally

---

## File Map (updated June 10, 2026)

### The live trading stack (v2 active)
| File | Purpose |
|------|---------|
| `scripts/paper_trade_carry.py` | Shadow trader — FROZEN config lives here (K=3, SMOOTH=9, EXIT_BAND=2, 0.03%/side) + synthetic fill measurement |
| `scripts/okx_demo_trade_carry.py` | OKX demo executor — real post-only orders; imports the frozen signal from paper_trade_carry (single source of truth) |
| `scripts/heartbeat_carry.py` | Staleness + drawdown-tripwire watchdog → ntfy push |
| `scripts/testnet_trade_carry.py` | Binance demo executor — kept, but geo-blocked from NL |

### The signal factory (v2 active)
| File | Purpose |
|------|---------|
| `src/data/derivs_client.py` | Venue-agnostic funding/perp/OI fetchers (funding fetch is INCREMENTAL — appends) |
| `scripts/fetch_derivs.py` | Driver: `--exchange binanceusdm\|okx`, `--what funding\|perp\|oi` |
| `scripts/carry_skill.py` | The carry diagnostic (IC/t-stats/stability/harvest, `--exchange` flag) |
| `scripts/backtest_carry.py` | The earned carry backtest (smoothing, band hysteresis, funding cashflows, cost knobs) |
| `scripts/momo_skill.py` | XS momentum + reversal diagnostic + the construction-matched tail-spread test |
| `scripts/tsmom_skill.py` | Time-series momentum diagnostic (equal-notional + canonical vol-scaled) |
| `scripts/backtest_reversal.py` | Reversal backtest (failed; kept as record) |

### v1 infrastructure (reusable; model-dependent parts dormant)
| File | Purpose |
|------|---------|
| `src/backtest/strategy_api.py`, `strategies.py`, `sizing.py`, `portfolio_engine.py`, `grid.py` | Pluggable multi-asset walk-forward engine |
| `src/backtest/costs.py`, `metrics.py` | Cost models; Sharpe/drawdown/deflated-Sharpe |
| `src/model/forecast_cache.py`, `build.py`, `batched_inference.py`, `ensemble.py` | Kronos stack (dormant) — cache pattern + batched MC are the reusable ideas |
| `src/regime/indicators.py`, `classifier.py` | hurst/adx/atr/realized_vol (reusable for vol sizing); composite gate (retired) |
| `src/risk/cqr.py` | Conformal calibration (works) |
| `scripts/forecast_skill.py`, `horizon_skill.py`, `vol_skill.py` | The Kronos falsification diagnostics — template for all future signal tests |
| `docs/STRATEGY_DESIGN.md` | Signed-off pluggable-architecture design doc |

### Key results on HPC (`results/`)
| Path | Contents |
|------|----------|
| `results/paper/` | LIVE: carry_state/history, okx_demo_state/history, heartbeat_state |
| `results/backtest/carry_backtest_v2_maker_8h.json` | The frozen brick's backtest |
| `results/backtest/{portfolio,reversal,carry}_backtest_*.json` | The full trial ledger (count these for deflation!) |
| `results/cqr/`, `results/forecasts/` | v1 artifacts (calibrations; forecast cache) |
| `data/raw/derivs/`, `data/raw/derivs_okx/` | Funding/perp/OI per venue |

---

## Common Pitfalls & Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| Small-n IC looks amazing | n=24 commodity IC +0.29 → +0.05 at n=384 | Never read an IC without n and a t-stat |
| Single-window Sharpe 4+ | Regime beta masquerading as alpha | Multi-year stability + decouple signal from strategy |
| Great IC, losing portfolio | Averages vs extremes (reversal corpse) | Construction-matched tail-spread test |
| Gross profit, net disaster | Turnover × fees (carry v1: −91% net on +124% gross) | Smooth signals, hysteresis bands, maker execution |
| Cron fires at wrong time | Head node is CEST, funding is UTC | Schedule in local time: 2,10,18 |
| Binance anything 404s | Binance withdrew from NL (2023) | OKX (MiCA) for accounts; public Binance data still fetches fine |
| OKX funding history short | API serves ~3 months only | Incremental fetch + weekly cron accumulates |
| OKX order sizing wrong | Swaps are sized in CONTRACTS (contractSize varies per coin) | Conversion handled in okx_demo_trade_carry.py |
| OKX order fails re posSide | Account in hedge mode | Demo settings → One-way position mode |
| Login-node python = CPU | No GPU on hpc-head | GPU work via sbatch; network-only crons on head node |
| SLURM job stuck PD days | Oversized request blocks backfill | 8 CPU / 32G |
| Long job times out, loses all | Results written only at end | Incremental save + resume |
| Kronos paths averaged | `predict(sample_count=N)` averages | `src/model/batched_inference.predict_paths` |
| HuggingFace on compute nodes | Offline | Cache on head node, `HF_HUB_OFFLINE=1` |
| Conda activation fails | Direct path doesn't exist | `source $(conda info --base)/etc/profile.d/conda.sh` |
| Git push rejected on laptop | Remote ahead | `git pull --rebase` then push |

---

*Single source of truth for project state. Update after every significant change.*

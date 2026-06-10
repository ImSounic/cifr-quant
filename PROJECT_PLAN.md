# CIFR-QUANT: Project Plan

> This plan has plain-language explanations (**In plain terms**) and reasoning (**Why**)
> throughout, so it stays readable without the technical background. Companion file:
> PROJECT_STATE.md (current status, glossary, full evidence chains).

## Overview (v2 — June 10, 2026)

CIFR-QUANT is a systematic-trading research project building toward a quant firm. It is now a **signal factory**: a validated research pipeline (skill diagnostics → pluggable walk-forward backtest → portfolio construction) pointed at **information-bearing market data** — starting with crypto derivatives data (funding rates, open interest, liquidations, basis) where documented, capacity-constrained edges exist that are accessible at small scale.

**In plain terms**: a quant firm is not one magic AI model — it's a *factory* that takes trading ideas in one end and, for each, answers "is this statistically real, and does it survive realistic trading fees?" cheaply and honestly. Most ideas die (normal!). The few survivors get traded, each small, together adding up. We built that factory, and its first survivor — a strategy that gets paid by over-leveraged crypto traders — is now running live on practice money.

**Why this shape**: we learned it the hard way. v1 (June 2025 – June 10, 2026) bet everything on one sophisticated model (Kronos) reading price charts, and was rigorously falsified — see "How the Project Changed" below. The factory approach is the opposite: many cheap, simple, statistically-policed bets on *data that still contains information*. The v1 plan is preserved at the bottom of this document as the historical record; its infrastructure survives, its thesis does not.

**Status snapshot (June 10, 2026)**: Brick #1 (funding carry, +9%/yr backtested, profitable every year 2022–26, market-neutral) FROZEN and running in a 30-day live shadow + OKX-demo window, fully automated on cron. Four other candidates tested and killed in one day. Live venue provisionally selected (OKX — Binance is geo-blocked from NL). Next: day-14/day-30 reviews, then the second signal as data ripens.

**License**: CIFR-QUANT is proprietary (no license). Kronos (vendored, retired from the signal path) is MIT.

---

## How the Project Changed (June 10, 2026): v1 → v2

### What v1 believed
1. A foundation model pretrained on 12B+ candles (Kronos, 102M params), finetuned per market, could forecast price direction well enough to trade.
2. Ensemble MC paths + CQR calibration would turn those forecasts into tradeable bands (SL/TP, confidence, sizing).
3. An **LLM strategy layer** on top would read the forecasts and adaptively pick/tune strategies as market conditions changed.

### What we measured (June 8–10, the falsification week)
Every component was built and tested end-to-end, then the signal itself was isolated and measured with proper statistics. The full evidence chain lives in PROJECT_STATE.md ("Phase 1 Post-Mortem"). The headline results:

| Test | Result |
|---|---|
| 90-day walk-forward backtest, directional strategy | crypto −23.7%; commodity +14.1% — but the asymmetry was the clue |
| Strategy A/B sweep (regime-gated, mean-reversion, off forecast cache) | every variant negative; regime gate killed the only profitable book |
| ATR-exit sweep, symmetric RR 1.0 | crypto win rate **48.9% = sub-coinflip** — the smoking gun |
| Directional skill, time-series, every horizon h=1..48 | IC ≈ 0 everywhere, both markets (n=2516 / n=384) |
| Cross-sectional ranking skill (basis of market-neutral) | t-stat +0.76 (crypto) / −0.01 (commodity) — not significant |
| Model confidence (MC-path agreement) | degenerate: 82% pinned >0.80, that bucket = coinflip |
| Volatility forecasting (quantile spread vs realised vol) | IC +0.34/+0.44 — **loses to naive persistence (+0.61/+0.46)** |
| The commodity +14% | **beta, not alpha**: coinflip hit rate + trending energy quarter |

**Verdict: Kronos on OHLCV adds zero extractable information beyond trailing realised volatility.** Not a tuning failure — strategy structure, exits, gating, confidence, horizon and cross-section were all tested.

### The three lessons that define v2
1. **Edge lives in the inputs, not the model.** Kronos already trained on huge price data and has nothing — OHLCV is the most-arbitraged dataset on earth. Building a bigger in-house model on the same candles would learn the same nothing at 1000× the cost. Change the data, not the architecture.
2. **Information cannot be created downstream.** The v1 LLM strategy layer consumed Kronos's output; intelligence applied to a coinflip is still a coinflip. (The strategy sweep WAS that layer's job, run manually — every variant lost.) An LLM belongs **upstream**, where information enters: converting text (news, events, sentiment) into features. And "adaptively changing strategies with the market" without t-stat discipline is an overfitting machine — LLMs always have a confident narrative.
3. **The factory is the firm.** What survived v1 — forecast cache, pluggable strategy engine, IC/t-stat diagnostics, CQR, cost-aware walk-forward, HPC workflow — is precisely the machine that killed the bad thesis in days instead of with real money. A quant firm = (1) that factory, (2) a portfolio of small validated uncorrelated edges, (3) execution & risk. We have (1); v2 builds (2), then (3).

### What is retained vs retired
| Asset | Status |
|---|---|
| Skill diagnostics (`forecast_skill`, `horizon_skill`, `vol_skill` patterns) | **Core of v2** — every new signal goes through them first |
| Pluggable backtest engine + forecast/signal cache | **Core of v2** — signals plug in as strategies |
| CQR / uncertainty quantification | Retained (works as designed) |
| Vol persistence finding (IC 0.61 for free) | Retained as the **risk/sizing layer** (vol targeting), not as alpha |
| Regime indicators (`src/regime/`) | Retained for vol sizing; Hurst+ADX gate retired |
| Kronos ensembles, finetuned checkpoints, batched inference | **Retired from signal path** (kept on disk; batched-inference technique reusable) |
| LLM strategy layer v1 (`src/strategy/` as strategy picker) | **Cancelled as designed**; reborn in Phase 3 as text→feature extractor |
| Meta-labeling plan | Cancelled — nothing upstream to filter |

---

## The Plan Now (v2): Signal Factory

### Architecture

```
INFORMATION-BEARING INPUTS                    (Phase 2A / 3)
  funding rates · open interest · liquidations · basis · [later: text→LLM features]
        │
        ▼
SIGNAL CANDIDATES                             (Phase 2B/2C)
  funding carry · OI-change · liquidation cascades · slow xs-momentum
        │
        ▼
THE GAUNTLET — skill diagnostics FIRST        (existing infra)
  IC / rank-IC / t-stat / multi-window / per-asset & cross-sectional
  RULE: |t| > 2 across windows or the signal dies here. No backtest before this.
        │
        ▼
PORTFOLIO OF SURVIVORS                        (Phase 2D, existing engine)
  pluggable strategies · VOL-TARGETED sizing (persistence IC 0.61, free)
  cost models · drawdown halt · deflated Sharpe across all trials
        │
        ▼
PAPER TRADING → LIVE                          (Phase 4 — LIVE since June 10)
  shadow book + OKX demo loop · execution data feeds back into research
```

**In plain terms, top to bottom**: find data that still contains information → turn it
into candidate trading ideas → make each idea prove itself statistically (most die
here, cheaply) → trade the survivors together with strict risk sizing → rehearse on
fake money until the live numbers match the promised ones → only then real capital.
**Why this order**: every stage exists to catch a specific way of fooling yourself
before money is exposed; v1 skipped the third stage and paid for it in weeks of GPU
time instead of (thankfully) cash.

### Phase 2A — Derivatives data layer ✅ DONE (June 10)
- **What**: `scripts/fetch_derivs.py` — historical **funding rates** (4.9k events/asset since 2022), **perp 1h prices**, **open interest** for the tier-1/2 crypto assets, from Binance (and now OKX via `--exchange`). Funding fetches are *incremental* (append-only) so history accumulates on venues whose APIs only serve a window (OKX: ~3 months).
- **In plain terms**: we downloaded the record of who-paid-whom between long and short traders, every 8 hours, for 4.5 years, across 14 coins — for free.
- **Why this data**: price charts are picked clean (v1 proved it on our own data); funding rates are *flows of actual money driven by trader positioning* — newer, less arbitraged, and the documented home of small-scale edges. Liquidation history turned out to be paywalled — skipped.

### Phase 2B — First signal: funding-rate carry ✅ PASSED → BRICK #1 FROZEN
- **In plain terms**: coins whose leveraged longs are paying heavy funding are crowded trades — bet against them (and collect their payments); bet on the coins where funding is cheap. The loser funding our profit is the leveraged trader knowingly paying for leverage — an edge with a willing payer is an edge that persists.
- **Diagnostic**: t = −4.83, negative in all 5 years, 93% of assets agree — the first (and so far only) gate pass of the project. Kronos's best-ever was t = 0.76, for contrast.
- **Backtest journey (the implementation lesson)**: naive version made +124% gross and lost **−91% net to trading fees**. Two mechanical fixes — smooth the ranking over 3 days, let incumbents keep their seats (hysteresis), execute with patient maker orders — cut trading 8× and produced the frozen config: **+9.0%/yr, Sharpe 0.52, profitable all 5 years, market-neutral**. *Why frozen*: 6 configs were tried (counted); more tuning = fitting noise.
- The pure "harvest" variant (always short perp/long spot) was found regime-dependent (negative in 2022/2026) → rejected in favor of the cross-sectional book, which is insensitive to the overall funding level by construction.

### Phase 2C — Signal candidates 2..N through the same gauntlet 🔄 (4 tested, 4 killed — normal)
- **In plain terms**: we fed four more classic ideas into the factory in one day; the statistics killed them all before they could cost anything. A 1-in-5 hit rate is what this business actually looks like — the factory's job is making the failures cheap.
- ❌ Cross-sectional momentum (winners keep winning): incoherent statistics, closed.
- ❌ Short-term reversal (losers bounce back): *passed* the statistical gate marginally but the portfolio lost every year → exposed a gauntlet gap: **averages and extremes can disagree** — the middle of the pack reverts while crypto's extremes (death spirals, moonshots) keep going, and a portfolio trades the extremes. **Gauntlet upgraded**: every diagnostic now also tests exactly what the portfolio would trade (construction-matched tail-spread test).
- ❌ Tail momentum (the mirror): was real in 2022, decayed to zero by 2026.
- ❌ Trend-timing/TSMOM (long what's rising, short what's falling — both naive and the canonical vol-scaled construction): 2025 negative everywhere, nothing near significance.
- **Queue (blocked on data, not effort)**: OI/squeeze signals (~end June, when accumulated OI history is thick enough); commodity futures carry (needs futures-curve/COT data — commodities deprioritized NOT scrapped: spot candles simply don't contain term-structure information); Phase 3 text features.
- Models stay simple: z-scores, rankings, GBM at most. **Complexity is earned**: ML returns to *combine* several validated signals; never to conjure signal from empty data (that was v1).

### Phase 2D — Portfolio construction ⏳ (needs ≥2 bricks)
- Survivors combined with **vol-targeted sizing** (the one thing v1 proved predictable — vol persistence, IC 0.61 — is the risk layer, not the alpha).
- **Why**: fixes carry's main weakness (−30% maxDD) at the portfolio level, and uncorrelated bricks raise combined Sharpe more than tuning any single one ever could.

### Phase 3 — LLM as a data source ⏳ (the v1 idea, pointed the right way)
- Text → features: news, Fed/macro events, protocol incidents, sentiment → scored numbers → the same gauntlet.
- **Why upstream, not downstream**: v1's LLM was a strategy-picker consuming a coinflip — intelligence cannot add information that isn't in its input. Text is the one input where an LLM extracts information nothing else can. Allocation-across-strategies returns much later, rules-first.

### Phase 4 — Paper trading 🔄 LIVE (June 10)
- **In plain terms**: the strategy now runs itself every 8 hours — once with imaginary money against real prices (shadow), once placing real practice orders on OKX (demo) — with an hourly watchdog that alerts Raj's phone. Thirty days of this is the only test that can't be accidentally cheated, because the data didn't exist when the code was written.
- **What each loop proves**: shadow = "does live PnL track the backtest?"; synthetic fill check = pessimistic bound on "do patient orders fill?" from public data alone; OKX demo = real fill rates on a venue we could legally use.
- **Judged by checklist, not PnL sign** (30 days of a Sharpe-0.5 strategy is statistical noise): funding collection ≈ backtest rate, turnover ≈ 0.1–0.2/event, fill rate high, no tripwire, book matches frozen logic. Day-14 review ~June 24; day-30 graduation ~July 10.
- **Venue reality discovered en route**: Binance is geo-blocked from NL (withdrew 2023) → live venue = **OKX** (MiCA-licensed; carry edge independently confirmed on OKX's own data, t=−2.68, 13/13 sign agreement) or Hyperliquid (DEX) — decision at graduation, on the fill evidence being collected now.

### Phase 4.5 — Venue decision & live capital ⏳ (after graduation)
- Choose OKX vs Hyperliquid on measured fills + funding comparability; size initial capital small; live-vs-backtest divergence tracking continues forever (it's how edge death gets detected early).

### Phase 5 — Proprietary platform ⏳ (thin slices, as pain appears)
- **In plain terms**: eventually a proper dashboard/software instead of terminals — but built piece by piece exactly when each piece earns its keep, because months of UI around one small strategy is the classic small-fund failure.
- Order: (a) ✅ alerting/heartbeat (done — protects the 30-day run); (b) monitoring dashboard at graduation; (c) research console + **automated trial ledger** (every config and t-stat logged automatically — makes the multiple-testing discipline mechanical) when signal #2 enters; (d) execution console + kill switch only when real money.

### Methodology rules (the constitution — each clause is a scar)
1. **Diagnostics before backtests** (|t|>2 or no backtest). *Scar*: v1's beautiful backtests of noise.
2. **Count every trial; deflate the winner.** *Scar*: best-of-6 carry config's Sharpe is optimistic by selection.
3. **Multi-year stability, never one window.** *Scar*: the +14% commodity quarter that was pure market trend.
4. **Construction-matched tests** — test what the portfolio trades, not the average. *Scar*: reversal's −105%.
5. **Costs from day one; execution assumptions verified live.** *Scar*: carry was −91% naive, +9% patient.
6. **Freeze what passes; changes re-earn the gate.** *Scar*: every tuning pass after a pass is noise-fitting.
7. **Final test window touched once; live shadow is the real out-of-sample.**
8. **Realistic targets**: portfolio Sharpe 1.0–1.5 from several small edges = excellent. No single killer model exists.

---

# ⚠️ EVERYTHING BELOW THIS LINE IS THE v1 PLAN (HISTORICAL RECORD — SUPERSEDED June 10, 2026)

Kept intact for reference: the Kronos architecture, finetuning pipeline, CQR method, and v1 phase log. The thesis it encodes was falsified; see "How the Project Changed" above.

---

## System Architecture

```
  CRYPTO MARKET (Binance)                    COMMODITY MARKET (TwelveData)
  BTC ETH SOL AVAX LINK +20 assets           XAU XAG XPT WTI Brent NatGas Copper
  15m candles, 24/7                           4h candles, trading hours
         │                                            │
         ▼                                            ▼
  ┌──────────────────────┐                 ┌──────────────────────┐
  │  CRYPTO ENSEMBLE     │                 │  COMMODITY ENSEMBLE  │
  │  Zero-shot Kronos    │                 │  Zero-shot Kronos    │
  │  + BTC-finetuned     │                 │  + XAU-finetuned v1  │
  │  (transfers to all)  │                 │  + XAU-finetuned v2  │
  │  50 MC paths/asset   │                 │  50 MC paths/asset   │
  └──────────┬───────────┘                 └──────────┬───────────┘
             │                                        │
             └────────────────┬───────────────────────┘
                              │
               ┌──────────────▼───────────────┐
               │  CQR-CALIBRATED UNCERTAINTY  │
               │  Per-asset: SL (q05) / TP    │
               │  (q95) / confidence / width  │
               └──────────────┬───────────────┘
                              │
               ┌──────────────▼───────────────┐
               │  PORTFOLIO ORCHESTRATOR       │
               │  Filter: confidence ≥ 55%     │
               │  Risk parity: 1/interval_width│
               │  Max 10% per position         │
               │  Cross-asset correlation adj.  │
               │  Regime detection             │
               └──────────────┬───────────────┘
                              │
               ┌──────────────▼───────────────┐
               │  BACKTEST / LIVE EXECUTION    │
               │  Walk-forward, realistic costs│
               │  Target Sharpe 1.0-1.5        │
               │  across 25-35 assets          │
               └──────────────────────────────┘
```

---

## Two Market Strategy (Dual-Market Multi-Asset)

### Market 1: Crypto (Binance, 15m candles)

| Parameter | Value |
|-----------|-------|
| Assets | Tier 1 (5 majors): BTC, ETH, BNB, SOL, XRP. Tier 2 (10 large): ADA, AVAX, DOGE, DOT, LINK, MATIC, UNI, ATOM, LTC, NEAR. Tier 3 (10 mid): APT, ARB, FIL, INJ, OP, SUI, TIA, SEI, AAVE, MKR. |
| Default | Tier 1+2 (15 assets) |
| Timeframe | 15-minute candles |
| Context window | 512 candles = ~5.3 days (24/7 market) |
| Prediction horizon | 48 candles (12 hours) |
| Data source | Binance API (free, no auth for historical OHLCV) |
| Finetuned checkpoint | BTC-finetuned (`checkpoints/cifr-btc/`) — transfers to ALL crypto assets |
| Costs | Spread ~0.01%, slippage ~0.02%, commission ~0.04% maker (VIP0) |
| Rationale | Kronos pre-training heavily upweighted crypto. 24/7 market, no gaps. BTC checkpoint captures crypto-specific patterns (momentum cascades, leverage liquidations, microstructure) that transfer broadly. |

### Market 2: Commodities (TwelveData, 4h candles)

| Parameter | Value |
|-----------|-------|
| Assets | Precious: XAU/USD, XAG/USD, XPT/USD. Energy: WTI, Brent, Natural Gas. Industrial: Copper. |
| Default | Precious + Energy (6 assets) |
| Timeframe | 4-hour candles |
| Context window | 512 candles = ~85 trading days (~3 months) |
| Prediction horizon | 6 candles (1 day) |
| Data source | TwelveData API (free tier: 800 req/day) |
| Finetuned checkpoint | XAU-finetuned v1 + v2 (`checkpoints/cifr-xau/`, `checkpoints/cifr-xau-v2/`) — transfers to ALL commodities |
| Costs | Spread ~0.015%, slippage ~0.005%, commission ~0.01% |
| Volume note | Spot has no real volume. Use `volume=0, amount=0` — Kronos handles this. Futures symbols (GC=F, SI=F, CL=F) have real volume. |
| Rationale | XAU ensemble showed best IC (+0.393). Commodity macro dynamics (real rates, central bank flows, risk-off) transfer across precious metals and energy. Decorrelated from crypto. |

### Why Not Forex?

EUR/USD was dropped because: no centralized volume data, weekend gaps require special handling, weaker model fit (IC=-0.264), and adding a third market with one asset doesn't help breadth. The EUR v1/v2 checkpoints remain available if needed.

### Transfer Learning Key Insight

Finetuning on one representative asset per market captures market-level patterns that transfer to all assets in that market. This is far more capital-efficient than finetuning 25+ individual checkpoints, and the shared patterns (market microstructure, regime dynamics) are more important than asset-specific idiosyncrasies for Kronos's price-level predictions.

---

## Model Details

### Why Kronos-base (102M)

| Model | Params | Context | Status | Notes |
|-------|--------|---------|--------|-------|
| Kronos-mini | 4.1M | 2,048 | Open | Too small, weak predictions despite long context |
| Kronos-small | 24.7M | 512 | Open | Good for prototyping, use for pipeline validation |
| **Kronos-base** | **102.3M** | **512** | **Open** | **Best available. Scaling laws confirmed: bigger = better** |
| Kronos-large | 499.2M | 512 | Closed | Not open-sourced, institutional licensing only |

### Architecture Internals

- **Type**: Decoder-only GPT-style Transformer
- **Position encoding**: RoPE (Rotary Position Embedding)
- **Normalization**: RMSNorm
- **Tokenizer**: Binary Spherical Quantization (BSQ), vocabulary size 2^20 (~1M tokens)
- **Tokenizer structure**: Hierarchical — upper 10 bits (coarse: price direction, volume level) + lower 10 bits (fine: exact positions of highs/lows)
- **Temporal embeddings**: 5 types summed together — minute, hour, day-of-week, day-of-month, month
- **Input dimensions**: D=6 — Open, High, Low, Close, Volume, Amount (OHLCVA)
- **Pre-training**: Autoregressive next-candle prediction on 12B+ K-line records from 45 exchanges
- **Inference**: Monte Carlo sampling with temperature (T) and nucleus sampling (top_p)

---

## Finetuning Pipeline

### Use `finetune_csv/` (NOT Qlib pipeline)

> **IMPORTANT**: Kronos provides TWO finetuning paths:
> 1. `finetune/` — designed for Chinese A-share data via Microsoft Qlib. **Not suitable for us.**
> 2. `finetune_csv/` — designed for arbitrary CSV data. **This is what we use.**
>
> The CSV pipeline accepts standard OHLCV CSVs and does not require Qlib installation.

### Two-Stage Finetuning (per market)

**Stage 1 — Finetune Tokenizer (~2 hours per checkpoint)**

Adapts the BSQ codebook to the specific market's statistical distribution. A tokenizer trained on 45 exchanges has generalized representations; finetuning aligns high-frequency codebook entries with the target instrument's typical candle patterns.

```bash
torchrun --standalone --nproc_per_node=NUM_GPUS finetune_csv/train_tokenizer.py
```

**Stage 2 — Finetune Predictor (~4 hours per checkpoint)**

Adapts the transformer weights to predict the specific instrument's next-candle sequence. The model learns instrument-specific temporal patterns, volatility regimes, and session effects.

```bash
torchrun --standalone --nproc_per_node=NUM_GPUS finetune_csv/train_predictor.py
```

### Data Volume Considerations

| Market | Timeframe | Candles/year | 2 years | 5 years | Sufficient for finetuning? |
|--------|-----------|-------------|---------|---------|---------------------------|
| BTC/USDT | 15-min | ~70,000 | ~140k | ~350k | 2 years is plenty |
| EUR/USD | 1-hour | ~6,200 | ~12.4k | ~31k | 2 years is marginal, 5 years preferred |
| XAU/USD | 4-hour | ~1,500 | ~3k | ~7.5k | **2 years is too few.** Use 5+ years, or augment with XAG/USD (silver) and GC=F (gold futures) |

For gold, consider finetuning on a basket of commodities (gold, silver, crude oil) to increase data volume, then evaluate on gold specifically.

### Data Splits (per market)

```
|<──────── Train ────────>|<─ Cal ─>|<── Val ──>|<── Test ──>|
|     Everything before   | Months  | Months    | Last 3     |
|     last 9 months       | -9 to -6| -6 to -3  | months     |
|                         | (CQR    | (early    | (UNTOUCHED |
|                         | calib.) | stopping) | until final|
|                         |         |           | evaluation)|
```

- **Train**: All data except last 9 months — used for tokenizer + predictor finetuning
- **Calibration**: Months -9 to -6 — used ONLY for CQR conformity scores (after finetuning is frozen)
- **Validation**: Months -6 to -3 — used for early stopping, hyperparameter selection
- **Test**: Last 3 months — touched exactly ONCE at the end for final evaluation

> **Why 4-way split?** Using the same validation data for both finetuning early stopping
> and CQR calibration introduces data leakage — the model was partially fit to validation
> data via early stopping, so CQR conformity scores computed on it would be optimistic.
> A separate calibration set keeps CQR coverage guarantees valid.

### Hardware Requirements

| Task | GPU | VRAM | Time |
|------|-----|------|------|
| Finetune tokenizer (per market) | L40S (Lovelace) | ~8 GB | ~1-2.5 hours |
| Finetune predictor (per market) | L40S (Lovelace) | ~12 GB | ~2-8 hours |
| Inference (per market) | Mac Mini 16GB / any GPU | ~4 GB | seconds |
| **Total for 3 markets** | | | **~12–18 hours** |

**Development**: Mac Mini 16GB (data pipeline, inference, backtesting)
**Training**: UTwente HPC SLURM cluster — L40S (Lovelace) GPUs preferred over A40 (Ampere), ~1.5-2x faster

### Ensemble Architecture

Based on research into how top quant firms (Renaissance, Two Sigma) operate:

```
┌────────────────────────────────────────────────────┐
│                 ENSEMBLE PREDICTOR                  │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ Zero-Shot│  │Finetuned │  │Finetuned │  ...     │
│  │  Kronos  │  │  v1      │  │  v2      │         │
│  │  (base)  │  │(seed=42) │  │(seed=137)│         │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘         │
│       │              │              │               │
│       └──────────────┼──────────────┘               │
│                      │                              │
│              Weighted Average                       │
│         (equal or inverse val loss)                 │
│                      │                              │
│              ┌───────▼────────┐                     │
│              │ Monte Carlo    │                     │
│              │ Path Sampling  │                     │
│              │ (50 paths)     │                     │
│              └───────┬────────┘                     │
│                      │                              │
│              ┌───────▼────────┐                     │
│              │ CQR-Calibrated │                     │
│              │ Quantile Bands │                     │
│              └───────┬────────┘                     │
│                      │                              │
│         Direction + Confidence + SL/TP              │
└────────────────────────────────────────────────────┘
```

**Key insight (Grinold's Law)**: IR = IC × √Breadth. Two models with IC=0.05 and low correlation produce combined IC >> 0.05. Diversity (different seeds, zero-shot vs finetuned) is more valuable than individual model accuracy.

---

## Quantile Risk Management

### Method: Monte Carlo Paths + Conformalized Quantile Regression

**Step 1 — Generate N forecast paths**

> **CRITICAL IMPLEMENTATION NOTE**: Kronos's `predict()` with `sample_count=N` generates N paths
> and **averages them into a single DataFrame**. It does NOT return individual paths. To get
> 30 separate trajectories for quantile analysis, we must call `predict()` 30 times with
> `sample_count=1` and collect each result independently.

```python
# CORRECT: Call predict() N times to get N individual paths
paths = []
for i in range(30):
    path_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,          # Temperature controls randomness
        top_p=0.9,      # Nucleus sampling
        sample_count=1   # Single path per call
    )
    paths.append(path_df)

# paths is now a list of 30 DataFrames, each a distinct future trajectory
```

Each call with `sample_count=1` and `T > 0` produces a stochastically sampled path. Temperature-controlled autoregressive generation ensures each path is different.

**Step 2 — Extract empirical quantiles**

From the 30 paths, compute:
- **5th percentile of lows** → Stop Loss level
- **50th percentile of closes** → Point forecast
- **95th percentile of highs** → Take Profit level
- **Interquartile range (25th–75th)** → Confidence band

These intervals are naturally adaptive: they widen in volatile regimes and tighten in calm ones.

**Step 3 — Conformalized Quantile Regression (CQR) calibration**

Raw quantiles from the model may be overconfident (the model doesn't know what it doesn't know). CQR adds a distribution-free correction:

1. Hold out a calibration set from validation data
2. Generate predictions and compute residuals against actual outcomes
3. Compute conformity scores: `score_i = max(q_lo - y_i, y_i - q_hi)`
4. Find the (1-α) quantile of scores as the correction term
5. Adjust all future intervals by this correction

**Result**: Prediction intervals with provable coverage guarantee (e.g., "90% of actual prices fall within this band") regardless of the underlying model's accuracy.

### Application to Trading

| Signal | Derivation | Use |
|--------|------------|-----|
| Stop Loss | 5th percentile low across 30 paths, CQR-adjusted | Hard stop below entry |
| Take Profit | 95th percentile high across 30 paths, CQR-adjusted | Limit order above entry |
| Position Size | Inversely proportional to interval width | Risk parity: wider interval = smaller position |
| Entry Confidence | Fraction of paths showing positive return | Only enter if >60% of paths agree on direction |
| Regime Detection | Variance of path endpoints | High variance = uncertain regime, reduce exposure |

---

## LLM Strategy Layer

### Role: Strategy Code Generator (NOT Direct Trader)

The LLM receives structured context and outputs executable Python strategy code. It never makes real-time trading decisions.

### Input Context Pack

```python
context = {
    "forecast": {
        "point_estimate": pred_df,           # Mean of 30 paths
        "confidence_band": (q25, q75),       # IQR
        "stop_loss": sl_level,               # CQR-adjusted 5th percentile
        "take_profit": tp_level,             # CQR-adjusted 95th percentile
        "directional_confidence": 0.73,      # 73% of paths show positive return
    },
    "regime": {
        "volatility_percentile": 0.82,       # Current vol vs historical
        "path_dispersion": 0.045,            # Variance of path endpoints
        "trend_strength": 0.61,              # Slope consistency across paths
    },
    "market_state": {
        "instrument": "BTC/USDT",
        "timeframe": "15min",
        "current_price": 67234.50,
        "session": "US",
        "recent_performance": {...},          # Last N trades P&L
    }
}
```

### Output: Executable Strategy

The LLM generates parameterized strategy code selecting from:
- **Trend following** — when directional confidence > 70% and trend strength > 0.5
- **Mean reversion** — when volatility percentile > 80th and price at band extremes
- **Breakout** — when path dispersion is low (compression) then expands
- **Volatility targeting** — position sized to target constant portfolio vol

### Guardrails

- LLM generates code, a deterministic engine executes it
- Maximum 5 refinement iterations per evaluation cycle
- Each iteration changes at most 2 parameters
- Test set (last 3 months) is touched exactly once
- All LLM-proposed strategies are logged with reasoning for audit

### Candidate Frameworks

- **TradingAgents** (58k+ GitHub stars) — multi-agent LLM framework with built-in analyst/trader/risk-manager agents
- **Custom Claude/GPT pipeline** — simpler, more control, less overhead

---

## Backtest & Evaluation

### Methodology: Walk-Forward Rolling Backtest

```
Window 1: [Train ──────][Predict][Eval]
Window 2:    [Train ──────][Predict][Eval]
Window 3:       [Train ──────][Predict][Eval]
...
```

The model is re-predicted (not re-trained) at each step using the rolling context window. Finetuning happens once; inference rolls forward.

### Realistic Cost Modeling

| Market | Spread | Slippage | Commission | Total per round-trip |
|--------|--------|----------|------------|---------------------|
| BTC/USDT (Binance) | ~0.01% | ~0.02% | 0.04% (maker) | ~0.07% |
| EUR/USD (OANDA) | ~0.8 pips | ~0.2 pips | 0 | ~1.0 pip |
| XAU/USD | ~0.30 USD | ~0.10 USD | varies | ~0.40 USD |

### Evaluation Metrics

**Signal Quality**:
- IC (Information Coefficient) — correlation between predicted and actual returns
- RankIC — rank correlation (more robust to outliers)
- Directional Accuracy — % of correct up/down predictions

**Strategy Performance**:
- Sharpe Ratio (annualized, after costs)
- Maximum Drawdown
- Calmar Ratio (return / max drawdown)
- Profit Factor (gross profit / gross loss)
- Win Rate
- Tail Ratio (95th percentile gain / 5th percentile loss)

**Statistical Rigor**:
- Deflated Sharpe Ratio (Bailey & Lopez de Prado) — adjusts for multiple testing
- Bonferroni correction when comparing multiple strategies
- Purged cross-validation for time series (prevent leakage)

---

## Implementation Phases

### Phase 1: Data Pipeline ✅ COMPLETE

- [x] Binance API client for BTC/USDT 15-min OHLCV (~97k candles)
- [x] TwelveData API client for EUR/USD 1-hour OHLCV (~23k candles)
- [x] TwelveData API client for XAU/USD 4-hour OHLCV (~9.5k candles)
- [x] Data cleaning, validation, Z-score clipping
- [x] Data splitting: tokenizer_train.csv, validation.csv, test.csv per market
- [x] Storage in `data/processed/{btc,eur,xau}/`

### Phase 2: Kronos Integration ✅ COMPLETE

- [x] Kronos cloned to `Kronos/` subdir, added to Python path
- [x] Pre-trained Kronos-base + Tokenizer-base loaded from HuggingFace (`NeoQuasar/Kronos-base`)
- [x] Model/tokenizer cached on HPC head node (compute nodes offline)
- [x] Zero-shot inference verified on all 3 markets
- [x] Zero-shot baseline: BTC IC=-0.34, DA=60%; EUR IC=-0.26, DA=40%

### Phase 3: Finetuning ✅ COMPLETE

- [x] YAML configs for all markets (`finetune/config_{btc,eur,xau,eur_v2,xau_v2}.yaml`)
- [x] SLURM scripts targeting Ampere/A40 GPUs (`slurm/finetune_*.sh`)
- [x] BTC v1 complete (tok val_loss=0.0027, pred trained ~10 epochs)
- [x] EUR v1 complete (val_loss=1.6098, 1h12m)
- [x] EUR v2 complete (val_loss=1.5521, 67min, seed=137)
- [x] XAU v1 complete (val_loss=0.9054, 1h39m) — overfitted
- [x] XAU v2 complete (val_loss=1.2316, 26min) — anti-overfitting config worked

### Phase 3.5: Ensemble Evaluation ✅ COMPLETE

- [x] `eval_ensemble.py` with 50 windows and bootstrap 95% CIs
- [x] XAU ensemble_full IC=+0.393 (best signal)
- [x] EUR ensembles DA=60% (best direction)
- [x] BTC finetuning improves RMSE, IC/DA flat (needs v2 for diversity)
- [x] Results saved to `results/ensemble/ensemble_metrics.csv`

### Phase 3.75: Architecture Redesign ✅ COMPLETE

- [x] Pivoted from 3-market×1-asset to 2-market×many-assets
- [x] Crypto universe defined: 3 tiers, 25 assets (`configs/crypto_universe.py`)
- [x] Commodity universe defined: precious+energy+industrial, 7 assets (`configs/commodity_universe.py`)
- [x] Portfolio orchestrator built (`src/portfolio/orchestrator.py`)
- [x] Multi-asset data fetcher created (`scripts/fetch_universe.py`)

### Phase 4: Multi-Asset Data + CQR Calibration 🔄 NEXT

- [x] Code written: `src/risk/cqr.py`, `src/risk/quantile.py`, `src/risk/position_sizer.py`
- [x] `EnsemblePredictor.predict_with_quantiles()` generates probabilistic forecasts
- [ ] Fetch crypto universe data (Binance, 15 assets tier 1+2)
- [ ] Fetch commodity universe data (TwelveData, 6 assets)
- [ ] Create BTC v2 config (different seed for crypto ensemble diversity)
- [ ] Run CQR calibration per-asset on calibration split
- [ ] Validate coverage guarantees (90% target)

### Phase 5: Multi-Asset Portfolio Backtest ⏳ PENDING

- [x] Code written: `src/backtest/engine.py`, `src/backtest/costs.py`, `src/backtest/metrics.py`
- [ ] Walk-forward backtest across full portfolio (crypto + commodities)
- [ ] Risk parity position sizing (1/interval_width)
- [ ] Portfolio-level Sharpe, drawdown, Calmar evaluation
- [ ] Per-asset contribution analysis

### Phase 5.5: Regime Detection ❌ NOT STARTED

- [ ] Implement HMM or volatility-based regime classifier
- [ ] Adjust signal weighting per regime (trending vs mean-reverting vs volatile)
- [ ] Adjust position sizing per regime
- [ ] This is the single highest-impact addition per quant research

### Phase 6: LLM Strategy Layer ⏳ PENDING

- [x] Code written: `src/strategy/llm_generator.py`, `src/strategy/executor.py`, `src/strategy/context_builder.py`
- [ ] Test strategy generation → backtest → evaluation loop
- [ ] Add guardrails (iteration limits, parameter budgets)

### Phase 7: Integration & Final Evaluation ⏳ PENDING

- [ ] End-to-end pipeline: data → ensemble forecast → CQR → regime → orchestrator → backtest
- [ ] Final test set evaluation (last 3 months, ONE TIME ONLY)
- [ ] Cross-market portfolio performance analysis
- [ ] Document results and findings

---

## Repository Structure

```
cifr-quant/
├── PROJECT_PLAN.md            # This document
├── README.md                  # Project overview
├── requirements.txt           # Python dependencies
├── .gitignore                 # Ignore data files, checkpoints, secrets
│
├── configs/                   # Market-specific configurations
│   ├── btc_config.py          # BTC/USDT: 15-min, Binance
│   ├── eur_config.py          # EUR/USD: 1-hour, OANDA
│   └── xau_config.py          # XAU/USD: 4-hour, MT5
│
├── data/                      # Data directory (gitignored)
│   ├── raw/                   # Raw OHLCV downloads
│   │   ├── btc/
│   │   ├── eur/
│   │   └── xau/
│   └── processed/             # Cleaned, split datasets
│       ├── btc/
│       ├── eur/
│       └── xau/
│
├── src/                       # Core source code
│   ├── data/                  # Data collection & processing
│   │   ├── __init__.py
│   │   ├── binance_client.py  # BTC data fetcher
│   │   ├── forex_client.py    # EUR/USD data fetcher
│   │   ├── gold_client.py     # XAU/USD data fetcher (yfinance GC=F or TwelveData)
│   │   ├── preprocessor.py    # Cleaning, validation, Z-score clipping
│   │   └── splitter.py        # Train/val/test temporal splits
│   │
│   ├── model/                 # Kronos model integration
│   │   ├── __init__.py
│   │   ├── predictor.py       # KronosPredictor wrapper
│   │   ├── sampler.py         # Multi-path Monte Carlo sampling
│   │   └── loader.py          # Model/tokenizer loading utilities
│   │
│   ├── risk/                  # Quantile risk management
│   │   ├── __init__.py
│   │   ├── quantile.py        # Empirical quantile extraction
│   │   ├── cqr.py             # Conformalized Quantile Regression
│   │   └── position_sizer.py  # Adaptive position sizing
│   │
│   ├── strategy/              # Strategy generation & execution
│   │   ├── __init__.py
│   │   ├── llm_generator.py   # LLM strategy code generation
│   │   ├── executor.py        # Strategy execution engine
│   │   ├── strategies/        # Strategy implementations
│   │   │   ├── trend.py
│   │   │   ├── mean_revert.py
│   │   │   ├── breakout.py
│   │   │   └── vol_target.py
│   │   └── context_builder.py # Build context packs for LLM
│   │
│   └── backtest/              # Backtesting engine
│       ├── __init__.py
│       ├── engine.py          # Walk-forward backtest runner
│       ├── costs.py           # Transaction cost models
│       ├── metrics.py         # Sharpe, drawdown, Calmar, etc.
│       └── report.py          # Generate performance reports
│
├── finetune/                  # Finetuning scripts (run on uni GPUs, uses Kronos finetune_csv/ pipeline)
│   ├── config_btc.py          # BTC finetuning config
│   ├── config_eur.py          # EUR finetuning config
│   ├── config_xau.py          # XAU finetuning config
│   ├── prepare_data.py        # Convert our data → Kronos CSV format
│   └── run_finetune.sh        # Shell script to run all 3 finetuning jobs
│
├── checkpoints/               # Finetuned model weights (gitignored)
│   ├── ckpt-btc/
│   ├── ckpt-eur/
│   └── ckpt-xau/
│
├── notebooks/                 # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_zero_shot_baseline.ipynb
│   ├── 03_finetuning_results.ipynb
│   ├── 04_quantile_analysis.ipynb
│   └── 05_backtest_results.ipynb
│
├── tests/                     # Unit tests
│   ├── test_data_pipeline.py
│   ├── test_predictor.py
│   ├── test_quantile.py
│   └── test_backtest.py
│
└── results/                   # Evaluation outputs (gitignored)
    ├── btc/
    ├── eur/
    └── xau/
```

---

## Key Technical Decisions

### 1. Why not extend the context window?

Kronos-base was trained at 512 tokens. While RoPE Position Interpolation or YaRN could theoretically extend it to 2048+, the tokenizer (Tokenizer-base) was co-trained for 512-length patterns. Extending the predictor without retraining the tokenizer creates a mismatch. Kronos-mini already offers 2048 context natively but with only 4.1M parameters — too weak. Instead, we use different timeframes per market to effectively control lookback within the 512 window.

### 2. Why separate checkpoints per market?

Different markets have fundamentally different statistical properties. Crypto is 24/7 with high vol and no gaps. Forex has session structure, lower vol, and macro-driven moves. Gold trends on multi-month cycles driven by real rates. A single finetuned model averaging these distributions would underperform specialized models for each.

### 3. Why LLM generates code, not trades?

Recent research (arXiv:2505.07078, May 2025) shows LLMs that excelled on static benchmarks performed worse in actual trading. LLMs are trained on text, not on stochastic price processes. They can reason about strategy design but can't reliably make moment-to-moment trading calls. The LLM proposes strategy logic; a deterministic engine executes it.

### 4. Why CQR over raw quantiles?

Raw empirical quantiles from Kronos's 30 sampled paths may be poorly calibrated — the model may be systematically overconfident or underconfident. CQR provides a distribution-free, finite-sample coverage guarantee. If you set 90% coverage, 90% of actual prices will fall within the adjusted interval, regardless of the model's internal calibration.

### 5. Why walk-forward instead of simple train/test split?

A single train/test split can be lucky or unlucky depending on the test period. Walk-forward rolling evaluation tests the model across multiple market conditions, giving a more robust estimate of out-of-sample performance. It also mirrors how the system would operate live — always predicting into unseen future data.

---

## Risk Management & Guardrails

### Overfitting Prevention

- Test set touched exactly ONCE (final evaluation only)
- Deflated Sharpe Ratio for multiple-testing correction
- Maximum 5 LLM refinement iterations per cycle
- Purged cross-validation for all hyperparameter searches
- Monitor live vs backtest performance divergence

### Operational Risks

- No live trading until backtest results are validated on test set
- All LLM-generated strategies logged with full reasoning chain
- Position sizing capped at max % of portfolio per trade
- Kill switch: if drawdown exceeds threshold, halt all trading
- No financial credentials stored in repo (use environment variables)

### What This System Will NOT Do

- Make real-time sub-second trading decisions (not HFT)
- Trade based on news, earnings, or fundamental data (candles only)
- Guarantee profits (no trading system can)
- Replace human judgment on risk limits and capital allocation

---

## Dependencies

```
# Core
torch>=2.0
transformers
pandas
numpy

# Kronos
# (cloned from https://github.com/shiyu-coder/Kronos)

# Data
ccxt              # Crypto exchange API (Binance) — also supports gold CFDs
yfinance          # Gold futures (GC=F), forex backup, equities
twelvedata        # Forex & gold OHLCV (free tier: 800 req/day)
# NOTE: MetaTrader5 is Windows-only, do NOT use on macOS
# NOTE: oandapyV20 requires OANDA account + API key

# Backtesting
vectorbt          # Vectorized backtesting
# or backtrader

# Risk
mapie             # Conformalized prediction (CQR)
scipy             # Statistical tests

# LLM
anthropic         # Claude API (or openai for GPT)

# Visualization
matplotlib
plotly

# Experiment tracking
wandb             # or comet_ml
```

---

## Environment Variables

```bash
# .env (gitignored, never committed)
BINANCE_API_KEY=           # Optional — public endpoints work without auth for historical data
BINANCE_SECRET=            # Optional
TWELVEDATA_API_KEY=        # Required for forex/gold data (free tier: 800 req/day)
ANTHROPIC_API_KEY=         # Required for LLM strategy layer (Claude)
WANDB_API_KEY=             # Optional — experiment tracking
```

## .gitignore Must Include

```
# Data (large, reproducible from scripts)
data/
*.csv
*.pkl

# Model checkpoints (large binary files)
checkpoints/

# Results
results/

# Secrets
.env
*.key

# Python
__pycache__/
*.pyc
.venv/
*.egg-info/

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
```

---

## References

- Shi et al. (2025). "Kronos: A Foundation Model for the Language of Financial Markets." arXiv:2508.02739. AAAI 2026.
- Romano et al. (2019). "Conformalized Quantile Regression." NeurIPS 2019. arXiv:1905.03222.
- Bailey & Lopez de Prado. "The Deflated Sharpe Ratio." Journal of Portfolio Management.
- TradingAgents (2026). Multi-Agent LLM Financial Trading Framework. https://github.com/TauricResearch/TradingAgents
- Chen et al. (2023). "Extending Context Window of LLMs via Positional Interpolation." arXiv:2306.15595.

---

---

## Issues Found During Plan Review (Fixed Above)

For posterity, these are the 7 issues caught during review before implementation began:

1. **`sample_count` bug**: `predict(sample_count=30)` returns an averaged single DataFrame, NOT 30 individual paths. Must call `predict(sample_count=1)` in a loop of 30 to get individual trajectories for quantile analysis. **Fixed in Quantile Risk section.**

2. **Wrong finetuning pipeline**: Plan originally referenced `finetune/` (Qlib-based, Chinese A-shares only). Our data is CSV-based, so we must use `finetune_csv/`. **Fixed in Finetuning Pipeline section.**

3. **MetaTrader5 is Windows-only**: The `MetaTrader5` Python package does not work on macOS. Replaced with `yfinance` (GC=F gold futures) and TwelveData API. **Fixed in Experiment 3 and Dependencies.**

4. **Gold data volume too small**: 4-hour candles × 2 years = ~3k candles. Insufficient for finetuning 102M params on a single instrument. Increased to 5+ years and suggested multi-commodity augmentation. **Fixed in Experiment 3 and Data Volume table.**

5. **CQR calibration leakage**: Original 3-way split (train/val/test) used validation set for both finetuning early stopping AND CQR calibration — double-dipping introduces leakage. Changed to 4-way split (train/calibration/validation/test). **Fixed in Data Splits section.**

6. **Missing volume/amount for forex and gold**: Forex spot has no real volume data; gold spot volume is unreliable. Kronos fills missing volume/amount with zeros, but the plan didn't acknowledge this. **Fixed with explicit notes in Experiments 2 and 3.**

7. **Missing `.gitignore` and `.env` specifications**: Plan said "gitignored" and "use environment variables" without specifying contents. **Fixed with explicit sections added.**

---

---

## Research Findings (June 6, 2026)

### Quant Firm & ML Trading Research

1. **No public profitable Kronos trading results** — Kronos paper acknowledges backtesting is "simplified example, not production-ready". AER and IR metrics reported, but no live P&L.
2. **Top firms use ensemble of weak signals** — Renaissance Medallion (~35% annualized), Two Sigma, Citadel all combine many uncorrelated signals rather than relying on one model.
3. **What kills ML trading**: overfitting to historical patterns, ignoring regime changes, not accounting for execution costs (slippage, fees).
4. **What works**: mean-reversion + momentum with regime detection, ensemble methods, robust position sizing (Kelly/fractional Kelly), walk-forward validation.
5. **Realistic targets**: Sharpe 1.0-1.5 excellent for retail quant, 2.0+ exceptional. Dir. accuracy of 53-55% with proper sizing can be very profitable.

### Applied to CIFR-QUANT

- Ensemble architecture (ZS + multi-seed FT) directly implements Grinold's Law
- CQR calibration provides distribution-free uncertainty quantification
- Regime detection is the single highest-impact feature not yet implemented
- Walk-forward backtest with realistic costs is non-negotiable before live trading

---

*Last updated: June 10, 2026 (rev 6 — plain-language + reasoning pass throughout; v2 plan brought current: carry brick #1 frozen and live in shadow/OKX-demo, 4 candidates killed, venue constraints mapped, methodology constitution expanded to 8 rules. v1 plan preserved above as historical record.)*

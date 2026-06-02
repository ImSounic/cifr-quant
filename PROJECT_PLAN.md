# CIFR-QUANT: Project Plan

## Overview

CIFR-QUANT is a multi-market algorithmic trading system built on top of **Kronos**, the first open-source foundation model for financial candlestick (K-line) data. The system finetunes Kronos-base (102M parameters) into three market-specialized checkpoints, combines probabilistic forecasting with quantile-based risk management, and uses an LLM strategy layer to generate, evaluate, and select trading strategies.

**Foundation Model**: [Kronos](https://github.com/shiyu-coder/Kronos) (Tsinghua University, AAAI 2026)
**License**: Kronos is MIT licensed. CIFR-QUANT is proprietary (no license).

---

## System Architecture

```
                    ┌─────────────────────────────┐
                    │       MARKET DATA LAYER      │
                    │  BTC/USDT  EUR/USD  XAU/USD  │
                    │  (Binance) (OANDA)  (MT5/TV) │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │     KRONOS FOUNDATION LAYER   │
                    │                               │
                    │  ┌─────────┐  ┌───────────┐  │
                    │  │Tokenizer│─▶│ Predictor  │  │
                    │  │(BSQ 2^20│  │(102M GPT)  │  │
                    │  │vocab)   │  │Decoder-only│  │
                    │  └─────────┘  └─────┬─────┘  │
                    │                     │        │
                    │  3 Finetuned Checkpoints:     │
                    │  ckpt-btc / ckpt-eur / ckpt-xau│
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │   PROBABILISTIC FORECAST LAYER │
                    │                               │
                    │  sample_count=30 Monte Carlo   │
                    │  → 30 future price paths       │
                    │  → Mean forecast               │
                    │  → Prediction intervals         │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │   QUANTILE RISK LAYER         │
                    │                               │
                    │  Empirical quantiles from      │
                    │  30 sampled paths:              │
                    │  • 5th percentile  → Stop Loss │
                    │  • 95th percentile → Take Profit│
                    │  • CQR calibration for coverage │
                    │    guarantee on held-out data   │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │   LLM STRATEGY LAYER          │
                    │                               │
                    │  Input: forecasts + SL/TP +    │
                    │  regime signals + market state  │
                    │                               │
                    │  Output: executable strategy    │
                    │  code (NOT direct trade calls)  │
                    │                               │
                    │  Strategy types:                │
                    │  • Trend following              │
                    │  • Mean reversion               │
                    │  • Breakout                     │
                    │  • Volatility targeting          │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │   BACKTEST & EVALUATION LAYER  │
                    │                               │
                    │  Walk-forward rolling backtest  │
                    │  Multi-strategy comparison      │
                    │  Realistic costs (spread,       │
                    │  slippage, commission)           │
                    │  Deflated Sharpe Ratio           │
                    └─────────────────────────────────┘
```

---

## Three Market Experiments

### Experiment 1: Crypto — BTC/USDT

| Parameter | Value |
|-----------|-------|
| Instrument | BTC/USDT perpetual |
| Timeframe | 15-minute candles |
| Context window | 512 candles = ~5.3 days (24/7 market) |
| Prediction horizon | 24–96 candles (6–24 hours) |
| Data source | Binance API (free, no auth for historical OHLCV) |
| Data volume | 2+ years of 15-min data (~70k candles/year) |
| Finetuned checkpoint | `checkpoints/ckpt-btc/` |
| Rationale | Highest volatility, most data availability, 24/7 market with no gaps. Kronos live demo already uses BTC/USDT. Training data upweighted crypto during pre-training. |

### Experiment 2: Forex — EUR/USD

| Parameter | Value |
|-----------|-------|
| Instrument | EUR/USD spot |
| Timeframe | 1-hour candles |
| Context window | 512 candles = ~21 trading days (~3 weeks) |
| Prediction horizon | 24–168 candles (1 day – 1 week) |
| Data source | OANDA API or FXCM |
| Data volume | 2+ years of 1-hour data (~6.2k candles/year) |
| Finetuned checkpoint | `checkpoints/ckpt-eur/` |
| Rationale | Most liquid instrument on earth, negligible spreads (~0.1 pips). Session-based trading (London/NY/Tokyo) captured by Kronos temporal embeddings. 21 days context captures weekly cycles. |
| **Volume note** | Forex spot has no centralized volume. Use `volume=0, amount=0` — Kronos handles this (fills with zeros). Tick volume from brokers is a poor proxy and should NOT be used. |
| **Gap handling** | Forex closes on weekends. 512 hourly candles = ~21 *trading* days (~30 calendar days). Weekend gaps must be handled: either skip gaps (let temporal embeddings handle it) or insert NaN rows and let Kronos truncate. |

### Experiment 3: Commodities — XAU/USD (Gold)

| Parameter | Value |
|-----------|-------|
| Instrument | XAU/USD (Gold spot) |
| Timeframe | 4-hour candles |
| Context window | 512 candles = ~85 trading days (~3 months) |
| Prediction horizon | 6–42 candles (1 day – 1 week) |
| Data source | `yfinance` (GC=F gold futures) or TwelveData API. **NOT MetaTrader5** (Windows-only, won't run on macOS) |
| Data volume | **5+ years** of 4-hour data (~1.5k candles/year = ~7.5k total). 2 years is too few for finetuning a 102M model on a single instrument. |
| Finetuned checkpoint | `checkpoints/ckpt-xau/` |
| Rationale | Strong trending behavior driven by real rates, central bank buying, risk-off flows. 3-month context captures macro trend structure. Futures exchanges (COMEX) were in Kronos training data. |
| **Volume note** | Gold spot volume is unreliable from most free sources. Use `volume=0, amount=0` — Kronos handles missing volume gracefully (fills with zeros). Alternatively, use gold futures (GC=F) which have real volume. |

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
| Finetune tokenizer (per market) | A10 or A100 | ~8 GB | ~2 hours |
| Finetune predictor (per market) | A10 or A100 | ~12 GB | ~4 hours |
| Inference (per market) | Mac Mini 16GB / any GPU | ~4 GB | seconds |
| **Total for 3 markets** | | | **~18–24 hours** |

**Development**: Mac Mini 16GB (data pipeline, inference, backtesting)
**Training**: University JupyterLab with A10/A100 GPUs

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

### Phase 1: Data Pipeline (Week 1)

- [ ] Set up data collection scripts for all three markets
  - [ ] Binance API client for BTC/USDT 15-min OHLCV
  - [ ] OANDA/FXCM API client for EUR/USD 1-hour OHLCV
  - [ ] MetaTrader5/TradingView client for XAU/USD 4-hour OHLCV
- [ ] Data cleaning and validation (Z-score clipping, gap handling, session filtering)
- [ ] Data splitting logic (train/val/test with temporal boundaries)
- [ ] Data format conversion to Kronos-compatible DataFrames
- [ ] Storage: raw data in `data/raw/`, processed in `data/processed/`

### Phase 2: Kronos Integration (Week 2)

- [ ] Install Kronos via `pip install kronos-model-arch` or clone repo and add `model/` to Python path
- [ ] Load pre-trained Kronos-base and Tokenizer-base from HuggingFace
- [ ] Build `KronosPredictor` wrapper for each market
- [ ] Verify zero-shot inference works on all three instruments
- [ ] Benchmark zero-shot RankIC/IC as baseline

### Phase 3: Finetuning (Week 3)

- [ ] Prepare finetuning configs for each market (`configs/btc.py`, `configs/eur.py`, `configs/xau.py`)
- [ ] Upload data + scripts to university JupyterLab
- [ ] Finetune tokenizer for each market (3 runs × ~2h)
- [ ] Finetune predictor for each market (3 runs × ~4h)
- [ ] Download finetuned checkpoints to local machine
- [ ] Compare finetuned vs zero-shot metrics on validation set

### Phase 4: Quantile Risk Management (Week 4)

- [ ] Implement multi-path sampling (sample_count=30)
- [ ] Build empirical quantile extraction (SL/TP levels)
- [ ] Implement CQR calibration on validation data
- [ ] Validate coverage guarantees on held-out calibration set
- [ ] Build adaptive position sizing based on interval width

### Phase 5: Backtest Engine (Week 5)

- [ ] Build walk-forward backtest framework
- [ ] Implement realistic cost models per market
- [ ] Build multi-strategy executor (trend/mean-revert/breakout/vol-target)
- [ ] Implement evaluation metrics suite (Sharpe, drawdown, Calmar, etc.)
- [ ] Run baseline backtests with simple strategies

### Phase 6: LLM Strategy Layer (Week 6)

- [ ] Design context pack format (forecasts + regime + market state)
- [ ] Build LLM strategy generation pipeline
- [ ] Implement strategy code execution sandbox
- [ ] Test strategy generation → backtest → evaluation loop
- [ ] Add guardrails (iteration limits, parameter budgets)

### Phase 7: Integration & Evaluation (Week 7–8)

- [ ] End-to-end pipeline: data → forecast → SL/TP → strategy → backtest
- [ ] Run full evaluation on all three markets
- [ ] Final test set evaluation (last 3 months, ONE TIME ONLY)
- [ ] Cross-market performance comparison
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

*Last updated: June 2, 2026 (rev 2 — post-review fixes applied)*

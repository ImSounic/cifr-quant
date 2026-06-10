# CIFR-QUANT: Project State

> **Purpose**: Cross-session memory for Claude. Read this file at the start of every new session.
> **Last updated**: June 10, 2026

---

## Quick Summary

CIFR-QUANT is a systematic-trading research project building toward a quant firm. User is Raj (s3702111), UTwente student. Works across HP Windows laptop (dev) + UTwente HPC (all heavy compute).

**The project pivoted on June 10, 2026.** Phase 1 — directional trading on the Kronos financial foundation model (102M, OHLCV candles) — was completed and **rigorously falsified**: the model has no extractable edge of any kind (directional, cross-sectional, or volatility-beyond-persistence) on crypto 15m or commodity 4h data. See "Phase 1 Post-Mortem" below for the full evidence chain.

**Current direction (v2): the Signal Factory.** Keep the validated research pipeline (forecast cache, pluggable strategy engine, IC/t-stat diagnostics, CQR, walk-forward backtest, HPC workflow) and point it at **information-bearing inputs** instead of price candles: funding rates, open interest, liquidations, basis — starting with funding-rate carry, the best-documented edge in crypto. Kronos is retired from the signal path. The LLM layer is repositioned from "strategy picker" (information-free) to a future **text→feature data source**. Detailed plan in PROJECT_PLAN.md ("The Plan Now (v2)").

---

## Current Status: Phase 2 — FIRST SIGNAL PASSED THE GATE (June 10, 2026)

**Funding-rate carry is real.** Phase 2A (data) + 2B (diagnostic) complete in one day:
- `scripts/fetch_derivs.py` fetched funding (4.9k events/asset since 2022), perp 1h
  klines (39k rows/asset), OI (30d, accumulating) for 14 assets (MATIC delisted) →
  `data/raw/derivs/` on HPC.
- `scripts/carry_skill.py`: **8h cross-sectional IC −0.020, t=−4.83 pooled;
  negative in ALL 5 years (2022-26), |t|≥2.2 in 4/5, strongest in 2026 (−2.79 —
  not decaying). Per-asset sign consistency 93%.** First |t|>2 of the project
  (Kronos's best was +0.76). Tradeable horizon = 8h (1d unstable, 3d noise).
- Harvest leg: majors yield +5-8% ann. unconditionally BUT regime-dependent
  (+11.6% 2024, negative 2022/2026) → conditional/cross-sectional designs only.
  BNB structurally negative funding (−5.8%/yr, shorts pay) — anomaly, investigate.
- **Earned backtest run & iterated (6 configs total, ledger counted). BRICK #1
  FROZEN: `v2_maker_8h` = K=3, --smooth 9, --exit-band 2, 8h cadence, maker
  costs (0.03%/side): +9.0%/yr, Sharpe 0.52, dollar-neutral, NET POSITIVE ALL
  5 YEARS** (2022 +14.9 / 2023 +22.6 / 2024 +1.8 / 2025 +4.1 / 2026 +5.5).
  Decomposition: price +31% / funding +35% / costs −17% over 4.5y. Turnover
  0.11 gross/event (smoothing+band hysteresis vs 0.91 naive — naive taker
  version lost −91% to fees despite +124% gross: implementation matters).
  Honest read: equity-curve t≈1.1 alone is weak; credibility = diagnostic
  t=−4.83 + mechanical funding leg + 5/5-year consistency. Deflated for 6
  trials → true expectation somewhat below Sharpe 0.5. NO MORE CARRY TUNING —
  next iteration risk is overfitting. Known gaps: maker fill risk (priced only
  by paper trading), maxDD −30% (fix = portfolio-level vol targeting, not
  carry tweaks). Results: `results/backtest/carry_backtest_v2_maker_8h.json`.

---

## Phase 1 Post-Mortem: The Kronos Thesis, Tested and Falsified (June 6–10, 2026)

### The original thesis
Finetune Kronos (foundation model for candlesticks) per market → ensemble MC-path forecasts → CQR-calibrated bands → directional trades sized by risk parity → an LLM layer on top picking/adapting strategies with market conditions. Two markets: crypto (15 assets, 15m, Binance) and commodities (6 assets, 4h, TwelveData/yfinance).

### What was built (all of it works and is reusable)
- Finetuned checkpoints: BTC v1/v2, XAU v1/v2 (EUR dropped earlier). Ensembles = zero-shot + 2 finetunes per market (`src/model/build.py`).
- Batched MC path generation (`src/model/batched_inference.py`) — ~20x effective speedup on L40S vs the per-path loop (Kronos's `sample_count=N` averages paths; we replicate the AR loop without averaging).
- CQR calibration for all 21 assets, 90% target → 91-92.5% achieved (`results/cqr/cqr_calibrations.json`).
- **Forecast cache** (`src/model/forecast_cache.py` + `scripts/build_forecast_cache.py`): run the GPU ensemble ONCE over the walk-forward grid, dump per-(asset, rebalance) forecasts to `results/forecasts/` → all strategy testing becomes CPU-only and instant. This is the single most valuable infra piece.
- **Pluggable strategy engine** (`src/backtest/portfolio_engine.py` + `strategy_api.py` + `strategies.py` + `sizing.py` + `grid.py`): Strategy/Sizer/RegimeClassifier injected; baseline proven logically equivalent to the original inline engine (regression test `scripts/test_strategy_equivalence.py`).
- **Skill diagnostics** (`scripts/forecast_skill.py`, `horizon_skill.py`, `vol_skill.py`): IC, rank IC, t-stats, hit rates, confidence stratification, horizon curves, vol-vs-persistence baselines.
- Regime classifier (Hurst+ADX, `src/regime/`) — built, tested, found net-negative (below).

### The evidence chain (chronological — every escape hatch tested and closed)

1. **Backtest #1 (job 511139, June 9)** — single directional strategy, 90-day walk-forward:
   crypto **−23.4%** (Sharpe −3.28), commodity **+16.0%** (Sharpe 4.36), combined −5.1%.
   Suspicious asymmetry → investigate rather than celebrate commodity.

2. **Strategy A/B sweep off the forecast cache** (baseline reproduced: −23.7%/+14.1% → refactor faithful):
   | Strategy | Crypto | Commodity | Key telltale |
   |---|---|---|---|
   | directional (baseline) | −23.7% | +14.1% | crypto exit mix 0 tp / 31 sl / **2341 timeout** |
   | regime_gated_trend (strict & relaxed) | −2 to −6% | **0 trades** | Hurst never labels trending energy "trend" |
   | mean_reversion (ATR exits) | −6.2% | +1.8% | exits actually fire (106 tp/195 sl/25 to) |
   - Finding A: **CQR band is inert as SL/TP** (calibrated for coverage, far too wide intrabar) — directional was a pure 12h horizon bet.
   - Finding B: **Hurst+ADX regime gating is net-negative** — killed the only profitable book.

3. **ATR-exit sweep (`directional_atr`, RR 1.0/1.5/2.0)** — "winning direction + working exits":
   crypto −26% at every RR; **atr_10 (symmetric RR) win rate 48.9% = sub-coinflip — the smoking gun**;
   commodity fell +14% → −5% (tight stops cut trend winners → its profit was drift-riding, not skill).

4. **forecast_skill.py (decouple forecaster from strategy)** —
   crypto n=2516: hit 50.8% (0.8σ), pooled IC +0.004 (noise floor 0.020), signed edge negative;
   commodity n=384: hit 50.3%, IC −0.022. **Confidence degenerate**: 82% of forecasts pinned >0.80 conf, that bucket = coinflip.
   **Cross-sectional rank IC** (can it pick relative winners?): crypto mean +0.019, **t = +0.76** (NS); commodity t = −0.01.
   → Commodity's +14% was **beta** (coinflip direction + big asymmetric trend moves), not alpha.

5. **horizon_skill.py (GPU, jobs 511874 + 511916)** — IC/hit at EVERY horizon step:
   crypto h=1..48: IC ∈ [−0.05, +0.04] around zero, hit mostly <50%. No short-horizon skill hiding.
   commodity with proper n=384: h=1 IC +0.053 / hit 49.0% (the n=24 "IC +0.29 flicker" was sampling noise).

6. **vol_skill.py** — last hypothesis: maybe it predicts volatility.
   pred_width vs realised vol: crypto IC +0.34, commodity +0.44 — **but naive persistence (trailing realised vol) scores +0.61 / +0.46**. The model's vol forecast is a degraded echo of recent vol. (The cross-sectional vol t=25 is the trivial DOGE-is-always-more-volatile-than-BTC ranking.)

### Final verdict table
| Capability | Result |
|---|---|
| Directional, time-series, any horizon, both markets | **none** |
| Cross-sectional ranking | **none** (t = 0.76 / −0.01) |
| Volatility forecasting | **≤ naive persistence** (which is free: IC 0.61) |
| CQR uncertainty calibration | works — but only because vol persists |

**Kronos on OHLCV alone adds zero extractable information beyond trailing realised vol.** Not a tuning failure: strategy structure, exits, gating, confidence, horizon, and cross-section were all tested with proper statistics.

### Why it failed (the lesson that defines v2)
- **Edge lives in the inputs, not the model.** Kronos was already trained on 12B+ candles from 45 exchanges and still has nothing — OHLCV is the most-arbitraged dataset on earth. A bigger in-house model on the same data would learn the same nothing.
- **Information cannot be created downstream.** The planned LLM strategy layer sat downstream of Kronos's output; an LLM reasoning about a coinflip is still betting on a coinflip. (We effectively ran that layer's job manually via the strategy sweep — every variant lost.) "Keep changing strategies with the market" without t-stat discipline = overfitting machine.
- **The +14% trap**: a beautiful single-window backtest (Sharpe 4.36!) that was pure regime beta. Without the skill diagnostics we'd have shipped it.
- **What a quant firm actually is**: (1) a signal factory that cheaply kills bad hypotheses ← **built and proven**, (2) a portfolio of small validated uncorrelated edges ← have zero, this is the gap, (3) execution + risk ← backtest half built.

---

## What's Next (v2 Roadmap — In Order)

1. ✅ **Phase 2A — Derivatives data layer**: `scripts/fetch_derivs.py` done June 10; 14 assets of funding / perp-1h / OI in `data/raw/derivs/` (HPC).
2. ✅ **Phase 2B — Carry signal diagnostic**: **GATE PASSED** — 8h XS t=−4.83, negative all 5 years, |t|≥2.2 in 4/5. Earned backtest built (`scripts/backtest_carry.py`), first run pending.
3. 🔄 **Phase 2C — More signal candidates** (June 10 results so far):
   - ❌ **XS momentum** (momo_skill.py): FAILED — grid signs incoherent, no cell
     near significance, year-unstable. Closed without a backtest.
   - ❌ **Short-term reversal** (7d→1d, found by scan at t=−2.86): passed the
     marginal gate but backtest price leg lost −105% across ALL years.
     **LESSON → GAUNTLET UPGRADE: rank IC measures the whole cross-section; a
     K-portfolio trades the TAILS, which in crypto trend violently (death
     spirals/pumps) while the middle reverts. Every future diagnostic must
     include a construction-matched tail-spread test (bottom-K minus top-K
     forward return) before any backtest is earned.** Added to momo_skill.py.
   - ✅ Tail-spread check confirmed the reversal post-mortem: bottom-3-minus-top-3
     spread −0.108%/day pooled, negative EVERY year → middle reverts, tails trend.
     Mirror trade (tail momentum) decayed −0.23%/day (2022) → −0.006% (2026): dead.
   - ❌ **TSMOM** (tsmom_skill.py): equal-notional AND canonical inverse-vol-scaled
     both fail — best t≈1.6, 2025 negative in every lookback/construction. All-cells-
     positive grid was suggestive but cells share trades; not 12 confirmations. Closed.
   - **2C scoreboard: carry ✅ / XS momo ❌ / reversal ❌ / tail-momo ❌ / TSMOM ❌.**
     1-in-5 hit rate = normal. Remaining queue BLOCKED ON DATA: OI/squeeze (OI history
     accumulating since June 10, ~30d), commodity futures carry (term structure/COT —
     needs new data source; commodities DEPRIORITIZED not scrapped — spot candles carry
     no term structure), Phase 3 text features.
   - NOTE: none of Phase 2 touches Kronos — all CPU statistics over derivs CSVs.
     Complexity must be earned; ML returns only behind validated features.
4. 🔄 **Phase 4 v1 — SHADOW PAPER TRADING (started June 10)**: `scripts/paper_trade_carry.py`
   runs the FROZEN carry config live every 8h via cron on the HPC head node (network
   I/O only). Fetches real funding + marks, accrues real funding on the held book,
   simulates fills at mark±3bp, persists `results/paper/carry_state.json` +
   `carry_history.csv`. Measures signal-live consistency vs backtest. Frozen params
   in the script header — ANY change = new strategy, must re-earn the gate.
   v2 (later): Binance testnet limit orders to price maker fill risk.
4. ⏳ **Phase 2D — Portfolio the survivors**: validated signals → pluggable engine (`--strategy` plug-ins) with **vol-targeted sizing** (vol persistence IC 0.61 is free and proven — it's the risk layer, not the alpha).
5. ⏳ **Phase 3 — LLM as data source** (the v1 vision, pointed the right way): text→features (news/sentiment/events) → same IC diagnostics. NOT a strategy picker.
6. ⏳ **Phase 4 — Paper trading loop** (Binance testnet) — makes it a firm, generates execution data.

**Methodology rules (non-negotiable, learned the hard way):**
- A signal must pass IC/t-stat diagnostics (|t|>2) BEFORE any backtest is run on it.
- Count every variant tried; deflated Sharpe / multiple-testing haircut on anything that looks good.
- Multi-window walk-forward, never a single 90-day window. Final test window touched once.
- Costs modeled from day one. Vol-targeted sizing as default risk layer.

### Status of old roadmap items
- ~~Regime detection~~ — built, tested, net-negative as a gate on a zero signal. `src/regime/` kept (indicators reusable for vol sizing).
- ~~Meta-labeling (task #9)~~ — cancelled: nothing upstream to filter.
- ~~LLM strategy layer (v1 design)~~ — cancelled as designed; repositioned to Phase 3 text-features.
- ~~Final Kronos test-set evaluation~~ — moot; verdict already conclusive at proper significance.

---

## HPC Environment

- **Cluster**: UTwente SLURM, head nodes `hpc-head1/2.ewi.utwente.nl`
- **Best GPUs in main-gpu**: Lovelace (L40S/L40, 48GB) on hpc-node01-18, `--gres=gpu:lovelace:1`
- **Conda env**: `trade`. Activate: `source $(conda info --base)/etc/profile.d/conda.sh && conda activate trade` (direct path doesn't work)
- **HuggingFace**: cached on head node; jobs use `HF_HUB_OFFLINE=1` (compute nodes offline)
- **SLURM sizing**: 8 CPU / 32G is right-sized and backfills fast; 32 CPU / 128G blocks backfill for ~days
- **Raj runs all SSH/sbatch/git commands himself** — provide commands, he executes & pastes output
- **Commits**: Raj's credentials only, NEVER any Claude/AI attribution

---

## Development Environment

### HP Windows 11 laptop (dev machine; small checks/tests ONLY — all real workloads on HPC)

- **Repo**: `W:\cifr-quant` (W: is a `subst` of `C:\Work Drive`)
- **Python env**: venv `mlenv` at `C:\Work Drive\envs\mlenv`; activate by typing `mlenv` in PowerShell. Python 3.12.10
- **GPU**: RTX PRO 1000 (Blackwell), torch cu128, CUDA works
- **Deps added for project**: `ccxt yfinance twelvedata anthropic python-dotenv` (do NOT install vectorbt/mapie — would downgrade shared numpy/pandas)
- **Kronos**: gitlink (mode 160000, commit `67b630e`), **no `.gitmodules`** → fresh clone leaves `Kronos/` empty. Restore: `git clone https://github.com/shiyu-coder/Kronos.git Kronos && cd Kronos && git checkout 67b630e`. **Edits inside `Kronos/` do NOT propagate via git — all new code goes in tracked `src/`.**
- **Not present locally** (gitignored, live on HPC): `data/`, `checkpoints/`, `results/`
- `PYTHONPATH` = repo root + `Kronos` when running locally

### Training/compute: UTwente HPC
- Checkpoints at `~/cifr-quant/Kronos/checkpoints/`, symlinked to `~/cifr-quant/checkpoints/`

---

## File Map (updated June 10, 2026)

### v2-relevant infrastructure (the keepers)

| File | Purpose |
|------|---------|
| `src/backtest/strategy_api.py` | Contracts: ForecastBundle, RegimeLabel, AssetDecision, TradeIntent, Strategy, Sizer |
| `src/backtest/strategies.py` | DirectionalMomentum / DirectionalMomentumATR / RegimeGatedTrend / MeanReversion |
| `src/backtest/sizing.py` | InverseWidthRiskParity, EqualWeight (VolTarget/Kelly: todo in v2) |
| `src/backtest/portfolio_engine.py` | Pluggable joint multi-asset walk-forward engine (CPU, off cache) |
| `src/backtest/grid.py` | Shared rebalance grid + context location (cache & engine must agree) |
| `src/backtest/costs.py` | Cost models incl. CRYPTO_COSTS / COMMODITY_COSTS |
| `src/backtest/metrics.py` | Sharpe, drawdown, Calmar, deflated Sharpe |
| `src/model/forecast_cache.py` | Build (GPU) / load (CPU) per-(asset,rebalance) forecast cache |
| `src/model/build.py` | build_market_ensemble — single source of truth for ensemble composition |
| `src/model/batched_inference.py` | Batched MC paths (fixes Kronos's path-averaging) |
| `src/model/ensemble.py` | EnsemblePredictor (predict_with_quantiles) |
| `src/regime/indicators.py` | hurst, adx, atr, realized_vol, vol_percentile (reusable for vol sizing) |
| `src/regime/classifier.py` | Hurst+ADX composite (empirically net-negative as a gate) |
| `src/risk/cqr.py` | CQR calibration (works as designed) |
| `scripts/backtest_portfolio.py` | CPU strategy A/B driver (--strategy/--sizer/--tag) |
| `scripts/build_forecast_cache.py` + `slurm/build_forecast_cache.sh` | One-time GPU cache build |
| `scripts/forecast_skill.py` | IC / hit / confidence-stratification / cross-sectional rank IC |
| `scripts/horizon_skill.py` + slurm wrappers | IC + hit at every horizon step |
| `scripts/vol_skill.py` | Vol-forecast IC vs naive persistence baseline |
| `scripts/test_strategy_equivalence.py` | Regression proof: refactor == original engine |
| `docs/STRATEGY_DESIGN.md` | Signed-off pluggable-architecture design doc |

### Key results on HPC (`results/`)
| Path | Contents |
|------|----------|
| `results/cqr/cqr_calibrations.json` | 21 assets, 90%→91-92.5% coverage |
| `results/forecasts/{crypto,commodity}_forecasts.csv` | The forecast cache (2516 + 384 rows) |
| `results/backtest/portfolio_backtest_*.json` | baseline / rgt_strict / rgt_relaxed / meanrev / atr_10 / atr_15 / atr_20 |

### v1 legacy (kept, not in active use)
Finetune configs (`finetune/config_*.yaml`), finetune SLURM scripts, eval scripts (`zero_shot_baseline.py`, `finetuned_eval.py`, `eval_ensemble.py`), CQR calibration scripts, single-asset engine (`src/backtest/engine.py`), orchestrator (`src/portfolio/orchestrator.py`), LLM strategy layer v1 (`src/strategy/` — llm_generator/executor/context_builder + single-asset strategies), data fetchers (`scripts/fetch_universe.py`, `fetch_commodities_yf.py` — patterns to reuse for fetch_derivs.py).

---

## Common Pitfalls & Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError` on data | Relative paths resolve wrong from `Kronos/finetune_csv/` | Absolute paths in YAML |
| HuggingFace 401 | Wrong repo name | `NeoQuasar/Kronos-base`; `hf auth login` with read token |
| HuggingFace httpx closed | Compute nodes offline | Cache on head node, `HF_HUB_OFFLINE=1` |
| Checkpoints not found | Saves to `Kronos/checkpoints/`, loader expects `checkpoints/` | `ln -s Kronos/checkpoints checkpoints` |
| Conda activation fails | Direct path doesn't exist | `source $(conda info --base)/etc/profile.d/conda.sh && conda activate trade` |
| SLURM job stuck PD for days | Oversized request (128G/32CPU) blocks backfill | Right-size to 32G/8CPU |
| Long job times out, loses results | Results written only at end | Save incrementally per asset + resume (see calibrate_cqr_multi.py) |
| SLURM logs invisible mid-run | NFS sync delay | `tee` inside script + `mkdir -p slurm/logs` first |
| Kronos paths averaged | `predict(sample_count=N)` averages N paths | Use `src/model/batched_inference.predict_paths` |
| Login-node python = CPU | No GPU on hpc-head | GPU work goes through sbatch, always |
| Small-n IC looks amazing | e.g. n=24 commodity h=1 IC +0.29 → +0.05 at n=384 | Never read an IC without n and a t-stat |
| Single-window Sharpe 4+ | Regime beta masquerading as alpha | Decouple signal from strategy; hit-rate + signed-edge separates skill from drift |
| TwelveData free tier | Only XAU/USD works | yfinance futures for other commodities |
| Git push rejected on laptop | HPC/remote ahead | `git pull --rebase` then push |

---

*This file is the single source of truth for project state. Update after every significant change.*

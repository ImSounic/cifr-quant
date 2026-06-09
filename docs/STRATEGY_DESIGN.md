# Pluggable Strategy Architecture — Design (signed off June 9 2026)

## Why
The June-9 backtest (job 511139) ran ONE strategy — probabilistic-directional —
and got crypto −23.4% (Sharpe −3.28) / commodity +16.0% (Sharpe +4.36). The
literature (regime filtering, López de Prado meta-labeling, vol-targeting) and
the result both say: the strategy only works in trending regimes. We need to
test many strategies, not hardcode one.

## Principle: separate the 4 fused concerns
`portfolio_engine.run_market_backtest` currently fuses forecast + regime + trade
logic + sizing into one loop. Split them; compute the expensive part (Kronos)
ONCE and cache it so strategy A/B is CPU-only and fast.

```
FORECAST (Kronos ensemble, GPU, run once) ──► cache: results/forecasts/{market}_forecasts.csv
                                              │
        ┌─────────────────────┬──────────────┴───────────┐
        ▼                     ▼                           ▼
  RegimeClassifier        Strategy                      Sizer        (all CPU, swappable)
  (Hurst+ADX+vol)         decide → TradeIntent          usd/symbol
        └─────────────────────┴──────────────┬───────────┘
                                              ▼
                                          Simulator (intent + future bars + costs → PnL)
```

**Biggest payoff:** Kronos forecasts cached → testing N strategies costs zero GPU,
runs in seconds on the laptop. Run Kronos once on HPC, iterate strategies locally.

## Signed-off decisions (June 9)
1. **Refactor + cache first.** Baseline `DirectionalMomentum`+`InverseWidthRiskParity`
   must reproduce today's logic exactly (logical-equivalence regression test —
   exact P&L can't be reproduced because the prior run's MC paths weren't saved).
2. **Regime = Hurst + ADX composite.** Both agree → TREND; either flags → RANGE.
3. **HPC builds the forecast cache; laptop iterates strategies (CPU-only).**

## Note on existing code
`src/strategy/` is the **LLM strategy layer** (separate subsystem, item #6):
single-asset `signal_fn(data, capital, position)` over a `MarketContext`, used by
the single-asset `src/backtest/engine.py`. The portfolio interface here lives in
`src/backtest/` so it does NOT touch that working code. Strategy names/semantics
are mirrored (trend / mean_revert / breakout) so the two layers can merge later.

## Data contracts
- **ForecastBundle** (one cache row per (symbol, rebalance_t)): entry_price, final-step
  q05/q25/q50/q75/q95, mean, direction, confidence, pred_len; `cqr_correction`
  attached at run time (NOT baked into the cache, so a strategy may use raw or widened).
- **RegimeLabel**: trend_state ∈ {TREND_UP, TREND_DOWN, RANGE}; vol_state ∈ {LOW,NORMAL,HIGH};
  features {hurst, adx, realized_vol, atr, vol_pct}.
- **TradeIntent**: symbol, direction, entry_ref, stop, target, conviction, meta.

## Interfaces
- `RegimeClassifier.classify(context_df) -> RegimeLabel`
- `Strategy.generate_intents(decisions: list[AssetDecision]) -> list[TradeIntent]`
  (per-asset strategies implement `_decide_one`; cross-sectional override the whole method;
   future bars are NOT passed → no lookahead by construction)
- `Sizer.size(intents, capital) -> dict[symbol -> position_usd]`
- `Simulator` = existing `_simulate_position`, fed a TradeIntent.

## Regime filter (v1)
| Signal | Computation | Threshold |
|--------|-------------|-----------|
| Hurst  | R/S on log-returns over lookback | >0.55 trend · <0.45 mean-revert |
| ADX    | Wilder ADX(14) | >25 trending · <20 range |
| Vol %  | rolling σ vs trailing distribution | top quartile → HIGH |
| ATR    | ATR(14) | feeds stop sizing |

Composite: TREND if Hurst>0.55 AND ADX>25 (directional enabled); RANGE if Hurst<0.45
OR ADX<20 (mean-reversion enabled, directional disabled); HIGH vol → widen stops / stand aside.

## Strategy library
1. DirectionalMomentum — today's logic (baseline). 2. MeanReversionBand — fade q05/q95.
3. RegimeGatedTrend — #1 only in TREND. 4. CrossSectionalNeutral — long top / short bottom by
expected return, dollar-neutral. 5. MetaLabeled(base) — RF/GBM vetoes weak intents (task #9).

## Sizers
InverseWidthRiskParity (today's), VolTarget, FractionalKelly, EqualWeight.

## File layout (all tracked src/, nothing in Kronos/)
```
src/backtest/strategy_api.py   # ForecastBundle, RegimeLabel, AssetDecision, TradeIntent, Strategy, Sizer
src/backtest/strategies.py     # DirectionalMomentum (Phase A); MeanReversion/RegimeGated/CrossSectional (B/D)
src/backtest/sizing.py         # InverseWidthRiskParity (Phase A); VolTarget/Kelly (Phase C)
src/model/forecast_cache.py    # build_forecast_cache (GPU) + load_forecast_cache (CPU)
src/regime/indicators.py       # hurst(), adx(), realized_vol(), atr()   (Phase B)
src/regime/classifier.py       # RegimeClassifier, RegimeLabel            (Phase B)
```

## Phasing
- A: forecast cache + engine refactor; baseline reproduces. ◄ building now
- B: regime classifier + RegimeGatedTrend + MeanReversionBand. First test of "does regime gating fix crypto."
- C: ATR stops + vol-target/Kelly sizers.
- D: meta-labeling, cross-sectional, multi-window walk-forward + deflated Sharpe.

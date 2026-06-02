"""LLM-based strategy selection and parameterization.

The LLM receives market context and proposes:
1. Which strategy to use (trend, mean_revert, breakout, vol_target)
2. What parameters to set
3. Rationale for the choice

The LLM NEVER makes direct trade decisions.
"""

import json
import os
from typing import Optional

from src.strategy.context_builder import MarketContext
from src.strategy.executor import STRATEGIES


SYSTEM_PROMPT = """You are a quantitative trading strategist. Given market analysis data,
you select the most appropriate trading strategy and its parameters.

Available strategies:
1. **trend** — Trend following. Best when directional confidence > 70% and trend strength > 60%.
   Params: min_confidence (0.6-0.9), min_trend_strength (0.5-0.8), min_rr_ratio (1.0-3.0), max_risk_pct (0.01-0.03)

2. **mean_revert** — Mean reversion. Best in high-volatility regimes when price is at band extremes.
   Params: min_vol_percentile (0.5-0.9), band_threshold (0.7-0.95), max_risk_pct (0.01-0.02)

3. **breakout** — Breakout. Best when interval width is narrow (compression) with directional conviction.
   Params: max_interval_width (0.01-0.05), min_confidence (0.55-0.75), max_risk_pct (0.01-0.03)

4. **vol_target** — Volatility targeting. Sizes positions to target constant portfolio risk.
   Params: target_vol (0.10-0.25), min_confidence (0.55-0.70), lookback (10-40)

Respond ONLY with valid JSON:
{
    "strategy": "<strategy_name>",
    "params": {<parameter_dict>},
    "rationale": "<brief explanation>"
}
"""


def propose_strategy(
    context: MarketContext,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """
    Use LLM to propose a strategy and parameters for the given market context.

    Args:
        context: MarketContext with all analysis data
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        model: Claude model to use

    Returns:
        Dict with 'strategy', 'params', and 'rationale' keys
    """
    try:
        import anthropic
    except ImportError:
        print("anthropic package not installed. Using fallback rule-based selection.")
        return fallback_strategy_selection(context)

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("No ANTHROPIC_API_KEY set. Using fallback rule-based selection.")
        return fallback_strategy_selection(context)

    client = anthropic.Anthropic(api_key=key)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": context.to_prompt(),
            }],
        )

        # Parse JSON response
        text = response.content[0].text.strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        result = json.loads(text)

        # Validate strategy name
        if result.get("strategy") not in STRATEGIES:
            print(f"LLM proposed unknown strategy: {result.get('strategy')}. Using fallback.")
            return fallback_strategy_selection(context)

        return result

    except Exception as e:
        print(f"LLM call failed: {e}. Using fallback.")
        return fallback_strategy_selection(context)


def fallback_strategy_selection(context: MarketContext) -> dict:
    """
    Rule-based strategy selection when LLM is unavailable.

    Logic:
    - High confidence + strong trend → trend following
    - High vol + price at extremes → mean reversion
    - Narrow bands + moderate confidence → breakout
    - Otherwise → vol targeting (always works)
    """
    dc = context.directional_confidence
    ts = context.trend_strength
    vp = context.volatility_percentile
    iw = context.interval_width

    if dc > 0.70 and ts > 0.60:
        return {
            "strategy": "trend",
            "params": {"min_confidence": 0.65, "min_trend_strength": 0.55, "min_rr_ratio": 1.5},
            "rationale": f"Strong directional signal ({dc:.0%}) with trend consistency ({ts:.0%})",
        }

    if vp > 0.75 and iw > 0.03:
        return {
            "strategy": "mean_revert",
            "params": {"min_vol_percentile": 0.65, "band_threshold": 0.85},
            "rationale": f"High volatility regime ({vp:.0%}) with wide bands ({iw:.1%})",
        }

    if iw < 0.02 and dc > 0.60:
        return {
            "strategy": "breakout",
            "params": {"max_interval_width": 0.025, "min_confidence": 0.58},
            "rationale": f"Tight compression ({iw:.1%}) with moderate conviction ({dc:.0%})",
        }

    return {
        "strategy": "vol_target",
        "params": {"target_vol": 0.15, "min_confidence": 0.55},
        "rationale": "Default vol-targeting — no strong regime signal detected",
    }

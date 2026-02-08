"""
SENTINEL — Forecast explanation generator.
Produces plain-English explanations for every forecast.
"""

import pandas as pd

from utils.logger import get_logger

log = get_logger("forecasting.explainer")


def explain_forecast(
    ticker: str,
    forecasts: dict,
    regime: dict,
    news_sentiment: float = 0.0,
    tech_features: pd.DataFrame | None = None,
) -> dict:
    """
    Generate a structured explanation for a forecast.
    Returns dict with summary, factors (bullish/bearish), and detail text.
    """
    bullish_factors = []
    bearish_factors = []
    neutral_factors = []

    # 1. Regime assessment
    regime_label = regime.get("regime", "Unknown")
    regime_probs = regime.get("probabilities", {})
    if regime_label == "Bull":
        bullish_factors.append(f"Market regime: {regime_label} ({regime_probs.get('Bull', 0):.0%} confidence)")
    elif regime_label == "Bear":
        bearish_factors.append(f"Market regime: {regime_label} ({regime_probs.get('Bear', 0):.0%} confidence)")
    else:
        neutral_factors.append(f"Market regime: {regime_label} (transitional)")

    # 2. Technical signals
    if tech_features is not None and not tech_features.empty:
        latest = tech_features.iloc[-1]

        # RSI
        rsi = latest.get("RSI_14")
        if rsi is not None:
            if rsi > 70:
                bearish_factors.append(f"RSI overbought ({rsi:.0f})")
            elif rsi < 30:
                bullish_factors.append(f"RSI oversold ({rsi:.0f})")

        # Moving average position
        price_vs_200 = latest.get("Price_vs_SMA200")
        if price_vs_200 is not None:
            if price_vs_200 > 0:
                bullish_factors.append(f"Price above 200-day MA (+{price_vs_200:.1f}%)")
            else:
                bearish_factors.append(f"Price below 200-day MA ({price_vs_200:.1f}%)")

        # MACD
        macd_hist = latest.get("MACD_Hist")
        if macd_hist is not None:
            if macd_hist > 0:
                bullish_factors.append("MACD histogram positive (upward momentum)")
            else:
                bearish_factors.append("MACD histogram negative (downward momentum)")

        # ADX trend strength
        adx = latest.get("ADX")
        if adx is not None and adx > 25:
            direction = "bullish" if latest.get("ADX_Pos", 0) > latest.get("ADX_Neg", 0) else "bearish"
            if direction == "bullish":
                bullish_factors.append(f"Strong {direction} trend (ADX={adx:.0f})")
            else:
                bearish_factors.append(f"Strong {direction} trend (ADX={adx:.0f})")

        # Bollinger Band position
        bb_pct = latest.get("BB_Pct")
        if bb_pct is not None:
            if bb_pct > 1.0:
                bearish_factors.append("Price above upper Bollinger Band (overbought)")
            elif bb_pct < 0.0:
                bullish_factors.append("Price below lower Bollinger Band (oversold)")

        # Volume
        vol_ratio = latest.get("Vol_Ratio_20d")
        if vol_ratio is not None and vol_ratio > 1.5:
            neutral_factors.append(f"Volume {vol_ratio:.1f}x above 20-day average (heightened activity)")

    # 3. News sentiment
    if news_sentiment > 0.15:
        bullish_factors.append(f"News sentiment positive ({news_sentiment:.2f})")
    elif news_sentiment < -0.15:
        bearish_factors.append(f"News sentiment negative ({news_sentiment:.2f})")

    # 4. Build summary
    primary_forecast = None
    for horizon in ["1W", "1M", "3M"]:
        if horizon in forecasts:
            primary_forecast = forecasts[horizon]
            break

    if primary_forecast:
        direction = primary_forecast.get("direction_label", "Neutral")
        confidence = primary_forecast.get("confidence", 0)
        point = primary_forecast.get("point", 0) * 100

        summary = f"{ticker} outlook: {direction} ({confidence:.0%} confidence, {point:+.1f}% expected)"
    else:
        summary = f"{ticker}: Insufficient data for forecast"

    # 5. Build detail text
    detail_lines = [summary, ""]

    if bullish_factors:
        detail_lines.append("Bullish factors:")
        for f in bullish_factors:
            detail_lines.append(f"  + {f}")

    if bearish_factors:
        detail_lines.append("Bearish factors:")
        for f in bearish_factors:
            detail_lines.append(f"  - {f}")

    if neutral_factors:
        detail_lines.append("Neutral / Watch:")
        for f in neutral_factors:
            detail_lines.append(f"  ~ {f}")

    detail_text = "\n".join(detail_lines)

    return {
        "summary": summary,
        "bullish_factors": bullish_factors,
        "bearish_factors": bearish_factors,
        "neutral_factors": neutral_factors,
        "detail": detail_text,
        "n_bullish": len(bullish_factors),
        "n_bearish": len(bearish_factors),
        "net_signal": len(bullish_factors) - len(bearish_factors),
    }

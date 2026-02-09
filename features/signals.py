"""
SENTINEL — Composite Signal Scoring Engine.
Combines technical, momentum, volume, trend, and macro signals into a single
0-100 conviction score for any asset. Higher = more bullish.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.technical import compute_all_technical
from utils.logger import get_logger

log = get_logger("features.signals")


def _score_rsi(rsi: float, trend_score: float = 50.0) -> float:
    """RSI signal — trend-aware.

    In strong uptrends (trend_score > 65): high RSI confirms strength, not overbought.
    In strong downtrends (trend_score < 35): low RSI confirms weakness, not oversold.
    In sideways markets: classic mean-reversion (oversold=buy, overbought=sell).
    """
    if pd.isna(rsi):
        return 50.0

    strong_uptrend = trend_score >= 65
    strong_downtrend = trend_score <= 35

    if strong_uptrend:
        # In uptrends, RSI 50-70 is healthy momentum, >70 is extended but not bearish
        if rsi < 30:
            return 75.0  # Oversold in uptrend — strong buy
        if rsi < 40:
            return 60.0
        if rsi < 60:
            return 55.0  # Neutral-to-bullish
        if rsi < 75:
            return 60.0  # Momentum confirmation — bullish
        return 45.0  # >75 is stretched even in uptrend — slight caution

    if strong_downtrend:
        # In downtrends, RSI 30-50 is healthy bearish momentum, <30 is extended
        if rsi > 70:
            return 25.0  # Overbought in downtrend — strong sell
        if rsi > 60:
            return 40.0
        if rsi > 40:
            return 45.0  # Neutral-to-bearish
        if rsi > 25:
            return 40.0  # Momentum confirmation — bearish
        return 55.0  # <25 is stretched even in downtrend — slight bounce possible

    # Sideways / no strong trend — classic mean reversion
    if rsi < 25:
        return 85.0  # Extremely oversold
    if rsi < 30:
        return 75.0
    if rsi < 40:
        return 60.0
    if rsi < 60:
        return 50.0  # Neutral
    if rsi < 70:
        return 40.0
    if rsi < 75:
        return 30.0
    return 15.0  # Extremely overbought


def _score_macd(macd_hist: float, macd_hist_prev: float) -> float:
    """MACD histogram: positive and rising = bullish."""
    if pd.isna(macd_hist) or pd.isna(macd_hist_prev):
        return 50.0
    if macd_hist > 0 and macd_hist > macd_hist_prev:
        return 80.0  # Positive and accelerating
    if macd_hist > 0:
        return 65.0  # Positive but decelerating
    if macd_hist < 0 and macd_hist > macd_hist_prev:
        return 45.0  # Negative but improving
    return 20.0  # Negative and worsening


def _score_trend(price: float, sma_20: float, sma_50: float, sma_200: float) -> float:
    """Moving average alignment: stacked bullish or bearish."""
    if any(pd.isna(v) for v in [price, sma_20, sma_50, sma_200]):
        return 50.0
    above_20 = price > sma_20
    above_50 = price > sma_50
    above_200 = price > sma_200
    sma_20_above_50 = sma_20 > sma_50
    sma_50_above_200 = sma_50 > sma_200

    bullish_count = sum([above_20, above_50, above_200, sma_20_above_50, sma_50_above_200])
    return 10 + bullish_count * 16  # Maps 0-5 to 10-90


def _score_volume(vol_ratio: float, daily_return: float = 0.0) -> float:
    """Volume relative to average, direction-aware: high volume on up day = bullish,
    high volume on down day = bearish, low volume = weak signal either way."""
    if pd.isna(vol_ratio):
        return 50.0
    up_day = daily_return >= 0 if not pd.isna(daily_return) else True

    if vol_ratio > 2.0:
        return 85.0 if up_day else 15.0  # Strong confirmation either direction
    if vol_ratio > 1.5:
        return 70.0 if up_day else 30.0
    if vol_ratio > 0.8:
        return 55.0 if up_day else 45.0  # Normal volume, slight directional lean
    return 40.0  # Low volume — move is suspect regardless of direction


def _score_adx(adx: float) -> float:
    """Trend strength: strong trends are tradeable."""
    if pd.isna(adx):
        return 50.0
    if adx > 40:
        return 85.0  # Very strong trend
    if adx > 25:
        return 70.0  # Trending
    if adx > 20:
        return 55.0  # Weak trend
    return 35.0  # No trend — choppy market


def _score_bollinger(bb_pct: float) -> float:
    """Bollinger Band %B: mean-reversion signal."""
    if pd.isna(bb_pct):
        return 50.0
    if bb_pct < 0:
        return 85.0  # Below lower band — oversold bounce
    if bb_pct < 0.2:
        return 70.0
    if bb_pct < 0.8:
        return 50.0
    if bb_pct < 1.0:
        return 30.0
    return 15.0  # Above upper band — overbought


def _score_momentum(ret_1w: float, ret_1m: float, ret_3m: float) -> float:
    """Multi-timeframe momentum alignment."""
    if all(pd.isna(v) for v in [ret_1w, ret_1m, ret_3m]):
        return 50.0
    scores = []
    for ret in [ret_1w, ret_1m, ret_3m]:
        if pd.isna(ret):
            scores.append(50.0)
        elif ret > 0.05:
            scores.append(80.0)
        elif ret > 0.02:
            scores.append(65.0)
        elif ret > -0.02:
            scores.append(50.0)
        elif ret > -0.05:
            scores.append(35.0)
        else:
            scores.append(20.0)
    return np.mean(scores)


def _score_volatility_regime(vol_7d: float, vol_30d: float) -> float:
    """Volatility contraction precedes expansion (big moves)."""
    if pd.isna(vol_7d) or pd.isna(vol_30d) or vol_30d == 0:
        return 50.0
    ratio = vol_7d / vol_30d
    if ratio < 0.6:
        return 80.0  # Vol squeeze — big move incoming
    if ratio < 0.8:
        return 65.0
    if ratio < 1.2:
        return 50.0  # Normal
    if ratio < 1.5:
        return 40.0  # Vol expanding
    return 25.0  # Vol explosion — late to the party


def compute_composite_score(ohlcv: pd.DataFrame) -> dict:
    """
    Compute a composite 0-100 conviction score for an asset.
    Returns dict with overall score, component scores, and signal details.
    """
    if ohlcv.empty or len(ohlcv) < 50:
        return {"score": 50, "label": "INSUFFICIENT DATA", "components": {}}

    tech = compute_all_technical(ohlcv)
    if tech.empty:
        return {"score": 50, "label": "INSUFFICIENT DATA", "components": {}}

    latest = tech.iloc[-1]
    prev = tech.iloc[-2] if len(tech) > 1 else latest
    close = ohlcv["Close"]

    # Component scores
    components = {}

    # Trend (MA alignment) — compute FIRST so RSI can use it
    components["Trend"] = _score_trend(
        close.iloc[-1],
        latest.get("SMA_20"),
        latest.get("SMA_50"),
        latest.get("SMA_200"),
    )

    # RSI — trend-aware: doesn't fight the trend anymore
    components["RSI"] = _score_rsi(latest.get("RSI_14"), trend_score=components["Trend"])

    # MACD
    components["MACD"] = _score_macd(
        latest.get("MACD_Hist"), prev.get("MACD_Hist")
    )

    # Volume (direction-aware: high volume + up day = bullish, high volume + down day = bearish)
    vol = ohlcv["Volume"] if "Volume" in ohlcv.columns else pd.Series(dtype=float)
    vol_ratio = vol.iloc[-1] / vol.rolling(20).mean().iloc[-1] if len(vol) > 20 and vol.rolling(20).mean().iloc[-1] > 0 else np.nan
    daily_ret = close.pct_change().iloc[-1] if len(close) > 1 else 0.0
    components["Volume"] = _score_volume(vol_ratio, daily_ret)

    # ADX (Trend Strength)
    components["Trend_Strength"] = _score_adx(latest.get("ADX"))

    # Bollinger Bands
    components["Mean_Reversion"] = _score_bollinger(latest.get("BB_Pct"))

    # Multi-timeframe momentum
    ret_1w = close.pct_change(5).iloc[-1] if len(close) > 5 else np.nan
    ret_1m = close.pct_change(21).iloc[-1] if len(close) > 21 else np.nan
    ret_3m = close.pct_change(63).iloc[-1] if len(close) > 63 else np.nan
    components["Momentum"] = _score_momentum(ret_1w, ret_1m, ret_3m)

    # Volatility regime
    vol_7d = close.pct_change().rolling(7).std().iloc[-1] if len(close) > 7 else np.nan
    vol_30d = close.pct_change().rolling(30).std().iloc[-1] if len(close) > 30 else np.nan
    components["Volatility"] = _score_volatility_regime(vol_7d, vol_30d)

    # Weighted composite
    weights = {
        "Trend": 0.20,
        "Momentum": 0.20,
        "MACD": 0.15,
        "RSI": 0.12,
        "Trend_Strength": 0.10,
        "Volume": 0.08,
        "Mean_Reversion": 0.08,
        "Volatility": 0.07,
    }

    score = sum(components[k] * weights.get(k, 0.1) for k in components)
    score = max(0, min(100, score))

    # Label
    if score >= 80:
        label = "STRONG BUY"
    elif score >= 65:
        label = "BUY"
    elif score >= 55:
        label = "LEAN BULLISH"
    elif score >= 45:
        label = "NEUTRAL"
    elif score >= 35:
        label = "LEAN BEARISH"
    elif score >= 20:
        label = "SELL"
    else:
        label = "STRONG SELL"

    # Key signals summary
    signals = []
    if components["Trend"] >= 75:
        signals.append("All MAs aligned bullish")
    elif components["Trend"] <= 25:
        signals.append("All MAs aligned bearish")

    if components["RSI"] >= 80:
        signals.append("Oversold — bounce likely")
    elif components["RSI"] <= 20:
        signals.append("Overbought — pullback likely")
    elif components["Trend"] >= 65 and latest.get("RSI_14", 50) > 60:
        signals.append("RSI confirms uptrend momentum")

    if components["Volatility"] >= 75:
        signals.append("Volatility squeeze — big move incoming")

    if components["Momentum"] >= 75:
        signals.append("Strong multi-timeframe momentum")
    elif components["Momentum"] <= 25:
        signals.append("Momentum deteriorating across timeframes")

    macd_hist = latest.get("MACD_Hist", 0)
    macd_hist_prev = prev.get("MACD_Hist", 0)
    if not pd.isna(macd_hist) and not pd.isna(macd_hist_prev):
        if macd_hist > 0 and macd_hist_prev < 0:
            signals.append("MACD bullish crossover")
        elif macd_hist < 0 and macd_hist_prev > 0:
            signals.append("MACD bearish crossover")

    return {
        "score": round(score, 1),
        "label": label,
        "components": components,
        "signals": signals,
        "details": {
            "price": close.iloc[-1],
            "rsi": latest.get("RSI_14"),
            "macd_hist": macd_hist,
            "adx": latest.get("ADX"),
            "bb_pct": latest.get("BB_Pct"),
            "ret_1w": ret_1w,
            "ret_1m": ret_1m,
            "ret_3m": ret_3m,
            "vol_ratio": vol_ratio,
        },
    }


def score_multiple(tickers_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Score multiple assets at once.
    Input: dict of ticker -> OHLCV DataFrame
    Returns: DataFrame sorted by score descending.
    """
    rows = []
    for ticker, ohlcv in tickers_data.items():
        result = compute_composite_score(ohlcv)
        rows.append({
            "Ticker": ticker,
            "Score": result["score"],
            "Signal": result["label"],
            "RSI": result["components"].get("RSI", 50),
            "Trend": result["components"].get("Trend", 50),
            "Momentum": result["components"].get("Momentum", 50),
            "MACD": result["components"].get("MACD", 50),
            "Volume": result["components"].get("Volume", 50),
            "Volatility": result["components"].get("Volatility", 50),
            "Price": result["details"].get("price", 0),
            "1W_Ret": result["details"].get("ret_1w", 0),
            "1M_Ret": result["details"].get("ret_1m", 0),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    return df

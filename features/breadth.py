"""
SENTINEL — Market Breadth Analysis.
The indicators that institutional traders use to call tops and bottoms.
Breadth divergences from price are the most reliable predictive signals.
"""

import numpy as np
import pandas as pd

from data.stocks import fetch_ohlcv
from config.assets import SECTORS
from utils.logger import get_logger

log = get_logger("features.breadth")


def compute_sector_breadth() -> dict:
    """
    Compute breadth indicators across all sector holdings.
    Returns dict with breadth metrics and per-stock status.
    """
    all_stocks = []
    for sector_name, info in SECTORS.items():
        for ticker in info["holdings"]:
            all_stocks.append((ticker, sector_name))

    results = []
    for ticker, sector in all_stocks:
        try:
            df = fetch_ohlcv(ticker)
            if df.empty or len(df) < 200:
                continue

            close = df["Close"]
            sma_50 = close.rolling(50).mean().iloc[-1]
            sma_200 = close.rolling(200).mean().iloc[-1]
            price = close.iloc[-1]
            high_52w = close.tail(252).max()
            low_52w = close.tail(252).min()
            ret_1d = close.pct_change().iloc[-1]
            ret_1w = close.pct_change(5).iloc[-1]
            ret_1m = close.pct_change(21).iloc[-1]

            results.append({
                "Ticker": ticker,
                "Sector": sector,
                "Price": price,
                "Above_50MA": price > sma_50,
                "Above_200MA": price > sma_200,
                "Golden_Cross": sma_50 > sma_200,
                "Pct_From_52W_High": (price / high_52w - 1) * 100,
                "Pct_From_52W_Low": (price / low_52w - 1) * 100,
                "Near_52W_High": (price / high_52w) > 0.95,
                "Near_52W_Low": (price / low_52w) < 1.05,
                "Ret_1D": ret_1d,
                "Ret_1W": ret_1w,
                "Ret_1M": ret_1m,
                "Advancing": ret_1d > 0,
            })
        except Exception:
            continue

    if not results:
        return {"breadth_df": pd.DataFrame(), "summary": {}}

    df = pd.DataFrame(results)
    n = len(df)

    summary = {
        "total_stocks": n,
        "pct_above_50ma": df["Above_50MA"].mean() * 100,
        "pct_above_200ma": df["Above_200MA"].mean() * 100,
        "pct_golden_cross": df["Golden_Cross"].mean() * 100,
        "pct_advancing": df["Advancing"].mean() * 100,
        "pct_near_52w_high": df["Near_52W_High"].mean() * 100,
        "pct_near_52w_low": df["Near_52W_Low"].mean() * 100,
        "avg_1d_return": df["Ret_1D"].mean() * 100,
        "avg_1w_return": df["Ret_1W"].mean() * 100,
        "avg_1m_return": df["Ret_1M"].mean() * 100,
        "advance_decline_ratio": df["Advancing"].sum() / max(1, (~df["Advancing"]).sum()),
    }

    # Breadth health score (0-100)
    health = (
        summary["pct_above_50ma"] * 0.25
        + summary["pct_above_200ma"] * 0.25
        + summary["pct_advancing"] * 0.20
        + summary["pct_golden_cross"] * 0.15
        + (100 - summary["pct_near_52w_low"]) * 0.15
    )
    summary["health_score"] = round(max(0, min(100, health)), 1)

    # Breadth interpretation
    if summary["health_score"] >= 75:
        summary["interpretation"] = "HEALTHY — Broad participation confirms uptrend"
    elif summary["health_score"] >= 60:
        summary["interpretation"] = "MODERATE — Selective participation, watch for narrowing"
    elif summary["health_score"] >= 40:
        summary["interpretation"] = "WEAKENING — Fewer stocks participating, risk rising"
    elif summary["health_score"] >= 25:
        summary["interpretation"] = "DETERIORATING — Breadth divergence, correction likely"
    else:
        summary["interpretation"] = "CRITICAL — Extreme weakness, capitulation possible"

    # Sector-level breadth
    sector_breadth = df.groupby("Sector").agg({
        "Above_50MA": "mean",
        "Above_200MA": "mean",
        "Advancing": "mean",
        "Ret_1D": "mean",
        "Ret_1W": "mean",
        "Ret_1M": "mean",
    }).round(4)
    sector_breadth.columns = [
        "Pct_Above_50MA", "Pct_Above_200MA", "Pct_Advancing",
        "Avg_1D_Ret", "Avg_1W_Ret", "Avg_1M_Ret",
    ]

    summary["sector_breadth"] = sector_breadth

    return {"breadth_df": df, "summary": summary}


def compute_breadth_divergence(spy_close: pd.Series, breadth_pct_above_50ma: float) -> str:
    """
    Detect breadth divergence: price making highs but breadth weakening.
    This is the #1 signal for calling market tops.
    """
    if spy_close.empty:
        return "NO DATA"

    high_52w = spy_close.tail(252).max()
    pct_from_high = (spy_close.iloc[-1] / high_52w - 1) * 100

    # Price near highs but breadth weak = BEARISH DIVERGENCE
    if pct_from_high > -3 and breadth_pct_above_50ma < 60:
        return "BEARISH DIVERGENCE — Price near highs but fewer stocks participating"

    # Price falling but breadth improving = BULLISH DIVERGENCE
    if pct_from_high < -10 and breadth_pct_above_50ma > 50:
        return "BULLISH DIVERGENCE — Price down but breadth improving"

    # Everything aligned
    if pct_from_high > -5 and breadth_pct_above_50ma > 70:
        return "CONFIRMED UPTREND — Price and breadth aligned"

    if pct_from_high < -10 and breadth_pct_above_50ma < 40:
        return "CONFIRMED DOWNTREND — Price and breadth aligned bearish"

    return "NEUTRAL — No significant divergence"

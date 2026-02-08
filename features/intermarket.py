"""
SENTINEL — Intermarket Intelligence.
When bonds, stocks, gold, and dollar disagree, the biggest moves follow.
These divergences are what separate amateurs from professionals.
"""

import numpy as np
import pandas as pd

from data.stocks import fetch_ohlcv
from utils.logger import get_logger

log = get_logger("features.intermarket")

# Key intermarket relationships:
# 1. Stocks vs Bonds (TLT) — Usually inverse. Same direction = something breaking.
# 2. Stocks vs Dollar (UUP) — Weak dollar = bullish stocks (usually).
# 3. Gold vs Dollar — Usually inverse. Both rising = panic.
# 4. VIX vs Stocks — Always inverse. Divergence = hidden risk.
# 5. Copper vs Gold — Rising ratio = economic optimism. Falling = fear.
# 6. High Yield (HYG) vs Investment Grade (LQD) — Credit stress indicator.


def compute_intermarket_signals() -> dict:
    """
    Analyze cross-asset relationships and detect divergences.
    Returns dict with signals, correlations, and divergence alerts.
    """
    tickers = {
        "SPY": "S&P 500",
        "QQQ": "Nasdaq 100",
        "TLT": "20Y+ Treasuries",
        "GLD": "Gold",
        "UUP": "US Dollar",
        "HYG": "High Yield Bonds",
        "LQD": "Inv. Grade Bonds",
        "USO": "Crude Oil",
    }

    # Fetch all data
    data = {}
    for ticker in tickers:
        df = fetch_ohlcv(ticker)
        if not df.empty:
            data[ticker] = df["Close"]

    if len(data) < 4:
        return {"signals": [], "correlations": pd.DataFrame(), "divergences": []}

    # Build returns
    closes = pd.DataFrame(data).dropna()
    if closes.empty or len(closes) < 60:
        return {"signals": [], "correlations": pd.DataFrame(), "divergences": []}

    returns_1w = closes.pct_change(5).iloc[-1]
    returns_1m = closes.pct_change(21).iloc[-1]
    returns_3m = closes.pct_change(63).iloc[-1] if len(closes) > 63 else pd.Series(dtype=float)

    # Rolling correlations (60-day)
    daily_returns = closes.pct_change().dropna()
    corr_60d = daily_returns.tail(60).corr()

    # 20-day rolling correlation for detecting regime shifts
    rolling_corr_spy_tlt = daily_returns["SPY"].rolling(20).corr(daily_returns.get("TLT", pd.Series(dtype=float)))

    signals = []
    divergences = []

    # ── Signal 1: Stock-Bond Correlation Regime ──────────────
    if "SPY" in returns_1m and "TLT" in returns_1m:
        spy_ret = returns_1m.get("SPY", 0)
        tlt_ret = returns_1m.get("TLT", 0)
        spy_tlt_corr = corr_60d.get("SPY", {}).get("TLT", 0) if "TLT" in corr_60d.columns else 0

        if spy_ret > 0.02 and tlt_ret > 0.02:
            signals.append({
                "signal": "RISK-ON EVERYTHING",
                "detail": "Stocks AND bonds both rising — liquidity flood or Fed pivot expectations",
                "strength": "STRONG",
                "direction": "BULLISH",
            })
        elif spy_ret < -0.02 and tlt_ret < -0.02:
            signals.append({
                "signal": "SELL EVERYTHING",
                "detail": "Stocks AND bonds both falling — liquidity drain or stagflation fear",
                "strength": "STRONG",
                "direction": "BEARISH",
            })

        if spy_tlt_corr > 0.3:
            divergences.append(
                "Stock-Bond correlation POSITIVE (unusual) — macro regime shift underway"
            )

    # ── Signal 2: Gold + Dollar ─────────────────────────────
    if "GLD" in returns_1m and "UUP" in returns_1m:
        gld_ret = returns_1m.get("GLD", 0)
        uup_ret = returns_1m.get("UUP", 0)

        if gld_ret > 0.02 and uup_ret > 0.01:
            signals.append({
                "signal": "FEAR TRADE ACTIVE",
                "detail": "Gold AND dollar both rising — flight to safety, crisis mode",
                "strength": "STRONG",
                "direction": "BEARISH",
            })
        elif gld_ret > 0.03 and uup_ret < -0.01:
            signals.append({
                "signal": "INFLATION HEDGE",
                "detail": "Gold up, dollar down — inflation expectations rising",
                "strength": "MODERATE",
                "direction": "NEUTRAL",
            })

    # ── Signal 3: Credit Stress ─────────────────────────────
    if "HYG" in returns_1m and "LQD" in returns_1m:
        hyg_ret = returns_1m.get("HYG", 0)
        lqd_ret = returns_1m.get("LQD", 0)
        spread = hyg_ret - lqd_ret

        if spread < -0.02:
            signals.append({
                "signal": "CREDIT STRESS RISING",
                "detail": "High yield underperforming investment grade — credit risk increasing",
                "strength": "STRONG",
                "direction": "BEARISH",
            })
        elif spread > 0.02:
            signals.append({
                "signal": "CREDIT APPETITE STRONG",
                "detail": "High yield outperforming — risk appetite healthy",
                "strength": "MODERATE",
                "direction": "BULLISH",
            })

    # ── Signal 4: Dollar Impact ─────────────────────────────
    if "UUP" in returns_1m and "SPY" in returns_1m:
        uup_ret = returns_1m.get("UUP", 0)
        if uup_ret > 0.02:
            signals.append({
                "signal": "STRONG DOLLAR HEADWIND",
                "detail": "Rising dollar pressures multinational earnings and emerging markets",
                "strength": "MODERATE",
                "direction": "BEARISH",
            })
        elif uup_ret < -0.02:
            signals.append({
                "signal": "WEAK DOLLAR TAILWIND",
                "detail": "Falling dollar boosts corporate earnings and risk assets",
                "strength": "MODERATE",
                "direction": "BULLISH",
            })

    # ── Signal 5: Oil + Stocks ──────────────────────────────
    if "USO" in returns_1m:
        uso_ret = returns_1m.get("USO", 0)
        if uso_ret > 0.10:
            signals.append({
                "signal": "OIL SURGE",
                "detail": "Crude oil spiking — inflation risk, consumer squeeze, energy sector boost",
                "strength": "STRONG",
                "direction": "MIXED",
            })
        elif uso_ret < -0.10:
            signals.append({
                "signal": "OIL COLLAPSE",
                "detail": "Crude oil crashing — deflation signal or demand destruction",
                "strength": "MODERATE",
                "direction": "MIXED",
            })

    # ── Signal 6: Tech vs Value Rotation ────────────────────
    if "QQQ" in returns_1m and "SPY" in returns_1m:
        qqq_ret = returns_1m.get("QQQ", 0)
        spy_ret = returns_1m.get("SPY", 0)
        rotation = qqq_ret - spy_ret

        if rotation > 0.03:
            signals.append({
                "signal": "GROWTH LEADERSHIP",
                "detail": "Tech/growth outperforming — risk-on, momentum chasing",
                "strength": "MODERATE",
                "direction": "BULLISH",
            })
        elif rotation < -0.03:
            signals.append({
                "signal": "VALUE ROTATION",
                "detail": "Broad market outperforming tech — defensive rotation underway",
                "strength": "MODERATE",
                "direction": "NEUTRAL",
            })

    # Summary interpretation
    bullish_count = sum(1 for s in signals if s["direction"] == "BULLISH")
    bearish_count = sum(1 for s in signals if s["direction"] == "BEARISH")

    if bullish_count > bearish_count + 1:
        overall = "BULLISH — Cross-asset signals favor risk-on positioning"
    elif bearish_count > bullish_count + 1:
        overall = "BEARISH — Cross-asset signals warn of risk-off environment"
    else:
        overall = "MIXED — Cross-asset signals are conflicting, stay nimble"

    return {
        "signals": signals,
        "divergences": divergences,
        "correlations": corr_60d,
        "returns_1w": returns_1w.to_dict() if not returns_1w.empty else {},
        "returns_1m": returns_1m.to_dict() if not returns_1m.empty else {},
        "overall": overall,
        "closes": closes,
    }

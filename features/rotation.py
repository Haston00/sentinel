"""
SENTINEL — Sector Rotation Model.
Maps current economic cycle position and predicts which sectors lead next.
Based on the classic economic cycle framework + momentum confirmation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.stocks import fetch_ohlcv
from config.assets import SECTORS
from utils.logger import get_logger

log = get_logger("features.rotation")

# Economic cycle sector leadership mapping
# Source: Standard institutional sector rotation framework
CYCLE_SECTORS = {
    "Early Recovery": {
        "leaders": ["Financials", "Consumer Discretionary", "Technology", "Real Estate"],
        "laggards": ["Utilities", "Consumer Staples", "Healthcare"],
        "description": "Economy recovering, rates low, credit expanding",
    },
    "Mid Cycle": {
        "leaders": ["Technology", "Industrials", "Materials", "Communication Services"],
        "laggards": ["Utilities", "Consumer Staples"],
        "description": "Economy growing, corporate earnings strong, moderate rates",
    },
    "Late Cycle": {
        "leaders": ["Energy", "Materials", "Consumer Staples", "Healthcare"],
        "laggards": ["Technology", "Consumer Discretionary", "Financials"],
        "description": "Economy overheating, inflation rising, rates high",
    },
    "Recession": {
        "leaders": ["Utilities", "Consumer Staples", "Healthcare"],
        "laggards": ["Consumer Discretionary", "Financials", "Technology", "Industrials"],
        "description": "Economy contracting, flight to safety, rates falling",
    },
}


def compute_sector_momentum() -> pd.DataFrame:
    """
    Compute relative momentum for each sector across multiple timeframes.
    Returns DataFrame with sector momentum scores.
    """
    spy_data = fetch_ohlcv("SPY")
    if spy_data.empty:
        return pd.DataFrame()

    spy_close = spy_data["Close"]
    rows = []

    for sector_name, info in SECTORS.items():
        etf_data = fetch_ohlcv(info["etf"])
        if etf_data.empty or len(etf_data) < 63:
            continue

        etf_close = etf_data["Close"]

        # Absolute returns
        ret_1w = etf_close.pct_change(5).iloc[-1] if len(etf_close) > 5 else 0
        ret_1m = etf_close.pct_change(21).iloc[-1] if len(etf_close) > 21 else 0
        ret_3m = etf_close.pct_change(63).iloc[-1] if len(etf_close) > 63 else 0

        # Relative returns (vs SPY)
        common = min(len(etf_close), len(spy_close))
        rel_1w = ret_1w - spy_close.pct_change(5).iloc[-1] if common > 5 else 0
        rel_1m = ret_1m - spy_close.pct_change(21).iloc[-1] if common > 21 else 0
        rel_3m = ret_3m - spy_close.pct_change(63).iloc[-1] if common > 63 else 0

        # Composite momentum score (weighted)
        momentum_score = rel_1w * 0.3 + rel_1m * 0.4 + rel_3m * 0.3

        # RSI of sector ETF
        delta = etf_close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100
        rsi = 100 - (100 / (1 + rs))

        rows.append({
            "Sector": sector_name,
            "ETF": info["etf"],
            "Price": etf_close.iloc[-1],
            "1W_Ret": ret_1w,
            "1M_Ret": ret_1m,
            "3M_Ret": ret_3m,
            "1W_Rel": rel_1w,
            "1M_Rel": rel_1m,
            "3M_Rel": rel_3m,
            "Momentum_Score": momentum_score,
            "RSI": rsi,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("Momentum_Score", ascending=False)

    # Rank sectors
    df["Rank"] = range(1, len(df) + 1)

    # Rotation signal
    df["Signal"] = df["Momentum_Score"].apply(
        lambda x: "OVERWEIGHT" if x > 0.02
        else "UNDERWEIGHT" if x < -0.02
        else "MARKET WEIGHT"
    )

    return df.set_index("Sector")


def detect_cycle_phase(sector_momentum: pd.DataFrame) -> dict:
    """
    Estimate current economic cycle phase based on sector leadership patterns.
    Compares actual sector performance to theoretical cycle framework.
    """
    if sector_momentum.empty:
        return {"phase": "UNKNOWN", "confidence": 0, "description": "Insufficient data"}

    best_match = None
    best_score = -1

    for phase, config in CYCLE_SECTORS.items():
        score = 0
        total_checks = 0

        # Check if theoretical leaders are actually leading
        for sector in config["leaders"]:
            if sector in sector_momentum.index:
                total_checks += 1
                if sector_momentum.loc[sector, "Momentum_Score"] > 0:
                    score += 1
                if sector_momentum.loc[sector, "Rank"] <= 5:
                    score += 0.5

        # Check if theoretical laggards are actually lagging
        for sector in config["laggards"]:
            if sector in sector_momentum.index:
                total_checks += 1
                if sector_momentum.loc[sector, "Momentum_Score"] < 0:
                    score += 1
                if sector_momentum.loc[sector, "Rank"] > 6:
                    score += 0.5

        if total_checks > 0:
            normalized = score / total_checks
            if normalized > best_score:
                best_score = normalized
                best_match = phase

    if best_match is None:
        return {"phase": "UNKNOWN", "confidence": 0, "description": "Cannot determine"}

    confidence = min(100, round(best_score * 100))
    config = CYCLE_SECTORS[best_match]

    return {
        "phase": best_match,
        "confidence": confidence,
        "description": config["description"],
        "expected_leaders": config["leaders"],
        "expected_laggards": config["laggards"],
    }


def get_rotation_recommendations(sector_momentum: pd.DataFrame, cycle_phase: dict) -> list[dict]:
    """
    Generate actionable sector rotation recommendations.
    """
    if sector_momentum.empty:
        return []

    recs = []
    phase = cycle_phase.get("phase", "UNKNOWN")
    expected_leaders = cycle_phase.get("expected_leaders", [])
    expected_laggards = cycle_phase.get("expected_laggards", [])

    for sector in sector_momentum.index:
        row = sector_momentum.loc[sector]
        momentum = row["Momentum_Score"]
        rank = row["Rank"]
        rsi = row["RSI"]

        is_expected_leader = sector in expected_leaders
        is_expected_laggard = sector in expected_laggards

        # High conviction: cycle says lead + momentum confirms
        if is_expected_leader and momentum > 0.01:
            action = "STRONG OVERWEIGHT"
            reason = f"Cycle phase ({phase}) + momentum both favor {sector}"
            conviction = "HIGH"
        elif is_expected_leader and momentum < -0.01:
            action = "WATCH"
            reason = f"Cycle favors {sector} but momentum hasn't confirmed yet"
            conviction = "LOW"
        elif is_expected_laggard and momentum < -0.01:
            action = "STRONG UNDERWEIGHT"
            reason = f"Cycle phase ({phase}) + momentum both against {sector}"
            conviction = "HIGH"
        elif is_expected_laggard and momentum > 0.01:
            action = "CAUTION"
            reason = f"Momentum favors {sector} but cycle phase suggests rotation away"
            conviction = "MODERATE"
        elif momentum > 0.02:
            action = "OVERWEIGHT"
            reason = "Strong relative momentum"
            conviction = "MODERATE"
        elif momentum < -0.02:
            action = "UNDERWEIGHT"
            reason = "Weak relative momentum"
            conviction = "MODERATE"
        else:
            action = "MARKET WEIGHT"
            reason = "No strong signal"
            conviction = "LOW"

        # RSI override for extremes
        if rsi > 80:
            reason += " | Caution: RSI overbought"
        elif rsi < 25:
            reason += " | Opportunity: RSI oversold"

        recs.append({
            "Sector": sector,
            "Action": action,
            "Conviction": conviction,
            "Reason": reason,
            "Momentum": momentum,
            "Rank": rank,
            "RSI": rsi,
        })

    return sorted(recs, key=lambda x: {"HIGH": 0, "MODERATE": 1, "LOW": 2}.get(x["Conviction"], 3))

"""
SENTINEL — Options Flow Intelligence.
Pulls options chain data, detects unusual activity, computes put/call ratios,
and identifies smart money positioning as leading indicators.

Hedge fund edge: Options market often leads stock market by 1-5 days.
Unusual options volume = institutional positioning that hasn't hit the tape yet.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config.settings import CACHE_DIR
from utils.logger import get_logger

log = get_logger("data.options")

OPTIONS_CACHE = CACHE_DIR / "options"
OPTIONS_CACHE.mkdir(parents=True, exist_ok=True)


def fetch_options_chain(ticker: str) -> dict:
    """
    Fetch full options chain for a ticker via yfinance.
    Returns dict with 'calls', 'puts', and 'expirations'.
    """
    import yfinance as yf

    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "expirations": []}

        all_calls = []
        all_puts = []
        for exp in expirations[:6]:  # First 6 expirations (nearest term most relevant)
            try:
                chain = stock.option_chain(exp)
                calls = chain.calls.copy()
                puts = chain.puts.copy()
                calls["expiration"] = exp
                puts["expiration"] = exp
                all_calls.append(calls)
                all_puts.append(puts)
            except Exception:
                continue

        calls_df = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
        puts_df = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()

        return {
            "calls": calls_df,
            "puts": puts_df,
            "expirations": list(expirations),
        }
    except Exception as e:
        log.warning(f"Options chain fetch failed for {ticker}: {e}")
        return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "expirations": []}


def compute_put_call_ratio(chain: dict) -> dict:
    """
    Compute put/call ratios from options chain.
    P/C > 1.0 = bearish sentiment, P/C < 0.7 = bullish.
    Uses both volume and open interest for robustness.
    """
    calls = chain.get("calls", pd.DataFrame())
    puts = chain.get("puts", pd.DataFrame())

    if calls.empty or puts.empty:
        return {"volume_pc": None, "oi_pc": None, "signal": "no_data"}

    # Volume-based P/C ratio
    call_volume = calls["volume"].sum() if "volume" in calls.columns else 0
    put_volume = puts["volume"].sum() if "volume" in puts.columns else 0
    volume_pc = put_volume / max(call_volume, 1)

    # Open Interest-based P/C ratio
    call_oi = calls["openInterest"].sum() if "openInterest" in calls.columns else 0
    put_oi = puts["openInterest"].sum() if "openInterest" in puts.columns else 0
    oi_pc = put_oi / max(call_oi, 1)

    # Signal classification
    avg_pc = (volume_pc + oi_pc) / 2
    if avg_pc > 1.2:
        signal = "strongly_bearish"
    elif avg_pc > 1.0:
        signal = "bearish"
    elif avg_pc > 0.7:
        signal = "neutral"
    elif avg_pc > 0.5:
        signal = "bullish"
    else:
        signal = "strongly_bullish"

    return {
        "volume_pc": round(volume_pc, 3),
        "oi_pc": round(oi_pc, 3),
        "avg_pc": round(avg_pc, 3),
        "call_volume": int(call_volume),
        "put_volume": int(put_volume),
        "call_oi": int(call_oi),
        "put_oi": int(put_oi),
        "signal": signal,
    }


def detect_unusual_activity(chain: dict, volume_threshold: float = 2.0) -> list[dict]:
    """
    Detect unusual options activity — contracts with volume >> open interest.
    This is the #1 smart money signal. When volume > 2× open interest,
    someone is opening a large new position.

    Returns list of unusual contracts sorted by volume/OI ratio.
    """
    unusual = []

    for option_type, df in [("call", chain.get("calls", pd.DataFrame())),
                             ("put", chain.get("puts", pd.DataFrame()))]:
        if df.empty:
            continue

        for col in ["volume", "openInterest", "strike", "impliedVolatility"]:
            if col not in df.columns:
                continue

        df = df.dropna(subset=["volume", "openInterest"])
        df = df[df["openInterest"] > 0]

        if df.empty:
            continue

        df = df.copy()
        df["vol_oi_ratio"] = df["volume"] / df["openInterest"]

        # Unusual = volume significantly exceeds open interest
        hot = df[df["vol_oi_ratio"] >= volume_threshold].copy()

        for _, row in hot.iterrows():
            unusual.append({
                "type": option_type,
                "strike": float(row.get("strike", 0)),
                "expiration": str(row.get("expiration", "")),
                "volume": int(row.get("volume", 0)),
                "open_interest": int(row.get("openInterest", 0)),
                "vol_oi_ratio": round(float(row["vol_oi_ratio"]), 2),
                "implied_vol": round(float(row.get("impliedVolatility", 0)), 4),
                "last_price": float(row.get("lastPrice", 0)),
                "in_the_money": bool(row.get("inTheMoney", False)),
            })

    unusual.sort(key=lambda x: x["vol_oi_ratio"], reverse=True)
    return unusual[:30]  # Top 30 most unusual


def compute_gamma_exposure(chain: dict, spot_price: float) -> dict:
    """
    Estimate dealer gamma exposure (GEX).
    When dealers are long gamma, they dampen moves. When short gamma, moves amplify.
    This is the institutional secret sauce — it predicts volatility regime.
    """
    calls = chain.get("calls", pd.DataFrame())
    puts = chain.get("puts", pd.DataFrame())

    if calls.empty and puts.empty:
        return {"net_gamma": 0, "gamma_flip": None, "regime": "unknown"}

    gamma_by_strike = {}

    for df, multiplier in [(calls, 1), (puts, -1)]:
        if df.empty or "strike" not in df.columns:
            continue
        for _, row in df.iterrows():
            strike = float(row.get("strike", 0))
            oi = float(row.get("openInterest", 0))
            if strike == 0 or oi == 0:
                continue
            # Approximate gamma using Black-Scholes delta relationship
            moneyness = spot_price / strike
            # Simplified gamma proxy: higher near ATM, lower far OTM/ITM
            gamma_proxy = np.exp(-0.5 * (np.log(moneyness) ** 2) / 0.04)
            gamma_dollars = multiplier * oi * 100 * gamma_proxy * spot_price * 0.01
            gamma_by_strike[strike] = gamma_by_strike.get(strike, 0) + gamma_dollars

    if not gamma_by_strike:
        return {"net_gamma": 0, "gamma_flip": None, "regime": "unknown"}

    net_gamma = sum(gamma_by_strike.values())

    # Find gamma flip point (where cumulative gamma crosses zero)
    sorted_strikes = sorted(gamma_by_strike.keys())
    cumulative = 0
    gamma_flip = None
    for strike in sorted_strikes:
        prev = cumulative
        cumulative += gamma_by_strike[strike]
        if prev < 0 and cumulative >= 0:
            gamma_flip = strike
            break
        elif prev >= 0 and cumulative < 0:
            gamma_flip = strike

    regime = "positive_gamma" if net_gamma > 0 else "negative_gamma"

    return {
        "net_gamma": round(net_gamma, 0),
        "gamma_flip": gamma_flip,
        "regime": regime,
        "n_strikes": len(gamma_by_strike),
        "interpretation": (
            "Dealers LONG gamma — expect mean reversion, lower volatility"
            if net_gamma > 0
            else "Dealers SHORT gamma — expect amplified moves, higher volatility"
        ),
    }


def compute_iv_percentile(chain: dict) -> dict:
    """
    Compute implied volatility percentile across the chain.
    High IV percentile = options are expensive = market expects big move.
    """
    calls = chain.get("calls", pd.DataFrame())
    puts = chain.get("puts", pd.DataFrame())

    all_iv = []
    for df in [calls, puts]:
        if not df.empty and "impliedVolatility" in df.columns:
            iv = df["impliedVolatility"].dropna()
            all_iv.extend(iv.tolist())

    if not all_iv:
        return {"mean_iv": None, "median_iv": None, "iv_skew": None}

    all_iv = [v for v in all_iv if v > 0]
    if not all_iv:
        return {"mean_iv": None, "median_iv": None, "iv_skew": None}

    # IV skew: compare put IV to call IV (puts usually more expensive)
    call_iv = calls["impliedVolatility"].dropna().mean() if not calls.empty and "impliedVolatility" in calls.columns else 0
    put_iv = puts["impliedVolatility"].dropna().mean() if not puts.empty and "impliedVolatility" in puts.columns else 0
    skew = put_iv - call_iv if call_iv > 0 and put_iv > 0 else 0

    return {
        "mean_iv": round(float(np.mean(all_iv)), 4),
        "median_iv": round(float(np.median(all_iv)), 4),
        "max_iv": round(float(np.max(all_iv)), 4),
        "min_iv": round(float(np.min(all_iv)), 4),
        "iv_skew": round(float(skew), 4),
        "skew_signal": "bearish_fear" if skew > 0.05 else ("complacent" if skew < -0.02 else "neutral"),
    }


def get_options_intelligence(ticker: str) -> dict:
    """
    Full options intelligence report for a single ticker.
    Combines all signals into one actionable package.
    """
    import yfinance as yf

    # Get spot price
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        spot = float(hist["Close"].iloc[-1]) if not hist.empty else 0
    except Exception:
        spot = 0

    chain = fetch_options_chain(ticker)
    pc_ratio = compute_put_call_ratio(chain)
    unusual = detect_unusual_activity(chain)
    gamma = compute_gamma_exposure(chain, spot) if spot > 0 else {}
    iv_data = compute_iv_percentile(chain)

    # Composite signal: combine all options intelligence
    signals = []
    if pc_ratio.get("signal") in ("strongly_bearish", "bearish"):
        signals.append(-1)
    elif pc_ratio.get("signal") in ("strongly_bullish", "bullish"):
        signals.append(1)
    else:
        signals.append(0)

    # Unusual call vs put activity
    unusual_calls = [u for u in unusual if u["type"] == "call"]
    unusual_puts = [u for u in unusual if u["type"] == "put"]
    if len(unusual_calls) > len(unusual_puts) * 1.5:
        signals.append(1)
    elif len(unusual_puts) > len(unusual_calls) * 1.5:
        signals.append(-1)
    else:
        signals.append(0)

    # Gamma regime
    if gamma.get("regime") == "positive_gamma":
        signals.append(0)  # Mean reversion, less directional
    elif gamma.get("regime") == "negative_gamma":
        signals.append(-0.5)  # Amplified moves, slight bearish tilt

    composite = round(float(np.mean(signals)), 3) if signals else 0

    return {
        "ticker": ticker,
        "spot_price": spot,
        "put_call_ratio": pc_ratio,
        "unusual_activity": unusual,
        "n_unusual": len(unusual),
        "gamma_exposure": gamma,
        "iv_analysis": iv_data,
        "composite_signal": composite,
        "composite_direction": "bullish" if composite > 0.3 else ("bearish" if composite < -0.3 else "neutral"),
    }

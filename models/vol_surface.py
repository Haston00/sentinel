"""
SENTINEL — Volatility Surface Engine.
Builds implied volatility term structure from options data.
Compares implied vol to GARCH realized vol for mispricing signals.

Hedge fund edge: When implied vol > realized vol, options are expensive
(sell vol). When implied < realized, options are cheap (buy vol).
The spread predicts the next move's magnitude.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("models.vol_surface")


def build_iv_term_structure(ticker: str) -> dict:
    """
    Build implied volatility term structure from options chain.
    Shows how IV changes across expiration dates (term structure).
    Normal: upward sloping. Inverted: market expects near-term event.
    """
    import yfinance as yf

    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return {"term_structure": [], "shape": "unknown"}

        hist = stock.history(period="1d")
        spot = float(hist["Close"].iloc[-1]) if not hist.empty else 0

        term_data = []
        for exp in expirations[:10]:
            try:
                chain = stock.option_chain(exp)
                # Get ATM options (nearest to spot price)
                calls = chain.calls
                if calls.empty or "strike" not in calls.columns:
                    continue

                # Find nearest ATM strike
                calls = calls.copy()
                calls["distance"] = abs(calls["strike"] - spot)
                atm = calls.nsmallest(3, "distance")

                if "impliedVolatility" in atm.columns:
                    avg_iv = float(atm["impliedVolatility"].mean())
                    if avg_iv > 0:
                        days_to_exp = (pd.Timestamp(exp) - pd.Timestamp.now()).days
                        term_data.append({
                            "expiration": exp,
                            "days_to_exp": max(days_to_exp, 1),
                            "atm_iv": round(avg_iv, 4),
                            "atm_iv_annualized": round(avg_iv * 100, 2),
                        })
            except Exception:
                continue

        if not term_data:
            return {"term_structure": [], "shape": "unknown"}

        # Classify shape
        if len(term_data) >= 3:
            short_iv = term_data[0]["atm_iv"]
            long_iv = term_data[-1]["atm_iv"]
            if long_iv > short_iv * 1.05:
                shape = "contango"  # Normal — longer term = higher IV
            elif short_iv > long_iv * 1.05:
                shape = "backwardation"  # Inverted — near term fear
            else:
                shape = "flat"
        else:
            shape = "insufficient_data"

        return {
            "ticker": ticker,
            "spot": spot,
            "term_structure": term_data,
            "shape": shape,
            "interpretation": {
                "contango": "Normal IV curve — no near-term event expected",
                "backwardation": "Inverted IV curve — market expects near-term volatility event",
                "flat": "Flat IV curve — uniform uncertainty across horizons",
            }.get(shape, ""),
        }
    except Exception as e:
        log.warning(f"IV term structure failed for {ticker}: {e}")
        return {"term_structure": [], "shape": "error"}


def build_vol_smile(ticker: str, expiration: str | None = None) -> dict:
    """
    Build volatility smile/skew for a specific expiration.
    Shows how IV changes across strike prices.
    Steep skew = market pricing in tail risk.
    """
    import yfinance as yf

    try:
        stock = yf.Ticker(ticker)
        if expiration is None:
            exps = stock.options
            if not exps:
                return {"smile": [], "skew_metric": None}
            expiration = exps[0]

        chain = stock.option_chain(expiration)
        hist = stock.history(period="1d")
        spot = float(hist["Close"].iloc[-1]) if not hist.empty else 0

        smile_data = []
        for df, opt_type in [(chain.calls, "call"), (chain.puts, "put")]:
            if df.empty:
                continue
            for _, row in df.iterrows():
                strike = float(row.get("strike", 0))
                iv = float(row.get("impliedVolatility", 0))
                if strike > 0 and iv > 0:
                    moneyness = strike / spot if spot > 0 else 0
                    smile_data.append({
                        "strike": strike,
                        "moneyness": round(moneyness, 3),
                        "iv": round(iv, 4),
                        "type": opt_type,
                        "volume": int(row.get("volume", 0)) if pd.notna(row.get("volume")) else 0,
                    })

        if not smile_data:
            return {"smile": [], "skew_metric": None}

        # Compute skew: 25-delta put IV minus 25-delta call IV
        puts_data = [s for s in smile_data if s["type"] == "put" and 0.85 < s["moneyness"] < 0.95]
        calls_data = [s for s in smile_data if s["type"] == "call" and 1.05 < s["moneyness"] < 1.15]

        skew_metric = None
        if puts_data and calls_data:
            otm_put_iv = np.mean([p["iv"] for p in puts_data])
            otm_call_iv = np.mean([c["iv"] for c in calls_data])
            skew_metric = round(float(otm_put_iv - otm_call_iv), 4)

        return {
            "ticker": ticker,
            "expiration": expiration,
            "spot": spot,
            "smile": sorted(smile_data, key=lambda x: x["strike"]),
            "skew_metric": skew_metric,
            "skew_signal": (
                "heavy_put_skew" if skew_metric and skew_metric > 0.1
                else ("call_skew" if skew_metric and skew_metric < -0.05
                      else "normal" if skew_metric else "unknown")
            ),
        }
    except Exception as e:
        log.warning(f"Vol smile failed for {ticker}: {e}")
        return {"smile": [], "skew_metric": None}


def compute_vol_risk_premium(ticker: str, lookback_days: int = 60) -> dict:
    """
    Compute the Volatility Risk Premium (VRP).
    VRP = Implied Volatility - Realized Volatility.
    Positive VRP = options are expensive (IV > realized). Historically,
    VRP is positive ~85% of the time, meaning options are usually overpriced.
    When VRP is negative, it's a major risk signal.
    """
    import yfinance as yf

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{lookback_days + 30}d")
        if hist.empty or len(hist) < 20:
            return {"vrp": None, "signal": "insufficient_data"}

        # Realized vol (annualized)
        returns = hist["Close"].pct_change().dropna()
        realized_vol_20d = float(returns.tail(20).std() * np.sqrt(252))
        realized_vol_60d = float(returns.tail(min(60, len(returns))).std() * np.sqrt(252))

        # Current implied vol (from nearest ATM option)
        exps = stock.options
        if not exps:
            return {
                "realized_vol_20d": round(realized_vol_20d, 4),
                "realized_vol_60d": round(realized_vol_60d, 4),
                "implied_vol": None,
                "vrp": None,
                "signal": "no_options_data",
            }

        chain = stock.option_chain(exps[0])
        spot = float(hist["Close"].iloc[-1])

        calls = chain.calls
        if calls.empty:
            return {
                "realized_vol_20d": round(realized_vol_20d, 4),
                "implied_vol": None,
                "vrp": None,
            }

        calls = calls.copy()
        calls["distance"] = abs(calls["strike"] - spot)
        atm = calls.nsmallest(3, "distance")
        implied_vol = float(atm["impliedVolatility"].mean()) if "impliedVolatility" in atm.columns else None

        if implied_vol is None:
            return {
                "realized_vol_20d": round(realized_vol_20d, 4),
                "implied_vol": None,
                "vrp": None,
            }

        vrp = implied_vol - realized_vol_20d

        # Signal
        if vrp > 0.1:
            signal = "options_very_expensive"
        elif vrp > 0.03:
            signal = "options_expensive"
        elif vrp > -0.03:
            signal = "fair_value"
        elif vrp > -0.1:
            signal = "options_cheap"
        else:
            signal = "options_very_cheap_danger"

        return {
            "ticker": ticker,
            "realized_vol_20d": round(realized_vol_20d, 4),
            "realized_vol_60d": round(realized_vol_60d, 4),
            "implied_vol": round(implied_vol, 4),
            "vrp": round(vrp, 4),
            "vrp_pct": round(vrp * 100, 2),
            "signal": signal,
            "interpretation": {
                "options_very_expensive": "IV >> Realized. Options overpriced. Sell vol or expect mean reversion.",
                "options_expensive": "IV > Realized. Mild premium. Normal market conditions.",
                "fair_value": "IV ≈ Realized. Options fairly priced.",
                "options_cheap": "IV < Realized. Options cheap. Possible catalyst ahead.",
                "options_very_cheap_danger": "IV << Realized. Major risk signal. Market underpricing volatility.",
            }.get(signal, ""),
        }
    except Exception as e:
        log.warning(f"VRP calc failed for {ticker}: {e}")
        return {"vrp": None, "signal": "error"}


def get_vol_intelligence(ticker: str) -> dict:
    """
    Full volatility intelligence report.
    Combines term structure, smile, and risk premium.
    """
    term_structure = build_iv_term_structure(ticker)
    smile = build_vol_smile(ticker)
    vrp = compute_vol_risk_premium(ticker)

    # Composite vol signal
    signals = []

    if term_structure.get("shape") == "backwardation":
        signals.append({"factor": "term_structure", "signal": "bearish", "weight": -1})
    elif term_structure.get("shape") == "contango":
        signals.append({"factor": "term_structure", "signal": "normal", "weight": 0})

    if smile.get("skew_signal") == "heavy_put_skew":
        signals.append({"factor": "skew", "signal": "bearish_fear", "weight": -1})
    elif smile.get("skew_signal") == "call_skew":
        signals.append({"factor": "skew", "signal": "bullish", "weight": 1})

    vrp_val = vrp.get("vrp")
    if vrp_val is not None:
        if vrp_val < -0.03:
            signals.append({"factor": "vrp", "signal": "danger", "weight": -2})
        elif vrp_val > 0.05:
            signals.append({"factor": "vrp", "signal": "overpriced", "weight": 0.5})

    composite = round(np.mean([s["weight"] for s in signals]), 3) if signals else 0

    return {
        "ticker": ticker,
        "term_structure": term_structure,
        "vol_smile": smile,
        "vol_risk_premium": vrp,
        "signals": signals,
        "composite_vol_signal": composite,
        "vol_regime": (
            "high_fear" if composite < -0.5
            else "elevated" if composite < 0
            else "complacent" if composite > 0.3
            else "normal"
        ),
    }

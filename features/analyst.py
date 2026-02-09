"""
SENTINEL — Market Analyst Engine v2.
Reads ALL signals across the system and produces a comprehensive daily intelligence briefing.
Scores EVERY sector, EVERY asset class, and crypto individually.
Tells you exactly what is happening, why, and what to do about it.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional

from config.assets import SECTORS, BENCHMARKS, BOND_TICKERS, COMMODITY_TICKERS, DOLLAR_TICKER
from data.stocks import fetch_ohlcv
from data.crypto import get_current_prices as get_crypto_prices
from data.news import fetch_crypto_news
from features.signals import compute_composite_score
from features.sentiment import score_text, extract_entities, classify_urgency
from features.intermarket import compute_intermarket_signals
from features.breadth import compute_sector_breadth, compute_breadth_divergence
from utils.logger import get_logger

logger = get_logger(__name__)


def _get_top_movers():
    """
    Pull today's % change for ALL stocks in the universe + crypto.
    Returns dict with 'high_flyers' (top 5 up) and 'sinking_ships' (top 5 down).
    """
    import yfinance as yf

    # Collect all individual stock tickers from sector holdings
    all_tickers = set()
    for sector_name, info in SECTORS.items():
        all_tickers.update(info["holdings"])
    # Add benchmarks and asset class ETFs
    for t in BENCHMARKS.values():
        all_tickers.add(t)

    all_tickers = sorted(all_tickers)

    movers = []

    # Batch download — yfinance handles this efficiently
    try:
        data = yf.download(all_tickers, period="5d", group_by="ticker", progress=False, threads=True)
        for ticker in all_tickers:
            try:
                if len(all_tickers) == 1:
                    tdf = data
                else:
                    tdf = data[ticker] if ticker in data.columns.get_level_values(0) else None

                if tdf is None or tdf.empty:
                    continue

                closes = tdf["Close"].dropna()
                if len(closes) < 2:
                    continue

                current = float(closes.iloc[-1])
                prev = float(closes.iloc[-2])
                if prev == 0:
                    continue
                day_pct = (current - prev) / prev * 100

                # Figure out which sector this stock belongs to
                from config.assets import get_sector_for_ticker
                sector = get_sector_for_ticker(ticker) or ""

                movers.append({
                    "name": ticker,
                    "type": "Stock",
                    "sector": sector,
                    "price": current,
                    "day_change": round(day_pct, 2),
                })
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"Batch download failed: {e}")

    # Add crypto
    try:
        crypto_raw = get_crypto_prices()
        if crypto_raw:
            from config.assets import CRYPTO_ALL
            for coin_id, info in CRYPTO_ALL.items():
                cdata = crypto_raw.get(coin_id, {})
                if cdata:
                    price = cdata.get("usd", 0)
                    chg = cdata.get("usd_24h_change", 0) or 0
                    movers.append({
                        "name": info["symbol"],
                        "type": "Crypto",
                        "sector": "Crypto",
                        "price": price,
                        "day_change": round(chg, 2),
                    })
    except Exception:
        pass

    # Sort
    movers_sorted = sorted(movers, key=lambda x: x["day_change"], reverse=True)
    high_flyers = movers_sorted[:5]
    sinking_ships = sorted(movers, key=lambda x: x["day_change"])[:5]

    return {"high_flyers": high_flyers, "sinking_ships": sinking_ships}


def _score_asset(ticker):
    """Score any single ticker. Returns dict with score, label, signals, price info."""
    df = fetch_ohlcv(ticker)
    if df.empty or len(df) < 50:
        return None

    result = compute_composite_score(df)
    score = result["score"]

    # Get price context
    close = df["Close"]
    current = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else current
    day_chg = (current - prev) / prev * 100

    # 1-week and 1-month returns
    ret_1w = (current / float(close.iloc[-5]) - 1) * 100 if len(close) >= 5 else 0
    ret_1m = (current / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0

    # SMA context
    sma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else current
    sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else current
    above_50 = current > sma_50
    above_200 = current > sma_200

    return {
        "score": score,
        "label": result["label"],
        "signals": result["signals"],
        "price": current,
        "day_change": round(day_chg, 2),
        "ret_1w": round(ret_1w, 2),
        "ret_1m": round(ret_1m, 2),
        "above_50_sma": above_50,
        "above_200_sma": above_200,
    }


def _interpret_score(score, name, result=None):
    """Turn a 0-100 score into plain English using actual market data."""
    if result is None:
        result = {}

    price = result.get("price", 0)
    day_chg = result.get("day_change", 0)
    ret_1w = result.get("ret_1w", 0)
    ret_1m = result.get("ret_1m", 0)
    above_50 = result.get("above_50_sma", None)
    above_200 = result.get("above_200_sma", None)
    signals = result.get("signals", {})

    # Build SMA context
    sma_parts = []
    if above_200 is True:
        sma_parts.append("trading above its 200-day moving average")
    elif above_200 is False:
        sma_parts.append("below its 200-day moving average")
    if above_50 is True:
        sma_parts.append("above the 50-day")
    elif above_50 is False:
        sma_parts.append("below the 50-day")
    sma_text = " and ".join(sma_parts) if sma_parts else ""

    # Build signal detail — pick the 2 most notable signals
    sig_text = ""
    if isinstance(signals, list) and signals:
        top_sigs = signals[:2]
        sig_text = " Signals: " + "; ".join(top_sigs) + "."
    elif isinstance(signals, dict):
        sig_details = []
        for sig_name, sig_val in signals.items():
            if isinstance(sig_val, dict):
                direction = sig_val.get("direction", "")
                if direction in ("BULLISH", "BEARISH"):
                    sig_details.append((sig_name.replace("_", " ").title(), direction.lower()))
        if sig_details:
            sig_text = " Signals: " + ", ".join([f"{n} is {d}" for n, d in sig_details[:2]]) + "."

    # Day direction word
    if day_chg > 1:
        day_word = f"up {day_chg:+.1f}% today"
    elif day_chg > 0:
        day_word = f"slightly green at {day_chg:+.1f}% today"
    elif day_chg > -1:
        day_word = f"slightly red at {day_chg:+.1f}% today"
    else:
        day_word = f"down {day_chg:+.1f}% today"

    # 1-month direction
    if ret_1m > 5:
        month_word = f"up {ret_1m:+.1f}% over the last month"
    elif ret_1m > 0:
        month_word = f"gained {ret_1m:+.1f}% this past month"
    elif ret_1m > -5:
        month_word = f"lost {ret_1m:+.1f}% this past month"
    else:
        month_word = f"dropped {ret_1m:+.1f}% over the last month"

    if score >= 75:
        return (
            f"{name} scores {score:.0f}/100 — one of the strongest areas right now. "
            f"It is {day_word}, {month_word}, and {sma_text if sma_text else 'trending well'}. "
            f"Buyers are in control and momentum is strong.{sig_text}"
        )
    elif score >= 60:
        return (
            f"{name} scores {score:.0f}/100 — in solid shape. "
            f"It is {day_word} and {month_word}. "
            f"{'Price is ' + sma_text + '.' if sma_text else ''} "
            f"Not the hottest area, but the trend is working.{sig_text}"
        )
    elif score >= 50:
        return (
            f"{name} scores {score:.0f}/100 — right in the middle. "
            f"It is {day_word}, {month_word}. "
            f"{'Currently ' + sma_text + '.' if sma_text else ''} "
            f"Could go either way from here — no strong edge.{sig_text}"
        )
    elif score >= 40:
        return (
            f"{name} scores {score:.0f}/100 — starting to struggle. "
            f"It is {day_word} and has {month_word}. "
            f"{'Price is ' + sma_text + ' — not ideal.' if sma_text else ''}"
            f"{sig_text} Momentum is fading."
        )
    elif score >= 30:
        return (
            f"{name} scores {score:.0f}/100 — weak. "
            f"It is {day_word}, {month_word}. "
            f"{'Sitting ' + sma_text + '.' if sma_text else ''} "
            f"Most signals point down.{sig_text} Probably best to avoid."
        )
    else:
        return (
            f"{name} scores only {score:.0f}/100 — one of the weakest spots. "
            f"It is {day_word}, {month_word}. "
            f"{'Well ' + sma_text + '.' if sma_text else ''} "
            f"Sellers are in control with no sign of a turnaround.{sig_text}"
        )


def generate_market_briefing() -> dict:
    """
    Generate a comprehensive daily intelligence briefing covering
    ALL sectors, ALL asset classes, and crypto.
    """
    briefing = {
        "headline": "",
        "takeaways": [],
        "market_scores": {},
        "sector_scores": {},
        "asset_class_scores": {},
        "crypto": {},
        "crypto_outlook": {},
        "sector_rankings": [],
        "regime": "",
        "regime_detail": "",
        "signals_explained": [],
        "risk_level": "",
        "risk_score": 50,
        "risk_factors": [],
        "action_items": [],
        "watch_list": [],
        "bottom_line": "",
        "market_drivers": [],
        "top_movers": {"high_flyers": [], "sinking_ships": []},
        "timestamp": datetime.now(ZoneInfo("America/New_York")).strftime("%I:%M %p ET — %b %d, %Y"),
    }

    # ══════════════════════════════════════════════════════════════
    # STEP 1: Score the major benchmarks
    # ══════════════════════════════════════════════════════════════
    bench_data = {}
    for name, ticker in BENCHMARKS.items():
        result = _score_asset(ticker)
        if result:
            bench_data[name] = result
            briefing["market_scores"][name] = {
                "score": result["score"],
                "label": result["label"],
                "day_change": result["day_change"],
                "ret_1w": result["ret_1w"],
                "ret_1m": result["ret_1m"],
            }

    # ══════════════════════════════════════════════════════════════
    # STEP 2: Score ALL 11 GICS sectors
    # ══════════════════════════════════════════════════════════════
    sector_data = {}
    for sector_name, info in SECTORS.items():
        etf = info["etf"]
        result = _score_asset(etf)
        if result:
            interpretation = _interpret_score(result["score"], sector_name, result)
            sector_data[sector_name] = result
            briefing["sector_scores"][sector_name] = {
                "etf": etf,
                "score": result["score"],
                "label": result["label"],
                "day_change": result["day_change"],
                "ret_1w": result["ret_1w"],
                "ret_1m": result["ret_1m"],
                "above_50_sma": result["above_50_sma"],
                "above_200_sma": result["above_200_sma"],
                "interpretation": interpretation,
            }

    # Rank sectors best to worst
    ranked = sorted(briefing["sector_scores"].items(), key=lambda x: x[1]["score"], reverse=True)
    briefing["sector_rankings"] = [
        {"rank": i + 1, "sector": name, "etf": data["etf"], "score": data["score"],
         "label": data["label"], "ret_1m": data["ret_1m"]}
        for i, (name, data) in enumerate(ranked)
    ]

    # ══════════════════════════════════════════════════════════════
    # STEP 3: Score bonds, gold, oil, dollar
    # ══════════════════════════════════════════════════════════════
    asset_map = {
        "Long Bonds (TLT)": "TLT",
        "High Yield (HYG)": "HYG",
        "Inv Grade (LQD)": "LQD",
        "Gold (GLD)": "GLD",
        "Oil (USO)": "USO",
        "Dollar (UUP)": "UUP",
    }
    for label, ticker in asset_map.items():
        result = _score_asset(ticker)
        if result:
            briefing["asset_class_scores"][label] = {
                "ticker": ticker,
                "score": result["score"],
                "label": result["label"],
                "price": result["price"],
                "day_change": result["day_change"],
                "ret_1w": result["ret_1w"],
                "ret_1m": result["ret_1m"],
                "interpretation": _interpret_score(result["score"], label.split("(")[0].strip(), result),
            }

    # ══════════════════════════════════════════════════════════════
    # STEP 4: Get crypto data
    # ══════════════════════════════════════════════════════════════
    try:
        crypto_raw = get_crypto_prices()
        if crypto_raw:
            coin_map = {"bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL"}
            crypto_data = {}
            for coin_id, symbol in coin_map.items():
                cinfo = crypto_raw.get(coin_id, {})
                if cinfo:
                    change_24h = cinfo.get("usd_24h_change", 0) or 0
                    price = cinfo.get("usd", 0)

                    # Crypto interpretation — plain English
                    if change_24h > 5:
                        crypto_signal = "SURGING"
                        crypto_desc = f"{symbol} is up {change_24h:+.1f}% in the last 24 hours — a big move. When crypto jumps like this, it usually means investors are feeling very confident and willing to take risks."
                    elif change_24h > 2:
                        crypto_signal = "BULLISH"
                        crypto_desc = f"{symbol} is up {change_24h:+.1f}% in 24 hours. A solid day. Buyers are in control and the mood is positive."
                    elif change_24h > -2:
                        crypto_signal = "QUIET"
                        crypto_desc = f"{symbol} barely moved ({change_24h:+.1f}% in 24h). Nothing exciting happening right now — the market is taking a breather."
                    elif change_24h > -5:
                        crypto_signal = "WEAK"
                        crypto_desc = f"{symbol} is down {change_24h:+.1f}% in 24 hours. Sellers are winning today. When crypto drops, it can sometimes mean the stock market will feel pressure next."
                    else:
                        crypto_signal = "SELLING"
                        crypto_desc = f"{symbol} is falling hard — down {change_24h:+.1f}% in 24 hours. This kind of drop usually means investors are scared and pulling money out of risky investments."

                    crypto_data[symbol] = {
                        "price": price,
                        "change_24h": change_24h,
                        "change_7d": 0,
                        "market_cap": cinfo.get("usd_market_cap", 0) or 0,
                        "signal": crypto_signal,
                        "description": crypto_desc,
                    }
            briefing["crypto"] = crypto_data
    except Exception:
        pass

    # ══════════════════════════════════════════════════════════════
    # STEP 4B: Crypto Daily Outlook (news-driven)
    # ══════════════════════════════════════════════════════════════
    briefing["crypto_outlook"] = _build_crypto_outlook(briefing.get("crypto", {}))

    # ══════════════════════════════════════════════════════════════
    # STEP 5: Intermarket signals
    # ══════════════════════════════════════════════════════════════
    inter = compute_intermarket_signals()
    inter_signals = inter.get("signals", [])
    inter_divergences = inter.get("divergences", [])
    overall_inter = inter.get("overall", "MIXED")
    returns_1w = inter.get("returns_1w", {})
    returns_1m = inter.get("returns_1m", {})
    correlations = inter.get("correlations", pd.DataFrame())

    # ══════════════════════════════════════════════════════════════
    # STEP 6: Breadth data
    # ══════════════════════════════════════════════════════════════
    breadth = compute_sector_breadth()
    breadth_summary = breadth.get("summary", {})
    health_score = breadth_summary.get("health_score", 50)

    # ══════════════════════════════════════════════════════════════
    # STEP 7: Market regime
    # ══════════════════════════════════════════════════════════════
    spy_score = bench_data.get("S&P 500", {}).get("score", 50)
    qqq_score = bench_data.get("Nasdaq 100", {}).get("score", 50)

    regime = _determine_regime(spy_score, health_score, overall_inter, inter_signals)
    briefing["regime"] = regime["name"]
    briefing["regime_detail"] = regime["detail"]

    # ══════════════════════════════════════════════════════════════
    # STEP 8: Headline
    # ══════════════════════════════════════════════════════════════
    briefing["headline"] = _build_headline(spy_score, qqq_score, health_score, regime, overall_inter, sector_data, briefing.get("crypto", {}))

    # ══════════════════════════════════════════════════════════════
    # STEP 8B: Market Drivers — WHAT is pushing the market right now
    # ══════════════════════════════════════════════════════════════
    briefing["market_drivers"] = _build_market_drivers(
        bench_data, sector_data, inter_signals, inter_divergences,
        breadth_summary, returns_1w, returns_1m, overall_inter,
        briefing.get("asset_class_scores", {}), briefing.get("crypto", {}),
    )

    # ══════════════════════════════════════════════════════════════
    # STEP 9: Takeaways (now much richer)
    # ══════════════════════════════════════════════════════════════
    briefing["takeaways"] = _build_takeaways(
        bench_data, sector_data, inter_signals, inter_divergences,
        breadth_summary, regime, overall_inter, returns_1w, returns_1m,
        briefing.get("crypto", {})
    )

    # ══════════════════════════════════════════════════════════════
    # STEP 10: Signal explanations
    # ══════════════════════════════════════════════════════════════
    briefing["signals_explained"] = _explain_signals(
        bench_data, inter_signals, inter_divergences,
        breadth_summary, correlations, returns_1w, returns_1m
    )

    # ══════════════════════════════════════════════════════════════
    # STEP 11: Risk assessment
    # ══════════════════════════════════════════════════════════════
    risk = _assess_risk(
        spy_score, health_score, inter_signals,
        inter_divergences, breadth_summary, returns_1m
    )
    briefing["risk_level"] = risk["level"]
    briefing["risk_score"] = risk["score"]
    briefing["risk_factors"] = risk["factors"]

    # ══════════════════════════════════════════════════════════════
    # STEP 12: Action items
    # ══════════════════════════════════════════════════════════════
    briefing["action_items"] = _build_actions(
        spy_score, health_score, regime, inter_signals,
        breadth_summary, bench_data, sector_data, briefing.get("crypto", {})
    )

    # ══════════════════════════════════════════════════════════════
    # STEP 13: Watch list
    # ══════════════════════════════════════════════════════════════
    briefing["watch_list"] = _build_watchlist(
        inter_signals, inter_divergences, breadth_summary, returns_1m,
        sector_data, briefing.get("crypto", {})
    )

    # ══════════════════════════════════════════════════════════════
    # STEP 14: Top movers — high flyers and sinking ships
    # ══════════════════════════════════════════════════════════════
    try:
        briefing["top_movers"] = _get_top_movers()
    except Exception as e:
        logger.warning(f"Top movers failed: {e}")

    # ══════════════════════════════════════════════════════════════
    # STEP 15: Bottom line
    # ══════════════════════════════════════════════════════════════
    briefing["bottom_line"] = _build_bottom_line(
        spy_score, health_score, regime, overall_inter, sector_data, briefing.get("crypto", {})
    )

    return briefing


# ── Helpers ──────────────────────────────────────────────────────

def _determine_regime(spy_score, health_score, overall_inter, inter_signals):
    """Classify market regime from all available signals."""
    bullish_count = 0
    bearish_count = 0

    if spy_score >= 60:
        bullish_count += 2
    elif spy_score <= 40:
        bearish_count += 2

    if health_score >= 60:
        bullish_count += 1
    elif health_score <= 40:
        bearish_count += 1

    if "BULLISH" in overall_inter:
        bullish_count += 1
    elif "BEARISH" in overall_inter:
        bearish_count += 1

    has_credit_stress = any("CREDIT" in s.get("signal", "").upper() for s in inter_signals)
    has_fear_trade = any("FEAR" in s.get("detail", "").upper() for s in inter_signals)

    if has_credit_stress:
        bearish_count += 2

    if bullish_count >= 4:
        return {
            "name": "STRONG AND HEALTHY",
            "detail": "Almost everything is pointing up. Stocks are rising, most sectors look good, and investors are confident. This is the kind of market where most people make money.",
        }
    elif bullish_count >= 2 and bearish_count <= 1:
        return {
            "name": "LEANING POSITIVE",
            "detail": "More things look good than bad, but it is not a slam dunk. The market is in decent shape — just keep an eye on things.",
        }
    elif bearish_count >= 4:
        return {
            "name": "WARNING — BE CAREFUL",
            "detail": "A lot of warning signs are flashing at the same time. Stocks, bonds, and other markets are all saying the same thing: something is not right. This is a time to be very cautious.",
        }
    elif bearish_count >= 2 and bullish_count <= 1:
        return {
            "name": "GETTING WORSE",
            "detail": "Things are sliding in the wrong direction. Not a full panic, but the trend is turning negative. Consider playing it safe until things stabilize.",
        }
    elif has_fear_trade:
        return {
            "name": "INVESTORS ARE NERVOUS",
            "detail": "Money is flowing into gold and other safe investments while riskier stuff struggles. When this happens, it usually means bigger trouble could be coming.",
        }
    else:
        return {
            "name": "MIXED SIGNALS",
            "detail": "Some things look good, some look bad. The market has not made up its mind yet. Best to wait and see before making big moves.",
        }


def _build_headline(spy_score, qqq_score, health_score, regime, overall_inter, sector_data, crypto_data=None):
    """
    Generate a plain English headline covering stocks, sectors, AND crypto.
    Specific numbers, what it means, and what to expect next.
    """
    # Sector context
    if sector_data:
        sorted_sectors = sorted(sector_data.items(), key=lambda x: x[1].get("score", 50), reverse=True)
        top = sorted_sectors[0]
        top2 = sorted_sectors[1] if len(sorted_sectors) > 1 else top
        bottom = sorted_sectors[-1]
        bottom2 = sorted_sectors[-2] if len(sorted_sectors) > 1 else bottom
        top_name = top[0]
        top_score = top[1].get("score", 50)
        top_1m = top[1].get("ret_1m", 0)
        bot_name = bottom[0]
        bot_score = bottom[1].get("score", 50)
        bot_1m = bottom[1].get("ret_1m", 0)
        above_200 = sum(1 for s in sector_data.values() if s.get("above_200_sma"))
        total_sectors = len(sector_data)

        # Offensive vs Defensive
        offensive = ["Technology", "Consumer Discretionary", "Financials", "Industrials", "Communication Services"]
        defensive = ["Utilities", "Healthcare", "Consumer Staples", "Real Estate"]
        off_scores = [sector_data[s]["score"] for s in offensive if s in sector_data]
        def_scores = [sector_data[s]["score"] for s in defensive if s in sector_data]
        off_avg = np.mean(off_scores) if off_scores else 50
        def_avg = np.mean(def_scores) if def_scores else 50
    else:
        above_200 = 0
        total_sectors = 11
        top_name = "N/A"
        top_score = 50
        bot_name = "N/A"
        bot_score = 50
        off_avg = 50
        def_avg = 50
        top_1m = 0
        bot_1m = 0

    # Crypto context — build a quick snapshot
    crypto_snippet = ""
    if crypto_data:
        btc = crypto_data.get("BTC", {})
        eth = crypto_data.get("ETH", {})
        sol = crypto_data.get("SOL", {})
        btc_chg = btc.get("change_24h", 0) if btc else 0
        eth_chg = eth.get("change_24h", 0) if eth else 0
        sol_chg = sol.get("change_24h", 0) if sol else 0
        btc_price = btc.get("price", 0) if btc else 0

        # Format BTC price
        btc_price_str = f"${btc_price:,.0f}" if btc_price >= 1000 else f"${btc_price:,.2f}"

        # Overall crypto mood
        avg_crypto_chg = np.mean([c for c in [btc_chg, eth_chg, sol_chg] if c != 0]) if any([btc_chg, eth_chg, sol_chg]) else 0

        if avg_crypto_chg > 5:
            crypto_snippet = f" Crypto is surging — BTC {btc_price_str} ({btc_chg:+.1f}%), ETH {eth_chg:+.1f}%, SOL {sol_chg:+.1f}%. Big risk appetite across the board."
        elif avg_crypto_chg > 2:
            crypto_snippet = f" Crypto having a good day — BTC {btc_price_str} ({btc_chg:+.1f}%), ETH {eth_chg:+.1f}%, SOL {sol_chg:+.1f}%. Buyers are showing up."
        elif avg_crypto_chg > 0:
            crypto_snippet = f" Crypto slightly green — BTC {btc_price_str} ({btc_chg:+.1f}%), ETH {eth_chg:+.1f}%, SOL {sol_chg:+.1f}%."
        elif avg_crypto_chg > -2:
            crypto_snippet = f" Crypto drifting lower — BTC {btc_price_str} ({btc_chg:+.1f}%), ETH {eth_chg:+.1f}%, SOL {sol_chg:+.1f}%."
        elif avg_crypto_chg > -5:
            crypto_snippet = f" Crypto under pressure — BTC {btc_price_str} ({btc_chg:+.1f}%), ETH {eth_chg:+.1f}%, SOL {sol_chg:+.1f}%. Risk appetite is fading."
        else:
            crypto_snippet = f" Crypto getting hit hard — BTC {btc_price_str} ({btc_chg:+.1f}%), ETH {eth_chg:+.1f}%, SOL {sol_chg:+.1f}%. Investors are running from risk."

    regime_name = regime["name"]

    # Build a headline anyone can understand — real numbers, real meaning
    if regime_name == "STRONG AND HEALTHY":
        return (
            f"The Market Looks Strong Right Now. {above_200} out of {total_sectors} sectors are in an uptrend. "
            f"Overall market health: {spy_score:.0f}/100. Breadth: {health_score:.0f}/100. "
            f"{top_name} ({top_score:.0f}) is the strongest area. "
            f"Most things are going up — this is a good environment to be invested.{crypto_snippet}"
        )
    elif regime_name == "WARNING — BE CAREFUL":
        return (
            f"Warning Signs Everywhere. Only {above_200} out of {total_sectors} sectors are still in an uptrend. "
            f"Market health dropped to {spy_score:.0f}/100. Breadth is weak at {health_score:.0f}/100. "
            f"{bot_name} ({bot_score:.0f}) is getting hit the hardest. "
            f"This is a time to be very careful — more things are going down than up.{crypto_snippet}"
        )
    elif regime_name == "INVESTORS ARE NERVOUS":
        return (
            f"Investors Are Getting Nervous. Safe sectors ({def_avg:.0f}) are outperforming risky ones ({off_avg:.0f}). "
            f"Market health: {spy_score:.0f}/100. Breadth: {health_score:.0f}/100. "
            f"When money moves to gold and safe investments, it usually means trouble is brewing.{crypto_snippet}"
        )
    elif regime_name == "GETTING WORSE":
        return (
            f"Things Are Getting Worse. Market health slipping to {spy_score:.0f}/100, breadth at {health_score:.0f}/100. "
            f"{above_200} out of {total_sectors} sectors still in uptrend but that number is shrinking. "
            f"{bot_name} is the weakest ({bot_1m:+.1f}% this month). "
            f"The trend is moving in the wrong direction.{crypto_snippet}"
        )
    elif spy_score >= 65 and health_score >= 60:
        return (
            f"The Market Is Healthy. Score: {spy_score:.0f}/100. Breadth: {health_score:.0f}/100. "
            f"{above_200} out of {total_sectors} sectors are trending up. "
            f"{top_name} ({top_score:.0f}) is leading with {top_1m:+.1f}% this month. "
            f"Growth sectors ({off_avg:.0f}) beating safe sectors ({def_avg:.0f}) — investors feel confident.{crypto_snippet}"
        )
    elif spy_score >= 55 and health_score >= 50:
        return (
            f"The Market Is OK But Not Great. Score: {spy_score:.0f}/100. Breadth: {health_score:.0f}/100. "
            f"{above_200} out of {total_sectors} sectors in uptrend. "
            f"Best sector: {top_name} ({top_score:.0f}). Worst: {bot_name} ({bot_score:.0f}). "
            f"Things are moving up slowly. Not exciting, but not scary either.{crypto_snippet}"
        )
    elif spy_score <= 40 and health_score <= 40:
        return (
            f"The Market Is Struggling. Score: {spy_score:.0f}/100. Breadth: {health_score:.0f}/100. "
            f"Only {above_200} out of {total_sectors} sectors are still going up. "
            f"{bot_name} down {bot_1m:+.1f}% this month. "
            f"Most stocks are falling. Not a great time to be putting new money in.{crypto_snippet}"
        )
    elif spy_score <= 45:
        return (
            f"The Market Is Weakening. Score only {spy_score:.0f}/100. "
            f"Breadth: {health_score:.0f}/100. {above_200} out of {total_sectors} sectors holding their uptrend. "
            f"{bot_name} ({bot_score:.0f}) is the weakest link. "
            f"Not a crisis, but momentum is fading.{crypto_snippet}"
        )
    elif off_avg > def_avg + 12:
        return (
            f"Money Is Moving Into Growth. Growth sectors ({off_avg:.0f}) are beating safe sectors ({def_avg:.0f}). "
            f"Market score: {spy_score:.0f}/100. {above_200} out of {total_sectors} sectors uptrend. "
            f"{top_name} ({top_score:.0f}) is where the action is. "
            f"Investors feel good about where things are headed.{crypto_snippet}"
        )
    elif def_avg > off_avg + 12:
        return (
            f"Money Is Moving To Safety. Safe sectors ({def_avg:.0f}) beating growth ({off_avg:.0f}). "
            f"Market score: {spy_score:.0f}/100. Breadth: {health_score:.0f}/100. "
            f"Investors are quietly shifting to safer investments like {top_name}. "
            f"This usually happens when people think the economy might slow down.{crypto_snippet}"
        )
    else:
        spread = top_score - bot_score
        return (
            f"Mixed Picture Today. Market score: {spy_score:.0f}/100. Breadth: {health_score:.0f}/100. "
            f"{above_200} out of {total_sectors} sectors in uptrend. "
            f"Best: {top_name} ({top_score:.0f}). Worst: {bot_name} ({bot_score:.0f}) — {spread:.0f}-point gap between them. "
            f"Some areas doing well, others not.{crypto_snippet}"
        )


def _build_takeaways(bench_data, sector_data, inter_signals, divergences,
                     breadth, regime, overall_inter, ret_1w, ret_1m, crypto_data):
    """5-7 key takeaways covering market + sectors + crypto."""
    takeaways = []

    # 1. Market direction
    spy = bench_data.get("S&P 500", {})
    if spy:
        score = spy.get("score", 50)
        spy_day = spy.get("day_change", 0)
        spy_1w = spy.get("ret_1w", 0)
        spy_1m = spy.get("ret_1m", 0)
        if score >= 60:
            takeaways.append(f"The S&P 500 scores {score:.0f}/100 — healthy. It is {spy_day:+.1f}% today, {spy_1w:+.1f}% this week, and {spy_1m:+.1f}% over the past month. The trend is working.")
        elif score >= 50:
            takeaways.append(f"The S&P 500 scores {score:.0f}/100 — slightly positive. Today it is {spy_day:+.1f}%, this week {spy_1w:+.1f}%. Not exciting but not scary.")
        elif score >= 40:
            takeaways.append(f"The S&P 500 scores {score:.0f}/100 — below average. It is {spy_day:+.1f}% today and {spy_1w:+.1f}% this week. No clear direction.")
        else:
            takeaways.append(f"The S&P 500 scores {score:.0f}/100 — weak. It is {spy_day:+.1f}% today, {spy_1w:+.1f}% this week, {spy_1m:+.1f}% this month. More stocks are going down than up.")

    # 2. Sector leadership/rotation
    if sector_data:
        sorted_sectors = sorted(sector_data.items(), key=lambda x: x[1].get("score", 50), reverse=True)
        top3 = sorted_sectors[:3]
        bot3 = sorted_sectors[-3:]
        top_names = ", ".join([f"{s[0]} ({s[1]['score']:.0f})" for s in top3])
        bot_names = ", ".join([f"{s[0]} ({s[1]['score']:.0f})" for s in bot3])
        takeaways.append(f"Strongest sectors right now: {top_names}. Weakest: {bot_names}. Where money flows tells you what investors believe is coming next.")

        # Offensive vs defensive check
        offensive = ["Technology", "Consumer Discretionary", "Financials", "Industrials", "Communication Services"]
        defensive = ["Utilities", "Healthcare", "Consumer Staples", "Real Estate"]
        off_avg = np.mean([sector_data[s]["score"] for s in offensive if s in sector_data]) if sector_data else 50
        def_avg = np.mean([sector_data[s]["score"] for s in defensive if s in sector_data]) if sector_data else 50
        if off_avg > def_avg + 10:
            takeaways.append(f"Growth sectors ({off_avg:.0f}) are beating safe sectors ({def_avg:.0f}). That means investors feel confident about the economy — they are taking more risk, which is a positive sign.")
        elif def_avg > off_avg + 10:
            takeaways.append(f"Safe sectors ({def_avg:.0f}) are beating growth sectors ({off_avg:.0f}). That means investors are getting nervous — they are moving money to safer places, which is a warning sign.")
        else:
            takeaways.append(f"Growth sectors ({off_avg:.0f}) and safe sectors ({def_avg:.0f}) are about equal. The market has not made up its mind yet about which direction things are heading.")

    # 3. Breadth — how many stocks are actually going up
    health = breadth.get("health_score", 50)
    pct_50ma = breadth.get("pct_above_50ma", 50)
    if health >= 60:
        takeaways.append(f"Market breadth scores {health:.0f}/100 — that is healthy. About {pct_50ma:.0f}% of stocks are in an uptrend. When lots of stocks go up together (not just a few big names), the rally is more trustworthy.")
    elif health <= 40:
        takeaways.append(f"Market breadth is weak at {health:.0f}/100. Only {pct_50ma:.0f}% of stocks are in an uptrend. Even if the main indexes look OK, most individual stocks are actually struggling. That is a red flag.")
    else:
        takeaways.append(f"Market breadth is average at {health:.0f}/100. About {pct_50ma:.0f}% of stocks are in an uptrend. Not great, not terrible — the market is in a wait-and-see zone.")

    # 4. Other markets (bonds, gold, dollar) — what they are saying
    bullish_sigs = [s for s in inter_signals if s.get("direction") == "BULLISH"]
    bearish_sigs = [s for s in inter_signals if s.get("direction") == "BEARISH"]
    if len(bearish_sigs) > len(bullish_sigs):
        takeaways.append(f"Outside of stocks, {len(bearish_sigs)} out of {len(bearish_sigs) + len(bullish_sigs)} signals are negative. Bonds, gold, and other markets are painting a more cautious picture than stocks alone. When these markets disagree with stocks, pay attention.")
    elif len(bullish_sigs) > len(bearish_sigs):
        takeaways.append(f"Outside of stocks, {len(bullish_sigs)} out of {len(bearish_sigs) + len(bullish_sigs)} signals are positive. Bonds, gold, and other markets all agree that things look decent. When everything lines up like this, the signal is stronger.")

    # 5. Crypto — what Bitcoin and friends are doing
    btc = crypto_data.get("BTC", {})
    if btc:
        btc_chg = btc.get("change_24h", 0)
        takeaways.append(f"Bitcoin is {btc_chg:+.1f}% in the last 24 hours. {btc.get('description', '')} Crypto tends to move before stocks — big crypto moves can give you a heads-up about what is coming next.")

    # 6. Divergences — when things do not add up
    if divergences:
        takeaways.append(f"Something does not add up: {divergences[0]}. When the stock market indexes say one thing but the underlying numbers say another, a big move is usually coming. Pay close attention.")

    return takeaways[:7]


def _explain_signals(bench_data, inter_signals, divergences, breadth, correlations, ret_1w, ret_1m):
    """Explain each signal in plain English anyone can understand."""
    explanations = []

    for sig in inter_signals:
        signal_name = sig.get("signal", "")
        detail = sig.get("detail", "")
        direction = sig.get("direction", "NEUTRAL")

        if "BOND" in signal_name.upper() or "STOCK-BOND" in signal_name.upper():
            explanations.append({
                "title": signal_name,
                "what_it_is": "Stocks and bonds usually move in opposite directions — like a seesaw. When they both go the same way at the same time, something unusual is happening.",
                "what_it_means": detail,
                "direction": direction,
                "so_what": _bond_stock_explanation(direction, ret_1m),
            })
        elif "GOLD" in signal_name.upper() or "FEAR" in signal_name.upper():
            explanations.append({
                "title": signal_name,
                "what_it_is": "Gold is what people buy when they are scared. Think of it as the market's panic button. When gold goes up a lot, it means big investors are worried about something.",
                "what_it_means": detail,
                "direction": direction,
                "so_what": "When gold starts leading the way up, it often means stocks will have a harder time over the next few months.",
            })
        elif "CREDIT" in signal_name.upper():
            explanations.append({
                "title": signal_name,
                "what_it_is": "This tracks whether banks and lenders are getting nervous. When risky company bonds start dropping compared to safe ones, it means lenders think some companies might not be able to pay back their debts.",
                "what_it_means": detail,
                "direction": direction,
                "so_what": "This is one of the best early warning signals out there. The lending market usually spots trouble before the stock market does.",
            })
        elif "DOLLAR" in signal_name.upper():
            explanations.append({
                "title": signal_name,
                "what_it_is": "When the US dollar gets stronger, it makes American products more expensive for other countries to buy. Big US companies that sell overseas earn less money when the dollar is strong.",
                "what_it_means": detail,
                "direction": direction,
                "so_what": "A strong dollar puts pressure on big companies like Apple and Microsoft (they sell globally). A weak dollar is good for things like gold, oil, and foreign investments.",
            })
        elif "OIL" in signal_name.upper():
            explanations.append({
                "title": signal_name,
                "what_it_is": "Oil prices affect almost everything — gas prices, shipping costs, food prices, airline tickets. When oil goes up a lot, it is like a hidden tax on everyone.",
                "what_it_means": detail,
                "direction": direction,
                "so_what": "When oil rises sharply, people spend more on gas and less on other things. That is bad for stores, restaurants, and the overall economy.",
            })
        else:
            explanations.append({
                "title": signal_name, "what_it_is": "",
                "what_it_means": detail, "direction": direction, "so_what": "",
            })

    # Breadth signal
    pct_above_50 = breadth.get("pct_above_50ma", 50)
    pct_above_200 = breadth.get("pct_above_200ma", 50)
    if pct_above_50 < 40 and pct_above_200 > 50:
        explanations.append({
            "title": "Short-Term Dip in a Healthy Market",
            "what_it_is": f"Right now {pct_above_50:.0f}% of stocks are in a short-term uptrend, but {pct_above_200:.0f}% are still in a long-term uptrend.",
            "what_it_means": "Stocks are pulling back in the short term, but the bigger picture still looks OK. Think of it like a rainy day during a mostly sunny week.",
            "direction": "MIXED",
            "so_what": "These kinds of dips in an otherwise healthy market have historically been good times to buy — as long as the bigger trend holds.",
        })
    elif pct_above_50 < 40 and pct_above_200 < 40:
        explanations.append({
            "title": "Most Stocks Are Falling",
            "what_it_is": f"Only {pct_above_50:.0f}% of stocks are in a short-term uptrend and only {pct_above_200:.0f}% in a long-term uptrend.",
            "what_it_means": "Most stocks are going down, both in the short term and long term. The market is in rough shape under the surface.",
            "direction": "BEARISH",
            "so_what": "When this few stocks are going up, it is usually best to wait on the sidelines until things improve. There is no rush to invest when the odds are against you.",
        })

    for div in divergences:
        explanations.append({
            "title": "SOMETHING DOES NOT ADD UP",
            "what_it_is": "The big stock indexes say one thing, but when you look at what individual stocks are doing, they tell a different story.",
            "what_it_means": div,
            "direction": "BEARISH",
            "so_what": "When the headline numbers and the real numbers disagree, the real numbers are usually right. A big move is likely coming — pay attention.",
        })

    return explanations


def _bond_stock_explanation(direction, ret_1m):
    tlt_ret = ret_1m.get("TLT", 0)
    spy_ret = ret_1m.get("SPY", 0)
    if tlt_ret > 0 and spy_ret < 0:
        return "Bonds are going up while stocks go down. That means investors are moving their money to safety. People are getting cautious."
    elif tlt_ret < 0 and spy_ret > 0:
        return "Stocks are going up while bonds go down. That is actually normal and healthy — it means investors believe the economy is strong enough to keep growing."
    elif tlt_ret > 0 and spy_ret > 0:
        return "Both stocks AND bonds are going up — which is unusual. This often means investors expect the Federal Reserve to cut interest rates soon."
    elif tlt_ret < 0 and spy_ret < 0:
        return "Both stocks AND bonds are going down — that is a warning sign. When everything falls together, it usually means big investors are selling across the board. Be very careful."
    else:
        return "Stocks and bonds are behaving normally — nothing unusual to report here."


def _assess_risk(spy_score, health_score, inter_signals, divergences, breadth, ret_1m):
    risk_score = 30
    factors = []

    if spy_score <= 40:
        risk_score += 15
        factors.append("The overall market score is below 40 — that means the trend is going against you")
    elif spy_score <= 50:
        risk_score += 5

    if health_score <= 35:
        risk_score += 20
        factors.append(f"Market breadth is only {health_score:.0f}/100 — most stocks are struggling even if the indexes look OK")
    elif health_score <= 45:
        risk_score += 10
        factors.append("Market breadth is below average — only a small number of stocks are actually going up")

    bearish_sigs = [s for s in inter_signals if s.get("direction") == "BEARISH"]
    if len(bearish_sigs) >= 3:
        risk_score += 20
        factors.append(f"{len(bearish_sigs)} different markets (bonds, gold, etc.) are sending warning signs at the same time")
    elif len(bearish_sigs) >= 2:
        risk_score += 10
        factors.append(f"{len(bearish_sigs)} different markets are flashing caution signals")

    credit_stress = any("CREDIT" in s.get("signal", "").upper() and s.get("direction") == "BEARISH" for s in inter_signals)
    if credit_stress:
        risk_score += 15
        factors.append("The lending market is showing stress — this is one of the best early warning systems and it is saying be careful")

    if divergences:
        risk_score += 10
        factors.append("The headline numbers and the real numbers do not match — something under the surface is off")

    tlt_ret = ret_1m.get("TLT", 0)
    spy_ret = ret_1m.get("SPY", 0)
    if tlt_ret < -0.02 and spy_ret < -0.02:
        risk_score += 15
        factors.append("Stocks AND bonds are both falling — when everything drops together it means big investors are selling everything")

    risk_score = min(risk_score, 100)

    if risk_score >= 70:
        level = "HIGH"
    elif risk_score >= 50:
        level = "ELEVATED"
    elif risk_score >= 30:
        level = "MODERATE"
    else:
        level = "LOW"

    return {"level": level, "score": risk_score, "factors": factors}


def _build_actions(spy_score, health_score, regime, inter_signals, breadth, bench_data, sector_data, crypto_data):
    """What to actually think about doing — explained simply."""
    actions = []
    regime_name = regime["name"]

    # Regime-based core actions
    if regime_name == "STRONG AND HEALTHY":
        actions.append("The market is in good shape right now. If you are already invested, this is a good time to stay invested. The numbers support it.")
    elif regime_name == "WARNING — BE CAREFUL":
        actions.append("Multiple warning signs are flashing. This is a time to think about moving some money to safety (like a savings account or bonds) rather than putting more into stocks.")
    elif regime_name == "INVESTORS ARE NERVOUS":
        actions.append("Investors are moving money to safe places like gold. When the smart money gets nervous, it usually pays to listen. Consider being more cautious with new investments.")
    elif regime_name == "GETTING WORSE":
        actions.append("Things are heading in the wrong direction. Consider taking some profits on your weakest investments. Having some cash on the side is not a bad idea right now.")
    else:
        actions.append("The signals are mixed — some good, some bad. When the market cannot make up its mind, the best move is to be patient and not make any big changes.")

    # Sector-specific actions
    if sector_data:
        sorted_sectors = sorted(sector_data.items(), key=lambda x: x[1].get("score", 50), reverse=True)
        top = sorted_sectors[:2]
        bottom = sorted_sectors[-2:]

        top_names = " and ".join([s[0] for s in top])
        bot_names = " and ".join([s[0] for s in bottom])

        actions.append(f"STRONGEST AREAS: {top_names}. These sectors have the best scores right now — if you are looking to invest, this is where the momentum is.")
        actions.append(f"WEAKEST AREAS: {bot_names}. These sectors are struggling. Probably best to stay away from these until they start improving.")

    # Crypto action
    btc = crypto_data.get("BTC", {})
    if btc:
        sig = btc.get("signal", "QUIET")
        if sig in ("SURGING", "BULLISH"):
            actions.append("Bitcoin is looking strong. If you own crypto, things are going in your favor right now. No need to rush to sell.")
        elif sig in ("WEAK", "SELLING"):
            actions.append("Bitcoin is showing weakness. If you own crypto, keep a close eye on it. Think about whether you are comfortable with the risk if it keeps dropping.")

    # Breadth-based
    pct_50 = breadth.get("pct_above_50ma", 50)
    if pct_50 <= 30:
        actions.append(f"Only {pct_50:.0f}% of stocks are in an uptrend — that is very low. Historically, when this number bounces back above 60%, it has been a good time to buy. Watch for that bounce.")
    elif pct_50 >= 80:
        actions.append(f"{pct_50:.0f}% of stocks are going up — which sounds great, but when almost everything is up, the market often takes a breather soon. Not a great time to pile in more money.")

    return actions


def _build_watchlist(inter_signals, divergences, breadth, ret_1m, sector_data, crypto_data):
    """Things to keep an eye on — explained like you are telling a friend."""
    watch = []

    watch.append({
        "item": "The Lending Market (HYG vs LQD)",
        "why": "Banks and lenders usually smell trouble before everyone else. If risky company bonds start falling compared to safe ones, it is an early warning that something bad might be coming for stocks too.",
    })

    gld_ret = ret_1m.get("GLD", 0)
    if gld_ret > 0.02:
        watch.append({
            "item": "Gold Prices",
            "why": f"Gold is up {gld_ret*100:.1f}% this month. Gold goes up when people are scared. If it keeps climbing, it means big investors are worried about something — even if stocks have not dropped yet.",
        })

    pct_50 = breadth.get("pct_above_50ma", 50)
    if 40 <= pct_50 <= 60:
        watch.append({
            "item": "How Many Stocks Are Going Up",
            "why": f"Right now {pct_50:.0f}% of stocks are in an uptrend — that is right in the middle. If this number goes above 60%, it is a good sign. If it drops below 40%, that is a bad sign. We are at a tipping point.",
        })

    watch.append({
        "item": "The VIX (Market Fear Level)",
        "why": "The VIX measures how nervous the market is. Below 15 means people are very calm (maybe too calm). Above 25 means people are scared. Both extremes often lead to big moves.",
    })

    # Sector-specific watches
    if sector_data:
        sorted_s = sorted(sector_data.items(), key=lambda x: x[1].get("score", 50))
        weakest = sorted_s[0]
        strongest = sorted_s[-1]
        watch.append({
            "item": f"Gap Between {strongest[0]} and {weakest[0]}",
            "why": f"{strongest[0]} scores {strongest[1]['score']:.0f} while {weakest[0]} only scores {weakest[1]['score']:.0f}. If the weak one starts catching up, it means money is shifting between sectors — which often signals a bigger change in the market.",
        })

    # Crypto watch
    btc = crypto_data.get("BTC", {})
    if btc and abs(btc.get("change_24h", 0)) > 3:
        watch.append({
            "item": "Bitcoin Movement",
            "why": f"Bitcoin moved {btc['change_24h']:+.1f}% in just 24 hours. Big moves in crypto often happen a day or two before stocks follow in the same direction. It is like a sneak preview.",
        })

    if divergences:
        watch.append({
            "item": "The Numbers Do Not Match",
            "why": "Right now the big stock indexes are saying one thing, but when you look at what individual stocks are doing, they tell a different story. When this happens, a big move usually follows. One of them is wrong.",
        })

    return watch


def _build_market_drivers(bench_data, sector_data, inter_signals, divergences,
                          breadth, returns_1w, returns_1m, overall_inter,
                          asset_class_scores, crypto_data):
    """
    Identify what is ACTUALLY driving the market right now.
    Returns a list of driver dicts: {direction, driver, detail, impact}
    """
    drivers = []

    # ── 1. Which sectors are pulling the market and WHY ───────
    if sector_data:
        ranked = sorted(sector_data.items(), key=lambda x: x[1].get("score", 50), reverse=True)
        top = ranked[0]
        bot = ranked[-1]

        # Leaders
        top_name, top_d = top
        top_day = top_d.get("day_change", 0)
        top_1w = top_d.get("ret_1w", 0)
        top_1m = top_d.get("ret_1m", 0)

        # Sector-specific real-world drivers
        sector_why = {
            "Energy": "oil prices" if asset_class_scores.get("Oil (USO)", {}).get("ret_1w", 0) > 0 else "energy demand",
            "Technology": "AI spending and tech earnings",
            "Financials": "interest rate expectations and bank earnings",
            "Healthcare": "pharma pipelines and defensive demand",
            "Consumer Discretionary": "consumer spending data",
            "Consumer Staples": "defensive rotation — investors want safety",
            "Industrials": "manufacturing data and infrastructure spending",
            "Materials": "commodity prices and global demand",
            "Utilities": "rate cut expectations and safety demand",
            "Real Estate": "interest rate outlook and housing data",
            "Communication Services": "ad revenue and streaming growth",
        }

        why_top = sector_why.get(top_name, "strong momentum")
        why_bot = sector_why.get(bot[0], "weak momentum")

        drivers.append({
            "direction": "BULLISH" if top_d.get("score", 50) >= 60 else "NEUTRAL",
            "driver": f"{top_name} is leading the market",
            "detail": (
                f"{top_name} scores {top_d.get('score', 50):.0f}/100, {top_day:+.1f}% today, "
                f"{top_1w:+.1f}% this week, {top_1m:+.1f}% this month. "
                f"The driver: {why_top}."
            ),
            "impact": "HIGH",
        })

        bot_name, bot_d = bot
        if bot_d.get("score", 50) < 45:
            drivers.append({
                "direction": "BEARISH",
                "driver": f"{bot_name} is dragging the market down",
                "detail": (
                    f"{bot_name} scores only {bot_d.get('score', 50):.0f}/100, "
                    f"{bot_d.get('day_change', 0):+.1f}% today, {bot_d.get('ret_1m', 0):+.1f}% this month. "
                    f"The problem: {why_bot}."
                ),
                "impact": "HIGH",
            })

        # Offensive vs defensive — tells you what kind of market this is
        offensive = ["Technology", "Consumer Discretionary", "Financials", "Industrials", "Communication Services"]
        defensive = ["Utilities", "Healthcare", "Consumer Staples", "Real Estate"]
        off_scores = [sector_data[s]["score"] for s in offensive if s in sector_data]
        def_scores = [sector_data[s]["score"] for s in defensive if s in sector_data]
        off_avg = np.mean(off_scores) if off_scores else 50
        def_avg = np.mean(def_scores) if def_scores else 50

        if off_avg > def_avg + 8:
            drivers.append({
                "direction": "BULLISH",
                "driver": "Money is flowing into growth over safety",
                "detail": (
                    f"Growth sectors average {off_avg:.0f} vs defensive sectors at {def_avg:.0f}. "
                    "Investors are betting on economic expansion — they want upside, not protection. "
                    "This pattern typically shows up in the early-to-mid phase of a bull market."
                ),
                "impact": "MEDIUM",
            })
        elif def_avg > off_avg + 8:
            drivers.append({
                "direction": "BEARISH",
                "driver": "Money is rotating into safe sectors",
                "detail": (
                    f"Defensive sectors average {def_avg:.0f} vs growth sectors at {off_avg:.0f}. "
                    "Investors are quietly moving to safety — utilities, healthcare, staples. "
                    "This rotation often precedes broader market weakness by 2-4 weeks."
                ),
                "impact": "HIGH",
            })

    # ── 2. Bond market — the smartest money in the room ───────
    tlt_1w = returns_1w.get("TLT", 0)
    tlt_1m = returns_1m.get("TLT", 0)
    spy_1w = returns_1w.get("SPY", 0)
    spy_1m = returns_1m.get("SPY", 0)

    if isinstance(tlt_1w, (int, float)) and isinstance(spy_1w, (int, float)):
        if tlt_1w > 0.01 and spy_1w > 0.01:
            drivers.append({
                "direction": "BULLISH",
                "driver": "Both stocks and bonds are rising — rate cut hopes",
                "detail": (
                    f"Bonds (TLT) up {tlt_1w*100:+.1f}% this week while stocks (SPY) up {spy_1w*100:+.1f}%. "
                    "When both rise together, the market is pricing in lower interest rates ahead. "
                    "Investors believe the Fed will cut, which is good for both stocks and bonds."
                ),
                "impact": "HIGH",
            })
        elif tlt_1w > 0.01 and spy_1w < -0.01:
            drivers.append({
                "direction": "BEARISH",
                "driver": "Flight to safety — bonds up, stocks down",
                "detail": (
                    f"Bonds (TLT) up {tlt_1w*100:+.1f}% while stocks (SPY) down {spy_1w*100:+.1f}% this week. "
                    "Money is leaving stocks and rushing into government bonds. "
                    "This is the classic fear trade — big investors are getting defensive."
                ),
                "impact": "HIGH",
            })
        elif tlt_1w < -0.01 and spy_1w > 0.01:
            drivers.append({
                "direction": "BULLISH",
                "driver": "Stocks up, bonds down — confidence in growth",
                "detail": (
                    f"Stocks (SPY) up {spy_1w*100:+.1f}% while bonds (TLT) down {tlt_1w*100:+.1f}% this week. "
                    "This is the normal healthy pattern — investors are selling safe bonds "
                    "and putting that money into stocks because they believe the economy will keep growing."
                ),
                "impact": "MEDIUM",
            })
        elif tlt_1w < -0.01 and spy_1w < -0.01:
            drivers.append({
                "direction": "BEARISH",
                "driver": "Everything is selling off — stocks AND bonds down",
                "detail": (
                    f"Both stocks ({spy_1w*100:+.1f}%) and bonds ({tlt_1w*100:+.1f}%) fell this week. "
                    "When there is nowhere to hide, it usually means rising interest rates or inflation fears "
                    "are pressuring all assets. This is one of the toughest environments for investors."
                ),
                "impact": "HIGH",
            })

    # ── 3. Gold — the fear gauge ──────────────────────────────
    gld_1w = returns_1w.get("GLD", 0)
    gld_1m = returns_1m.get("GLD", 0)
    if isinstance(gld_1w, (int, float)) and gld_1w > 0.015:
        drivers.append({
            "direction": "BEARISH",
            "driver": f"Gold is surging — up {gld_1w*100:+.1f}% this week",
            "detail": (
                f"Gold (GLD) gained {gld_1w*100:+.1f}% this week and {gld_1m*100:+.1f}% this month. "
                "Rising gold means institutional investors are buying protection against uncertainty. "
                "Could be inflation worries, geopolitical tension, or loss of confidence in central banks. "
                "Whatever the reason, smart money is hedging."
            ),
            "impact": "MEDIUM",
        })
    elif isinstance(gld_1w, (int, float)) and gld_1w < -0.015:
        drivers.append({
            "direction": "BULLISH",
            "driver": f"Gold is falling — investors do not need safety",
            "detail": (
                f"Gold (GLD) dropped {gld_1w*100:+.1f}% this week. "
                "Falling gold means investors feel confident enough to sell their safety nets. "
                "Money leaving gold usually flows into stocks and riskier assets."
            ),
            "impact": "LOW",
        })

    # ── 4. Dollar — global competitiveness ────────────────────
    uup_1w = returns_1w.get("UUP", 0)
    if isinstance(uup_1w, (int, float)) and abs(uup_1w) > 0.005:
        if uup_1w > 0:
            drivers.append({
                "direction": "BEARISH",
                "driver": f"Dollar is strengthening — headwind for stocks",
                "detail": (
                    f"The US dollar (UUP) is up {uup_1w*100:+.1f}% this week. "
                    "A stronger dollar hurts US companies that sell overseas (about 40% of S&P 500 revenue). "
                    "Tech giants like Apple, Microsoft, and Nvidia all earn heavily abroad. "
                    "It also puts pressure on gold, commodities, and emerging markets."
                ),
                "impact": "MEDIUM",
            })
        else:
            drivers.append({
                "direction": "BULLISH",
                "driver": f"Dollar is weakening — tailwind for stocks",
                "detail": (
                    f"The US dollar (UUP) is down {uup_1w*100:+.1f}% this week. "
                    "A weaker dollar boosts earnings for US multinationals when they convert foreign revenue back to dollars. "
                    "It also lifts gold, commodities, and international investments."
                ),
                "impact": "MEDIUM",
            })

    # ── 5. Oil — inflation driver ─────────────────────────────
    uso_1w = returns_1w.get("USO", 0)
    if isinstance(uso_1w, (int, float)) and abs(uso_1w) > 0.02:
        if uso_1w > 0:
            drivers.append({
                "direction": "BEARISH",
                "driver": f"Oil prices jumping — inflation risk",
                "detail": (
                    f"Oil (USO) is up {uso_1w*100:+.1f}% this week. "
                    "Rising oil raises costs for everything — shipping, manufacturing, food, gas. "
                    "Higher energy costs act like a tax on consumers and squeeze corporate margins. "
                    "Energy stocks benefit but almost everything else gets hurt."
                ),
                "impact": "MEDIUM",
            })
        else:
            drivers.append({
                "direction": "BULLISH",
                "driver": f"Oil prices falling — good for consumers",
                "detail": (
                    f"Oil (USO) is down {uso_1w*100:+.1f}% this week. "
                    "Falling oil is like a tax cut for consumers — cheaper gas, cheaper shipping, lower food costs. "
                    "Good for consumer discretionary, airlines, and retail. Bad for energy stocks."
                ),
                "impact": "MEDIUM",
            })

    # ── 6. Breadth divergence — hidden weakness or strength ───
    pct_50 = breadth.get("pct_above_50ma", 50)
    pct_200 = breadth.get("pct_above_200ma", 50)
    health = breadth.get("health_score", 50)

    spy_score = bench_data.get("S&P 500", {}).get("score", 50)
    if spy_score >= 55 and pct_50 < 45:
        drivers.append({
            "direction": "BEARISH",
            "driver": "Hidden weakness — indexes up but most stocks are not",
            "detail": (
                f"The S&P 500 scores {spy_score:.0f}/100 but only {pct_50:.0f}% of stocks are above their 50-day average. "
                "This means a handful of big stocks are carrying the whole market. "
                "When leadership narrows this much, the rally is fragile — one bad day in the mega-caps "
                "and there is nothing underneath to catch the fall."
            ),
            "impact": "HIGH",
        })
    elif spy_score <= 45 and pct_50 > 55:
        drivers.append({
            "direction": "BULLISH",
            "driver": "Hidden strength — breadth is better than the index",
            "detail": (
                f"The S&P 500 scores only {spy_score:.0f}/100 but {pct_50:.0f}% of stocks are in uptrends. "
                "This means the average stock is doing better than the index suggests. "
                "Improving breadth under a weak headline number is often the start of a new leg up."
            ),
            "impact": "MEDIUM",
        })

    # ── 7. Crypto risk appetite ───────────────────────────────
    btc = crypto_data.get("BTC", {})
    if btc:
        btc_chg = btc.get("change_24h", 0)
        if btc_chg > 3:
            drivers.append({
                "direction": "BULLISH",
                "driver": f"Bitcoin surging {btc_chg:+.1f}% — risk appetite is strong",
                "detail": (
                    "Big up days in Bitcoin tend to correlate with risk-on sentiment across all markets. "
                    "When crypto is ripping, it means traders are willing to take big bets — "
                    "that confidence usually spills into stocks too."
                ),
                "impact": "LOW",
            })
        elif btc_chg < -3:
            drivers.append({
                "direction": "BEARISH",
                "driver": f"Bitcoin dropping {btc_chg:+.1f}% — risk appetite is fading",
                "detail": (
                    "Sharp Bitcoin selloffs often signal broader risk aversion. "
                    "Crypto tends to be the first domino — when it falls hard, "
                    "growth stocks and other speculative assets often follow within 24-48 hours."
                ),
                "impact": "MEDIUM",
            })

    # ── 8. Divergences — the smart money warning ──────────────
    if divergences:
        drivers.append({
            "direction": "BEARISH",
            "driver": "Something does not add up in the data",
            "detail": (
                f"{divergences[0]} "
                "When the surface looks fine but the internals tell a different story, "
                "pay attention to the internals. They are usually right."
            ),
            "impact": "HIGH",
        })

    # Sort by impact: HIGH first
    impact_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    drivers.sort(key=lambda x: impact_order.get(x.get("impact", "LOW"), 3))

    return drivers


def _build_bottom_line(spy_score, health_score, regime, overall_inter, sector_data, crypto_data):
    """The one-paragraph summary anyone can understand."""
    regime_name = regime["name"]

    # Sector context
    sector_ctx = ""
    if sector_data:
        sorted_s = sorted(sector_data.items(), key=lambda x: x[1].get("score", 50), reverse=True)
        top = sorted_s[0]
        bot = sorted_s[-1]
        sector_ctx = f" The strongest area is {top[0]} (score: {top[1]['score']:.0f}) and the weakest is {bot[0]} (score: {bot[1]['score']:.0f})."

    crypto_ctx = ""
    btc = crypto_data.get("BTC", {})
    if btc:
        chg = btc.get("change_24h", 0)
        sig = btc.get("signal", "quiet").lower()
        if sig in ("surging", "bullish"):
            crypto_ctx = f" Bitcoin is doing well today ({chg:+.1f}%), which supports the positive mood."
        elif sig in ("weak", "selling"):
            crypto_ctx = f" Bitcoin is down ({chg:+.1f}%), which adds to the cautious mood."
        else:
            crypto_ctx = f" Bitcoin is mostly flat today ({chg:+.1f}%)."

    if regime_name == "STRONG AND HEALTHY":
        return (
            f"In simple terms: the market looks good right now. The overall score is {spy_score:.0f}/100, "
            f"breadth is {health_score:.0f}/100, and most other markets agree.{sector_ctx}{crypto_ctx} "
            "If you are already invested, this is a good time to stay the course. The biggest mistake right now would be getting too scared and selling when things are actually working."
        )
    elif regime_name == "WARNING — BE CAREFUL":
        return (
            f"In simple terms: be careful. The market scores {spy_score:.0f}/100 and breadth is only {health_score:.0f}/100. "
            f"Multiple warning signs are flashing at the same time.{sector_ctx}{crypto_ctx} "
            "This is not the time to be putting a lot of new money in. Focus on protecting what you have. Wait for things to settle down before making big moves."
        )
    elif regime_name == "INVESTORS ARE NERVOUS":
        return (
            f"In simple terms: people are getting nervous.{sector_ctx}{crypto_ctx} "
            "Money is moving into safe places like gold instead of stocks. When smart money starts hiding, it usually means they see trouble that most people have not noticed yet. Better to be a little cautious right now."
        )
    elif regime_name == "GETTING WORSE":
        return (
            f"In simple terms: things are heading in the wrong direction. Market score is {spy_score:.0f}/100 "
            f"and breadth is {health_score:.0f}/100 — both below average.{sector_ctx}{crypto_ctx} "
            "Consider taking some money off the table, especially from your weakest investments. Having cash available is smart when the market is going downhill."
        )
    else:
        return (
            f"In simple terms: the market is sending mixed signals. Score is {spy_score:.0f}/100, "
            f"breadth is {health_score:.0f}/100, and other markets cannot agree on a direction.{sector_ctx}{crypto_ctx} "
            "When nothing is clear, the best thing to do is be patient. Do not make big moves based on confusion. Wait for the picture to get clearer."
        )


def _build_crypto_outlook(crypto_prices):
    """
    Build a practical daily crypto outlook from real news + price data.
    Pulls crypto headlines, scores sentiment, groups by theme, and writes
    a plain English outlook anyone can understand.
    """
    outlook = {
        "summary": "",
        "sentiment_score": 0.0,
        "sentiment_label": "Neutral",
        "news_count": 0,
        "top_headlines": [],
        "themes": [],
        "btc_outlook": "",
        "eth_outlook": "",
        "sol_outlook": "",
        "what_to_watch": [],
        "practical_take": "",
    }

    # ── Pull crypto news ──────────────────────────────────────
    try:
        news_df = fetch_crypto_news(use_cache=True)
    except Exception as e:
        logger.warning(f"Crypto news fetch failed: {e}")
        news_df = pd.DataFrame()

    if news_df.empty or "Title" not in news_df.columns:
        outlook["summary"] = "No crypto news available right now. Check back later — we pull from CoinDesk, CoinTelegraph, Decrypt, The Block, and more."
        outlook["practical_take"] = "Without current news, focus on the price action above. The numbers tell the story when headlines are quiet."
        return outlook

    # ── Score sentiment on all headlines ───────────────────────
    sentiments = []
    scored_articles = []
    for _, row in news_df.head(100).iterrows():
        title = str(row.get("Title", ""))
        desc = str(row.get("Description", "")) if pd.notna(row.get("Description")) else ""
        text = f"{title} {desc}".strip()
        if not title or len(title) < 10:
            continue

        sent = score_text(text)
        urgency = classify_urgency(text, sent["compound"])
        entities = extract_entities(text)

        scored_articles.append({
            "title": title,
            "source": str(row.get("Source", "")),
            "url": str(row.get("URL", "")),
            "sentiment": sent["compound"],
            "sentiment_label": sent["label"],
            "urgency": urgency,
            "coins_mentioned": entities.get("cryptos", []),
        })
        sentiments.append(sent["compound"])

    if not scored_articles:
        outlook["summary"] = "Found news articles but could not analyze them. The feeds may be temporarily unavailable."
        return outlook

    # ── Overall sentiment ─────────────────────────────────────
    avg_sentiment = np.mean(sentiments)
    outlook["sentiment_score"] = round(float(avg_sentiment), 3)
    outlook["news_count"] = len(scored_articles)

    positive_pct = sum(1 for s in sentiments if s > 0.05) / len(sentiments) * 100
    negative_pct = sum(1 for s in sentiments if s < -0.05) / len(sentiments) * 100
    neutral_pct = 100 - positive_pct - negative_pct

    if avg_sentiment > 0.15:
        outlook["sentiment_label"] = "Very Positive"
    elif avg_sentiment > 0.05:
        outlook["sentiment_label"] = "Leaning Positive"
    elif avg_sentiment > -0.05:
        outlook["sentiment_label"] = "Neutral"
    elif avg_sentiment > -0.15:
        outlook["sentiment_label"] = "Leaning Negative"
    else:
        outlook["sentiment_label"] = "Very Negative"

    # ── Top headlines (most impactful) ────────────────────────
    # Sort by abs sentiment * urgency weight
    urgency_weight = {"high": 3, "medium": 2, "low": 1}
    for a in scored_articles:
        a["impact"] = abs(a["sentiment"]) * urgency_weight.get(a["urgency"], 1)

    sorted_articles = sorted(scored_articles, key=lambda x: x["impact"], reverse=True)
    outlook["top_headlines"] = [
        {
            "title": a["title"],
            "source": a["source"],
            "sentiment": "Positive" if a["sentiment"] > 0.05 else ("Negative" if a["sentiment"] < -0.05 else "Neutral"),
            "urgency": a["urgency"],
        }
        for a in sorted_articles[:8]
    ]

    # ── Detect themes from headlines ──────────────────────────
    themes = []
    theme_keywords = {
        "Regulation & Government": ["sec", "regulation", "regulat", "congress", "law", "ban", "legal", "compliance", "government", "policy", "senator", "bill"],
        "ETF & Institutional": ["etf", "institutional", "blackrock", "fidelity", "grayscale", "fund", "asset manager", "wall street", "inflow", "outflow"],
        "Bitcoin Specific": ["bitcoin", "btc", "halving", "mining", "miner", "satoshi", "lightning"],
        "Ethereum & DeFi": ["ethereum", "eth", "defi", "staking", "layer 2", "l2", "dencun", "pectra", "merge"],
        "Solana & Alt-L1": ["solana", "sol", "avalanche", "cardano", "polkadot"],
        "Stablecoins & CBDC": ["stablecoin", "usdt", "usdc", "tether", "cbdc", "digital dollar"],
        "Hacks & Security": ["hack", "exploit", "breach", "stolen", "vulnerability", "rug pull", "scam", "fraud"],
        "Adoption & Partnerships": ["adopt", "partner", "accept", "payment", "merchant", "integration", "launch"],
        "Market & Trading": ["price", "rally", "crash", "surge", "plunge", "all-time", "ath", "bull", "bear", "whale"],
    }

    all_titles_lower = " ".join([a["title"].lower() for a in scored_articles])
    for theme_name, keywords in theme_keywords.items():
        matching = [kw for kw in keywords if kw in all_titles_lower]
        if matching:
            # Get sentiment of articles matching this theme
            theme_articles = [
                a for a in scored_articles
                if any(kw in a["title"].lower() for kw in keywords)
            ]
            theme_sent = np.mean([a["sentiment"] for a in theme_articles]) if theme_articles else 0
            if theme_sent > 0.05:
                mood = "positive"
            elif theme_sent < -0.05:
                mood = "negative"
            else:
                mood = "neutral"

            themes.append({
                "theme": theme_name,
                "article_count": len(theme_articles),
                "mood": mood,
                "avg_sentiment": round(float(theme_sent), 3),
            })

    # Sort by article count
    themes.sort(key=lambda x: x["article_count"], reverse=True)
    outlook["themes"] = themes[:6]

    # ── Per-coin outlooks ─────────────────────────────────────
    btc_price = crypto_prices.get("BTC", {})
    eth_price = crypto_prices.get("ETH", {})
    sol_price = crypto_prices.get("SOL", {})

    btc_articles = [a for a in scored_articles if "bitcoin" in a.get("coins_mentioned", [])]
    eth_articles = [a for a in scored_articles if "ethereum" in a.get("coins_mentioned", [])]
    sol_articles = [a for a in scored_articles if "solana" in a.get("coins_mentioned", [])]

    outlook["btc_outlook"] = _coin_outlook("Bitcoin", btc_price, btc_articles, avg_sentiment)
    outlook["eth_outlook"] = _coin_outlook("Ethereum", eth_price, eth_articles, avg_sentiment)
    outlook["sol_outlook"] = _coin_outlook("Solana", sol_price, sol_articles, avg_sentiment)

    # ── What to watch today ───────────────────────────────────
    watch = []
    high_urgency = [a for a in scored_articles if a["urgency"] == "high"]
    if high_urgency:
        watch.append(f"There are {len(high_urgency)} high-impact crypto stories right now. When big news drops, prices can move fast — especially in crypto where things trade 24/7.")

    reg_theme = next((t for t in themes if t["theme"] == "Regulation & Government"), None)
    if reg_theme and reg_theme["article_count"] >= 2:
        if reg_theme["mood"] == "positive":
            watch.append("Regulation news is trending positive — governments being crypto-friendly tends to push prices up because it makes big investors feel safe getting in.")
        elif reg_theme["mood"] == "negative":
            watch.append("There is negative regulation news in the headlines. Government crackdowns or new rules can cause sharp drops, especially for smaller coins. Bitcoin usually holds up best in these situations.")
        else:
            watch.append("Regulation is in the news but the tone is neutral. Keep an eye on it — regulation stories can shift from neutral to very negative (or positive) quickly.")

    etf_theme = next((t for t in themes if t["theme"] == "ETF & Institutional"), None)
    if etf_theme and etf_theme["article_count"] >= 2:
        watch.append("Institutional and ETF news is active. When big money (like BlackRock or Fidelity) makes moves in crypto, it tends to set the direction for weeks, not just days.")

    hack_theme = next((t for t in themes if t["theme"] == "Hacks & Security"), None)
    if hack_theme:
        watch.append("There are security or hack-related stories in the news. These can cause sudden drops in affected coins. Make sure your own crypto is on a secure wallet, not just sitting on an exchange.")

    if not watch:
        watch.append("No major breaking stories right now. The crypto market is driven by the overall mood today rather than any single headline.")

    outlook["what_to_watch"] = watch

    # ── Practical take (the money paragraph) ──────────────────
    btc_chg = btc_price.get("change_24h", 0) if btc_price else 0

    if avg_sentiment > 0.15 and btc_chg > 2:
        outlook["practical_take"] = (
            f"The news is positive and prices are backing it up — Bitcoin is up {btc_chg:+.1f}% in 24 hours. "
            f"Out of {len(scored_articles)} recent stories, {positive_pct:.0f}% are positive vs {negative_pct:.0f}% negative. "
            "When good news AND price action line up like this, the move tends to have legs. "
            "If you are already holding crypto, this is a good time to let it ride. If you have been thinking about buying, the momentum is on your side — but never invest more than you can afford to lose."
        )
    elif avg_sentiment > 0.05 and btc_chg > 0:
        outlook["practical_take"] = (
            f"The news is mildly positive and prices are inching up (BTC {btc_chg:+.1f}%). "
            f"{positive_pct:.0f}% of headlines lean positive, {negative_pct:.0f}% negative, {neutral_pct:.0f}% neutral. "
            "The mood is decent but not euphoric. This is a reasonable environment to hold your existing positions. "
            "No urgency to buy more, but no reason to panic either."
        )
    elif avg_sentiment < -0.15 and btc_chg < -2:
        outlook["practical_take"] = (
            f"The news is negative and prices are dropping — Bitcoin down {btc_chg:+.1f}% in 24 hours. "
            f"Only {positive_pct:.0f}% of headlines are positive while {negative_pct:.0f}% are negative. "
            "When bad news and falling prices come together, things can get worse before they get better. "
            "If you are holding, do not panic sell at the bottom — but also do not try to catch a falling knife by buying the dip yet. Wait for the news to calm down."
        )
    elif avg_sentiment < -0.05 and btc_chg < 0:
        outlook["practical_take"] = (
            f"News sentiment is leaning negative and prices reflect that (BTC {btc_chg:+.1f}%). "
            f"{negative_pct:.0f}% of recent headlines are negative vs {positive_pct:.0f}% positive. "
            "The mood is cautious. Not a crash, but the wind is blowing against crypto right now. "
            "Probably not the best time to add new money. Patience is your best friend here."
        )
    elif abs(avg_sentiment) <= 0.05:
        outlook["practical_take"] = (
            f"The news is balanced today — {positive_pct:.0f}% positive, {negative_pct:.0f}% negative, {neutral_pct:.0f}% neutral. "
            f"Bitcoin is {btc_chg:+.1f}% in 24 hours. "
            "No strong push in either direction from the headlines. The market is taking a breather. "
            "These quiet periods are normal — the next big move will come when a real catalyst drops. In the meantime, there is no rush to do anything."
        )
    else:
        # News and price disagree
        if avg_sentiment > 0 and btc_chg < 0:
            outlook["practical_take"] = (
                f"Interesting disconnect: the news is positive but prices are down (BTC {btc_chg:+.1f}%). "
                f"{positive_pct:.0f}% of headlines are positive. "
                "When good news fails to push prices up, it can mean the market needs more selling before it can rally. "
                "Or it could mean the good news is already old and priced in. Either way, watch closely — this kind of disconnect usually resolves within a day or two."
            )
        else:
            outlook["practical_take"] = (
                f"Prices are up (BTC {btc_chg:+.1f}%) even though news sentiment is mixed to negative. "
                f"{negative_pct:.0f}% of headlines lean negative. "
                "When prices go up despite bad news, it is actually a strong sign — it means buyers are confident enough to ignore the negativity. "
                "This kind of resilience often leads to more upside. But stay alert in case the negative headlines eventually win out."
            )

    # ── Summary paragraph ─────────────────────────────────────
    top_theme_str = ""
    if themes:
        top_theme = themes[0]
        top_theme_str = f" The biggest story theme today is {top_theme['theme']} ({top_theme['article_count']} articles, mood: {top_theme['mood']})."

    outlook["summary"] = (
        f"We scanned {len(scored_articles)} crypto news articles from CoinDesk, CoinTelegraph, Decrypt, The Block, GDELT, and more. "
        f"Overall sentiment: {outlook['sentiment_label']} (score: {avg_sentiment:+.3f}). "
        f"{positive_pct:.0f}% positive, {negative_pct:.0f}% negative, {neutral_pct:.0f}% neutral.{top_theme_str}"
    )

    return outlook


def _coin_outlook(coin_name, price_data, coin_articles, overall_sentiment):
    """Build a plain English outlook for one specific coin."""
    if not price_data:
        return f"No price data available for {coin_name} right now."

    chg = price_data.get("change_24h", 0)
    price = price_data.get("price", 0)

    if price >= 1000:
        price_str = f"${price:,.0f}"
    elif price >= 1:
        price_str = f"${price:,.2f}"
    else:
        price_str = f"${price:.4f}"

    if not coin_articles:
        # No specific news — base it on price + overall mood
        if chg > 3:
            return f"{coin_name} is at {price_str} ({chg:+.1f}% today). Strong move up, though no major {coin_name}-specific headlines are driving it. It is riding the overall crypto mood."
        elif chg > 0:
            return f"{coin_name} is at {price_str} ({chg:+.1f}% today). Slightly up with no big {coin_name} news. Just moving with the flow."
        elif chg > -3:
            return f"{coin_name} is at {price_str} ({chg:+.1f}% today). Small dip, nothing specific in the news causing it. Normal day."
        else:
            return f"{coin_name} is at {price_str} ({chg:+.1f}% today). Bigger drop, but no {coin_name}-specific bad news. Likely following the broader crypto sell-off."

    # Has specific news
    coin_sent = np.mean([a["sentiment"] for a in coin_articles])
    n_articles = len(coin_articles)
    top_headline = max(coin_articles, key=lambda a: abs(a["sentiment"]))["title"]

    if coin_sent > 0.1 and chg > 0:
        mood = "The news AND the price both point up"
        action = "Momentum is on its side right now."
    elif coin_sent > 0.1 and chg < 0:
        mood = "News is positive but the price is not following yet"
        action = "Could be a setup for a bounce, or the market might need more time."
    elif coin_sent < -0.1 and chg < 0:
        mood = "Bad news and falling prices — not a great combo"
        action = "Wait for things to stabilize before making any moves."
    elif coin_sent < -0.1 and chg > 0:
        mood = "Price is holding up despite negative headlines"
        action = "That kind of resilience is actually a good sign."
    else:
        mood = "News is mixed and so is the price action"
        action = "Nothing screaming buy or sell right now."

    return (
        f"{coin_name} at {price_str} ({chg:+.1f}% today). "
        f"{n_articles} {coin_name}-specific stories in the news. {mood}. {action} "
        f"Top headline: \"{top_headline[:80]}{'...' if len(top_headline) > 80 else ''}\""
    )

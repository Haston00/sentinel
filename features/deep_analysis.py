"""
SENTINEL — Deep Stock Analysis Engine.
Pulls EVERYTHING on a ticker: financials, earnings, news, analyst ratings,
insider trades, institutional holdings, technicals — and synthesizes it all
into AI-powered price targets across 6 horizons.

This is the "put in a ticker, get a full Wall Street research report" engine.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from data.stocks import fetch_ohlcv
from features.technical import compute_all_technical as compute_all
from utils.logger import get_logger

logger = get_logger(__name__)


def deep_analyze(ticker: str) -> dict:
    """
    Master function: pull everything, analyze everything, return everything.
    Returns a massive dict with all analysis sections.
    """
    logger.info(f"Starting deep analysis for {ticker}")
    result = {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "company": {},
        "price": {},
        "technicals": {},
        "fundamentals": {},
        "earnings": {},
        "analyst_ratings": {},
        "insider_activity": {},
        "institutional": {},
        "news": [],
        "upgrades": [],
        "risk_factors": [],
        "bull_case": [],
        "bear_case": [],
        "price_targets": {},
        "verdict": {},
    }

    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info or {}
    except Exception as e:
        logger.error(f"Failed to get info for {ticker}: {e}")
        result["error"] = f"Could not load data for {ticker}"
        return result

    # ══════════════════════════════════════════════════════════════
    # COMPANY OVERVIEW
    # ══════════════════════════════════════════════════════════════
    result["company"] = {
        "name": info.get("longName", info.get("shortName", ticker)),
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
        "description": info.get("longBusinessSummary", ""),
        "employees": info.get("fullTimeEmployees", 0),
        "website": info.get("website", ""),
        "market_cap": info.get("marketCap", 0),
        "enterprise_value": info.get("enterpriseValue", 0),
    }

    # ══════════════════════════════════════════════════════════════
    # PRICE DATA
    # ══════════════════════════════════════════════════════════════
    price = info.get("currentPrice", info.get("regularMarketPrice", 0))
    prev_close = info.get("previousClose", price)
    day_change = price - prev_close if price and prev_close else 0
    day_change_pct = (day_change / prev_close * 100) if prev_close else 0

    high_52w = info.get("fiftyTwoWeekHigh", 0)
    low_52w = info.get("fiftyTwoWeekLow", 0)
    pct_from_high = ((price - high_52w) / high_52w * 100) if high_52w else 0
    pct_from_low = ((price - low_52w) / low_52w * 100) if low_52w else 0

    sma_50 = info.get("fiftyDayAverage", 0)
    sma_200 = info.get("twoHundredDayAverage", 0)
    above_50 = price > sma_50 if sma_50 else None
    above_200 = price > sma_200 if sma_200 else None

    result["price"] = {
        "current": price,
        "previous_close": prev_close,
        "day_change": day_change,
        "day_change_pct": day_change_pct,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "pct_from_52w_high": pct_from_high,
        "pct_from_52w_low": pct_from_low,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "above_50_sma": above_50,
        "above_200_sma": above_200,
        "beta": info.get("beta", 0),
        "avg_volume": info.get("averageVolume", 0),
        "volume": info.get("volume", 0),
    }

    # ══════════════════════════════════════════════════════════════
    # TECHNICAL ANALYSIS
    # ══════════════════════════════════════════════════════════════
    try:
        ohlcv = fetch_ohlcv(ticker)
        if not ohlcv.empty:
            ta_data = compute_all(ohlcv)
            latest = ta_data.iloc[-1] if not ta_data.empty else pd.Series()

            rsi = float(latest.get("RSI_14", 50))
            macd = float(latest.get("MACD", 0))
            macd_signal = float(latest.get("MACD_Signal", 0))
            bb_upper = float(latest.get("BB_Upper", 0))
            bb_lower = float(latest.get("BB_Lower", 0))
            atr = float(latest.get("ATR_14", 0))
            adx = float(latest.get("ADX_14", 0))

            # RSI interpretation
            if rsi > 70:
                rsi_signal = "OVERBOUGHT"
                rsi_desc = f"RSI at {rsi:.0f} — stock is overbought. Often pulls back from these levels."
            elif rsi < 30:
                rsi_signal = "OVERSOLD"
                rsi_desc = f"RSI at {rsi:.0f} — stock is oversold. Could be a buying opportunity."
            elif rsi > 60:
                rsi_signal = "BULLISH"
                rsi_desc = f"RSI at {rsi:.0f} — bullish momentum but not extreme."
            elif rsi < 40:
                rsi_signal = "BEARISH"
                rsi_desc = f"RSI at {rsi:.0f} — bearish momentum but not extreme."
            else:
                rsi_signal = "NEUTRAL"
                rsi_desc = f"RSI at {rsi:.0f} — neutral territory."

            # MACD interpretation
            macd_bullish = macd > macd_signal
            if macd_bullish and macd > 0:
                macd_desc = "MACD is above signal AND above zero — strong bullish momentum."
            elif macd_bullish:
                macd_desc = "MACD crossed above signal line — bullish crossover."
            elif not macd_bullish and macd < 0:
                macd_desc = "MACD is below signal AND below zero — strong bearish momentum."
            else:
                macd_desc = "MACD crossed below signal line — bearish crossover."

            # Bollinger Band position
            if price and bb_upper and bb_lower:
                bb_range = bb_upper - bb_lower
                if bb_range > 0:
                    bb_position = (price - bb_lower) / bb_range * 100
                else:
                    bb_position = 50
            else:
                bb_position = 50

            # Trend strength
            if adx > 40:
                trend_str = "VERY STRONG"
                trend_desc = f"ADX at {adx:.0f} — very strong trend in place."
            elif adx > 25:
                trend_str = "STRONG"
                trend_desc = f"ADX at {adx:.0f} — strong trend. Trade with the trend."
            elif adx > 20:
                trend_str = "MODERATE"
                trend_desc = f"ADX at {adx:.0f} — trend is developing."
            else:
                trend_str = "WEAK"
                trend_desc = f"ADX at {adx:.0f} — no strong trend. Choppy/range-bound market."

            # Moving average signals
            ma_signals = []
            if above_50 is True:
                ma_signals.append("Price above 50-day MA (bullish)")
            elif above_50 is False:
                ma_signals.append("Price below 50-day MA (bearish)")

            if above_200 is True:
                ma_signals.append("Price above 200-day MA (long-term bullish)")
            elif above_200 is False:
                ma_signals.append("Price below 200-day MA (long-term bearish)")

            if sma_50 and sma_200:
                if sma_50 > sma_200:
                    ma_signals.append("Golden Cross active (50-day > 200-day) — bullish structure")
                else:
                    ma_signals.append("Death Cross active (50-day < 200-day) — bearish structure")

            result["technicals"] = {
                "rsi": rsi,
                "rsi_signal": rsi_signal,
                "rsi_description": rsi_desc,
                "macd": macd,
                "macd_signal_line": macd_signal,
                "macd_bullish": macd_bullish,
                "macd_description": macd_desc,
                "bb_upper": bb_upper,
                "bb_lower": bb_lower,
                "bb_position": bb_position,
                "atr": atr,
                "atr_pct": (atr / price * 100) if price else 0,
                "adx": adx,
                "trend_strength": trend_str,
                "trend_description": trend_desc,
                "ma_signals": ma_signals,
            }
    except Exception as e:
        logger.warning(f"Technical analysis failed for {ticker}: {e}")

    # ══════════════════════════════════════════════════════════════
    # FUNDAMENTALS
    # ══════════════════════════════════════════════════════════════
    pe_trailing = info.get("trailingPE", 0)
    pe_forward = info.get("forwardPE", 0)
    peg = info.get("trailingPegRatio", 0)
    ps = info.get("priceToSalesTrailing12Months", 0)
    pb = info.get("priceToBook", 0)
    ev_ebitda = info.get("enterpriseToEbitda", 0)

    # Valuation assessment
    val_signals = []
    if pe_forward and pe_trailing:
        if pe_forward < pe_trailing:
            val_signals.append(f"Forward P/E ({pe_forward:.1f}) lower than trailing ({pe_trailing:.1f}) — earnings expected to grow")
        else:
            val_signals.append(f"Forward P/E ({pe_forward:.1f}) higher than trailing ({pe_trailing:.1f}) — earnings may slow")

    if peg:
        if peg < 1:
            val_signals.append(f"PEG ratio {peg:.2f} — stock appears UNDERVALUED relative to growth")
        elif peg < 1.5:
            val_signals.append(f"PEG ratio {peg:.2f} — fairly valued relative to growth")
        else:
            val_signals.append(f"PEG ratio {peg:.2f} — stock appears EXPENSIVE relative to growth")

    result["fundamentals"] = {
        "pe_trailing": pe_trailing,
        "pe_forward": pe_forward,
        "peg_ratio": peg,
        "price_to_sales": ps,
        "price_to_book": pb,
        "ev_to_ebitda": ev_ebitda,
        "revenue": info.get("totalRevenue", 0),
        "revenue_growth": info.get("revenueGrowth", 0),
        "earnings_growth": info.get("earningsGrowth", 0),
        "profit_margin": info.get("profitMargins", 0),
        "operating_margin": info.get("operatingMargins", 0),
        "gross_margin": info.get("grossMargins", 0),
        "roe": info.get("returnOnEquity", 0),
        "roa": info.get("returnOnAssets", 0),
        "debt_to_equity": info.get("debtToEquity", 0),
        "current_ratio": info.get("currentRatio", 0),
        "free_cash_flow": info.get("freeCashflow", 0),
        "dividend_yield": info.get("dividendYield", 0),
        "payout_ratio": info.get("payoutRatio", 0),
        "valuation_signals": val_signals,
    }

    # ══════════════════════════════════════════════════════════════
    # EARNINGS
    # ══════════════════════════════════════════════════════════════
    try:
        ed = yf_ticker.earnings_dates
        if ed is not None and not ed.empty:
            recent_earnings = []
            for dt, row in ed.head(8).iterrows():
                eps_est = row.get("EPS Estimate", None)
                eps_actual = row.get("Reported EPS", None)
                surprise = row.get("Surprise(%)", None)

                entry = {
                    "date": str(dt),
                    "eps_estimate": float(eps_est) if pd.notna(eps_est) else None,
                    "eps_actual": float(eps_actual) if pd.notna(eps_actual) else None,
                    "surprise_pct": float(surprise) if pd.notna(surprise) else None,
                }
                recent_earnings.append(entry)

            # Count beats
            beats = sum(1 for e in recent_earnings if e["surprise_pct"] and e["surprise_pct"] > 0)
            misses = sum(1 for e in recent_earnings if e["surprise_pct"] and e["surprise_pct"] < 0)
            total_reported = beats + misses

            result["earnings"] = {
                "history": recent_earnings,
                "beats": beats,
                "misses": misses,
                "beat_rate": (beats / total_reported * 100) if total_reported > 0 else 0,
                "next_date": recent_earnings[0]["date"] if recent_earnings and recent_earnings[0]["eps_actual"] is None else "Unknown",
                "eps_current_year": info.get("epsCurrentYear", 0),
                "eps_forward": info.get("epsForward", 0),
            }
    except Exception as e:
        logger.warning(f"Earnings data failed: {e}")

    # ══════════════════════════════════════════════════════════════
    # ANALYST RATINGS
    # ══════════════════════════════════════════════════════════════
    try:
        recs = yf_ticker.recommendations
        if recs is not None and not recs.empty:
            latest_rec = recs.iloc[0]
            total_analysts = int(latest_rec.get("strongBuy", 0) + latest_rec.get("buy", 0) +
                               latest_rec.get("hold", 0) + latest_rec.get("sell", 0) +
                               latest_rec.get("strongSell", 0))

            strong_buy = int(latest_rec.get("strongBuy", 0))
            buy = int(latest_rec.get("buy", 0))
            hold = int(latest_rec.get("hold", 0))
            sell = int(latest_rec.get("sell", 0))
            strong_sell = int(latest_rec.get("strongSell", 0))

            bullish_pct = ((strong_buy + buy) / total_analysts * 100) if total_analysts > 0 else 0

            result["analyst_ratings"] = {
                "total_analysts": total_analysts,
                "strong_buy": strong_buy,
                "buy": buy,
                "hold": hold,
                "sell": sell,
                "strong_sell": strong_sell,
                "bullish_pct": bullish_pct,
                "consensus": info.get("recommendationKey", "N/A"),
                "mean_rating": info.get("recommendationMean", 0),
                "target_high": info.get("targetHighPrice", 0),
                "target_low": info.get("targetLowPrice", 0),
                "target_mean": info.get("targetMeanPrice", 0),
                "target_median": info.get("targetMedianPrice", 0),
                "target_upside": ((info.get("targetMeanPrice", price) - price) / price * 100) if price else 0,
            }
    except Exception as e:
        logger.warning(f"Analyst ratings failed: {e}")

    # ══════════════════════════════════════════════════════════════
    # UPGRADES / DOWNGRADES (Recent)
    # ══════════════════════════════════════════════════════════════
    try:
        ud = yf_ticker.upgrades_downgrades
        if ud is not None and not ud.empty:
            recent = ud.head(15)
            upgrades_list = []
            for dt, row in recent.iterrows():
                upgrades_list.append({
                    "date": str(dt)[:10],
                    "firm": row.get("Firm", ""),
                    "to_grade": row.get("ToGrade", ""),
                    "from_grade": row.get("FromGrade", ""),
                    "action": row.get("Action", ""),
                    "price_target": float(row.get("currentPriceTarget", 0)) if pd.notna(row.get("currentPriceTarget")) else None,
                })
            result["upgrades"] = upgrades_list
    except Exception as e:
        logger.warning(f"Upgrades data failed: {e}")

    # ══════════════════════════════════════════════════════════════
    # INSIDER ACTIVITY
    # ══════════════════════════════════════════════════════════════
    try:
        ins = yf_ticker.insider_transactions
        if ins is not None and not ins.empty:
            recent_ins = ins.head(10)
            insider_list = []
            buys = 0
            sells = 0
            for _, row in recent_ins.iterrows():
                txn = str(row.get("Transaction", row.get("Text", "")))
                insider_list.append({
                    "insider": str(row.get("Insider", row.get("Insider Trading", "Unknown"))),
                    "transaction": txn,
                    "shares": int(row.get("Shares", 0)),
                    "date": str(row.get("Start Date", ""))[:10],
                })
                txn_lower = txn.lower()
                if "buy" in txn_lower or "purchase" in txn_lower:
                    buys += 1
                elif "sell" in txn_lower or "sale" in txn_lower:
                    sells += 1

            result["insider_activity"] = {
                "transactions": insider_list,
                "recent_buys": buys,
                "recent_sells": sells,
                "net_signal": "BULLISH" if buys > sells else ("BEARISH" if sells > buys else "NEUTRAL"),
            }
    except Exception as e:
        logger.warning(f"Insider data failed: {e}")

    # ══════════════════════════════════════════════════════════════
    # INSTITUTIONAL HOLDINGS
    # ══════════════════════════════════════════════════════════════
    try:
        ih = yf_ticker.institutional_holders
        if ih is not None and not ih.empty:
            holders = []
            for _, row in ih.head(10).iterrows():
                holders.append({
                    "holder": str(row.get("Holder", "")),
                    "shares": int(row.get("Shares", 0)) if pd.notna(row.get("Shares")) else 0,
                    "pct_change": float(row.get("pctChange", 0)) if pd.notna(row.get("pctChange")) else 0,
                })

            result["institutional"] = {
                "holders": holders,
                "pct_held": info.get("heldPercentInstitutions", 0),
                "insider_pct": info.get("heldPercentInsiders", 0),
            }
    except Exception as e:
        logger.warning(f"Institutional data failed: {e}")

    # ══════════════════════════════════════════════════════════════
    # NEWS
    # ══════════════════════════════════════════════════════════════
    try:
        news_raw = yf_ticker.news
        if news_raw:
            for item in news_raw[:10]:
                content = item.get("content", {})
                title = content.get("title", "")
                summary = content.get("summary", "")
                pub = content.get("provider", {})
                pub_name = pub.get("displayName", "") if isinstance(pub, dict) else str(pub)
                pub_date = content.get("pubDate", "")

                if title:
                    result["news"].append({
                        "title": title,
                        "summary": summary[:300] if summary else "",
                        "publisher": pub_name,
                        "date": str(pub_date)[:10] if pub_date else "",
                    })
    except Exception as e:
        logger.warning(f"News data failed: {e}")

    # ══════════════════════════════════════════════════════════════
    # SYNTHESIZE: Bull/Bear Case + Risk Factors
    # ══════════════════════════════════════════════════════════════
    bull, bear, risks = _build_cases(result, info)
    result["bull_case"] = bull
    result["bear_case"] = bear
    result["risk_factors"] = risks

    # ══════════════════════════════════════════════════════════════
    # AI PRICE TARGETS
    # ══════════════════════════════════════════════════════════════
    result["price_targets"] = _compute_price_targets(result, info)

    # ══════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ══════════════════════════════════════════════════════════════
    result["verdict"] = _build_verdict(result)

    logger.info(f"Deep analysis complete for {ticker}")
    return result


def _build_cases(data, info):
    """Build bull case, bear case, and risk factors from all data."""
    bull = []
    bear = []
    risks = []
    price_data = data.get("price", {})
    fund = data.get("fundamentals", {})
    tech = data.get("technicals", {})
    analyst = data.get("analyst_ratings", {})
    earnings = data.get("earnings", {})

    # Technical signals
    if tech.get("rsi_signal") == "OVERSOLD":
        bull.append("RSI is oversold — historically a buying opportunity")
    elif tech.get("rsi_signal") == "OVERBOUGHT":
        bear.append("RSI is overbought — may be due for a pullback")
        risks.append("Overbought on RSI — short-term pullback risk")

    if tech.get("macd_bullish"):
        bull.append("MACD bullish crossover — positive momentum")
    else:
        bear.append("MACD bearish crossover — negative momentum")

    if tech.get("trend_strength") in ("STRONG", "VERY STRONG"):
        if price_data.get("above_50_sma"):
            bull.append(f"Strong uptrend (ADX {tech.get('adx', 0):.0f}) with price above 50-day MA")
        else:
            bear.append(f"Strong downtrend (ADX {tech.get('adx', 0):.0f}) with price below 50-day MA")

    # Moving averages
    if price_data.get("above_200_sma"):
        bull.append("Price above 200-day MA — long-term trend is up")
    else:
        bear.append("Price below 200-day MA — long-term trend is down")
        risks.append("Below 200-day moving average — institutional sellers may be active")

    # Fundamentals
    rev_growth = fund.get("revenue_growth", 0)
    if rev_growth and rev_growth > 0.1:
        bull.append(f"Revenue growing {rev_growth*100:.0f}% — strong top-line growth")
    elif rev_growth and rev_growth < 0:
        bear.append(f"Revenue declining {rev_growth*100:.0f}% — top-line shrinking")
        risks.append("Revenue in decline")

    earnings_growth = fund.get("earnings_growth", 0)
    if earnings_growth and earnings_growth > 0.15:
        bull.append(f"Earnings growing {earnings_growth*100:.0f}% — strong profit growth")
    elif earnings_growth and earnings_growth < 0:
        bear.append(f"Earnings declining {earnings_growth*100:.0f}%")

    profit_margin = fund.get("profit_margin", 0)
    if profit_margin and profit_margin > 0.2:
        bull.append(f"Profit margin {profit_margin*100:.0f}% — highly profitable business")

    dte = fund.get("debt_to_equity", 0)
    if dte and dte > 200:
        bear.append(f"Debt-to-equity {dte:.0f}% — heavily leveraged")
        risks.append(f"High debt load (D/E: {dte:.0f}%)")

    fcf = fund.get("free_cash_flow", 0)
    if fcf and fcf > 0:
        bull.append(f"Generating ${fcf/1e9:.1f}B in free cash flow")
    elif fcf and fcf < 0:
        bear.append("Negative free cash flow — burning cash")
        risks.append("Negative free cash flow")

    # PEG
    peg = fund.get("peg_ratio", 0)
    if peg and 0 < peg < 1:
        bull.append(f"PEG ratio {peg:.2f} — undervalued relative to growth")
    elif peg and peg > 2:
        bear.append(f"PEG ratio {peg:.2f} — expensive relative to growth")

    # Analyst consensus
    bullish_pct = analyst.get("bullish_pct", 0)
    if bullish_pct > 70:
        bull.append(f"{bullish_pct:.0f}% of analysts rate Buy or Strong Buy")
    elif bullish_pct < 40:
        bear.append(f"Only {bullish_pct:.0f}% of analysts are bullish")

    target_upside = analyst.get("target_upside", 0)
    if target_upside > 15:
        bull.append(f"Analyst mean target implies {target_upside:.0f}% upside")
    elif target_upside < -10:
        bear.append(f"Analyst mean target implies {abs(target_upside):.0f}% downside")

    # Earnings track record
    beat_rate = earnings.get("beat_rate", 0)
    if beat_rate > 75:
        bull.append(f"Beat earnings estimates {beat_rate:.0f}% of the time — reliable execution")
    elif beat_rate < 50:
        bear.append(f"Only beat earnings {beat_rate:.0f}% of the time — execution concerns")

    # Proximity to 52-week levels
    pct_from_high = price_data.get("pct_from_52w_high", 0)
    pct_from_low = price_data.get("pct_from_52w_low", 0)
    if pct_from_high and pct_from_high > -5:
        bull.append("Trading near 52-week high — breakout potential")
        risks.append("Near 52-week high — could face resistance")
    if pct_from_low and pct_from_low < 15:
        bear.append("Trading near 52-week low — weak price action")
        bull.append("Near 52-week low — could be value opportunity if fundamentals are solid")

    return bull, bear, risks


def _compute_price_targets(data, info):
    """
    Compute price targets for 6 horizons using decorrelated multi-factor analysis.

    Key accuracy improvements over naive approach:
      - Student-t distribution (fat tails) instead of Normal for probability
      - Three decorrelated signal groups (trend, value, analyst) instead of
        correlated components counted multiple times
      - Short horizons (1D/1W) show ranges, not false-precision point estimates
      - Analyst targets labeled as stale consensus, not blended silently
      - Mean-reversion drag applied to extreme readings
      - Confidence derived from signal agreement, not arbitrary decay
    """
    price = data["price"].get("current", 0)
    if not price:
        return {}

    tech = data.get("technicals", {})
    fund = data.get("fundamentals", {})
    analyst = data.get("analyst_ratings", {})
    price_data = data.get("price", {})

    # ── GROUP 1: Trend signal (-1 to +1) ──────────────────────
    # Moving averages + ADX measure the SAME thing: trend direction and strength.
    # Score them as ONE input, not three separate correlated signals.
    trend_signal = 0
    if price_data.get("above_200_sma"):
        trend_signal += 0.35
    else:
        trend_signal -= 0.35
    if price_data.get("above_50_sma"):
        trend_signal += 0.25
    else:
        trend_signal -= 0.25
    # MACD direction
    if tech.get("macd_bullish"):
        trend_signal += 0.20
    else:
        trend_signal -= 0.20
    # ADX scales the conviction — strong trend amplifies, weak trend dampens
    adx = tech.get("adx", 20)
    if adx and adx > 25:
        trend_signal *= min(1.3, 1.0 + (adx - 25) / 50)
    elif adx and adx < 20:
        trend_signal *= 0.6  # Choppy market — discount trend signals
    trend_signal = max(-1, min(1, trend_signal))

    # ── GROUP 2: Value / fundamental signal (-1 to +1) ────────
    # Growth + valuation as ONE signal (they're correlated)
    value_signal = 0
    rev_g = fund.get("revenue_growth", 0)
    if rev_g:
        value_signal += max(-0.25, min(0.25, rev_g))
    earn_g = fund.get("earnings_growth", 0)
    if earn_g:
        value_signal += max(-0.25, min(0.25, earn_g))
    peg = fund.get("peg_ratio", 0)
    if peg and 0 < peg < 1:
        value_signal += 0.20
    elif peg and peg > 2.5:
        value_signal -= 0.20
    margin = fund.get("profit_margin", 0)
    if margin and margin > 0.2:
        value_signal += 0.10
    elif margin and margin < 0.05:
        value_signal -= 0.10
    value_signal = max(-1, min(1, value_signal))

    # ── GROUP 3: Analyst consensus (-1 to +1) ─────────────────
    # Capped and dampened because sell-side analysts are biased bullish
    analyst_signal = 0
    target_upside = analyst.get("target_upside", 0)
    if target_upside:
        # Dampen bullish analyst targets (known bullish bias)
        if target_upside > 0:
            analyst_signal = min(0.7, target_upside / 40)  # 40% upside = 0.7, not 1.0
        else:
            analyst_signal = max(-1, target_upside / 25)  # Downside gets more weight

    # ── Mean-reversion adjustment ─────────────────────────────
    # If RSI is extreme, apply drag to the combined score
    rsi = tech.get("rsi", 50)
    mean_rev_drag = 0
    if rsi and rsi > 75:
        mean_rev_drag = -0.10 * ((rsi - 75) / 25)  # Up to -10% drag at RSI 100
    elif rsi and rsi < 25:
        mean_rev_drag = 0.10 * ((25 - rsi) / 25)   # Up to +10% boost at RSI 0

    # ── Combine with decorrelated weights ─────────────────────
    # Trend is real-time, value is slow-moving, analyst is stale
    combined = (trend_signal * 0.40 + value_signal * 0.35 + analyst_signal * 0.25)
    combined += mean_rev_drag
    combined = max(-1, min(1, combined))

    # ── Confidence from signal agreement ──────────────────────
    # If all 3 groups agree in direction, confidence is high.
    # If they disagree, confidence drops.
    signs = [
        1 if trend_signal > 0.1 else (-1 if trend_signal < -0.1 else 0),
        1 if value_signal > 0.1 else (-1 if value_signal < -0.1 else 0),
        1 if analyst_signal > 0.1 else (-1 if analyst_signal < -0.1 else 0),
    ]
    agreement = abs(sum(signs)) / 3  # 0 = total disagreement, 1 = full agreement
    base_conf = 40 + agreement * 30  # Range: 40-70%

    # ── Expected return (data-anchored) ───────────────────────
    # Use historical SPY CAGR (10.5%) as baseline, adjusted by combined score
    # Score of +1 = double the base, -1 = negative base
    annual_return = 0.105 * (1 + combined)  # Range: 0% to 21%
    if combined < 0:
        annual_return = 0.105 + combined * 0.20  # Range: -9.5% to 10.5%

    daily_ret = annual_return / 252

    # Volatility from ATR (real data, not assumed)
    atr_pct = tech.get("atr_pct", 1.5)
    if not atr_pct:
        atr_pct = 1.5
    daily_vol = atr_pct / 100

    horizons = {
        "1_day": 1,
        "1_week": 5,
        "1_month": 21,
        "3_months": 63,
        "6_months": 126,
        "1_year": 252,
    }

    # Use Student-t distribution (fat tails) instead of Normal
    from scipy.stats import t as student_t
    df_t = 5  # Degrees of freedom — fatter tails than Normal, matches equity returns

    targets = {}
    for name, days in horizons.items():
        expected_move = daily_ret * days
        volatility = daily_vol * np.sqrt(days)

        target_price = price * (1 + expected_move)
        # Use wider bands (Student-t 90% CI instead of Normal 1.5-sigma)
        t_mult = student_t.ppf(0.95, df_t)  # ~2.015 for df=5 (wider than 1.645 Normal)
        high_target = price * (1 + expected_move + t_mult * volatility)
        low_target = price * (1 + expected_move - t_mult * volatility)

        # Probability of being up — Student-t accounts for fat tails
        if volatility > 0:
            z_score = expected_move / volatility
            prob_up = float(student_t.cdf(z_score, df_t)) * 100
        else:
            prob_up = 52 if expected_move > 0 else 48

        # Confidence: starts from signal agreement, fades with horizon
        horizon_decay = max(0.5, 1 - (days / 252) * 0.3)  # Lose up to 30% over 1 year
        conf = base_conf * horizon_decay

        # Short horizons get extra confidence penalty — noise dominates
        if days <= 5:
            conf = min(conf, 45)  # 1D/1W capped at 45% — honest about noise

        # Direction thresholds scale with horizon
        dir_threshold = 0.005 * np.sqrt(days)  # Wider threshold for longer horizons
        if expected_move > dir_threshold:
            direction = "BULLISH"
        elif expected_move < -dir_threshold:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        label_map = {
            "1_day": "1 Day",
            "1_week": "1 Week",
            "1_month": "1 Month",
            "3_months": "3 Months",
            "6_months": "6 Months",
            "1_year": "1 Year",
        }

        # Short horizons: round aggressively to avoid false precision
        if days <= 5:
            target_price = round(target_price, 1)
            high_target = round(high_target, 1)
            low_target = round(low_target, 1)
        else:
            target_price = round(target_price, 2)
            high_target = round(high_target, 2)
            low_target = round(low_target, 2)

        targets[name] = {
            "label": label_map[name],
            "days": days,
            "target_price": target_price,
            "high_target": high_target,
            "low_target": low_target,
            "expected_move_pct": round(expected_move * 100, 2),
            "volatility_pct": round(volatility * 100, 2),
            "probability_up": round(prob_up, 1),
            "confidence": round(conf, 0),
            "direction": direction,
        }

    # Analyst targets shown as SEPARATE reference, not blended into model targets
    analyst_mean = analyst.get("target_mean", 0)
    analyst_high = analyst.get("target_high", 0)
    analyst_low = analyst.get("target_low", 0)
    if analyst_mean and "1_year" in targets:
        targets["1_year"]["analyst_consensus"] = {
            "mean": round(analyst_mean, 2),
            "high": round(analyst_high, 2),
            "low": round(analyst_low, 2),
            "note": "Sell-side consensus — typically biased 5-15% bullish",
        }

    return targets


def _build_verdict(data):
    """Build final verdict and overall recommendation."""
    targets = data.get("price_targets", {})
    bull_count = len(data.get("bull_case", []))
    bear_count = len(data.get("bear_case", []))
    risk_count = len(data.get("risk_factors", []))

    # Score from price targets
    target_scores = []
    for name, t in targets.items():
        if t.get("direction") == "BULLISH":
            target_scores.append(1)
        elif t.get("direction") == "BEARISH":
            target_scores.append(-1)
        else:
            target_scores.append(0)

    avg_target = np.mean(target_scores) if target_scores else 0

    # Combine
    case_score = (bull_count - bear_count) / max(bull_count + bear_count, 1)
    overall = (avg_target * 0.5 + case_score * 0.5)

    if overall > 0.3:
        verdict = "BULLISH"
        color = "#00C853"
        summary = "The weight of evidence favors the bulls. Technicals, fundamentals, and analyst consensus align positively."
    elif overall > 0.1:
        verdict = "LEAN BULLISH"
        color = "#8BC34A"
        summary = "Slightly more bullish than bearish signals, but conviction is moderate. Watch for confirmation."
    elif overall > -0.1:
        verdict = "NEUTRAL"
        color = "#FFD600"
        summary = "Mixed signals across technicals and fundamentals. No clear directional edge right now."
    elif overall > -0.3:
        verdict = "LEAN BEARISH"
        color = "#FF9800"
        summary = "More bearish than bullish signals. Caution warranted unless you see a catalyst."
    else:
        verdict = "BEARISH"
        color = "#FF1744"
        summary = "The weight of evidence favors the bears. Multiple red flags across technicals and fundamentals."

    # 1-month target for quick reference
    one_month = targets.get("1_month", {})
    one_year = targets.get("1_year", {})

    return {
        "verdict": verdict,
        "color": color,
        "summary": summary,
        "overall_score": round(overall, 2),
        "bull_points": bull_count,
        "bear_points": bear_count,
        "risk_count": risk_count,
        "short_term_target": one_month.get("target_price", 0),
        "short_term_direction": one_month.get("direction", "NEUTRAL"),
        "long_term_target": one_year.get("target_price", 0),
        "long_term_direction": one_year.get("direction", "NEUTRAL"),
    }

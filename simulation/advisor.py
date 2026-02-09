"""
SENTINEL — Wharton-Level Investment Advisor Engine.
Uses ALL SENTINEL signals to generate specific trade recommendations
with institutional-grade thesis, sizing, and risk reasoning.
Teaches investment principles with every recommendation.
"""

import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

from config.assets import SECTORS, BENCHMARKS, CRYPTO_ALL, get_sector_for_ticker
from data.stocks import fetch_ohlcv
from data.crypto import get_current_prices as get_crypto_prices
from features.signals import compute_composite_score
from features.intermarket import compute_intermarket_signals
from features.breadth import compute_sector_breadth
from simulation.portfolio import get_portfolio
from utils.logger import get_logger

logger = get_logger(__name__)

ET = ZoneInfo("America/New_York")

# ── Educational concepts tied to each recommendation reason ────
LESSONS = {
    "momentum": {
        "title": "Momentum Investing",
        "what": "Stocks that have been going up tend to keep going up — and stocks going down tend to keep going down. This is one of the most proven patterns in finance.",
        "why": "It works because investors are slow to react to new information. Good news takes time to fully get priced in. Academic research by Jegadeesh & Titman (1993) showed momentum generates 1% per month excess returns.",
        "risk": "Momentum crashes hard during market reversals. The strategy that made money for 11 months can lose it all in month 12 if you are not careful.",
    },
    "value": {
        "title": "Value Investing",
        "what": "Buying things that are cheap relative to their fundamentals. Warren Buffett built his fortune on this idea.",
        "why": "Markets overreact to bad news. When investors panic and sell, prices drop below fair value. Patient buyers profit when the price recovers. Fama & French (1992) proved value stocks outperform over time.",
        "risk": "Sometimes cheap is cheap for a reason. The key is distinguishing between 'on sale' and 'going out of business.'",
    },
    "sector_rotation": {
        "title": "Sector Rotation",
        "what": "Different sectors lead at different points in the economic cycle. Technology leads in expansion, Utilities lead in recession. Smart money rotates ahead of the crowd.",
        "why": "The economy moves in cycles. When growth is accelerating, cyclical sectors (tech, discretionary, industrials) outperform. When growth slows, defensive sectors (utilities, staples, healthcare) hold up better.",
        "risk": "Timing the rotation is the hard part. Move too early and you leave money on the table. Move too late and you catch the downturn.",
    },
    "diversification": {
        "title": "Diversification — The Only Free Lunch",
        "what": "Nobel Prize winner Harry Markowitz called diversification 'the only free lunch in finance.' By spreading your money across different assets that don't move together, you reduce risk without reducing expected returns.",
        "why": "When one investment drops, another often rises. A portfolio of uncorrelated assets has lower volatility than any single holding — meaning smoother returns and smaller drawdowns.",
        "risk": "Over-diversification kills returns. Owning 50 stocks is not much different from owning an index fund. The sweet spot is 15-25 positions across different sectors.",
    },
    "risk_management": {
        "title": "Position Sizing — How Much Matters More Than What",
        "what": "Professional traders know that HOW MUCH you invest in each position matters more than WHAT you invest in. A 2% position that doubles makes you 2%. A 20% position that drops 50% loses you 10%.",
        "why": "The Kelly Criterion (from Bell Labs, used by hedge funds) says to size positions based on your edge and the odds. Higher conviction = larger position, but never more than you can afford to lose.",
        "risk": "The biggest mistake beginners make is putting too much in one trade. Even if you are right 70% of the time, one oversized loser can wipe out months of gains.",
    },
    "regime_awareness": {
        "title": "Regime Awareness — Know When the Rules Change",
        "what": "Markets alternate between bull, bear, and transition regimes. The strategy that works in a bull market can destroy you in a bear market. Hedge funds use Hidden Markov Models to detect these shifts.",
        "why": "In a bull market, buying dips works. In a bear market, buying dips is catching falling knives. The first step to making money is knowing which game you are playing.",
        "risk": "Regime shifts happen fast and are only obvious in hindsight. That is why SENTINEL monitors multiple signals — no single indicator catches every shift.",
    },
    "mean_reversion": {
        "title": "Mean Reversion — What Goes Up Too Fast Must Come Down",
        "what": "When prices stretch too far from their average (measured by Bollinger Bands, RSI, etc.), they tend to snap back. This is one of the oldest trading principles.",
        "why": "Extreme moves attract profit-takers and contrarian buyers. A stock that drops 20% in a week often bounces 5-10% in the following days — not because the company improved, but because selling exhausted itself.",
        "risk": "Mean reversion fails during regime changes. A stock dropping from $100 to $80 might 'mean revert' to $85 before dropping to $60. Always check the bigger trend first.",
    },
    "intermarket": {
        "title": "Intermarket Analysis — Everything Is Connected",
        "what": "Stocks, bonds, gold, oil, and the dollar all affect each other. When bonds rally and stocks fall, it means money is fleeing to safety. When oil spikes, consumer stocks suffer. Professional investors watch ALL markets, not just stocks.",
        "why": "The bond market is smarter than the stock market. It has more institutional players and less retail noise. When bonds and stocks disagree, bonds are usually right.",
        "risk": "Correlations change during crises. In 2008, everything except government bonds fell together. Diversification across asset classes helps, but does not eliminate risk during panics.",
    },
    "crypto_risk": {
        "title": "Crypto — Asymmetric Risk/Reward",
        "what": "Crypto offers the potential for massive gains but also devastating losses. Bitcoin has dropped 80%+ three times in its history and recovered each time to new highs. Most altcoins never recover.",
        "why": "Crypto is still in the adoption phase — like the internet in 1997. The winners will generate enormous returns, but most projects will fail. Stick to BTC and ETH for core positions.",
        "risk": "Never invest more than you can afford to lose in crypto. A 5-10% allocation to your overall portfolio gives you exposure to the upside without risking your financial stability.",
    },
}


def generate_recommendations(max_recs: int = 5) -> list:
    """
    Generate up to max_recs trade recommendations using ALL SENTINEL signals.
    Each recommendation includes: what to do, why, education, risk warning, sizing.
    """
    recs = []
    portfolio = get_portfolio()
    cash = portfolio["cash"]
    total_value = portfolio["total_value"]
    positions = {p["ticker"]: p for p in portfolio.get("positions_detail", [])}

    # ── Gather market intelligence ────────────────────────────
    # Score all sector ETFs
    sector_scores = {}
    for sector_name, info in SECTORS.items():
        etf = info["etf"]
        df = fetch_ohlcv(etf)
        if df.empty or len(df) < 50:
            continue
        result = compute_composite_score(df)
        close = df["Close"]
        current = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) > 1 else current
        day_chg = (current - prev) / prev * 100
        ret_1m = (current / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0

        sector_scores[sector_name] = {
            "etf": etf,
            "score": result["score"],
            "label": result["label"],
            "signals": result.get("signals", []),
            "price": current,
            "day_change": round(day_chg, 2),
            "ret_1m": round(ret_1m, 2),
        }

    # Breadth and intermarket
    breadth = compute_sector_breadth()
    health_score = breadth.get("summary", {}).get("health_score", 50)
    inter = compute_intermarket_signals()
    overall_inter = inter.get("overall", "MIXED")

    # SPY score for regime
    spy_df = fetch_ohlcv("SPY")
    spy_result = compute_composite_score(spy_df) if not spy_df.empty and len(spy_df) >= 50 else {"score": 50}
    spy_score = spy_result["score"]

    # Determine regime
    if spy_score >= 60 and health_score >= 55:
        regime = "BULLISH"
    elif spy_score <= 40 or health_score <= 35:
        regime = "BEARISH"
    else:
        regime = "NEUTRAL"

    # ── Rule 1: Portfolio-level guidance first ────────────────
    cash_pct = (cash / total_value * 100) if total_value > 0 else 100
    num_positions = len(positions)

    if cash_pct > 80 and regime == "BULLISH":
        recs.append(_rec_deploy_cash(cash, total_value, sector_scores, regime, health_score))

    if cash_pct < 10 and regime == "BEARISH":
        recs.append(_rec_raise_cash(positions, sector_scores, regime))

    # ── Rule 2: Sector rotation opportunities ─────────────────
    if sector_scores:
        ranked = sorted(sector_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        top_sector_name, top_sector = ranked[0]
        bot_sector_name, bot_sector = ranked[-1]

        # Recommend buying strongest sector if not owned
        if top_sector["score"] >= 60 and top_sector["etf"] not in positions:
            recs.append(_rec_buy_sector(
                top_sector_name, top_sector, cash, total_value, regime, health_score,
            ))

        # Recommend selling weakest sector if owned
        if bot_sector["score"] <= 40 and bot_sector["etf"] in positions:
            recs.append(_rec_sell_sector(
                bot_sector_name, bot_sector, positions[bot_sector["etf"]],
            ))

        # Find sector with biggest score improvement (momentum shift)
        for name, data in ranked[:3]:
            if data["score"] >= 55 and data["ret_1m"] > 3 and data["etf"] not in positions:
                if len(recs) < max_recs:
                    recs.append(_rec_momentum_sector(name, data, cash, total_value))
                break

    # ── Rule 3: Individual stock opportunities ────────────────
    if sector_scores and len(recs) < max_recs:
        best_sector_name = ranked[0][0] if ranked else None
        if best_sector_name and best_sector_name in SECTORS:
            holdings = SECTORS[best_sector_name]["holdings"][:5]
            best_stock = None
            best_score = 0
            for ticker in holdings:
                if ticker in positions:
                    continue
                df = fetch_ohlcv(ticker)
                if df.empty or len(df) < 50:
                    continue
                result = compute_composite_score(df)
                if result["score"] > best_score:
                    best_score = result["score"]
                    close = df["Close"]
                    current = float(close.iloc[-1])
                    ret_1m = (current / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0
                    best_stock = {
                        "ticker": ticker,
                        "score": result["score"],
                        "label": result["label"],
                        "price": current,
                        "ret_1m": round(ret_1m, 2),
                        "signals": result.get("signals", []),
                        "sector": best_sector_name,
                    }

            if best_stock and best_stock["score"] >= 55:
                recs.append(_rec_buy_stock(best_stock, cash, total_value, regime))

    # ── Rule 4: Crypto opportunities ──────────────────────────
    if len(recs) < max_recs:
        try:
            cprices = get_crypto_prices()
            if cprices:
                btc = cprices.get("bitcoin", {})
                btc_chg = btc.get("usd_24h_change", 0) or 0
                btc_price = btc.get("usd", 0)

                # Only recommend crypto if regime is not bearish and BTC is not crashing
                crypto_weight = sum(p["weight"] for p in positions.values() if isinstance(p, dict) and p.get("is_crypto"))
                if not isinstance(crypto_weight, (int, float)):
                    crypto_weight = 0
                # Get crypto weight from detail
                crypto_weight = sum(
                    p["weight"] for p in portfolio.get("positions_detail", [])
                    if p.get("is_crypto")
                )

                if crypto_weight < 10 and btc_chg > -5 and "BTC" not in positions and regime != "BEARISH":
                    recs.append(_rec_buy_crypto(btc_price, btc_chg, cash, total_value))
        except Exception:
            pass

    # ── Rule 5: Diversification check ─────────────────────────
    if num_positions > 0 and num_positions < 3 and len(recs) < max_recs:
        recs.append({
            "action": "DIVERSIFY",
            "ticker": "",
            "urgency": "MEDIUM",
            "headline": f"You only have {num_positions} position(s) — your portfolio is too concentrated",
            "thesis": (
                f"With only {num_positions} position(s), a single bad day in one stock could seriously hurt your portfolio. "
                "Aim for 8-15 positions across at least 4 different sectors. This is not about being timid — "
                "it is about making sure one bad bet does not ruin your whole account."
            ),
            "education": LESSONS["diversification"],
            "suggested_shares": 0,
            "suggested_cost": 0,
            "suggested_pct": 0,
        })

    # ── Rule 6: Check for losers to cut ───────────────────────
    for p in portfolio.get("positions_detail", []):
        if p["pnl_pct"] < -10 and len(recs) < max_recs:
            recs.append({
                "action": "SELL",
                "ticker": p["ticker"],
                "urgency": "HIGH",
                "headline": f"Consider cutting {p['ticker']} — down {p['pnl_pct']:.1f}% from your entry",
                "thesis": (
                    f"You bought {p['ticker']} at ${p['avg_cost']:.2f} and it is now ${p['current_price']:.2f} — "
                    f"a {p['pnl_pct']:.1f}% loss. "
                    "One of the hardest lessons in investing: the market does not care what you paid. "
                    "If the signals say this is going lower, cutting the loss now and redeploying that capital "
                    "into something working is usually the smarter move. "
                    "Professional traders have a rule: 'Cut your losers fast, let your winners run.'"
                ),
                "education": LESSONS["risk_management"],
                "suggested_shares": p["shares"],
                "suggested_cost": p["market_value"],
                "suggested_pct": p["weight"],
            })
            break  # Only one cut recommendation at a time

    return recs[:max_recs]


def _rec_deploy_cash(cash, total_value, sector_scores, regime, health):
    """Recommend deploying idle cash in a bullish regime."""
    cash_pct = cash / total_value * 100
    ranked = sorted(sector_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    top3 = [f"{name} ({data['etf']}, score {data['score']:.0f})" for name, data in ranked[:3]]

    return {
        "action": "DEPLOY CASH",
        "ticker": ranked[0][1]["etf"] if ranked else "SPY",
        "urgency": "MEDIUM",
        "headline": f"You have {cash_pct:.0f}% of your portfolio sitting in cash during a bullish market",
        "thesis": (
            f"Your portfolio is ${total_value:,.0f} but ${cash:,.0f} ({cash_pct:.0f}%) is uninvested. "
            f"The market regime is {regime} with health score {health:.0f}/100. "
            "Cash earns nothing while the market moves up without you. "
            f"Top sectors right now: {', '.join(top3)}. "
            "Consider putting 20-30% of your cash to work in the strongest areas. "
            "Do not invest it all at once — spread your buys over a few days to get a better average price."
        ),
        "education": LESSONS["sector_rotation"],
        "suggested_shares": 0,
        "suggested_cost": round(cash * 0.25, 2),
        "suggested_pct": 25,
    }


def _rec_raise_cash(positions, sector_scores, regime):
    """Recommend raising cash in a bearish regime."""
    weakest = None
    for ticker, pos in positions.items():
        if weakest is None or pos.get("pnl_pct", 0) < weakest.get("pnl_pct", 0):
            weakest = {**pos, "ticker": ticker} if isinstance(pos, dict) else None

    ticker = weakest["ticker"] if weakest else "your weakest position"

    return {
        "action": "RAISE CASH",
        "ticker": ticker,
        "urgency": "HIGH",
        "headline": "Market regime is bearish — consider raising cash",
        "thesis": (
            f"The market regime is {regime}. In bear markets, cash is a position. "
            "Having cash means you can buy quality assets at lower prices when the selling stops. "
            "The biggest mistake in a downturn is being fully invested and having no ammunition when great opportunities appear. "
            f"Consider selling your weakest positions to build a 20-30% cash reserve."
        ),
        "education": LESSONS["regime_awareness"],
        "suggested_shares": 0,
        "suggested_cost": 0,
        "suggested_pct": 0,
    }


def _rec_buy_sector(sector_name, sector_data, cash, total_value, regime, health):
    """Recommend buying a strong sector ETF."""
    etf = sector_data["etf"]
    score = sector_data["score"]
    price = sector_data["price"]
    signals = sector_data.get("signals", [])
    sig_text = "; ".join(signals[:3]) if signals else "multiple indicators aligned"

    # Position sizing: 5-10% of portfolio based on conviction
    target_pct = min(10, max(5, score / 10))
    target_dollars = total_value * target_pct / 100
    target_dollars = min(target_dollars, cash * 0.5)  # Never more than half cash
    suggested_shares = int(target_dollars / price) if price > 0 else 0

    return {
        "action": "BUY",
        "ticker": etf,
        "urgency": "MEDIUM" if score >= 65 else "LOW",
        "headline": f"Buy {sector_name} ({etf}) — strongest sector, scoring {score:.0f}/100",
        "thesis": (
            f"{sector_name} is the top-ranked sector right now with a score of {score:.0f}/100. "
            f"It is {sector_data['day_change']:+.1f}% today and {sector_data['ret_1m']:+.1f}% over the past month. "
            f"Key signals: {sig_text}. "
            f"Market regime is {regime} with health {health:.0f}/100. "
            f"Buying the ETF ({etf}) at ${price:.2f} gives you exposure to the entire sector "
            "without the risk of picking the wrong individual stock."
        ),
        "education": LESSONS["sector_rotation"],
        "suggested_shares": suggested_shares,
        "suggested_cost": round(suggested_shares * price, 2),
        "suggested_pct": round(target_pct, 1),
        "is_crypto": False,
        "sector": sector_name,
    }


def _rec_sell_sector(sector_name, sector_data, position):
    """Recommend selling a weak sector."""
    etf = sector_data["etf"]
    return {
        "action": "SELL",
        "ticker": etf,
        "urgency": "HIGH" if sector_data["score"] <= 30 else "MEDIUM",
        "headline": f"Consider selling {sector_name} ({etf}) — weakest sector at {sector_data['score']:.0f}/100",
        "thesis": (
            f"{sector_name} has dropped to a score of {sector_data['score']:.0f}/100 — that is weak. "
            f"It is {sector_data['day_change']:+.1f}% today and {sector_data['ret_1m']:+.1f}% this month. "
            f"You own {position.get('shares', 0)} shares with P&L of {position.get('pnl_pct', 0):.1f}%. "
            "When a sector scores below 40, the odds of it continuing to fall are higher than the odds of a recovery. "
            "Selling now and rotating into the strongest sector is what institutional investors do — they follow the money, not hope."
        ),
        "education": LESSONS["sector_rotation"],
        "suggested_shares": position.get("shares", 0),
        "suggested_cost": position.get("market_value", 0),
        "suggested_pct": position.get("weight", 0),
    }


def _rec_momentum_sector(name, data, cash, total_value):
    """Recommend a sector with strong momentum."""
    price = data["price"]
    target_pct = 5
    target_dollars = min(total_value * target_pct / 100, cash * 0.4)
    suggested_shares = int(target_dollars / price) if price > 0 else 0

    return {
        "action": "BUY",
        "ticker": data["etf"],
        "urgency": "LOW",
        "headline": f"{name} ({data['etf']}) has strong momentum — up {data['ret_1m']:+.1f}% this month",
        "thesis": (
            f"{name} scores {data['score']:.0f}/100 and has gained {data['ret_1m']:+.1f}% over the past month. "
            "Academic research shows that assets with strong recent performance tend to continue outperforming "
            "for the next 3-12 months. This is called the 'momentum factor' and it has worked across every market "
            "and every time period ever studied."
        ),
        "education": LESSONS["momentum"],
        "suggested_shares": suggested_shares,
        "suggested_cost": round(suggested_shares * price, 2),
        "suggested_pct": round(target_pct, 1),
        "is_crypto": False,
        "sector": name,
    }


def _rec_buy_stock(stock, cash, total_value, regime):
    """Recommend buying an individual stock."""
    price = stock["price"]
    target_pct = min(7, max(3, stock["score"] / 15))
    target_dollars = min(total_value * target_pct / 100, cash * 0.3)
    suggested_shares = int(target_dollars / price) if price > 0 else 0
    sig_text = "; ".join(stock.get("signals", [])[:3]) if stock.get("signals") else "solid technical setup"

    return {
        "action": "BUY",
        "ticker": stock["ticker"],
        "urgency": "LOW",
        "headline": f"Buy {stock['ticker']} — score {stock['score']:.0f}/100 in the leading sector ({stock['sector']})",
        "thesis": (
            f"{stock['ticker']} scores {stock['score']:.0f}/100 ({stock['label']}). "
            f"It is in {stock['sector']}, the strongest sector right now. "
            f"Price: ${price:.2f}, up {stock['ret_1m']:+.1f}% this month. "
            f"Signals: {sig_text}. "
            "Buying the best stock in the best sector is a classic institutional strategy — "
            "you are combining sector rotation with stock selection, two independent sources of edge."
        ),
        "education": LESSONS["momentum"],
        "suggested_shares": suggested_shares,
        "suggested_cost": round(suggested_shares * price, 2),
        "suggested_pct": round(target_pct, 1),
        "is_crypto": False,
        "sector": stock["sector"],
    }


def _rec_buy_crypto(btc_price, btc_chg, cash, total_value):
    """Recommend a small BTC position."""
    target_pct = 5
    target_dollars = min(total_value * target_pct / 100, cash * 0.2)
    # BTC uses fractional shares
    suggested_shares = round(target_dollars / btc_price, 6) if btc_price > 0 else 0

    return {
        "action": "BUY",
        "ticker": "BTC",
        "urgency": "LOW",
        "headline": f"Consider a small Bitcoin position (5% max) — currently ${btc_price:,.0f} ({btc_chg:+.1f}% today)",
        "thesis": (
            f"Bitcoin is at ${btc_price:,.0f} ({btc_chg:+.1f}% in 24h). "
            "A 5% portfolio allocation to Bitcoin gives you exposure to the potential upside of crypto "
            "without putting your overall portfolio at risk. "
            "Institutional investors (BlackRock, Fidelity) now allocate 1-5% of client portfolios to Bitcoin. "
            "The key rule: only invest what you can emotionally handle losing 50% of overnight, "
            "because that has happened before and will happen again."
        ),
        "education": LESSONS["crypto_risk"],
        "suggested_shares": suggested_shares,
        "suggested_cost": round(suggested_shares * btc_price, 2),
        "suggested_pct": round(target_pct, 1),
        "is_crypto": True,
        "sector": "Crypto",
    }


def analyze_closed_trade(trade: dict) -> dict:
    """
    Post-trade education: analyze a completed sell trade.
    What went right, what went wrong, what to learn.
    """
    ticker = trade.get("ticker", "")
    pnl = trade.get("realized_pnl", 0)
    pnl_pct = trade.get("realized_pnl_pct", 0)
    entry_price = trade.get("cost_basis_per_share", 0)
    exit_price = trade.get("price", 0)

    won = pnl > 0

    # Get current signals to check if the exit was good
    analysis = {
        "ticker": ticker,
        "result": "WIN" if won else "LOSS",
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "lessons": [],
    }

    if won:
        analysis["grade"] = "A" if pnl_pct > 10 else "B" if pnl_pct > 5 else "C"
        analysis["summary"] = (
            f"You made ${pnl:,.2f} ({pnl_pct:+.1f}%) on {ticker}. "
            f"Bought at ${entry_price:.2f}, sold at ${exit_price:.2f}. "
        )
        if pnl_pct > 10:
            analysis["summary"] += "Excellent trade — double-digit gain. The key question: did you let it run long enough, or did you leave money on the table?"
            analysis["lessons"].append("Review whether you sold because of a real signal change, or because you were nervous about giving back gains. Winners should be sold on signal changes, not emotions.")
        elif pnl_pct > 5:
            analysis["summary"] += "Solid gain. Consistent 5%+ trades compound into serious wealth over time."
            analysis["lessons"].append("A 5% gain every month is 80% per year. The goal is not home runs — it is consistent base hits.")
        else:
            analysis["summary"] += "Small gain. Better than a loss, but think about whether this position was worth the capital allocation."
            analysis["lessons"].append("Small gains on large positions mean your capital was not working hard enough. Look for higher-conviction setups next time.")
    else:
        analysis["grade"] = "D" if pnl_pct > -10 else "F"
        analysis["summary"] = (
            f"You lost ${abs(pnl):,.2f} ({pnl_pct:+.1f}%) on {ticker}. "
            f"Bought at ${entry_price:.2f}, sold at ${exit_price:.2f}. "
        )
        if pnl_pct > -5:
            analysis["summary"] += "Small loss — well managed. Cutting losers fast is what separates professionals from amateurs."
            analysis["lessons"].append("Good discipline cutting this early. Most successful traders lose money on 40-50% of their trades — they just make sure the losses are small.")
        elif pnl_pct > -10:
            analysis["summary"] += "Moderate loss. Not devastating, but think about whether you held too long hoping for a recovery."
            analysis["lessons"].append("A common mistake: 'I will sell when I get back to even.' The market does not know your entry price. Sell on signals, not hope.")
        else:
            analysis["summary"] += "Significant loss. This is the kind of trade that teaches the most expensive lessons."
            analysis["lessons"].append("Large losses usually happen because (1) the position was too big, (2) there was no stop-loss plan, or (3) you ignored warning signals hoping for a recovery. Which one was it?")

    analysis["lessons"].append(
        "Every trade is data. Win or lose, the question is: would you make the same trade again with the same information? "
        "If yes, the process was good even if the outcome was bad. If no, write down what you would do differently."
    )

    return analysis

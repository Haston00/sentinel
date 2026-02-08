"""
SENTINEL — Page 15: SENTINEL Academy.
Interactive learning modules that teach you what everything means
using YOUR live data. Not theory — real numbers, real examples.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from config.settings import COLORS
from data.stocks import fetch_ohlcv


def render():
    st.title("SENTINEL Academy")
    st.caption("Learn what all of this means — using live market data, not textbooks")

    surface = COLORS["surface"]
    text_color = COLORS["text"]
    bull = COLORS["bull"]
    bear = COLORS["bear"]
    neutral = COLORS["neutral"]
    primary = COLORS["primary"]
    bg = COLORS["background"]

    # Organized into categories — logical learning progression
    category = st.radio(
        "Category",
        ["Trading School Basics", "Crypto Mastery", "SENTINEL Features", "Advanced & Strategy"],
        horizontal=True,
    )

    if category == "Trading School Basics":
        modules = [
            "1. Stock Market 101 — How It All Works",
            "2. Orders, Brokers & How to Actually Trade",
            "3. Reading Price Charts",
            "4. What Indicators Mean",
            "5. The Fed, Interest Rates & Why They Move Everything",
            "6. Earnings, Dividends & Corporate Events",
            "7. How Markets Connect",
        ]
    elif category == "Crypto Mastery":
        modules = [
            "8. Crypto Fundamentals",
            "9. Blockchain — How the Technology Works",
            "10. Crypto Deep Dive — Wallets, Exchanges & Gas Fees",
            "11. Tokenomics — Supply, Burns & Why Coins Move",
            "12. Major Chains — Ethereum, Solana, and Beyond",
            "13. Reading Crypto Markets — Funding, OI & Whales",
            "14. The Crypto Cycle — BTC Season, Alt Season & Narratives",
            "15. DeFi Masterclass — Pools, Yield & Real Risks",
            "16. On-Chain Analysis — Reading the Blockchain",
            "17. Crypto Security — Scams, Hacks & Protection",
        ]
    elif category == "SENTINEL Features":
        modules = [
            "18. Conviction Scores Explained",
            "19. Probability Forecasts",
            "20. Market Breadth",
            "21. Sentiment & News Intelligence",
            "22. Regimes Explained",
            "23. What AI/ML Does",
            "24. Deep Analysis — Reading a Research Report",
        ]
    else:
        modules = [
            "25. Intermarket Rules",
            "26. Sector Rotation & Business Cycle",
            "27. Volatility — What It Is and Why It Matters",
            "28. Trading Styles — Day vs Swing vs Investing",
            "29. Trading Psychology — Your Biggest Enemy Is You",
            "30. Risk Management — Real Rules",
            "31. Building Your Daily Trading Routine",
            "32. Using SENTINEL — Putting It All Together",
        ]

    st.markdown(
        f"<p style='color:#888;font-size:13px;margin-bottom:5px'>"
        f"32 modules — your complete trading education</p>",
        unsafe_allow_html=True,
    )

    module = st.selectbox("Choose a Module", modules)

    # Load SPY data for examples throughout
    spy = fetch_ohlcv("SPY")
    spy_close = spy["Close"] if not spy.empty else pd.Series(dtype=float)

    # ═══════════════════════════════════════════════════════════
    # TRADING SCHOOL BASICS (1-7)
    if module.startswith("1."):
        _module_stock_market_101(surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("2."):
        _module_orders_brokers(surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("3."):
        _module_price_charts(spy, surface, text_color, bull, bear, primary, bg)
    elif module.startswith("4."):
        _module_indicators(spy, surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("5."):
        _module_fed_rates(surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("6."):
        _module_earnings_dividends(surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("7."):
        _module_markets_connect(surface, text_color, bull, bear, neutral, primary, bg)
    # CRYPTO MASTERY (8-17)
    elif module.startswith("8."):
        _module_crypto(surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("9."):
        _module_blockchain(surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("10."):
        _module_crypto_deep_dive(surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("11."):
        _module_tokenomics(surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("12."):
        _module_major_chains(surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("13."):
        _module_crypto_markets(surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("14."):
        _module_crypto_cycles(surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("15."):
        _module_defi_masterclass(surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("16."):
        _module_onchain_analysis(surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("17."):
        _module_crypto_security(surface, text_color, bull, bear, neutral, primary, bg)
    # SENTINEL FEATURES (18-24)
    elif module.startswith("18."):
        _module_conviction_scores(spy, surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("19."):
        _module_probability(surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("20."):
        _module_breadth(surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("21."):
        _module_sentiment(surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("22."):
        _module_regimes(surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("23."):
        _module_ai_ml(surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("24."):
        _module_deep_analysis(surface, text_color, bull, bear, neutral, primary, bg)
    # ADVANCED & STRATEGY (25-32)
    elif module.startswith("25."):
        _module_intermarket(surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("26."):
        _module_sector_rotation(surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("27."):
        _module_volatility(spy, surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("28."):
        _module_trading_styles(surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("29."):
        _module_psychology(surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("30."):
        _module_risk_management(surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("31."):
        _module_daily_routine(surface, text_color, bull, bear, neutral, primary, bg)
    elif module.startswith("32."):
        _module_using_sentinel(surface, text_color, bull, bear, neutral, primary, bg)


def _card(title, content, color, surface, text_color):
    st.markdown(
        f"<div style='background:{surface};padding:20px;border-radius:12px;"
        f"border-left:4px solid {color};margin-bottom:15px'>"
        f"<h4 style='margin:0 0 8px;color:{color}'>{title}</h4>"
        f"<p style='margin:0;color:{text_color};font-size:14px;line-height:1.7'>{content}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── MODULE 1: How Markets Connect ─────────────────────────────

def _module_markets_connect(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 7: How Markets Connect")
    st.write("Markets are not isolated. Stocks, bonds, gold, oil, and the dollar all influence each other. Understanding these connections is the foundation of everything SENTINEL does.")

    _card("Stocks (S&P 500, Nasdaq)",
          "When you buy a stock, you own a piece of a company. Stock prices go up when people expect the company to make more money in the future, and down when they expect less. The S&P 500 tracks the 500 biggest US companies — it IS the market for most purposes.",
          primary, surface, text_color)

    _card("Bonds (TLT = Long-Term Treasuries)",
          "Bonds are loans to the government. They pay a fixed interest rate. When people get scared, they buy bonds (safe) and sell stocks (risky) — this is called 'flight to safety.' When bond prices go UP, interest rates go DOWN. They move opposite.",
          "#00BCD4", surface, text_color)

    _card("Gold (GLD)",
          "Gold is the oldest safe haven. People buy gold when they are afraid — afraid of inflation, afraid of war, afraid of the economy collapsing. When gold and the dollar BOTH go up, that means serious fear. When gold goes up but the dollar goes down, people are worried about inflation specifically.",
          neutral, surface, text_color)

    _card("The Dollar (UUP)",
          "The US dollar affects everything. A strong dollar hurts US companies that sell overseas (about 40% of S&P 500 revenue). It also squeezes countries that borrowed in dollars. A weak dollar helps commodities and emerging markets. The dollar is like gravity — everything else moves relative to it.",
          "#FF9800", surface, text_color)

    _card("Oil (USO)",
          "Oil is the lifeblood of the economy. When oil spikes, it acts like a tax — everything costs more to transport, manufacture, and deliver. Consumers spend more on gas and less on everything else. Rising oil also forces the Fed to keep interest rates higher to fight inflation.",
          bear, surface, text_color)

    _card("Credit (HYG = Junk Bonds, LQD = Safe Corporate Bonds)",
          "This is the canary in the coal mine. When companies borrow money, riskier companies pay higher interest rates. When the gap between risky and safe bond yields WIDENS, it means lenders are getting nervous about defaults. Credit stress almost always shows up BEFORE stock market crashes.",
          "#9C27B0", surface, text_color)

    st.subheader("The Key Relationships")
    relationships = [
        ("Stocks UP + Bonds DOWN", "Growth optimism — economy strong, rates can rise", bull),
        ("Stocks DOWN + Bonds UP", "Risk-off — people fleeing to safety", bear),
        ("Stocks DOWN + Bonds DOWN", "LIQUIDATION — everything selling. Very rare, very bad", bear),
        ("Gold UP + Dollar UP", "FEAR TRADE — serious worry, buying all safe havens", "#FF9800"),
        ("Gold UP + Dollar DOWN", "Inflation trade — worried about purchasing power", neutral),
        ("Oil UP sharply", "Inflation risk — hurts consumers, forces Fed to stay tight", "#FF9800"),
        ("Credit spreads widening", "Stress building — lenders getting nervous", bear),
    ]
    for rel, meaning, color in relationships:
        st.markdown(
            f"<div style='display:flex;padding:10px;border-bottom:1px solid #333'>"
            f"<span style='width:250px;font-weight:600;color:{color}'>{rel}</span>"
            f"<span style='color:{text_color}'>{meaning}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── MODULE 2: Reading Price Charts ────────────────────────────

def _module_price_charts(spy, surface, text_color, bull, bear, primary, bg):
    st.header("Module 3: Reading Price Charts")
    st.write("A price chart tells a story. Here is SPY (S&P 500 ETF) over the last year. Let me show you how to read it.")

    if spy.empty:
        st.warning("No SPY data available")
        return

    recent = spy.tail(252)
    close = recent["Close"]

    # Chart with moving averages
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=recent.index, open=recent["Open"], high=recent["High"],
        low=recent["Low"], close=recent["Close"], name="SPY",
    ))
    fig.add_trace(go.Scatter(x=recent.index, y=sma50, name="50-day MA",
                             line=dict(color="#FF9800", width=2)))
    fig.add_trace(go.Scatter(x=recent.index, y=sma200, name="200-day MA",
                             line=dict(color="#2196F3", width=2)))
    fig.update_layout(
        title="SPY — Last Year", template="plotly_dark",
        paper_bgcolor=bg, plot_bgcolor=surface, height=450,
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    _card("What are Moving Averages?",
          "The orange line (50-day MA) shows the average price over the last 50 days. The blue line (200-day MA) shows 200 days. When price is ABOVE the moving average, the trend is UP. When it is BELOW, the trend is DOWN. When the 50 crosses ABOVE the 200, it is called a Golden Cross — very bullish. When it crosses BELOW, it is called a Death Cross — bearish.",
          primary, surface, text_color)

    _card("Candlesticks",
          "Each candlestick shows one day. Green means the price went UP that day (closed higher than it opened). Red means it went DOWN. The thin lines (wicks) show how high and how low it went during the day. Long wicks mean there was a fight between buyers and sellers.",
          bull, surface, text_color)

    current = close.iloc[-1]
    ma50_val = sma50.iloc[-1] if not sma50.empty else 0
    ma200_val = sma200.iloc[-1] if not sma200.empty else 0

    if not np.isnan(ma50_val) and not np.isnan(ma200_val):
        above_50 = current > ma50_val
        above_200 = current > ma200_val
        golden = ma50_val > ma200_val

        st.subheader("What SPY Is Telling Us Right Now")
        st.write(f"**Current Price:** ${current:.2f}")
        st.write(f"**50-day MA:** ${ma50_val:.2f} — Price is {'ABOVE' if above_50 else 'BELOW'} ({'bullish' if above_50 else 'bearish'})")
        st.write(f"**200-day MA:** ${ma200_val:.2f} — Price is {'ABOVE' if above_200 else 'BELOW'} ({'bullish' if above_200 else 'bearish'})")
        st.write(f"**Golden Cross Status:** {'YES — 50 MA above 200 MA (bullish)' if golden else 'NO — 50 MA below 200 MA (bearish)'}")


# ── MODULE 3: What Indicators Mean ────────────────────────────

def _module_indicators(spy, surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 4: What Indicators Mean")
    st.write("SENTINEL uses over 40 technical indicators. Here are the most important ones, with live values.")

    if spy.empty:
        st.warning("No data")
        return

    close = spy["Close"]

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_now = rsi.iloc[-1]

    st.subheader(f"RSI (Relative Strength Index): {rsi_now:.0f}")
    if rsi_now > 70:
        rsi_reading = f"OVERBOUGHT ({rsi_now:.0f}). The stock has been rising fast. A pullback is likely soon."
        rsi_color = bear
    elif rsi_now < 30:
        rsi_reading = f"OVERSOLD ({rsi_now:.0f}). The stock has been falling fast. A bounce is likely soon."
        rsi_color = bull
    else:
        rsi_reading = f"NEUTRAL ({rsi_now:.0f}). Not stretched in either direction."
        rsi_color = neutral

    _card("RSI Right Now", rsi_reading, rsi_color, surface, text_color)
    _card("What RSI Measures",
          "RSI measures momentum — how fast and how far a stock has moved recently. It goes from 0 to 100. Above 70 means the stock has gone up too fast too quickly (overbought). Below 30 means it has fallen too fast (oversold). It does NOT predict direction — it predicts when a move is exhausted and likely to reverse or pause.",
          primary, surface, text_color)

    _card("Important: RSI Changes Meaning Based on Trend",
          "SENTINEL does not use RSI blindly. In a strong uptrend, an RSI of 65 is not 'overbought' — it is confirming momentum. In a strong downtrend, an RSI of 35 is not 'oversold' — it is confirming weakness. Only in a sideways market does the classic 'above 70 = sell, below 30 = buy' rule work well. SENTINEL checks the trend first and adjusts the RSI reading accordingly.",
          "#FF9800", surface, text_color)

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    macd_hist = macd - signal
    macd_now = macd.iloc[-1]
    signal_now = signal.iloc[-1]

    st.subheader(f"MACD: {macd_now:.2f} (Signal: {signal_now:.2f})")
    if macd_now > signal_now:
        macd_reading = "MACD is ABOVE the signal line — momentum is bullish. The trend has upward energy."
        macd_color = bull
    else:
        macd_reading = "MACD is BELOW the signal line — momentum is bearish. The trend has downward energy."
        macd_color = bear

    _card("MACD Right Now", macd_reading, macd_color, surface, text_color)
    _card("What MACD Measures",
          "MACD measures the DIFFERENCE between a fast moving average (12-day) and a slow one (26-day). When the fast MA is above the slow MA, momentum is positive — the stock is accelerating upward. The signal line (9-day average of MACD) acts as a trigger. When MACD crosses above the signal, many traders take that as a buy signal.",
          primary, surface, text_color)

    # Volume
    vol_avg = spy["Volume"].rolling(20).mean().iloc[-1]
    vol_today = spy["Volume"].iloc[-1]
    vol_ratio = vol_today / vol_avg if vol_avg > 0 else 1

    st.subheader(f"Volume: {vol_today/1e6:.0f}M (Avg: {vol_avg/1e6:.0f}M)")
    _card("Why Volume Matters",
          f"Today's volume is {vol_ratio:.1f}x the 20-day average. Volume confirms moves. A price jump on HIGH volume is real — big money is behind it. A price jump on LOW volume is suspect — it could reverse easily. Think of volume as conviction. High volume = the crowd agrees. Low volume = nobody cares.",
          primary, surface, text_color)

    _card("Volume Direction Matters Too",
          "SENTINEL does not just look at whether volume is high or low. It looks at which DIRECTION the price moved on that volume. High volume on an UP day is bullish — buyers are aggressive. High volume on a DOWN day is bearish — sellers are dumping. Same volume number, completely different meaning depending on the direction.",
          "#FF9800", surface, text_color)

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = (sma20 + 2 * std20).iloc[-1]
    bb_lower = (sma20 - 2 * std20).iloc[-1]
    bb_width = ((bb_upper - bb_lower) / sma20.iloc[-1]) * 100
    price = close.iloc[-1]

    bb_pos = (price - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50

    st.subheader(f"Bollinger Bands: Price at {bb_pos:.0f}% of range")
    _card("What Bollinger Bands Show",
          f"Price: ${price:.2f} | Upper Band: ${bb_upper:.2f} | Lower Band: ${bb_lower:.2f}. "
          f"Bollinger Bands draw a channel 2 standard deviations above and below the 20-day average. "
          f"Price near the upper band ({bb_pos:.0f}%) means stretched high. Near the lower band means stretched low. "
          f"Band width ({bb_width:.1f}%) measures volatility — narrow bands mean low volatility, and a big move is coming. Wide bands mean high volatility already happened.",
          primary, surface, text_color)

    # ADX
    st.subheader("Other Key Indicators SENTINEL Uses")
    _card("ADX (Average Directional Index)",
          "ADX measures how STRONG a trend is, not which direction. It goes from 0 to 100. Below 20 means no trend — the market is choppy and sideways. Above 25 means a real trend is forming. Above 40 means a very strong trend. SENTINEL uses ADX to decide whether to trust trend signals — if ADX is low, trend signals are unreliable because there is no trend to follow.",
          primary, surface, text_color)

    _card("ATR (Average True Range)",
          "ATR measures how much a stock moves per day on average. If SPY has an ATR of $5, that means it typically moves about $5 per day. This matters for setting stop losses — if you put your stop $2 away on a stock with $5 ATR, you will get stopped out by normal daily noise. SENTINEL uses ATR to size positions and set realistic targets.",
          primary, surface, text_color)

    _card("OBV (On-Balance Volume)",
          "OBV keeps a running total: adds volume on up days, subtracts volume on down days. If OBV is trending up while price is flat, it means big money is quietly accumulating shares — bullish. If OBV is trending down while price is flat, smart money is distributing (selling) — bearish. OBV often moves BEFORE price does.",
          primary, surface, text_color)

    _card("Stochastic Oscillator",
          "Similar to RSI but measures where the closing price is relative to the high-low range over 14 days. Above 80 = overbought, below 20 = oversold. The twist: it has two lines (K and D). When %K crosses above %D in oversold territory, that is a buy signal. When %K crosses below %D in overbought territory, that is a sell signal.",
          primary, surface, text_color)


# ── MODULE 4: Market Breadth ──────────────────────────────────

def _module_breadth(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 20: Market Breadth")
    st.write("Breadth answers the question: is the WHOLE market moving, or just a few big stocks?")

    _card("What Is Breadth?",
          "The S&P 500 is 'weighted' — Apple, Microsoft, and Nvidia have a huge impact on the index. The index could go UP even if most stocks are going DOWN, just because those few giants are rising. Breadth measures what the AVERAGE stock is doing. If the index is at new highs but most stocks are falling, that is a DIVERGENCE — and it is one of the most reliable warning signs in the market.",
          primary, surface, text_color)

    _card("% Above 50-Day MA",
          "This tells you what percentage of stocks are above their 50-day moving average. Above 60% = healthy, broad participation. Below 40% = weak, narrow market. Below 25% = extremely oversold (often a bounce is coming).",
          bull, surface, text_color)

    _card("% Above 200-Day MA",
          "Same idea but using the 200-day MA, which captures the LONG-term trend. If this drops below 50%, the majority of stocks are in long-term downtrends. That is a bear market under the surface, even if the index looks okay.",
          "#00BCD4", surface, text_color)

    _card("Advance/Decline Ratio",
          "Simply: how many stocks went UP today divided by how many went DOWN. Above 2.0 = very strong buying. Below 0.5 = very strong selling. A 'breadth thrust' (ratio above 3.0) is one of the most powerful buy signals in existence — it means buying is so broad that it overwhelms all selling.",
          bull, surface, text_color)

    _card("Why Breadth Divergences Matter",
          "In 2021, the S&P 500 kept making new highs, but the percentage of stocks above their 50-day MA was falling. The index was being propped up by a handful of mega-cap tech stocks. When those stocks finally broke, the whole market crashed. Breadth warned you 3 months early. SENTINEL checks for this automatically.",
          bear, surface, text_color)


# ── MODULE 5: Intermarket Rules ───────────────────────────────

def _module_intermarket(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 25: Intermarket Rules")
    st.write("These are the relationships between asset classes that institutional traders watch. When you understand these, you are thinking like a professional.")

    rules = [
        ("Gold UP + Dollar UP", "Fear Trade", "Both safe havens rising = serious worry. Stocks usually fall within 1-3 months. This happened before 2008 crash and 2020 crash.", bear),
        ("Gold UP + Dollar DOWN", "Inflation Trade", "Gold rising while dollar weakens = inflation fear. Good for commodities, bad for bonds. Mixed for stocks — depends on the sector.", neutral),
        ("Stocks UP + Bonds DOWN", "Growth Optimism", "Money rotating OUT of safety INTO risk. The market believes the economy is strong. Bullish.", bull),
        ("Stocks DOWN + Bonds UP", "Risk-Off", "Classic flight to safety. People are scared and parking money in government bonds. Defensive sectors outperform.", bear),
        ("Stocks DOWN + Bonds DOWN", "Liquidation", "The scariest signal. When EVERYTHING sells, it means either a liquidity crisis or the Fed is squeezing too hard. Very rare, very bearish.", bear),
        ("Credit Spreads Widening", "Stress Building", "When junk bond yields spike vs safe bonds, lenders smell trouble. The credit market is usually right 6-12 months before the stock market figures it out.", bear),
        ("Oil UP >15%/month", "Consumer Tax", "Sharp oil spikes act like a tax on the economy. Consumers spend more on gas, less on everything else. Hurts corporate margins. Forces Fed hawkish.", "#FF9800"),
        ("Dollar UP >3%/month", "Headwind", "A surging dollar hurts S&P 500 earnings (40% overseas revenue), squeezes emerging markets, and tightens global financial conditions.", "#FF9800"),
        ("VIX Spike + RSI Oversold", "Capitulation", "Maximum fear + oversold prices = contrarian buy signal. The crowd is panicking, which historically is the worst time to sell and the best time to buy.", bull),
        ("Breadth Thrust (A/D >3)", "Rare Bullish", "When buying is SO broad that the A/D ratio exceeds 3, the market rallied over the next 6 months 92% of the time historically. One of the most reliable signals.", bull),
    ]

    for name, label, explanation, color in rules:
        st.markdown(
            f"<div style='background:{surface};padding:18px;border-radius:10px;margin-bottom:12px;"
            f"border-left:4px solid {color}'>"
            f"<p style='margin:0;font-weight:700;color:{color};font-size:15px'>{name}</p>"
            f"<p style='margin:3px 0;color:{text_color};font-weight:600;font-size:13px'>{label}</p>"
            f"<p style='margin:5px 0 0;color:#aaa;font-size:13px;line-height:1.6'>{explanation}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── MODULE 6: What AI/ML Does ─────────────────────────────────

def _module_ai_ml(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 23: What AI/ML Does")
    st.write("SENTINEL uses real machine learning — not just if/then rules. Here is why that matters.")

    _card("The Problem with Human Analysis",
          "A human can look at maybe 5-10 indicators at once. But what about the interaction between RSI AND MACD AND volume AND Bollinger Bands AND 37 other features simultaneously? You cannot hold all of that in your head. And even if you could, your biases (fear, greed, recency) would distort your judgment.",
          "#FF9800", surface, text_color)

    _card("What XGBoost Does",
          "XGBoost (Extreme Gradient Boosting) is a machine learning algorithm used by quant funds and Kaggle competition winners. It takes all 41 features and builds hundreds of decision trees. Each tree asks simple questions like 'Is RSI above 70 AND is volume below average AND is MACD negative?' and then combines thousands of these rules to make a prediction. It finds patterns no human could see.",
          primary, surface, text_color)

    _card("What Random Forest Does",
          "Random Forest is another algorithm that builds many decision trees, but each tree only looks at a random subset of features. This makes it more robust — it is less likely to overfit (memorize the past instead of learning real patterns). By combining XGBoost and Random Forest, we get a more reliable prediction than either alone.",
          primary, surface, text_color)

    _card("The Ensemble — Why We Use Multiple Models",
          "SENTINEL does not rely on one model. It runs XGBoost, Random Forest, ARIMA (time series), GARCH (volatility), and a factor model. Then it combines them using an adaptive ensemble — a meta-model that learns which model performs best in the CURRENT market regime. In a trending bull market, the trend-following models get more weight. In a choppy market, the mean-reversion models get more weight. The ensemble adapts in real time.",
          primary, surface, text_color)

    _card("Walk-Forward Testing (Why Our Numbers Are Honest)",
          "Many people 'backtest' by training on ALL the data and then checking accuracy on the same data. That is cheating — of course it looks good, it memorized the answers. We do WALK-FORWARD testing: train on the first 80% of data, then test on the remaining 20% the model has NEVER seen. This gives you the REAL accuracy — what would have happened if you used this model in real time.",
          bull, surface, text_color)

    _card("Why 45% Accuracy Can Be Profitable",
          "In a 3-class problem (UP, DOWN, FLAT), random guessing gets 33%. So even 45% accuracy means the model has an EDGE. If the model is right 45% of the time but picks bigger winners than losers, it makes money. This is exactly how quantitative trading firms operate — small edge, applied many times.",
          bull, surface, text_color)

    _card("Feature Importance = What the AI Thinks Matters",
          "After training, we can ask the model: which features were most useful for making predictions? This is feature importance. If the model says RSI matters most, that tells you momentum is the dominant force right now. If it says volume matters most, the market is being driven by institutional flows. This changes over time — and the model adapts.",
          primary, surface, text_color)

    _card("What AI Cannot Do",
          "AI models trained on historical data CANNOT predict black swans (unprecedented events like COVID, 9/11, or banking crises). They also degrade when market structure changes fundamentally. This is why SENTINEL also uses regime detection and intermarket rules — the AI is one tool among many, not a crystal ball.",
          bear, surface, text_color)


# ── MODULE 7: Regimes Explained ───────────────────────────────

def _module_regimes(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 22: Regimes Explained")
    st.write("The same trading strategy does NOT work in every market. What works in a bull market can destroy you in a bear market. Regime detection tells you which rules apply RIGHT NOW.")

    _card("What Is a Regime?",
          "A market regime is the overall 'mode' the market is in. Think of it like weather: sometimes it is sunny (bull), sometimes stormy (bear), sometimes cloudy and uncertain (transition). You would not wear the same clothes in a storm that you wear on a sunny day. Same with investing — you need different strategies for different regimes.",
          primary, surface, text_color)

    _card("BULL Regime",
          "Low volatility, positive returns, rising prices. The trend is up and corrections are short and shallow. Strategy: be aggressive, buy dips, favor growth and cyclical stocks. Risk: complacency. Bull markets end when everyone thinks they cannot end.",
          bull, surface, text_color)

    _card("BEAR Regime",
          "High volatility, negative or flat returns, falling prices. Rallies are short and sell off quickly. Strategy: reduce exposure, raise cash, favor defensive sectors (utilities, healthcare), consider gold. Risk: catching falling knives. Wait for confirmation before buying.",
          bear, surface, text_color)

    _card("TRANSITION Regime",
          "Mixed signals, rising uncertainty. The market is deciding which way to go. This is often the hardest environment to trade. Strategy: patience, smaller positions, focus only on highest-conviction ideas. Risk: getting whipsawed — buying right before a drop or selling right before a rally.",
          neutral, surface, text_color)

    _card("How the HMM Detects Regimes",
          "A Hidden Markov Model (HMM) is a statistical model that assumes there are hidden states (regimes) that we cannot directly observe. What we CAN observe are returns, volatility, VIX, bond prices, and credit spreads. The HMM learns the statistical 'fingerprint' of each regime from 5 years of history. Then it looks at today's data and says: based on what I see, there is an X% chance we are in Bull, Y% chance in Bear, Z% chance in Transition. It is not perfect — no model is. But it removes human emotion from the assessment.",
          primary, surface, text_color)

    _card("Why Regime Probabilities Are Not 100%",
          "You will never see SENTINEL say '100% Bull' or '100% Bear.' That would be dishonest. The model always assigns some probability to each regime because the future is uncertain. If it says 75% Bull, 15% Transition, 10% Bear — that means conditions strongly favor a bull market, but there is still meaningful uncertainty. The probabilities change every day as new data comes in. A gradual shift from Bull toward Transition is often the first warning sign that the good times are ending.",
          "#FF9800", surface, text_color)


# ── MODULE 8: Crypto Fundamentals ─────────────────────────────

def _module_crypto(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 8: Crypto Fundamentals")
    st.write("Crypto is a different animal from stocks. Different rules, different metrics, different risks. Here is what you need to know.")

    _card("How Crypto Is Different From Stocks",
          "When you buy Apple stock, you own a piece of a company with revenue, profits, and employees. When you buy Bitcoin, you own a piece of a decentralized network. There are no earnings reports, no PE ratios, no dividends. Crypto trades 24/7/365 — there is no closing bell. Volatility is 3-5x higher than stocks. A 10% move in a day is unusual for stocks but totally normal for crypto.",
          primary, surface, text_color)

    _card("Bitcoin (BTC) — Digital Gold",
          "Bitcoin is the original cryptocurrency and the largest by market cap. It has a fixed supply of 21 million coins — no government or company can print more. This scarcity is why people call it 'digital gold.' Bitcoin drives the entire crypto market. When BTC rallies, almost everything else rallies. When BTC crashes, almost everything else crashes harder. Always check BTC first.",
          "#FF9800", surface, text_color)

    _card("Ethereum (ETH) — The Platform",
          "Ethereum is not just a coin — it is a platform where other projects build. Think of BTC as gold and ETH as the internet. Most DeFi (decentralized finance), NFTs, and smart contracts run on Ethereum. When the crypto ecosystem is growing, ETH often outperforms BTC because more activity means more demand for the platform.",
          "#00BCD4", surface, text_color)

    _card("Altcoins (SOL, ADA, AVAX, etc.)",
          "Everything that is not BTC or ETH is called an altcoin. Altcoins are higher risk and higher reward. In a bull market, altcoins can 5-10x while BTC does 2-3x. But in a crash, altcoins can lose 80-90% while BTC might lose 50%. The rule: altcoins amplify whatever BTC is doing. They go up more AND down more.",
          neutral, surface, text_color)

    st.subheader("Crypto-Specific Metrics SENTINEL Tracks")

    _card("Market Cap vs. Volume Ratio",
          "Market cap is the total value of all coins (price x supply). Daily volume is how much trades in 24 hours. If a coin has $10 billion market cap but only $100 million in daily volume, it is illiquid — a few big sellers can crash the price. SENTINEL flags coins where the volume-to-market-cap ratio is unusually low or high.",
          primary, surface, text_color)

    _card("Drawdown From All-Time High",
          "How far is the coin from its highest price ever? BTC at $60,000 when its ATH was $69,000 is a 13% drawdown. A coin at 80% drawdown from ATH is either a bargain or dying — you need other signals to tell which. SENTINEL tracks this for every coin and compares it to historical patterns.",
          primary, surface, text_color)

    _card("Volatility Regimes (7d vs 30d vs 90d)",
          "SENTINEL compares short-term volatility (7 days) to medium-term (30 days) to long-term (90 days). When 7-day volatility is much higher than 90-day, the coin is in a 'hot' phase — big moves happening now. When 7-day is lower than 90-day, the coin is quiet — a breakout is building. Think of it like a coiled spring.",
          primary, surface, text_color)

    _card("The Halving Cycle",
          "Every 4 years, Bitcoin's mining reward is cut in half — this is called the halving. Historically, BTC has rallied significantly in the 12-18 months after each halving (2012, 2016, 2020). The theory: less new supply + same or growing demand = price goes up. This is not guaranteed to work forever, but it has worked three times in a row.",
          bull, surface, text_color)

    st.subheader("Crypto Dangers")

    _card("Rug Pulls",
          "A rug pull is when the creators of a crypto project take all the money and disappear. Common in small altcoins and new DeFi projects. Warning signs: anonymous team, unrealistic promises, locked liquidity that suddenly unlocks. Stick to established coins (BTC, ETH, SOL, top 20 by market cap) unless you truly understand the project.",
          bear, surface, text_color)

    _card("Leverage and Liquidation",
          "Many crypto exchanges let you trade with 10x, 50x, even 100x leverage. This means you can control $100,000 worth of BTC with just $1,000. Sounds amazing until a 1% move against you wipes out your entire position. This is called liquidation. Billions of dollars in leveraged positions get liquidated during big moves, which makes the crash even worse. SENTINEL does not use leverage signals, but cascading liquidations cause the flash crashes you see.",
          bear, surface, text_color)

    _card("Correlation With Stocks",
          "Crypto used to trade independently from stocks. That changed in 2020. Now BTC and ETH are highly correlated with the Nasdaq. When the Fed raises rates and tech stocks sell off, crypto usually sells off too. SENTINEL monitors this correlation — when crypto starts decorrelating from stocks, that is a signal that crypto-specific forces are taking over.",
          "#FF9800", surface, text_color)


# ── MODULE 9: Sentiment & News Intelligence ───────────────────

def _module_sentiment(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 21: Sentiment & News Intelligence")
    st.write("Markets move on news and emotions. SENTINEL reads thousands of articles and scores them automatically so you do not have to.")

    _card("What Is Sentiment Analysis?",
          "Sentiment analysis is using a computer to read text and determine if it is positive, negative, or neutral. When a headline says 'Markets surge on strong jobs data,' that is positive. When it says 'Stocks plunge as recession fears mount,' that is negative. SENTINEL reads headlines from Reuters, Bloomberg, CNBC, WSJ, crypto news sites, Reddit, and global event databases — then scores each one.",
          primary, surface, text_color)

    _card("How VADER Works",
          "SENTINEL uses an algorithm called VADER (Valence Aware Dictionary for Sentiment Reasoning). It looks at each word in a sentence and assigns it a score. 'Surge' is very positive. 'Crash' is very negative. 'Market' is neutral. Then it combines all the word scores into one number from -1.0 (extremely negative) to +1.0 (extremely positive). Anything between -0.05 and +0.05 is considered neutral.",
          primary, surface, text_color)

    _card("Custom Financial Vocabulary",
          "Regular VADER does not know financial language. The word 'bearish' means nothing to a generic sentiment tool. SENTINEL adds 87 custom financial and crypto terms. 'Bullish' scores +2.0. 'Rug pull' scores -3.0. 'ETF approved' scores +2.5. 'Rate hike' scores -0.8. This makes the sentiment scores much more accurate for market-related text.",
          bull, surface, text_color)

    _card("Source Credibility Weighting",
          "Not all news is equal. A Reuters article is more reliable than a random blog post. SENTINEL weights sources by credibility. Tier 1 (Reuters, Bloomberg, WSJ) get a 1.5x multiplier — their sentiment counts more. Tier 2 (AP, BBC, NYT) get 1.2x. Unknown sources get 0.8x — a discount. This prevents low-quality sources from skewing the signal.",
          "#FF9800", surface, text_color)

    _card("Recency Weighting",
          "A news article from 2 hours ago matters more than one from 2 days ago. SENTINEL uses exponential decay with a 24-hour half-life. An article from right now gets full weight. One from 24 hours ago gets half weight. One from 48 hours ago gets a quarter weight. This means the sentiment score always reflects CURRENT conditions, not old news.",
          "#FF9800", surface, text_color)

    _card("Entity Extraction — What Is the News About?",
          "SENTINEL does not just score sentiment — it figures out WHO the news is about. It extracts stock tickers ($AAPL, $TSLA), sectors (Technology, Healthcare), and crypto coins (Bitcoin, Ethereum, Solana) from each headline. This lets it route the sentiment to the right place. A negative article about banks only affects the Financials sector score, not Technology.",
          primary, surface, text_color)

    _card("Urgency Classification",
          "Not all news has the same market impact. SENTINEL classifies each article as HIGH, MEDIUM, or LOW urgency. Words like 'crash,' 'emergency,' 'halt,' and 'bankruptcy' trigger HIGH urgency. Words like 'rally,' 'earnings,' and 'forecast' trigger MEDIUM. Everything else is LOW. High-urgency articles get extra attention in the dashboard.",
          bear, surface, text_color)

    st.subheader("News Sources SENTINEL Monitors")
    sources = [
        ("GDELT", "Global events database — updated every 15 minutes, covers events in 65 languages worldwide"),
        ("NewsAPI", "Financial headlines from 80,000+ news sources, filtered for market relevance"),
        ("RSS Feeds", "Direct feeds from Federal Reserve, Reuters, Bloomberg, CNBC, WSJ, ECB"),
        ("Crypto RSS", "CoinDesk, CoinTelegraph, Decrypt, The Block, Bitcoin Magazine"),
        ("Reddit", "r/wallstreetbets, r/cryptocurrency, r/stocks, r/economics, r/investing"),
    ]
    for name, desc in sources:
        st.markdown(
            f"<div style='padding:8px 15px;border-bottom:1px solid #333;display:flex'>"
            f"<span style='width:120px;font-weight:600;color:{primary}'>{name}</span>"
            f"<span style='color:{text_color};font-size:13px'>{desc}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── MODULE 10: Conviction Scores Explained ────────────────────

def _module_conviction_scores(spy, surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 18: Conviction Scores Explained")
    st.write("SENTINEL gives every stock and ETF a score from 0 to 100. Here is exactly what that number means and how it is built.")

    _card("The Big Picture",
          "The conviction score combines FIVE different signal categories into one number. A score of 75 does not mean 'the stock will go up 75%.' It means: across all five categories, the evidence is strongly bullish. A score of 25 means the evidence is strongly bearish. Around 50 is neutral — signals are mixed or weak.",
          primary, surface, text_color)

    st.subheader("The Five Components")

    components = [
        ("Trend (25% weight)", "Is the price trending up or down?",
         "Looks at moving average alignment — is the price above the 20, 50, and 200-day averages? Are the shorter MAs above the longer ones (bullish alignment)? Score ranges from 15 (all bearish) to 85 (all bullish).",
         primary),
        ("Momentum (25% weight)", "Is the move accelerating or decelerating?",
         "Combines RSI and MACD. But here is the key: SENTINEL makes RSI context-aware. In a strong uptrend, RSI of 65 scores as 'confirming momentum' (mildly bullish), not 'approaching overbought' (bearish). The trend is checked FIRST, then RSI is interpreted through that lens.",
         primary),
        ("Volume (20% weight)", "Is big money confirming the move?",
         "Compares today's volume to the 20-day average AND checks the direction. Volume 2x above average on an up day scores 85 (very bullish — institutions are buying). The same 2x volume on a down day scores 15 (very bearish — institutions are selling). Low volume scores around 40 regardless of direction.",
         primary),
        ("Volatility (15% weight)", "Is the price stable or chaotic?",
         "Uses Bollinger Band width and position. When price is at the upper band in a tight range, it is stretched. When volatility is very low (tight bands), a big move is coming but the direction is uncertain. Moderate, declining volatility in an uptrend scores highest.",
         primary),
        ("Mean Reversion (15% weight)", "Is the price stretched too far?",
         "Checks how far price has moved from its average. Extreme moves tend to snap back. If the stock is very far above its 20-day average, this component scores bearish (expects pullback). If very far below, it scores bullish (expects bounce).",
         primary),
    ]

    for name, question, explanation, color in components:
        st.markdown(
            f"<div style='background:{surface};padding:18px;border-radius:10px;margin-bottom:12px;"
            f"border-left:4px solid {color}'>"
            f"<p style='margin:0;font-weight:700;color:{color};font-size:15px'>{name}</p>"
            f"<p style='margin:3px 0;color:{neutral};font-style:italic;font-size:13px'>{question}</p>"
            f"<p style='margin:5px 0 0;color:{text_color};font-size:13px;line-height:1.6'>{explanation}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.subheader("How to Read the Score")
    ranges = [
        ("80-100", "Extremely Bullish", "Nearly all signals aligned bullish. Very rare. High conviction long.", bull),
        ("65-79", "Bullish", "Most signals positive. Good setup for a long position with normal sizing.", bull),
        ("55-64", "Mildly Bullish", "Slight bullish lean but not strong. Proceed with caution, smaller position.", bull),
        ("45-54", "Neutral", "Signals are mixed or absent. No clear edge. Best to wait.", neutral),
        ("35-44", "Mildly Bearish", "Slight bearish lean. Consider reducing exposure or hedging.", bear),
        ("20-34", "Bearish", "Most signals negative. Strong case for caution or short positions.", bear),
        ("0-19", "Extremely Bearish", "Nearly all signals aligned bearish. Very rare. Maximum caution.", bear),
    ]
    for score_range, label, meaning, color in ranges:
        st.markdown(
            f"<div style='display:flex;padding:10px;border-bottom:1px solid #333;align-items:center'>"
            f"<span style='width:80px;font-weight:700;color:{color};font-size:15px'>{score_range}</span>"
            f"<span style='width:150px;font-weight:600;color:{color}'>{label}</span>"
            f"<span style='color:{text_color};font-size:13px'>{meaning}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _card("Key Signals List",
          "Below the score, SENTINEL lists the key signals driving it. Things like 'Golden Cross active,' 'RSI confirms uptrend momentum,' or 'High volume selling.' These tell you WHY the score is what it is. Always read the signals, not just the number — two stocks with the same score might have very different reasons behind it.",
          "#FF9800", surface, text_color)


# ── MODULE 11: Probability Forecasts ──────────────────────────

def _module_probability(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 19: Probability Forecasts")
    st.write("SENTINEL does not just say 'up' or 'down.' It gives you probabilities. Here is how to read them.")

    _card("Why Probabilities Instead of Predictions?",
          "Nobody can predict the market with certainty. Anyone who tells you they can is lying. What you CAN do is estimate the odds. If there is a 65% chance the S&P 500 goes up this month and a 35% chance it goes down, that is useful. You do not need to be right every time — you need to be right MORE than the odds imply over many trades.",
          primary, surface, text_color)

    _card("How SENTINEL Calculates Probabilities",
          "SENTINEL combines 14 different intermarket rules, technical signals, and historical patterns. Each one contributes to the probability. For example: if the yield curve is steepening (bullish), credit spreads are tightening (bullish), AND breadth is strong (bullish), the probability of an up move increases. If those same signals conflict with each other, probabilities move toward 50/50 (uncertain).",
          primary, surface, text_color)

    _card("Confidence Levels",
          "Every probability comes with a confidence level. A 60% UP probability with HIGH confidence means multiple independent signals agree. A 60% UP probability with LOW confidence means only one or two weak signals point that direction. Same probability, very different conviction. SENTINEL caps short-term forecasts (1 day, 1 week) at 45% confidence because short-term markets are basically random noise.",
          "#FF9800", surface, text_color)

    _card("Time Horizons Matter",
          "SENTINEL gives forecasts for multiple timeframes: 1 day, 1 week, 1 month, 3 months, and longer. Short-term forecasts are MUCH less reliable than longer-term ones. A 1-day forecast is barely better than a coin flip because daily moves are dominated by random noise. A 3-month forecast can be quite powerful because trends, fundamentals, and macro forces have time to play out.",
          primary, surface, text_color)

    _card("Price Target Bands",
          "Instead of saying 'the stock will hit $500,' SENTINEL gives a range: 'There is a 70% chance the price will be between $480 and $520 in one month.' The range is wider for longer timeframes and more volatile assets. SENTINEL uses a Student-t distribution (fat tails) instead of a normal bell curve — this means it accounts for the fact that extreme moves happen more often in financial markets than simple math would predict.",
          primary, surface, text_color)

    st.subheader("How to Use Probability Forecasts")

    _card("Rule 1: Ignore Daily Forecasts for Trading Decisions",
          "The 1-day forecast is there for awareness only. It is too noisy to trade on. Focus on the weekly and monthly forecasts for actual decisions.",
          "#FF9800", surface, text_color)

    _card("Rule 2: Look at Where Multiple Timeframes Agree",
          "If the 1-week, 1-month, AND 3-month forecasts all say bullish, that is much stronger than just one timeframe. When all timeframes align, the signal is most powerful.",
          bull, surface, text_color)

    _card("Rule 3: Wide Bands Mean Uncertainty",
          "If the price target band is very wide (say $450-$550 on a $500 stock), the model is uncertain. Wide bands mean: there is a real chance of a big move in either direction. Narrow bands mean the model is more confident about the range.",
          neutral, surface, text_color)

    _card("Rule 4: Analyst Price Targets Are Biased",
          "SENTINEL shows Wall Street analyst targets as a reference, but flags them as 'biased 5-15% bullish.' Analysts almost always have higher targets than reality because their firms want you to buy stocks. Use them as one data point, not gospel.",
          bear, surface, text_color)


# ── MODULE 12: Sector Rotation & Business Cycle ───────────────

def _module_sector_rotation(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 26: Sector Rotation & Business Cycle")
    st.write("The economy moves in cycles. Different sectors lead at different phases. Professionals rotate their money through sectors to stay ahead of the cycle.")

    _card("What Is the Business Cycle?",
          "The economy does not grow in a straight line. It goes through repeating phases: recovery, expansion, slowdown, and recession. Each phase lasts months to years. The stock market LEADS the economy — it starts going up before the recession ends and starts going down before the expansion ends. Understanding where we are in the cycle tells you where to put your money.",
          primary, surface, text_color)

    st.subheader("The Four Phases")

    phases = [
        ("Early Recovery", "Economy coming out of recession",
         "Interest rates are falling, Fed is cutting, unemployment peaked and starting to drop. This is when you want to be aggressive. The market has already bottomed and the rally is underway, but most people are still scared.",
         "Technology, Consumer Discretionary, Industrials, Real Estate",
         "Utilities, Consumer Staples, Healthcare",
         bull),
        ("Mid Cycle (Expansion)", "Economy growing steadily",
         "Job growth is strong, corporate earnings are rising, consumer confidence is high. This is the longest phase and when most money is made. The market rises broadly.",
         "Technology, Communication Services, Industrials, Materials",
         "Utilities (boring when growth is easy)",
         bull),
        ("Late Cycle", "Economy overheating",
         "Inflation is rising, the Fed is raising rates, wages are up but so are costs. Profit margins start getting squeezed. This is when the smart money starts getting cautious. The market may still go up, but it is living on borrowed time.",
         "Energy, Materials, Healthcare, Consumer Staples",
         "Technology, Consumer Discretionary, Real Estate",
         "#FF9800"),
        ("Recession", "Economy contracting",
         "GDP is falling, layoffs are rising, corporate earnings are dropping. Fear dominates. This is painful but also where the NEXT bull market is born. Cash and defensive sectors are king.",
         "Utilities, Healthcare, Consumer Staples, Gold",
         "Everything cyclical — Industrials, Materials, Consumer Discretionary",
         bear),
    ]

    for name, subtitle, explanation, leaders, laggards, color in phases:
        st.markdown(
            f"<div style='background:{surface};padding:20px;border-radius:12px;margin-bottom:15px;"
            f"border-left:4px solid {color}'>"
            f"<h4 style='margin:0 0 3px;color:{color}'>{name}</h4>"
            f"<p style='margin:0 0 8px;color:#888;font-size:12px'>{subtitle}</p>"
            f"<p style='margin:0 0 10px;color:{text_color};font-size:13px;line-height:1.6'>{explanation}</p>"
            f"<p style='margin:0;font-size:12px'>"
            f"<span style='color:{bull}'>Leaders: {leaders}</span><br>"
            f"<span style='color:{bear}'>Laggards: {laggards}</span>"
            f"</p></div>",
            unsafe_allow_html=True,
        )

    _card("How SENTINEL Detects the Cycle Phase",
          "SENTINEL looks at multiple macro indicators: the yield curve slope (steepening vs flattening), PMI (manufacturing activity), CPI acceleration (inflation direction), unemployment claims, and credit spreads. It compares current readings to historical thresholds for each phase. It also uses sector momentum — which sectors are leading tells you where in the cycle the market THINKS we are.",
          primary, surface, text_color)

    _card("Relative Momentum",
          "SENTINEL measures each sector's momentum over 1, 3, and 6 months. Sectors that are outperforming the S&P 500 have positive relative momentum — money is flowing in. Sectors that are underperforming have negative relative momentum — money is flowing out. The shift in leadership often happens BEFORE the economic data confirms the cycle change.",
          primary, surface, text_color)

    _card("The Copper/Gold Ratio Trick",
          "Copper is used in construction and manufacturing — it rises when the economy is growing. Gold is a fear asset — it rises when people are worried. The copper/gold ratio is one of the best real-time indicators of economic health. Ratio rising = economy strengthening (favor cyclicals). Ratio falling = economy weakening (favor defensives). SENTINEL tracks this on the Intermarket page.",
          "#FF9800", surface, text_color)


# ── MODULE 13: Volatility ─────────────────────────────────────

def _module_volatility(spy, surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 27: Volatility — What It Is and Why It Matters")
    st.write("Volatility is probably the most misunderstood concept in investing. It is not just 'how much the stock moves.' It is the foundation of risk management.")

    _card("What Is Volatility?",
          "Volatility measures how much a price bounces around. A stock that moves 1% per day has low volatility. A stock that moves 5% per day has high volatility. Volatility is NOT direction — a stock can be very volatile and go nowhere, or have low volatility and trend steadily upward. Think of volatility as the SIZE of the moves, not the direction.",
          primary, surface, text_color)

    _card("VIX — The Fear Index",
          "The VIX measures how much volatility the options market EXPECTS in the S&P 500 over the next 30 days. Below 15 = calm, nobody worried. 15-20 = normal. 20-30 = elevated concern. Above 30 = fear. Above 40 = panic. The VIX usually spikes when stocks DROP because fear causes people to buy protective options. SENTINEL monitors VIX as a key input to regime detection.",
          "#FF9800", surface, text_color)

    if not spy.empty:
        close = spy["Close"]
        # Calculate realized volatility
        returns = close.pct_change().dropna()
        vol_20d = returns.rolling(20).std() * np.sqrt(252) * 100
        vol_now = vol_20d.iloc[-1] if len(vol_20d) > 0 else 0

        if not np.isnan(vol_now):
            if vol_now < 12:
                vol_reading = f"VERY LOW ({vol_now:.1f}%). Markets are unusually calm. A big move may be building."
                vol_color = neutral
            elif vol_now < 18:
                vol_reading = f"NORMAL ({vol_now:.1f}%). Standard market conditions."
                vol_color = bull
            elif vol_now < 25:
                vol_reading = f"ELEVATED ({vol_now:.1f}%). Above average uncertainty. Be cautious with position sizes."
                vol_color = "#FF9800"
            else:
                vol_reading = f"HIGH ({vol_now:.1f}%). Significant fear in the market. Reduce exposure or hedge."
                vol_color = bear

            st.subheader(f"SPY Realized Volatility Right Now: {vol_now:.1f}%")
            _card("Current Reading", vol_reading, vol_color, surface, text_color)

    _card("Realized vs. Implied Volatility",
          "Realized volatility is what actually happened — how much the stock moved over the last 20 days. Implied volatility is what the market EXPECTS to happen — it comes from options prices. When implied is much higher than realized, the market is pricing in a big move that has not happened yet. When implied is lower than realized, the market thinks the wild moves are over. SENTINEL tracks both.",
          primary, surface, text_color)

    _card("Volatility Clustering",
          "Here is a pattern that matters: volatility comes in clusters. After a big move (up or down), you tend to get MORE big moves. After a calm period, you tend to get MORE calm. This is called clustering. SENTINEL uses a GARCH model (Generalized AutoRegressive Conditional Heteroskedasticity) to capture this. In plain English: if yesterday was wild, today is probably wild too. If yesterday was calm, today is probably calm too.",
          primary, surface, text_color)

    _card("What GARCH Does",
          "GARCH is a model that forecasts future volatility based on recent volatility patterns. It knows that a quiet 0.2% day followed by a sudden 3% crash means volatility is spiking — and it forecasts that tomorrow will also be volatile. Traditional models assume volatility is constant. GARCH knows it changes. This matters because SENTINEL adjusts position sizing, stop losses, and confidence levels based on expected volatility.",
          primary, surface, text_color)

    _card("Fat Tails — Extreme Moves Happen More Than You Think",
          "If stock returns followed a perfect bell curve, a 5-standard-deviation move should happen once every 14,000 years. In reality, it happens every few years. Financial markets have 'fat tails' — extreme moves are much more common than basic statistics predict. This is why SENTINEL uses a Student-t distribution instead of a normal distribution for price targets. It gives wider bands for extreme outcomes, which is more honest.",
          bear, surface, text_color)

    _card("How to Use Volatility",
          "High volatility means wider stop losses (so you do not get stopped out by noise), smaller position sizes (because each position can lose more), and lower confidence in short-term forecasts. Low volatility means tighter stops, larger positions, and watch for breakouts. SENTINEL automatically adjusts for all of this in its scoring.",
          "#FF9800", surface, text_color)


# ── MODULE 14: Deep Analysis ──────────────────────────────────

def _module_deep_analysis(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 24: Deep Analysis — Reading a Research Report")
    st.write("The Deep Analysis page gives you a full Wall Street research report on any stock. Here is how to read it.")

    _card("What It Pulls Together",
          "Deep Analysis grabs everything: price data and technical indicators, fundamental data (P/E ratio, revenue, margins), analyst ratings and price targets, insider buying and selling, institutional ownership, recent news and sentiment, and earnings history. Then it synthesizes all of this into one view.",
          primary, surface, text_color)

    st.subheader("Understanding Fundamental Data")

    _card("P/E Ratio (Price to Earnings)",
          "How much you pay for each dollar of profit. If a stock has a P/E of 25, you are paying $25 for every $1 the company earns per year. A high P/E means investors expect big growth ahead (or the stock is overpriced). A low P/E means investors expect slow growth (or the stock is a bargain). Compare P/E to the sector average — a tech stock with a P/E of 30 might be cheap compared to its peers at 40.",
          primary, surface, text_color)

    _card("Revenue and Earnings Growth",
          "Revenue is the total money coming in. Earnings is what is left after all costs. Revenue growth tells you if the business is expanding. Earnings growth tells you if it is getting more profitable. The best stocks have BOTH growing. Revenue growth without earnings growth means costs are eating the gains. Earnings growth without revenue growth means cost-cutting, which has limits.",
          primary, surface, text_color)

    _card("Profit Margin",
          "What percentage of revenue turns into profit. A 20% margin means the company keeps $0.20 of every dollar it earns. High margins = strong competitive position (pricing power, efficient operations). Falling margins = trouble (rising costs, competitive pressure). SENTINEL shows margin trends, not just the current number.",
          primary, surface, text_color)

    st.subheader("Insider and Institutional Activity")

    _card("Insider Buying vs. Selling",
          "Insiders are company executives and board members. When they BUY stock with their own money, that is a strong bullish signal — they know the company best and they are betting their own money. Insider SELLING is less meaningful because insiders sell for many reasons (paying taxes, buying a house, diversifying). But a cluster of MULTIPLE insiders selling at the same time is a red flag.",
          bull, surface, text_color)

    _card("Institutional Ownership",
          "What percentage of the stock is owned by big funds (mutual funds, pension funds, hedge funds). High institutional ownership (70%+) means smart money approves. But watch the TREND — if institutional ownership is dropping, the big players are getting out. Rising institutional ownership means they are accumulating.",
          primary, surface, text_color)

    st.subheader("Price Targets")

    _card("How SENTINEL Computes Price Targets",
          "SENTINEL does NOT just average analyst targets. It builds its own targets from three independent signal groups: Trend signals (moving averages, MACD, ADX) weighted 40%, Value signals (how far from fair value) weighted 35%, and Analyst consensus weighted 25%. These three groups are deliberately uncorrelated — they look at different things. When all three agree, confidence is high. When they disagree, confidence drops.",
          primary, surface, text_color)

    _card("Support and Resistance Levels",
          "Support is a price level where buyers step in — the stock bounces off this floor. Resistance is where sellers step in — the stock gets rejected at this ceiling. These levels form because traders remember them. If a stock bounced at $100 three times, the fourth time it hits $100, buyers show up again. SENTINEL identifies these levels from recent price history.",
          "#FF9800", surface, text_color)

    _card("Why Analyst Targets Are Marked as Biased",
          "Wall Street analyst targets are biased bullish by 5-15% on average. This is not a conspiracy — it is an incentive problem. Analysts work for banks that make money when you buy stocks. 'Buy' recommendations outnumber 'Sell' recommendations roughly 10 to 1 across the industry. SENTINEL shows analyst targets as a reference but discounts them. Use the SENTINEL-computed targets as your primary guide.",
          bear, surface, text_color)


# ── MODULE 15: Risk Management ────────────────────────────────

def _module_risk_management(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 30: Risk Management — Real Rules")
    st.write("This is the most important module. You can have the best signals in the world and still lose everything without risk management. Professionals spend more time managing risk than picking stocks.")

    _card("Rule 1: Never Risk More Than 2% on a Single Trade",
          "If you have $10,000, never risk more than $200 on any single trade. This does not mean you can only buy $200 worth of stock — it means your STOP LOSS should be set so that if the trade goes wrong, you lose at most $200. If you buy a $50 stock and set a stop at $48, you are risking $2 per share. $200 / $2 = 100 shares maximum. This is called position sizing.",
          bear, surface, text_color)

    _card("Rule 2: Use ATR for Stop Losses",
          "The biggest mistake beginners make is putting stop losses too tight. If a stock moves $3 per day on average (its ATR) and you put your stop $1 away, you will get stopped out by normal daily noise. The rule: set your stop at 1.5x to 2x the ATR below your entry for long positions. This gives the stock room to breathe while still protecting you from a real breakdown.",
          "#FF9800", surface, text_color)

    _card("Rule 3: Scale Position Size to Conviction",
          "SENTINEL gives you conviction scores. Use them for sizing. Score above 75 (very bullish): use your full 2% risk. Score 60-75 (bullish): use 1.5% risk. Score 50-60 (neutral lean): use 1% risk or skip the trade. Below 50: do not go long. This means your biggest positions are your highest-conviction ideas, and your smaller positions are the uncertain ones.",
          primary, surface, text_color)

    _card("Rule 4: The Risk/Reward Ratio",
          "Before entering any trade, calculate: how much can I make vs. how much can I lose? If your target is $5 above your entry and your stop is $2 below, your risk/reward ratio is 2.5:1. Never take a trade below 2:1. Professional traders aim for 3:1 or better. This means even if you are wrong 50% of the time, you still make money because your winners are bigger than your losers.",
          bull, surface, text_color)

    _card("Rule 5: Have a Maximum Portfolio Risk",
          "Do not just manage individual trades — manage total exposure. If you have 5 positions each risking 2%, your total portfolio risk is 10%. If they are all in the same sector or all highly correlated, a single bad event could hit all 5. Cap total portfolio risk at 6-8%. Diversify across sectors and asset classes (stocks, crypto, bonds, cash).",
          "#FF9800", surface, text_color)

    _card("Rule 6: Reduce Size in High Volatility",
          "When SENTINEL shows elevated volatility (VIX above 25, ATR expanding), CUT your position sizes by 30-50%. High volatility means bigger daily swings, which means your 2% risk gets hit faster. Think of it like driving in a storm — you slow down and leave more room. Same thing with your portfolio.",
          bear, surface, text_color)

    _card("Rule 7: Cash Is a Position",
          "When signals conflict or the regime is in Transition, there is no rule that says you must be invested. Holding cash means you are not losing money in a bad market AND you have money ready to deploy when a clear opportunity appears. Professional traders spend a lot of time in cash. Beginners feel like they must always be in a trade. That feeling will lose you money.",
          neutral, surface, text_color)

    _card("Rule 8: Never Average Down Without a Plan",
          "Averaging down means buying more of a stock that has dropped. If you bought at $50 and it is now at $40, buying more brings your average cost to $45. Sounds smart until the stock goes to $20. Only average down if: the original reason for the trade is still valid, the conviction score is still bullish, AND you planned for it before entering the trade. Never average down just because you are losing.",
          bear, surface, text_color)

    _card("Rule 9: Take Profits on the Way Up",
          "You do not have to sell everything at once. If your position is up 20%, consider selling half and letting the rest ride with a trailing stop. This locks in profit while keeping upside exposure. Greed — holding for the absolute top — is one of the most common ways traders turn winners into losers.",
          bull, surface, text_color)

    _card("The Math That Matters",
          "If you lose 50% of your portfolio, you need a 100% gain just to get back to even. If you lose 25%, you need 33% to recover. If you lose 10%, you need 11%. This is why protecting against big losses is MORE important than chasing big gains. A 10% annual return with small drawdowns will make you more money over 10 years than a 20% return with occasional 40% crashes. Risk management is the entire game.",
          bear, surface, text_color)


# ── MODULE 16: Using SENTINEL ──────────────────────────────────

def _module_using_sentinel(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 32: Using SENTINEL — Putting It All Together")
    st.write("Here is the recommended workflow — from big picture to specific trades.")

    steps = [
        ("Step 1: Check the Genius Briefing",
         "Start here every morning. The briefing reads ALL signals and tells you what they mean in plain English. It gives you the regime, risk level, key takeaways, and action items. This is your 30-second morning briefing.",
         primary),
        ("Step 2: Check the Regime Monitor",
         "Know what environment you are in. Bull, Bear, or Transition. This determines your overall aggression level. In Bull, you buy dips. In Bear, you sell rallies. In Transition, you wait for clarity.",
         primary),
        ("Step 3: Check the AI Forecast & Probability Page",
         "See what the machine learning models predict. Look at the probability breakdown (UP/DOWN/FLAT) and the confidence level. Check the model accuracy — if it is significantly above 33% (random), the model has a real edge. Look at multiple timeframes — 1 week, 1 month, 3 months.",
         primary),
        ("Step 4: Check Intermarket Signals",
         "Are bonds, gold, credit, and the dollar confirming or diverging from stocks? If intermarket signals agree with the AI forecast, your conviction should be higher. If they disagree, be cautious.",
         "#FF9800"),
        ("Step 5: Check Market Breadth",
         "Is the rally broad or narrow? A healthy market has 60%+ of stocks above their 50-day MA. If breadth is weak while the index looks fine, be suspicious — that divergence is a warning sign.",
         "#FF9800"),
        ("Step 6: Sector Rotation",
         "Check which economic cycle phase we are in (Early Recovery, Mid Cycle, Late Cycle, Recession) and favor the sectors that historically outperform in that phase. See Module 12 for details.",
         bull),
        ("Step 7: Alpha Screener for Specific Ideas",
         "Once you know the direction, regime, and conviction level, use the Alpha Screener to find the highest-conviction individual stocks. The composite score combines all signals into one number. See Module 10 for how to read it.",
         bull),
        ("Step 8: Deep Analysis Before You Buy",
         "Before entering any position, run it through Deep Analysis. Check fundamentals, insider activity, price targets, and news sentiment. This is your final checklist before committing money.",
         bull),
        ("Step 9: Risk Management (Most Important)",
         "Size your position based on conviction score and current volatility. Set your stop loss using ATR. Calculate your risk/reward ratio — never below 2:1. Know your maximum portfolio risk. See Module 15 for the full rulebook.",
         bear),
        ("Step 10: Monitor and Adjust",
         "Check SENTINEL daily. Scores and regimes change. If your stock's conviction score drops below 40, reassess the position. If the regime shifts from Bull to Transition, reduce exposure. If news sentiment spikes negative on your sector, pay attention. SENTINEL refreshes data every 5 minutes for stocks and every 3 minutes for crypto.",
         "#FF9800"),
    ]

    for i, (title, content, color) in enumerate(steps, 1):
        st.markdown(
            f"<div style='background:{surface};padding:20px;border-radius:12px;margin-bottom:12px;"
            f"border-left:4px solid {color}'>"
            f"<h4 style='margin:0 0 8px;color:{color}'>{title}</h4>"
            f"<p style='margin:0;color:{text_color};font-size:14px;line-height:1.7'>{content}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.subheader("Key Principles")
    principles = [
        "The TREND is your friend — do not fight it. If price is above all moving averages, the trend is up.",
        "BREADTH confirms or denies. A healthy rally has broad participation. A narrow rally is fragile.",
        "INTERMARKET signals are leading indicators. Bonds, gold, and credit often move before stocks do.",
        "REGIME matters more than any single indicator. The same setup works differently in Bull vs Bear.",
        "CONVICTION SCORES combine everything — use them for position sizing, not just direction.",
        "AI gives you an EDGE, not certainty. 55% accuracy applied consistently beats 90% accuracy applied randomly.",
        "RISK MANAGEMENT is the only thing that keeps you in the game. You can be wrong 40% of the time and still make money if you manage risk properly.",
        "PATIENCE is a position. When signals conflict, doing nothing is the smartest trade.",
        "VOLATILITY is your speedometer. When it is high, slow down. When it is low, watch for breakouts.",
        "LEARN from losses. Every losing trade teaches you something if you pay attention to what went wrong.",
    ]
    for p in principles:
        st.markdown(f"- {p}")


# ── MODULE 17: Stock Market 101 ───────────────────────────────

def _module_stock_market_101(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 1: Stock Market 101 — How It All Works")
    st.write("Before you trade anything, you need to understand what the stock market actually is. This is day one of trading school.")

    _card("What Is the Stock Market?",
          "The stock market is a marketplace where people buy and sell pieces of companies. When a company 'goes public,' it sells shares (tiny ownership pieces) to the public. After that, those shares trade on an exchange. The price goes up and down based on what people are willing to pay. That is it — it is just buyers and sellers agreeing on a price.",
          primary, surface, text_color)

    _card("The Two Major Exchanges",
          "NYSE (New York Stock Exchange) is the oldest and largest exchange. It is on Wall Street in New York. Big established companies trade here — Walmart, JPMorgan, Coca-Cola. NASDAQ is the electronic exchange known for tech companies — Apple, Microsoft, Google, Amazon, Tesla. When people say 'the market,' they usually mean the S&P 500, which includes stocks from both exchanges.",
          primary, surface, text_color)

    _card("Market Hours",
          "The stock market is open Monday through Friday, 9:30 AM to 4:00 PM Eastern Time. That is it — 6.5 hours per day. It is closed on weekends and major holidays. Pre-market trading runs from 4:00 AM to 9:30 AM, and after-hours trading runs from 4:00 PM to 8:00 PM. Pre-market and after-hours have much lower volume, so prices can be more volatile and spreads are wider.",
          primary, surface, text_color)

    _card("What Is a Stock?",
          "A stock (also called a share or equity) is a tiny piece of ownership in a company. If a company has 1 billion shares and you buy 100, you own 0.00001% of that company. You make money two ways: the stock price goes up (capital gain) or the company pays you a portion of its profits (dividend). You lose money when the stock price goes down.",
          primary, surface, text_color)

    st.subheader("Stock Categories by Size")

    sizes = [
        ("Large Cap", "$10 billion+", "The giants. Apple, Microsoft, Amazon. Stable, slow-moving, less likely to go to zero. Most of the S&P 500.", primary),
        ("Mid Cap", "$2-10 billion", "Growing companies. More volatile than large caps but more growth potential. The sweet spot for many traders.", "#FF9800"),
        ("Small Cap", "$300M-2 billion", "Younger or niche companies. Higher risk, higher reward. Can double in a year or lose 50%.", "#FF9800"),
        ("Micro/Penny", "Under $300M", "Tiny companies. Very high risk. Low volume means it is easy to get stuck in a position you cannot exit. Most beginners should avoid these entirely.", bear),
    ]
    for name, size, desc, color in sizes:
        _card(f"{name} ({size})", desc, color, surface, text_color)

    st.subheader("What Is an Index?")

    _card("S&P 500",
          "The 500 largest US companies weighted by market cap. This IS 'the market' for most purposes. When people say stocks are up or down, they usually mean the S&P 500. SPY is the ETF that tracks it — the most traded security in the world.",
          primary, surface, text_color)

    _card("Dow Jones Industrial Average (DJIA)",
          "30 huge US companies. The oldest major index (started in 1896). It is price-weighted, which is weird — a $300 stock affects the Dow more than a $50 stock regardless of company size. Most professionals pay more attention to the S&P 500, but the Dow is what you see on the evening news.",
          primary, surface, text_color)

    _card("NASDAQ Composite",
          "All stocks listed on the NASDAQ exchange — over 3,000 companies. Heavily weighted toward technology. When tech is doing well, the Nasdaq outperforms. When tech is selling off, the Nasdaq gets hit harder. QQQ is the ETF that tracks the top 100 Nasdaq stocks.",
          primary, surface, text_color)

    _card("Russell 2000",
          "2,000 small-cap stocks. This is the best gauge of how the AVERAGE American company is doing. While the S&P 500 can be propped up by 5 mega-cap tech stocks, the Russell 2000 reflects the broader economy. IWM is the ETF. When the Russell is strong, the economy is genuinely healthy.",
          primary, surface, text_color)

    st.subheader("What Is an ETF?")

    _card("ETFs Explained",
          "An ETF (Exchange Traded Fund) is a basket of stocks packaged into one thing you can buy. Instead of buying all 500 S&P companies individually, you buy SPY and instantly own a piece of all 500. ETFs trade just like stocks — you buy and sell them during market hours at whatever the current price is. They are the easiest way for beginners to get diversified exposure.",
          bull, surface, text_color)

    _card("Common ETFs You Should Know",
          "SPY = S&P 500. QQQ = Nasdaq 100 (tech heavy). IWM = Russell 2000 (small caps). TLT = 20+ year Treasury bonds. GLD = Gold. USO = Oil. UUP = US Dollar. XLF = Financials sector. XLK = Technology sector. XLE = Energy sector. Each sector has its own ETF, which makes it easy to bet on a specific part of the economy.",
          primary, surface, text_color)

    st.subheader("What Is Short Selling?")

    _card("How Shorting Works",
          "Short selling is betting that a stock will go DOWN. You borrow shares from your broker, sell them immediately at today's price, and hope to buy them back later at a lower price. If the stock goes from $100 to $80, you profit $20 per share. But if it goes from $100 to $150, you lose $50 per share. The danger: your losses are theoretically UNLIMITED because a stock can go up forever, but it can only go down to zero.",
          bear, surface, text_color)

    _card("Why Beginners Should Avoid Shorting",
          "Short selling is how many beginners blow up their accounts. The risk is asymmetric — you can lose more than 100% of your investment. Short squeezes (like GameStop in 2021) can cause massive, sudden losses. If you want to bet against the market, buy inverse ETFs (like SH for short S&P 500) instead — your maximum loss is limited to what you paid.",
          bear, surface, text_color)

    _card("What Is Margin?",
          "Margin is borrowed money from your broker. A margin account lets you buy $20,000 worth of stock with only $10,000 of your own money. Sounds great until the stock drops — you still owe the broker. If your account falls below a certain level, you get a 'margin call' — the broker forces you to deposit more money or they sell your positions at whatever price they can get, which is usually the worst possible moment.",
          bear, surface, text_color)


# ── MODULE 18: Orders, Brokers & How to Actually Trade ────────

def _module_orders_brokers(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 2: Orders, Brokers & How to Actually Trade")
    st.write("Knowing what to buy is only half the battle. You need to know HOW to buy it. Here is how trading actually works.")

    st.subheader("Types of Orders")

    _card("Market Order",
          "Buy or sell RIGHT NOW at whatever the current price is. This is the simplest order. The upside: guaranteed to execute. The downside: in a fast-moving market, you might get a worse price than you expected. For large, liquid stocks (SPY, AAPL) during market hours, market orders are fine. For small stocks or after-hours, use limit orders.",
          primary, surface, text_color)

    _card("Limit Order",
          "Buy or sell only at YOUR price or better. A limit buy at $95 means you will only buy if the price hits $95 or lower. A limit sell at $105 means you will only sell if the price hits $105 or higher. The upside: you control your price. The downside: your order might never fill if the price does not reach your limit. Use limit orders for entries when you are not in a rush.",
          primary, surface, text_color)

    _card("Stop Loss Order",
          "An order that triggers when the price reaches a certain level to LIMIT YOUR LOSSES. If you buy at $100 and set a stop loss at $90, your shares will be sold automatically if the price drops to $90. This is the single most important order type for protecting your money. Every single trade should have a stop loss. No exceptions.",
          bear, surface, text_color)

    _card("Stop-Limit Order",
          "Combines a stop and a limit. A stop-limit sell at $90 with a limit of $88 means: if the price drops to $90, place a limit sell order at $88 or better. The risk: in a fast crash, the price might blow past both your stop and your limit, and your order never fills. You are still holding while the stock keeps falling. For most situations, a regular stop loss is safer.",
          "#FF9800", surface, text_color)

    _card("Trailing Stop",
          "A stop loss that moves UP with the price but never moves DOWN. If you set a 5% trailing stop on a stock at $100, your stop is at $95. If the stock goes to $110, your stop automatically moves to $104.50 (5% below $110). If the stock then drops to $104.50, you are sold out — locking in a $4.50 gain instead of a loss. This is the best tool for letting winners run while protecting profits.",
          bull, surface, text_color)

    st.subheader("The Bid-Ask Spread")

    _card("What Is the Spread?",
          "Every stock has two prices: the bid (what buyers are willing to pay) and the ask (what sellers are willing to accept). The difference is the spread. If the bid is $99.95 and the ask is $100.05, the spread is $0.10. When you buy, you pay the ask. When you sell, you get the bid. The spread is a hidden cost of every trade. Wide spreads (low-volume stocks) cost you more. Tight spreads (SPY, AAPL) cost almost nothing.",
          primary, surface, text_color)

    _card("Why the Spread Matters",
          "If the spread is $0.50 on a $20 stock, you are already down 2.5% the instant you buy. For day traders making many small trades, the spread eats into profits fast. For long-term investors, it barely matters. Rule of thumb: if the spread is more than 0.5% of the stock price, be cautious — it is hard to trade profitably.",
          "#FF9800", surface, text_color)

    st.subheader("Choosing a Broker")

    _card("What Is a Broker?",
          "A broker is the company that holds your account and executes your trades. You cannot buy stocks directly on the exchange — you go through a broker. All major US brokers now offer commission-free trading for stocks and ETFs. The main ones: Fidelity (best for beginners — great research tools, no payment for order flow), Charles Schwab (great all-around, merged with TD Ameritrade), and Interactive Brokers (best for active traders — advanced tools, low margin rates).",
          primary, surface, text_color)

    _card("Paper Trading — Practice Before You Risk Real Money",
          "Every major broker offers paper trading — simulated trading with fake money. It works exactly like real trading but with zero risk. Trade for at least 2-4 weeks on paper before using real money. This lets you learn the platform, test strategies, and make mistakes that do not cost you anything. If you cannot make money paper trading, you will definitely not make money with real money.",
          bull, surface, text_color)

    _card("Account Types",
          "Taxable Brokerage Account: The standard account. You pay taxes on gains. No restrictions on when you can withdraw. Roth IRA: Tax-free growth and withdrawals in retirement. You contribute after-tax money. Best for long-term investing. Traditional IRA/401k: Tax-deferred. You get a tax deduction now, pay taxes when you withdraw in retirement. For active trading, use a taxable brokerage account.",
          primary, surface, text_color)

    st.subheader("The Pattern Day Trader Rule")

    _card("PDT Rule — Read This Before Day Trading",
          "If you make 4 or more 'day trades' (buy and sell the same stock on the same day) within 5 business days, your broker will flag you as a Pattern Day Trader. If your account has less than $25,000, you will be restricted from day trading for 90 days. This is an SEC rule, not a broker rule — all US brokers enforce it. If you have under $25,000, stick to swing trades (holding for days or weeks) instead of day trades.",
          bear, surface, text_color)


# ── MODULE 19: The Fed, Interest Rates & Why They Move Everything ──

def _module_fed_rates(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 5: The Fed, Interest Rates & Why They Move Everything")
    st.write("The Federal Reserve is the most powerful force in financial markets. When the Fed speaks, everything moves. Here is why.")

    _card("What Is the Federal Reserve?",
          "The Fed is the central bank of the United States. It controls the money supply and sets the baseline interest rate (the Federal Funds Rate). The Fed has two jobs: keep inflation low (target 2% per year) and keep unemployment low. These two goals often conflict — fighting inflation usually means slowing the economy, which causes job losses.",
          primary, surface, text_color)

    _card("What Is the Federal Funds Rate?",
          "This is the interest rate banks charge each other for overnight loans. It sounds boring but it affects EVERYTHING. When the Fed raises this rate, it becomes more expensive to borrow money — mortgages go up, car loans go up, credit card rates go up, and companies pay more to borrow. When they lower it, borrowing gets cheaper and money flows more freely.",
          primary, surface, text_color)

    st.subheader("How Rate Changes Affect Markets")

    _card("Rate Hike (Fed Raises Rates)",
          "The Fed raises rates to SLOW the economy and fight inflation. Higher rates mean: bonds pay more interest so money moves from stocks to bonds. Companies borrow less so growth slows. Mortgages get expensive so housing cools. Consumers spend less because borrowing costs more. Stocks generally go DOWN when the Fed raises rates, especially growth and tech stocks. The market usually starts falling BEFORE the hike — it prices things in early.",
          bear, surface, text_color)

    _card("Rate Cut (Fed Lowers Rates)",
          "The Fed cuts rates to STIMULATE the economy. Lower rates mean: borrowing is cheap so companies invest and expand. Mortgages get cheaper so housing booms. Bonds pay less so money moves into stocks for better returns. Consumers borrow and spend more. Stocks generally go UP when the Fed cuts rates. The first rate cut after a hiking cycle is one of the most bullish signals in the market.",
          bull, surface, text_color)

    _card("Quantitative Easing (QE) — Money Printing",
          "When rates are already near zero and the economy still needs help, the Fed buys government bonds and mortgage-backed securities directly. This pumps money into the financial system. QE is rocket fuel for stocks and crypto — it is what drove the massive rallies in 2009-2021. The S&P 500 tripled during QE. When the Fed stops QE or reverses it (Quantitative Tightening), markets lose that fuel and often drop.",
          primary, surface, text_color)

    _card("Quantitative Tightening (QT) — Draining Liquidity",
          "QT is the opposite of QE. The Fed stops buying bonds and lets existing ones expire, which pulls money OUT of the financial system. Less money in the system means less buying pressure on stocks, bonds, and crypto. QT started in 2022 and contributed to the bear market that year. SENTINEL tracks QE/QT status because it affects everything.",
          bear, surface, text_color)

    st.subheader("Key Economic Reports That Move Markets")

    reports = [
        ("CPI (Consumer Price Index)", "Monthly", "Measures inflation — how fast prices are rising. Higher than expected = Fed might raise rates = stocks drop. Lower than expected = less pressure on rates = stocks rally. This is often the most volatile report day of the month.", bear),
        ("Jobs Report (Non-Farm Payrolls)", "1st Friday of month", "How many jobs were added. Too many jobs = economy overheating = Fed stays tough = stocks drop. Fewer jobs = economy cooling = Fed might cut = stocks rally. Yes, bad economic news can be good for stocks. Welcome to markets.", "#FF9800"),
        ("FOMC Meeting", "8 times per year", "The Fed announces its rate decision and gives guidance. Markets move violently on these days. The press conference after the announcement often matters more than the decision itself. Watch the language — 'hawkish' (tough on inflation) is bearish, 'dovish' (accommodating) is bullish.", primary),
        ("GDP (Gross Domestic Product)", "Quarterly", "The total economic output of the country. Two consecutive quarters of negative GDP = recession. Markets usually move on the GDP number vs expectations, not the absolute level.", primary),
        ("PMI (Purchasing Managers Index)", "Monthly", "Survey of manufacturers. Above 50 = economy expanding. Below 50 = economy contracting. The ISM Manufacturing PMI is one of the best leading indicators — it turns down before the economy does.", "#FF9800"),
        ("Unemployment Claims", "Weekly", "How many people filed for unemployment benefits. Rising claims = economy weakening. Falling claims = economy strengthening. The 4-week moving average is more useful than any single week.", primary),
    ]

    for name, freq, explanation, color in reports:
        st.markdown(
            f"<div style='background:{surface};padding:18px;border-radius:10px;margin-bottom:12px;"
            f"border-left:4px solid {color}'>"
            f"<p style='margin:0;font-weight:700;color:{color};font-size:15px'>{name}</p>"
            f"<p style='margin:3px 0;color:{neutral};font-size:12px'>Released: {freq}</p>"
            f"<p style='margin:5px 0 0;color:{text_color};font-size:13px;line-height:1.6'>{explanation}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _card("The Yield Curve — The Most Reliable Recession Predictor",
          "The yield curve shows interest rates across different time periods. Normally, long-term bonds pay more than short-term bonds (you want more reward for locking up money longer). When this INVERTS (short-term pays more than long-term), it has predicted every US recession in the last 50 years. An inverted yield curve means the bond market believes a recession is coming. SENTINEL monitors the yield curve automatically on the Intermarket page.",
          bear, surface, text_color)


# ── MODULE 20: Earnings, Dividends & Corporate Events ─────────

def _module_earnings_dividends(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 6: Earnings, Dividends & Corporate Events")
    st.write("Companies report financial results every quarter. These reports move stocks more than almost anything else. Here is how to understand them.")

    st.subheader("Earnings Season")

    _card("What Is Earnings Season?",
          "Four times per year (January, April, July, October), most public companies report their financial results for the previous quarter. This is earnings season. It lasts about 6 weeks each time. During earnings season, individual stocks can move 5-20% in a single day based on their results. Most of the year's biggest stock moves happen during earnings.",
          primary, surface, text_color)

    _card("The Four Numbers That Matter",
          "Revenue (top line): Total money the company brought in. Wall Street sets an estimate — beat it and the stock usually goes up. Miss it and it drops. Earnings Per Share (EPS): Profit divided by number of shares. This is the most watched number. Guidance: What the company says about NEXT quarter. This often matters more than the current results. Margins: Profit as a percentage of revenue. Expanding margins = getting more efficient. Shrinking margins = costs rising.",
          primary, surface, text_color)

    _card("Beat and Raise vs. Beat and Lower",
          "The best earnings result is 'beat and raise' — the company beat estimates AND raised guidance for next quarter. Stocks often gap up 5-15% on this. The worst is 'miss and lower' — missed estimates AND lowered guidance. Can cause 10-30% drops. The tricky one: 'beat and lower.' The company did well this quarter but warns about the future. The stock often drops because Wall Street cares more about the future than the past.",
          "#FF9800", surface, text_color)

    _card("Earnings Whisper — The Real Estimate",
          "Wall Street analyst estimates are public, but the REAL expectation is often higher. This is called the 'whisper number.' If analysts estimate $1.50 EPS but the whisper is $1.60, the company needs to beat $1.60 for the stock to go up — beating $1.50 will not be enough because the market already priced in more. This is why some stocks drop even after 'beating' estimates.",
          "#FF9800", surface, text_color)

    _card("When to Buy Around Earnings",
          "Buying right before earnings is gambling — the stock can move 10% either way and there is no way to know. Most professional traders either: (1) buy weeks BEFORE earnings if the setup looks good and sell a portion before the report, or (2) wait until AFTER earnings and buy the reaction. The second approach gives up some potential upside but eliminates the binary risk.",
          bear, surface, text_color)

    st.subheader("Dividends")

    _card("What Is a Dividend?",
          "A dividend is a cash payment a company makes to shareholders, usually quarterly. If a stock pays a $1.00 annual dividend and you own 100 shares, you get $100 per year. Dividend yield is the annual dividend divided by the stock price. A $50 stock paying $2.00 per year has a 4% yield. Dividend stocks tend to be large, stable, profitable companies — utilities, banks, consumer staples.",
          bull, surface, text_color)

    _card("Ex-Dividend Date",
          "To receive a dividend, you must own the stock BEFORE the ex-dividend date. On the ex-dividend date, the stock price drops by approximately the dividend amount because new buyers will not receive that payment. If a stock is $50 and pays a $0.50 dividend, it opens at roughly $49.50 on the ex-dividend date. This is normal, not a selloff.",
          primary, surface, text_color)

    _card("Dividend Growth Investing",
          "Some investors focus entirely on buying stocks that consistently RAISE their dividend every year. These are called Dividend Aristocrats (25+ years of consecutive raises) and Dividend Kings (50+ years). Companies like Johnson & Johnson, Coca-Cola, and Procter & Gamble. The strategy: buy quality dividend growers, reinvest the dividends, and let compounding do the work over decades. Boring but very effective.",
          bull, surface, text_color)

    st.subheader("Other Corporate Events")

    _card("Stock Splits",
          "When a company divides its shares to lower the price. A 4-for-1 split on a $400 stock means you now have 4 shares at $100 each. Your total value does not change. Splits are cosmetically bullish — they make the stock 'feel' cheaper, which attracts retail buyers. Apple, Tesla, Amazon, and Google have all split in recent years. Stocks tend to run up before a split and sometimes pull back after.",
          primary, surface, text_color)

    _card("Buybacks (Share Repurchases)",
          "When a company uses its cash to buy back its own shares on the open market. This reduces the number of shares outstanding, which increases earnings per share and makes each remaining share more valuable. Buybacks are a bullish signal — management believes the stock is undervalued. Apple spends over $90 billion per year on buybacks.",
          bull, surface, text_color)

    _card("Mergers and Acquisitions (M&A)",
          "When one company buys another. The target company stock usually jumps to near the offer price. The acquiring company stock often drops slightly (they are spending a lot of money). If a deal falls through, the target stock crashes back down. M&A activity tends to increase in bull markets when companies have cash and confidence.",
          "#FF9800", surface, text_color)

    _card("IPOs (Initial Public Offerings)",
          "When a private company sells shares to the public for the first time. IPOs are exciting but risky. Many IPOs pop 20-50% on day one and then slowly fade over the next 6-12 months as insiders sell their shares. The rule: do not chase IPOs on day one. Wait at least 3-6 months for the hype to settle and the financials to speak for themselves.",
          "#FF9800", surface, text_color)


# ── MODULE 21: Crypto Deep Dive ───────────────────────────────

def _module_crypto_deep_dive(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 10: Crypto Deep Dive — Wallets, Exchanges & Gas Fees")
    st.write("Module 8 covered what crypto IS. This module covers how it WORKS — the stuff you need to know to trade safely.")

    st.subheader("How to Buy Crypto")

    _card("Centralized Exchanges (CEX)",
          "The easiest way to buy crypto. Coinbase is the most beginner-friendly (US regulated, insured, easy interface). Kraken is solid for more experienced users (lower fees, more coins). Binance.US has the most coins and lowest fees but has had regulatory issues. You deposit dollars, buy crypto, and the exchange holds it for you. This is similar to how a stock broker works.",
          primary, surface, text_color)

    _card("Decentralized Exchanges (DEX)",
          "Uniswap, Jupiter (Solana), and others let you trade directly from your own wallet without a company in the middle. No account, no KYC, no withdrawal limits. But also no customer support, no insurance, and easy to make irreversible mistakes. DEXs are for experienced users. Start with Coinbase or Kraken.",
          "#FF9800", surface, text_color)

    st.subheader("Wallets — Where Your Crypto Lives")

    _card("Hot Wallets (Software)",
          "A hot wallet is an app on your phone or computer. MetaMask (Ethereum), Phantom (Solana), and Trust Wallet (multi-chain) are popular. They are connected to the internet, which makes them convenient but also vulnerable to hacking. Good for small amounts you actively trade. NOT for your life savings.",
          "#FF9800", surface, text_color)

    _card("Cold Wallets (Hardware)",
          "A cold wallet is a physical device (like a USB stick) that stores your crypto offline. Ledger and Trezor are the big names. Because they are not connected to the internet, they are nearly impossible to hack remotely. The rule: if you have more than $1,000 in crypto, buy a hardware wallet. If you have more than $10,000, you need one.",
          bull, surface, text_color)

    _card("Seed Phrases — Guard These With Your Life",
          "When you create a wallet, you get a 12 or 24-word recovery phrase. This is the MASTER KEY to your crypto. If you lose it, your crypto is gone forever — no customer support can help. If someone else gets it, they can steal everything instantly. Write it on paper (not digital), store it somewhere safe (fireproof safe or safety deposit box), and NEVER type it into a website. Most crypto theft happens through fake websites that trick people into entering their seed phrase.",
          bear, surface, text_color)

    st.subheader("Key Crypto Concepts")

    _card("Gas Fees",
          "Every transaction on a blockchain costs a fee called 'gas.' On Ethereum, gas fees can range from $1 to $50+ depending on network congestion. On Solana, fees are fractions of a cent. On Bitcoin, fees range from $1 to $30. Gas fees matter because they eat into small trades. If you are buying $100 of a token and the gas fee is $15, you are already down 15%. Batch your transactions and use low-fee chains for small amounts.",
          "#FF9800", surface, text_color)

    _card("Stablecoins — Your Cash Position in Crypto",
          "Stablecoins are cryptocurrencies pegged to the US dollar. 1 USDC = 1 USDT = roughly $1.00 at all times. They serve the same purpose as holding cash in a stock account. When you want to exit a crypto position but stay 'in the system' (avoid transferring back to your bank), you sell to a stablecoin. USDC (by Circle) is the most trusted. USDT (Tether) has the most liquidity but has faced transparency concerns.",
          primary, surface, text_color)

    _card("DeFi (Decentralized Finance)",
          "DeFi is financial services (lending, borrowing, trading, earning interest) built on blockchain without banks. You can lend your crypto and earn 3-10% interest. You can borrow against your crypto without a credit check. You can provide liquidity to exchanges and earn trading fees. The risks: smart contract bugs can lose your money, and returns that seem too good to be true usually are. Stick to established DeFi protocols (Aave, Uniswap, Lido) if you explore this space.",
          "#FF9800", surface, text_color)

    _card("Staking — Earning Yield on Your Crypto",
          "Some cryptocurrencies let you 'stake' your coins — lock them up to help secure the network — and earn rewards in return. Ethereum staking earns about 3-5% per year. Solana staking earns about 6-8%. This is the crypto equivalent of earning interest on your savings account. The risk: your coins are locked for a period, and if the price drops, you are stuck. But for coins you plan to hold long-term anyway, staking is essentially free money.",
          bull, surface, text_color)

    _card("Layer 1 vs Layer 2",
          "Layer 1 is the main blockchain — Bitcoin, Ethereum, Solana. Layer 2 is a network built ON TOP of a Layer 1 to make it faster and cheaper. Arbitrum and Optimism are Layer 2s on Ethereum. They process transactions off the main chain and then settle back to Ethereum. Think of Layer 1 as the highway and Layer 2 as express lanes. Layer 2 tokens (ARB, OP) can be good investments if the underlying Layer 1 grows.",
          primary, surface, text_color)

    st.subheader("Crypto Tax Rules (US)")

    _card("You Owe Taxes on Every Trade",
          "In the US, every time you sell, trade, or swap crypto, it is a taxable event. Swapping ETH for SOL counts. Selling BTC for USD counts. Even spending crypto to buy something counts. Short-term gains (held less than 1 year) are taxed as ordinary income (up to 37%). Long-term gains (held over 1 year) are taxed at 0%, 15%, or 20% depending on your income. Track everything. Use a crypto tax tool like Koinly or CoinTracker.",
          bear, surface, text_color)

    _card("Wash Sale Rules and Crypto",
          "For stocks, the wash sale rule says you cannot sell at a loss and immediately buy back within 30 days to claim the tax deduction. As of 2025, crypto was NOT subject to this rule — you could sell BTC at a loss and immediately buy it back. However, this may change. Always check current tax law or ask a tax professional.",
          "#FF9800", surface, text_color)


# ── MODULE 22: Trading Styles ─────────────────────────────────

def _module_trading_styles(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 28: Trading Styles — Day vs Swing vs Investing")
    st.write("There is no single 'right way' to trade. Your style depends on your time, risk tolerance, and goals. Here are the main approaches.")

    _card("Day Trading",
          "Buy and sell within the same day. Never hold overnight. Day traders watch charts all day, make multiple trades, and profit from small price movements (0.5-3%). Requirements: $25,000+ account (PDT rule), fast internet, multiple monitors, full-time attention. Success rate: less than 10% of day traders are profitable after 1 year. Most beginners should NOT start here.",
          bear, surface, text_color)

    _card("Swing Trading (Best for Most People)",
          "Hold positions for days to weeks. Buy based on technical setups and sell when the trade plays out or hits your stop. Swing traders check charts 1-2 times per day, not all day. No PDT rule issues. This is what SENTINEL is built for — its scores and signals work best on a multi-day to multi-week timeframe. Most people with a day job should be swing traders.",
          bull, surface, text_color)

    _card("Position Trading",
          "Hold for weeks to months. Focus on bigger trends and macro forces. Position traders care less about daily noise and more about regime changes, sector rotation, and economic cycles. They use the Genius Briefing and Regime Monitor heavily. Requires patience — you might only make 3-6 trades per month.",
          primary, surface, text_color)

    _card("Long-Term Investing (Buy and Hold)",
          "Buy quality stocks or ETFs and hold for years or decades. Warren Buffett's approach. You do not care about daily price swings — you care about business quality, earnings growth, and valuation. The S&P 500 has returned roughly 10% per year over the last 100 years. Dollar-cost averaging (buying a fixed amount every month regardless of price) into SPY is one of the most reliable wealth-building strategies in existence.",
          bull, surface, text_color)

    st.subheader("Matching Your Style to Your Life")

    styles = [
        ("Full-time job, check phone at lunch", "Swing Trading or Position Trading", "Set alerts for your price levels. Check SENTINEL's briefing in the morning. Make decisions in the evening. Execute trades before or after market hours."),
        ("Work from home, flexible schedule", "Swing Trading", "Check SENTINEL 2-3 times per day. Enter trades during market hours when spreads are tightest. Monitor positions but do not stare at charts."),
        ("Full-time trader, dedicated hours", "Day Trading or Active Swing", "Use Signal Dashboard and Alpha Screener in real time. Layer in Deep Analysis for conviction. Monitor intraday volume and momentum."),
        ("Hands-off, just want growth", "Long-Term Investing", "Buy SPY and QQQ monthly. Check SENTINEL quarterly for regime changes. Only adjust when the regime shifts to Bear — then move to defensive positions."),
    ]

    for situation, style, explanation in styles:
        _card(f"{situation}", f"<strong>{style}</strong> — {explanation}", primary, surface, text_color)

    _card("The Timeframe Truth",
          "The shorter your timeframe, the more noise dominates your results. On a 1-minute chart, moves are nearly random. On a daily chart, trends become visible. On a weekly chart, major trends are clear. This is why day trading is so hard and long-term investing is so reliable. SENTINEL's accuracy improves on longer timeframes for the same reason.",
          "#FF9800", surface, text_color)


# ── MODULE 23: Trading Psychology ─────────────────────────────

def _module_psychology(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 29: Trading Psychology — Your Biggest Enemy Is You")
    st.write("The number one reason traders lose money is not bad analysis — it is bad psychology. Your emotions will try to sabotage every trade. Understanding these patterns is the difference between success and failure.")

    _card("Why Most Traders Lose Money",
          "Studies consistently show that 70-90% of retail traders lose money. The data is not because the market is rigged — it is because humans are emotionally terrible at making financial decisions. We hold losers too long hoping they come back. We sell winners too early because we fear losing the gain. We trade too much because sitting still feels lazy. We double down on bad trades because admitting a mistake hurts. SENTINEL exists to fight these biases by giving you objective, emotionless signals.",
          bear, surface, text_color)

    st.subheader("The Emotional Traps")

    _card("FOMO (Fear of Missing Out)",
          "You see a stock up 20% and think: 'I need to get in before it goes higher.' This is almost always wrong. By the time you notice a big move, the easy money has been made. Chasing stocks that already moved is how most beginners lose money. The fix: SENTINEL's signals tell you when a setup is forming BEFORE the move. Trust the process, not the fear. There is always another trade.",
          "#FF9800", surface, text_color)

    _card("Revenge Trading",
          "You lose money on a trade and immediately jump into another trade to 'make it back.' This never works. You are emotional, not thinking clearly, and you usually pick a worse trade than the one you just lost on. The fix: after a loss, walk away for at least 1 hour. Better yet, stop trading for the day. Come back tomorrow with fresh eyes.",
          bear, surface, text_color)

    _card("Confirmation Bias",
          "You decide a stock is going up, then only look for information that supports your view while ignoring warning signs. This is why traders hold losing positions too long — they keep finding reasons to stay in while ignoring the screaming sell signals. The fix: before entering any trade, actively try to find reasons NOT to take it. If you cannot find any, the trade might actually be good. SENTINEL's conviction score does this automatically — it weighs bearish AND bullish signals.",
          "#FF9800", surface, text_color)

    _card("Loss Aversion",
          "Losing $100 feels about twice as bad as gaining $100 feels good. This causes traders to: hold losers too long (hoping to break even instead of cutting losses) and sell winners too early (locking in the gain before it can be taken away). The math says you should do the exact opposite — cut losers fast and let winners run. Every professional trader will tell you this is the single hardest thing to do consistently.",
          bear, surface, text_color)

    _card("Anchoring",
          "You bought at $50 and the stock drops to $35. You keep thinking 'it was at $50, so $35 is cheap.' But $50 is irrelevant — the stock is worth whatever the market says it is worth today. Your purchase price does not change the stock's value. The fix: ask yourself 'if I did not already own this stock, would I buy it at today's price with the current signals?' If the answer is no, sell.",
          "#FF9800", surface, text_color)

    _card("Overconfidence After Wins",
          "You hit three winning trades in a row and start thinking you have figured out the market. You increase your position size, skip your stop losses, and take riskier trades. Then one bad trade wipes out all three wins and more. The market does not care about your winning streak. The fix: keep your position sizing rules the same whether you have won 5 in a row or lost 5 in a row.",
          bear, surface, text_color)

    _card("The Sunk Cost Fallacy",
          "You have been holding a losing stock for 3 months and refuse to sell because 'I have already invested so much time and money in this.' The money you lost is gone regardless of whether you sell now or hold. The only question is: based on the signals TODAY, is this stock likely to go up or down? If the conviction score is below 40, sell. The past is irrelevant.",
          "#FF9800", surface, text_color)

    st.subheader("Emotional Discipline Tools")

    _card("Keep a Trading Journal",
          "Write down every trade: what you bought, why you bought it, what the SENTINEL score was, your stop loss, your target, and what happened. Review it weekly. You will start seeing patterns — like always getting FOMO on Mondays, or always ignoring your stops when the trade is 'close to breaking even.' Self-awareness is the first step to fixing bad habits.",
          bull, surface, text_color)

    _card("Pre-Commit to Your Rules",
          "Before the market opens, write down your trading rules for the day. 'I will only buy stocks with conviction above 65. My stop is at 1.5x ATR. I will not make more than 3 trades today.' Writing it down makes you accountable. When FOMO hits at 2 PM, you look at your rules and say 'no, the conviction is only 52.' Rules protect you from yourself.",
          bull, surface, text_color)

    _card("The 3-Question Test",
          "Before every trade, answer these three questions: (1) What is the SENTINEL conviction score? (2) What is my stop loss and where is it based on? (3) What is my risk/reward ratio? If you cannot answer all three, do not take the trade. This simple test eliminates 80% of bad trades.",
          bull, surface, text_color)


# ── MODULE 24: Building Your Daily Trading Routine ────────────

def _module_daily_routine(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 31: Building Your Daily Trading Routine")
    st.write("Consistent profits come from consistent routines. Here is a real daily workflow built around SENTINEL.")

    st.subheader("Pre-Market (Before 9:30 AM ET)")

    _card("Step 1: Open SENTINEL Genius Briefing (2 min)",
          "This is your morning briefing. It tells you the regime (Bull/Bear/Transition), overall risk level, every sector's score, and the key headlines. Read it like a morning newspaper. If the regime is bullish and risk is low, you can be aggressive today. If the regime is transitioning and risk is elevated, stay defensive.",
          primary, surface, text_color)

    _card("Step 2: Check the Economic Calendar (1 min)",
          "Is there a Fed meeting today? CPI report? Jobs data? Major earnings? If there is a high-impact event, be extra cautious with new positions — the market can whipsaw violently around these events. Widen your stops or reduce your size. If there is nothing major, trade normally.",
          "#FF9800", surface, text_color)

    _card("Step 3: Review Your Open Positions (2 min)",
          "Check the conviction score for every stock you currently own. Has it changed significantly? If a score dropped from 70 to 45, that is a warning — something changed. Check the news and Deep Analysis. Adjust your stops if needed. Decide if you need to exit anything today.",
          primary, surface, text_color)

    st.subheader("Market Hours (9:30 AM - 4:00 PM ET)")

    _card("The First 30 Minutes — Do NOT Trade",
          "The first 30 minutes after the market opens are the most volatile and deceptive. Prices whip around on overnight orders being filled and algorithms fighting each other. Many trades that look great at 9:35 AM look terrible by 10:00 AM. Wait until 10:00 AM for the dust to settle before entering new positions. This one rule alone will save you from dozens of bad trades per year.",
          bear, surface, text_color)

    _card("Mid-Day: Scan and Plan (11 AM - 2 PM)",
          "This is the quietest part of the trading day. Use this time to: run the Alpha Screener for new ideas, check Deep Analysis on any stocks that interest you, set up limit orders for entries you want, and review your watchlist. Do not force trades during the mid-day lull — volume is low and moves are unreliable.",
          primary, surface, text_color)

    _card("Power Hour (3:00 - 4:00 PM)",
          "The last hour of trading is the second most active (after the open). Institutional traders make their moves here. If a stock has been strong all day and volume picks up in the last hour, that is very bullish — smart money is buying into the close. If a stock sells off in the last hour, that is bearish. Some traders focus exclusively on this window.",
          "#FF9800", surface, text_color)

    st.subheader("Post-Market (After 4:00 PM ET)")

    _card("Step 4: Review the Day (5 min)",
          "Did any of your positions hit stops? Did any new signals fire? Check the Signal Dashboard for any major changes. Look at the Intermarket page — did bonds, gold, or credit do anything unusual? Write down anything notable in your trading journal.",
          primary, surface, text_color)

    _card("Step 5: Plan Tomorrow (5 min)",
          "Based on today's signals, set up your trades for tomorrow. Identify which stocks you might buy, at what price, with what stop loss. Write it down. When the market opens tomorrow, you already have a plan — you are not reacting emotionally to whatever the market throws at you.",
          bull, surface, text_color)

    st.subheader("Weekend Review (30 min on Saturday or Sunday)")

    _card("The Weekly Review",
          "This is the most valuable 30 minutes of your trading week. Review ALL trades from the past week. What worked? What did you do wrong? Did you follow your rules? Calculate your win rate and average risk/reward. Check the Regime Monitor — is the regime changing? Look at Sector Rotation — which sectors are gaining momentum? Adjust your watchlist for next week.",
          bull, surface, text_color)

    st.subheader("Rules for Your Routine")

    rules = [
        "Same time every day. Markets reward consistency, not intensity.",
        "Briefing BEFORE the market opens. Never start the day blind.",
        "No trading in the first 30 minutes. Let the chaos settle.",
        "Maximum 3 new trades per day. Quality over quantity.",
        "Every trade has a stop loss BEFORE you enter. No exceptions.",
        "Trading journal updated daily. Even two sentences is enough.",
        "Weekend review every week. This is where real improvement happens.",
        "If you are angry, scared, or euphoric — step away. Emotions destroy edge.",
        "Your SENTINEL conviction threshold is your rulebook. Below it, no trade.",
        "The goal is not to trade every day. The goal is to trade WELL.",
    ]
    for i, rule in enumerate(rules, 1):
        st.markdown(f"**{i}.** {rule}")


# ═══════════════════════════════════════════════════════════
# MODULE 25: BLOCKCHAIN — HOW THE TECHNOLOGY WORKS
# ═══════════════════════════════════════════════════════════
def _module_blockchain(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 9: Blockchain — How the Technology Works")
    st.caption("The engine under the hood of every cryptocurrency")

    st.subheader("What Is a Blockchain?")

    _card("The Core Idea",
          "A blockchain is a digital ledger that records transactions across thousands of computers simultaneously. No single person or company controls it. Every participant holds a copy of the entire history. When a new transaction happens, every copy updates at the same time. This makes it nearly impossible to cheat — you would have to hack thousands of computers simultaneously to change a single record.",
          primary, surface, text_color)

    _card("Blocks and Chains",
          "Transactions are grouped into 'blocks.' Each block contains a batch of transactions (usually a few hundred to a few thousand), a timestamp, and a cryptographic fingerprint of the previous block. That fingerprint is called a hash. Because each block references the one before it, they form a chain — hence 'blockchain.' If someone tries to alter a past block, the hash changes, which breaks every block after it. The chain becomes invalid and the network rejects it.",
          bull, surface, text_color)

    _card("Why This Matters for Trading",
          "Blockchain gives crypto its core properties: no middleman (no bank needed), censorship resistance (nobody can freeze your account), transparency (anyone can verify any transaction), and immutability (once confirmed, a transaction cannot be reversed). These properties are why people assign value to crypto. When you trade crypto, you are trading access to these networks.",
          neutral, surface, text_color)

    st.subheader("Consensus Mechanisms — How the Network Agrees")

    _card("The Problem: Who Validates Transactions?",
          "In a traditional bank, the bank decides which transactions are valid. In crypto, there is no bank. So how do thousands of strangers agree on which transactions are real? This is called the consensus problem. Different blockchains solve it different ways, and the method they choose affects speed, security, cost, and energy usage.",
          primary, surface, text_color)

    _card("Proof of Work (PoW) — Bitcoin's Method",
          "Miners compete to solve a mathematical puzzle. The first one to solve it gets to add the next block and earns a reward (currently 3.125 BTC per block). The puzzle requires massive computing power, which costs electricity. This energy expenditure is what secures the network — attacking Bitcoin would require more electricity than most countries consume. PoW is extremely secure but slow (Bitcoin processes about 7 transactions per second) and energy-intensive.",
          bull, surface, text_color)

    _card("Proof of Stake (PoS) — Ethereum's Method",
          "Instead of burning electricity, validators lock up (stake) their coins as collateral. The network randomly selects validators to propose new blocks, weighted by how much they have staked. If a validator tries to cheat, their stake gets 'slashed' — they lose real money. PoS uses 99.9% less energy than PoW and can process more transactions per second. Ethereum switched from PoW to PoS in September 2022 (called 'The Merge').",
          bull, surface, text_color)

    _card("Delegated Proof of Stake (DPoS) — Solana, Cardano",
          "A variation where token holders vote for a smaller set of validators. Faster than standard PoS because fewer validators need to agree. Solana uses a version of this combined with Proof of History (a cryptographic clock) to achieve 400+ transactions per second. The tradeoff: fewer validators means more centralization.",
          neutral, surface, text_color)

    st.subheader("Key Blockchain Concepts")

    _card("Nodes",
          "A node is any computer running the blockchain software. Full nodes store the entire blockchain history and validate every transaction independently. Light nodes store only block headers and rely on full nodes for verification. More nodes means more decentralization and security. Bitcoin has roughly 15,000 full nodes worldwide. Anyone can run one.",
          primary, surface, text_color)

    _card("Mining and Validators",
          "In PoW chains, miners use specialized hardware (ASICs for Bitcoin, GPUs for some others) to compete for block rewards. In PoS chains, validators stake tokens and earn rewards proportional to their stake. Both serve the same purpose: securing the network and processing transactions. The economics of mining and staking directly affect token supply and price.",
          bull, surface, text_color)

    _card("Hash Rate",
          "The total computing power securing a PoW blockchain, measured in hashes per second. Bitcoin's hash rate is measured in exahashes (quintillions of hashes per second). Higher hash rate means more security. A sudden drop in hash rate can signal that miners are shutting down (unprofitable) or a major disruption. SENTINEL tracks hash rate as a security indicator.",
          neutral, surface, text_color)

    _card("Block Time and Finality",
          "Block time is how often new blocks are created. Bitcoin: ~10 minutes. Ethereum: ~12 seconds. Solana: ~400 milliseconds. But block time is not the same as finality. Finality means the transaction is irreversible. Bitcoin is considered final after 6 confirmations (~60 minutes). Ethereum after 2 epochs (~13 minutes). Faster finality means faster settlements, which matters for exchanges and DeFi.",
          primary, surface, text_color)

    _card("Gas Fees",
          "The cost to execute a transaction on the blockchain. Think of it like postage — you pay to use the network. On Ethereum, gas fees fluctuate with demand. During busy periods (NFT mints, market crashes), fees can spike to $50-$200 per transaction. During quiet periods, fees drop to $1-$5. Solana fees are fractions of a cent. Gas fees are paid in the chain's native token (ETH for Ethereum, SOL for Solana).",
          bear, surface, text_color)

    st.subheader("Forks — When Blockchains Split")

    _card("Hard Forks vs Soft Forks",
          "A fork happens when a blockchain's rules change. A soft fork is backward-compatible — old nodes can still participate. A hard fork is not backward-compatible — the chain splits into two separate chains. Famous hard forks: Bitcoin Cash split from Bitcoin in 2017 over a disagreement about block size. Ethereum Classic split from Ethereum in 2016 after The DAO hack. Forks can create new tokens and cause significant price volatility.",
          bear, surface, text_color)

    st.subheader("Smart Contracts")

    _card("Programs That Run on the Blockchain",
          "A smart contract is code that executes automatically when conditions are met. Ethereum introduced this concept in 2015. Example: 'If Alice sends 1 ETH to this address, automatically send her 1000 USDC.' No middleman needed. Smart contracts power DeFi (decentralized finance), NFTs, DAOs, and most of the crypto ecosystem beyond simple transfers. Solidity is the main programming language for Ethereum smart contracts.",
          bull, surface, text_color)

    _card("Why Smart Contracts Matter for Prices",
          "The more useful smart contracts a blockchain runs, the more demand for its native token (needed to pay gas fees). Total Value Locked (TVL) measures how much money is deposited in a chain's smart contracts. Higher TVL generally means more adoption and demand for the native token. Ethereum has the highest TVL by far, which supports ETH's value.",
          primary, surface, text_color)


# ═══════════════════════════════════════════════════════════
# MODULE 26: TOKENOMICS — SUPPLY, BURNS & WHY COINS MOVE
# ═══════════════════════════════════════════════════════════
def _module_tokenomics(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 11: Tokenomics — Supply, Burns & Why Coins Move")
    st.caption("Understand the economics behind every token")

    st.subheader("Why Tokenomics Matters More Than Hype")

    _card("The Most Important Lesson in Crypto",
          "Price is determined by supply and demand. Tokenomics is the study of a token's supply — how many exist, how many will exist, how fast new ones are created, and whether any are being destroyed. You can have the most exciting project in the world, but if it prints 10% more tokens every year and dumps them on the market, the price goes down. Many traders lose money because they chase hype without checking the supply schedule.",
          bear, surface, text_color)

    st.subheader("Supply Metrics You Must Know")

    _card("Circulating Supply",
          "The number of tokens currently available and trading on the market right now. This is what matters for current price. Market cap = circulating supply times price. Example: if a token has 100 million circulating supply and trades at $10, market cap is $1 billion.",
          primary, surface, text_color)

    _card("Total Supply",
          "All tokens that currently exist, including those locked in vesting contracts, staking, or team wallets. Total supply is always greater than or equal to circulating supply. The gap between circulating and total supply tells you how much potential selling pressure exists from unlocks.",
          neutral, surface, text_color)

    _card("Max Supply",
          "The absolute maximum number of tokens that will ever exist. Bitcoin's max supply is 21 million — no more can ever be created. Ethereum has no max supply, but its burn mechanism (EIP-1559) can make it deflationary. Some tokens have no max supply at all, meaning unlimited inflation. Always check max supply before investing.",
          bull, surface, text_color)

    _card("Fully Diluted Valuation (FDV)",
          "Market cap calculated using max supply instead of circulating supply. FDV = max supply times current price. If a token has $1B market cap but $10B FDV, that means 90% of tokens have not entered circulation yet. Those tokens will eventually hit the market. A huge gap between market cap and FDV is a warning sign — it means massive future dilution.",
          bear, surface, text_color)

    st.subheader("Token Distribution — Who Holds the Coins?")

    _card("Founder and Team Allocation",
          "Most projects allocate 15-25% of tokens to the founding team. These tokens are usually locked with a vesting schedule (e.g., 4-year vest with 1-year cliff). When team tokens unlock, founders often sell — creating selling pressure. Always check when the next team unlock happens. A major unlock event can crash a token 20-40% if the market is not expecting it.",
          bear, surface, text_color)

    _card("Venture Capital (VC) Allocation",
          "VCs who funded the project early get tokens at massive discounts — often 10-100x below the public price. When their tokens unlock, they almost always sell because they are already in huge profit. VC unlock schedules are the single biggest source of predictable selling pressure in crypto. Check the token's documentation for the vesting schedule.",
          bear, surface, text_color)

    _card("Community and Ecosystem",
          "Tokens reserved for airdrops, liquidity mining, grants, and community incentives. These tokens enter circulation over time. Large community allocations can create sustained selling pressure but also drive adoption. The best projects balance distribution — enough to incentivize users, not so much that it crashes the price.",
          neutral, surface, text_color)

    _card("Treasury",
          "Tokens held by the project's DAO or foundation. Used to fund development, partnerships, and marketing. Treasury tokens are not in circulation but can be sold by governance vote. Large treasuries can be bullish (well-funded project) or bearish (potential dump). Check how the treasury is managed and what governance controls exist.",
          neutral, surface, text_color)

    st.subheader("Inflation and Deflation")

    _card("Emission Rate (Inflation)",
          "How fast new tokens are created. Bitcoin emits 3.125 BTC per block (about 450 BTC per day). This rate halves every 4 years (the halving). Ethereum emits about 1,700 ETH per day to validators. High emission rates dilute existing holders — your tokens represent a smaller percentage of total supply over time. Annual inflation above 5-10% is a red flag for long-term holdings.",
          primary, surface, text_color)

    _card("Token Burns (Deflation)",
          "When tokens are permanently destroyed, reducing total supply. Ethereum burns a portion of every gas fee (EIP-1559, implemented August 2021). During high-activity periods, Ethereum burns more ETH than it emits — making it deflationary. BNB does quarterly burns. Some projects do buyback-and-burn using revenue. Burns reduce supply, which can support price if demand stays constant.",
          bull, surface, text_color)

    _card("The Halving (Bitcoin-Specific)",
          "Every 210,000 blocks (roughly 4 years), Bitcoin's block reward is cut in half. The most recent halving was April 2024 (6.25 BTC to 3.125 BTC). Halvings reduce new supply by 50%, which historically triggers massive bull runs 12-18 months later. The next halving is expected around 2028. This is the most predictable supply event in all of crypto.",
          bull, surface, text_color)

    st.subheader("Staking Economics")

    _card("Staking Yield and Its Effect on Supply",
          "When users stake tokens, they lock them up and remove them from circulating supply. High staking participation (like Ethereum at ~28% staked) reduces available supply on exchanges, which can support price. Staking rewards are new token emissions — so the yield comes from inflation. A 5% staking yield with 5% inflation means you are just keeping up, not actually earning. Always compare staking yield to inflation rate.",
          primary, surface, text_color)

    _card("Liquid Staking",
          "Protocols like Lido let you stake ETH and receive a liquid token (stETH) that you can trade or use in DeFi while still earning staking rewards. Liquid staking has grown massively — Lido alone holds over 30% of all staked ETH. This changes the economics because staked tokens are no longer truly locked. Watch liquid staking ratios as a measure of market sentiment.",
          neutral, surface, text_color)

    st.subheader("Red Flags in Tokenomics")

    red_flags = [
        "FDV is 5x or more above market cap — massive future dilution incoming.",
        "Team allocation above 30% — too much insider control.",
        "No vesting schedule or very short vesting — insiders can dump immediately.",
        "Annual inflation above 10% with no burn mechanism — your tokens lose value every day.",
        "Major VC unlocks within 3-6 months — predictable selling pressure ahead.",
        "Top 10 wallets hold more than 50% of supply — extreme concentration risk.",
        "No max supply and no burn mechanism — unlimited money printing.",
        "Project revenue does not flow back to token holders — the token has no value capture.",
    ]
    for i, flag in enumerate(red_flags, 1):
        st.markdown(f"**{i}.** {flag}")


# ═══════════════════════════════════════════════════════════
# MODULE 27: READING CRYPTO MARKETS — FUNDING, OI & WHALES
# ═══════════════════════════════════════════════════════════
def _module_crypto_markets(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 13: Reading Crypto Markets — Funding, OI & Whales")
    st.caption("The indicators that separate retail from smart money")

    st.subheader("Crypto Markets Are Different")

    _card("Why Traditional TA Is Not Enough",
          "Crypto markets trade 24/7/365 with no circuit breakers, no market makers of last resort, and massive leverage available to retail traders. The result: faster moves, wilder swings, and unique indicators that do not exist in stock markets. Funding rates, open interest, liquidation cascades, and whale wallet tracking are tools specific to crypto. Learn them and you have an edge over 90% of crypto traders.",
          primary, surface, text_color)

    st.subheader("Perpetual Futures and Funding Rates")

    _card("What Are Perpetual Futures (Perps)?",
          "A perpetual future is a derivative contract that lets you bet on a crypto's price without owning the actual coin. Unlike traditional futures, perps have no expiration date — you can hold them forever. Perps are the most traded instrument in crypto, with daily volumes often 5-10x spot volume. Binance, Bybit, and dYdX are the biggest perp exchanges.",
          primary, surface, text_color)

    _card("Funding Rate — The Most Important Crypto Indicator",
          "Funding rate is a periodic payment between long and short traders that keeps the perp price close to the spot price. If the funding rate is positive, longs pay shorts — meaning the market is over-leveraged to the upside. If funding is negative, shorts pay longs — the market is over-leveraged to the downside. Extremely high positive funding (above 0.05% per 8 hours or 0.15% per day) often precedes sharp corrections because the market is too one-sided. Negative funding during a rally is bullish — shorts are paying longs and the rally has room to run.",
          bull, surface, text_color)

    _card("How to Use Funding Rate",
          "Consistently high positive funding (several days of 0.03%+): the long trade is crowded. Expect a flush down to liquidate over-leveraged longs. Consistently negative funding during an uptrend: shorts are fighting the trend and paying for it. This often fuels further upside as shorts get squeezed. Funding rate flipping from positive to negative (or vice versa): the market is resetting. Watch for the new direction.",
          neutral, surface, text_color)

    st.subheader("Open Interest — How Much Money Is in the Game")

    _card("What Is Open Interest (OI)?",
          "Open interest is the total value of all outstanding derivative contracts (futures and options). Rising OI means new money is entering the market. Falling OI means positions are being closed. OI is different from volume — volume measures activity, OI measures commitment.",
          primary, surface, text_color)

    _card("OI + Price Combinations",
          "Price up + OI up: New longs are opening. The move is backed by new money. Bullish. | Price up + OI down: Short positions are closing (short squeeze). The move may lack follow-through. | Price down + OI up: New shorts are opening. Bears are committing capital. Bearish. | Price down + OI down: Long positions are closing (long liquidation). If OI drops sharply, the selling may be exhausted.",
          bull, surface, text_color)

    _card("OI Extremes",
          "When OI reaches all-time highs, the market is maximally leveraged. This is when you get the biggest liquidation cascades. All-time high OI with high positive funding is the classic setup for a violent correction — sometimes 15-30% in hours. Watch for OI extremes on aggregated data (not just one exchange).",
          bear, surface, text_color)

    st.subheader("Liquidations — Forced Selling Cascades")

    _card("What Are Liquidations?",
          "When a leveraged trader's losses exceed their collateral, the exchange forcibly closes their position. This is a liquidation. At 10x leverage, a 10% move against you means total liquidation. At 50x leverage, a 2% move wipes you out. Liquidations create forced selling (or forced buying for short liquidations), which can cascade — one liquidation pushes the price further, triggering more liquidations.",
          bear, surface, text_color)

    _card("Liquidation Maps",
          "Tools like Coinglass and Hyblock show where liquidation clusters are concentrated. If $500M in long liquidations are stacked at $58,000 BTC and the price is at $60,000, there is a 'magnet' pulling price toward $58,000 — because market makers know that breaking that level triggers forced selling that they can profit from. Liquidation maps help you identify where the market is likely to 'hunt' — and where you should NOT place your stop losses.",
          primary, surface, text_color)

    st.subheader("Whale Watching")

    _card("Exchange Inflows and Outflows",
          "When large amounts of crypto move FROM wallets TO exchanges, it often signals intent to sell (bearish). When crypto moves FROM exchanges TO wallets, holders are taking custody — they are not planning to sell (bullish). Bitcoin exchange reserves at multi-year lows is one of the most bullish on-chain signals. Track net exchange flow for major coins.",
          bull, surface, text_color)

    _card("Whale Wallet Tracking",
          "Whales are wallets holding large amounts of crypto (100+ BTC, 10,000+ ETH). Their behavior often predicts market moves. Whales accumulating during fear is bullish. Whales sending to exchanges during euphoria is bearish. Services like Whale Alert, Arkham Intelligence, and Nansen track whale movements in real time. SENTINEL incorporates these signals in its crypto analysis.",
          primary, surface, text_color)

    _card("Stablecoin Supply on Exchanges",
          "Stablecoins sitting on exchanges represent dry powder — money ready to buy crypto. Rising stablecoin balances on exchanges is bullish because it means buying power is building. Declining stablecoin balances means the buying power is being used up or withdrawn. USDT and USDC supply on exchanges is a leading indicator of potential rallies.",
          bull, surface, text_color)

    st.subheader("Order Book and Market Microstructure")

    _card("Bid-Ask Spread in Crypto",
          "Major coins like BTC and ETH on top exchanges have tight spreads (fractions of a cent). Altcoins on smaller exchanges can have spreads of 1-5%. Wide spreads mean less liquidity and higher slippage — your market orders will fill at worse prices. Always check the spread and order book depth before trading smaller coins.",
          neutral, surface, text_color)

    _card("Spoofing and Fake Walls",
          "Crypto order books are full of fake orders designed to manipulate sentiment. A large sell wall at $60,000 makes traders think 'it will never break through' — then the wall disappears seconds before price reaches it. This is called spoofing. Never base trading decisions on order book walls alone. Volume and actual executions matter more than displayed orders.",
          bear, surface, text_color)


# ═══════════════════════════════════════════════════════════
# MODULE 28: THE CRYPTO CYCLE — BTC SEASON, ALT SEASON
# ═══════════════════════════════════════════════════════════
def _module_crypto_cycles(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 14: The Crypto Cycle — BTC Season, Alt Season & Narratives")
    st.caption("Crypto moves in predictable cycles — learn to ride them")

    st.subheader("The Four-Year Cycle")

    _card("Bitcoin's Halving Cycle",
          "Bitcoin has followed a roughly four-year cycle tied to the halving event. The pattern: halving reduces new supply → supply shock → price rises → euphoria attracts new buyers → bubble → crash → accumulation → next halving. This has played out in 2012, 2016, 2020, and 2024. While past performance does not guarantee future results, the supply mechanics are mathematical, not speculative. The halving literally cuts new BTC production in half.",
          primary, surface, text_color)

    _card("Cycle Phases",
          "Accumulation (6-12 months after a crash): Smart money buys while everyone else is scared. Low volume, low volatility, maximum despair. | Early Bull (3-6 months): Price starts rising but most people do not notice. Disbelief rallies — every pump is called a dead cat bounce. | Parabolic Bull (3-6 months): Mainstream media coverage, new retail investors, exponential price increases. This is when everyone talks about crypto at dinner parties. | Distribution and Crash (3-12 months): Smart money sells to retail. Price collapses 70-85% from the peak. The cycle resets.",
          bull, surface, text_color)

    st.subheader("BTC Season vs Alt Season")

    _card("What Is BTC Dominance?",
          "BTC dominance measures Bitcoin's market cap as a percentage of total crypto market cap. When BTC dominance is rising, Bitcoin is outperforming altcoins — this is 'BTC season.' When BTC dominance is falling, altcoins are outperforming Bitcoin — this is 'alt season.' BTC dominance typically ranges between 40% and 70%. At cycle tops, dominance is usually low (alt season euphoria). At cycle bottoms, dominance is high (flight to quality).",
          primary, surface, text_color)

    _card("The Rotation Pattern",
          "In a typical bull cycle, money flows in a specific order: Bitcoin first (institutional money and largest market cap). Then Ethereum (the smart contract platform). Then large-cap altcoins (SOL, ADA, AVAX). Then mid-caps. Then small-caps and meme coins. Finally, everything crashes. This rotation happens because each successive category is riskier and more speculative. Smart money buys BTC early. Retail buys meme coins at the top.",
          bull, surface, text_color)

    _card("Identifying Alt Season",
          "Alt season is confirmed when 75% of the top 50 altcoins outperform Bitcoin over a 90-day period. Key signals: BTC dominance falling below 50%. ETH/BTC ratio rising. Social media dominated by altcoin talk. Your uber driver asks about a specific altcoin. Alt season is extremely profitable if you enter early and extremely destructive if you enter late.",
          neutral, surface, text_color)

    st.subheader("Narrative Trading")

    _card("What Are Crypto Narratives?",
          "Crypto moves on narratives — compelling stories that attract capital. Narratives create temporary demand for specific categories of tokens. Examples: DeFi Summer (2020), NFTs (2021), Layer 2 scaling (2023), AI tokens (2024), Real World Assets / RWA (2024-2025). Narrative traders identify which story the market will care about next and position before the crowd.",
          primary, surface, text_color)

    _card("Anatomy of a Narrative",
          "Stage 1 — Innovation: A new technology or concept emerges. Only insiders know about it. Stage 2 — Early Adopters: Smart money starts buying. Crypto Twitter starts discussing it. Stage 3 — Mainstream Crypto Attention: Major exchanges list related tokens. Prices explode 5-50x. Stage 4 — Retail FOMO: Everyone is talking about it. New projects launch daily in the category. Stage 5 — Exhaustion: The narrative is fully priced in. Insiders sell. 90% of projects in the category go to zero.",
          bull, surface, text_color)

    _card("Current and Emerging Narratives",
          "Narratives that have already played out are not opportunities — they are traps. The money is made by identifying the NEXT narrative. Look for: real technological breakthroughs (not just marketing), growing developer activity (GitHub commits), increasing TVL in new categories, and regulatory tailwinds. SENTINEL's news intelligence can help you spot emerging narratives early.",
          neutral, surface, text_color)

    st.subheader("Macro Correlation")

    _card("Crypto and the Federal Reserve",
          "Since 2020, crypto has become increasingly correlated with macro factors. When the Fed prints money (quantitative easing), crypto rallies. When the Fed tightens (raising rates, quantitative tightening), crypto falls. Bitcoin's correlation with the Nasdaq reached 0.8+ during 2022. This correlation weakens during extreme crypto-specific events but strengthens during macro-driven markets. Always check what the Fed is doing.",
          primary, surface, text_color)

    _card("Dollar Strength (DXY)",
          "The US Dollar Index (DXY) measures the dollar against a basket of major currencies. Crypto and DXY are generally inversely correlated — strong dollar means weak crypto, weak dollar means strong crypto. When DXY breaks below key support levels, it often coincides with major crypto rallies. SENTINEL tracks DXY in its cross-asset correlation analysis.",
          bear, surface, text_color)

    _card("Global Liquidity",
          "The total amount of money in the global financial system. When central banks worldwide are adding liquidity (printing money), risk assets including crypto rally. When liquidity is being drained, risk assets fall. Global M2 money supply is the broadest measure. Crypto traders who ignore macro liquidity conditions get destroyed in tightening cycles.",
          neutral, surface, text_color)


# ═══════════════════════════════════════════════════════════
# MODULE 29: DEFI MASTERCLASS — POOLS, YIELD & REAL RISKS
# ═══════════════════════════════════════════════════════════
def _module_defi_masterclass(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 15: DeFi Masterclass — Pools, Yield & Real Risks")
    st.caption("Decentralized finance — the most powerful and dangerous part of crypto")

    st.subheader("What Is DeFi?")

    _card("Finance Without Banks",
          "Decentralized Finance (DeFi) replaces banks, brokers, and exchanges with smart contracts on a blockchain. Instead of a bank holding your deposit and lending it out, a smart contract does it automatically. Instead of a stock exchange matching buyers and sellers, an automated market maker does it with math. DeFi is permissionless (anyone can use it), transparent (all code is public), and composable (protocols can plug into each other like Legos).",
          primary, surface, text_color)

    _card("Total Value Locked (TVL)",
          "TVL measures all the money deposited in DeFi protocols. It is the primary metric for DeFi adoption. Peak TVL was about $180 billion in November 2021. It crashed to $40 billion by late 2022. Rising TVL means capital is flowing into DeFi. Falling TVL means capital is leaving. Ethereum holds the majority of TVL, but Solana, Arbitrum, and Base are growing fast.",
          bull, surface, text_color)

    st.subheader("Decentralized Exchanges (DEXs)")

    _card("Automated Market Makers (AMMs)",
          "Traditional exchanges use order books — buyers and sellers place orders. DEXs use a different model called an Automated Market Maker. Instead of matching orders, an AMM uses liquidity pools — two tokens deposited in a smart contract. The price is determined by a mathematical formula (usually x * y = k, where x and y are the quantities of each token). When you trade, you swap against the pool, and the price adjusts automatically based on supply and demand within the pool.",
          primary, surface, text_color)

    _card("Major DEXs",
          "Uniswap (Ethereum) — the original AMM, largest DEX by volume. SushiSwap — Uniswap fork with added features. Curve — specialized for stablecoin and similar-asset swaps with very low slippage. Jupiter (Solana) — the dominant Solana DEX aggregator. Raydium (Solana) — major Solana AMM. PancakeSwap (BNB Chain) — largest on BNB Chain.",
          neutral, surface, text_color)

    st.subheader("Liquidity Providing (LP)")

    _card("How Liquidity Providing Works",
          "You deposit two tokens into a liquidity pool in equal dollar value (e.g., $500 of ETH + $500 of USDC). Traders swap against your pool, and you earn a share of the trading fees (usually 0.3% per trade). Your share is proportional to your percentage of the total pool. Sounds great — free money from fees. But there is a catch called impermanent loss.",
          bull, surface, text_color)

    _card("Impermanent Loss — The Hidden Cost",
          "When you provide liquidity, if one token's price changes relative to the other, you end up with more of the cheaper token and less of the expensive one. This is called impermanent loss. Example: you deposit ETH/USDC. ETH doubles. The AMM formula rebalances your position — you now have less ETH and more USDC than if you had just held. The loss is 'impermanent' because if the price returns to the original ratio, it disappears. But in practice, prices rarely return exactly. The bigger the price move, the bigger the impermanent loss. A 2x price move causes about 5.7% impermanent loss. A 5x move causes about 25%.",
          bear, surface, text_color)

    _card("When LP Is Worth It",
          "Liquidity providing is profitable when: trading fees exceed impermanent loss, the two tokens are stable relative to each other (like two stablecoins), or you receive additional token rewards (liquidity mining). Stablecoin pools (USDC/USDT) have minimal impermanent loss and can earn 3-15% APY. Volatile pairs require much higher fee income to be profitable.",
          neutral, surface, text_color)

    st.subheader("Lending and Borrowing")

    _card("DeFi Lending Protocols",
          "Protocols like Aave, Compound, and Morpho let you lend your crypto and earn interest, or borrow against your crypto holdings. You deposit collateral (e.g., ETH) and can borrow against it (e.g., USDC). The loan-to-value ratio is usually 70-80% — you can borrow $700 worth of USDC against $1,000 of ETH. Interest rates are variable and set by supply/demand algorithms.",
          primary, surface, text_color)

    _card("Liquidation in DeFi Lending",
          "If your collateral value drops below the required ratio, your position gets liquidated. A liquidation bot buys your collateral at a discount to repay your loan. You lose your collateral plus a liquidation penalty (usually 5-10%). This is why DeFi lending is dangerous with volatile collateral. If ETH drops 30% overnight, your position may be liquidated before you can add more collateral.",
          bear, surface, text_color)

    st.subheader("Yield Farming")

    _card("What Is Yield Farming?",
          "Yield farming is depositing tokens into DeFi protocols to earn rewards, often paid in the protocol's governance token. During DeFi Summer 2020, yields of 100-1,000% APY were common. These yields were unsustainable — they were paid from token emissions (inflation), not real revenue. Most yield farming tokens lost 90-99% of their value.",
          neutral, surface, text_color)

    _card("Real Yield vs Token Emission Yield",
          "Real yield comes from actual protocol revenue — trading fees, lending interest, liquidation penalties. Token emission yield comes from newly minted governance tokens. Real yield is sustainable. Token emission yield is not — it dilutes the governance token and eventually collapses. Always ask: where does the yield come from? If the answer is 'new token emissions,' the yield is temporary.",
          bull, surface, text_color)

    st.subheader("DeFi Risks — What Can Go Wrong")

    risks = [
        "Smart contract bugs: Code is public but can still have vulnerabilities. Billions have been lost to hacks. Even audited contracts can be exploited.",
        "Rug pulls: The developer drains the liquidity pool and disappears with user funds. Common in new, unaudited projects.",
        "Oracle manipulation: DeFi protocols rely on price oracles (data feeds). If the oracle is manipulated, attackers can drain protocols.",
        "Governance attacks: Someone acquires enough governance tokens to pass a malicious proposal. Flash loan attacks enable this with zero upfront capital.",
        "Regulatory risk: Governments may classify DeFi tokens as unregistered securities. The SEC has already taken action against several protocols.",
        "Impermanent loss: As covered above, price divergence between paired tokens erodes LP returns.",
        "Gas fee spikes: During market crashes, Ethereum gas fees can spike to $200+. Your $500 DeFi position may cost $200 to exit.",
        "Composability risk: Because DeFi protocols plug into each other, a failure in one can cascade through others. This happened during the Terra/LUNA collapse in May 2022.",
    ]
    for i, risk in enumerate(risks, 1):
        st.markdown(f"**{i}.** {risk}")


# ═══════════════════════════════════════════════════════════
# MODULE 30: CRYPTO SECURITY — SCAMS, HACKS & PROTECTION
# ═══════════════════════════════════════════════════════════
def _module_crypto_security(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 17: Crypto Security — Scams, Hacks & Protection")
    st.caption("In crypto, YOU are the bank — security is your responsibility")

    st.subheader("The Golden Rule")

    _card("Not Your Keys, Not Your Coins",
          "If your crypto is on an exchange (Coinbase, Binance, Kraken), the exchange controls your private keys. If the exchange gets hacked, goes bankrupt, or freezes withdrawals — you lose everything. FTX users lost $8 billion in November 2022. Mt. Gox users lost $450 million in 2014. Celsius, Voyager, and BlockFi all went bankrupt in 2022. The only way to truly own your crypto is to hold your own private keys in a self-custody wallet.",
          bear, surface, text_color)

    st.subheader("Wallet Types")

    _card("Hot Wallets (Software Wallets)",
          "Software that runs on your phone or computer. Examples: MetaMask (browser extension, Ethereum), Phantom (Solana), Trust Wallet (multi-chain). Hot wallets are connected to the internet, which makes them convenient but vulnerable. Use hot wallets for small amounts you actively trade or use in DeFi. Never keep your life savings in a hot wallet.",
          neutral, surface, text_color)

    _card("Cold Wallets (Hardware Wallets)",
          "Physical devices that store your private keys offline. Examples: Ledger Nano X, Ledger Nano S Plus, Trezor Model T. The private key never touches the internet — transactions are signed on the device itself. This makes hardware wallets virtually immune to remote hacking. Use cold wallets for any amount you cannot afford to lose. A $79 Ledger protects against millions in potential losses.",
          bull, surface, text_color)

    _card("Seed Phrase — The Master Key",
          "When you create a wallet, you receive a 12 or 24-word seed phrase (also called recovery phrase or mnemonic). This phrase IS your wallet. Anyone who has your seed phrase has complete control of all your crypto. If you lose your seed phrase, you lose access to your crypto forever — there is no 'forgot password' option. Write it on paper or stamp it on metal. Never store it digitally (no photos, no cloud, no notes app). Never share it with anyone for any reason.",
          bear, surface, text_color)

    st.subheader("Common Scams — How People Lose Money")

    _card("Phishing Attacks",
          "Fake websites that look identical to real ones. You connect your wallet and approve a malicious transaction that drains all your tokens. Common targets: Uniswap, OpenSea, MetaMask. Always verify the URL manually. Bookmark legitimate sites. Never click links from Discord, Telegram, or email. The real MetaMask will never DM you.",
          bear, surface, text_color)

    _card("Approval Exploits (Token Approvals)",
          "When you interact with a DeFi protocol, you grant it permission to move your tokens. Many protocols request 'unlimited approval' — meaning they can move your tokens forever. If that protocol gets hacked, the hacker can drain your wallet using your old approvals. Use tools like Revoke.cash to regularly check and revoke unused approvals. Set limited approvals when possible.",
          bear, surface, text_color)

    _card("Rug Pulls",
          "The developer creates a token, hypes it on social media, waits for people to buy, then pulls all liquidity from the pool and disappears. You are left holding a worthless token with no liquidity to sell. Red flags: anonymous team, no audit, locked liquidity is missing, extremely high APY promises, aggressive shilling, the contract has a function that lets the deployer drain funds.",
          bear, surface, text_color)

    _card("Pump and Dump Groups",
          "Telegram or Discord groups that coordinate buying a low-cap token to inflate the price, then sell to latecomers. The group organizers buy before the announcement. By the time you see the 'signal,' you are the exit liquidity. These groups are illegal in regulated markets and should be avoided completely.",
          bear, surface, text_color)

    _card("Fake Customer Support",
          "Someone on Discord or Telegram pretending to be from the project team or exchange. They ask you to 'verify your wallet' or 'connect to this link for support.' This is always a scam. Real support teams will never ask for your seed phrase or ask you to connect your wallet to an unknown site.",
          bear, surface, text_color)

    _card("Honeypot Tokens",
          "A token where the smart contract allows buying but blocks selling. You buy, see the price going up, try to sell, and cannot. Your money is trapped forever. The deployer is the only one who can sell. Always check if a token can be sold before buying significant amounts. Tools like Token Sniffer and GoPlus can analyze contracts for honeypot code.",
          bear, surface, text_color)

    st.subheader("Security Best Practices")

    practices = [
        "Use a hardware wallet for any significant holdings. This is non-negotiable.",
        "Never share your seed phrase. Not with support, not with friends, not with anyone claiming to be from the project.",
        "Store your seed phrase on paper or metal in a secure location. Never digitally.",
        "Use a separate browser or browser profile for crypto. Keep your crypto browser clean — no random extensions.",
        "Revoke token approvals regularly using Revoke.cash. Old approvals are ticking time bombs.",
        "Verify every URL manually. Bookmark all sites you use regularly. Phishing sites look identical to real ones.",
        "Enable 2FA on every exchange account. Use an authenticator app (Google Authenticator, Authy), NOT SMS.",
        "Use a dedicated email address for crypto exchanges. Do not use your personal or work email.",
        "Start with small test transactions before sending large amounts. A $5 test can save you from a $50,000 mistake.",
        "If someone contacts you about an 'opportunity,' it is a scam. 100% of the time.",
    ]
    for i, practice in enumerate(practices, 1):
        st.markdown(f"**{i}.** {practice}")


# ═══════════════════════════════════════════════════════════
# MODULE 31: MAJOR CHAINS — ETHEREUM, SOLANA, AND BEYOND
# ═══════════════════════════════════════════════════════════
def _module_major_chains(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 12: Major Chains — Ethereum, Solana, and Beyond")
    st.caption("Understanding the platforms that power the crypto ecosystem")

    st.subheader("Layer 1 vs Layer 2")

    _card("What Is a Layer 1?",
          "A Layer 1 (L1) is a base blockchain that processes and finalizes transactions on its own. Bitcoin, Ethereum, Solana, Avalanche, and Cardano are all Layer 1s. Each L1 has its own consensus mechanism, validator set, and native token. L1s compete on speed, cost, security, and developer ecosystem.",
          primary, surface, text_color)

    _card("What Is a Layer 2?",
          "A Layer 2 (L2) is built on top of a Layer 1 to increase speed and reduce costs. L2s process transactions off the main chain but settle them back on the L1 for security. Ethereum has the most L2 activity: Arbitrum, Optimism, Base, zkSync, and Starknet. L2s can process thousands of transactions per second at fractions of a penny while inheriting Ethereum's security. L2 adoption has exploded — some L2s now process more transactions than Ethereum itself.",
          bull, surface, text_color)

    _card("Rollups — How L2s Work",
          "Most Ethereum L2s use rollups — they bundle (roll up) hundreds of transactions into one batch and post it to Ethereum. Optimistic rollups (Arbitrum, Optimism, Base) assume transactions are valid unless challenged. Zero-knowledge rollups (zkSync, Starknet) use mathematical proofs to guarantee validity. ZK rollups are more technically advanced and potentially more secure, but optimistic rollups currently handle more volume.",
          neutral, surface, text_color)

    st.subheader("Ethereum — The Smart Contract King")

    _card("Ethereum Overview",
          "Created by Vitalik Buterin in 2015. Ethereum introduced smart contracts — programmable money. It hosts the vast majority of DeFi, NFTs, and tokenized assets. ETH is the second-largest cryptocurrency by market cap. Ethereum transitioned from Proof of Work to Proof of Stake in September 2022 (The Merge), reducing energy consumption by 99.9%. About 28% of all ETH is currently staked.",
          bull, surface, text_color)

    _card("Ethereum's Roadmap",
          "Ethereum is evolving in stages. The Merge (2022): PoW to PoS. Shanghai/Capella (2023): enabled staking withdrawals. Dencun (2024): introduced proto-danksharding (EIP-4844), reducing L2 fees by 90%+. Future upgrades focus on sharding (splitting the blockchain into parallel chains for higher throughput), statelessness (reducing node requirements), and account abstraction (making wallets easier to use). Ethereum's roadmap stretches to 2028+.",
          primary, surface, text_color)

    _card("EIP-1559 and ETH as an Asset",
          "Ethereum Improvement Proposal 1559, implemented in August 2021, changed how gas fees work. Instead of all fees going to validators, a base fee is burned (destroyed). During high activity periods, more ETH is burned than emitted, making ETH deflationary. This gives ETH a value proposition similar to a dividend-paying stock — you can stake for yield while supply potentially decreases. The 'ultrasound money' narrative emerged from this mechanism.",
          bull, surface, text_color)

    st.subheader("Solana — Speed and Low Cost")

    _card("Solana Overview",
          "Created by Anatoly Yakovenko in 2020. Solana prioritizes speed and low cost above all else. It processes 400+ transactions per second with sub-second finality and fees under $0.01. This makes Solana ideal for high-frequency applications: DEX trading, gaming, payments, and meme coins. SOL is a top-5 cryptocurrency by market cap.",
          bull, surface, text_color)

    _card("Proof of History",
          "Solana's innovation is Proof of History (PoH) — a cryptographic clock that timestamps transactions before consensus. This allows validators to process transactions in parallel without waiting to agree on ordering. The result is much higher throughput than chains that must agree on order first. The tradeoff: Solana requires powerful hardware to run a validator (128GB RAM, fast SSD, high bandwidth), which limits decentralization.",
          primary, surface, text_color)

    _card("Solana's Challenges",
          "Solana experienced multiple network outages in 2022-2023, with the blockchain halting for hours at a time. These outages damaged trust and raised questions about reliability. The network has been significantly more stable since late 2023. Solana also faced existential risk during the FTX collapse (FTX/Alameda held large SOL positions), but the ecosystem survived and thrived. Solana's meme coin culture (Bonk, WIF, POPCAT) has driven massive adoption but also controversy.",
          neutral, surface, text_color)

    st.subheader("Other Major Layer 1s")

    _card("Avalanche (AVAX)",
          "Uses a novel consensus mechanism (Avalanche consensus) that is fast and energy-efficient. Features subnets — custom blockchains built on Avalanche's network. Strong in institutional DeFi and real-world asset tokenization. C-Chain is EVM-compatible (Ethereum code runs on it).",
          neutral, surface, text_color)

    _card("Cardano (ADA)",
          "Founded by Charles Hoskinson (Ethereum co-founder). Known for peer-reviewed academic research before implementation. Uses Ouroboros PoS consensus. Slower development pace than competitors but emphasizes formal verification and security. Popular in Africa and developing markets.",
          neutral, surface, text_color)

    _card("Polygon (POL, formerly MATIC)",
          "Originally a sidechain for Ethereum, now building a comprehensive L2 ecosystem using ZK technology. Polygon has strong partnerships with major corporations (Starbucks, Nike, Reddit) for bringing real-world applications to blockchain. Very low fees and fast transactions.",
          neutral, surface, text_color)

    _card("BNB Chain (BNB)",
          "Binance's blockchain. EVM-compatible with low fees. Largest ecosystem outside of Ethereum and Solana by TVL. Centralized compared to other L1s (only 21 validators). Popular for retail DeFi and PancakeSwap.",
          neutral, surface, text_color)

    st.subheader("How to Evaluate a Blockchain")

    criteria = [
        "Transaction throughput (TPS): How many transactions per second can it handle? Ethereum L1: ~15 TPS. Solana: 400+ TPS. L2s: 2,000+ TPS.",
        "Finality time: How long until a transaction is irreversible? Ranges from 400ms (Solana) to 13 minutes (Ethereum).",
        "Transaction cost: Average fee per transaction. Ranges from $0.001 (Solana) to $1-50 (Ethereum L1, varies).",
        "Developer ecosystem: Number of active developers, GitHub activity, hackathon participation. Ethereum leads by a wide margin.",
        "Total Value Locked: How much capital is deployed on the chain. Indicates trust and adoption.",
        "Decentralization: Number of validators, geographic distribution, hardware requirements. More decentralized = more censorship-resistant.",
        "Security track record: Has the chain been exploited or experienced outages? How did the team respond?",
        "Ecosystem growth: New protocols launching, user growth, transaction volume trends.",
    ]
    for i, item in enumerate(criteria, 1):
        st.markdown(f"**{i}.** {item}")


# ═══════════════════════════════════════════════════════════
# MODULE 32: ON-CHAIN ANALYSIS — READING THE BLOCKCHAIN
# ═══════════════════════════════════════════════════════════
def _module_onchain_analysis(surface, text_color, bull, bear, neutral, primary, bg):
    st.header("Module 16: On-Chain Analysis — Reading the Blockchain")
    st.caption("The blockchain is a public ledger — learn to read it like a pro")

    st.subheader("What Is On-Chain Analysis?")

    _card("The Blockchain Tells the Truth",
          "Every transaction on a public blockchain is recorded permanently and viewable by anyone. On-chain analysis examines these transactions to understand what market participants are actually doing — not what they say they are doing. Are whales accumulating or distributing? Are long-term holders selling? Is exchange supply increasing? The blockchain cannot lie. This gives on-chain analysts a massive edge over traders who rely only on price charts and news.",
          primary, surface, text_color)

    st.subheader("Key On-Chain Metrics — Bitcoin")

    _card("HODL Waves",
          "HODL waves show the age distribution of all Bitcoin in existence. Each colored band represents coins that have not moved in a specific time period (1 day, 1 week, 1 month, 3 months, 6 months, 1 year, 2 years, etc.). When the 'young coin' bands expand (coins moving that have been dormant), it signals distribution — long-term holders are selling. When old coin bands grow, it signals accumulation — holders are NOT selling. At market tops, young coins dominate. At market bottoms, old coins dominate.",
          bull, surface, text_color)

    _card("Realized Price and MVRV Ratio",
          "Realized price is the average price at which all Bitcoin last moved (the average cost basis of the entire network). MVRV (Market Value to Realized Value) compares the current market cap to the realized market cap. MVRV above 3.5: market is overheated, holders sitting on massive unrealized profit — historically signals a top. MVRV below 1.0: market is trading below average cost basis — holders are underwater. Historically signals a bottom. This is one of the most reliable on-chain indicators.",
          bull, surface, text_color)

    _card("Net Unrealized Profit/Loss (NUPL)",
          "NUPL measures the overall profit or loss of all Bitcoin holders. Calculated as (market cap minus realized cap) divided by market cap. NUPL above 0.75: Euphoria/Greed — almost everyone is in profit. Market top zone. NUPL between 0.5-0.75: Belief/Optimism — healthy bull market. NUPL between 0-0.25: Hope/Fear — early recovery or late decline. NUPL below 0: Capitulation — the market is in net loss. Historically the best time to buy.",
          primary, surface, text_color)

    _card("Spent Output Profit Ratio (SOPR)",
          "SOPR measures whether coins being moved are in profit or loss. SOPR above 1: coins are being spent at a profit. SOPR below 1: coins are being spent at a loss. In bull markets, SOPR dipping to 1 and bouncing is a buying opportunity — holders refuse to sell at a loss. In bear markets, SOPR rising to 1 and getting rejected is a selling signal — holders sell as soon as they break even.",
          neutral, surface, text_color)

    st.subheader("Exchange Metrics")

    _card("Exchange Reserves",
          "The total amount of Bitcoin (or any crypto) held on exchange wallets. Declining exchange reserves: coins are moving to self-custody, reducing available supply. Bullish. Rising exchange reserves: coins are moving to exchanges, likely to be sold. Bearish. Bitcoin exchange reserves have been declining since 2020, reaching multi-year lows. This is one of the strongest long-term bullish signals.",
          bull, surface, text_color)

    _card("Exchange Net Flow",
          "The difference between coins flowing into and out of exchanges. Positive net flow (inflows > outflows): more coins arriving at exchanges. Selling pressure building. Negative net flow (outflows > inflows): more coins leaving exchanges. Buying and holding behavior. A single large inflow (whale deposit) can signal an incoming sell-off. Track this daily for major coins.",
          primary, surface, text_color)

    _card("Stablecoin Exchange Reserves",
          "While declining BTC exchange reserves are bullish, rising stablecoin reserves on exchanges are ALSO bullish — they represent dry powder waiting to buy. When both conditions align (low BTC on exchanges + high stablecoins on exchanges), the setup is extremely bullish. Maximum buying power with minimum selling pressure.",
          bull, surface, text_color)

    st.subheader("Miner Metrics")

    _card("Hash Rate",
          "Total computing power securing the Bitcoin network. Rising hash rate: miners are investing in hardware, confident in future profitability. Bullish for long-term price. Falling hash rate: miners shutting down, may need to sell BTC reserves to cover costs. Temporarily bearish. Hash rate reaching all-time highs is a sign of network strength and miner confidence.",
          bull, surface, text_color)

    _card("Miner Revenue and Reserves",
          "Miners earn BTC from block rewards and transaction fees. They must sell some BTC to cover electricity and hardware costs. When miner reserves are declining rapidly, it means miners are selling aggressively — usually during bear markets when margins are thin. When miner reserves stabilize or grow, selling pressure from miners decreases. The Puell Multiple (daily miner revenue vs 365-day average) helps identify when miner revenue is extreme in either direction.",
          primary, surface, text_color)

    st.subheader("Network Activity Metrics")

    _card("Active Addresses",
          "The number of unique addresses transacting on the network each day. Rising active addresses: growing adoption and usage. Bullish. Falling active addresses: declining interest. Bearish. Active addresses tend to lead price — they start rising before major bull runs and start declining before major tops. Think of it like foot traffic in a store.",
          bull, surface, text_color)

    _card("Transaction Volume",
          "The total dollar value of transactions on the blockchain. High volume with rising price confirms the move. High volume with flat price can indicate accumulation (bullish) or distribution (bearish). Very low volume with rising price is suspicious — the rally may lack genuine demand.",
          neutral, surface, text_color)

    _card("NVT Ratio (Network Value to Transactions)",
          "NVT is crypto's equivalent of the P/E ratio. It compares market cap to daily transaction volume. High NVT: the network is overvalued relative to its usage. The price is ahead of fundamentals. Low NVT: the network is undervalued relative to its usage. Strong fundamentals supporting the price. NVT Signal (a smoothed version) is more useful than raw NVT for identifying tops and bottoms.",
          primary, surface, text_color)

    st.subheader("On-Chain Analysis Tools")

    _card("Free Tools",
          "Glassnode (limited free tier): the gold standard for Bitcoin on-chain data. CryptoQuant: exchange flows, miner data, market indicators. IntoTheBlock: ownership concentration, large transactions, network stats. Blockchain.com: basic Bitcoin network stats. Etherscan/Solscan: individual transaction and wallet lookup for Ethereum and Solana.",
          neutral, surface, text_color)

    _card("What SENTINEL Tracks",
          "SENTINEL pulls on-chain data where available through free APIs and integrates it into the crypto analysis pages. Exchange flow trends, network activity, and key on-chain ratios feed into the composite conviction score. When on-chain data contradicts price action (e.g., exchange reserves dropping while price falls), SENTINEL flags the divergence — these are often the highest-conviction setups.",
          bull, surface, text_color)

    st.subheader("On-Chain Analysis Framework")

    framework = [
        "Check MVRV and NUPL for cycle positioning. Are we in euphoria or capitulation? This sets your macro bias.",
        "Check exchange reserves and net flows. Is supply moving to or from exchanges? This tells you about selling pressure.",
        "Check stablecoin reserves on exchanges. Is buying power building? This tells you about potential demand.",
        "Check active addresses and transaction volume. Is the network growing or shrinking? This confirms or denies the price trend.",
        "Check whale behavior. Are large holders accumulating or distributing? Follow the smart money.",
        "Check miner behavior. Are miners selling or holding? Miner capitulation often marks bottoms.",
        "Combine on-chain data with SENTINEL's technical and sentiment signals for the complete picture.",
        "On-chain analysis works best for Bitcoin and Ethereum. Altcoin on-chain data is less reliable due to smaller sample sizes.",
    ]
    for i, item in enumerate(framework, 1):
        st.markdown(f"**{i}.** {item}")

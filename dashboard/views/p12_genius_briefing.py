"""
SENTINEL — Page 12: Genius Market Briefing v2.
Comprehensive daily intelligence: scores EVERY sector, EVERY asset class, crypto.
Plain-English intelligence that tells you what every signal means and what to do.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
import plotly.graph_objects as go

from config.settings import COLORS
from features.analyst import generate_market_briefing


@st.cache_data(ttl=300, show_spinner=False)
def _cached_briefing() -> dict:
    """Cache the full briefing for 5 minutes."""
    return generate_market_briefing()


def _greeting() -> str:
    """Return time-based greeting for Eastern time."""
    hour = datetime.now(ZoneInfo("America/New_York")).hour
    if hour < 12:
        return "Good morning"
    elif hour < 17:
        return "Good afternoon"
    return "Good evening"


def render():
    st.markdown(
        f'<h2 style="color:{COLORS["text"]}; margin-bottom:-10px;">'
        f'{_greeting()}, Brandon</h2>',
        unsafe_allow_html=True,
    )
    st.title("Market Intelligence Briefing")
    st.caption("Your daily Wall Street strategist — every sector, every asset class, every signal scored and explained")

    surface = COLORS["surface"]
    text_color = COLORS["text"]
    bull_color = COLORS["bull"]
    bear_color = COLORS["bear"]
    neutral_color = COLORS["neutral"]
    primary = COLORS["primary"]
    bg = COLORS["background"]

    # Refresh controls
    rc1, rc2 = st.columns([3, 1])
    with rc2:
        if st.button("Refresh Now", type="primary", key="briefing_refresh"):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Scoring every sector, asset class, and crypto signal..."):
        briefing = _cached_briefing()

    # Show last-updated timestamp
    ts = briefing.get("timestamp", "")
    if ts:
        st.caption(f"Last updated: {ts}")

    # ═══════════════════════════════════════════════════════════
    # HEADLINE BANNER
    # ═══════════════════════════════════════════════════════════
    regime = briefing["regime"]
    risk_level = briefing["risk_level"]

    if "STRONG" in regime or "POSITIVE" in regime:
        regime_color = bull_color
    elif "WARNING" in regime or "NERVOUS" in regime or "WORSE" in regime:
        regime_color = bear_color
    else:
        regime_color = neutral_color

    if risk_level == "HIGH":
        risk_color = bear_color
    elif risk_level == "ELEVATED":
        risk_color = "#FF9800"
    elif risk_level == "MODERATE":
        risk_color = neutral_color
    else:
        risk_color = bull_color

    st.markdown(
        f"<div style='background:linear-gradient(135deg, {surface}, #1a1a2e);padding:30px;border-radius:16px;"
        f"border-left:6px solid {regime_color};margin-bottom:25px'>"
        f"<h2 style='margin:0 0 10px;color:{text_color};font-size:26px'>{briefing['headline']}</h2>"
        f"<div style='display:flex;gap:20px;margin-top:15px;flex-wrap:wrap'>"
        f"<span style='background:{regime_color};color:white;padding:6px 16px;border-radius:20px;font-size:13px;font-weight:600'>{regime}</span>"
        f"<span style='background:{risk_color};color:white;padding:6px 16px;border-radius:20px;font-size:13px;font-weight:600'>Risk: {risk_level} ({briefing['risk_score']}/100)</span>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ═══════════════════════════════════════════════════════════
    # WHAT THE SCORE MEANS (explainer)
    # ═══════════════════════════════════════════════════════════
    with st.expander("What does the 0-100 score mean?", expanded=False):
        st.markdown(
            f"""<div style='color:{text_color};font-size:14px;line-height:1.8'>
            <p>Every asset gets scored <b>0 to 100</b> based on 8 things, weighted by how reliable they are:</p>
            <table style='width:100%;border-collapse:collapse;margin:10px 0'>
            <tr style='border-bottom:1px solid #333'>
                <td style='padding:6px;color:{primary};font-weight:600'>Trend (20%)</td>
                <td style='padding:6px'>Is the price above or below its moving averages? Are the averages lined up bullish or bearish?</td>
            </tr>
            <tr style='border-bottom:1px solid #333'>
                <td style='padding:6px;color:{primary};font-weight:600'>Momentum (20%)</td>
                <td style='padding:6px'>Is the price going up over the last week, month, and 3 months? All three lining up = strong signal.</td>
            </tr>
            <tr style='border-bottom:1px solid #333'>
                <td style='padding:6px;color:{primary};font-weight:600'>MACD (15%)</td>
                <td style='padding:6px'>A popular indicator that catches when momentum is speeding up or slowing down. Positive and rising = bullish.</td>
            </tr>
            <tr style='border-bottom:1px solid #333'>
                <td style='padding:6px;color:{primary};font-weight:600'>RSI (12%)</td>
                <td style='padding:6px'>Measures if something is overbought or oversold. Smart twist: in strong uptrends, high RSI is good (confirms strength), not bad.</td>
            </tr>
            <tr style='border-bottom:1px solid #333'>
                <td style='padding:6px;color:{primary};font-weight:600'>Trend Strength (10%)</td>
                <td style='padding:6px'>ADX indicator — tells you HOW STRONG the trend is, not which direction. Strong trends (up or down) are more tradeable.</td>
            </tr>
            <tr style='border-bottom:1px solid #333'>
                <td style='padding:6px;color:{primary};font-weight:600'>Volume (8%)</td>
                <td style='padding:6px'>Is trading volume higher or lower than normal? High volume on an up day = real buying. High volume on a down day = real selling.</td>
            </tr>
            <tr style='border-bottom:1px solid #333'>
                <td style='padding:6px;color:{primary};font-weight:600'>Mean Reversion (8%)</td>
                <td style='padding:6px'>Bollinger Bands — when price stretches too far from average, it tends to snap back. Catches extremes.</td>
            </tr>
            <tr>
                <td style='padding:6px;color:{primary};font-weight:600'>Volatility (7%)</td>
                <td style='padding:6px'>When volatility gets unusually quiet, a big move is brewing. This catches those setups before they explode.</td>
            </tr>
            </table>
            <p style='margin-top:12px'><b>80-100</b> = Strong Buy &nbsp;|&nbsp; <b>65-80</b> = Buy &nbsp;|&nbsp; <b>55-65</b> = Lean Bullish &nbsp;|&nbsp; <b>45-55</b> = Neutral &nbsp;|&nbsp; <b>35-45</b> = Lean Bearish &nbsp;|&nbsp; <b>20-35</b> = Sell &nbsp;|&nbsp; <b>0-20</b> = Strong Sell</p>
            </div>""",
            unsafe_allow_html=True,
        )

    # ═══════════════════════════════════════════════════════════
    # TOP MOVERS — HIGH FLYERS & SINKING SHIPS
    # ═══════════════════════════════════════════════════════════
    top_movers = briefing.get("top_movers", {})
    high_flyers = top_movers.get("high_flyers", [])
    sinking_ships = top_movers.get("sinking_ships", [])

    if high_flyers or sinking_ships:
        mc1, mc2 = st.columns(2)

        with mc1:
            st.markdown(
                f"<h3 style='color:{bull_color};margin-bottom:10px'>Top 5 High Flyers Today</h3>",
                unsafe_allow_html=True,
            )
            for i, m in enumerate(high_flyers, 1):
                price = m.get("price", 0)
                p_str = f"${price:,.2f}" if price < 1000 else f"${price:,.0f}"
                if m.get("type") == "Crypto" and price < 1:
                    p_str = f"${price:.4f}"
                sector_tag = f"<span style='color:#888;font-size:11px;margin-left:6px'>{m.get('sector', '')}</span>" if m.get("sector") else ""
                st.markdown(
                    f"<div style='background:{surface};padding:12px 16px;border-radius:8px;margin-bottom:6px;"
                    f"border-left:4px solid {bull_color};display:flex;align-items:center;justify-content:space-between'>"
                    f"<div>"
                    f"<span style='color:{bull_color};font-weight:700;font-size:18px;margin-right:8px'>#{i}</span>"
                    f"<span style='color:{text_color};font-weight:600;font-size:15px'>{m['name']}</span>"
                    f"<span style='background:rgba(0,200,83,0.15);color:{bull_color};padding:2px 8px;border-radius:10px;font-size:10px;margin-left:8px'>{m.get('type', '')}</span>"
                    f"{sector_tag}"
                    f"</div>"
                    f"<div style='text-align:right'>"
                    f"<span style='color:{text_color};font-size:14px;margin-right:12px'>{p_str}</span>"
                    f"<span style='color:{bull_color};font-weight:700;font-size:16px'>{m['day_change']:+.1f}%</span>"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        with mc2:
            st.markdown(
                f"<h3 style='color:{bear_color};margin-bottom:10px'>Top 5 Sinking Ships Today</h3>",
                unsafe_allow_html=True,
            )
            for i, m in enumerate(sinking_ships, 1):
                price = m.get("price", 0)
                p_str = f"${price:,.2f}" if price < 1000 else f"${price:,.0f}"
                if m.get("type") == "Crypto" and price < 1:
                    p_str = f"${price:.4f}"
                sector_tag = f"<span style='color:#888;font-size:11px;margin-left:6px'>{m.get('sector', '')}</span>" if m.get("sector") else ""
                st.markdown(
                    f"<div style='background:{surface};padding:12px 16px;border-radius:8px;margin-bottom:6px;"
                    f"border-left:4px solid {bear_color};display:flex;align-items:center;justify-content:space-between'>"
                    f"<div>"
                    f"<span style='color:{bear_color};font-weight:700;font-size:18px;margin-right:8px'>#{i}</span>"
                    f"<span style='color:{text_color};font-weight:600;font-size:15px'>{m['name']}</span>"
                    f"<span style='background:rgba(255,23,68,0.15);color:{bear_color};padding:2px 8px;border-radius:10px;font-size:10px;margin-left:8px'>{m.get('type', '')}</span>"
                    f"{sector_tag}"
                    f"</div>"
                    f"<div style='text-align:right'>"
                    f"<span style='color:{text_color};font-size:14px;margin-right:12px'>{p_str}</span>"
                    f"<span style='color:{bear_color};font-weight:700;font-size:16px'>{m['day_change']:+.1f}%</span>"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("")

    # ═══════════════════════════════════════════════════════════
    # KEY TAKEAWAYS
    # ═══════════════════════════════════════════════════════════
    st.subheader("Key Takeaways")
    for i, takeaway in enumerate(briefing["takeaways"], 1):
        st.markdown(
            f"<div style='background:{surface};padding:15px 20px;border-radius:8px;margin-bottom:10px;"
            f"border-left:3px solid {primary}'>"
            f"<span style='color:{primary};font-weight:700;font-size:18px;margin-right:10px'>{i}.</span>"
            f"<span style='color:{text_color};font-size:15px;line-height:1.6'>{takeaway}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ═══════════════════════════════════════════════════════════
    # MARKET BENCHMARK SCORES
    # ═══════════════════════════════════════════════════════════
    if briefing["market_scores"]:
        st.markdown("---")
        st.subheader("Market Conviction Scores")
        score_cols = st.columns(len(briefing["market_scores"]))
        for col, (name, data) in zip(score_cols, briefing["market_scores"].items()):
            score = data["score"]
            label = data["label"]
            day_chg = data.get("day_change", 0)
            ret_1m = data.get("ret_1m", 0)

            if score >= 65:
                s_color = bull_color
            elif score >= 55:
                s_color = "#8BC34A"
            elif score >= 45:
                s_color = neutral_color
            elif score >= 35:
                s_color = "#FF9800"
            else:
                s_color = bear_color

            chg_color = bull_color if day_chg >= 0 else bear_color

            col.markdown(
                f"<div style='text-align:center;background:{surface};padding:20px;border-radius:12px;"
                f"border-top:4px solid {s_color}'>"
                f"<p style='margin:0;color:#aaa;font-size:11px;text-transform:uppercase;letter-spacing:1px'>{name}</p>"
                f"<p style='margin:8px 0 4px;font-size:42px;font-weight:700;color:{s_color}'>{score:.0f}</p>"
                f"<p style='margin:0;font-size:12px;color:{s_color};font-weight:600'>{label}</p>"
                f"<p style='margin:4px 0 0;font-size:11px;color:{chg_color}'>Today: {day_chg:+.1f}% | 1M: {ret_1m:+.1f}%</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ═══════════════════════════════════════════════════════════
    # SECTOR RANKINGS (THE BIG NEW SECTION)
    # ═══════════════════════════════════════════════════════════
    if briefing.get("sector_scores"):
        st.markdown("---")
        st.subheader("All 11 Sectors — Ranked Best to Worst")
        st.caption("Each sector scored 0-100 on trend, momentum, volume, and relative strength")

        rankings = briefing.get("sector_rankings", [])
        for r in rankings:
            score = r["score"]
            rank = r["rank"]
            sector = r["sector"]
            etf = r["etf"]
            label = r["label"]
            ret_1m = r["ret_1m"]

            if score >= 65:
                s_color = bull_color
            elif score >= 55:
                s_color = "#8BC34A"
            elif score >= 45:
                s_color = neutral_color
            elif score >= 35:
                s_color = "#FF9800"
            else:
                s_color = bear_color

            # Get interpretation
            sector_detail = briefing["sector_scores"].get(sector, {})
            interp = sector_detail.get("interpretation", "")
            above_50 = sector_detail.get("above_50_sma", None)
            above_200 = sector_detail.get("above_200_sma", None)
            day_chg = sector_detail.get("day_change", 0)

            ma_badges = ""
            if above_200 is True:
                ma_badges += f"<span style='background:rgba(0,200,83,0.15);color:{bull_color};padding:2px 8px;border-radius:10px;font-size:10px;margin-left:5px'>Above 200d</span>"
            elif above_200 is False:
                ma_badges += f"<span style='background:rgba(255,23,68,0.15);color:{bear_color};padding:2px 8px;border-radius:10px;font-size:10px;margin-left:5px'>Below 200d</span>"

            if above_50 is True:
                ma_badges += f"<span style='background:rgba(0,200,83,0.1);color:{bull_color};padding:2px 8px;border-radius:10px;font-size:10px;margin-left:5px'>Above 50d</span>"
            elif above_50 is False:
                ma_badges += f"<span style='background:rgba(255,23,68,0.1);color:{bear_color};padding:2px 8px;border-radius:10px;font-size:10px;margin-left:5px'>Below 50d</span>"

            chg_color = bull_color if day_chg >= 0 else bear_color

            # Score bar width
            bar_width = max(5, min(100, score))

            st.markdown(
                f"<div style='background:{surface};padding:15px 20px;border-radius:10px;margin-bottom:8px;"
                f"border-left:4px solid {s_color}'>"
                f"<div style='display:flex;align-items:center;gap:15px;flex-wrap:wrap'>"
                f"<span style='font-size:20px;font-weight:700;color:{s_color};min-width:35px'>#{rank}</span>"
                f"<div style='flex:1;min-width:200px'>"
                f"<span style='font-size:16px;font-weight:600;color:{text_color}'>{sector}</span>"
                f"<span style='color:#888;font-size:12px;margin-left:8px'>({etf})</span>"
                f"{ma_badges}"
                f"<div style='background:#333;border-radius:4px;height:8px;margin-top:6px;width:100%'>"
                f"<div style='background:{s_color};height:8px;border-radius:4px;width:{bar_width}%'></div></div>"
                f"</div>"
                f"<div style='text-align:right;min-width:80px'>"
                f"<span style='font-size:28px;font-weight:700;color:{s_color}'>{score:.0f}</span>"
                f"<span style='color:#888;font-size:12px'>/100</span>"
                f"</div>"
                f"<div style='text-align:right;min-width:80px'>"
                f"<p style='margin:0;font-size:12px;color:{chg_color}'>Today: {day_chg:+.1f}%</p>"
                f"<p style='margin:2px 0 0;font-size:12px;color:#888'>1M: {ret_1m:+.1f}%</p>"
                f"</div>"
                f"</div>"
                f"<p style='margin:8px 0 0;color:#aaa;font-size:13px;line-height:1.5'>{interp}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ═══════════════════════════════════════════════════════════
    # ASSET CLASS SCORES (Bonds, Gold, Oil, Dollar)
    # ═══════════════════════════════════════════════════════════
    if briefing.get("asset_class_scores"):
        st.markdown("---")
        st.subheader("Asset Class Scores")
        st.caption("Bonds, gold, oil, dollar — what the cross-asset picture is telling you")

        ac_cols = st.columns(len(briefing["asset_class_scores"]))
        for col, (name, data) in zip(ac_cols, briefing["asset_class_scores"].items()):
            score = data["score"]
            label = data["label"]
            day_chg = data.get("day_change", 0)
            ret_1m = data.get("ret_1m", 0)

            if score >= 60:
                s_color = bull_color
            elif score >= 45:
                s_color = neutral_color
            else:
                s_color = bear_color

            chg_color = bull_color if day_chg >= 0 else bear_color

            col.markdown(
                f"<div style='text-align:center;background:{surface};padding:18px;border-radius:12px;"
                f"border-top:4px solid {s_color}'>"
                f"<p style='margin:0;color:#aaa;font-size:10px;text-transform:uppercase;letter-spacing:1px'>{name}</p>"
                f"<p style='margin:6px 0 2px;font-size:32px;font-weight:700;color:{s_color}'>{score:.0f}</p>"
                f"<p style='margin:0;font-size:11px;color:{s_color}'>{label}</p>"
                f"<p style='margin:4px 0 0;font-size:10px;color:{chg_color}'>{day_chg:+.1f}% today | {ret_1m:+.1f}% 1M</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Asset class interpretations
        with st.expander("What each asset class score means"):
            for name, data in briefing["asset_class_scores"].items():
                interp = data.get("interpretation", "")
                if interp:
                    st.write(f"**{name}:** {interp}")

    # ═══════════════════════════════════════════════════════════
    # DAILY CRYPTO OUTLOOK
    # ═══════════════════════════════════════════════════════════
    crypto_data = briefing.get("crypto", {})
    crypto_outlook = briefing.get("crypto_outlook", {})

    st.markdown("---")
    st.subheader("Daily Crypto Outlook")
    st.caption("Real news from CoinDesk, CoinTelegraph, Decrypt, The Block & more — analyzed and explained")

    # Price cards
    if crypto_data:
        crypto_cols = st.columns(len(crypto_data))
        for col, (symbol, cdata) in zip(crypto_cols, crypto_data.items()):
            price = cdata["price"]
            change_24h = cdata["change_24h"]
            signal = cdata.get("signal", "QUIET")
            c_color = bull_color if change_24h > 0 else bear_color

            if signal in ("SURGING", "BULLISH"):
                sig_color = bull_color
            elif signal in ("WEAK", "SELLING"):
                sig_color = bear_color
            else:
                sig_color = neutral_color

            if price >= 1000:
                price_str = f"${price:,.0f}"
            elif price >= 1:
                price_str = f"${price:,.2f}"
            else:
                price_str = f"${price:.4f}"

            col.markdown(
                f"<div style='text-align:center;background:{surface};padding:20px;border-radius:12px;"
                f"border-top:4px solid {c_color}'>"
                f"<p style='margin:0;color:#aaa;font-size:11px;text-transform:uppercase;letter-spacing:1px'>{symbol}</p>"
                f"<p style='margin:8px 0 4px;font-size:28px;font-weight:700;color:{text_color}'>{price_str}</p>"
                f"<p style='margin:0;font-size:13px;color:{c_color};font-weight:600'>{change_24h:+.1f}%</p>"
                f"<p style='margin:4px 0 0;font-size:11px'>"
                f"<span style='background:{sig_color};color:white;padding:2px 8px;border-radius:8px;font-size:10px'>{signal}</span>"
                f"</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # News sentiment summary
    if crypto_outlook.get("summary"):
        sent_label = crypto_outlook.get("sentiment_label", "Neutral")
        sent_score = crypto_outlook.get("sentiment_score", 0)
        news_count = crypto_outlook.get("news_count", 0)

        if "Positive" in sent_label:
            sent_color = bull_color
        elif "Negative" in sent_label:
            sent_color = bear_color
        else:
            sent_color = neutral_color

        st.markdown(
            f"<div style='background:linear-gradient(135deg, {surface}, #1a1a2e);padding:20px;border-radius:12px;"
            f"margin-top:15px;border-left:5px solid {sent_color}'>"
            f"<div style='display:flex;align-items:center;gap:15px;margin-bottom:12px;flex-wrap:wrap'>"
            f"<span style='font-size:18px;font-weight:700;color:{text_color}'>News Sentiment</span>"
            f"<span style='background:{sent_color};color:white;padding:4px 14px;border-radius:15px;font-size:12px;font-weight:600'>{sent_label}</span>"
            f"<span style='color:#888;font-size:12px'>{news_count} articles analyzed</span>"
            f"</div>"
            f"<p style='margin:0;color:{text_color};font-size:14px;line-height:1.6'>{crypto_outlook['summary']}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Practical take (the money paragraph)
    if crypto_outlook.get("practical_take"):
        st.markdown(
            f"<div style='background:{surface};padding:20px;border-radius:12px;margin-top:12px;"
            f"border-left:5px solid {primary}'>"
            f"<p style='margin:0 0 8px;font-weight:700;color:{primary};font-size:16px'>What This Means For You</p>"
            f"<p style='margin:0;color:{text_color};font-size:14px;line-height:1.7'>{crypto_outlook['practical_take']}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Per-coin outlook
    coin_outlooks = [
        ("BTC", crypto_outlook.get("btc_outlook", "")),
        ("ETH", crypto_outlook.get("eth_outlook", "")),
        ("SOL", crypto_outlook.get("sol_outlook", "")),
    ]
    has_outlooks = any(o for _, o in coin_outlooks)
    if has_outlooks:
        for symbol, coin_outlook in coin_outlooks:
            if not coin_outlook:
                continue
            cdata = crypto_data.get(symbol, {})
            change_24h = cdata.get("change_24h", 0)
            sig = cdata.get("signal", "QUIET")

            if sig in ("SURGING", "BULLISH"):
                sig_color = bull_color
            elif sig in ("WEAK", "SELLING"):
                sig_color = bear_color
            else:
                sig_color = "#FF9800"

            st.markdown(
                f"<div style='background:{surface};padding:14px 18px;border-radius:8px;margin-top:8px;"
                f"border-left:3px solid {sig_color}'>"
                f"<span style='color:{sig_color};font-weight:700;font-size:15px'>{symbol}:</span> "
                f"<span style='color:{text_color};font-size:13px;line-height:1.6'>{coin_outlook}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # News themes
    themes = crypto_outlook.get("themes", [])
    if themes:
        st.markdown(f"<p style='margin:20px 0 10px;font-weight:700;color:{text_color};font-size:15px'>What The News Is Talking About</p>", unsafe_allow_html=True)
        theme_cols = st.columns(min(len(themes), 3))
        for i, theme in enumerate(themes[:6]):
            col = theme_cols[i % 3]
            mood = theme.get("mood", "neutral")
            if mood == "positive":
                t_color = bull_color
                t_icon = "+"
            elif mood == "negative":
                t_color = bear_color
                t_icon = "!"
            else:
                t_color = neutral_color
                t_icon = "~"

            col.markdown(
                f"<div style='text-align:center;background:{surface};padding:14px;border-radius:10px;"
                f"border-top:3px solid {t_color};margin-bottom:8px'>"
                f"<p style='margin:0;color:{text_color};font-weight:600;font-size:13px'>{theme['theme']}</p>"
                f"<p style='margin:4px 0;font-size:20px;font-weight:700;color:{t_color}'>{t_icon}</p>"
                f"<p style='margin:0;color:#888;font-size:11px'>{theme['article_count']} articles | Mood: {mood}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Top headlines
    headlines = crypto_outlook.get("top_headlines", [])
    if headlines:
        with st.expander(f"Top {len(headlines)} Crypto Headlines Right Now", expanded=False):
            for h in headlines:
                sent = h.get("sentiment", "Neutral")
                if sent == "Positive":
                    h_color = bull_color
                    h_icon = "+"
                elif sent == "Negative":
                    h_color = bear_color
                    h_icon = "!"
                else:
                    h_color = "#888"
                    h_icon = "~"

                urgency = h.get("urgency", "low")
                urg_badge = ""
                if urgency == "high":
                    urg_badge = f"<span style='background:{bear_color};color:white;padding:1px 6px;border-radius:6px;font-size:9px;margin-left:6px'>HIGH IMPACT</span>"

                st.markdown(
                    f"<div style='padding:8px 12px;margin:4px 0;border-left:3px solid {h_color};border-radius:4px'>"
                    f"<span style='color:{h_color};font-weight:700;margin-right:6px'>{h_icon}</span>"
                    f"<span style='color:{text_color};font-size:13px'>{h['title']}</span>"
                    f"<span style='color:#666;font-size:11px;margin-left:8px'>— {h['source']}</span>"
                    f"{urg_badge}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # What to watch
    watch_items = crypto_outlook.get("what_to_watch", [])
    if watch_items:
        st.markdown(f"<p style='margin:20px 0 10px;font-weight:700;color:{text_color};font-size:15px'>Crypto — What To Watch</p>", unsafe_allow_html=True)
        for w in watch_items:
            st.markdown(
                f"<div style='background:{surface};padding:10px 16px;border-radius:8px;margin-bottom:6px;"
                f"border-left:3px solid {neutral_color}'>"
                f"<span style='color:{text_color};font-size:13px;line-height:1.5'>{w}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ═══════════════════════════════════════════════════════════
    # WHAT THE SIGNALS MEAN
    # ═══════════════════════════════════════════════════════════
    if briefing["signals_explained"]:
        st.markdown("---")
        st.subheader("What The Signals Mean")
        st.caption("Each signal explained in plain English — no jargon, no BS")

        for sig in briefing["signals_explained"]:
            direction = sig.get("direction", "NEUTRAL")
            if direction == "BULLISH":
                sig_color = bull_color
                icon = "+"
            elif direction == "BEARISH":
                sig_color = bear_color
                icon = "!"
            elif direction == "MIXED":
                sig_color = neutral_color
                icon = "~"
            else:
                sig_color = "#888"
                icon = "-"

            title = sig.get("title", "")
            what_it_is = sig.get("what_it_is", "")
            what_it_means = sig.get("what_it_means", "")
            so_what = sig.get("so_what", "")

            html_parts = [
                f"<div style='background:{surface};padding:20px;border-radius:12px;margin-bottom:15px;"
                f"border-left:4px solid {sig_color}'>",
                f"<div style='display:flex;align-items:center;margin-bottom:10px'>",
                f"<span style='background:{sig_color};color:white;width:28px;height:28px;border-radius:50%;"
                f"display:flex;align-items:center;justify-content:center;font-weight:700;font-size:16px;margin-right:12px'>{icon}</span>",
                f"<span style='font-size:17px;font-weight:600;color:{text_color}'>{title}</span>",
                f"<span style='margin-left:auto;background:{sig_color};color:white;padding:3px 10px;border-radius:10px;font-size:11px'>{direction}</span>",
                f"</div>",
            ]

            if what_it_is:
                html_parts.append(
                    f"<p style='color:#aaa;font-size:13px;margin:5px 0;font-style:italic'>What it is: {what_it_is}</p>"
                )
            if what_it_means:
                html_parts.append(
                    f"<p style='color:{text_color};font-size:14px;margin:8px 0;line-height:1.5'>What it means: {what_it_means}</p>"
                )
            if so_what:
                html_parts.append(
                    f"<p style='color:{sig_color};font-size:14px;margin:8px 0;font-weight:600;line-height:1.5'>So what? {so_what}</p>"
                )

            html_parts.append("</div>")
            st.markdown("".join(html_parts), unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # RISK ASSESSMENT
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Risk Assessment")

    risk_score = briefing["risk_score"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title=dict(text="Market Risk Level", font=dict(size=16, color=text_color)),
        number=dict(font=dict(size=48, color=risk_color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=text_color),
            bar=dict(color=risk_color, thickness=0.3),
            bgcolor=surface,
            bordercolor="#444",
            steps=[
                dict(range=[0, 25], color="rgba(0,200,83,0.15)"),
                dict(range=[25, 50], color="rgba(255,214,0,0.08)"),
                dict(range=[50, 75], color="rgba(255,152,0,0.12)"),
                dict(range=[75, 100], color="rgba(255,23,68,0.15)"),
            ],
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=250,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    if briefing["risk_factors"]:
        st.markdown("**Active Risk Factors:**")
        for factor in briefing["risk_factors"]:
            st.markdown(
                f"<div style='padding:8px 15px;margin:5px 0;background:rgba(255,23,68,0.08);"
                f"border-left:3px solid {bear_color};border-radius:4px'>"
                f"<span style='color:{bear_color};font-weight:600'>WARNING:</span> "
                f"<span style='color:{text_color}'>{factor}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.success("No major risk factors detected at this time")

    # ═══════════════════════════════════════════════════════════
    # ACTION ITEMS
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("What To Do Right Now")

    for action in briefing["action_items"]:
        st.markdown(
            f"<div style='padding:12px 18px;margin:8px 0;background:{surface};"
            f"border-radius:8px;border-left:3px solid {primary}'>"
            f"<span style='color:{primary};font-weight:700;margin-right:8px'>ACTION:</span>"
            f"<span style='color:{text_color};font-size:14px'>{action}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ═══════════════════════════════════════════════════════════
    # WATCH LIST
    # ═══════════════════════════════════════════════════════════
    if briefing["watch_list"]:
        st.markdown("---")
        st.subheader("What To Watch This Week")

        for item in briefing["watch_list"]:
            st.markdown(
                f"<div style='background:{surface};padding:15px;border-radius:8px;margin-bottom:10px'>"
                f"<p style='margin:0;font-weight:600;color:{primary};font-size:15px'>{item['item']}</p>"
                f"<p style='margin:5px 0 0;color:#aaa;font-size:13px;line-height:1.5'>{item['why']}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ═══════════════════════════════════════════════════════════
    # BOTTOM LINE
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    bottom_line = briefing["bottom_line"]
    st.markdown(
        f"<div style='background:linear-gradient(135deg, {surface}, #1a1a2e);padding:25px;border-radius:16px;"
        f"border:1px solid {regime_color};margin-top:10px'>"
        f"<h3 style='margin:0 0 10px;color:{regime_color}'>THE BOTTOM LINE</h3>"
        f"<p style='margin:0;color:{text_color};font-size:16px;line-height:1.7'>{bottom_line}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<p style='text-align:center;color:#555;font-size:11px;margin-top:20px'>"
        f"Current Regime: {regime} | {briefing['regime_detail']}"
        f"</p>",
        unsafe_allow_html=True,
    )

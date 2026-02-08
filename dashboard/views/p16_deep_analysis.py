"""
SENTINEL — Page 16: Deep Stock Analysis.
Enter ANY ticker. Get EVERYTHING: financials, technicals, news, analyst ratings,
insider trades, institutional holdings. AI synthesizes it all into price targets
for 1 day, 1 week, 1 month, 3 months, 6 months, and 1 year.
"""

import streamlit as st
import plotly.graph_objects as go

from config.settings import COLORS


def _fmt_big_number(n):
    """Format large numbers: 1.5T, 250B, 4.2M, etc."""
    if not n or n == 0:
        return "N/A"
    abs_n = abs(n)
    if abs_n >= 1e12:
        return f"${n/1e12:.2f}T"
    if abs_n >= 1e9:
        return f"${n/1e9:.1f}B"
    if abs_n >= 1e6:
        return f"${n/1e6:.1f}M"
    if abs_n >= 1e3:
        return f"${n/1e3:.1f}K"
    return f"${n:,.0f}"


def _fmt_pct(n, decimals=1):
    """Format as percentage."""
    if n is None or n == 0:
        return "N/A"
    return f"{n*100:.{decimals}f}%" if abs(n) < 5 else f"{n:.{decimals}f}%"


def render():
    st.title("Deep Stock Analysis")
    st.caption("Enter any ticker — get EVERYTHING: financials, technicals, news, analyst targets, insider moves, AI price targets")

    surface = COLORS["surface"]
    text_color = COLORS["text"]
    bull = COLORS["bull"]
    bear = COLORS["bear"]
    neutral = COLORS["neutral"]
    primary = COLORS["primary"]
    bg = COLORS["background"]

    # ── Ticker input ─────────────────────────────────────────────
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        ticker = st.text_input("Ticker Symbol", value="AAPL", max_chars=10,
                               placeholder="AAPL, TSLA, NVDA, MSFT...").upper().strip()
    with col_btn:
        st.write("")  # spacer
        analyze = st.button("Analyze", type="primary", use_container_width=True)

    if not ticker:
        return

    if not analyze and "last_deep_ticker" not in st.session_state:
        st.info("Enter a ticker and click **Analyze** to get the full research report.")
        return

    if analyze:
        st.session_state["last_deep_ticker"] = ticker

    active_ticker = st.session_state.get("last_deep_ticker", ticker)

    with st.spinner(f"Running deep analysis on {active_ticker}... pulling financials, news, technicals, analyst data..."):
        from features.deep_analysis import deep_analyze
        data = deep_analyze(active_ticker)

    if data.get("error"):
        st.error(data["error"])
        return

    company = data["company"]
    price_data = data["price"]
    tech = data.get("technicals", {})
    fund = data.get("fundamentals", {})
    earnings = data.get("earnings", {})
    analyst = data.get("analyst_ratings", {})
    insider = data.get("insider_activity", {})
    institutional = data.get("institutional", {})
    news = data.get("news", [])
    upgrades = data.get("upgrades", [])
    targets = data.get("price_targets", {})
    verdict = data.get("verdict", {})

    # ═══════════════════════════════════════════════════════════════
    # VERDICT BANNER
    # ═══════════════════════════════════════════════════════════════
    v_color = verdict.get("color", neutral)
    v_text = verdict.get("verdict", "NEUTRAL")
    v_summary = verdict.get("summary", "")
    bull_pts = verdict.get("bull_points", 0)
    bear_pts = verdict.get("bear_points", 0)

    current_price = price_data.get("current", 0)
    day_chg = price_data.get("day_change_pct", 0)
    day_color = bull if day_chg >= 0 else bear

    st.markdown(
        f"<div style='background:linear-gradient(135deg,{surface},#1a1a2e);padding:30px;border-radius:16px;"
        f"border-left:6px solid {v_color};margin-bottom:25px'>"
        f"<div style='display:flex;align-items:center;gap:20px;flex-wrap:wrap'>"
        f"<div>"
        f"<h2 style='margin:0;color:{text_color}'>{company.get('name', active_ticker)}</h2>"
        f"<p style='margin:5px 0;color:#888;font-size:13px'>{company.get('sector','')}"
        f"{' | ' + company.get('industry','') if company.get('industry') else ''}</p>"
        f"</div>"
        f"<div style='margin-left:auto;text-align:right'>"
        f"<p style='margin:0;font-size:36px;font-weight:700;color:{text_color}'>${current_price:,.2f}</p>"
        f"<p style='margin:0;font-size:15px;color:{day_color};font-weight:600'>{day_chg:+.2f}%</p>"
        f"</div>"
        f"</div>"
        f"<div style='margin-top:15px;display:flex;gap:12px;flex-wrap:wrap'>"
        f"<span style='background:{v_color};color:white;padding:6px 16px;border-radius:20px;font-size:14px;font-weight:700'>{v_text}</span>"
        f"<span style='background:rgba(0,200,83,0.15);color:{bull};padding:6px 12px;border-radius:20px;font-size:12px'>{bull_pts} Bull Points</span>"
        f"<span style='background:rgba(255,23,68,0.15);color:{bear};padding:6px 12px;border-radius:20px;font-size:12px'>{bear_pts} Bear Points</span>"
        f"<span style='background:rgba(255,255,255,0.05);color:#888;padding:6px 12px;border-radius:20px;font-size:12px'>Mkt Cap: {_fmt_big_number(company.get('market_cap', 0))}</span>"
        f"</div>"
        f"<p style='margin:12px 0 0;color:{text_color};font-size:14px;line-height:1.6'>{v_summary}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ═══════════════════════════════════════════════════════════════
    # PRICE TARGETS (The star of the show)
    # ═══════════════════════════════════════════════════════════════
    if targets:
        st.subheader("AI Price Targets")
        st.caption("Multi-factor model combining technicals, fundamentals, analyst consensus, and volatility")

        target_cols = st.columns(len(targets))
        for col, (key, t) in zip(target_cols, targets.items()):
            t_dir = t.get("direction", "NEUTRAL")
            if t_dir == "BULLISH":
                t_color = bull
            elif t_dir == "BEARISH":
                t_color = bear
            else:
                t_color = neutral

            move_pct = t.get("expected_move_pct", 0)
            prob = t.get("probability_up", 50)
            t_price = t.get("target_price", 0)
            conf = t.get("confidence", 0)

            col.markdown(
                f"<div style='text-align:center;background:{surface};padding:18px 10px;border-radius:12px;"
                f"border-top:4px solid {t_color}'>"
                f"<p style='margin:0;color:#aaa;font-size:11px;text-transform:uppercase;letter-spacing:1px'>{t.get('label','')}</p>"
                f"<p style='margin:8px 0 2px;font-size:28px;font-weight:700;color:{text_color}'>${t_price:,.2f}</p>"
                f"<p style='margin:0;font-size:14px;color:{t_color};font-weight:600'>{move_pct:+.1f}%</p>"
                f"<p style='margin:6px 0 2px;font-size:11px;color:#888'>Prob Up: {prob:.0f}%</p>"
                f"<div style='background:#333;border-radius:4px;height:6px;margin:5px 10px'>"
                f"<div style='background:{t_color};height:6px;border-radius:4px;width:{prob:.0f}%'></div>"
                f"</div>"
                f"<p style='margin:4px 0 0;font-size:10px;color:#666'>Range: ${t.get('low_target',0):,.0f} - ${t.get('high_target',0):,.0f}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Analyst targets comparison (if available)
        yr_target = targets.get("1_year", {})
        if yr_target.get("analyst_target"):
            st.markdown(
                f"<div style='background:{surface};padding:12px 18px;border-radius:8px;margin-top:10px;"
                f"border-left:3px solid {primary}'>"
                f"<span style='color:{primary};font-weight:600'>WALL STREET COMPARISON:</span> "
                f"<span style='color:{text_color}'>Analyst consensus 12-month target: "
                f"<b>${yr_target['analyst_target']:,.2f}</b> "
                f"(Range: ${yr_target.get('analyst_low',0):,.0f} - ${yr_target.get('analyst_high',0):,.0f}) "
                f"vs SENTINEL AI target: <b>${yr_target['target_price']:,.2f}</b></span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ═══════════════════════════════════════════════════════════════
    # BULL vs BEAR
    # ═══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Bull Case vs Bear Case")

    bull_col, bear_col = st.columns(2)

    with bull_col:
        st.markdown(f"<p style='color:{bull};font-weight:700;font-size:16px;margin-bottom:8px'>BULL CASE ({bull_pts} points)</p>", unsafe_allow_html=True)
        for point in data.get("bull_case", []):
            st.markdown(
                f"<div style='padding:8px 12px;margin:4px 0;background:rgba(0,200,83,0.06);"
                f"border-left:3px solid {bull};border-radius:4px'>"
                f"<span style='color:{text_color};font-size:13px'>{point}</span></div>",
                unsafe_allow_html=True,
            )

    with bear_col:
        st.markdown(f"<p style='color:{bear};font-weight:700;font-size:16px;margin-bottom:8px'>BEAR CASE ({bear_pts} points)</p>", unsafe_allow_html=True)
        for point in data.get("bear_case", []):
            st.markdown(
                f"<div style='padding:8px 12px;margin:4px 0;background:rgba(255,23,68,0.06);"
                f"border-left:3px solid {bear};border-radius:4px'>"
                f"<span style='color:{text_color};font-size:13px'>{point}</span></div>",
                unsafe_allow_html=True,
            )

    # Risk factors
    if data.get("risk_factors"):
        st.markdown(f"<p style='color:#FF9800;font-weight:600;margin-top:15px'>RISK FACTORS:</p>", unsafe_allow_html=True)
        for risk in data["risk_factors"]:
            st.markdown(
                f"<div style='padding:6px 12px;margin:3px 0;background:rgba(255,152,0,0.06);"
                f"border-left:3px solid #FF9800;border-radius:4px'>"
                f"<span style='color:{text_color};font-size:13px'>{risk}</span></div>",
                unsafe_allow_html=True,
            )

    # ═══════════════════════════════════════════════════════════════
    # TECHNICAL ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    if tech:
        st.markdown("---")
        st.subheader("Technical Analysis")

        tc = st.columns(4)

        # RSI
        rsi = tech.get("rsi", 50)
        rsi_sig = tech.get("rsi_signal", "NEUTRAL")
        if rsi_sig in ("OVERBOUGHT", "BEARISH"):
            rsi_color = bear
        elif rsi_sig in ("OVERSOLD", "BULLISH"):
            rsi_color = bull
        else:
            rsi_color = neutral

        tc[0].markdown(
            f"<div style='text-align:center;background:{surface};padding:18px;border-radius:12px;"
            f"border-top:3px solid {rsi_color}'>"
            f"<p style='margin:0;color:#888;font-size:11px'>RSI (14)</p>"
            f"<p style='margin:8px 0 4px;font-size:36px;font-weight:700;color:{rsi_color}'>{rsi:.0f}</p>"
            f"<p style='margin:0;font-size:12px;color:{rsi_color}'>{rsi_sig}</p></div>",
            unsafe_allow_html=True,
        )

        # MACD
        macd_bull = tech.get("macd_bullish", False)
        macd_color = bull if macd_bull else bear
        tc[1].markdown(
            f"<div style='text-align:center;background:{surface};padding:18px;border-radius:12px;"
            f"border-top:3px solid {macd_color}'>"
            f"<p style='margin:0;color:#888;font-size:11px'>MACD</p>"
            f"<p style='margin:8px 0 4px;font-size:36px;font-weight:700;color:{macd_color}'>"
            f"{'BULL' if macd_bull else 'BEAR'}</p>"
            f"<p style='margin:0;font-size:12px;color:{macd_color}'>{'Bullish Crossover' if macd_bull else 'Bearish Crossover'}</p></div>",
            unsafe_allow_html=True,
        )

        # Trend Strength
        adx = tech.get("adx", 0)
        trend = tech.get("trend_strength", "WEAK")
        if trend in ("STRONG", "VERY STRONG"):
            trend_color = primary
        elif trend == "MODERATE":
            trend_color = neutral
        else:
            trend_color = "#888"

        tc[2].markdown(
            f"<div style='text-align:center;background:{surface};padding:18px;border-radius:12px;"
            f"border-top:3px solid {trend_color}'>"
            f"<p style='margin:0;color:#888;font-size:11px'>TREND (ADX)</p>"
            f"<p style='margin:8px 0 4px;font-size:36px;font-weight:700;color:{trend_color}'>{adx:.0f}</p>"
            f"<p style='margin:0;font-size:12px;color:{trend_color}'>{trend}</p></div>",
            unsafe_allow_html=True,
        )

        # Bollinger Position
        bb_pos = tech.get("bb_position", 50)
        if bb_pos > 80:
            bb_color = bear
            bb_label = "UPPER BAND"
        elif bb_pos < 20:
            bb_color = bull
            bb_label = "LOWER BAND"
        else:
            bb_color = neutral
            bb_label = "MID RANGE"

        tc[3].markdown(
            f"<div style='text-align:center;background:{surface};padding:18px;border-radius:12px;"
            f"border-top:3px solid {bb_color}'>"
            f"<p style='margin:0;color:#888;font-size:11px'>BOLLINGER POS</p>"
            f"<p style='margin:8px 0 4px;font-size:36px;font-weight:700;color:{bb_color}'>{bb_pos:.0f}%</p>"
            f"<p style='margin:0;font-size:12px;color:{bb_color}'>{bb_label}</p></div>",
            unsafe_allow_html=True,
        )

        # Moving average signals
        if tech.get("ma_signals"):
            for sig in tech["ma_signals"]:
                sig_color = bull if "bullish" in sig.lower() else bear
                st.markdown(
                    f"<div style='padding:6px 12px;margin:4px 0;background:{surface};border-left:3px solid {sig_color};border-radius:4px'>"
                    f"<span style='color:{text_color};font-size:13px'>{sig}</span></div>",
                    unsafe_allow_html=True,
                )

        # Descriptions
        with st.expander("What do these indicators mean?"):
            st.write(tech.get("rsi_description", ""))
            st.write(tech.get("macd_description", ""))
            st.write(tech.get("trend_description", ""))

    # ═══════════════════════════════════════════════════════════════
    # FUNDAMENTALS
    # ═══════════════════════════════════════════════════════════════
    if fund:
        st.markdown("---")
        st.subheader("Fundamentals")

        fc = st.columns(4)
        pe_t = fund.get("pe_trailing", 0)
        pe_f = fund.get("pe_forward", 0)
        fc[0].metric("P/E (Trailing)", f"{pe_t:.1f}" if pe_t else "N/A")
        fc[1].metric("P/E (Forward)", f"{pe_f:.1f}" if pe_f else "N/A")
        fc[2].metric("PEG Ratio", f"{fund.get('peg_ratio', 0):.2f}" if fund.get("peg_ratio") else "N/A")
        fc[3].metric("EV/EBITDA", f"{fund.get('ev_to_ebitda', 0):.1f}" if fund.get("ev_to_ebitda") else "N/A")

        fc2 = st.columns(4)
        fc2[0].metric("Revenue", _fmt_big_number(fund.get("revenue", 0)))
        rev_g = fund.get("revenue_growth", 0)
        fc2[1].metric("Revenue Growth", f"{rev_g*100:.1f}%" if rev_g else "N/A",
                       delta=f"{rev_g*100:.1f}%" if rev_g else None)
        fc2[2].metric("Profit Margin", f"{fund.get('profit_margin', 0)*100:.1f}%" if fund.get("profit_margin") else "N/A")
        fc2[3].metric("Free Cash Flow", _fmt_big_number(fund.get("free_cash_flow", 0)))

        fc3 = st.columns(4)
        fc3[0].metric("ROE", f"{fund.get('roe', 0)*100:.1f}%" if fund.get("roe") else "N/A")
        fc3[1].metric("Debt/Equity", f"{fund.get('debt_to_equity', 0):.0f}%" if fund.get("debt_to_equity") else "N/A")
        div_y = fund.get("dividend_yield", 0)
        fc3[2].metric("Dividend Yield", f"{div_y*100:.2f}%" if div_y else "None")
        fc3[3].metric("Current Ratio", f"{fund.get('current_ratio', 0):.2f}" if fund.get("current_ratio") else "N/A")

        if fund.get("valuation_signals"):
            for sig in fund["valuation_signals"]:
                st.markdown(
                    f"<div style='padding:6px 12px;margin:4px 0;background:{surface};border-left:3px solid {primary};border-radius:4px'>"
                    f"<span style='color:{text_color};font-size:13px'>{sig}</span></div>",
                    unsafe_allow_html=True,
                )

    # ═══════════════════════════════════════════════════════════════
    # EARNINGS
    # ═══════════════════════════════════════════════════════════════
    if earnings:
        st.markdown("---")
        st.subheader("Earnings Track Record")

        ec = st.columns(3)
        beat_rate = earnings.get("beat_rate", 0)
        br_color = bull if beat_rate >= 70 else (neutral if beat_rate >= 50 else bear)
        ec[0].markdown(
            f"<div style='text-align:center;background:{surface};padding:18px;border-radius:12px;"
            f"border-top:3px solid {br_color}'>"
            f"<p style='margin:0;color:#888;font-size:11px'>BEAT RATE</p>"
            f"<p style='margin:8px 0;font-size:42px;font-weight:700;color:{br_color}'>{beat_rate:.0f}%</p>"
            f"<p style='margin:0;font-size:12px;color:#888'>{earnings.get('beats',0)} beats, {earnings.get('misses',0)} misses</p></div>",
            unsafe_allow_html=True,
        )

        eps_curr = earnings.get("eps_current_year", 0)
        eps_fwd = earnings.get("eps_forward", 0)
        ec[1].metric("EPS (Current Year)", f"${eps_curr:.2f}" if eps_curr else "N/A")
        ec[2].metric("EPS (Forward)", f"${eps_fwd:.2f}" if eps_fwd else "N/A")

        # Earnings history chart
        history = earnings.get("history", [])
        reported = [e for e in history if e.get("surprise_pct") is not None]
        if reported:
            dates = [e["date"][:10] for e in reported]
            surprises = [e["surprise_pct"] for e in reported]
            colors = [bull if s > 0 else bear for s in surprises]

            fig = go.Figure(go.Bar(
                x=dates, y=surprises,
                marker_color=colors,
                text=[f"{s:+.1f}%" for s in surprises],
                textposition="outside",
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="#666")
            fig.update_layout(
                title="Earnings Surprise History",
                template="plotly_dark", paper_bgcolor=bg, plot_bgcolor=surface,
                height=250, yaxis_title="Surprise %",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════
    # ANALYST RATINGS
    # ═══════════════════════════════════════════════════════════════
    if analyst:
        st.markdown("---")
        st.subheader("Analyst Ratings")

        ac = st.columns(3)

        # Consensus badge
        consensus = analyst.get("consensus", "N/A").upper()
        if "BUY" in consensus or "OVER" in consensus:
            cons_color = bull
        elif "SELL" in consensus or "UNDER" in consensus:
            cons_color = bear
        else:
            cons_color = neutral

        ac[0].markdown(
            f"<div style='text-align:center;background:{surface};padding:18px;border-radius:12px;"
            f"border-top:3px solid {cons_color}'>"
            f"<p style='margin:0;color:#888;font-size:11px'>CONSENSUS</p>"
            f"<p style='margin:8px 0;font-size:28px;font-weight:700;color:{cons_color}'>{consensus}</p>"
            f"<p style='margin:0;font-size:12px;color:#888'>{analyst.get('total_analysts',0)} analysts</p></div>",
            unsafe_allow_html=True,
        )

        ac[1].markdown(
            f"<div style='text-align:center;background:{surface};padding:18px;border-radius:12px;"
            f"border-top:3px solid {primary}'>"
            f"<p style='margin:0;color:#888;font-size:11px'>TARGET PRICE</p>"
            f"<p style='margin:8px 0;font-size:28px;font-weight:700;color:{text_color}'>${analyst.get('target_mean',0):,.2f}</p>"
            f"<p style='margin:0;font-size:12px;color:#888'>${analyst.get('target_low',0):,.0f} - ${analyst.get('target_high',0):,.0f}</p></div>",
            unsafe_allow_html=True,
        )

        upside = analyst.get("target_upside", 0)
        up_color = bull if upside > 0 else bear
        ac[2].markdown(
            f"<div style='text-align:center;background:{surface};padding:18px;border-radius:12px;"
            f"border-top:3px solid {up_color}'>"
            f"<p style='margin:0;color:#888;font-size:11px'>IMPLIED UPSIDE</p>"
            f"<p style='margin:8px 0;font-size:28px;font-weight:700;color:{up_color}'>{upside:+.1f}%</p>"
            f"<p style='margin:0;font-size:12px;color:#888'>from current price</p></div>",
            unsafe_allow_html=True,
        )

        # Rating distribution bar
        sb = analyst.get("strong_buy", 0)
        b = analyst.get("buy", 0)
        h = analyst.get("hold", 0)
        s = analyst.get("sell", 0)
        ss = analyst.get("strong_sell", 0)
        total = sb + b + h + s + ss

        if total > 0:
            fig = go.Figure(go.Bar(
                x=[sb, b, h, s, ss],
                y=["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"],
                orientation="h",
                marker_color=["#00C853", "#8BC34A", "#FFD600", "#FF9800", "#FF1744"],
                text=[str(x) for x in [sb, b, h, s, ss]],
                textposition="outside",
            ))
            fig.update_layout(
                title="Analyst Rating Distribution",
                template="plotly_dark", paper_bgcolor=bg, plot_bgcolor=surface,
                height=220, margin=dict(l=100, r=50, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════
    # RECENT UPGRADES / DOWNGRADES
    # ═══════════════════════════════════════════════════════════════
    if upgrades:
        st.markdown("---")
        st.subheader("Recent Analyst Actions")

        for u in upgrades[:10]:
            action = u.get("action", "")
            if action == "up":
                a_color = bull
                a_icon = "UPGRADE"
            elif action == "down":
                a_color = bear
                a_icon = "DOWNGRADE"
            else:
                a_color = neutral
                a_icon = "MAINTAIN"

            pt_str = f" | Target: ${u['price_target']:,.0f}" if u.get("price_target") else ""

            st.markdown(
                f"<div style='padding:8px 15px;margin:4px 0;background:{surface};"
                f"border-left:3px solid {a_color};border-radius:4px;display:flex;align-items:center;gap:10px'>"
                f"<span style='background:{a_color};color:white;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700'>{a_icon}</span>"
                f"<span style='color:{text_color};font-size:13px'>"
                f"<b>{u.get('firm','')}</b>: {u.get('from_grade','')} → {u.get('to_grade','')}{pt_str}"
                f"</span>"
                f"<span style='margin-left:auto;color:#666;font-size:11px'>{u.get('date','')}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ═══════════════════════════════════════════════════════════════
    # INSIDER ACTIVITY
    # ═══════════════════════════════════════════════════════════════
    if insider and insider.get("transactions"):
        st.markdown("---")
        st.subheader("Insider Activity")

        signal = insider.get("net_signal", "NEUTRAL")
        sig_color = bull if signal == "BULLISH" else (bear if signal == "BEARISH" else neutral)
        st.markdown(
            f"<div style='background:{surface};padding:12px 18px;border-radius:8px;"
            f"border-left:3px solid {sig_color};margin-bottom:10px'>"
            f"<span style='color:{sig_color};font-weight:700'>INSIDER SIGNAL: {signal}</span>"
            f"<span style='color:#888;margin-left:15px'>({insider.get('recent_buys',0)} buys, {insider.get('recent_sells',0)} sells)</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        for txn in insider["transactions"][:8]:
            st.markdown(
                f"<div style='padding:6px 12px;margin:3px 0;background:{surface};border-radius:4px'>"
                f"<span style='color:{text_color};font-size:13px'><b>{txn.get('insider','')}</b>: "
                f"{txn.get('transaction','')} — {txn.get('shares',0):,} shares</span>"
                f"<span style='color:#666;font-size:11px;margin-left:10px'>{txn.get('date','')}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ═══════════════════════════════════════════════════════════════
    # INSTITUTIONAL HOLDERS
    # ═══════════════════════════════════════════════════════════════
    if institutional and institutional.get("holders"):
        st.markdown("---")
        st.subheader("Top Institutional Holders")

        inst_pct = institutional.get("pct_held", 0)
        ins_pct = institutional.get("insider_pct", 0)

        ic = st.columns(2)
        ic[0].metric("Institutional Ownership", f"{inst_pct*100:.1f}%" if inst_pct else "N/A")
        ic[1].metric("Insider Ownership", f"{ins_pct*100:.1f}%" if ins_pct else "N/A")

        for h in institutional["holders"][:8]:
            chg = h.get("pct_change", 0)
            chg_color = bull if chg > 0 else (bear if chg < 0 else "#888")
            chg_str = f"{chg*100:+.1f}%" if chg else ""

            st.markdown(
                f"<div style='padding:6px 12px;margin:3px 0;background:{surface};border-radius:4px;"
                f"display:flex;align-items:center'>"
                f"<span style='color:{text_color};font-size:13px;flex:1'>{h.get('holder','')}</span>"
                f"<span style='color:{text_color};font-size:13px;margin-right:15px'>{h.get('shares',0):,} shares</span>"
                f"<span style='color:{chg_color};font-size:12px;font-weight:600'>{chg_str}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ═══════════════════════════════════════════════════════════════
    # NEWS
    # ═══════════════════════════════════════════════════════════════
    if news:
        st.markdown("---")
        st.subheader("Latest News")

        for n in news:
            st.markdown(
                f"<div style='background:{surface};padding:12px 18px;border-radius:8px;margin-bottom:8px'>"
                f"<p style='margin:0;font-weight:600;color:{text_color};font-size:14px'>{n.get('title','')}</p>"
                f"<p style='margin:5px 0 0;color:#888;font-size:12px'>{n.get('publisher','')} | {n.get('date','')}</p>"
                f"{'<p style=\"margin:5px 0 0;color:#aaa;font-size:12px;line-height:1.4\">' + n.get('summary','')[:200] + '</p>' if n.get('summary') else ''}"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ═══════════════════════════════════════════════════════════════
    # POSITION SIZING
    # ═══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Position Sizing Calculator")
    st.caption("How much should you buy based on the analysis?")

    ps_cols = st.columns(3)
    portfolio_size = ps_cols[0].number_input("Portfolio Size ($)", value=10000, step=1000, min_value=100)
    risk_pct = ps_cols[1].number_input("Max Risk Per Trade (%)", value=2.0, step=0.5, min_value=0.5, max_value=10.0)
    stop_pct = ps_cols[2].number_input("Stop Loss (%)", value=round(tech.get("atr_pct", 2.0) * 2, 1), step=0.5, min_value=0.5)

    risk_amount = portfolio_size * (risk_pct / 100)
    shares = int(risk_amount / (current_price * stop_pct / 100)) if current_price and stop_pct else 0
    position_value = shares * current_price

    psc = st.columns(4)
    psc[0].metric("Risk Amount", f"${risk_amount:,.0f}")
    psc[1].metric("Shares to Buy", f"{shares:,}")
    psc[2].metric("Position Value", f"${position_value:,.0f}")
    psc[3].metric("Stop Loss Price", f"${current_price * (1 - stop_pct/100):,.2f}" if current_price else "N/A")

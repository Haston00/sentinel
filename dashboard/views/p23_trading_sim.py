"""
SENTINEL — Page 23: Paper Trading Simulator.
$100K virtual portfolio with Wharton-level investment education.
Guided trades, post-trade analysis, performance analytics.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
import plotly.graph_objects as go

from config.settings import COLORS
from config.assets import SECTORS, CRYPTO_ALL, get_sector_for_ticker
from simulation.portfolio import (
    get_portfolio, execute_trade, reset_portfolio,
    get_trade_history, get_performance_stats, take_daily_snapshot,
)
from simulation.advisor import generate_recommendations, analyze_closed_trade

ET = ZoneInfo("America/New_York")


def render():
    surface = COLORS["surface"]
    text_color = COLORS["text"]
    bull = COLORS["bull"]
    bear = COLORS["bear"]
    neutral = COLORS["neutral"]
    primary = COLORS["primary"]

    st.title("Paper Trading Simulator")
    st.caption("$100K virtual portfolio — learn to invest with real market data and institutional-grade guidance")

    # Take daily snapshot on page load
    try:
        take_daily_snapshot()
    except Exception:
        pass

    # ── Portfolio summary bar ─────────────────────────────────
    port = get_portfolio()
    stats = get_performance_stats()

    total_val = port["total_value"]
    cash = port["cash"]
    pnl = port.get("total_pnl", 0)
    pnl_pct = port.get("total_pnl_pct", 0)
    pnl_color = bull if pnl >= 0 else bear

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.markdown(
        f"<div style='text-align:center;background:{surface};padding:15px;border-radius:10px'>"
        f"<p style='margin:0;color:#aaa;font-size:10px'>PORTFOLIO VALUE</p>"
        f"<p style='margin:4px 0;font-size:28px;font-weight:700;color:{text_color}'>${total_val:,.0f}</p>"
        f"</div>", unsafe_allow_html=True,
    )
    sc2.markdown(
        f"<div style='text-align:center;background:{surface};padding:15px;border-radius:10px'>"
        f"<p style='margin:0;color:#aaa;font-size:10px'>TOTAL P&L</p>"
        f"<p style='margin:4px 0;font-size:28px;font-weight:700;color:{pnl_color}'>{pnl:+,.0f}</p>"
        f"<p style='margin:0;font-size:12px;color:{pnl_color}'>{pnl_pct:+.2f}%</p>"
        f"</div>", unsafe_allow_html=True,
    )
    sc3.markdown(
        f"<div style='text-align:center;background:{surface};padding:15px;border-radius:10px'>"
        f"<p style='margin:0;color:#aaa;font-size:10px'>CASH</p>"
        f"<p style='margin:4px 0;font-size:28px;font-weight:700;color:{text_color}'>${cash:,.0f}</p>"
        f"<p style='margin:0;font-size:12px;color:#aaa'>{cash/total_val*100:.0f}% of portfolio</p>"
        f"</div>", unsafe_allow_html=True,
    )
    sc4.markdown(
        f"<div style='text-align:center;background:{surface};padding:15px;border-radius:10px'>"
        f"<p style='margin:0;color:#aaa;font-size:10px'>WIN RATE</p>"
        f"<p style='margin:4px 0;font-size:28px;font-weight:700;color:{text_color}'>{stats['win_rate']:.0f}%</p>"
        f"<p style='margin:0;font-size:12px;color:#aaa'>{stats['num_trades']} trades</p>"
        f"</div>", unsafe_allow_html=True,
    )
    sc5.markdown(
        f"<div style='text-align:center;background:{surface};padding:15px;border-radius:10px'>"
        f"<p style='margin:0;color:#aaa;font-size:10px'>POSITIONS</p>"
        f"<p style='margin:4px 0;font-size:28px;font-weight:700;color:{text_color}'>{stats['num_positions']}</p>"
        f"<p style='margin:0;font-size:12px;color:#aaa'>Diversification: {stats['diversification_score']:.0f}/100</p>"
        f"</div>", unsafe_allow_html=True,
    )

    st.markdown("")

    # ── Tabs ──────────────────────────────────────────────────
    tabs = st.tabs([
        "Advisor & Trade",
        "My Holdings",
        "Manual Trade",
        "Trade Journal",
        "Performance",
        "Settings",
    ])

    # ══════════════════════════════════════════════════════════
    # TAB 1: ADVISOR & TRADE
    # ══════════════════════════════════════════════════════════
    with tabs[0]:
        st.subheader("Your Investment Advisor")
        st.caption("SENTINEL analyzes every signal and tells you exactly what to do — and teaches you why")

        with st.spinner("Analyzing markets and building recommendations..."):
            try:
                recs = generate_recommendations()
            except Exception as e:
                st.error(f"Advisor error: {e}")
                recs = []

        if not recs:
            st.info("No recommendations right now. The advisor only suggests trades when the signals are clear. Patience is a strategy.")
        else:
            for i, rec in enumerate(recs):
                action = rec.get("action", "")
                ticker = rec.get("ticker", "")
                urgency = rec.get("urgency", "LOW")

                if urgency == "HIGH":
                    urg_color = bear
                elif urgency == "MEDIUM":
                    urg_color = "#FF9800"
                else:
                    urg_color = neutral

                if action == "BUY" or action == "DEPLOY CASH":
                    act_color = bull
                elif action == "SELL" or action == "RAISE CASH":
                    act_color = bear
                else:
                    act_color = primary

                st.markdown(
                    f"<div style='background:{surface};padding:20px;border-radius:12px;margin-bottom:15px;"
                    f"border-left:5px solid {act_color}'>"
                    f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap'>"
                    f"<span style='background:{act_color};color:white;padding:4px 14px;border-radius:15px;font-size:12px;font-weight:700'>{action}</span>"
                    f"<span style='background:{urg_color};color:white;padding:4px 10px;border-radius:15px;font-size:11px'>{urgency}</span>"
                    f"{'<span style=font-size:15px;font-weight:600;color:' + text_color + '>' + ticker + '</span>' if ticker else ''}"
                    f"</div>"
                    f"<h4 style='margin:0 0 10px;color:{text_color};font-size:16px'>{rec.get('headline', '')}</h4>"
                    f"<p style='color:{text_color};font-size:14px;line-height:1.7;margin:0'>{rec.get('thesis', '')}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Suggested position size
                if rec.get("suggested_shares") and rec.get("suggested_cost"):
                    st.markdown(
                        f"<div style='background:rgba(0,176,255,0.08);padding:12px 16px;border-radius:8px;margin:-8px 0 8px'>"
                        f"<span style='color:{primary};font-weight:600;font-size:13px'>Suggested Size:</span> "
                        f"<span style='color:{text_color};font-size:13px'>"
                        f"{rec['suggested_shares']:.4f} shares × current price = ~${rec['suggested_cost']:,.2f} "
                        f"({rec.get('suggested_pct', 0):.1f}% of portfolio)</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # Education box
                edu = rec.get("education", {})
                if edu:
                    with st.expander(f"Learn: {edu.get('title', 'Investment Lesson')}", expanded=False):
                        st.markdown(f"**What it is:** {edu.get('what', '')}")
                        st.markdown(f"**Why it works:** {edu.get('why', '')}")
                        st.markdown(f"**The risk:** {edu.get('risk', '')}")

                # Quick execute button
                if action in ("BUY", "SELL") and ticker and rec.get("suggested_shares"):
                    shares = rec["suggested_shares"]
                    is_crypto = rec.get("is_crypto", False)
                    sector = rec.get("sector", "")

                    col_btn, col_space = st.columns([1, 3])
                    with col_btn:
                        btn_label = f"{action} {shares:.4f} {ticker}" if is_crypto else f"{action} {shares} {ticker}"
                        if st.button(btn_label, key=f"advisor_{i}_{ticker}", type="primary"):
                            result = execute_trade(
                                ticker, action, shares,
                                thesis=rec.get("headline", ""),
                                is_crypto=is_crypto,
                                sector=sector,
                            )
                            if result["success"]:
                                st.success(f"Executed: {action} {shares} {ticker} @ ${result['price']:.2f}")
                                st.rerun()
                            else:
                                st.error(result["error"])

                st.markdown("")

    # ══════════════════════════════════════════════════════════
    # TAB 2: MY HOLDINGS
    # ══════════════════════════════════════════════════════════
    with tabs[1]:
        st.subheader("Current Holdings")
        positions = port.get("positions_detail", [])

        if not positions:
            st.info("No positions yet. Go to the Advisor tab for recommendations on where to start.")
        else:
            for p in positions:
                pnl_c = bull if p["pnl"] >= 0 else bear
                type_badge = "Crypto" if p.get("is_crypto") else p.get("sector", "Stock")

                st.markdown(
                    f"<div style='background:{surface};padding:16px 20px;border-radius:10px;margin-bottom:8px;"
                    f"border-left:4px solid {pnl_c}'>"
                    f"<div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap'>"
                    f"<div>"
                    f"<span style='font-size:18px;font-weight:700;color:{text_color}'>{p['ticker']}</span>"
                    f"<span style='background:rgba(0,176,255,0.1);color:{primary};padding:2px 8px;border-radius:8px;font-size:10px;margin-left:8px'>{type_badge}</span>"
                    f"<span style='color:#888;font-size:12px;margin-left:8px'>{p['shares']:.4f} shares</span>"
                    f"</div>"
                    f"<div style='text-align:right'>"
                    f"<span style='font-size:16px;font-weight:600;color:{text_color}'>${p['market_value']:,.2f}</span>"
                    f"<span style='color:{pnl_c};font-size:14px;margin-left:12px;font-weight:600'>{p['pnl']:+,.2f} ({p['pnl_pct']:+.1f}%)</span>"
                    f"</div>"
                    f"</div>"
                    f"<div style='display:flex;gap:20px;margin-top:8px;flex-wrap:wrap'>"
                    f"<span style='color:#888;font-size:12px'>Avg Cost: ${p['avg_cost']:.2f}</span>"
                    f"<span style='color:#888;font-size:12px'>Current: ${p['current_price']:.2f}</span>"
                    f"<span style='color:#888;font-size:12px'>Weight: {p['weight']:.1f}%</span>"
                    f"</div>"
                    f"{'<p style=margin:6px 0 0;color:#666;font-size:11px;font-style:italic>Thesis: ' + p.get('thesis', '') + '</p>' if p.get('thesis') else ''}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Allocation pie chart
            if len(positions) > 1:
                st.markdown("")
                st.markdown(f"**Portfolio Allocation**")
                labels = [p["ticker"] for p in positions] + ["Cash"]
                values = [p["market_value"] for p in positions] + [cash]
                colors_list = [bull if p["pnl"] >= 0 else bear for p in positions] + ["#555"]

                fig = go.Figure(go.Pie(
                    labels=labels, values=values,
                    hole=0.4,
                    marker=dict(colors=colors_list),
                    textinfo="label+percent",
                    textfont=dict(size=12, color="white"),
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # TAB 3: MANUAL TRADE
    # ══════════════════════════════════════════════════════════
    with tabs[2]:
        st.subheader("Execute a Trade")
        st.caption("Enter your own trades — the system will track everything and teach you from the results")

        tc1, tc2 = st.columns(2)

        with tc1:
            trade_action = st.selectbox("Action", ["BUY", "SELL"], key="manual_action")
            asset_type = st.selectbox("Asset Type", ["Stock / ETF", "Crypto"], key="manual_type")

            if asset_type == "Crypto":
                crypto_options = sorted([info["symbol"] for info in CRYPTO_ALL.values()])
                trade_ticker = st.selectbox("Coin", crypto_options, key="manual_crypto")
                is_crypto = True
            else:
                trade_ticker = st.text_input("Ticker Symbol", value="", key="manual_ticker").upper().strip()
                is_crypto = False

        with tc2:
            trade_shares = st.number_input(
                "Shares" if not is_crypto else "Amount",
                min_value=0.0001 if is_crypto else 1.0,
                value=10.0 if not is_crypto else 0.01,
                step=1.0 if not is_crypto else 0.01,
                key="manual_shares",
            )
            trade_thesis = st.text_area("Why are you making this trade? (optional — helps you learn)", key="manual_thesis", height=80)

            sector = ""
            if not is_crypto and trade_ticker:
                sector = get_sector_for_ticker(trade_ticker) or ""
                if sector:
                    st.caption(f"Sector: {sector}")

        if trade_ticker:
            if st.button(f"Execute {trade_action} — {trade_shares} {trade_ticker}", type="primary", key="manual_exec"):
                result = execute_trade(
                    trade_ticker, trade_action, trade_shares,
                    thesis=trade_thesis,
                    is_crypto=is_crypto,
                    sector=sector,
                )
                if result["success"]:
                    trade = result["trade"]
                    st.success(
                        f"Done! {trade_action} {trade_shares} {trade_ticker} @ ${result['price']:.2f} "
                        f"(Total: ${trade.get('total', 0):,.2f})"
                    )
                    if trade_action == "SELL" and "realized_pnl" in trade:
                        pnl_val = trade["realized_pnl"]
                        pnl_word = "profit" if pnl_val >= 0 else "loss"
                        st.info(f"Realized {pnl_word}: ${abs(pnl_val):,.2f} ({trade.get('realized_pnl_pct', 0):+.1f}%)")
                    st.rerun()
                else:
                    st.error(result["error"])

    # ══════════════════════════════════════════════════════════
    # TAB 4: TRADE JOURNAL
    # ══════════════════════════════════════════════════════════
    with tabs[3]:
        st.subheader("Trade Journal")
        st.caption("Every trade you have made — with grades and lessons on closed positions")

        history = get_trade_history()
        if not history:
            st.info("No trades yet. Your journal will fill up as you start investing.")
        else:
            for trade in history[:50]:
                action = trade.get("action", "")
                ticker = trade.get("ticker", "")
                price = trade.get("price", 0)
                shares = trade.get("shares", 0)
                ts = trade.get("timestamp", "")
                is_sell = action == "SELL"

                if is_sell:
                    pnl_val = trade.get("realized_pnl", 0)
                    pnl_pct = trade.get("realized_pnl_pct", 0)
                    tc = bull if pnl_val >= 0 else bear

                    st.markdown(
                        f"<div style='background:{surface};padding:14px 18px;border-radius:8px;margin-bottom:8px;"
                        f"border-left:4px solid {tc}'>"
                        f"<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap'>"
                        f"<div>"
                        f"<span style='background:{tc};color:white;padding:2px 10px;border-radius:8px;font-size:11px;font-weight:600'>SELL</span>"
                        f"<span style='color:{text_color};font-weight:600;font-size:14px;margin-left:8px'>{ticker}</span>"
                        f"<span style='color:#888;font-size:12px;margin-left:8px'>{shares} shares @ ${price:.2f}</span>"
                        f"</div>"
                        f"<div>"
                        f"<span style='color:{tc};font-weight:700;font-size:14px'>{pnl_val:+,.2f} ({pnl_pct:+.1f}%)</span>"
                        f"</div>"
                        f"</div>"
                        f"<p style='margin:4px 0 0;color:#666;font-size:11px'>{ts}</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # Post-trade analysis
                    analysis = analyze_closed_trade(trade)
                    with st.expander(f"Trade Analysis: {analysis.get('grade', '?')} — {ticker}", expanded=False):
                        st.write(analysis.get("summary", ""))
                        for lesson in analysis.get("lessons", []):
                            st.markdown(
                                f"<div style='background:rgba(0,176,255,0.06);padding:10px 14px;border-radius:6px;margin:6px 0;"
                                f"border-left:3px solid {primary}'>"
                                f"<span style='color:{text_color};font-size:13px'>{lesson}</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                else:
                    st.markdown(
                        f"<div style='background:{surface};padding:14px 18px;border-radius:8px;margin-bottom:8px;"
                        f"border-left:4px solid {bull}'>"
                        f"<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap'>"
                        f"<div>"
                        f"<span style='background:{bull};color:white;padding:2px 10px;border-radius:8px;font-size:11px;font-weight:600'>BUY</span>"
                        f"<span style='color:{text_color};font-weight:600;font-size:14px;margin-left:8px'>{ticker}</span>"
                        f"<span style='color:#888;font-size:12px;margin-left:8px'>{shares} shares @ ${price:.2f}</span>"
                        f"</div>"
                        f"<div>"
                        f"<span style='color:#888;font-size:13px'>-${trade.get('total', 0):,.2f}</span>"
                        f"</div>"
                        f"</div>"
                        f"{'<p style=margin:4px 0 0;color:#666;font-size:11px;font-style:italic>' + trade.get('thesis', '') + '</p>' if trade.get('thesis') else ''}"
                        f"<p style='margin:4px 0 0;color:#666;font-size:11px'>{ts}</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    # ══════════════════════════════════════════════════════════
    # TAB 5: PERFORMANCE
    # ══════════════════════════════════════════════════════════
    with tabs[4]:
        st.subheader("Performance Analytics")
        st.caption("How your portfolio stacks up — the same metrics hedge funds report to their investors")

        # Key metrics cards
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(
            f"<div style='text-align:center;background:{surface};padding:15px;border-radius:10px'>"
            f"<p style='margin:0;color:#aaa;font-size:10px'>TOTAL RETURN</p>"
            f"<p style='margin:4px 0;font-size:24px;font-weight:700;color:{bull if stats['total_return_pct'] >= 0 else bear}'>{stats['total_return_pct']:+.2f}%</p>"
            f"</div>", unsafe_allow_html=True,
        )
        m2.markdown(
            f"<div style='text-align:center;background:{surface};padding:15px;border-radius:10px'>"
            f"<p style='margin:0;color:#aaa;font-size:10px'>SHARPE RATIO</p>"
            f"<p style='margin:4px 0;font-size:24px;font-weight:700;color:{text_color}'>{stats['sharpe_ratio']:.2f}</p>"
            f"<p style='margin:0;color:#888;font-size:10px'>{'Good' if stats['sharpe_ratio'] > 1 else 'Needs work' if stats['sharpe_ratio'] > 0 else 'Negative'}</p>"
            f"</div>", unsafe_allow_html=True,
        )
        m3.markdown(
            f"<div style='text-align:center;background:{surface};padding:15px;border-radius:10px'>"
            f"<p style='margin:0;color:#aaa;font-size:10px'>MAX DRAWDOWN</p>"
            f"<p style='margin:4px 0;font-size:24px;font-weight:700;color:{bear}'>{stats['max_drawdown_pct']:.1f}%</p>"
            f"<p style='margin:0;color:#888;font-size:10px'>{'Excellent' if stats['max_drawdown_pct'] < 5 else 'Acceptable' if stats['max_drawdown_pct'] < 15 else 'High'}</p>"
            f"</div>", unsafe_allow_html=True,
        )
        m4.markdown(
            f"<div style='text-align:center;background:{surface};padding:15px;border-radius:10px'>"
            f"<p style='margin:0;color:#aaa;font-size:10px'>PROFIT FACTOR</p>"
            f"<p style='margin:4px 0;font-size:24px;font-weight:700;color:{text_color}'>{stats['profit_factor']}</p>"
            f"<p style='margin:0;color:#888;font-size:10px'>{'Strong' if str(stats['profit_factor']) == '∞' or (isinstance(stats['profit_factor'], (int, float)) and stats['profit_factor'] > 2) else 'OK' if isinstance(stats['profit_factor'], (int, float)) and stats['profit_factor'] > 1 else 'Losing'}</p>"
            f"</div>", unsafe_allow_html=True,
        )

        # Metric explanations
        with st.expander("What do these metrics mean?", expanded=False):
            st.markdown(f"""
**Sharpe Ratio** — Risk-adjusted return. How much return you earn per unit of risk.
- Above 2.0 = Excellent (hedge fund level)
- 1.0-2.0 = Good
- 0.5-1.0 = Average
- Below 0.5 = You are taking too much risk for your returns

**Max Drawdown** — The biggest peak-to-trough drop. If your portfolio hit $110K then dropped to $95K, that is a 13.6% drawdown.
- Below 10% = Excellent risk control
- 10-20% = Normal for an equity portfolio
- Above 20% = You need better risk management

**Profit Factor** — Total money won ÷ total money lost.
- Above 2.0 = Very good — you make $2 for every $1 you lose
- 1.5-2.0 = Solid
- Below 1.0 = You are losing money overall

**Win Rate** — Percentage of trades that made money. Most successful traders win 50-60% of the time. The key is making more on winners than you lose on losers.

**Sortino Ratio** ({stats['sortino_ratio']:.2f}) — Like Sharpe, but only penalizes downside volatility. A more fair measure because upside volatility (big gains) should not count against you.

**Annualized Volatility** ({stats['volatility_annual_pct']:.1f}%) — How much your portfolio bounces around. S&P 500 averages about 15-16%. Below 12% is conservative, above 20% is aggressive.
""")

        # P&L breakdown
        st.markdown("---")
        p1, p2 = st.columns(2)
        with p1:
            st.markdown(
                f"<div style='background:{surface};padding:18px;border-radius:10px'>"
                f"<p style='margin:0;color:#aaa;font-size:11px'>REALIZED P&L (closed trades)</p>"
                f"<p style='margin:6px 0;font-size:22px;font-weight:700;color:{bull if stats['total_realized_pnl'] >= 0 else bear}'>${stats['total_realized_pnl']:+,.2f}</p>"
                f"<p style='margin:0;color:#888;font-size:12px'>Avg Win: ${stats['avg_win']:,.2f} | Avg Loss: ${stats['avg_loss']:,.2f}</p>"
                f"</div>", unsafe_allow_html=True,
            )
        with p2:
            st.markdown(
                f"<div style='background:{surface};padding:18px;border-radius:10px'>"
                f"<p style='margin:0;color:#aaa;font-size:11px'>UNREALIZED P&L (open positions)</p>"
                f"<p style='margin:6px 0;font-size:22px;font-weight:700;color:{bull if stats['total_unrealized_pnl'] >= 0 else bear}'>${stats['total_unrealized_pnl']:+,.2f}</p>"
                f"<p style='margin:0;color:#888;font-size:12px'>{stats['num_positions']} open positions</p>"
                f"</div>", unsafe_allow_html=True,
            )

        # Sector allocation
        sector_alloc = stats.get("sector_allocation", {})
        if sector_alloc:
            st.markdown("---")
            st.markdown("**Sector Allocation**")
            for sec, weight in sorted(sector_alloc.items(), key=lambda x: x[1], reverse=True):
                bar_w = max(3, min(100, weight))
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:10px;margin:4px 0'>"
                    f"<span style='color:{text_color};font-size:12px;min-width:140px'>{sec}</span>"
                    f"<div style='flex:1;background:#333;height:14px;border-radius:4px'>"
                    f"<div style='background:{primary};height:14px;border-radius:4px;width:{bar_w}%'></div></div>"
                    f"<span style='color:#888;font-size:12px;min-width:40px;text-align:right'>{weight:.1f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Portfolio value chart (from snapshots)
        state = port
        snapshots = get_portfolio()  # Already has state
        # Get raw state for snapshots
        from simulation.portfolio import _load_state
        raw = _load_state()
        snaps = raw.get("daily_snapshots", [])
        if len(snaps) >= 2:
            st.markdown("---")
            st.markdown("**Portfolio Value Over Time**")
            dates = [s["date"] for s in snaps]
            values = [s["total_value"] for s in snaps]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=values,
                mode="lines+markers",
                line=dict(color=primary, width=2),
                fill="tozeroy",
                fillcolor="rgba(0,176,255,0.08)",
            ))
            fig.add_hline(y=100000, line_dash="dash", line_color="#555",
                          annotation_text="Starting $100K", annotation_position="bottom right")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(color="#888", gridcolor="#222"),
                yaxis=dict(color="#888", gridcolor="#222", tickprefix="$", tickformat=","),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # TAB 6: SETTINGS
    # ══════════════════════════════════════════════════════════
    with tabs[5]:
        st.subheader("Portfolio Settings")

        st.markdown(
            f"<div style='background:{surface};padding:20px;border-radius:10px;margin-bottom:15px'>"
            f"<p style='color:{text_color};font-size:15px;font-weight:600;margin:0 0 8px'>About This Simulator</p>"
            f"<p style='color:{text_color};font-size:13px;line-height:1.6;margin:0'>"
            "This is a paper trading account — no real money is involved. "
            "You start with $100,000 in virtual cash and can buy and sell any stock, ETF, or cryptocurrency using real market prices. "
            "Every trade is tracked, graded, and analyzed so you learn from both your wins and losses. "
            "The advisor uses SENTINEL's full signal engine (the same 8-factor scoring system, regime detection, "
            "sector analysis, and intermarket signals) to suggest trades with institutional-grade reasoning. "
            "Think of it as your personal trading mentor.</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.warning("Resetting your portfolio will delete ALL positions and trade history. This cannot be undone.")
        if st.button("Reset Portfolio to $100,000", type="secondary", key="reset_port"):
            reset_portfolio()
            st.success("Portfolio reset to $100,000. Fresh start!")
            st.rerun()

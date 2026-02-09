"""
SENTINEL — Page 19: Options Flow Intelligence.
Smart money tracking. Put/call ratios. Unusual activity detection.
Gamma exposure. The options market leads the stock market.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from config.settings import COLORS


def render():
    st.title("Options Flow Intelligence")
    st.caption(
        "The options market leads the stock market. Track smart money positioning, "
        "unusual activity, and dealer gamma exposure in real time."
    )

    surface = COLORS["surface"]
    text_color = COLORS["text"]
    bull = COLORS["bull"]
    bear = COLORS["bear"]
    primary = COLORS["primary"]
    bg = COLORS["background"]

    # ── Ticker Selection ──────────────────────────────────────────
    default_tickers = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "AMD"]
    c1, c2 = st.columns([2, 1])
    with c1:
        ticker = st.selectbox("Select Ticker", default_tickers, key="opt_ticker")
    with c2:
        if st.button("Analyze Options", type="primary", key="opt_analyze"):
            st.session_state["opt_run"] = True

    if not st.session_state.get("opt_run"):
        st.info(
            "**How it works:**\n\n"
            "1. Select a ticker above and click 'Analyze Options'\n"
            "2. The system pulls the full options chain (calls + puts)\n"
            "3. Computes put/call ratios, unusual activity, gamma exposure, and IV analysis\n"
            "4. Generates a composite signal showing smart money direction\n\n"
            "**Key signals:** Put/Call ratio > 1.0 = bearish, < 0.7 = bullish. "
            "Volume >> Open Interest = new position being opened (smart money)."
        )
        return

    with st.spinner(f"Pulling options chain for {ticker}..."):
        try:
            from data.options import get_options_intelligence
            intel = get_options_intelligence(ticker)
        except Exception as e:
            st.error(f"Options analysis failed: {e}")
            return

    if not intel or intel.get("put_call_ratio", {}).get("signal") == "no_data":
        st.warning(f"No options data available for {ticker}. Try a different ticker with listed options.")
        return

    # ═══════════════════════════════════════════════════════════════
    # COMPOSITE SIGNAL BANNER
    # ═══════════════════════════════════════════════════════════════
    composite = intel.get("composite_signal", 0)
    direction = intel.get("composite_direction", "neutral")
    d_color = bull if direction == "bullish" else (bear if direction == "bearish" else COLORS["neutral"])

    st.markdown(
        f"<div style='background:{surface};padding:20px;border-radius:12px;"
        f"border-left:5px solid {d_color};margin-bottom:20px'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap'>"
        f"<div>"
        f"<span style='font-size:14px;color:#888'>OPTIONS FLOW SIGNAL</span><br>"
        f"<span style='font-size:32px;font-weight:700;color:{d_color}'>"
        f"{direction.upper()}</span>"
        f"<span style='color:#888;margin-left:15px'>Composite: {composite:+.3f}</span>"
        f"</div>"
        f"<div style='text-align:right'>"
        f"<span style='color:{text_color};font-size:24px;font-weight:700'>"
        f"${intel.get('spot_price', 0):,.2f}</span>"
        f"<br><span style='color:#888'>{ticker} Spot Price</span>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    # ═══════════════════════════════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4 = st.tabs([
        "Put/Call Ratio", "Unusual Activity", "Gamma Exposure", "IV Analysis"
    ])

    # ── TAB 1: Put/Call Ratio ─────────────────────────────────────
    with tab1:
        pc = intel.get("put_call_ratio", {})
        st.subheader("Put/Call Ratio Analysis")

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Volume P/C", f"{pc.get('volume_pc', 0):.3f}")
        with mc2:
            st.metric("OI P/C", f"{pc.get('oi_pc', 0):.3f}")
        with mc3:
            st.metric("Call Volume", f"{pc.get('call_volume', 0):,}")
        with mc4:
            st.metric("Put Volume", f"{pc.get('put_volume', 0):,}")

        signal = pc.get("signal", "neutral")
        sig_color = bear if "bearish" in signal else (bull if "bullish" in signal else COLORS["neutral"])
        st.markdown(
            f"<div style='background:{surface};padding:15px;border-radius:8px;"
            f"border-left:4px solid {sig_color}'>"
            f"<strong style='color:{sig_color}'>{signal.replace('_', ' ').upper()}</strong>"
            f"<span style='color:#888;margin-left:10px'>Average P/C: {pc.get('avg_pc', 0):.3f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # P/C Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pc.get("avg_pc", 0),
            title={"text": "Put/Call Ratio"},
            gauge={
                "axis": {"range": [0, 2]},
                "bar": {"color": sig_color},
                "steps": [
                    {"range": [0, 0.7], "color": "#1a3d1a"},
                    {"range": [0.7, 1.0], "color": "#2a2a1a"},
                    {"range": [1.0, 2.0], "color": "#3d1a1a"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "value": 1.0,
                    "thickness": 0.75,
                },
            },
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=bg,
            height=300, margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 2: Unusual Activity ───────────────────────────────────
    with tab2:
        unusual = intel.get("unusual_activity", [])
        st.subheader(f"Unusual Options Activity ({len(unusual)} contracts)")
        st.markdown(
            "When **Volume >> Open Interest**, someone is opening a large new position. "
            "This is the #1 institutional signal."
        )

        if not unusual:
            st.info("No unusual activity detected today.")
        else:
            for u in unusual[:15]:
                u_color = bull if u["type"] == "call" else bear
                itm = " (ITM)" if u["in_the_money"] else ""

                st.markdown(
                    f"<div style='background:{surface};padding:12px;border-radius:8px;"
                    f"border-left:3px solid {u_color};margin-bottom:6px;"
                    f"display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap'>"
                    f"<div>"
                    f"<strong style='color:{u_color}'>"
                    f"{u['type'].upper()}</strong>"
                    f" <span style='color:{text_color}'>${u['strike']:.0f}</span>"
                    f" <span style='color:#888'>exp {u['expiration']}{itm}</span>"
                    f"</div>"
                    f"<div style='display:flex;gap:20px'>"
                    f"<div><span style='color:#888'>Vol: </span>"
                    f"<span style='color:{text_color}'>{u['volume']:,}</span></div>"
                    f"<div><span style='color:#888'>OI: </span>"
                    f"<span style='color:{text_color}'>{u['open_interest']:,}</span></div>"
                    f"<div><span style='color:#888'>Ratio: </span>"
                    f"<span style='color:{u_color};font-weight:700'>"
                    f"{u['vol_oi_ratio']:.1f}×</span></div>"
                    f"<div><span style='color:#888'>IV: </span>"
                    f"<span style='color:{text_color}'>{u['implied_vol']:.1%}</span></div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

    # ── TAB 3: Gamma Exposure ─────────────────────────────────────
    with tab3:
        gamma = intel.get("gamma_exposure", {})
        st.subheader("Dealer Gamma Exposure (GEX)")
        st.markdown(
            "**Positive gamma:** Dealers hedge by buying dips/selling rallies → dampens moves.\n\n"
            "**Negative gamma:** Dealers hedge by selling dips/buying rallies → amplifies moves."
        )

        if not gamma or gamma.get("regime") == "unknown":
            st.info("Gamma exposure data not available for this ticker.")
        else:
            regime = gamma.get("regime", "unknown")
            g_color = bull if regime == "positive_gamma" else bear

            st.markdown(
                f"<div style='background:{surface};padding:20px;border-radius:12px;"
                f"border-left:5px solid {g_color};margin-bottom:15px'>"
                f"<span style='font-size:14px;color:#888'>GAMMA REGIME</span><br>"
                f"<span style='font-size:28px;font-weight:700;color:{g_color}'>"
                f"{regime.replace('_', ' ').upper()}</span><br>"
                f"<span style='color:#888;font-size:13px'>"
                f"{gamma.get('interpretation', '')}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            gc1, gc2 = st.columns(2)
            with gc1:
                st.metric("Net Gamma ($)", f"${gamma.get('net_gamma', 0):,.0f}")
            with gc2:
                flip = gamma.get("gamma_flip")
                st.metric("Gamma Flip Point", f"${flip:,.0f}" if flip else "N/A")

    # ── TAB 4: IV Analysis ────────────────────────────────────────
    with tab4:
        iv = intel.get("iv_analysis", {})
        st.subheader("Implied Volatility Analysis")

        if not iv or iv.get("mean_iv") is None:
            st.info("IV data not available.")
        else:
            ic1, ic2, ic3, ic4 = st.columns(4)
            with ic1:
                st.metric("Mean IV", f"{iv.get('mean_iv', 0):.1%}")
            with ic2:
                st.metric("Median IV", f"{iv.get('median_iv', 0):.1%}")
            with ic3:
                st.metric("IV Skew", f"{iv.get('iv_skew', 0):.4f}")
            with ic4:
                skew_sig = iv.get("skew_signal", "neutral")
                st.metric("Skew Signal", skew_sig.replace("_", " ").title())

            st.markdown(
                f"**IV Range:** {iv.get('min_iv', 0):.1%} — {iv.get('max_iv', 0):.1%}"
            )

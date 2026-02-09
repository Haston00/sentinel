"""
SENTINEL — Page 21: Volatility Surface.
Implied vol term structure, volatility smile/skew, and the
Volatility Risk Premium — the institutional edge in options markets.
"""

import streamlit as st
import plotly.graph_objects as go

from config.settings import COLORS


def render():
    st.title("Volatility Surface")
    st.caption(
        "Implied volatility term structure, smile analysis, and the Volatility Risk Premium. "
        "When implied vol diverges from realized vol, there is a tradeable edge."
    )

    surface = COLORS["surface"]
    text_color = COLORS["text"]
    bull = COLORS["bull"]
    bear = COLORS["bear"]
    primary = COLORS["primary"]
    bg = COLORS["background"]
    neutral = COLORS["neutral"]

    # ── Ticker Selection ──────────────────────────────────────────
    default_tickers = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL"]
    c1, c2 = st.columns([2, 1])
    with c1:
        ticker = st.selectbox("Select Ticker", default_tickers, key="vol_ticker")
    with c2:
        if st.button("Analyze Volatility", type="primary", key="vol_analyze"):
            st.session_state["vol_run"] = True

    if not st.session_state.get("vol_run"):
        st.info(
            "**Volatility Risk Premium (VRP):**\n"
            "- IV > Realized Vol = options overpriced (sell vol / expect mean reversion)\n"
            "- IV < Realized Vol = options cheap (buy vol / expect breakout)\n"
            "- VRP is positive ~85% of the time — but when it flips negative, watch out\n\n"
            "**Term Structure:**\n"
            "- Contango (upward) = normal, calm markets\n"
            "- Backwardation (inverted) = near-term fear, event expected"
        )
        return

    with st.spinner(f"Building volatility surface for {ticker}..."):
        try:
            from models.vol_surface import get_vol_intelligence
            intel = get_vol_intelligence(ticker)
        except Exception as e:
            st.error(f"Volatility analysis failed: {e}")
            return

    # ═══════════════════════════════════════════════════════════════
    # VRP BANNER
    # ═══════════════════════════════════════════════════════════════
    vrp_data = intel.get("vol_risk_premium", {})
    vrp = vrp_data.get("vrp")
    vol_regime = intel.get("vol_regime", "unknown")
    regime_color = (
        bear if vol_regime == "high_fear"
        else neutral if vol_regime == "elevated"
        else bull if vol_regime == "complacent"
        else text_color
    )

    if vrp is not None:
        st.markdown(
            f"<div style='background:{surface};padding:20px;border-radius:12px;"
            f"border-left:5px solid {regime_color};margin-bottom:20px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap'>"
            f"<div>"
            f"<span style='font-size:14px;color:#888'>VOLATILITY REGIME</span><br>"
            f"<span style='font-size:28px;font-weight:700;color:{regime_color}'>"
            f"{vol_regime.replace('_', ' ').upper()}</span>"
            f"</div>"
            f"<div style='display:flex;gap:30px'>"
            f"<div style='text-align:center'>"
            f"<span style='font-size:24px;font-weight:700;color:{text_color}'>"
            f"{vrp_data.get('implied_vol', 0):.1%}</span>"
            f"<br><span style='color:#888;font-size:11px'>IMPLIED VOL</span></div>"
            f"<div style='text-align:center'>"
            f"<span style='font-size:24px;font-weight:700;color:{text_color}'>"
            f"{vrp_data.get('realized_vol_20d', 0):.1%}</span>"
            f"<br><span style='color:#888;font-size:11px'>REALIZED VOL (20D)</span></div>"
            f"<div style='text-align:center'>"
            f"<span style='font-size:24px;font-weight:700;color:"
            f"{bull if vrp > 0 else bear}'>{vrp:+.1%}</span>"
            f"<br><span style='color:#888;font-size:11px'>VRP</span></div>"
            f"</div></div>"
            f"<p style='color:#888;font-size:12px;margin-top:10px'>"
            f"{vrp_data.get('interpretation', '')}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ═══════════════════════════════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════════════════════════════
    tab1, tab2, tab3 = st.tabs(["Term Structure", "Vol Smile", "Risk Premium"])

    # ── TAB 1: IV Term Structure ──────────────────────────────────
    with tab1:
        ts = intel.get("term_structure", {})
        term_data = ts.get("term_structure", [])
        st.subheader("IV Term Structure")

        if not term_data:
            st.info("No term structure data available.")
        else:
            shape = ts.get("shape", "unknown")
            shape_color = bull if shape == "contango" else (bear if shape == "backwardation" else neutral)

            st.markdown(
                f"**Shape:** <span style='color:{shape_color}'>"
                f"{shape.upper()}</span> — {ts.get('interpretation', '')}",
                unsafe_allow_html=True,
            )

            days = [d["days_to_exp"] for d in term_data]
            ivs = [d["atm_iv_annualized"] for d in term_data]
            labels = [d["expiration"] for d in term_data]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=days, y=ivs, mode="lines+markers",
                marker=dict(size=10, color=primary),
                line=dict(color=primary, width=2),
                text=labels, hovertemplate="%{text}<br>%{y:.1f}% IV<br>%{x} days to exp",
            ))
            fig.update_layout(
                title="ATM Implied Volatility by Expiration",
                template="plotly_dark",
                paper_bgcolor=bg, plot_bgcolor=surface,
                xaxis_title="Days to Expiration",
                yaxis_title="IV (Annualized %)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 2: Volatility Smile ───────────────────────────────────
    with tab2:
        smile_data = intel.get("vol_smile", {})
        smile = smile_data.get("smile", [])
        st.subheader("Volatility Smile / Skew")

        if not smile:
            st.info("No smile data available.")
        else:
            skew = smile_data.get("skew_metric")
            if skew is not None:
                skew_color = bear if skew > 0.05 else (bull if skew < -0.02 else neutral)
                st.markdown(
                    f"**Put-Call Skew:** <span style='color:{skew_color}'>{skew:.4f}</span>"
                    f" — {smile_data.get('skew_signal', 'unknown').replace('_', ' ').title()}",
                    unsafe_allow_html=True,
                )

            calls = [s for s in smile if s["type"] == "call"]
            puts = [s for s in smile if s["type"] == "put"]

            fig = go.Figure()
            if calls:
                fig.add_trace(go.Scatter(
                    x=[c["strike"] for c in calls],
                    y=[c["iv"] * 100 for c in calls],
                    mode="markers+lines", name="Calls",
                    marker=dict(color=bull, size=6),
                    line=dict(color=bull, width=1),
                ))
            if puts:
                fig.add_trace(go.Scatter(
                    x=[p["strike"] for p in puts],
                    y=[p["iv"] * 100 for p in puts],
                    mode="markers+lines", name="Puts",
                    marker=dict(color=bear, size=6),
                    line=dict(color=bear, width=1),
                ))

            spot = smile_data.get("spot", 0)
            if spot:
                fig.add_vline(x=spot, line_dash="dash", line_color="white",
                              annotation_text=f"Spot ${spot:.0f}")

            fig.update_layout(
                title=f"Volatility Smile — {smile_data.get('expiration', '')}",
                template="plotly_dark",
                paper_bgcolor=bg, plot_bgcolor=surface,
                xaxis_title="Strike Price",
                yaxis_title="Implied Volatility %",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 3: Vol Risk Premium Detail ────────────────────────────
    with tab3:
        st.subheader("Volatility Risk Premium Detail")

        if not vrp_data or vrp is None:
            st.info("VRP data not available.")
        else:
            vc1, vc2, vc3 = st.columns(3)
            with vc1:
                st.metric("Implied Vol", f"{vrp_data.get('implied_vol', 0):.1%}")
            with vc2:
                st.metric("Realized Vol (20d)", f"{vrp_data.get('realized_vol_20d', 0):.1%}")
            with vc3:
                st.metric("Realized Vol (60d)", f"{vrp_data.get('realized_vol_60d', 0):.1%}")

            vrp_signal = vrp_data.get("signal", "")
            sig_color = (
                bear if "cheap" in vrp_signal or "danger" in vrp_signal
                else bull if "expensive" in vrp_signal
                else neutral
            )

            st.markdown(
                f"<div style='background:{surface};padding:15px;border-radius:8px;"
                f"border-left:4px solid {sig_color}'>"
                f"<strong style='color:{sig_color}'>"
                f"{vrp_signal.replace('_', ' ').upper()}</strong>"
                f"<br><span style='color:#888'>{vrp_data.get('interpretation', '')}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # VRP gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=vrp * 100,
                number={"suffix": "%"},
                title={"text": "Volatility Risk Premium"},
                delta={"reference": 3, "suffix": "%"},
                gauge={
                    "axis": {"range": [-15, 15]},
                    "bar": {"color": primary},
                    "steps": [
                        {"range": [-15, -3], "color": "#3d1a1a"},
                        {"range": [-3, 3], "color": "#2a2a1a"},
                        {"range": [3, 15], "color": "#1a3d1a"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 2},
                        "value": 0, "thickness": 0.75,
                    },
                },
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor=bg,
                height=300, margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

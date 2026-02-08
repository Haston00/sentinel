"""
SENTINEL — Page 11: Sector Rotation & Market Breadth.
Where is money flowing? What does the cycle say? Are we at a top or bottom?
The indicators that institutional traders watch.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from config.settings import COLORS
from data.stocks import fetch_ohlcv
from features.breadth import compute_sector_breadth, compute_breadth_divergence
from features.rotation import (
    compute_sector_momentum,
    detect_cycle_phase,
    get_rotation_recommendations,
    CYCLE_SECTORS,
)


@st.cache_data(ttl=300, show_spinner=False)
def _load_ticker(ticker: str) -> pd.DataFrame:
    return fetch_ohlcv(ticker)


def render():
    st.title("Sector Rotation & Market Breadth")
    st.caption("Follow the smart money — see where capital is flowing before it moves the market")
    surface = COLORS["surface"]
    text_color = COLORS["text"]
    bull_color = COLORS["bull"]
    bear_color = COLORS["bear"]

    tab_rotation, tab_breadth = st.tabs(["Sector Rotation", "Market Breadth"])

    # ═══════════════════════════════════════════════════════
    # TAB 1: SECTOR ROTATION
    # ═══════════════════════════════════════════════════════
    with tab_rotation:
        with st.spinner("Computing sector momentum and cycle position..."):
            momentum = compute_sector_momentum()

        if momentum.empty:
            st.warning("Insufficient data for sector rotation analysis")
            return

        cycle = detect_cycle_phase(momentum)

        # ── Cycle Phase Indicator ──────────────────────────
        st.subheader("Economic Cycle Position")
        phase = cycle["phase"]
        confidence = cycle["confidence"]

        phase_colors = {
            "Early Recovery": COLORS["bull"],
            "Mid Cycle": "#8BC34A",
            "Late Cycle": "#FF9800",
            "Recession": COLORS["bear"],
        }
        phase_color = phase_colors.get(phase, "#888")

        st.markdown(
            f"<div style='background:{surface};padding:25px;border-radius:12px;"
            f"border-left:5px solid {phase_color};margin-bottom:20px'>"
            f"<h2 style='margin:0;color:{phase_color}'>{phase}</h2>"
            f"<p style='margin:8px 0;color:#ccc;font-size:15px'>{cycle['description']}</p>"
            f"<p style='margin:0;color:#888'>Confidence: {confidence}%</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Expected leaders/laggards
        leaders = cycle.get("expected_leaders", [])
        laggards = cycle.get("expected_laggards", [])
        lc1, lc2 = st.columns(2)
        with lc1:
            st.markdown(f"**Expected Leaders** ({phase}):")
            for s in leaders:
                st.markdown(f"<span style='color:{bull_color}'>+ {s}</span>", unsafe_allow_html=True)
        with lc2:
            st.markdown(f"**Expected Laggards** ({phase}):")
            for s in laggards:
                st.markdown(f"<span style='color:{bear_color}'>- {s}</span>", unsafe_allow_html=True)

        # ── Sector Momentum Rankings ───────────────────────
        st.markdown("---")
        st.subheader("Sector Momentum Rankings")

        # Momentum bar chart
        sorted_mom = momentum.sort_values("Momentum_Score", ascending=True)
        fig = go.Figure(go.Bar(
            x=sorted_mom["Momentum_Score"] * 100,
            y=sorted_mom.index,
            orientation="h",
            marker_color=[
                COLORS["bull"] if m > 0.01 else COLORS["bear"] if m < -0.01 else COLORS["neutral"]
                for m in sorted_mom["Momentum_Score"]
            ],
            text=[f"{m*100:+.1f}%" for m in sorted_mom["Momentum_Score"]],
            textposition="outside",
        ))
        fig.add_vline(x=0, line_color="#555")
        fig.update_layout(
            title="Relative Momentum vs S&P 500 (composite score)",
            template="plotly_dark",
            paper_bgcolor=COLORS["background"],
            plot_bgcolor=COLORS["surface"],
            height=450,
            xaxis_title="Relative Momentum (%)",
            margin=dict(l=150, r=80, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Performance table
        display_cols = ["ETF", "Price", "1W_Ret", "1M_Ret", "3M_Ret", "Momentum_Score", "RSI", "Rank", "Signal"]
        format_dict = {
            "Price": "${:,.2f}",
            "1W_Ret": "{:+.2%}",
            "1M_Ret": "{:+.2%}",
            "3M_Ret": "{:+.2%}",
            "Momentum_Score": "{:+.4f}",
            "RSI": "{:.1f}",
            "Rank": "{:.0f}",
        }
        available_cols = [c for c in display_cols if c in momentum.columns]
        st.dataframe(
            momentum[available_cols].style.background_gradient(
                cmap="RdYlGn", subset=["Momentum_Score"], vmin=-0.05, vmax=0.05
            ).format(format_dict),
            use_container_width=True,
        )

        # ── Rotation Recommendations ───────────────────────
        st.subheader("Rotation Recommendations")
        recs = get_rotation_recommendations(momentum, cycle)

        for rec in recs:
            action = rec["Action"]
            conviction = rec["Conviction"]

            if "OVERWEIGHT" in action:
                color = COLORS["bull"]
            elif "UNDERWEIGHT" in action:
                color = COLORS["bear"]
            elif "CAUTION" in action:
                color = "#FF9800"
            else:
                color = "#888"

            conv_badge = {
                "HIGH": f"<span style='background:{color};color:white;padding:2px 8px;border-radius:3px;font-size:10px'>HIGH</span>",
                "MODERATE": f"<span style='background:#555;color:white;padding:2px 8px;border-radius:3px;font-size:10px'>MOD</span>",
                "LOW": "<span style='background:#333;color:#888;padding:2px 8px;border-radius:3px;font-size:10px'>LOW</span>",
            }.get(conviction, "")

            st.markdown(
                f"<div style='display:flex;align-items:center;padding:8px 0;border-bottom:1px solid #333'>"
                f"<span style='width:160px;font-weight:600;color:{text_color}'>{rec['Sector']}</span>"
                f"<span style='width:180px;color:{color};font-weight:600'>{action}</span>"
                f"<span style='width:80px'>{conv_badge}</span>"
                f"<span style='color:#aaa;font-size:13px'>{rec['Reason']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ═══════════════════════════════════════════════════════
    # TAB 2: MARKET BREADTH
    # ═══════════════════════════════════════════════════════
    with tab_breadth:
        with st.spinner("Scanning 110+ stocks for breadth indicators (this takes a minute)..."):
            breadth_result = compute_sector_breadth()

        summary = breadth_result.get("summary", {})
        breadth_df = breadth_result.get("breadth_df", pd.DataFrame())

        if not summary:
            st.warning("Insufficient data for breadth analysis")
            return

        # ── Breadth Health Score ───────────────────────────
        health = summary.get("health_score", 50)
        interpretation = summary.get("interpretation", "")

        if health >= 60:
            h_color = COLORS["bull"]
        elif health >= 40:
            h_color = COLORS["neutral"]
        else:
            h_color = COLORS["bear"]

        st.markdown(
            f"<div style='background:{surface};padding:25px;border-radius:12px;"
            f"border-left:5px solid {h_color};margin-bottom:20px'>"
            f"<div style='display:flex;align-items:center'>"
            f"<span style='font-size:56px;font-weight:700;color:{h_color};margin-right:20px'>{health:.0f}</span>"
            f"<div>"
            f"<h3 style='margin:0;color:{text_color}'>Market Breadth Health</h3>"
            f"<p style='margin:5px 0 0;color:#aaa'>{interpretation}</p>"
            f"</div></div></div>",
            unsafe_allow_html=True,
        )

        # ── Breadth Divergence Check ───────────────────────
        spy_data = _load_ticker("SPY")
        if not spy_data.empty:
            divergence = compute_breadth_divergence(
                spy_data["Close"], summary.get("pct_above_50ma", 50)
            )
            if "DIVERGENCE" in divergence:
                st.error(divergence)
            elif "CONFIRMED" in divergence:
                st.success(divergence)
            else:
                st.info(divergence)

        # ── Key Breadth Metrics ────────────────────────────
        st.subheader("Key Breadth Indicators")
        mc = st.columns(4)
        mc[0].metric("% Above 50-day MA", f"{summary.get('pct_above_50ma', 0):.1f}%")
        mc[1].metric("% Above 200-day MA", f"{summary.get('pct_above_200ma', 0):.1f}%")
        mc[2].metric("% Advancing Today", f"{summary.get('pct_advancing', 0):.1f}%")
        mc[3].metric("A/D Ratio", f"{summary.get('advance_decline_ratio', 0):.2f}")

        mc2 = st.columns(4)
        mc2[0].metric("% Golden Cross", f"{summary.get('pct_golden_cross', 0):.1f}%")
        mc2[1].metric("% Near 52W High", f"{summary.get('pct_near_52w_high', 0):.1f}%")
        mc2[2].metric("% Near 52W Low", f"{summary.get('pct_near_52w_low', 0):.1f}%")
        mc2[3].metric("Total Stocks", summary.get("total_stocks", 0))

        # ── Breadth Gauge Chart ────────────────────────────
        metrics_for_gauge = {
            "Above 50MA": summary.get("pct_above_50ma", 0),
            "Above 200MA": summary.get("pct_above_200ma", 0),
            "Advancing": summary.get("pct_advancing", 0),
            "Golden Cross": summary.get("pct_golden_cross", 0),
        }

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(metrics_for_gauge.keys()),
            y=list(metrics_for_gauge.values()),
            marker_color=[
                COLORS["bull"] if v > 60 else COLORS["neutral"] if v > 40 else COLORS["bear"]
                for v in metrics_for_gauge.values()
            ],
            text=[f"{v:.1f}%" for v in metrics_for_gauge.values()],
            textposition="outside",
        ))
        fig.add_hline(y=50, line_dash="dash", line_color="#888", annotation_text="50% threshold")
        fig.update_layout(
            title="Breadth Indicators (%)",
            template="plotly_dark",
            paper_bgcolor=COLORS["background"],
            plot_bgcolor=COLORS["surface"],
            height=350,
            yaxis_range=[0, 105],
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Sector Breadth Breakdown ───────────────────────
        sector_breadth = summary.get("sector_breadth")
        if sector_breadth is not None and not sector_breadth.empty:
            st.subheader("Sector Breadth Breakdown")
            st.dataframe(
                sector_breadth.style.background_gradient(
                    cmap="RdYlGn",
                    subset=["Pct_Above_50MA", "Pct_Above_200MA", "Pct_Advancing"],
                ).format({
                    "Pct_Above_50MA": "{:.1%}",
                    "Pct_Above_200MA": "{:.1%}",
                    "Pct_Advancing": "{:.1%}",
                    "Avg_1D_Ret": "{:+.2%}",
                    "Avg_1W_Ret": "{:+.2%}",
                    "Avg_1M_Ret": "{:+.2%}",
                }),
                use_container_width=True,
            )

        # ── Individual Stock Table ─────────────────────────
        if not breadth_df.empty:
            with st.expander("All Stocks Detail"):
                display = breadth_df.sort_values("Ret_1M", ascending=False)
                st.dataframe(
                    display.style.format({
                        "Price": "${:,.2f}",
                        "Pct_From_52W_High": "{:+.1f}%",
                        "Pct_From_52W_Low": "{:+.1f}%",
                        "Ret_1D": "{:+.2%}",
                        "Ret_1W": "{:+.2%}",
                        "Ret_1M": "{:+.2%}",
                    }),
                    use_container_width=True,
                )

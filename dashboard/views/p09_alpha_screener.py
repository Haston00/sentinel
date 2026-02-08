"""
SENTINEL — Page 9: Alpha Screener.
Scans entire stock universe, ranks by composite conviction score.
Finds the highest-conviction plays right now.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config.assets import SECTORS, BENCHMARKS
from config.settings import COLORS
from data.stocks import fetch_ohlcv
from features.signals import compute_composite_score, score_multiple


@st.cache_data(ttl=300, show_spinner=False)
def _load_ticker(ticker: str) -> pd.DataFrame:
    return fetch_ohlcv(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def _scan_universe(tickers: tuple) -> pd.DataFrame:
    data = {}
    for t in tickers:
        df = fetch_ohlcv(t)
        if not df.empty:
            data[t] = df
    return score_multiple(data)


def render():
    st.title("Alpha Screener")
    st.caption("Find the highest-conviction setups across your entire universe")
    surface = COLORS["surface"]
    text_color = COLORS["text"]

    # ── Scan Configuration ─────────────────────────────────
    scan_mode = st.radio(
        "Scan Universe",
        ["Sector ETFs (11)", "All Sector Holdings (110+)", "Custom"],
        horizontal=True,
    )

    if scan_mode == "Sector ETFs (11)":
        tickers = tuple(s["etf"] for s in SECTORS.values())
    elif scan_mode == "All Sector Holdings (110+)":
        all_t = set()
        for s in SECTORS.values():
            all_t.add(s["etf"])
            all_t.update(s["holdings"])
        tickers = tuple(sorted(all_t))
    else:
        custom = st.text_input("Enter tickers (comma separated)", "AAPL,MSFT,NVDA,GOOGL,AMZN,META,TSLA")
        tickers = tuple(t.strip().upper() for t in custom.split(",") if t.strip())

    if not tickers:
        st.info("Enter tickers to scan")
        return

    st.markdown(f"**Scanning {len(tickers)} assets...**")

    with st.spinner(f"Analyzing {len(tickers)} stocks — computing composite scores..."):
        results = _scan_universe(tickers)

    if results.empty:
        st.warning("No results. Check tickers and try again.")
        return

    # ── Top Picks ──────────────────────────────────────────
    st.subheader("Top Conviction Plays")
    top = results.head(5)
    cols = st.columns(min(5, len(top)))
    for i, (_, row) in enumerate(top.iterrows()):
        score = row["Score"]
        if score >= 70:
            color = COLORS["bull"]
        elif score >= 55:
            color = "#8BC34A"
        elif score >= 45:
            color = COLORS["neutral"]
        else:
            color = COLORS["bear"]

        with cols[i]:
            ret_1w = row.get("1W_Ret", 0) or 0
            ret_1m = row.get("1M_Ret", 0) or 0
            st.markdown(
                f"<div style='text-align:center;background:{surface};padding:15px;border-radius:8px;border-top:3px solid {color}'>"
                f"<p style='margin:0;font-size:20px;font-weight:700;color:{text_color}'>{row['Ticker']}</p>"
                f"<p style='margin:8px 0;font-size:36px;font-weight:700;color:{color}'>{score:.0f}</p>"
                f"<p style='margin:0;font-size:12px;color:{color}'>{row['Signal']}</p>"
                f"<p style='margin:5px 0 0;font-size:11px;color:#aaa'>1W: {ret_1w:+.1%} | 1M: {ret_1m:+.1%}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Worst Picks (Shorts/Avoid) ─────────────────────────
    st.subheader("Weakest — Avoid or Short")
    bottom = results.tail(5).iloc[::-1]
    cols2 = st.columns(min(5, len(bottom)))
    for i, (_, row) in enumerate(bottom.iterrows()):
        score = row["Score"]
        color = COLORS["bear"] if score < 35 else "#FF9800" if score < 45 else COLORS["neutral"]
        with cols2[i]:
            ret_1w = row.get("1W_Ret", 0) or 0
            ret_1m = row.get("1M_Ret", 0) or 0
            st.markdown(
                f"<div style='text-align:center;background:{surface};padding:15px;border-radius:8px;border-top:3px solid {color}'>"
                f"<p style='margin:0;font-size:20px;font-weight:700;color:{text_color}'>{row['Ticker']}</p>"
                f"<p style='margin:8px 0;font-size:36px;font-weight:700;color:{color}'>{score:.0f}</p>"
                f"<p style='margin:0;font-size:12px;color:{color}'>{row['Signal']}</p>"
                f"<p style='margin:5px 0 0;font-size:11px;color:#aaa'>1W: {ret_1w:+.1%} | 1M: {ret_1m:+.1%}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Score Distribution Chart ───────────────────────────
    st.markdown("---")
    st.subheader("Score Distribution")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=results["Ticker"],
        y=results["Score"],
        marker_color=[
            COLORS["bull"] if s >= 65 else "#8BC34A" if s >= 55
            else COLORS["neutral"] if s >= 45 else "#FF9800" if s >= 35
            else COLORS["bear"]
            for s in results["Score"]
        ],
        text=[f"{s:.0f}" for s in results["Score"]],
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="#888",
                  annotation_text="Neutral (50)")
    fig.update_layout(
        title="Conviction Score Ranking",
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        height=400,
        yaxis_range=[0, 105],
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Signal Component Heatmap ───────────────────────────
    st.subheader("Signal Component Heatmap")
    component_cols = ["RSI", "Trend", "Momentum", "MACD", "Volume", "Volatility"]
    available = [c for c in component_cols if c in results.columns]
    if available:
        heatmap_data = results.set_index("Ticker")[available]
        fig = go.Figure(go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns.tolist(),
            y=heatmap_data.index.tolist(),
            colorscale=[
                [0, COLORS["bear"]],
                [0.35, "#FF9800"],
                [0.5, COLORS["neutral"]],
                [0.65, "#8BC34A"],
                [1, COLORS["bull"]],
            ],
            zmin=0, zmax=100,
            text=[[f"{v:.0f}" for v in row] for row in heatmap_data.values],
            texttemplate="%{text}",
            textfont=dict(size=10),
        ))
        fig.update_layout(
            title="Component Scores by Asset",
            template="plotly_dark",
            paper_bgcolor=COLORS["background"],
            plot_bgcolor=COLORS["surface"],
            height=max(300, len(heatmap_data) * 30),
            margin=dict(l=80, r=30, t=50, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Full Results Table ─────────────────────────────────
    st.subheader("Full Scan Results")
    display_cols = ["Ticker", "Score", "Signal", "Price"] + [c for c in available if c in results.columns]
    if "1W_Ret" in results.columns:
        display_cols.append("1W_Ret")
    if "1M_Ret" in results.columns:
        display_cols.append("1M_Ret")

    format_dict = {"Score": "{:.0f}", "Price": "${:,.2f}"}
    for c in available:
        format_dict[c] = "{:.0f}"
    if "1W_Ret" in display_cols:
        format_dict["1W_Ret"] = "{:+.2%}"
    if "1M_Ret" in display_cols:
        format_dict["1M_Ret"] = "{:+.2%}"

    st.dataframe(
        results[display_cols].style.background_gradient(
            cmap="RdYlGn", subset=["Score"] + available, vmin=0, vmax=100
        ).format(format_dict),
        use_container_width=True,
    )

"""
SENTINEL — Page 8: Signal Intelligence Dashboard.
The genius dashboard. Composite conviction scores, key signals, and breadth analysis
that shows you what's happening before everyone else sees it.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.assets import BENCHMARKS, SECTORS
from config.settings import COLORS
from data.stocks import fetch_ohlcv
from features.signals import compute_composite_score


@st.cache_data(ttl=300, show_spinner=False)
def _load_ticker(ticker: str) -> pd.DataFrame:
    return fetch_ohlcv(ticker)


def _score_color(score: float) -> str:
    if score >= 70:
        return COLORS["bull"]
    if score >= 55:
        return "#8BC34A"
    if score >= 45:
        return COLORS["neutral"]
    if score >= 30:
        return "#FF9800"
    return COLORS["bear"]


def _score_gauge(score: float, label: str, title: str) -> go.Figure:
    """Create a dramatic gauge chart for a conviction score."""
    color = _score_color(score)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title=dict(text=f"<b>{title}</b>", font=dict(size=16, color=COLORS["text"])),
        number=dict(font=dict(size=36, color=color), suffix=""),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=COLORS["text"],
                      tick0=0, dtick=20),
            bar=dict(color=color, thickness=0.3),
            bgcolor=COLORS["surface"],
            bordercolor="#444",
            steps=[
                dict(range=[0, 20], color="rgba(255,23,68,0.15)"),
                dict(range=[20, 35], color="rgba(255,152,0,0.1)"),
                dict(range=[35, 45], color="rgba(255,214,0,0.08)"),
                dict(range=[45, 55], color="rgba(200,200,200,0.05)"),
                dict(range=[55, 65], color="rgba(255,214,0,0.08)"),
                dict(range=[65, 80], color="rgba(139,195,74,0.1)"),
                dict(range=[80, 100], color="rgba(0,200,83,0.15)"),
            ],
            threshold=dict(line=dict(color="white", width=2), thickness=0.75, value=score),
        ),
    ))

    fig.add_annotation(
        text=f"<b>{label}</b>", xref="paper", yref="paper",
        x=0.5, y=-0.15, showarrow=False,
        font=dict(size=14, color=color),
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        height=250,
        margin=dict(l=30, r=30, t=60, b=50),
    )
    return fig


def _component_radar(components: dict, title: str) -> go.Figure:
    """Spider/radar chart showing all signal components."""
    cats = list(components.keys())
    vals = list(components.values())
    # Close the polygon
    cats.append(cats[0])
    vals.append(vals[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals,
        theta=cats,
        fill="toself",
        fillcolor="rgba(41,98,255,0.15)",
        line=dict(color=COLORS["primary"], width=2),
        name="Current",
    ))

    # Add 50-line reference
    fig.add_trace(go.Scatterpolar(
        r=[50] * len(cats),
        theta=cats,
        line=dict(color="#555", width=1, dash="dash"),
        name="Neutral",
    ))

    fig.update_layout(
        polar=dict(
            bgcolor=COLORS["surface"],
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#333",
                            tickfont=dict(size=9, color="#888")),
            angularaxis=dict(gridcolor="#444", tickfont=dict(size=11, color=COLORS["text"])),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, font=dict(size=14, color=COLORS["text"])),
        height=350,
        margin=dict(l=60, r=60, t=50, b=30),
        showlegend=False,
    )
    return fig


def render():
    st.title("Signal Intelligence")
    st.caption("Composite conviction scoring — see what others miss")
    surface = COLORS["surface"]
    text_color = COLORS["text"]

    # ── Quick Score: Major Benchmarks ───────────────────────
    st.subheader("Market Pulse")
    with st.spinner("Computing conviction scores across major indices..."):
        bench_cols = st.columns(len(BENCHMARKS))
        bench_scores = {}
        for col, (name, ticker) in zip(bench_cols, BENCHMARKS.items()):
            df = _load_ticker(ticker)
            if not df.empty:
                result = compute_composite_score(df)
                score = result["score"]
                label = result["label"]
                bench_scores[name] = result
                color = _score_color(score)
                col.markdown(
                    f"<div style='text-align:center;background:{surface};padding:15px;border-radius:8px;border-left:4px solid {color}'>"
                    f"<p style='margin:0;color:#aaa;font-size:12px'>{name}</p>"
                    f"<p style='margin:5px 0;font-size:32px;font-weight:700;color:{color}'>{score:.0f}</p>"
                    f"<p style='margin:0;font-size:11px;color:{color}'>{label}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Deep Dive: Selected Asset ──────────────────────────
    st.markdown("---")
    st.subheader("Deep Signal Analysis")

    all_tickers = sorted(set(
        list(BENCHMARKS.values())
        + [s["etf"] for s in SECTORS.values()]
    ))
    ticker = st.selectbox("Select Asset", all_tickers, index=0, key="sig_ticker")

    with st.spinner(f"Analyzing {ticker}..."):
        df = _load_ticker(ticker)
        if df.empty:
            st.error(f"No data for {ticker}")
            return

        result = compute_composite_score(df)

    # Score gauge + radar side by side
    g1, g2 = st.columns([1, 1])
    with g1:
        fig = _score_gauge(result["score"], result["label"], f"{ticker} Conviction Score")
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        if result["components"]:
            fig = _component_radar(result["components"], f"{ticker} Signal Components")
            st.plotly_chart(fig, use_container_width=True)

    # Key signals
    if result["signals"]:
        st.subheader("Key Signals Detected")
        for sig in result["signals"]:
            st.markdown(f"**{sig}**")

    # Component breakdown table
    st.subheader("Component Breakdown")
    if result["components"]:
        comp_df = pd.DataFrame([result["components"]]).T
        comp_df.columns = ["Score"]
        comp_df["Rating"] = comp_df["Score"].apply(
            lambda x: "BULLISH" if x >= 65 else "BEARISH" if x <= 35 else "NEUTRAL"
        )
        st.dataframe(
            comp_df.style.background_gradient(cmap="RdYlGn", subset=["Score"], vmin=0, vmax=100),
            use_container_width=True,
        )

    # ── Sector Signal Heatmap ──────────────────────────────
    st.markdown("---")
    st.subheader("Sector Signal Heatmap")
    with st.spinner("Scoring all 11 sectors..."):
        sector_scores = []
        for sector_name, info in SECTORS.items():
            sdf = _load_ticker(info["etf"])
            if not sdf.empty:
                sr = compute_composite_score(sdf)
                sector_scores.append({
                    "Sector": sector_name,
                    "Score": sr["score"],
                    "Signal": sr["label"],
                    "RSI": sr["components"].get("RSI", 50),
                    "Trend": sr["components"].get("Trend", 50),
                    "Momentum": sr["components"].get("Momentum", 50),
                })

    if sector_scores:
        sdf = pd.DataFrame(sector_scores).set_index("Sector")

        # Heatmap
        fig = go.Figure(go.Heatmap(
            z=[sdf["Score"].values],
            x=sdf.index.tolist(),
            y=["Conviction"],
            colorscale=[
                [0, COLORS["bear"]],
                [0.35, "#FF9800"],
                [0.5, COLORS["neutral"]],
                [0.65, "#8BC34A"],
                [1, COLORS["bull"]],
            ],
            zmin=0, zmax=100,
            text=[[f"{v:.0f}" for v in sdf["Score"].values]],
            texttemplate="%{text}",
            textfont=dict(size=14, color="white"),
        ))
        fig.update_layout(
            title="Sector Conviction Scores (0-100)",
            template="plotly_dark",
            paper_bgcolor=COLORS["background"],
            plot_bgcolor=COLORS["surface"],
            height=150,
            margin=dict(l=80, r=30, t=50, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.dataframe(
            sdf.style.background_gradient(
                cmap="RdYlGn", subset=["Score", "RSI", "Trend", "Momentum"],
                vmin=0, vmax=100,
            ).format({"Score": "{:.0f}", "RSI": "{:.0f}", "Trend": "{:.0f}", "Momentum": "{:.0f}"}),
            use_container_width=True,
        )

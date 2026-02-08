"""
SENTINEL — Page 2: Sector Analysis.
GICS sector deep-dive with forecasts, top stocks, relative strength, news.
"""

import streamlit as st
import pandas as pd

from config.assets import SECTORS
from config.settings import COLORS
from data.stocks import fetch_ohlcv, fetch_fundamentals
from features.technical import compute_all_technical
from dashboard.components.charts import candlestick_chart, forecast_cone_chart
from dashboard.components.widgets import forecast_card, explanation_panel


@st.cache_data(ttl=300, show_spinner=False)
def _load_ticker(ticker: str) -> pd.DataFrame:
    return fetch_ohlcv(ticker)


def render():
    st.title("Sector Analysis")

    # Sector selector
    sector_name = st.selectbox("Select Sector", list(SECTORS.keys()))
    sector = SECTORS[sector_name]
    etf = sector["etf"]
    holdings = sector["holdings"]

    # ── Sector ETF Overview ───────────────────────────────────
    st.subheader(f"{sector_name} — {etf}")
    with st.spinner(f"Loading {etf} data..."):
        etf_data = _load_ticker(etf)

    if not etf_data.empty:
        c = etf_data["Close"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Price", f"${c.iloc[-1]:,.2f}")
        col2.metric("1-Day", f"{c.pct_change().iloc[-1]:+.2%}")
        col3.metric("1-Month", f"{c.pct_change(21).iloc[-1]:+.2%}" if len(c) > 21 else "N/A")
        col4.metric("YTD", f"{(c.iloc[-1] / c.iloc[0] - 1):+.2%}")

        # Chart
        timeframe = st.selectbox("Chart Timeframe", ["3M", "6M", "1Y", "2Y"], index=2, key="sector_tf")
        days_map = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504}
        trimmed = etf_data.tail(days_map.get(timeframe, 252))
        fig = candlestick_chart(trimmed, title=f"{sector_name} ({etf})")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Could not load {etf} data")

    # ── Forecast ──────────────────────────────────────────────
    st.subheader("Sector Forecast")
    if st.button("Generate Forecast", key="sector_fc"):
        with st.spinner("Running forecast engine..."):
            try:
                from forecasting.engine import ForecastEngine
                engine = ForecastEngine()
                result = engine.forecast_sector(sector_name)

                if "error" not in result:
                    for horizon in ["1W", "1M", "3M"]:
                        forecast_card(result["forecasts"], horizon)

                    if "explanation" in result:
                        st.subheader("Explanation")
                        explanation_panel(result["explanation"])
                else:
                    st.error(result["error"])
            except Exception as e:
                st.error(f"Forecast failed: {e}")

    # ── Top Holdings ──────────────────────────────────────────
    st.subheader("Top Holdings")
    with st.spinner(f"Loading {len(holdings)} holdings..."):
        holdings_data = []
        for ticker in holdings:
            try:
                df = _load_ticker(ticker)
                if not df.empty:
                    c = df["Close"]
                    holdings_data.append({
                        "Ticker": ticker,
                        "Price": c.iloc[-1],
                        "1D": c.pct_change().iloc[-1],
                        "1W": c.pct_change(5).iloc[-1] if len(c) > 5 else None,
                        "1M": c.pct_change(21).iloc[-1] if len(c) > 21 else None,
                    })
            except Exception:
                pass

    if holdings_data:
        df = pd.DataFrame(holdings_data).set_index("Ticker")
        st.dataframe(
            df.style.format({
                "Price": "${:,.2f}",
                "1D": "{:+.2%}",
                "1W": "{:+.2%}",
                "1M": "{:+.2%}",
            }).background_gradient(cmap="RdYlGn", subset=["1D", "1W", "1M"]),
            use_container_width=True,
        )
    else:
        st.info("Loading holdings data...")

    # ── Relative Strength ─────────────────────────────────────
    st.subheader("Relative Strength vs S&P 500")
    spy_data = _load_ticker("SPY")
    if not etf_data.empty and not spy_data.empty:
        etf_returns = etf_data["Close"].pct_change().dropna()
        spy_returns = spy_data["Close"].pct_change().dropna()
        common_idx = etf_returns.index.intersection(spy_returns.index)

        relative = ((1 + etf_returns.loc[common_idx]).cumprod() /
                     (1 + spy_returns.loc[common_idx]).cumprod())

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=relative.tail(252).index, y=relative.tail(252).values,
            name="Relative Strength", line=dict(color=COLORS["primary"], width=2),
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color="#555")
        fig.update_layout(
            title=f"{sector_name} vs S&P 500 — Relative Strength",
            template="plotly_dark",
            paper_bgcolor=COLORS["background"],
            plot_bgcolor=COLORS["surface"],
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

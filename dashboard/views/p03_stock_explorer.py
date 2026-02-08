"""
SENTINEL — Page 3: Stock Explorer.
Individual stock deep-dive: forecast, technicals, factor exposure, news.
"""

import streamlit as st
import pandas as pd

from config.assets import SECTORS, get_sector_for_ticker
from config.settings import COLORS
from data.stocks import fetch_ohlcv, fetch_fundamentals
from features.technical import compute_all_technical
from dashboard.components.charts import candlestick_chart, forecast_cone_chart
from dashboard.components.widgets import forecast_card, explanation_panel, metric_row


@st.cache_data(ttl=300, show_spinner=False)
def _load_ticker(ticker: str) -> pd.DataFrame:
    return fetch_ohlcv(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def _load_fundamentals(ticker: str) -> dict:
    return fetch_fundamentals(ticker)


def render():
    st.title("Stock Explorer")

    # Ticker input
    all_tickers = []
    for s in SECTORS.values():
        all_tickers.extend(s["holdings"])
    all_tickers = sorted(set(all_tickers))

    col1, col2 = st.columns([1, 3])
    with col1:
        ticker = st.text_input("Enter Ticker", "AAPL").upper()
    with col2:
        quick_pick = st.selectbox("Or quick pick", [""] + all_tickers, key="quick_pick")
        if quick_pick:
            ticker = quick_pick

    if not ticker:
        st.info("Enter a ticker symbol to begin.")
        return

    # ── Fetch Data ────────────────────────────────────────────
    with st.spinner(f"Loading {ticker} data..."):
        ohlcv = _load_ticker(ticker)

    if ohlcv.empty:
        st.error(f"No data found for {ticker}. Check the ticker symbol.")
        return

    c = ohlcv["Close"]
    sector = get_sector_for_ticker(ticker)

    # ── Header Metrics ────────────────────────────────────────
    st.subheader(f"{ticker}" + (f" — {sector}" if sector else ""))
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Price", f"${c.iloc[-1]:,.2f}")
    m2.metric("1-Day", f"{c.pct_change().iloc[-1]:+.2%}")
    m3.metric("1-Week", f"{c.pct_change(5).iloc[-1]:+.2%}" if len(c) > 5 else "N/A")
    m4.metric("1-Month", f"{c.pct_change(21).iloc[-1]:+.2%}" if len(c) > 21 else "N/A")
    m5.metric("YTD", f"{(c.iloc[-1] / c.iloc[0] - 1):+.2%}")

    # ── Chart ─────────────────────────────────────────────────
    timeframe = st.selectbox("Timeframe", ["1M", "3M", "6M", "1Y", "2Y", "5Y"], index=3)
    days_map = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260}
    trimmed = ohlcv.tail(days_map.get(timeframe, 252))
    fig = candlestick_chart(trimmed, title=f"{ticker} — {timeframe}")
    st.plotly_chart(fig, use_container_width=True)

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Technicals", "Fundamentals", "Factor Exposure"])

    with tab1:
        if st.button("Generate Forecast", key="stock_fc"):
            with st.spinner("Running models..."):
                try:
                    from forecasting.engine import ForecastEngine
                    engine = ForecastEngine()
                    result = engine.forecast_equity(ticker)

                    if "error" not in result:
                        for horizon in ["1W", "1M", "3M"]:
                            forecast_card(result["forecasts"], horizon)

                        # Forecast cone
                        fig = forecast_cone_chart(c.tail(126), result["forecasts"], "1M", f"{ticker} Forecast")
                        st.plotly_chart(fig, use_container_width=True)

                        if "explanation" in result:
                            explanation_panel(result["explanation"])
                    else:
                        st.error(result["error"])
                except Exception as e:
                    st.error(f"Forecast failed: {e}")

    with tab2:
        st.subheader("Technical Indicators")
        with st.spinner("Computing technical indicators..."):
            tech = compute_all_technical(ohlcv)
        if not tech.empty:
            latest = tech.iloc[-1]

            tcol1, tcol2, tcol3, tcol4 = st.columns(4)
            rsi = latest.get("RSI_14")
            if rsi is not None:
                rsi_color = COLORS["bear"] if rsi > 70 else COLORS["bull"] if rsi < 30 else COLORS["text"]
                tcol1.markdown(f"**RSI (14):** <span style='color:{rsi_color}'>{rsi:.1f}</span>", unsafe_allow_html=True)

            macd = latest.get("MACD_Hist")
            if macd is not None:
                macd_color = COLORS["bull"] if macd > 0 else COLORS["bear"]
                tcol2.markdown(f"**MACD Hist:** <span style='color:{macd_color}'>{macd:.4f}</span>", unsafe_allow_html=True)

            adx = latest.get("ADX")
            if adx is not None:
                tcol3.markdown(f"**ADX:** {adx:.1f}")

            bb_pct = latest.get("BB_Pct")
            if bb_pct is not None:
                tcol4.markdown(f"**BB %B:** {bb_pct:.2f}")

            # Technical summary table
            tech_cols = [col for col in tech.columns if col not in ["Open", "High", "Low", "Close", "Volume"]]
            summary = tech[tech_cols].tail(1).T
            summary.columns = ["Current"]
            st.dataframe(summary, use_container_width=True)

    with tab3:
        st.subheader("Fundamentals")
        with st.spinner(f"Loading {ticker} fundamentals..."):
            fundamentals = _load_fundamentals(ticker)
        if fundamentals:
            fund_df = pd.DataFrame([fundamentals]).T
            fund_df.columns = ["Value"]
            st.dataframe(fund_df, use_container_width=True)

    with tab4:
        st.subheader("Factor Exposure")
        if st.button("Calculate Factor Betas", key="factor_btn"):
            with st.spinner("Estimating factor loadings..."):
                try:
                    from models.factors import FactorModel
                    fm = FactorModel()
                    betas = fm.estimate_betas(ticker)
                    if betas:
                        beta_df = pd.DataFrame([betas]).T
                        beta_df.columns = ["Loading"]
                        st.dataframe(beta_df.style.format("{:.4f}"), use_container_width=True)

                        import plotly.graph_objects as go
                        factor_betas = {k: v for k, v in betas.items() if k not in ("Alpha", "R_Squared")}
                        fig = go.Figure(go.Bar(
                            x=list(factor_betas.keys()),
                            y=list(factor_betas.values()),
                            marker_color=[COLORS["bull"] if v > 0 else COLORS["bear"] for v in factor_betas.values()],
                        ))
                        fig.update_layout(
                            title=f"{ticker} Factor Loadings",
                            template="plotly_dark",
                            paper_bgcolor=COLORS["background"],
                            plot_bgcolor=COLORS["surface"],
                            height=350,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Factor analysis failed: {e}")

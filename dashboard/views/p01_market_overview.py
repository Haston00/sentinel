"""
SENTINEL — Page 1: Market Overview.
Broad market dashboard: S&P 500, sector heatmap, regime, top movers, macro.
"""

import streamlit as st
import pandas as pd

from config.assets import BENCHMARKS, SECTORS
from config.settings import COLORS
from data.stocks import fetch_ohlcv, get_close_prices
from dashboard.components.charts import (
    candlestick_chart,
    correlation_matrix_chart,
)
from dashboard.components.widgets import metric_row, regime_indicator


@st.cache_data(ttl=300, show_spinner=False)
def _load_ticker(ticker: str) -> pd.DataFrame:
    """Cache-wrapped fetch for Streamlit (5-min TTL)."""
    return fetch_ohlcv(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def _load_closes(tickers: tuple) -> pd.DataFrame:
    """Cache-wrapped close prices (tuple for hashability)."""
    return get_close_prices(list(tickers))


def render():
    st.title("Market Overview")

    # ── Key Benchmarks Row ────────────────────────────────────
    st.subheader("Benchmarks")
    with st.spinner("Loading benchmark prices..."):
        cols = st.columns(len(BENCHMARKS))
        loaded_any = False
        for col, (name, ticker) in zip(cols, BENCHMARKS.items()):
            try:
                df = _load_ticker(ticker)
                if not df.empty:
                    price = df["Close"].iloc[-1]
                    change = df["Close"].pct_change().iloc[-1]
                    col.metric(name, f"${price:,.2f}", f"{change:+.2%}")
                    loaded_any = True
            except Exception as e:
                col.caption(f"{name}: error")

        if not loaded_any:
            st.warning("Could not load benchmark data. Check your internet connection and try refreshing.")
            return

    # ── S&P 500 Chart ─────────────────────────────────────────
    st.subheader("S&P 500")
    timeframe = st.selectbox("Timeframe", ["3M", "6M", "1Y", "2Y", "5Y"], index=2)
    days_map = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260}

    spy_data = _load_ticker("SPY")
    if not spy_data.empty:
        trimmed = spy_data.tail(days_map.get(timeframe, 252))
        fig = candlestick_chart(trimmed, title=f"S&P 500 (SPY) — {timeframe}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("SPY data not available")

    # ── Regime Indicator ──────────────────────────────────────
    st.subheader("Market Regime")
    try:
        from models.regime import RegimeDetector
        detector = RegimeDetector()
        try:
            detector.load("equity_regime")
        except Exception:
            if not spy_data.empty:
                with st.spinner("Training regime model (one-time)..."):
                    detector.fit(spy_data["Close"])

        if detector.model is not None and not spy_data.empty:
            regime_info = detector.current_regime(spy_data["Close"])
            regime_indicator(regime_info)
        else:
            st.info("Regime model not yet trained. Run forecast engine to initialize.")
    except Exception as e:
        st.warning(f"Regime detection unavailable: {e}")

    # ── Sector Performance ────────────────────────────────────
    st.subheader("Sector Performance")
    with st.spinner("Loading sector data (11 sectors)..."):
        sector_data = []
        for sector_name, info in SECTORS.items():
            try:
                df = _load_ticker(info["etf"])
                if not df.empty:
                    c = df["Close"]
                    sector_data.append({
                        "Sector": sector_name,
                        "ETF": info["etf"],
                        "Price": c.iloc[-1],
                        "1D": c.pct_change().iloc[-1],
                        "1W": c.pct_change(5).iloc[-1] if len(c) > 5 else None,
                        "1M": c.pct_change(21).iloc[-1] if len(c) > 21 else None,
                        "3M": c.pct_change(63).iloc[-1] if len(c) > 63 else None,
                    })
            except Exception:
                pass

    if sector_data:
        sector_df = pd.DataFrame(sector_data).set_index("Sector")
        st.dataframe(
            sector_df.style.format({
                "Price": "${:,.2f}",
                "1D": "{:+.2%}",
                "1W": "{:+.2%}",
                "1M": "{:+.2%}",
                "3M": "{:+.2%}",
            }).background_gradient(
                cmap="RdYlGn", subset=["1D", "1W", "1M", "3M"]
            ),
            use_container_width=True,
        )
    else:
        st.info("Sector data loading... refresh in a moment.")

    # ── Macro Dashboard ───────────────────────────────────────
    st.subheader("Macro Indicators")
    try:
        from data.macro import get_latest_values
        with st.spinner("Loading FRED macro data..."):
            latest = get_latest_values()
        if latest:
            macro_cols = st.columns(4)
            items = list(latest.items())
            for i, (key, val) in enumerate(items[:8]):
                macro_cols[i % 4].metric(key, f"{val:,.2f}")
        else:
            st.info("Set FRED_API_KEY to enable macro data (or data is loading)")
    except Exception:
        st.info("Set FRED_API_KEY environment variable for macro indicators")

    # ── Cross-Asset Correlation ───────────────────────────────
    st.subheader("Cross-Asset Correlation")
    with st.spinner("Computing cross-asset correlations..."):
        corr_tickers = ("SPY", "QQQ", "IWM", "TLT", "GLD", "USO", "UUP")
        closes = _load_closes(corr_tickers)
        if not closes.empty:
            returns = closes.pct_change().dropna()
            fig = correlation_matrix_chart(returns, "60-Day Rolling Correlation")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Correlation data loading...")

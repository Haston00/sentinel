"""
SENTINEL — Page 4: Crypto Market.
Crypto overview, individual coin forecasts, on-chain metrics, news.
"""

import streamlit as st
import pandas as pd

from config.assets import CRYPTO_ALL, CRYPTO_MAJOR
from config.settings import COLORS
from data.crypto import fetch_crypto_ohlcv, fetch_crypto_market_data, get_current_prices
from features.onchain import compute_crypto_features
from dashboard.components.charts import candlestick_chart, forecast_cone_chart
from dashboard.components.widgets import forecast_card


@st.cache_data(ttl=180, show_spinner=False)
def _load_crypto_prices():
    return get_current_prices()


@st.cache_data(ttl=180, show_spinner=False)
def _load_crypto_ohlcv(coin_id: str, days: int):
    return fetch_crypto_ohlcv(coin_id, days=days)


def render():
    st.title("Crypto Market")

    # ── Current Prices Overview ───────────────────────────────
    st.subheader("Market Overview")
    with st.spinner("Fetching crypto prices from CoinGecko..."):
        try:
            prices = _load_crypto_prices()
            if prices:
                cols = st.columns(min(len(CRYPTO_MAJOR), 4))
                for i, (coin_id, info) in enumerate(CRYPTO_MAJOR.items()):
                    with cols[i % len(cols)]:
                        coin_data = prices.get(coin_id, {})
                        price = coin_data.get("usd", 0)
                        change_24h = coin_data.get("usd_24h_change", 0)

                        st.metric(
                            f"{info['symbol']}",
                            f"${price:,.2f}",
                            f"{change_24h:+.1f}%",
                        )
            else:
                st.info("CoinGecko API rate limited — try again in a minute")
        except Exception as e:
            st.warning(f"Could not fetch current prices: {e}")

    # ── Coin Selector ─────────────────────────────────────────
    st.markdown("---")
    coin_options = {f"{v['symbol']} — {v['name']}": k for k, v in CRYPTO_ALL.items()}
    selected_label = st.selectbox("Select Cryptocurrency", list(coin_options.keys()))
    coin_id = coin_options[selected_label]
    coin_info = CRYPTO_ALL[coin_id]

    # ── Price Chart ───────────────────────────────────────────
    st.subheader(f"{coin_info['symbol']} — {coin_info['name']}")
    timeframe = st.selectbox("Timeframe", ["1M", "3M", "6M", "1Y", "Max"], index=2, key="crypto_tf")
    days_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "Max": 1825}

    with st.spinner(f"Loading {coin_info['symbol']} price data..."):
        ohlcv = _load_crypto_ohlcv(coin_id, days=days_map.get(timeframe, 180))

    if not ohlcv.empty:
        fig = candlestick_chart(ohlcv, title=f"{coin_info['symbol']} ({timeframe})", volume=False)
        st.plotly_chart(fig, use_container_width=True)

        # Key metrics
        c = ohlcv["Close"]
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Price", f"${c.iloc[-1]:,.2f}")
        mc2.metric("7D Change", f"{c.pct_change(7).iloc[-1]:+.2%}" if len(c) > 7 else "N/A")
        mc3.metric("30D Change", f"{c.pct_change(30).iloc[-1]:+.2%}" if len(c) > 30 else "N/A")
        ath = c.max()
        dd = (c.iloc[-1] / ath - 1) * 100
        mc4.metric("From ATH", f"{dd:+.1f}%")
    else:
        st.info(f"No price data available for {coin_info['symbol']}. CoinGecko may be rate limiting.")

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["Forecast", "On-Chain Metrics", "Market Data"])

    with tab1:
        if st.button("Generate Crypto Forecast", key="crypto_fc"):
            with st.spinner("Running crypto models..."):
                try:
                    from forecasting.engine import ForecastEngine
                    engine = ForecastEngine()
                    result = engine.forecast_crypto(coin_id)

                    if "error" not in result:
                        for horizon in ["1W", "1M", "3M"]:
                            if horizon in result.get("forecasts", {}):
                                forecast_card(result["forecasts"], horizon)
                    else:
                        st.error(result["error"])
                except Exception as e:
                    st.error(f"Forecast failed: {e}")

    with tab2:
        st.subheader("On-Chain Metrics")
        with st.spinner("Computing on-chain features..."):
            features = compute_crypto_features(coin_id)
            if not features.empty:
                latest = features.iloc[-1]

                ocol1, ocol2, ocol3 = st.columns(3)
                nvt = latest.get("NVT_Ratio")
                if nvt is not None and pd.notna(nvt):
                    ocol1.metric("NVT Ratio", f"{nvt:.1f}")
                mvrv = latest.get("MVRV_Proxy")
                if mvrv is not None and pd.notna(mvrv):
                    ocol2.metric("MVRV Proxy", f"{mvrv:.2f}")
                vol_ratio = latest.get("Vol_Ratio_7_30")
                if vol_ratio is not None and pd.notna(vol_ratio):
                    ocol3.metric("Vol Ratio (7/30d)", f"{vol_ratio:.2f}")

                # Feature table
                display_cols = [c for c in features.columns if not c.startswith("MarketCap")]
                if display_cols:
                    st.dataframe(
                        features[display_cols].tail(30).style.format("{:.4f}"),
                        use_container_width=True,
                    )
            else:
                st.info("Insufficient data for on-chain metrics")

    with tab3:
        st.subheader("Market Data")
        with st.spinner("Loading market data..."):
            market = fetch_crypto_market_data(coin_id)
        if not market.empty:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.6, 0.4], vertical_spacing=0.05)

            fig.add_trace(go.Scatter(
                x=market.index, y=market["Price"],
                name="Price", line=dict(color=COLORS["primary"]),
            ), row=1, col=1)

            if "Volume" in market.columns:
                fig.add_trace(go.Bar(
                    x=market.index, y=market["Volume"],
                    name="Volume", marker_color=COLORS["secondary"], opacity=0.5,
                ), row=2, col=1)

            fig.update_layout(
                title=f"{coin_info['symbol']} Market Data",
                template="plotly_dark",
                paper_bgcolor=COLORS["background"],
                plot_bgcolor=COLORS["surface"],
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Market data not available")

    # ── All Coins Table ───────────────────────────────────────
    st.markdown("---")
    st.subheader("Full Crypto Universe")
    with st.spinner("Loading all coin prices..."):
        try:
            all_prices = _load_crypto_prices()
            if all_prices:
                rows = []
                for cid, cinfo in CRYPTO_ALL.items():
                    cd = all_prices.get(cid, {})
                    rows.append({
                        "Symbol": cinfo["symbol"],
                        "Name": cinfo["name"],
                        "Price": cd.get("usd", 0),
                        "24h Change": cd.get("usd_24h_change", 0),
                        "Market Cap": cd.get("usd_market_cap", 0),
                        "24h Volume": cd.get("usd_24h_vol", 0),
                    })
                table_df = pd.DataFrame(rows)
                st.dataframe(
                    table_df.style.format({
                        "Price": "${:,.2f}",
                        "24h Change": "{:+.1f}%",
                        "Market Cap": "${:,.0f}",
                        "24h Volume": "${:,.0f}",
                    }).background_gradient(cmap="RdYlGn", subset=["24h Change"]),
                    use_container_width=True,
                )
            else:
                st.info("CoinGecko rate limited — try again in a minute")
        except Exception:
            pass

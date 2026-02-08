"""
SENTINEL — Page 6: Regime Monitor.
Current regime probability, historical timeline, transition alerts.
"""

import streamlit as st
import pandas as pd

from config.settings import COLORS
from data.stocks import fetch_ohlcv
from dashboard.components.charts import regime_timeline_chart
from dashboard.components.widgets import regime_indicator


@st.cache_data(ttl=300, show_spinner=False)
def _load_ticker(ticker: str) -> pd.DataFrame:
    return fetch_ohlcv(ticker)


def render():
    st.title("Regime Monitor")

    # ── Equity Regime ─────────────────────────────────────────
    st.subheader("Equity Market Regime")

    with st.spinner("Loading market data..."):
        spy_data = _load_ticker("SPY")

    if spy_data.empty:
        st.error("Cannot load SPY data for regime detection.")
        return

    # Train or load model
    try:
        from models.regime import RegimeDetector

        detector = RegimeDetector()
        try:
            detector.load("equity_regime")
            st.success("Loaded trained regime model")
        except Exception:
            with st.spinner("Training regime model on S&P 500 history (one-time)..."):
                vix_data = _load_ticker("^VIX")
                vix_close = vix_data["Close"] if not vix_data.empty else None
                detector.fit(spy_data["Close"], vix=vix_close)
                detector.save("equity_regime")
            st.success("Regime model trained and saved")

        # Current regime
        vix_data = _load_ticker("^VIX")
        vix_close = vix_data["Close"] if not vix_data.empty else None
        current = detector.current_regime(spy_data["Close"], vix=vix_close)
        regime_indicator(current)

        # Historical regime timeline
        st.subheader("Regime History")
        with st.spinner("Computing regime history..."):
            regime_history = detector.predict(spy_data["Close"], vix=vix_close)

        if not regime_history.empty:
            fig = regime_timeline_chart(regime_history, "S&P 500 Regime History")
            st.plotly_chart(fig, use_container_width=True)

            # Regime statistics
            st.subheader("Regime Statistics")
            stats = []
            for label in regime_history["Regime_Label"].unique():
                mask = regime_history["Regime_Label"] == label
                subset = regime_history[mask]
                spy_returns = spy_data["Close"].pct_change().reindex(subset.index)

                stats.append({
                    "Regime": label,
                    "Days": len(subset),
                    "% of History": f"{len(subset) / len(regime_history):.1%}",
                    "Avg Daily Return": f"{spy_returns.mean():.4%}",
                    "Annualized Return": f"{spy_returns.mean() * 252:.1%}",
                    "Daily Vol": f"{spy_returns.std():.4%}",
                    "Annualized Vol": f"{spy_returns.std() * (252**0.5):.1%}",
                })

            stats_df = pd.DataFrame(stats).set_index("Regime")
            st.dataframe(stats_df, use_container_width=True)

            # Regime probability over time
            st.subheader("Regime Probability Over Time")
            import plotly.graph_objects as go
            prob_cols = [c for c in regime_history.columns if c.startswith("Prob_")]
            fig = go.Figure()
            regime_colors = {"Prob_Bull": COLORS["bull"], "Prob_Bear": COLORS["bear"], "Prob_Transition": COLORS["transition"]}
            for col in prob_cols:
                fig.add_trace(go.Scatter(
                    x=regime_history.tail(504).index,
                    y=regime_history[col].tail(504),
                    name=col.replace("Prob_", ""),
                    line=dict(width=1.5, color=regime_colors.get(col, "#888")),
                    stackgroup="one",
                ))
            fig.update_layout(
                title="Regime Probabilities (Last 2 Years)",
                template="plotly_dark",
                paper_bgcolor=COLORS["background"],
                plot_bgcolor=COLORS["surface"],
                height=400,
                yaxis_range=[0, 1],
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Regime detection error: {e}")

    # ── Crypto Regime ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("Crypto Market Regime")

    try:
        from data.crypto import fetch_crypto_ohlcv
        from models.regime import RegimeDetector

        with st.spinner("Loading Bitcoin data for crypto regime..."):
            btc = fetch_crypto_ohlcv("bitcoin")

        if not btc.empty:
            crypto_detector = RegimeDetector()
            try:
                crypto_detector.load("crypto_regime")
            except Exception:
                with st.spinner("Training crypto regime model..."):
                    crypto_detector.fit(btc["Close"])
                    crypto_detector.save("crypto_regime")

            crypto_regime = crypto_detector.current_regime(btc["Close"])
            regime_indicator(crypto_regime)

            crypto_history = crypto_detector.predict(btc["Close"])
            if not crypto_history.empty:
                fig = regime_timeline_chart(crypto_history, "Bitcoin Regime History")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No Bitcoin data available for crypto regime detection")
    except Exception as e:
        st.warning(f"Crypto regime unavailable: {e}")

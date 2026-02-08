"""
SENTINEL — Page 7: Backtesting.
Model performance dashboard, accuracy by regime, calibration plots.
"""

import streamlit as st
import pandas as pd
import numpy as np

from config.settings import COLORS
from data.stocks import fetch_ohlcv
from dashboard.components.charts import performance_chart
from dashboard.components.widgets import metric_row


def render():
    st.title("Backtesting")
    st.markdown("Walk-forward out-of-sample backtesting — no lookahead bias.")

    # ── Configuration ─────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ticker = st.text_input("Ticker", "SPY", key="bt_ticker").upper()
    with col2:
        horizon = st.selectbox("Forecast Horizon", [5, 21, 63], index=1, key="bt_horizon")
    with col3:
        step_size = st.selectbox("Step Size (days)", [5, 10, 21], index=1, key="bt_step")
    with col4:
        min_train = st.selectbox("Min Training Window", [252, 504, 756], index=0, key="bt_train")

    if not st.button("Run Backtest", key="run_bt"):
        st.info("Configure parameters and click 'Run Backtest' to begin.")
        st.markdown("""
        **How it works:**
        1. Start with the minimum training window
        2. Train model on all data up to point T
        3. Forecast forward by the horizon period
        4. Record predicted vs actual direction
        5. Move forward by step size and repeat
        6. Report out-of-sample accuracy metrics

        **No lookahead bias:** The model never sees future data during training.
        """)
        return

    # ── Run Backtest ──────────────────────────────────────────
    with st.spinner(f"Running walk-forward backtest on {ticker}..."):
        try:
            ohlcv = fetch_ohlcv(ticker)
            if ohlcv.empty:
                st.error(f"No data for {ticker}")
                return

            from forecasting.backtester import WalkForwardBacktester
            from models.timeseries import ARIMAForecaster

            prices = ohlcv["Close"]

            def arima_model_factory(train_prices, train_features):
                """Simple ARIMA forecast for backtesting."""
                model = ARIMAForecaster(order=(2, 1, 2))
                model.fit(train_prices)
                fc = model.forecast(steps=horizon)
                if fc.empty:
                    return {"direction": 0, "confidence": 0.5}

                last_price = train_prices.iloc[-1]
                fc_price = fc["Forecast"].iloc[-1]
                expected_return = (fc_price / last_price) - 1

                direction = 1 if expected_return > 0.005 else (-1 if expected_return < -0.005 else 0)
                return {"direction": direction, "confidence": 0.6}

            backtester = WalkForwardBacktester(
                min_train_window=min_train,
                step_size=step_size,
                forecast_horizon=horizon,
            )

            results = backtester.run(prices, arima_model_factory)

            if results.empty:
                st.warning("Backtest produced no results. Try different parameters.")
                return

            # ── Results ───────────────────────────────────────
            st.success(f"Backtest complete: {len(results)} forecast periods")

            metrics = backtester.compute_metrics(results)

            # Key metrics
            st.subheader("Performance Metrics")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Hit Rate", f"{metrics.get('hit_rate_pct', 'N/A')}")
            m2.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
            m3.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
            m4.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
            m5.metric("N Forecasts", metrics.get("n_forecasts", 0))

            # Secondary metrics
            st.subheader("Detailed Metrics")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Long Accuracy", f"{metrics.get('long_accuracy', 0):.1%}")
            s2.metric("Short Accuracy", f"{metrics.get('short_accuracy', 0):.1%}")
            s3.metric("High Conf Accuracy", f"{metrics.get('high_conf_accuracy', 0):.1%}")
            s4.metric("Low Conf Accuracy", f"{metrics.get('low_conf_accuracy', 0):.1%}")

            # Equity curve
            st.subheader("Strategy vs Buy & Hold")
            fig = performance_chart(results, f"{ticker} ARIMA Strategy Backtest")
            st.plotly_chart(fig, use_container_width=True)

            # Regime breakdown
            st.subheader("Performance by Regime")
            try:
                from models.regime import RegimeDetector
                detector = RegimeDetector()
                try:
                    detector.load("equity_regime")
                except Exception:
                    detector.fit(prices)

                regime_data = detector.predict(prices)
                if not regime_data.empty:
                    regime_metrics = backtester.compute_regime_metrics(
                        results, regime_data["Regime_Label"]
                    )
                    if not regime_metrics.empty:
                        st.dataframe(
                            regime_metrics.style.format({
                                "Hit_Rate": "{:.1%}",
                                "Avg_Return": "{:.4%}",
                                "Sharpe": "{:.2f}",
                            }),
                            use_container_width=True,
                        )
            except Exception:
                st.info("Train regime model from the Regime Monitor page for regime-based analysis")

            # Hit rate over time
            st.subheader("Rolling Hit Rate")
            rolling_accuracy = results["correct"].rolling(20).mean()
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rolling_accuracy.index, y=rolling_accuracy.values,
                name="Rolling 20-period Hit Rate",
                line=dict(color=COLORS["primary"], width=2),
            ))
            fig.add_hline(y=0.5, line_dash="dash", line_color="#888",
                          annotation_text="Random (50%)")
            fig.update_layout(
                title="Rolling Hit Rate (20-period window)",
                template="plotly_dark",
                paper_bgcolor=COLORS["background"],
                plot_bgcolor=COLORS["surface"],
                height=350,
                yaxis_range=[0, 1],
            )
            st.plotly_chart(fig, use_container_width=True)

            # Raw results table
            with st.expander("Raw Results"):
                st.dataframe(results, use_container_width=True)

        except Exception as e:
            st.error(f"Backtest failed: {e}")
            import traceback
            st.code(traceback.format_exc())

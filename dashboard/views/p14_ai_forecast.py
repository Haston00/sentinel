"""
SENTINEL — Page 14: AI-Powered Forecast Dashboard.
Real machine learning predictions, not rules of thumb.
Shows you what the AI sees, how confident it is, and its track record.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from config.settings import COLORS
from config.assets import BENCHMARKS, SECTORS


def render():
    st.title("AI Forecast Engine")
    st.caption("XGBoost + Random Forest trained on your data — real ML, real accuracy numbers")

    surface = COLORS["surface"]
    text_color = COLORS["text"]
    bull = COLORS["bull"]
    bear = COLORS["bear"]
    neutral = COLORS["neutral"]
    primary = COLORS["primary"]
    bg = COLORS["background"]

    # ═══════════════════════════════════════════════════════════
    # REGIME DETECTION (HMM)
    # ═══════════════════════════════════════════════════════════
    st.subheader("Market Regime (Hidden Markov Model)")
    st.caption("The HMM watches 5 market signals and mathematically detects Bull, Bear, or Transition")

    with st.spinner("Training HMM on 5 years of multi-asset data..."):
        from models.regime import detect_current_regime
        regime = detect_current_regime()

    regime_name = regime.get("regime", "UNKNOWN")
    regime_conf = regime.get("confidence", 0)
    regime_color = regime.get("color", "#888")
    regime_desc = regime.get("description", "")
    regime_explanation = regime.get("explanation", "")

    st.markdown(
        f"<div style='background:{surface};padding:25px;border-radius:12px;"
        f"border-left:6px solid {regime_color};margin-bottom:20px'>"
        f"<div style='display:flex;align-items:center;gap:15px'>"
        f"<span style='font-size:48px;font-weight:700;color:{regime_color}'>{regime_name}</span>"
        f"<div>"
        f"<p style='margin:0;color:{text_color};font-size:15px'>{regime_desc}</p>"
        f"<p style='margin:5px 0 0;color:#888;font-size:13px'>Confidence: {regime_conf:.0f}%</p>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    # Regime probabilities
    probs = regime.get("probabilities", {})
    if probs:
        prob_cols = st.columns(len(probs))
        for col, (rname, rdata) in zip(prob_cols, probs.items()):
            p = rdata["probability"]
            c = rdata["color"]
            col.markdown(
                f"<div style='text-align:center;background:{surface};padding:15px;border-radius:8px;"
                f"border-top:3px solid {c}'>"
                f"<p style='margin:0;color:#888;font-size:11px'>{rname}</p>"
                f"<p style='margin:5px 0;font-size:28px;font-weight:700;color:{c}'>{p:.0f}%</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Regime history chart
    history = regime.get("history", pd.DataFrame())
    if not history.empty and "regime" in history.columns:
        st.subheader("Regime History (5 Years)")
        color_map = {"BULL": bull, "BEAR": bear, "TRANSITION": neutral}
        fig = go.Figure()
        for rname, rcolor in color_map.items():
            mask = history["regime"] == rname
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=history.index[mask], y=[rname] * mask.sum(),
                    mode="markers", marker=dict(color=rcolor, size=3),
                    name=rname, showlegend=True,
                ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=bg, plot_bgcolor=surface,
            height=200, margin=dict(l=80, r=30, t=10, b=30),
            yaxis=dict(categoryorder="array", categoryarray=["BEAR", "TRANSITION", "BULL"]),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Regime explanation
    if regime_explanation:
        with st.expander("How the HMM works (plain English)"):
            st.write(regime_explanation)
            st.write("---")
            st.write(
                "**What is an HMM?** A Hidden Markov Model assumes the market has hidden 'states' "
                "(like Bull, Bear, Transition) that we cannot directly see. We CAN see things like "
                "returns, volatility, VIX, bond prices, and credit spreads. The HMM learns the "
                "statistical relationship between these observable signals and the hidden states. "
                "Then it tells us: given what we can see today, which hidden state are we most "
                "likely in? That is what the percentages above represent."
            )

    # ═══════════════════════════════════════════════════════════
    # ML PREDICTION
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("AI Prediction (XGBoost + Random Forest)")

    all_tickers = sorted(set(
        list(BENCHMARKS.values()) + [s["etf"] for s in SECTORS.values()]
    ))
    ticker = st.selectbox("Select Asset to Predict", all_tickers, index=0, key="ai_ticker")

    horizon = st.radio("Prediction Horizon", [5, 10, 21], format_func=lambda x: {5: "1 Week", 10: "2 Weeks", 21: "1 Month"}[x], horizontal=True)

    with st.spinner(f"Training ML models on {ticker} ({horizon}-day horizon)... first time takes ~10 seconds"):
        from models.ml_models import train_ml_models
        try:
            forecaster = train_ml_models(ticker=ticker, horizon=horizon)
            prediction = forecaster.predict_current(ticker)
            metrics = forecaster.metrics
        except Exception as e:
            st.error(f"ML training failed: {e}")
            return

    if "error" in prediction:
        st.error(prediction["error"])
        return

    # Prediction card
    direction = prediction["direction"]
    confidence = prediction["confidence"]
    model_acc = prediction["model_accuracy"]

    if direction == "UP":
        dir_color = bull
    elif direction == "DOWN":
        dir_color = bear
    else:
        dir_color = neutral

    horizon_label = {5: "1 Week", 10: "2 Weeks", 21: "1 Month"}[horizon]

    st.markdown(
        f"<div style='background:linear-gradient(135deg,{surface},#1a1a2e);padding:30px;"
        f"border-radius:16px;border-left:6px solid {dir_color};margin-bottom:20px'>"
        f"<div style='display:flex;align-items:center;gap:20px'>"
        f"<div style='text-align:center'>"
        f"<p style='margin:0;color:#888;font-size:12px'>PREDICTION</p>"
        f"<p style='margin:5px 0;font-size:48px;font-weight:700;color:{dir_color}'>{direction}</p>"
        f"<p style='margin:0;font-size:13px;color:#aaa'>{ticker} over next {horizon_label}</p>"
        f"</div>"
        f"<div style='flex:1'>"
        f"<p style='color:{text_color};font-size:14px;margin:5px 0'>AI Confidence: <b style='color:{dir_color}'>{confidence:.0f}%</b></p>"
        f"<p style='color:{text_color};font-size:14px;margin:5px 0'>Model Accuracy (backtest): <b>{model_acc*100:.1f}%</b></p>"
        f"<p style='color:#888;font-size:12px;margin:5px 0'>Trained on {metrics.get('train_samples',0)} samples, tested on {metrics.get('test_samples',0)}</p>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    # Probability breakdown
    prob_data = prediction.get("probabilities", {})
    if prob_data:
        st.subheader("Probability Breakdown")
        p_cols = st.columns(len(prob_data))
        for col, (label, pdata) in zip(p_cols, prob_data.items()):
            p = pdata["probability"]
            xgb_p = pdata.get("xgb", p)
            rf_p = pdata.get("rf", p)
            if label == "UP":
                p_color = bull
            elif label == "DOWN":
                p_color = bear
            else:
                p_color = neutral

            col.markdown(
                f"<div style='text-align:center;background:{surface};padding:20px;border-radius:12px;"
                f"border-top:4px solid {p_color}'>"
                f"<p style='margin:0;font-size:13px;color:#888'>{label}</p>"
                f"<p style='margin:8px 0;font-size:36px;font-weight:700;color:{p_color}'>{p:.0f}%</p>"
                f"<p style='margin:0;font-size:11px;color:#666'>XGB: {xgb_p:.0f}% | RF: {rf_p:.0f}%</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Model metrics
    st.markdown("---")
    st.subheader("Model Performance (Honest Numbers)")
    st.caption("These are REAL out-of-sample accuracy numbers — the model never saw the test data during training")

    mc = st.columns(4)
    mc[0].metric("XGBoost Accuracy", f"{metrics.get('xgb_accuracy',0)*100:.1f}%")
    mc[1].metric("Random Forest Accuracy", f"{metrics.get('rf_accuracy',0)*100:.1f}%")
    mc[2].metric("Ensemble Accuracy", f"{metrics.get('ensemble_accuracy',0)*100:.1f}%")
    mc[3].metric("5-Fold CV Mean", f"{metrics.get('cv_mean_accuracy',0)*100:.1f}%")

    mc2 = st.columns(4)
    mc2[0].metric("Training Samples", f"{metrics.get('train_samples',0):,}")
    mc2[1].metric("Test Samples", f"{metrics.get('test_samples',0):,}")
    mc2[2].metric("Features Used", metrics.get("n_features", 0))
    test_start = metrics.get("test_period_start", "?")
    test_end = metrics.get("test_period_end", "?")
    mc2[3].metric("Test Period", f"{test_start} to {test_end}")

    # CV fold chart
    cv_scores = metrics.get("cv_fold_scores", [])
    if cv_scores:
        fig = go.Figure(go.Bar(
            x=[f"Fold {i+1}" for i in range(len(cv_scores))],
            y=[s * 100 for s in cv_scores],
            marker_color=[bull if s > 0.4 else bear for s in cv_scores],
            text=[f"{s*100:.1f}%" for s in cv_scores],
            textposition="outside",
        ))
        fig.add_hline(y=50, line_dash="dash", line_color="#FF5722",
                      annotation_text="Random guess (50%)")
        fig.update_layout(
            title="Cross-Validation Accuracy by Fold",
            template="plotly_dark", paper_bgcolor=bg, plot_bgcolor=surface,
            height=300, yaxis_range=[0, 80],
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    top_features = prediction.get("top_features", [])
    if top_features:
        st.subheader("What the AI is Looking At")
        st.caption("The features with the most influence on the prediction")

        feat_names = [f["feature"] for f in top_features]
        feat_imp = [f["importance"] * 100 for f in top_features]
        feat_vals = [f["value"] for f in top_features]

        fig = go.Figure(go.Bar(
            x=feat_imp,
            y=feat_names,
            orientation="h",
            marker_color=primary,
            text=[f"{v:.1f}%" for v in feat_imp],
            textposition="outside",
        ))
        fig.update_layout(
            title="Feature Importance (top 10)",
            template="plotly_dark", paper_bgcolor=bg, plot_bgcolor=surface,
            height=350, margin=dict(l=150, r=80, t=50, b=30),
            xaxis_title="Importance (%)",
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("What do these features mean?"):
            feature_explanations = {
                "RSI_14": "Relative Strength Index — measures if the stock is overbought (>70) or oversold (<30)",
                "MACD": "Moving Average Convergence/Divergence — momentum indicator",
                "MACD_Signal": "MACD signal line — when MACD crosses above, it is bullish",
                "MACD_Hist": "MACD histogram — shows the strength of momentum",
                "BB_Upper": "Bollinger Band upper — price near upper band = stretched up",
                "BB_Lower": "Bollinger Band lower — price near lower band = stretched down",
                "BB_Width": "Bollinger Band width — wider = more volatile",
                "ATR_14": "Average True Range — measures daily price volatility",
                "ADX_14": "Average Directional Index — measures trend strength (>25 = strong trend)",
                "OBV": "On-Balance Volume — accumulation/distribution pressure",
                "VWAP": "Volume-Weighted Average Price — institutional fair value",
                "SMA_20": "20-day Simple Moving Average — short-term trend",
                "SMA_50": "50-day Simple Moving Average — medium-term trend",
                "SMA_200": "200-day Simple Moving Average — long-term trend",
                "EMA_12": "12-day Exponential Moving Average — fast trend",
                "EMA_26": "26-day Exponential Moving Average — slow trend",
                "Returns_1d": "Yesterday's return",
                "Returns_5d": "Last week's return",
                "Returns_21d": "Last month's return",
                "Vol_21d": "21-day realized volatility",
                "Stoch_K": "Stochastic %K — momentum oscillator (>80 overbought, <20 oversold)",
                "Stoch_D": "Stochastic %D — smoothed version of %K",
                "Williams_R": "Williams %R — similar to Stochastic, measures overbought/oversold",
                "CCI_20": "Commodity Channel Index — measures price deviation from average",
                "MFI_14": "Money Flow Index — volume-weighted RSI",
            }
            for f in top_features:
                name = f["feature"]
                explanation = feature_explanations.get(name, "Technical indicator used by the model")
                val = f["value"]
                st.markdown(f"**{name}** = {val:.2f} — {explanation}")

    # How it works
    with st.expander("How the AI Forecast Works (plain English)"):
        st.write(
            f"**Step 1: Data.** We take 5 years of daily price data for {ticker}.\n\n"
            f"**Step 2: Features.** We compute {metrics.get('n_features', 41)} technical indicators — "
            "things like RSI, MACD, Bollinger Bands, moving averages, volume patterns, and volatility. "
            "These are the 'clues' the AI uses.\n\n"
            f"**Step 3: Target.** For every day in history, we ask: was {ticker} UP or DOWN "
            f"over the next {horizon} trading days? This gives us the 'answer key.'\n\n"
            f"**Step 4: Training.** We give the first 80% of the data ({metrics.get('train_samples',0)} days) "
            "to two AI models — XGBoost and Random Forest. They learn the patterns between the clues "
            "and the outcomes.\n\n"
            f"**Step 5: Testing.** We test on the remaining 20% ({metrics.get('test_samples',0)} days) "
            "that the AI has NEVER seen. This is the honest accuracy number.\n\n"
            f"**Step 6: Today's Prediction.** We compute today's features and ask the trained AI: "
            "based on what you learned, what happens next?\n\n"
            f"**The result:** {direction} with {confidence:.0f}% confidence. "
            f"On historical data the model never saw, it was right {model_acc*100:.1f}% of the time."
        )

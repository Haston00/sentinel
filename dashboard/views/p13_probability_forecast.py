"""
SENTINEL — Page 13: Probability Forecast Engine.
Shows you the actual probabilities based on intermarket rules,
technical signals, and historical patterns. Real forecasting, not guessing.
"""

import streamlit as st
import plotly.graph_objects as go

from config.settings import COLORS
from features.forecaster import generate_forecasts


def render():
    st.title("Probability Forecast Engine")
    st.caption("Real probability-weighted forecasts based on 18 institutional trading rules")

    surface = COLORS["surface"]
    text_color = COLORS["text"]
    bull_color = COLORS["bull"]
    bear_color = COLORS["bear"]
    neutral_color = COLORS["neutral"]
    primary = COLORS["primary"]
    bg = COLORS["background"]

    with st.spinner("Running 14 forecasting rules across all asset classes..."):
        forecasts = generate_forecasts()

    rules = forecasts["rules_triggered"]
    confidence = forecasts["confidence"]

    # ═══════════════════════════════════════════════════════════
    # CONFIDENCE BANNER
    # ═══════════════════════════════════════════════════════════
    if confidence == "HIGH":
        conf_color = bull_color
        conf_msg = "High confidence — multiple independent rules agree on direction"
    elif confidence == "MODERATE":
        conf_color = neutral_color
        conf_msg = "Moderate confidence — some agreement but not unanimous"
    else:
        conf_color = "#888"
        conf_msg = "Low confidence — rules are conflicting, stay cautious"

    bullish_rules = [r for r in rules if r["bias"] == "BULLISH"]
    bearish_rules = [r for r in rules if r["bias"] == "BEARISH"]

    st.markdown(
        f"<div style='background:{surface};padding:20px;border-radius:12px;"
        f"border-left:5px solid {conf_color};margin-bottom:20px'>"
        f"<div style='display:flex;align-items:center;gap:15px;flex-wrap:wrap'>"
        f"<span style='background:{conf_color};color:white;padding:5px 14px;border-radius:15px;font-size:13px;font-weight:600'>{confidence} CONFIDENCE</span>"
        f"<span style='color:#aaa;font-size:13px'>{conf_msg}</span>"
        f"</div>"
        f"<div style='display:flex;gap:20px;margin-top:12px'>"
        f"<span style='color:{bull_color};font-weight:600'>{len(bullish_rules)} Bullish Rules</span>"
        f"<span style='color:{bear_color};font-weight:600'>{len(bearish_rules)} Bearish Rules</span>"
        f"<span style='color:#888'>{len(rules)} Total Triggered (of 14)</span>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ═══════════════════════════════════════════════════════════
    # FORECAST CARDS (1W, 1M, 3M)
    # ═══════════════════════════════════════════════════════════
    st.subheader("S&P 500 Forecast")

    fc_cols = st.columns(3)
    horizons = [
        ("1 Week", forecasts["forecast_1w"]),
        ("1 Month", forecasts["forecast_1m"]),
        ("3 Months", forecasts["forecast_3m"]),
    ]

    for col, (label, fc) in zip(fc_cols, horizons):
        direction = fc.get("direction", "NEUTRAL")
        probability = fc.get("probability", 50)
        expected = fc.get("expected_return", 0)
        range_low = fc.get("range_low", 0)
        range_high = fc.get("range_high", 0)
        price_low = fc.get("price_target_low", 0)
        price_high = fc.get("price_target_high", 0)
        current = fc.get("current_price", 0)

        if direction == "BULLISH":
            dir_color = bull_color
        elif direction == "BEARISH":
            dir_color = bear_color
        else:
            dir_color = neutral_color

        with col:
            st.markdown(
                f"<div style='background:{surface};padding:20px;border-radius:12px;text-align:center;"
                f"border-top:4px solid {dir_color}'>"
                f"<p style='margin:0;color:#888;font-size:12px;text-transform:uppercase;letter-spacing:2px'>{label}</p>"
                f"<p style='margin:10px 0 5px;font-size:38px;font-weight:700;color:{dir_color}'>{probability}%</p>"
                f"<p style='margin:0;font-size:13px;color:{dir_color};font-weight:600'>{direction}</p>"
                f"<p style='margin:10px 0 0;font-size:12px;color:#aaa'>Expected: {expected:+.1f}%</p>"
                f"<p style='margin:3px 0 0;font-size:11px;color:#666'>Range: {range_low:+.1f}% to {range_high:+.1f}%</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Price targets
    current_price = forecasts["forecast_1m"].get("current_price", 0)
    if current_price > 0:
        st.markdown(
            f"<p style='text-align:center;color:#888;font-size:13px;margin-top:10px'>"
            f"SPY Current: ${current_price:,.2f}</p>",
            unsafe_allow_html=True,
        )

    # ═══════════════════════════════════════════════════════════
    # PROBABILITY GAUGE CHART
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Probability Gauges")

    gauge_cols = st.columns(3)
    for col, (label, fc) in zip(gauge_cols, horizons):
        prob = fc.get("probability", 50)
        direction = fc.get("direction", "NEUTRAL")

        if prob >= 60:
            g_color = bull_color
        elif prob <= 40:
            g_color = bear_color
        else:
            g_color = neutral_color

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title=dict(text=f"{label}", font=dict(size=14, color=text_color)),
            number=dict(font=dict(size=36, color=g_color), suffix="%"),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor="#555"),
                bar=dict(color=g_color, thickness=0.3),
                bgcolor=surface,
                bordercolor="#444",
                steps=[
                    dict(range=[0, 30], color="rgba(255,23,68,0.12)"),
                    dict(range=[30, 45], color="rgba(255,152,0,0.08)"),
                    dict(range=[45, 55], color="rgba(200,200,200,0.05)"),
                    dict(range=[55, 70], color="rgba(139,195,74,0.08)"),
                    dict(range=[70, 100], color="rgba(0,200,83,0.12)"),
                ],
                threshold=dict(
                    line=dict(color="white", width=2),
                    thickness=0.75,
                    value=prob,
                ),
            ),
        ))
        fig.add_annotation(
            text=f"<b>{direction}</b>",
            xref="paper", yref="paper",
            x=0.5, y=-0.1, showarrow=False,
            font=dict(size=13, color=g_color),
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=220,
            margin=dict(l=30, r=30, t=50, b=40),
        )
        col.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════════════════════
    # TRIGGERED RULES
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Forecasting Rules Triggered")
    st.caption("Each rule is based on documented institutional trading patterns with historical hit rates")

    if not rules:
        st.info("No strong directional rules triggered — the market is in a neutral zone. Be patient.")
    else:
        # Sort by weight (importance)
        sorted_rules = sorted(rules, key=lambda x: x["weight"], reverse=True)

        for rule in sorted_rules:
            bias = rule["bias"]
            weight = rule["weight"]
            name = rule["name"]
            explanation = rule["explanation"]
            historical = rule.get("historical", "")

            if bias == "BULLISH":
                r_color = bull_color
                icon = "+"
            elif bias == "BEARISH":
                r_color = bear_color
                icon = "!"
            else:
                r_color = neutral_color
                icon = "~"

            # Weight bar
            weight_pct = weight * 100

            html_parts = [
                f"<div style='background:{surface};padding:20px;border-radius:12px;margin-bottom:12px;"
                f"border-left:4px solid {r_color}'>",
                f"<div style='display:flex;align-items:center;margin-bottom:8px'>",
                f"<span style='background:{r_color};color:white;width:24px;height:24px;border-radius:50%;"
                f"display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px;margin-right:10px'>{icon}</span>",
                f"<span style='font-size:16px;font-weight:600;color:{text_color}'>{name}</span>",
                f"<span style='margin-left:auto;color:{r_color};font-weight:600;font-size:13px'>{bias}</span>",
                f"</div>",
                f"<p style='color:{text_color};font-size:14px;margin:8px 0;line-height:1.6'>{explanation}</p>",
            ]

            if historical:
                html_parts.append(
                    f"<p style='color:#888;font-size:12px;margin:8px 0 0;padding:8px;background:rgba(255,255,255,0.03);"
                    f"border-radius:4px;font-style:italic'>Historical: {historical}</p>"
                )

            # Weight indicator
            html_parts.append(
                f"<div style='margin-top:10px;display:flex;align-items:center;gap:8px'>"
                f"<span style='color:#666;font-size:11px'>Weight:</span>"
                f"<div style='flex:1;height:4px;background:#333;border-radius:2px;overflow:hidden'>"
                f"<div style='width:{weight_pct}%;height:100%;background:{r_color};border-radius:2px'></div>"
                f"</div>"
                f"<span style='color:#888;font-size:11px'>{weight_pct:.0f}%</span>"
                f"</div>"
            )
            html_parts.append("</div>")
            st.markdown("".join(html_parts), unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # BULL vs BEAR SCORECARD
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Bull vs Bear Scorecard")

    bull_weight = sum(r["weight"] for r in bullish_rules)
    bear_weight = sum(r["weight"] for r in bearish_rules)
    total_weight = bull_weight + bear_weight if (bull_weight + bear_weight) > 0 else 1

    # Use the actual probability from the 1-month forecast for the scorecard
    # This prevents the misleading 100%/0% when only one side has rules
    fc_1m_prob = forecasts["forecast_1m"].get("probability", 50)
    bull_pct = fc_1m_prob
    bear_pct = 100 - fc_1m_prob

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[bull_pct],
        y=["Forecast"],
        orientation="h",
        name="Bullish",
        marker_color=bull_color,
        text=[f"BULL {bull_pct:.0f}%"],
        textposition="inside",
        textfont=dict(size=16, color="white"),
    ))
    fig.add_trace(go.Bar(
        x=[bear_pct],
        y=["Forecast"],
        orientation="h",
        name="Bearish",
        marker_color=bear_color,
        text=[f"BEAR {bear_pct:.0f}%"],
        textposition="inside",
        textfont=dict(size=16, color="white"),
    ))
    fig.update_layout(
        barmode="stack",
        template="plotly_dark",
        paper_bgcolor=bg,
        plot_bgcolor=surface,
        height=120,
        margin=dict(l=80, r=30, t=10, b=10),
        showlegend=False,
        xaxis=dict(range=[0, 100], showticklabels=False),
        yaxis=dict(showticklabels=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Rule lists
    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown(f"**Bullish Factors ({len(bullish_rules)}):**")
        for r in bullish_rules:
            st.markdown(
                f"<span style='color:{bull_color}'>+ {r['name']}</span> "
                f"<span style='color:#666;font-size:11px'>(weight: {r['weight']*100:.0f}%)</span>",
                unsafe_allow_html=True,
            )
    with bc2:
        st.markdown(f"**Bearish Factors ({len(bearish_rules)}):**")
        for r in bearish_rules:
            st.markdown(
                f"<span style='color:{bear_color}'>- {r['name']}</span> "
                f"<span style='color:#666;font-size:11px'>(weight: {r['weight']*100:.0f}%)</span>",
                unsafe_allow_html=True,
            )

    # ═══════════════════════════════════════════════════════════
    # HISTORICAL CONTEXT
    # ═══════════════════════════════════════════════════════════
    if forecasts["historical_context"]:
        st.markdown("---")
        st.subheader("Historical Context")
        st.caption("What happened in the past when similar conditions existed")

        for ctx in forecasts["historical_context"]:
            bias = ctx["bias"]
            if bias == "BULLISH":
                ctx_color = bull_color
            elif bias == "BEARISH":
                ctx_color = bear_color
            else:
                ctx_color = "#888"

            st.markdown(
                f"<div style='padding:12px 16px;margin:6px 0;background:{surface};"
                f"border-radius:8px;border-left:3px solid {ctx_color}'>"
                f"<span style='color:{ctx_color};font-weight:600;font-size:13px'>{ctx['rule']}:</span> "
                f"<span style='color:#bbb;font-size:13px'>{ctx['context']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ═══════════════════════════════════════════════════════════
    # CONVICTION FACTORS
    # ═══════════════════════════════════════════════════════════
    if forecasts["conviction_factors"]:
        st.markdown("---")
        st.subheader("Highest Conviction Factors")

        for factor in forecasts["conviction_factors"]:
            bias = factor["bias"]
            f_color = bull_color if bias == "BULLISH" else bear_color if bias == "BEARISH" else neutral_color
            strength = factor["strength"] * 100

            st.markdown(
                f"<div style='background:{surface};padding:15px;border-radius:8px;margin-bottom:8px'>"
                f"<div style='display:flex;align-items:center;justify-content:space-between'>"
                f"<span style='color:{text_color};font-weight:600'>{factor['factor']}</span>"
                f"<span style='background:{f_color};color:white;padding:3px 10px;border-radius:10px;font-size:11px'>"
                f"{bias} | {strength:.0f}% weight</span>"
                f"</div>"
                f"<p style='color:#aaa;font-size:13px;margin:8px 0 0'>{factor['explanation']}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ═══════════════════════════════════════════════════════════
    # DISCLAIMER
    # ═══════════════════════════════════════════════════════════
    st.markdown(
        f"<p style='text-align:center;color:#444;font-size:10px;margin-top:30px;padding:10px'>"
        f"Probabilities are based on historical patterns and intermarket relationships. "
        f"Past performance does not guarantee future results. These are analytical tools, not financial advice. "
        f"Always do your own research and manage risk appropriately."
        f"</p>",
        unsafe_allow_html=True,
    )

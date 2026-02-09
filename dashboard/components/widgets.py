"""
SENTINEL — Reusable Streamlit UI widgets.
"""

import streamlit as st
import streamlit.components.v1 as components

from config.settings import COLORS


def render_ticker_bar(ticker_data: list) -> None:
    """Render a Wall Street-style scrolling stock ticker bar."""
    if not ticker_data:
        return

    items_html = ""
    for item in ticker_data:
        symbol = item["symbol"]
        price = item["price"]
        change = item["change"]
        color = "#00C853" if change >= 0 else "#FF1744"
        arrow = "\u25b2" if change >= 0 else "\u25bc"
        sign = "+" if change >= 0 else ""

        if price >= 10000:
            price_str = f"${price:,.0f}"
        elif price >= 1:
            price_str = f"${price:,.2f}"
        else:
            price_str = f"${price:.4f}"

        items_html += (
            f'<span class="ti">'
            f'<span class="ts">{symbol}</span>'
            f'<span class="tp">{price_str}</span>'
            f'<span class="tc" style="color:{color}">{arrow} {sign}{change:.2%}</span>'
            f'</span>'
        )

    ticker_html = f"""<!DOCTYPE html>
<html><head><style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0e1a;overflow:hidden;font-family:'Segoe UI',Consolas,monospace}}
@keyframes scroll{{0%{{transform:translateX(0)}}100%{{transform:translateX(-50%)}}}}
.tw{{background:linear-gradient(90deg,#0a0e1a,#111827 20%,#111827 80%,#0a0e1a);
border-top:1px solid #1e3a5f;border-bottom:1px solid #1e3a5f;overflow:hidden;padding:8px 0;position:relative}}
.tw::before,.tw::after{{content:'';position:absolute;top:0;bottom:0;width:40px;z-index:2;pointer-events:none}}
.tw::before{{left:0;background:linear-gradient(90deg,#0a0e1a,transparent)}}
.tw::after{{right:0;background:linear-gradient(90deg,transparent,#0a0e1a)}}
.tt{{display:flex;align-items:center;animation:scroll 40s linear infinite;width:max-content}}
.tw:hover .tt{{animation-play-state:paused}}
.ti{{display:inline-flex;align-items:center;gap:5px;margin:0 18px;white-space:nowrap}}
.ts{{color:#8892b0;font-weight:800;font-size:11px;letter-spacing:1px}}
.tp{{color:#e6e6e6;font-weight:600;font-size:12px}}
.tc{{font-weight:700;font-size:11px}}
.sep{{color:#1e3a5f;margin:0 4px;font-size:10px}}
</style></head><body>
<div class="tw"><div class="tt">{items_html}{items_html}</div></div>
</body></html>"""

    components.html(ticker_html, height=38, scrolling=False)


def render_brand_bar() -> None:
    """Render the 'From the Mind of Brandon Haston' brand bar."""
    st.markdown(
        '<div style="text-align:center;padding:6px 0 2px;margin-bottom:8px">'
        '<span style="color:#3a5a7a;font-size:10px;letter-spacing:4px;'
        "font-style:italic;font-family:'Segoe UI',sans-serif;"
        'text-transform:uppercase">'
        'from the mind of Brandon Haston'
        '</span></div>',
        unsafe_allow_html=True,
    )


def forecast_card(forecast: dict, horizon: str) -> None:
    """Display a forecast summary card."""
    fc = forecast.get(horizon, {})
    if not fc:
        st.warning(f"No forecast available for {horizon}")
        return

    direction = fc.get("direction_label", "Neutral")
    confidence = fc.get("confidence", 0)
    point = fc.get("point", 0) * 100

    color = COLORS["bull"] if direction == "Bullish" else COLORS["bear"] if direction == "Bearish" else COLORS["neutral"]
    surface = COLORS["surface"]
    text_color = COLORS["text"]
    n_models = fc.get("n_models", "N/A")

    st.markdown(
        f"""
        <div style="
            background-color: {surface};
            border-left: 4px solid {color};
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
        ">
            <h3 style="margin:0; color:{color}">{horizon} - {direction}</h3>
            <p style="margin:5px 0; font-size:24px; color:{text_color}">{point:+.2f}%</p>
            <p style="margin:0; color:#aaa">Confidence: {confidence:.0%} | Models: {n_models}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def regime_indicator(regime_info: dict) -> None:
    """Display current regime with probability bars."""
    regime = regime_info.get("regime", "Unknown")
    probs = regime_info.get("probabilities", {})

    color_map = {"Bull": COLORS["bull"], "Bear": COLORS["bear"], "Transition": COLORS["transition"]}
    color = color_map.get(regime, "#888")
    surface = COLORS["surface"]

    st.markdown(
        f"""
        <div style="
            background-color: {surface};
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        ">
            <h4 style="margin:0; color:#aaa">Current Regime</h4>
            <h2 style="margin:5px 0; color:{color}">{regime}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if probs:
        for regime_name, prob in probs.items():
            bar_color = color_map.get(regime_name, "#888")
            width_pct = prob * 100
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; margin:3px 0;">
                    <span style="width:80px; color:#aaa">{regime_name}</span>
                    <div style="flex:1; background:#333; border-radius:4px; height:12px; margin:0 8px;">
                        <div style="width:{width_pct:.0f}%; background:{bar_color}; height:100%; border-radius:4px;"></div>
                    </div>
                    <span style="width:40px; color:#aaa">{prob:.0%}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def metric_row(metrics: dict) -> None:
    """Display a row of key metrics."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        if isinstance(value, float):
            if abs(value) < 1:
                display = f"{value:.2%}"
            else:
                display = f"{value:,.2f}"
        else:
            display = str(value)
        col.metric(label, display)


def explanation_panel(explanation: dict) -> None:
    """Display forecast explanation with bullish/bearish factors."""
    st.markdown(f"**{explanation.get('summary', '')}**")

    col1, col2 = st.columns(2)
    bull_color = COLORS["bull"]
    bear_color = COLORS["bear"]

    with col1:
        st.markdown(f"<h4 style='color:{bull_color}'>Bullish Factors</h4>", unsafe_allow_html=True)
        for factor in explanation.get("bullish_factors", []):
            st.markdown(f"+ {factor}")
        if not explanation.get("bullish_factors"):
            st.markdown("_None identified_")

    with col2:
        st.markdown(f"<h4 style='color:{bear_color}'>Bearish Factors</h4>", unsafe_allow_html=True)
        for factor in explanation.get("bearish_factors", []):
            st.markdown(f"- {factor}")
        if not explanation.get("bearish_factors"):
            st.markdown("_None identified_")

    if explanation.get("neutral_factors"):
        st.markdown("**Watch Factors:**")
        for factor in explanation.get("neutral_factors", []):
            st.markdown(f"~ {factor}")


def news_feed(news_df, max_items: int = 20) -> None:
    """Display a scrollable news feed with sentiment coloring."""
    if news_df is None or news_df.empty:
        st.info("No news data available")
        return

    bull_color = COLORS["bull"]
    bear_color = COLORS["bear"]

    for _, row in news_df.head(max_items).iterrows():
        sentiment = row.get("Sentiment", 0)
        if sentiment > 0.15:
            indicator = f"<span style='color:{bull_color}'>&#9650;</span>"
        elif sentiment < -0.15:
            indicator = f"<span style='color:{bear_color}'>&#9660;</span>"
        else:
            indicator = "<span style='color:#888'>&#9679;</span>"

        urgency = row.get("Urgency", "low")
        urgency_badge = ""
        if urgency == "high":
            urgency_badge = f"<span style='background:{bear_color};color:white;padding:1px 6px;border-radius:3px;font-size:10px'>HIGH</span>"

        title = row.get("Title", "Untitled")
        source = row.get("Source", "")
        date = str(row.get("Date", ""))[:16]

        st.markdown(
            f"{indicator} **{title}** {urgency_badge}<br>"
            f"<small style='color:#888'>{source} | {date} | Sentiment: {sentiment:.2f}</small>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

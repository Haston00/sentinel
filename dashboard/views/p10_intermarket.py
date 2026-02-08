"""
SENTINEL — Page 10: Intermarket Intelligence.
Cross-asset analysis that detects the hidden signals most traders miss.
When bonds, stocks, gold, and dollar disagree — the biggest moves follow.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config.settings import COLORS
from features.intermarket import compute_intermarket_signals


def render():
    st.title("Intermarket Intelligence")
    st.caption("Cross-asset signals that predict the biggest moves")
    surface = COLORS["surface"]
    text_color = COLORS["text"]

    with st.spinner("Analyzing cross-asset relationships..."):
        result = compute_intermarket_signals()

    if not result["signals"] and not result["divergences"]:
        st.info("Insufficient data for intermarket analysis. Ensure market data is loaded.")
        return

    # ── Overall Assessment ─────────────────────────────────
    overall = result.get("overall", "ANALYZING...")
    if "BULLISH" in overall:
        bg_color = "rgba(0,200,83,0.1)"
        border_color = COLORS["bull"]
    elif "BEARISH" in overall:
        bg_color = "rgba(255,23,68,0.1)"
        border_color = COLORS["bear"]
    else:
        bg_color = "rgba(255,214,0,0.1)"
        border_color = COLORS["neutral"]

    st.markdown(
        f"<div style='background:{bg_color};border-left:4px solid {border_color};"
        f"padding:20px;border-radius:8px;margin-bottom:20px'>"
        f"<h3 style='margin:0;color:{border_color}'>{overall}</h3>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Active Signals ─────────────────────────────────────
    st.subheader("Active Signals")
    if result["signals"]:
        for sig in result["signals"]:
            direction = sig.get("direction", "NEUTRAL")
            strength = sig.get("strength", "MODERATE")

            if direction == "BULLISH":
                icon = "🟢"
                color = COLORS["bull"]
            elif direction == "BEARISH":
                icon = "🔴"
                color = COLORS["bear"]
            elif direction == "MIXED":
                icon = "🟡"
                color = COLORS["neutral"]
            else:
                icon = "⚪"
                color = "#aaa"

            strength_badge = f"<span style='background:{color};color:white;padding:2px 8px;border-radius:3px;font-size:10px'>{strength}</span>"

            st.markdown(
                f"<div style='background:{surface};padding:15px;border-radius:8px;margin-bottom:10px;border-left:3px solid {color}'>"
                f"<p style='margin:0;font-size:16px;font-weight:600;color:{text_color}'>{icon} {sig['signal']} {strength_badge}</p>"
                f"<p style='margin:5px 0 0;color:#aaa;font-size:13px'>{sig['detail']}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No strong intermarket signals detected right now")

    # ── Divergence Alerts ──────────────────────────────────
    if result["divergences"]:
        st.subheader("Divergence Alerts")
        for div in result["divergences"]:
            st.warning(div)

    # ── Cross-Asset Returns ────────────────────────────────
    st.markdown("---")
    st.subheader("Cross-Asset Performance")

    ret_1w = result.get("returns_1w", {})
    ret_1m = result.get("returns_1m", {})

    if ret_1w and ret_1m:
        asset_names = {
            "SPY": "S&P 500", "QQQ": "Nasdaq", "TLT": "Treasuries",
            "GLD": "Gold", "UUP": "Dollar", "HYG": "High Yield",
            "LQD": "Inv. Grade", "USO": "Crude Oil",
        }

        # 1W and 1M side-by-side bar chart
        tickers = list(ret_1m.keys())
        labels = [asset_names.get(t, t) for t in tickers]
        vals_1w = [ret_1w.get(t, 0) * 100 for t in tickers]
        vals_1m = [ret_1m.get(t, 0) * 100 for t in tickers]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="1 Week", x=labels, y=vals_1w,
            marker_color=[COLORS["bull"] if v > 0 else COLORS["bear"] for v in vals_1w],
            opacity=0.7,
        ))
        fig.add_trace(go.Bar(
            name="1 Month", x=labels, y=vals_1m,
            marker_color=[COLORS["primary"] if v > 0 else "#FF5722" for v in vals_1m],
            opacity=0.9,
        ))
        fig.update_layout(
            title="Cross-Asset Returns (%)",
            barmode="group",
            template="plotly_dark",
            paper_bgcolor=COLORS["background"],
            plot_bgcolor=COLORS["surface"],
            height=400,
            yaxis_title="%",
        )
        fig.add_hline(y=0, line_color="#555")
        st.plotly_chart(fig, use_container_width=True)

    # ── Correlation Matrix ─────────────────────────────────
    corr = result.get("correlations", pd.DataFrame())
    if not corr.empty:
        st.subheader("60-Day Correlation Matrix")
        asset_labels = {
            "SPY": "S&P 500", "QQQ": "Nasdaq", "TLT": "Treasuries",
            "GLD": "Gold", "UUP": "Dollar", "HYG": "High Yield",
            "LQD": "Inv. Grade", "USO": "Crude Oil",
        }
        display_labels = [asset_labels.get(c, c) for c in corr.columns]

        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=display_labels,
            y=display_labels,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}",
            textfont=dict(size=11),
        ))
        fig.update_layout(
            title="Cross-Asset Correlation (60-day rolling)",
            template="plotly_dark",
            paper_bgcolor=COLORS["background"],
            plot_bgcolor=COLORS["surface"],
            height=500,
            margin=dict(l=100, r=30, t=50, b=80),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Highlight unusual correlations
        st.subheader("Unusual Correlations")
        unusual = []
        for i in range(len(corr)):
            for j in range(i + 1, len(corr)):
                val = corr.iloc[i, j]
                name_i = display_labels[i]
                name_j = display_labels[j]
                # Stock-Bond positive correlation is unusual
                if ("S&P" in name_i or "Nasdaq" in name_i) and "Treasuries" in name_j:
                    if val > 0.2:
                        unusual.append(f"**{name_i} / {name_j}**: Correlation {val:+.2f} — Positive (unusual, suggests macro regime shift)")
                # Gold-Dollar positive correlation is unusual
                if "Gold" in name_i and "Dollar" in name_j:
                    if val > 0.1:
                        unusual.append(f"**{name_i} / {name_j}**: Correlation {val:+.2f} — Positive (fear trade active)")
                # Very high correlations
                if abs(val) > 0.8 and abs(val) < 1.0:
                    unusual.append(f"**{name_i} / {name_j}**: Correlation {val:+.2f} — Very high, watch for decorrelation")

        if unusual:
            for u in unusual:
                st.markdown(u)
        else:
            st.info("No unusual correlation patterns detected — markets in normal regime")

    # ── Normalized Performance Chart ───────────────────────
    closes = result.get("closes", pd.DataFrame())
    if not closes.empty:
        st.subheader("Normalized Performance (90-day)")
        normalized = closes.tail(90) / closes.tail(90).iloc[0] * 100

        fig = go.Figure()
        line_colors = {
            "SPY": COLORS["bull"], "QQQ": "#8BC34A", "TLT": COLORS["primary"],
            "GLD": COLORS["neutral"], "UUP": "#FF9800",
            "HYG": "#9C27B0", "LQD": "#00BCD4", "USO": COLORS["bear"],
        }
        for col in normalized.columns:
            fig.add_trace(go.Scatter(
                x=normalized.index, y=normalized[col],
                name=col, line=dict(width=2, color=line_colors.get(col, "#888")),
            ))
        fig.add_hline(y=100, line_dash="dash", line_color="#555")
        fig.update_layout(
            title="Relative Performance (indexed to 100)",
            template="plotly_dark",
            paper_bgcolor=COLORS["background"],
            plot_bgcolor=COLORS["surface"],
            height=450,
            yaxis_title="Indexed Value",
        )
        st.plotly_chart(fig, use_container_width=True)

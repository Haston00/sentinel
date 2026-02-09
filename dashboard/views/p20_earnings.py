"""
SENTINEL — Page 20: Earnings Surprise Model.
Track earnings beats/misses, learn stock-specific reaction patterns,
predict price moves around earnings announcements.
"""

import streamlit as st
import plotly.graph_objects as go

from config.settings import COLORS


def render():
    st.title("Earnings Surprise Model")
    st.caption(
        "Track actual vs estimated earnings. Learn how each stock reacts to beats and misses. "
        "Predict price moves before the next earnings report."
    )

    surface = COLORS["surface"]
    text_color = COLORS["text"]
    bull = COLORS["bull"]
    bear = COLORS["bear"]
    primary = COLORS["primary"]
    bg = COLORS["background"]

    try:
        from models.earnings import EarningsSurpriseModel
        model = EarningsSurpriseModel()
    except Exception as e:
        st.error(f"Earnings model failed to load: {e}")
        return

    stats = model.get_stats()

    # ── Status Banner ─────────────────────────────────────────────
    if stats["total_records"] == 0:
        st.warning("Earnings database is empty. Build it below to start tracking.")
    else:
        st.markdown(
            f"<div style='background:{surface};padding:20px;border-radius:12px;"
            f"border-left:5px solid {primary};margin-bottom:20px'>"
            f"<div style='display:flex;gap:30px;flex-wrap:wrap'>"
            f"<div><span style='font-size:36px;font-weight:700;color:{primary}'>"
            f"{stats['total_records']}</span>"
            f"<br><span style='color:#888;font-size:12px'>EARNINGS RECORDS</span></div>"
            f"<div><span style='font-size:36px;font-weight:700;color:{text_color}'>"
            f"{stats['tickers_tracked']}</span>"
            f"<br><span style='color:#888;font-size:12px'>TICKERS TRACKED</span></div>"
            f"<div><span style='font-size:36px;font-weight:700;color:{bull}'>"
            f"{stats['patterns_learned']}</span>"
            f"<br><span style='color:#888;font-size:12px'>PATTERNS LEARNED</span></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    # ── Build Controls ────────────────────────────────────────────
    from config.assets import SECTORS

    st.subheader("Build Earnings Database")
    bc1, bc2 = st.columns(2)
    with bc1:
        selected_sectors = st.multiselect(
            "Select sectors to track",
            list(SECTORS.keys()),
            default=["Technology", "Financials", "Healthcare"],
            key="earn_sectors",
        )
    with bc2:
        custom_tickers = st.text_input(
            "Additional tickers (comma-separated)",
            value="AAPL,MSFT,NVDA,TSLA,AMZN,META,GOOGL",
            key="earn_custom",
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Build Earnings History", type="primary"):
            tickers = []
            for sector in selected_sectors:
                tickers.extend(SECTORS[sector]["holdings"][:5])  # Top 5 per sector
            if custom_tickers:
                tickers.extend([t.strip() for t in custom_tickers.split(",") if t.strip()])
            tickers = list(set(tickers))

            with st.spinner(f"Downloading earnings for {len(tickers)} tickers..."):
                n = model.build_history(tickers)
            st.success(f"Downloaded {n} earnings records")
            st.rerun()

    with col2:
        if st.button("Learn Reaction Patterns"):
            with st.spinner("Analyzing earnings reaction patterns..."):
                patterns = model.learn_patterns()
            st.success(f"Learned patterns for {len(patterns)} tickers")
            st.rerun()

    if stats["total_records"] == 0:
        st.info(
            "**Step 1:** Click 'Build Earnings History' to download earnings data.\n\n"
            "**Step 2:** Click 'Learn Reaction Patterns' to analyze how each stock reacts.\n\n"
            "**Step 3:** Use the tabs below to predict moves and view upcoming earnings."
        )
        return

    # ═══════════════════════════════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════════════════════════════
    tab1, tab2, tab3 = st.tabs(["Predict Reaction", "Upcoming Earnings", "Learned Patterns"])

    # ── TAB 1: Predict Reaction ───────────────────────────────────
    with tab1:
        st.subheader("Predict Earnings Reaction")
        st.markdown("Select a ticker and expected surprise to see predicted price reaction.")

        pc1, pc2 = st.columns(2)
        with pc1:
            pred_ticker = st.selectbox(
                "Ticker",
                sorted(model._patterns.keys()) if model._patterns else ["AAPL"],
                key="earn_pred_ticker",
            )
        with pc2:
            surprise = st.slider(
                "Expected Surprise (%)",
                min_value=-20.0, max_value=20.0, value=5.0, step=0.5,
                key="earn_surprise",
            )

        if st.button("Predict Reaction", key="earn_predict"):
            prediction = model.predict_reaction(pred_ticker, surprise)

            if "error" in prediction:
                st.warning(prediction["error"])
            else:
                direction = prediction.get("direction", "neutral")
                d_color = bull if direction == "up" else (bear if direction == "down" else COLORS["neutral"])
                day1 = prediction.get("predicted_day1_pct", 0)

                st.markdown(
                    f"<div style='background:{surface};padding:20px;border-radius:12px;"
                    f"border-left:5px solid {d_color};margin-bottom:15px'>"
                    f"<span style='font-size:14px;color:#888'>PREDICTED REACTION TO "
                    f"{surprise:+.1f}% SURPRISE</span><br>"
                    f"<span style='font-size:36px;font-weight:700;color:{d_color}'>"
                    f"{day1:+.2f}%</span>"
                    f"<span style='color:#888;margin-left:15px'>Day 1 Move</span>"
                    f"<br><span style='color:#888'>Confidence: "
                    f"{prediction.get('confidence', 0):.0%} | "
                    f"Based on {prediction.get('based_on_n', 0)} historical quarters</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                drift = prediction.get("predicted_drift_5d_pct")
                if drift is not None:
                    drift_color = bull if drift > 0 else bear
                    st.markdown(
                        f"**5-Day Drift:** "
                        f"<span style='color:{drift_color}'>{drift:+.2f}%</span>",
                        unsafe_allow_html=True,
                    )

    # ── TAB 2: Upcoming Earnings ──────────────────────────────────
    with tab2:
        st.subheader("Upcoming Earnings Calendar")

        tracked_tickers = list(model._patterns.keys())[:50]
        if tracked_tickers:
            with st.spinner("Checking upcoming earnings dates..."):
                upcoming = model.get_upcoming_earnings(tracked_tickers)

            if not upcoming:
                st.info("No upcoming earnings found for tracked tickers.")
            else:
                for u in upcoming[:20]:
                    has_pattern = u.get("has_pattern", False)
                    badge = f"<span style='background:{primary};color:white;padding:2px 8px;border-radius:4px;font-size:11px'>PATTERN</span>" if has_pattern else ""

                    st.markdown(
                        f"<div style='background:{surface};padding:12px;border-radius:8px;"
                        f"margin-bottom:6px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap'>"
                        f"<div>"
                        f"<strong style='color:{text_color}'>{u['ticker']}</strong> "
                        f"{badge}"
                        f"</div>"
                        f"<div>"
                        f"<span style='color:#888'>Date: </span>"
                        f"<span style='color:{text_color}'>{u['date']}</span>"
                        f"<span style='color:#888;margin-left:15px'>"
                        f"({u['days_until']} days)</span>"
                        f"</div>"
                        f"<div>"
                        f"<span style='color:#888'>EPS Est: </span>"
                        f"<span style='color:{text_color}'>"
                        f"{'$' + str(round(u['eps_estimate'], 2)) if u.get('eps_estimate') else 'N/A'}"
                        f"</span>"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Build earnings history first to see upcoming dates.")

    # ── TAB 3: Learned Patterns ───────────────────────────────────
    with tab3:
        st.subheader("Learned Reaction Patterns")
        st.markdown("How each stock historically reacts to earnings beats and misses.")

        if not model._patterns:
            st.info("No patterns learned yet. Click 'Learn Reaction Patterns' above.")
        else:
            # Summary chart
            tickers_with_beat = []
            beat_moves = []
            miss_moves = []

            for t, p in sorted(model._patterns.items()):
                beat = p.get("on_beat", {})
                miss = p.get("on_miss", {})
                if beat and miss:
                    tickers_with_beat.append(t)
                    beat_moves.append(beat.get("avg_day1_pct", 0))
                    miss_moves.append(miss.get("avg_day1_pct", 0))

            if tickers_with_beat:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="On Beat", x=tickers_with_beat[:20], y=beat_moves[:20],
                    marker_color=bull,
                ))
                fig.add_trace(go.Bar(
                    name="On Miss", x=tickers_with_beat[:20], y=miss_moves[:20],
                    marker_color=bear,
                ))
                fig.update_layout(
                    title="Average Day-1 Reaction to Earnings",
                    template="plotly_dark",
                    paper_bgcolor=bg, plot_bgcolor=surface,
                    barmode="group",
                    yaxis_title="Day 1 Move %",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Detail cards
            for t, p in sorted(model._patterns.items()):
                drift_rate = p.get("drift_continuation_rate")
                drift_text = f" | Drift continuation: {drift_rate:.0%}" if drift_rate else ""

                st.markdown(
                    f"<div style='background:{surface};padding:12px;border-radius:8px;margin-bottom:6px'>"
                    f"<strong style='color:{primary}'>{t}</strong>"
                    f"<span style='color:#888;margin-left:10px'>"
                    f"{p.get('n_quarters', 0)} quarters | "
                    f"Avg surprise: {p.get('avg_surprise_pct', 0):+.1f}%{drift_text}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

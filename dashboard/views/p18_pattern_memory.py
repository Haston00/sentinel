"""
SENTINEL — Page 18: Cross-Asset Pattern Memory.
Decades of market memory. When gold drops, what follows?
When yields spike, which sectors lead? The system remembers everything.
"""

import streamlit as st
import plotly.graph_objects as go

from config.settings import COLORS


def render():
    st.title("Cross-Asset Pattern Memory")
    st.caption(
        "Decades of market relationships, permanently stored. "
        "The system remembers what happened every time a pattern occurred."
    )

    surface = COLORS["surface"]
    text_color = COLORS["text"]
    bull = COLORS["bull"]
    bear = COLORS["bear"]
    neutral = COLORS["neutral"]
    primary = COLORS["primary"]
    bg = COLORS["background"]

    try:
        from learning.pattern_memory import PatternMemory, ALL_ASSETS
        memory = PatternMemory()
    except Exception as e:
        st.error(f"Pattern memory failed to load: {e}")
        return

    stats = memory.get_memory_stats()

    # ═══════════════════════════════════════════════════════════
    # STATUS BANNER
    # ═══════════════════════════════════════════════════════════
    if stats["history_days"] == 0:
        st.warning("Pattern memory is empty. Click 'Build Memory' to download decades of market data.")
    else:
        st.markdown(
            f"<div style='background:{surface};padding:20px;border-radius:12px;"
            f"border-left:5px solid {primary};margin-bottom:20px'>"
            f"<div style='display:flex;gap:30px;flex-wrap:wrap;align-items:center'>"
            f"<div><span style='font-size:36px;font-weight:700;color:{primary}'>"
            f"{stats['years_of_data']} Years</span>"
            f"<br><span style='color:#888;font-size:12px'>OF MARKET MEMORY</span></div>"
            f"<div style='border-left:1px solid #333;padding-left:20px'>"
            f"<span style='color:{text_color}'>{stats['history_days']:,}</span>"
            f"<span style='color:#888'> trading days</span>"
            f"<span style='color:#555;margin:0 8px'>|</span>"
            f"<span style='color:{text_color}'>{stats['history_assets']}</span>"
            f"<span style='color:#888'> assets tracked</span>"
            f"<span style='color:#555;margin:0 8px'>|</span>"
            f"<span style='color:{text_color}'>{stats.get('unique_patterns', 0):,}</span>"
            f"<span style='color:#888'> learned patterns</span>"
            f"</div></div>"
            f"<p style='color:#555;font-size:11px;margin:8px 0 0'>"
            f"Data: {stats['history_start']} → {stats['history_end']}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Build / Update Controls ────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        years = st.selectbox("History Depth", [5, 10, 15, 20], index=3)
    with c2:
        if st.button("Download History", type="primary"):
            with st.spinner(f"Downloading {years} years of data for {len(ALL_ASSETS)} assets..."):
                new_days = memory.update_history(max_years=years)
            st.success(f"Downloaded {new_days:,} new days of data")
            st.rerun()
    with c3:
        if st.button("Build Pattern Database"):
            with st.spinner("Scanning all history for cross-asset patterns... this takes a minute..."):
                n_patterns = memory.build_pattern_database()
            st.success(f"Built {n_patterns:,} patterns")
            st.rerun()

    if stats["history_days"] == 0:
        st.info(
            "**Step 1:** Click 'Download History' to fetch 20 years of daily prices for all tracked assets.\n\n"
            "**Step 2:** Click 'Build Pattern Database' to analyze every cross-asset relationship.\n\n"
            "**Step 3:** Query the database below to find what historically follows any market move."
        )
        return

    # ═══════════════════════════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4 = st.tabs([
        "Today's Signals", "Query Patterns", "Asset Relationships", "What Predicts?"
    ])

    # ── TAB 1: Today's Triggered Patterns ──────────────────────
    with tab1:
        st.subheader("Patterns Triggered by Today's Market")
        st.markdown("Based on today's moves, here's what history says happens next.")

        window = st.selectbox("Forward Window", list(["1D", "1W", "2W", "1M", "3M"]),
                              index=3, key="today_window")

        triggered = memory.query_current_conditions(window=window, top_n=20)

        if not triggered:
            st.info("No significant patterns triggered today, or pattern database hasn't been built yet.")
        else:
            for p in triggered:
                rel = p["reliability"]
                avg = p["avg_forward_return"]
                direction = "UP" if avg > 0 else "DOWN"
                d_color = bull if avg > 0 else bear

                st.markdown(
                    f"<div style='background:{surface};padding:14px;border-radius:8px;"
                    f"border-left:4px solid {d_color};margin-bottom:8px'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap'>"
                    f"<div>"
                    f"<strong style='color:{text_color}'>{p['trigger_name']}</strong>"
                    f"<span style='color:#888'> {p['move_type'].replace('_', ' ')} "
                    f"({p['triggered_by_return']:+.1%} today)</span>"
                    f"<br><span style='color:#888'>→ </span>"
                    f"<strong style='color:{d_color}'>{p['response_name']}</strong>"
                    f"<span style='color:#888'> tends to go </span>"
                    f"<strong style='color:{d_color}'>{direction} {abs(avg):.2%}</strong>"
                    f"<span style='color:#888'> over {window}</span>"
                    f"</div>"
                    f"<div style='text-align:right'>"
                    f"<span style='font-size:20px;font-weight:700;color:{d_color}'>{rel:.0%}</span>"
                    f"<br><span style='color:#888;font-size:11px'>{p['n_occurrences']} occurrences</span>"
                    f"</div></div></div>",
                    unsafe_allow_html=True,
                )

    # ── TAB 2: Query Patterns ──────────────────────────────────
    with tab2:
        st.subheader("Query Pattern Database")
        st.markdown("Ask: *When [asset] does [move], what happens to [other assets]?*")

        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            trigger = st.selectbox(
                "When this asset moves:",
                list(ALL_ASSETS.keys()),
                format_func=lambda x: f"{x} ({ALL_ASSETS[x]})",
                key="q_trigger",
            )
        with qc2:
            move = st.selectbox(
                "Move type:",
                ["large_drop", "moderate_drop", "small_drop",
                 "small_rally", "moderate_rally", "large_rally"],
                format_func=lambda x: x.replace("_", " ").title(),
                key="q_move",
            )
        with qc3:
            q_window = st.selectbox("Over next:", ["1D", "1W", "2W", "1M", "3M"],
                                    index=3, key="q_window")

        if st.button("Search Patterns", key="search_btn"):
            results = memory.query(
                trigger, move_type=move, window=q_window,
                min_occurrences=5, min_reliability=0.0,
            )

            if not results:
                st.info("No patterns found. Try building the pattern database first.")
            else:
                st.markdown(f"**{len(results)} patterns found** for {ALL_ASSETS.get(trigger, trigger)} "
                            f"{move.replace('_', ' ')} → {q_window} forward")

                # Summary chart
                names = [r["response_name"] for r in results[:15]]
                returns = [r["avg_forward_return"] * 100 for r in results[:15]]
                reliabilities = [r["reliability"] * 100 for r in results[:15]]
                colors = [bull if r > 0 else bear for r in returns]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=names, x=returns, orientation="h",
                    marker_color=colors, name="Avg Forward Return %",
                    text=[f"{r:+.2f}%" for r in returns],
                    textposition="outside",
                ))
                fig.update_layout(
                    title=f"What follows {ALL_ASSETS.get(trigger, trigger)} {move.replace('_', ' ')} ({q_window})",
                    template="plotly_dark",
                    paper_bgcolor=bg,
                    plot_bgcolor=surface,
                    height=max(400, len(names) * 30),
                    xaxis_title="Average Forward Return %",
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Detail table
                for r in results:
                    avg = r["avg_forward_return"]
                    d_color = bull if avg > 0 else bear
                    st.markdown(
                        f"**{r['response_name']}** ({r['response']}): "
                        f"<span style='color:{d_color}'>{avg:+.2%}</span> avg | "
                        f"Up {r['pct_positive']:.0%} / Down {r['pct_negative']:.0%} | "
                        f"Reliability: {r['reliability']:.0%} | "
                        f"N={r['n_occurrences']}",
                        unsafe_allow_html=True,
                    )

    # ── TAB 3: Asset Relationships ─────────────────────────────
    with tab3:
        st.subheader("Two-Asset Relationship")
        st.markdown("See the full historical relationship between any two assets.")

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            asset_a = st.selectbox(
                "Asset A (trigger):",
                list(ALL_ASSETS.keys()),
                format_func=lambda x: f"{x} ({ALL_ASSETS[x]})",
                key="rel_a",
            )
        with rc2:
            asset_b = st.selectbox(
                "Asset B (response):",
                [k for k in ALL_ASSETS.keys() if k != asset_a],
                format_func=lambda x: f"{x} ({ALL_ASSETS[x]})",
                key="rel_b",
            )
        with rc3:
            rel_window = st.selectbox("Window:", ["1D", "1W", "2W", "1M", "3M"],
                                       index=3, key="rel_window")

        if st.button("Show Relationship", key="rel_btn"):
            rel = memory.get_relationship(asset_a, asset_b, window=rel_window)
            patterns = rel.get("patterns", {})

            if not patterns:
                st.info("No pattern data. Build the pattern database first.")
            else:
                st.markdown(
                    f"### {rel['asset_a_name']} → {rel['asset_b_name']} ({rel_window})"
                )

                for move, data in patterns.items():
                    avg = data["avg_response"]
                    d_color = bull if avg > 0 else bear
                    move_label = move.replace("_", " ").title()

                    pct_bar_width = int(data["pct_positive"] * 100)

                    st.markdown(
                        f"<div style='background:{surface};padding:12px;border-radius:8px;"
                        f"margin-bottom:6px;display:flex;justify-content:space-between;align-items:center'>"
                        f"<div style='flex:1'>"
                        f"<strong style='color:{text_color}'>{move_label}</strong>"
                        f"<span style='color:#888'> ({data['n']} times)</span>"
                        f"</div>"
                        f"<div style='flex:1;text-align:center'>"
                        f"<div style='background:#333;border-radius:4px;height:8px;position:relative'>"
                        f"<div style='background:{bull};width:{pct_bar_width}%;height:100%;border-radius:4px'></div>"
                        f"</div>"
                        f"<span style='color:#888;font-size:11px'>Up {data['pct_positive']:.0%} / Down {data['pct_negative']:.0%}</span>"
                        f"</div>"
                        f"<div style='text-align:right;min-width:80px'>"
                        f"<span style='color:{d_color};font-weight:700'>{avg:+.2%}</span>"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )

    # ── TAB 4: What Predicts This Asset? ───────────────────────
    with tab4:
        st.subheader("Predictive Signals")
        st.markdown("What other assets' moves are the best predictors of this one?")

        pc1, pc2 = st.columns(2)
        with pc1:
            target = st.selectbox(
                "I want to predict:",
                list(ALL_ASSETS.keys()),
                format_func=lambda x: f"{x} ({ALL_ASSETS[x]})",
                key="pred_target",
            )
        with pc2:
            pred_window = st.selectbox("Over next:", ["1D", "1W", "2W", "1M", "3M"],
                                       index=3, key="pred_window")

        if st.button("Find Predictors", key="pred_btn"):
            signals = memory.get_strongest_signals_for(target, window=pred_window)

            if not signals:
                st.info("No strong predictors found. Build the pattern database first.")
            else:
                st.markdown(f"**Top predictors of {ALL_ASSETS.get(target, target)} ({pred_window}):**")

                for s in signals:
                    avg = s["avg_forward_return"]
                    d_color = bull if avg > 0 else bear

                    st.markdown(
                        f"<div style='background:{surface};padding:12px;border-radius:8px;"
                        f"border-left:3px solid {d_color};margin-bottom:6px'>"
                        f"When <strong style='color:{primary}'>{s['trigger_name']}</strong>"
                        f" has a <strong>{s['move_type'].replace('_', ' ')}</strong>"
                        f" → {ALL_ASSETS.get(target, target)} goes "
                        f"<strong style='color:{d_color}'>{'+' if avg > 0 else ''}{avg:.2%}</strong>"
                        f" over {pred_window}"
                        f"<span style='color:#888'> | Reliability: {s['reliability']:.0%}"
                        f" | N={s['n_occurrences']}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

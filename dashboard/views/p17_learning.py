"""
SENTINEL — Page 17: Learning Engine Dashboard.
Shows how the system learns from its predictions vs actuals.
Tracks accuracy, bias corrections, regime performance, confidence calibration.
"""

import streamlit as st
import plotly.graph_objects as go

from config.settings import COLORS


def render():
    st.title("Learning Engine")
    st.caption(
        "Every prediction is tracked. Every actual result is scored. "
        "The system learns from its mistakes and gets smarter over time."
    )

    surface = COLORS["surface"]
    text_color = COLORS["text"]
    bull = COLORS["bull"]
    bear = COLORS["bear"]
    neutral = COLORS["neutral"]
    primary = COLORS["primary"]
    bg = COLORS["background"]

    # ── Initialize Learning System ─────────────────────────────
    try:
        from learning.feedback_loop import FeedbackLoop
        loop = FeedbackLoop()
    except Exception as e:
        st.error(f"Learning engine failed to initialize: {e}")
        return

    # ── Run Learning Cycle Button ──────────────────────────────
    col_run, col_status = st.columns([1, 3])
    with col_run:
        if st.button("Run Learning Cycle", type="primary"):
            with st.spinner("Scoring matured predictions and optimizing..."):
                result = loop.run_cycle()

            st.success(
                f"Cycle complete: {result['newly_scored']} new scores, "
                f"{result['total_scored']} total, hit rate: {result['hit_rate']}"
            )

    # ── Get Dashboard Data ─────────────────────────────────────
    data = loop.get_dashboard_data()
    accuracy = data["accuracy"]
    scored = data["scored_predictions"]
    pending = data["pending_predictions"]
    learning = data["learning_summary"]

    # ═══════════════════════════════════════════════════════════
    # OVERVIEW BANNER
    # ═══════════════════════════════════════════════════════════
    total = accuracy.get("total_predictions", 0)
    n_scored = accuracy.get("scored", 0)
    n_pending = accuracy.get("pending", 0)
    hit_rate = accuracy.get("hit_rate", 0)
    hit_pct = accuracy.get("hit_rate_pct", "N/A")
    ci_rate = accuracy.get("ci_capture_rate", 0)

    if total == 0:
        st.info(
            "No predictions logged yet. The learning engine starts tracking "
            "when you generate forecasts from the AI Forecast, Probability Forecast, "
            "or Stock Explorer pages. Each prediction is stored, and when the target "
            "date arrives, the actual result is compared to what SENTINEL predicted."
        )
        st.markdown(
            f"<div style='background:{surface};padding:20px;border-radius:12px;"
            f"border-left:5px solid {primary};margin:20px 0'>"
            f"<h3 style='color:{primary};margin:0'>How It Works</h3>"
            f"<ol style='color:{text_color};margin:10px 0;padding-left:20px'>"
            f"<li>Generate forecasts from any page (AI Forecast, Stock Explorer, etc.)</li>"
            f"<li>Each prediction is logged with direction, confidence, and regime</li>"
            f"<li>When the target date passes, SENTINEL fetches the actual price</li>"
            f"<li>Predictions are scored: was the direction right? How close was the estimate?</li>"
            f"<li>The optimizer learns: adjusts model weights, corrects biases, calibrates confidence</li>"
            f"<li>Next time you forecast, the system applies everything it learned</li>"
            f"</ol>"
            f"<p style='color:#888;font-size:13px;margin:0'>"
            f"Uses: exponential decay weighting, Bayesian reliability updating, "
            f"regime-conditional calibration, Kelly criterion conviction scoring, "
            f"adversarial self-testing, and drawdown-aware confidence scaling.</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    # Hit rate color
    if hit_rate >= 0.6:
        hr_color = bull
    elif hit_rate >= 0.45:
        hr_color = neutral
    else:
        hr_color = bear

    st.markdown(
        f"<div style='background:{surface};padding:20px;border-radius:12px;"
        f"border-left:5px solid {hr_color};margin-bottom:20px'>"
        f"<div style='display:flex;align-items:center;gap:20px;flex-wrap:wrap'>"
        f"<div><span style='font-size:42px;font-weight:700;color:{hr_color}'>{hit_pct}</span>"
        f"<br><span style='color:#888;font-size:12px'>DIRECTION ACCURACY</span></div>"
        f"<div style='border-left:1px solid #333;padding-left:20px'>"
        f"<span style='color:{text_color}'>{n_scored}</span><span style='color:#888'> scored</span>"
        f"<span style='color:#555;margin:0 8px'>|</span>"
        f"<span style='color:{neutral}'>{n_pending}</span><span style='color:#888'> pending</span>"
        f"<span style='color:#555;margin:0 8px'>|</span>"
        f"<span style='color:#888'>CI capture: {ci_rate:.0%}</span>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    # ═══════════════════════════════════════════════════════════
    # KEY METRICS ROW
    # ═══════════════════════════════════════════════════════════
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Predictions", total)
    m2.metric("Scored", n_scored)
    m3.metric("Hit Rate", hit_pct)
    m4.metric("Avg Error", f"{accuracy.get('avg_abs_error', 0):.4f}")
    m5.metric("CI Capture", f"{ci_rate:.0%}")

    # ═══════════════════════════════════════════════════════════
    # TABS: ACCURACY | CALIBRATION | RELIABILITY | ALERTS | HISTORY
    # ═══════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Accuracy Breakdown", "Confidence Calibration",
        "System Reliability", "Health Alerts", "Prediction Log"
    ])

    # ── TAB 1: Accuracy Breakdown ──────────────────────────────
    with tab1:
        # By regime
        by_regime = accuracy.get("by_regime", {})
        if by_regime:
            st.subheader("Accuracy by Market Regime")
            regime_cols = st.columns(len(by_regime))
            for col, (regime_name, rdata) in zip(regime_cols, by_regime.items()):
                r_hr = rdata.get("hit_rate", 0)
                r_color = bull if r_hr >= 0.55 else (bear if r_hr < 0.4 else neutral)
                col.markdown(
                    f"<div style='background:{surface};padding:15px;border-radius:8px;"
                    f"text-align:center;border-top:3px solid {r_color}'>"
                    f"<h4 style='margin:0;color:{r_color}'>{regime_name}</h4>"
                    f"<p style='font-size:28px;font-weight:700;color:{r_color};margin:5px 0'>{r_hr:.0%}</p>"
                    f"<p style='color:#888;font-size:12px;margin:0'>{rdata.get('count', 0)} predictions</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # By horizon
        by_horizon = accuracy.get("by_horizon", {})
        if by_horizon:
            st.subheader("Accuracy by Forecast Horizon")
            h_cols = st.columns(len(by_horizon))
            for col, (h_name, hdata) in zip(h_cols, by_horizon.items()):
                h_hr = hdata.get("hit_rate", 0)
                h_color = bull if h_hr >= 0.55 else (bear if h_hr < 0.4 else neutral)
                col.metric(h_name, f"{h_hr:.0%}", f"n={hdata.get('count', 0)}")

        # By asset type
        by_asset = accuracy.get("by_asset_type", {})
        if by_asset:
            st.subheader("Accuracy by Asset Type")
            for atype, adata in by_asset.items():
                a_hr = adata.get("hit_rate", 0)
                a_color = bull if a_hr >= 0.55 else bear
                st.markdown(
                    f"**{atype.title()}**: {a_hr:.0%} ({adata.get('count', 0)} predictions, "
                    f"avg error: {adata.get('avg_error', 0):.4f})"
                )

    # ── TAB 2: Confidence Calibration ──────────────────────────
    with tab2:
        calibration = accuracy.get("calibration", {})
        if calibration:
            st.subheader("Confidence Calibration")
            st.markdown(
                "Are we calibrated? When we say 70% confident, are we right 70% of the time?"
            )

            # Calibration chart
            cal_labels = []
            predicted_vals = []
            actual_vals = []
            for bucket, cdata in calibration.items():
                cal_labels.append(bucket)
                predicted_vals.append(cdata.get("avg_confidence", 0))
                actual_vals.append(cdata.get("actual_accuracy", 0))

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=cal_labels, y=predicted_vals,
                name="Predicted Confidence",
                marker_color=primary, opacity=0.6,
            ))
            fig.add_trace(go.Bar(
                x=cal_labels, y=actual_vals,
                name="Actual Accuracy",
                marker_color=bull,
            ))
            fig.add_trace(go.Scatter(
                x=cal_labels, y=cal_labels,
                name="Perfect Calibration",
                line=dict(color="#888", dash="dash"),
                mode="lines",
                visible="legendonly",
            ))
            fig.update_layout(
                title="Confidence Calibration",
                template="plotly_dark",
                paper_bgcolor=bg,
                plot_bgcolor=surface,
                height=400,
                barmode="group",
                yaxis_range=[0, 1],
            )
            st.plotly_chart(fig, use_container_width=True)

            for bucket, cdata in calibration.items():
                gap = cdata.get("avg_confidence", 0) - cdata.get("actual_accuracy", 0)
                status = "OVERCONFIDENT" if gap > 0.1 else ("UNDERCONFIDENT" if gap < -0.1 else "CALIBRATED")
                status_color = bear if status == "OVERCONFIDENT" else (bull if status == "UNDERCONFIDENT" else "#888")
                st.markdown(
                    f"**{bucket}**: predicted {cdata.get('avg_confidence', 0):.0%}, "
                    f"actual {cdata.get('actual_accuracy', 0):.0%} — "
                    f"<span style='color:{status_color}'>{status}</span> "
                    f"({cdata.get('count', 0)} predictions)",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Not enough scored predictions for calibration analysis (need 3+ per confidence bucket)")

    # ── TAB 3: System Reliability ──────────────────────────────
    with tab3:
        reliability = learning.get("model_reliability", {})
        system = reliability.get("system", {})

        if system:
            st.subheader("Bayesian System Reliability")

            r_mean = system.get("reliability_mean", 0.5)
            r_ci_low = system.get("ci_95_lower", 0)
            r_ci_high = system.get("ci_95_upper", 1)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=r_mean * 100,
                title=dict(text="System Reliability", font=dict(size=16)),
                number=dict(suffix="%", font=dict(size=36)),
                gauge=dict(
                    axis=dict(range=[0, 100]),
                    bar=dict(color=bull if r_mean > 0.55 else (bear if r_mean < 0.45 else neutral)),
                    bgcolor=surface,
                    threshold=dict(line=dict(color="white", width=2), value=50),
                ),
            ))
            fig.update_layout(
                paper_bgcolor=bg,
                font=dict(color=text_color),
                height=250,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                f"**Bayesian estimate**: {r_mean:.1%} "
                f"(95% credible interval: {r_ci_low:.1%} – {r_ci_high:.1%})"
            )
            st.markdown(
                f"Based on {system.get('n', 0)} scored predictions. "
                f"Alpha={system.get('alpha', 0):.0f}, Beta={system.get('beta', 0):.0f}"
            )

        # Bias corrections
        bias = learning.get("bias_corrections", {})
        if bias:
            st.subheader("Learned Bias Corrections")
            overall = bias.get("overall", {})
            if overall:
                bias_val = overall.get("weighted_bias", 0)
                corr_val = overall.get("correction", 0)
                momentum = overall.get("momentum", 0)

                st.markdown(
                    f"**Overall bias**: {bias_val:+.4%} "
                    f"(correction applied: {corr_val:+.4%}, "
                    f"momentum: {momentum:.0%})"
                )

            for key, bdata in bias.items():
                if key != "overall" and isinstance(bdata, dict):
                    st.markdown(
                        f"**{key}**: bias={bdata.get('bias', 0):+.4%}, "
                        f"correction={bdata.get('correction', 0):+.4%} "
                        f"(n={bdata.get('n', 'N/A')})"
                    )

        # Learning progress chart
        learning_curve = data.get("learning_curve")
        if learning_curve is not None and not learning_curve.empty:
            st.subheader("Learning Progress")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=learning_curve["timestamp"],
                y=learning_curve["rolling_hit_rate"],
                name="Rolling Hit Rate",
                line=dict(color=bull, width=2),
            ))
            fig.add_hline(y=0.5, line_dash="dash", line_color="#555",
                          annotation_text="Random (50%)")
            fig.update_layout(
                title="Rolling Accuracy Over Time (20-prediction window)",
                template="plotly_dark",
                paper_bgcolor=bg,
                plot_bgcolor=surface,
                height=350,
                yaxis_range=[0, 1],
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 4: Health Alerts ───────────────────────────────────
    with tab4:
        st.subheader("Adversarial Self-Test")
        st.markdown("The system tests itself for problems and degradation")

        if n_scored >= 10:
            with st.spinner("Running adversarial tests..."):
                from learning.optimizer import LearningOptimizer
                optimizer = LearningOptimizer()
                test_results = optimizer.optimize(scored)

            adversarial = test_results.get("adversarial", {})
            alerts = adversarial.get("alerts", [])
            healthy = adversarial.get("system_healthy", True)

            if healthy:
                st.success("System is healthy — no critical alerts detected")
            else:
                st.error("System health issues detected")

            for alert in alerts:
                severity = alert["severity"]
                s_color = bear if severity == "HIGH" else (neutral if severity == "MEDIUM" else "#888")
                st.markdown(
                    f"<div style='background:{surface};padding:12px;border-radius:8px;"
                    f"border-left:4px solid {s_color};margin-bottom:8px'>"
                    f"<strong style='color:{s_color}'>[{severity}] {alert['type']}</strong><br>"
                    f"<span style='color:{text_color}'>{alert['detail']}</span><br>"
                    f"<span style='color:#888;font-size:12px'>Action: {alert['action']}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Kelly criterion
            kelly = test_results.get("kelly", {})
            if kelly and kelly.get("edge_exists") is not None:
                st.subheader("Kelly Criterion Analysis")
                edge = kelly.get("edge_exists", False)
                k_color = bull if edge else bear

                st.markdown(
                    f"<div style='background:{surface};padding:15px;border-radius:8px;"
                    f"border-left:4px solid {k_color}'>"
                    f"<h4 style='color:{k_color};margin:0'>{'EDGE EXISTS' if edge else 'NO EDGE DETECTED'}</h4>"
                    f"<p style='color:{text_color};margin:5px 0'>"
                    f"Win rate: {kelly.get('win_rate', 0):.1%} | "
                    f"Win/Loss ratio: {kelly.get('win_loss_ratio', 0):.2f} | "
                    f"Full Kelly: {kelly.get('full_kelly', 0):.1%} | "
                    f"Half Kelly: {kelly.get('half_kelly', 0):.1%}"
                    f"</p></div>",
                    unsafe_allow_html=True,
                )

            # Drawdown analysis
            drawdown = test_results.get("drawdown", {})
            if drawdown:
                st.subheader("Streak Analysis")
                d1, d2, d3 = st.columns(3)
                streak_type = drawdown.get("current_streak_type", "N/A")
                streak_color = bull if streak_type == "winning" else bear
                d1.metric("Current Streak",
                          f"{drawdown.get('current_streak', 0)} {streak_type}")
                d2.metric("Max Losing Streak", drawdown.get("max_losing_streak", 0))
                d3.metric("Recent Accuracy (20)",
                          f"{drawdown.get('recent_accuracy_20', 0):.0%}")
        else:
            st.info(f"Need at least 10 scored predictions for health analysis (have {n_scored})")

    # ── TAB 5: Prediction Log ──────────────────────────────────
    with tab5:
        st.subheader("Recent Predictions")

        view_mode = st.radio("View", ["Scored", "Pending", "All"], horizontal=True)

        if view_mode == "Scored" and not scored.empty:
            display = scored.sort_values("timestamp", ascending=False).head(50)
            display_cols = [
                "timestamp", "asset_ticker", "horizon",
                "predicted_direction", "actual_direction", "correct_direction",
                "confidence", "predicted_return", "actual_return", "abs_error",
                "regime",
            ]
            existing_cols = [c for c in display_cols if c in display.columns]
            st.dataframe(display[existing_cols], use_container_width=True)
        elif view_mode == "Pending" and not pending.empty:
            display = pending.sort_values("maturity_date").head(50)
            display_cols = [
                "timestamp", "asset_ticker", "horizon", "maturity_date",
                "predicted_direction", "confidence", "predicted_return", "regime",
            ]
            existing_cols = [c for c in display_cols if c in display.columns]
            st.dataframe(display[existing_cols], use_container_width=True)
        elif view_mode == "All":
            all_preds = loop.tracker.get_all_predictions()
            if not all_preds.empty:
                st.dataframe(
                    all_preds.sort_values("timestamp", ascending=False).head(100),
                    use_container_width=True,
                )
            else:
                st.info("No predictions logged yet")
        else:
            st.info("No data for this view")

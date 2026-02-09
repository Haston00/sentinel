"""
SENTINEL — Page 22: Position Sizing & Risk Management.
Kelly-based position sizing, portfolio VaR, stress testing,
correlation analysis, and drawdown monitoring.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from config.settings import COLORS


def render():
    st.title("Position Sizing & Risk Management")
    st.caption(
        "Institutional-grade position sizing with Kelly criterion, volatility targeting, "
        "and risk limits. Plus portfolio VaR, stress testing, and drawdown monitoring."
    )

    surface = COLORS["surface"]
    text_color = COLORS["text"]
    bull = COLORS["bull"]
    bear = COLORS["bear"]
    primary = COLORS["primary"]
    bg = COLORS["background"]
    neutral = COLORS["neutral"]

    # ═══════════════════════════════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Position Sizer", "Portfolio Risk (VaR)", "Stress Test",
        "Correlation Matrix", "Risk Config"
    ])

    # ── TAB 1: Position Sizer ─────────────────────────────────────
    with tab1:
        st.subheader("Kelly-Based Position Sizer")
        st.markdown(
            "Enter a forecast and get optimal position size using three independent methods. "
            "The system takes the most conservative (smallest) recommendation."
        )

        pc1, pc2 = st.columns(2)
        with pc1:
            ps_ticker = st.text_input("Ticker", value="SPY", key="ps_ticker")
            ps_direction = st.selectbox(
                "Direction", ["Long (Bullish)", "Short (Bearish)"], key="ps_direction"
            )
            ps_confidence = st.slider(
                "Forecast Confidence", 0.0, 1.0, 0.65, 0.05, key="ps_conf"
            )
        with pc2:
            ps_expected = st.number_input(
                "Expected Return (%)", value=3.0, step=0.5, key="ps_expected"
            )
            ps_downside = st.number_input(
                "Max Downside (%)", value=-5.0, step=0.5, key="ps_downside"
            )
            ps_upside = st.number_input(
                "Max Upside (%)", value=8.0, step=0.5, key="ps_upside"
            )

        if st.button("Calculate Position Size", type="primary", key="ps_calc"):
            try:
                from risk.position_sizing import PositionSizer
                sizer = PositionSizer()

                forecast = {
                    "confidence": ps_confidence,
                    "point": ps_expected / 100,
                    "direction": 1 if "Long" in ps_direction else -1,
                    "lower": ps_downside / 100,
                    "upper": ps_upside / 100,
                }

                result = sizer.size_position(ps_ticker, forecast)

                action = result.get("action", "NO_TRADE")
                a_color = bull if action == "BUY" else (bear if action == "SHORT" else neutral)

                st.markdown(
                    f"<div style='background:{surface};padding:20px;border-radius:12px;"
                    f"border-left:5px solid {a_color};margin-bottom:15px'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap'>"
                    f"<div>"
                    f"<span style='font-size:14px;color:#888'>RECOMMENDATION</span><br>"
                    f"<span style='font-size:32px;font-weight:700;color:{a_color}'>"
                    f"{action}</span>"
                    f"</div>"
                    f"<div style='text-align:right'>"
                    f"<span style='font-size:28px;font-weight:700;color:{text_color}'>"
                    f"{result.get('position_pct', 0):.1%}</span>"
                    f"<br><span style='color:#888'>of portfolio</span>"
                    f"<br><span style='color:{primary};font-size:20px'>"
                    f"${result.get('position_value', 0):,.0f}</span>"
                    f"</div></div></div>",
                    unsafe_allow_html=True,
                )

                if result.get("shares_approx"):
                    st.markdown(
                        f"**Approximate shares:** {result['shares_approx']} "
                        f"@ ${result.get('current_price', 0):,.2f}"
                    )

                # Method breakdown
                st.markdown("**Sizing Methods (took the smallest):**")
                methods = {
                    "Kelly": result.get("kelly_pct", 0),
                    "Volatility": result.get("vol_pct", 0),
                    "Risk Limit": result.get("limit_pct", 0),
                }

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(methods.keys()),
                    y=[v * 100 for v in methods.values()],
                    marker_color=[
                        primary if result.get("sizing_method") == k.lower()
                        else "#444"
                        for k in methods.keys()
                    ],
                    text=[f"{v:.1%}" for v in methods.values()],
                    textposition="outside",
                ))
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor=bg, plot_bgcolor=surface,
                    yaxis_title="Position Size %", height=300,
                    title="Three Independent Sizing Methods",
                )
                st.plotly_chart(fig, use_container_width=True)

                if result.get("reason"):
                    st.info(result["reason"])

            except Exception as e:
                st.error(f"Position sizing failed: {e}")

    # ── TAB 2: Portfolio VaR ──────────────────────────────────────
    with tab2:
        st.subheader("Value at Risk (VaR)")
        st.markdown(
            "Three VaR methods: Historical, Parametric (Normal), and Monte Carlo. "
            "Uses the most conservative estimate."
        )

        vc1, vc2 = st.columns(2)
        with vc1:
            var_tickers = st.text_input(
                "Portfolio Tickers (comma-separated)",
                value="SPY,QQQ,AAPL,GLD,TLT",
                key="var_tickers",
            )
            var_weights = st.text_input(
                "Weights (comma-separated, should sum to 1)",
                value="0.3,0.2,0.2,0.15,0.15",
                key="var_weights",
            )
        with vc2:
            var_conf = st.selectbox(
                "Confidence Level", [0.95, 0.99], key="var_conf"
            )
            var_horizon = st.selectbox(
                "Horizon (days)", [1, 5, 10, 21], key="var_horizon"
            )

        if st.button("Calculate VaR", type="primary", key="var_calc"):
            tickers = [t.strip() for t in var_tickers.split(",") if t.strip()]
            weights = [float(w.strip()) for w in var_weights.split(",") if w.strip()]

            if len(tickers) != len(weights):
                st.error("Number of tickers must match number of weights")
            else:
                try:
                    from risk.risk_manager import RiskManager
                    rm = RiskManager(portfolio_value=100_000)

                    with st.spinner("Computing VaR (3 methods + Monte Carlo simulation)..."):
                        var_result = rm.compute_var(
                            tickers, weights, var_conf, var_horizon
                        )

                    if "error" in var_result:
                        st.error(var_result["error"])
                    else:
                        # VaR Banner
                        max_var = var_result.get("conservative_var", 0)
                        var_dollars = var_result.get("var_dollars", 0)
                        var_color = bear if max_var > 0.03 else (neutral if max_var > 0.015 else bull)

                        st.markdown(
                            f"<div style='background:{surface};padding:20px;border-radius:12px;"
                            f"border-left:5px solid {var_color};margin-bottom:15px'>"
                            f"<span style='font-size:14px;color:#888'>"
                            f"{var_conf:.0%} VaR ({var_horizon}-DAY)</span><br>"
                            f"<span style='font-size:36px;font-weight:700;color:{var_color}'>"
                            f"${var_dollars:,.0f}</span>"
                            f"<span style='color:#888;margin-left:15px'>"
                            f"({max_var:.2%} of portfolio)</span>"
                            f"<br><span style='color:#888;font-size:12px'>"
                            f"{var_result.get('interpretation', '')}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        # Three methods comparison
                        mc1, mc2, mc3 = st.columns(3)
                        with mc1:
                            st.metric(
                                "Historical VaR",
                                f"{var_result.get('historical_var', 0):.2%}",
                            )
                        with mc2:
                            st.metric(
                                "Parametric VaR",
                                f"{var_result.get('parametric_var', 0):.2%}",
                            )
                        with mc3:
                            st.metric(
                                "Monte Carlo VaR",
                                f"{var_result.get('monte_carlo_var', 0):.2%}",
                            )

                        cvar = var_result.get("cvar", 0)
                        st.markdown(
                            f"**Conditional VaR (Expected Shortfall):** "
                            f"<span style='color:{bear}'>{cvar:.2%}</span> "
                            f"(${var_result.get('cvar_dollars', 0):,.0f})",
                            unsafe_allow_html=True,
                        )

                except Exception as e:
                    st.error(f"VaR calculation failed: {e}")

    # ── TAB 3: Stress Test ────────────────────────────────────────
    with tab3:
        st.subheader("Historical Stress Test")
        st.markdown(
            "How would your portfolio have performed during major market crises?"
        )

        st_tickers = st.text_input(
            "Portfolio Tickers", value="SPY,QQQ,AAPL,GLD,TLT", key="st_tickers",
        )
        st_weights = st.text_input(
            "Weights", value="0.3,0.2,0.2,0.15,0.15", key="st_weights",
        )

        if st.button("Run Stress Test", type="primary", key="st_run"):
            tickers = [t.strip() for t in st_tickers.split(",") if t.strip()]
            weights = [float(w.strip()) for w in st_weights.split(",") if w.strip()]

            try:
                from risk.risk_manager import RiskManager
                rm = RiskManager(portfolio_value=100_000)

                with st.spinner("Running stress tests against historical crises..."):
                    stress = rm.stress_test(tickers, weights)

                scenarios = stress.get("scenarios", {})
                if not scenarios:
                    st.warning("No stress test data available.")
                else:
                    for name, data in scenarios.items():
                        port_ret = data.get("portfolio_return", 0)
                        d_color = bull if port_ret > 0 else bear
                        loss = data.get("portfolio_loss_dollars", 0)

                        st.markdown(
                            f"<div style='background:{surface};padding:14px;border-radius:8px;"
                            f"border-left:4px solid {d_color};margin-bottom:8px'>"
                            f"<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap'>"
                            f"<div><strong style='color:{text_color}'>{name}</strong></div>"
                            f"<div style='text-align:right'>"
                            f"<span style='font-size:20px;font-weight:700;color:{d_color}'>"
                            f"{port_ret:+.1%}</span>"
                            f"<br><span style='color:#888;font-size:12px'>"
                            f"${loss:+,.0f}</span>"
                            f"</div></div>"
                            f"<div style='color:#888;font-size:12px;margin-top:5px'>"
                            f"Worst: {data.get('worst_asset', '?')} ({data.get('worst_return', 0):+.1%}) | "
                            f"Best: {data.get('best_asset', '?')} ({data.get('best_return', 0):+.1%})"
                            f"</div></div>",
                            unsafe_allow_html=True,
                        )

                    avg_loss = stress.get("avg_crisis_loss", 0)
                    st.markdown(
                        f"**Average crisis loss:** <span style='color:{bear}'>"
                        f"{avg_loss:+.1%}</span>",
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.error(f"Stress test failed: {e}")

    # ── TAB 4: Correlation Matrix ─────────────────────────────────
    with tab4:
        st.subheader("Portfolio Correlation Matrix")
        st.markdown(
            "High correlation = concentrated risk. Good portfolios have low average correlation."
        )

        corr_tickers = st.text_input(
            "Tickers to analyze", value="SPY,QQQ,AAPL,GLD,TLT,BTC-USD", key="corr_tickers"
        )

        if st.button("Compute Correlations", type="primary", key="corr_calc"):
            tickers = [t.strip() for t in corr_tickers.split(",") if t.strip()]

            try:
                from risk.risk_manager import RiskManager
                rm = RiskManager()

                with st.spinner("Computing correlation matrix..."):
                    corr = rm.correlation_matrix(tickers)

                if "error" in corr:
                    st.error(corr["error"])
                else:
                    # Assessment
                    div_score = corr.get("diversification_score", 0)
                    assessment = corr.get("risk_assessment", "")
                    a_color = bull if "GOOD" in assessment else (neutral if "MODERATE" in assessment else bear)

                    st.markdown(
                        f"<div style='background:{surface};padding:15px;border-radius:8px;"
                        f"border-left:4px solid {a_color};margin-bottom:15px'>"
                        f"<strong style='color:{a_color}'>{assessment}</strong>"
                        f"<span style='color:#888;margin-left:10px'>"
                        f"Avg correlation: {corr.get('avg_correlation', 0):.3f} | "
                        f"Diversification score: {div_score:.3f}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # Heatmap
                    matrix = corr.get("matrix", {})
                    if matrix:
                        labels = list(matrix.keys())
                        values = [[matrix[r][c] for c in labels] for r in labels]

                        fig = go.Figure(data=go.Heatmap(
                            z=values, x=labels, y=labels,
                            colorscale="RdBu_r", zmid=0,
                            text=[[f"{v:.2f}" for v in row] for row in values],
                            texttemplate="%{text}",
                        ))
                        fig.update_layout(
                            title="Correlation Matrix",
                            template="plotly_dark",
                            paper_bgcolor=bg, plot_bgcolor=surface,
                            height=500,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # High correlation pairs
                    pairs = corr.get("high_correlation_pairs", [])
                    if pairs:
                        st.markdown("**High Correlation Pairs (>0.70):**")
                        for p in pairs:
                            p_color = bear if p["risk"] == "high" else neutral
                            st.markdown(
                                f"- <span style='color:{p_color}'>"
                                f"{p['pair']}: {p['correlation']:.3f}</span>"
                                f" ({p['risk']} risk)",
                                unsafe_allow_html=True,
                            )

            except Exception as e:
                st.error(f"Correlation analysis failed: {e}")

    # ── TAB 5: Risk Configuration ─────────────────────────────────
    with tab5:
        st.subheader("Risk Configuration")
        st.markdown("Set your portfolio size and risk limits.")

        try:
            from risk.position_sizing import PositionSizer
            sizer = PositionSizer()
            config = sizer.get_config()
        except Exception:
            config = {}

        if config:
            rc1, rc2 = st.columns(2)
            with rc1:
                new_portfolio = st.number_input(
                    "Portfolio Value ($)", value=int(config.get("total_portfolio_value", 100_000)),
                    step=10_000, key="rc_portfolio",
                )
                new_max_single = st.slider(
                    "Max Single Position %", 1, 25,
                    int(config.get("max_single_position_pct", 0.10) * 100),
                    key="rc_single",
                )
                new_max_sector = st.slider(
                    "Max Sector Exposure %", 10, 50,
                    int(config.get("max_sector_exposure_pct", 0.30) * 100),
                    key="rc_sector",
                )
            with rc2:
                new_vol_target = st.slider(
                    "Annual Vol Target %", 5, 40,
                    int(config.get("vol_target_annual", 0.15) * 100),
                    key="rc_vol",
                )
                new_kelly = st.slider(
                    "Kelly Fraction", 0.1, 1.0,
                    float(config.get("kelly_fraction", 0.5)),
                    0.05, key="rc_kelly",
                )
                new_max_loss = st.slider(
                    "Max Daily Loss %", 1, 10,
                    int(config.get("max_daily_loss_pct", 0.02) * 100),
                    key="rc_loss",
                )

            if st.button("Save Configuration", key="rc_save"):
                new_config = {
                    "total_portfolio_value": new_portfolio,
                    "max_single_position_pct": new_max_single / 100,
                    "max_sector_exposure_pct": new_max_sector / 100,
                    "vol_target_annual": new_vol_target / 100,
                    "kelly_fraction": new_kelly,
                    "max_daily_loss_pct": new_max_loss / 100,
                }
                sizer.save_config(new_config)
                st.success("Risk configuration saved!")

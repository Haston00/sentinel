"""
SENTINEL — Reusable Plotly chart components.
Consistent styling across the dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import COLORS


def _base_layout(title: str = "", height: int = 500) -> dict:
    """Standard layout settings for all charts."""
    return dict(
        title=dict(text=title, font=dict(size=18, color=COLORS["text"])),
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        height=height,
        margin=dict(l=60, r=30, t=60, b=40),
        font=dict(color=COLORS["text"]),
        xaxis=dict(gridcolor="#333", zerolinecolor="#555"),
        yaxis=dict(gridcolor="#333", zerolinecolor="#555"),
    )


def candlestick_chart(
    ohlcv: pd.DataFrame,
    title: str = "",
    volume: bool = True,
    sma_periods: list[int] | None = None,
) -> go.Figure:
    """Create a candlestick chart with optional volume and moving averages."""
    if volume:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
        )
    else:
        fig = go.Figure()

    row = 1 if volume else None

    fig.add_trace(
        go.Candlestick(
            x=ohlcv.index,
            open=ohlcv["Open"],
            high=ohlcv["High"],
            low=ohlcv["Low"],
            close=ohlcv["Close"],
            name="Price",
            increasing_line_color=COLORS["bull"],
            decreasing_line_color=COLORS["bear"],
        ),
        row=row, col=1 if volume else None,
    )

    # Moving averages
    sma_periods = sma_periods or [20, 50, 200]
    sma_colors = ["#FFD600", "#FF9100", "#2962FF"]
    for i, period in enumerate(sma_periods):
        if len(ohlcv) >= period:
            sma = ohlcv["Close"].rolling(period).mean()
            fig.add_trace(
                go.Scatter(
                    x=ohlcv.index, y=sma,
                    name=f"SMA {period}",
                    line=dict(width=1, color=sma_colors[i % len(sma_colors)]),
                ),
                row=row, col=1 if volume else None,
            )

    # Volume
    if volume and "Volume" in ohlcv.columns:
        colors = [
            COLORS["bull"] if c >= o else COLORS["bear"]
            for o, c in zip(ohlcv["Open"], ohlcv["Close"])
        ]
        fig.add_trace(
            go.Bar(x=ohlcv.index, y=ohlcv["Volume"], name="Volume", marker_color=colors, opacity=0.5),
            row=2, col=1,
        )

    fig.update_layout(**_base_layout(title, height=600))
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def forecast_cone_chart(
    historical: pd.Series,
    forecast: dict,
    horizon_name: str = "1M",
    title: str = "",
) -> go.Figure:
    """
    Plot price history with forward forecast cone (confidence interval).
    """
    fig = go.Figure()

    # Historical price
    fig.add_trace(go.Scatter(
        x=historical.index, y=historical.values,
        name="Historical", line=dict(color=COLORS["primary"], width=2),
    ))

    # Forecast cone
    fc = forecast.get(horizon_name, {})
    if fc and "point" in fc:
        last_price = historical.iloc[-1]
        last_date = historical.index[-1]

        # Generate future dates
        horizon_days = {"1W": 5, "1M": 21, "3M": 63}.get(horizon_name, 21)
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days)

        point_return = fc["point"]
        lower_return = fc.get("lower", point_return - 0.05)
        upper_return = fc.get("upper", point_return + 0.05)

        # Interpolate the cone
        t = np.linspace(0, 1, horizon_days)
        point_prices = last_price * (1 + point_return * t)
        lower_prices = last_price * (1 + lower_return * t)
        upper_prices = last_price * (1 + upper_return * t)

        # Confidence band
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(upper_prices) + list(lower_prices[::-1]),
            fill="toself",
            fillcolor="rgba(41, 98, 255, 0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI",
        ))

        # Point forecast line
        fig.add_trace(go.Scatter(
            x=future_dates, y=point_prices,
            name="Forecast",
            line=dict(color=COLORS["primary"], width=2, dash="dash"),
        ))

    fig.update_layout(**_base_layout(title or f"Forecast Cone ({horizon_name})", height=450))
    return fig


def regime_timeline_chart(regime_df: pd.DataFrame, title: str = "Market Regime History") -> go.Figure:
    """Plot regime history as a colored timeline."""
    fig = go.Figure()

    regime_colors = {
        "Bull": COLORS["bull"],
        "Bear": COLORS["bear"],
        "Transition": COLORS["transition"],
    }

    if "Regime_Label" in regime_df.columns:
        for regime_name, color in regime_colors.items():
            mask = regime_df["Regime_Label"] == regime_name
            subset = regime_df[mask]
            if not subset.empty:
                fig.add_trace(go.Scatter(
                    x=subset.index,
                    y=[regime_name] * len(subset),
                    mode="markers",
                    marker=dict(color=color, size=3),
                    name=regime_name,
                ))

    # Probability traces
    for col in regime_df.columns:
        if col.startswith("Prob_"):
            label = col.replace("Prob_", "")
            fig.add_trace(go.Scatter(
                x=regime_df.index,
                y=regime_df[col],
                name=f"{label} Prob",
                line=dict(width=1.5, color=regime_colors.get(label, "#888")),
                visible="legendonly",
            ))

    fig.update_layout(**_base_layout(title, height=350))
    return fig


def sentiment_heatmap(sector_sentiment: pd.DataFrame, title: str = "Sector Sentiment") -> go.Figure:
    """Create a heatmap of sector sentiment scores."""
    if sector_sentiment.empty:
        fig = go.Figure()
        fig.update_layout(**_base_layout(title))
        return fig

    fig = go.Figure(go.Heatmap(
        z=[sector_sentiment["Mean_Sentiment"].values],
        x=sector_sentiment.index.tolist(),
        y=["Sentiment"],
        colorscale=[[0, COLORS["bear"]], [0.5, COLORS["neutral"]], [1, COLORS["bull"]]],
        zmid=0,
        text=[[f"{v:.2f}" for v in sector_sentiment["Mean_Sentiment"].values]],
        texttemplate="%{text}",
    ))

    fig.update_layout(**_base_layout(title, height=200))
    return fig


def correlation_matrix_chart(returns_df: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    """Plot correlation matrix as a heatmap."""
    corr = returns_df.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
    ))

    fig.update_layout(**_base_layout(title, height=500))
    return fig


def metric_gauge(value: float, title: str, range_min: float = -1, range_max: float = 1) -> go.Figure:
    """Create a gauge chart for a single metric (e.g., sentiment, confidence)."""
    color = COLORS["bull"] if value > 0 else COLORS["bear"] if value < 0 else COLORS["neutral"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title=dict(text=title, font=dict(size=14)),
        number=dict(font=dict(size=28)),
        gauge=dict(
            axis=dict(range=[range_min, range_max], tickcolor=COLORS["text"]),
            bar=dict(color=color),
            bgcolor=COLORS["surface"],
            bordercolor="#444",
            steps=[
                dict(range=[range_min, 0], color="rgba(255, 23, 68, 0.1)"),
                dict(range=[0, range_max], color="rgba(0, 200, 83, 0.1)"),
            ],
        ),
    ))

    fig.update_layout(
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        height=200,
        margin=dict(l=30, r=30, t=50, b=10),
    )
    return fig


def performance_chart(
    backtest_results: pd.DataFrame,
    title: str = "Strategy Performance",
) -> go.Figure:
    """Plot cumulative returns from backtest results."""
    fig = go.Figure()

    if "actual_return" in backtest_results.columns and "predicted_direction" in backtest_results.columns:
        strategy = backtest_results["actual_return"] * backtest_results["predicted_direction"]
        cum_strategy = (1 + strategy).cumprod()
        cum_buyhold = (1 + backtest_results["actual_return"]).cumprod()

        fig.add_trace(go.Scatter(
            x=backtest_results.index, y=cum_strategy,
            name="Strategy", line=dict(color=COLORS["primary"], width=2),
        ))
        fig.add_trace(go.Scatter(
            x=backtest_results.index, y=cum_buyhold,
            name="Buy & Hold", line=dict(color="#888", width=1, dash="dash"),
        ))

    fig.update_layout(**_base_layout(title, height=400))
    return fig

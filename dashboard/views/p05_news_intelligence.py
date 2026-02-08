"""
SENTINEL — Page 5: News Intelligence.
Live news feed, sentiment gauge, sector impact tracker.
"""

import streamlit as st
import pandas as pd

from config.settings import COLORS
from dashboard.components.charts import metric_gauge, sentiment_heatmap
from dashboard.components.widgets import news_feed


def render():
    st.title("News Intelligence")

    # ── Fetch & Analyze ───────────────────────────────────────
    if st.button("Refresh News", key="refresh_news"):
        st.session_state["news_refreshed"] = True

    news_df = None
    analyzed_df = None

    with st.spinner("Fetching and analyzing news..."):
        try:
            from data.news import fetch_all_news
            from features.sentiment import analyze_news_batch, get_sector_sentiment

            news_df = fetch_all_news()
            if not news_df.empty:
                analyzed_df = analyze_news_batch(news_df)
                st.session_state["analyzed_news"] = analyzed_df
        except Exception as e:
            st.warning(f"News fetch error: {e}")

    # Use cached if available
    if analyzed_df is None:
        analyzed_df = st.session_state.get("analyzed_news")

    if analyzed_df is None or analyzed_df.empty:
        st.info("No news data available. Click 'Refresh News' or check API keys.")
        st.markdown("""
        **Required API keys:**
        - `NEWSAPI_KEY` — Get one at newsapi.org (free tier)
        - GDELT requires no API key
        - RSS feeds require no API key
        """)
        return

    # ── Overall Sentiment Gauge ───────────────────────────────
    st.subheader("Market Sentiment")
    col1, col2, col3 = st.columns(3)

    overall_sentiment = analyzed_df["Sentiment"].mean()
    with col1:
        fig = metric_gauge(overall_sentiment, "Overall Sentiment")
        st.plotly_chart(fig, use_container_width=True)

    positive_pct = (analyzed_df["Sentiment_Label"] == "positive").mean()
    with col2:
        fig = metric_gauge(positive_pct, "Positive %", range_min=0, range_max=1)
        st.plotly_chart(fig, use_container_width=True)

    high_urgency = (analyzed_df["Urgency"] == "high").sum()
    with col3:
        st.metric("High Urgency Articles", high_urgency)
        st.metric("Total Articles", len(analyzed_df))

    # ── Sector Sentiment Heatmap ──────────────────────────────
    st.subheader("Sector Sentiment")
    try:
        from features.sentiment import get_sector_sentiment
        sector_sent = get_sector_sentiment(analyzed_df)
        if not sector_sent.empty:
            fig = sentiment_heatmap(sector_sent, "Sector Sentiment Scores")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                sector_sent.style.format({
                    "Mean_Sentiment": "{:.3f}",
                    "Positive_Pct": "{:.1%}",
                    "Negative_Pct": "{:.1%}",
                }).background_gradient(cmap="RdYlGn", subset=["Mean_Sentiment"]),
                use_container_width=True,
            )
    except Exception:
        pass

    # ── News Feed ─────────────────────────────────────────────
    st.subheader("Latest Headlines")

    # Filter controls
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        provider_filter = st.multiselect(
            "Source",
            analyzed_df["Provider"].dropna().unique().tolist(),
            default=analyzed_df["Provider"].dropna().unique().tolist(),
        )
    with fcol2:
        sentiment_filter = st.multiselect(
            "Sentiment",
            ["positive", "negative", "neutral"],
            default=["positive", "negative", "neutral"],
        )
    with fcol3:
        urgency_filter = st.multiselect(
            "Urgency",
            ["high", "medium", "low"],
            default=["high", "medium", "low"],
        )

    filtered = analyzed_df[
        (analyzed_df["Provider"].isin(provider_filter))
        & (analyzed_df["Sentiment_Label"].isin(sentiment_filter))
        & (analyzed_df["Urgency"].isin(urgency_filter))
    ]

    news_feed(filtered, max_items=30)

    # ── Sentiment Distribution ────────────────────────────────
    st.subheader("Sentiment Distribution")
    import plotly.graph_objects as go
    fig = go.Figure(go.Histogram(
        x=analyzed_df["Sentiment"],
        nbinsx=50,
        marker_color=COLORS["primary"],
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="white")
    fig.update_layout(
        title="Article Sentiment Distribution",
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

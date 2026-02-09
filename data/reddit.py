"""
SENTINEL — Reddit sentiment data collection.
Monitors financial subreddits for market sentiment signals.
"""
from __future__ import annotations

import pandas as pd

from config.settings import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_POST_LIMIT,
    REDDIT_SUBREDDITS,
    REDDIT_USER_AGENT,
)
from utils.helpers import cache_key, is_cache_fresh, load_parquet, save_parquet
from utils.logger import get_logger

log = get_logger("data.reddit")


def _get_reddit():
    """Initialize Reddit API client."""
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        log.warning("Reddit API credentials not set — skipping Reddit data")
        return None

    import praw
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )


def fetch_subreddit_posts(
    subreddit_name: str,
    sort: str = "hot",
    limit: int = REDDIT_POST_LIMIT,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch recent posts from a subreddit.
    Returns DataFrame with: title, text, score, num_comments, created, subreddit.
    """
    path = cache_key("reddit", f"{subreddit_name}_{sort}")
    if use_cache and is_cache_fresh(path, max_age_hours=2):
        log.info(f"Cache hit: r/{subreddit_name}")
        return load_parquet(path)

    reddit = _get_reddit()
    if reddit is None:
        return pd.DataFrame()

    log.info(f"Fetching r/{subreddit_name} ({sort}, limit={limit})")
    try:
        sub = reddit.subreddit(subreddit_name)
        if sort == "hot":
            posts = sub.hot(limit=limit)
        elif sort == "new":
            posts = sub.new(limit=limit)
        elif sort == "top":
            posts = sub.top(limit=limit, time_filter="week")
        else:
            posts = sub.hot(limit=limit)

        rows = []
        for post in posts:
            rows.append({
                "Title": post.title,
                "Text": post.selftext[:1000] if post.selftext else "",
                "Score": post.score,
                "Upvote_Ratio": post.upvote_ratio,
                "Num_Comments": post.num_comments,
                "Created_UTC": pd.Timestamp(post.created_utc, unit="s", tz="UTC"),
                "Subreddit": subreddit_name,
                "URL": f"https://reddit.com{post.permalink}",
                "Is_Self": post.is_self,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            save_parquet(df, path)
        log.info(f"r/{subreddit_name}: {len(df)} posts")
        return df

    except Exception as e:
        log.error(f"Reddit fetch failed for r/{subreddit_name}: {e}")
        return pd.DataFrame()


def fetch_all_subreddits(use_cache: bool = True) -> pd.DataFrame:
    """Fetch posts from all configured financial subreddits."""
    all_posts = []
    for sub in REDDIT_SUBREDDITS:
        df = fetch_subreddit_posts(sub, use_cache=use_cache)
        if not df.empty:
            all_posts.append(df)

    if not all_posts:
        return pd.DataFrame()

    combined = pd.concat(all_posts, ignore_index=True)
    log.info(f"Total Reddit posts: {len(combined)} from {len(REDDIT_SUBREDDITS)} subreddits")
    return combined


def get_trending_tickers(posts_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Extract and count ticker mentions from Reddit posts.
    Looks for $TICKER patterns and common ticker symbols.
    """
    import re

    if posts_df is None:
        posts_df = fetch_all_subreddits()

    if posts_df.empty:
        return pd.DataFrame()

    ticker_pattern = re.compile(r'\$([A-Z]{1,5})\b')
    ticker_counts: dict[str, int] = {}

    for _, row in posts_df.iterrows():
        text = f"{row.get('Title', '')} {row.get('Text', '')}"
        matches = ticker_pattern.findall(text)
        for ticker in matches:
            ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1

    if not ticker_counts:
        return pd.DataFrame()

    df = pd.DataFrame(
        [{"Ticker": k, "Mentions": v} for k, v in ticker_counts.items()]
    ).sort_values("Mentions", ascending=False)

    return df

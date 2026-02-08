"""
SENTINEL — News aggregation pipeline.
Pulls from GDELT (global events), NewsAPI (financial headlines), and RSS feeds.
"""

from datetime import datetime, timedelta

import feedparser
import pandas as pd

from config.settings import (
    CRYPTO_RSS_FEEDS,
    GDELT_LOOKBACK_DAYS,
    NEWS_API_CACHE_MINUTES,
    NEWS_MAX_ARTICLES_PER_FETCH,
    NEWS_RSS_CACHE_MINUTES,
    NEWSAPI_KEY,
    RSS_FEEDS,
)
from utils.helpers import cache_key, is_cache_fresh, load_parquet, save_parquet
from utils.logger import get_logger

log = get_logger("data.news")


def fetch_gdelt_news(
    query: str = "market OR economy OR inflation OR federal reserve",
    days: int = GDELT_LOOKBACK_DAYS,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch recent news articles from GDELT DOC API.
    Returns DataFrame with: title, url, source, date, tone.
    """
    path = cache_key("gdelt", query[:30])
    if use_cache and is_cache_fresh(path, max_age_hours=NEWS_RSS_CACHE_MINUTES / 60):
        log.info("Cache hit: GDELT")
        return load_parquet(path)

    log.info(f"Fetching GDELT articles (last {days} days)")
    try:
        from gdeltdoc import GdeltDoc, Filters

        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        filters = Filters(
            keyword=query,
            start_date=start_date,
            end_date=end_date,
            num_records=250,
            country="US",
        )

        gd = GdeltDoc()
        df = gd.article_search(filters)

        if df.empty:
            log.warning("No GDELT articles returned")
            return pd.DataFrame()

        # Standardize columns
        rename_map = {}
        if "title" in df.columns:
            rename_map["title"] = "Title"
        if "url" in df.columns:
            rename_map["url"] = "URL"
        if "domain" in df.columns:
            rename_map["domain"] = "Source"
        if "seendate" in df.columns:
            rename_map["seendate"] = "Date"
        if "tone" in df.columns:
            rename_map["tone"] = "Tone"

        df = df.rename(columns=rename_map)
        df["Provider"] = "GDELT"

        save_parquet(df, path)
        log.info(f"GDELT: {len(df)} articles fetched")
        return df

    except Exception as e:
        log.error(f"GDELT fetch failed: {e}")
        return pd.DataFrame()


def fetch_newsapi(
    query: str = "stock market OR economy OR crypto",
    language: str = "en",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch headlines from NewsAPI.
    Free tier: 100 requests/day, last 30 days.
    """
    if not NEWSAPI_KEY:
        log.warning("NEWSAPI_KEY not set — skipping NewsAPI")
        return pd.DataFrame()

    path = cache_key("newsapi", query[:30])
    if use_cache and is_cache_fresh(path, max_age_hours=NEWS_API_CACHE_MINUTES / 60):
        log.info("Cache hit: NewsAPI")
        return load_parquet(path)

    log.info("Fetching NewsAPI headlines")
    try:
        from newsapi import NewsApiClient

        api = NewsApiClient(api_key=NEWSAPI_KEY)
        response = api.get_everything(
            q=query,
            language=language,
            sort_by="publishedAt",
            page_size=min(NEWS_MAX_ARTICLES_PER_FETCH, 100),
        )

        articles = response.get("articles", [])
        if not articles:
            log.warning("No NewsAPI articles returned")
            return pd.DataFrame()

        rows = []
        for a in articles:
            rows.append({
                "Title": a.get("title", ""),
                "Description": a.get("description", ""),
                "Source": a.get("source", {}).get("name", ""),
                "URL": a.get("url", ""),
                "Date": a.get("publishedAt", ""),
                "Provider": "NewsAPI",
            })

        df = pd.DataFrame(rows)
        save_parquet(df, path)
        log.info(f"NewsAPI: {len(df)} articles")
        return df

    except Exception as e:
        log.error(f"NewsAPI fetch failed: {e}")
        return pd.DataFrame()


def fetch_rss_feeds(use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch latest articles from configured RSS feeds.
    """
    path = cache_key("rss", "all_feeds")
    if use_cache and is_cache_fresh(path, max_age_hours=NEWS_RSS_CACHE_MINUTES / 60):
        log.info("Cache hit: RSS feeds")
        return load_parquet(path)

    log.info(f"Fetching {len(RSS_FEEDS)} RSS feeds")
    all_articles = []

    for feed_name, feed_url in RSS_FEEDS.items():
        try:
            parsed = feedparser.parse(feed_url)
            for entry in parsed.entries[:20]:  # Limit per feed
                all_articles.append({
                    "Title": entry.get("title", ""),
                    "Description": entry.get("summary", ""),
                    "Source": feed_name,
                    "URL": entry.get("link", ""),
                    "Date": entry.get("published", ""),
                    "Provider": "RSS",
                })
        except Exception as e:
            log.warning(f"RSS feed failed ({feed_name}): {e}")

    if not all_articles:
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)
    save_parquet(df, path)
    log.info(f"RSS: {len(df)} articles from {len(RSS_FEEDS)} feeds")
    return df


def fetch_crypto_news(use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch crypto-specific news from GDELT, NewsAPI, and crypto RSS feeds.
    Returns DataFrame with: Title, Description, Source, URL, Date, Provider.
    """
    sources = []

    # 1. GDELT with crypto query (keep short — GDELT has query length limits)
    crypto_gdelt = fetch_gdelt_news(
        query="bitcoin OR ethereum OR cryptocurrency",
        days=3,
        use_cache=use_cache,
    )
    if not crypto_gdelt.empty:
        sources.append(crypto_gdelt)

    # 2. NewsAPI with crypto query
    if NEWSAPI_KEY:
        path = cache_key("newsapi", "crypto_news")
        if use_cache and is_cache_fresh(path, max_age_hours=NEWS_API_CACHE_MINUTES / 60):
            log.info("Cache hit: crypto NewsAPI")
            cached = load_parquet(path)
            if not cached.empty:
                sources.append(cached)
        else:
            try:
                from newsapi import NewsApiClient
                api = NewsApiClient(api_key=NEWSAPI_KEY)
                response = api.get_everything(
                    q="bitcoin OR ethereum OR solana OR crypto OR cryptocurrency",
                    language="en",
                    sort_by="publishedAt",
                    page_size=50,
                )
                articles = response.get("articles", [])
                if articles:
                    rows = []
                    for a in articles:
                        rows.append({
                            "Title": a.get("title", ""),
                            "Description": a.get("description", ""),
                            "Source": a.get("source", {}).get("name", ""),
                            "URL": a.get("url", ""),
                            "Date": a.get("publishedAt", ""),
                            "Provider": "NewsAPI",
                        })
                    df = pd.DataFrame(rows)
                    save_parquet(df, path)
                    sources.append(df)
                    log.info(f"Crypto NewsAPI: {len(df)} articles")
            except Exception as e:
                log.warning(f"Crypto NewsAPI failed: {e}")

    # 3. Crypto-specific RSS feeds
    crypto_rss_path = cache_key("rss", "crypto_feeds")
    if use_cache and is_cache_fresh(crypto_rss_path, max_age_hours=NEWS_RSS_CACHE_MINUTES / 60):
        log.info("Cache hit: crypto RSS")
        cached = load_parquet(crypto_rss_path)
        if not cached.empty:
            sources.append(cached)
    else:
        rss_articles = []
        for feed_name, feed_url in CRYPTO_RSS_FEEDS.items():
            try:
                parsed = feedparser.parse(feed_url)
                for entry in parsed.entries[:15]:
                    rss_articles.append({
                        "Title": entry.get("title", ""),
                        "Description": entry.get("summary", "")[:300] if entry.get("summary") else "",
                        "Source": feed_name,
                        "URL": entry.get("link", ""),
                        "Date": entry.get("published", ""),
                        "Provider": "CryptoRSS",
                    })
            except Exception as e:
                log.warning(f"Crypto RSS feed failed ({feed_name}): {e}")
        if rss_articles:
            df = pd.DataFrame(rss_articles)
            save_parquet(df, crypto_rss_path)
            sources.append(df)
            log.info(f"Crypto RSS: {len(df)} articles from {len(CRYPTO_RSS_FEEDS)} feeds")

    if not sources:
        log.warning("No crypto news from any source")
        return pd.DataFrame()

    combined = pd.concat(sources, ignore_index=True)

    # Remove duplicates by title similarity
    if "Title" in combined.columns:
        combined = combined.drop_duplicates(subset=["Title"], keep="first")

    # Parse dates
    if "Date" in combined.columns:
        combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce", utc=True)
        combined = combined.sort_values("Date", ascending=False)

    log.info(f"Total crypto news articles: {len(combined)}")
    return combined


def fetch_all_news(use_cache: bool = True) -> pd.DataFrame:
    """
    Aggregate news from all sources into a single DataFrame.
    Columns: Title, Description, Source, URL, Date, Provider, Tone.
    """
    sources = [
        fetch_gdelt_news(use_cache=use_cache),
        fetch_newsapi(use_cache=use_cache),
        fetch_rss_feeds(use_cache=use_cache),
    ]

    non_empty = [df for df in sources if not df.empty]
    if not non_empty:
        log.warning("No news from any source")
        return pd.DataFrame()

    combined = pd.concat(non_empty, ignore_index=True)

    # Parse dates
    if "Date" in combined.columns:
        combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce", utc=True)
        combined = combined.sort_values("Date", ascending=False)

    log.info(f"Total news articles: {len(combined)}")
    return combined

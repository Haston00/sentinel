"""
SENTINEL — Global configuration and settings.
API keys loaded from environment variables. File paths, parameters, defaults.
"""

import os
from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for d in [CACHE_DIR, MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys (from environment variables) ──────────────────────
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "sentinel-market-intel/1.0")

# CoinGecko — free tier, no key needed
# Yahoo Finance — free via yfinance, no key needed
# GDELT — free, no key needed

# ── Data Parameters ────────────────────────────────────────────
DEFAULT_LOOKBACK_YEARS = 5
CACHE_EXPIRY_HOURS = 6          # Default fallback (used by macro data, etc.)
STOCK_CACHE_MINUTES = 5         # Stock prices refresh every 5 minutes
CRYPTO_CACHE_MINUTES = 3        # Crypto prices refresh every 3 minutes
NEWS_RSS_CACHE_MINUTES = 10     # RSS news feeds refresh every 10 minutes
NEWS_API_CACHE_MINUTES = 120    # NewsAPI refresh every 2 hours (free tier: 100/day)
TRADING_DAYS_PER_YEAR = 252

# ── Forecast Horizons ─────────────────────────────────────────
HORIZONS = {
    "1W": 5,      # 5 trading days
    "1M": 21,     # 21 trading days
    "3M": 63,     # 63 trading days
}

# ── Model Parameters ──────────────────────────────────────────
REGIME_N_STATES = 3  # Bull, Bear, Transition
HMM_N_ITER = 200
GARCH_P = 1
GARCH_Q = 1

# XGBoost defaults
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}

# Random Forest defaults
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_leaf": 20,
    "max_features": "sqrt",
    "random_state": 42,
}

# Ensemble — rolling window for model weighting
ENSEMBLE_EVAL_WINDOW = 60  # trading days

# ── Sentiment Parameters ──────────────────────────────────────
SENTIMENT_LOOKBACK_HOURS = 48
VADER_COMPOUND_THRESHOLD = 0.05  # abs value below this = neutral

# ── News Parameters ───────────────────────────────────────────
NEWS_MAX_ARTICLES_PER_FETCH = 100
GDELT_LOOKBACK_DAYS = 7
RSS_FEEDS = {
    "Federal Reserve": "https://www.federalreserve.gov/feeds/press_all.xml",
    "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
    "Bloomberg Markets": "https://feeds.bloomberg.com/markets/news.rss",
    "CNBC": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147",
    "WSJ Markets": "https://feeds.wsj.com/rss/RSSMarketsMain.xml",
    "ECB": "https://www.ecb.europa.eu/rss/press.html",
}

CRYPTO_RSS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CoinTelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
    "The Block": "https://www.theblock.co/rss.xml",
    "Bitcoin Magazine": "https://bitcoinmagazine.com/feed",
}

# ── Reddit Subreddits ─────────────────────────────────────────
REDDIT_SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "cryptocurrency",
    "economics",
    "investing",
]
REDDIT_POST_LIMIT = 100  # per subreddit per fetch

# ── CoinGecko Rate Limiting ──────────────────────────────────
COINGECKO_CALLS_PER_MINUTE = 25  # Stay under 30 free-tier limit

# ── Dashboard Settings ────────────────────────────────────────
STREAMLIT_PAGE_TITLE = "SENTINEL — Market Intelligence"
STREAMLIT_PAGE_ICON = "🛡️"
STREAMLIT_LAYOUT = "wide"

# Color palette for consistent theming
COLORS = {
    "bull": "#00C853",
    "bear": "#FF1744",
    "neutral": "#FFD600",
    "transition": "#FF9100",
    "primary": "#2962FF",
    "secondary": "#6200EA",
    "background": "#0E1117",
    "surface": "#1E1E2E",
    "text": "#FAFAFA",
}

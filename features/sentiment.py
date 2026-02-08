"""
SENTINEL — NLP sentiment scoring engine.
VADER (fast financial sentiment) + entity extraction + urgency classification.
"""

import re

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config.assets import CRYPTO_ALL, SECTORS, get_sector_for_ticker
from config.settings import VADER_COMPOUND_THRESHOLD
from utils.logger import get_logger

log = get_logger("features.sentiment")

_vader = SentimentIntensityAnalyzer()

# Financial + crypto context boosters for VADER
# Weights calibrated: -3 = catastrophic, -2 = very bad, -1 = mildly negative
#                     +1 = mildly positive, +2 = very good, +3 = euphoric
_FINANCIAL_LEXICON = {
    # ── Traditional finance ─────────────────
    "bullish": 2.0,
    "bearish": -2.0,
    "rally": 1.5,
    "crash": -3.0,
    "surge": 2.0,
    "plunge": -2.5,
    "soar": 2.0,
    "tank": -2.0,
    "recession": -2.5,
    "recovery": 1.5,
    "inflation": -0.5,          # Mildly negative (context-dependent)
    "disinflation": 0.5,
    "stagflation": -2.0,
    "rate hike": -0.8,
    "rate cut": 1.5,
    "hawkish": -0.8,
    "dovish": 1.0,
    "earnings beat": 2.0,
    "earnings miss": -2.0,
    "revenue beat": 1.5,
    "revenue miss": -1.5,
    "guidance raised": 2.0,
    "guidance lowered": -2.0,
    "guidance cut": -2.0,
    "default": -3.0,
    "bankruptcy": -3.0,
    "chapter 11": -2.5,
    "delisted": -2.5,
    "upgrade": 1.5,
    "downgrade": -1.5,
    "bubble": -1.5,
    "breakout": 1.5,
    "all-time high": 1.5,
    "new high": 1.0,
    "52-week low": -1.0,
    "short squeeze": 1.5,
    "buyback": 1.0,
    "dividend increase": 1.0,
    "dividend cut": -2.0,
    "layoffs": -1.5,
    "restructuring": -0.5,
    "merger": 0.5,
    "acquisition": 0.5,
    "antitrust": -1.0,
    "sec investigation": -2.0,
    "fraud": -3.0,
    "insider buying": 1.5,
    "insider selling": -0.8,    # Insiders sell for many reasons

    # ── Crypto-specific (modern 2024-2026 language) ──
    "halving": 1.0,
    "moon": 1.5,
    "mooning": 2.0,
    "dump": -2.0,
    "dumping": -2.0,
    "pump": 0.5,                # Pump alone is ambiguous (could be pump & dump)
    "pumping": 0.8,
    "rug pull": -3.0,
    "rugged": -3.0,
    "hodl": 1.0,
    "hodling": 1.0,
    "fud": -0.5,                # The word "FUD" itself is often used defensively
    "rekt": -2.0,
    "degen": 0.0,               # Neutral — cultural term
    "ape in": 0.5,
    "diamond hands": 1.0,
    "paper hands": -0.5,
    "whale": 0.0,               # Neutral — depends on context
    "whale buying": 2.0,
    "whale selling": -2.0,
    "etf approved": 2.5,
    "etf approval": 2.5,
    "etf rejected": -2.0,
    "etf denied": -2.0,
    "staking": 0.5,
    "unstaking": -0.5,
    "hack": -3.0,
    "hacked": -3.0,
    "exploit": -2.5,
    "bridge hack": -2.5,
    "defi": 0.3,
    "nft": 0.0,                 # Neutral
    "airdrop": 0.8,
    "token burn": 1.0,
    "supply shock": 1.5,
    "capitulation": -2.0,
    "accumulation": 1.5,
    "institutional adoption": 2.0,
    "ban": -2.0,
    "regulation": -0.3,         # Mildly negative for crypto
    "regulatory clarity": 1.0,
    "cbdc": 0.0,                # Neutral

    # ── Individual financial signal words ─────────
    # (VADER can't match multi-word phrases, so add key single words)
    "outperform": 1.5,
    "underperform": -1.5,
    "overweight": 1.0,
    "underweight": -1.0,
    "raised": 0.8,             # "guidance raised", "target raised"
    "lowered": -0.8,           # "guidance lowered"
    "reiterated": 0.3,         # Mildly positive — maintaining stance
    "initiated": 0.3,
    "beat": 1.0,               # "earnings beat"
    "topped": 1.0,
    "missed": -1.0,
    "shortfall": -1.5,
    "exceeded": 1.5,
    "disappointing": -1.5,
    "accelerating": 1.0,
    "decelerating": -0.8,
    "slowing": -0.5,
    "resilient": 1.0,
    "deteriorating": -1.5,
    "weakening": -1.0,
    "strengthening": 1.0,
}

# Update VADER lexicon with financial terms
_vader.lexicon.update(_FINANCIAL_LEXICON)


def score_text(text: str) -> dict:
    """
    Score a single text for sentiment using VADER.
    Returns: compound (-1 to 1), positive, negative, neutral scores.
    """
    if not text or not isinstance(text, str):
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0, "label": "neutral"}

    scores = _vader.polarity_scores(text)
    compound = scores["compound"]

    if compound >= VADER_COMPOUND_THRESHOLD:
        label = "positive"
    elif compound <= -VADER_COMPOUND_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"

    return {
        "compound": compound,
        "pos": scores["pos"],
        "neg": scores["neg"],
        "neu": scores["neu"],
        "label": label,
    }


def score_dataframe(df: pd.DataFrame, text_column: str = "Title") -> pd.DataFrame:
    """
    Score all rows in a DataFrame for sentiment.
    Adds columns: Sentiment, Sentiment_Label, Sentiment_Pos, Sentiment_Neg.
    """
    if df.empty or text_column not in df.columns:
        return df

    scores = df[text_column].apply(lambda t: score_text(str(t)))
    df = df.copy()
    df["Sentiment"] = scores.apply(lambda s: s["compound"])
    df["Sentiment_Label"] = scores.apply(lambda s: s["label"])
    df["Sentiment_Pos"] = scores.apply(lambda s: s["pos"])
    df["Sentiment_Neg"] = scores.apply(lambda s: s["neg"])

    return df


def extract_entities(text: str) -> dict:
    """
    Extract market entities (tickers, sectors, cryptos) from text.
    Returns dict with matched entities.
    """
    text_upper = text.upper()
    entities = {"tickers": [], "sectors": [], "cryptos": []}

    # Ticker pattern: $AAPL or standalone known tickers
    ticker_pattern = re.compile(r'\$([A-Z]{1,5})\b')
    matches = ticker_pattern.findall(text_upper)
    entities["tickers"].extend(matches)

    # Sector detection
    sector_keywords = {
        "Technology": ["tech", "software", "semiconductor", "chip", "ai ", "artificial intelligence"],
        "Healthcare": ["health", "pharma", "biotech", "drug", "fda", "medical"],
        "Financials": ["bank", "financial", "lending", "credit", "insurance"],
        "Energy": ["oil", "gas", "energy", "crude", "opec", "petroleum"],
        "Consumer Discretionary": ["retail", "consumer", "e-commerce", "luxury"],
        "Industrials": ["industrial", "manufacturing", "defense", "aerospace"],
        "Materials": ["mining", "steel", "chemical", "gold", "copper"],
        "Utilities": ["utility", "electric", "power grid", "renewable"],
        "Real Estate": ["real estate", "reit", "housing", "mortgage", "property"],
        "Consumer Staples": ["grocery", "food", "beverage", "household"],
        "Communication Services": ["social media", "streaming", "telecom", "advertising"],
    }

    text_lower = text.lower()
    for sector, keywords in sector_keywords.items():
        if any(kw in text_lower for kw in keywords):
            entities["sectors"].append(sector)

    # Crypto detection — all coins in CRYPTO_ALL, plus common name variants
    crypto_keywords = {
        "bitcoin": ["bitcoin", " btc ", " btc,", " btc.", "$btc"],
        "ethereum": ["ethereum", " eth ", " eth,", " eth.", "$eth", "ether "],
        "solana": ["solana", " sol ", " sol,", " sol.", "$sol"],
        "binancecoin": ["binance", " bnb ", "$bnb"],
        "ripple": ["ripple", " xrp ", "$xrp"],
        "cardano": ["cardano", " ada ", "$ada"],
        "dogecoin": ["dogecoin", "doge ", "$doge"],
        "avalanche-2": ["avalanche", " avax ", "$avax"],
        "polkadot": ["polkadot", " dot ", "$dot"],
        "chainlink": ["chainlink", " link ", "$link"],
        "polygon-ecosystem-token": ["polygon", " pol ", " matic ", "$matic"],
        "uniswap": ["uniswap", " uni ", "$uni"],
        "litecoin": ["litecoin", " ltc ", "$ltc"],
        "sui": [" sui ", "$sui"],
        "arbitrum": ["arbitrum", " arb ", "$arb"],
        "optimism": ["optimism", " op ", "$op"],
        "near": ["near protocol", " near ", "$near"],
    }
    text_padded = f" {text_lower} "  # Pad so " btc " matches at boundaries
    for crypto_id, keywords in crypto_keywords.items():
        if any(kw in text_padded for kw in keywords):
            entities["cryptos"].append(crypto_id)

    return entities


def classify_urgency(text: str, sentiment_score: float) -> str:
    """
    Classify market impact urgency of a news item.
    Returns: "high", "medium", or "low".
    """
    high_urgency_words = [
        "breaking", "urgent", "crash", "plunge", "soar", "emergency",
        "halt", "circuit breaker", "black swan", "crisis", "collapse",
        "default", "bankruptcy", "war", "invasion", "sanctions",
    ]
    medium_urgency_words = [
        "rally", "surge", "drop", "tumble", "spike", "fed ",
        "rate", "earnings", "miss", "beat", "guidance", "forecast",
    ]

    text_lower = text.lower()

    if any(word in text_lower for word in high_urgency_words) or abs(sentiment_score) > 0.7:
        return "high"
    elif any(word in text_lower for word in medium_urgency_words) or abs(sentiment_score) > 0.4:
        return "medium"
    return "low"


def analyze_news_batch(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full sentiment analysis pipeline on a news DataFrame.
    Adds: sentiment scores, entity tags, urgency, sector impact.
    """
    if news_df.empty:
        return news_df

    # Score sentiment
    df = score_dataframe(news_df, text_column="Title")

    # Extract entities and urgency
    entities_list = df["Title"].apply(lambda t: extract_entities(str(t)))
    df["Entities_Tickers"] = entities_list.apply(lambda e: e.get("tickers", []))
    df["Entities_Sectors"] = entities_list.apply(lambda e: e.get("sectors", []))
    df["Entities_Cryptos"] = entities_list.apply(lambda e: e.get("cryptos", []))

    df["Urgency"] = df.apply(
        lambda row: classify_urgency(str(row.get("Title", "")), row.get("Sentiment", 0)),
        axis=1,
    )

    log.info(
        f"Analyzed {len(df)} articles — "
        f"Positive: {(df['Sentiment_Label'] == 'positive').sum()}, "
        f"Negative: {(df['Sentiment_Label'] == 'negative').sum()}, "
        f"Neutral: {(df['Sentiment_Label'] == 'neutral').sum()}"
    )

    return df


_SOURCE_CREDIBILITY = {
    # Tier 1: Major financial news (weight 1.5x)
    "reuters": 1.5, "bloomberg": 1.5, "wsj": 1.5, "wall street journal": 1.5,
    "financial times": 1.5, "cnbc": 1.4, "barron's": 1.4, "marketwatch": 1.3,
    # Tier 2: Reliable business press (weight 1.2x)
    "ap": 1.2, "associated press": 1.2, "bbc": 1.2, "nyt": 1.2, "new york times": 1.2,
    "the economist": 1.3, "fortune": 1.2, "business insider": 1.1,
    # Tier 3: Crypto-specific trusted sources (weight 1.2x)
    "coindesk": 1.2, "the block": 1.2, "decrypt": 1.1, "cointelegraph": 1.0,
    # Default: unknown sources get 0.8x weight
}


def _get_source_weight(source: str) -> float:
    """Return credibility multiplier for a news source."""
    if not source:
        return 0.8
    source_lower = source.lower()
    for name, weight in _SOURCE_CREDIBILITY.items():
        if name in source_lower:
            return weight
    return 0.8  # Unknown source — slight discount


def get_sector_sentiment(analyzed_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment by sector with recency and source credibility weighting."""
    if analyzed_df.empty or "Entities_Sectors" not in analyzed_df.columns:
        return pd.DataFrame()

    # Compute recency weights: newer articles count more
    now = pd.Timestamp.now(tz="UTC")
    if "Date" in analyzed_df.columns:
        dates = pd.to_datetime(analyzed_df["Date"], errors="coerce", utc=True)
        hours_old = (now - dates).dt.total_seconds() / 3600
        # Exponential decay: half-life of 24 hours
        recency_weight = np.exp(-0.693 * hours_old / 24).fillna(0.5)
    else:
        recency_weight = pd.Series(1.0, index=analyzed_df.index)

    # Source credibility weights
    if "Source" in analyzed_df.columns:
        source_weight = analyzed_df["Source"].apply(_get_source_weight)
    else:
        source_weight = pd.Series(1.0, index=analyzed_df.index)

    combined_weight = recency_weight * source_weight

    rows = []
    for sector in SECTORS:
        mask = analyzed_df["Entities_Sectors"].apply(lambda s: sector in s)
        sector_news = analyzed_df[mask]
        if sector_news.empty:
            continue

        w = combined_weight[mask]
        w_sum = w.sum()
        if w_sum > 0:
            weighted_sentiment = (sector_news["Sentiment"] * w).sum() / w_sum
        else:
            weighted_sentiment = sector_news["Sentiment"].mean()

        rows.append({
            "Sector": sector,
            "Mean_Sentiment": weighted_sentiment,
            "Article_Count": len(sector_news),
            "Positive_Pct": (sector_news["Sentiment_Label"] == "positive").mean(),
            "Negative_Pct": (sector_news["Sentiment_Label"] == "negative").mean(),
            "High_Urgency": (sector_news["Urgency"] == "high").sum(),
        })

    return pd.DataFrame(rows).set_index("Sector") if rows else pd.DataFrame()

"""
SENTINEL — Crypto data pipeline.
Fetches price, market cap, and volume data via CoinGecko free API.
"""
from __future__ import annotations

import time

import pandas as pd
from pycoingecko import CoinGeckoAPI

from config.assets import CRYPTO_ALL, CRYPTO_MAJOR
from config.settings import COINGECKO_CALLS_PER_MINUTE, CRYPTO_CACHE_MINUTES, DEFAULT_LOOKBACK_YEARS
from utils.helpers import cache_key, is_cache_fresh, load_parquet, save_parquet
from utils.logger import get_logger

log = get_logger("data.crypto")

cg = CoinGeckoAPI()

# Rate-limiting delay
_MIN_DELAY = 60.0 / COINGECKO_CALLS_PER_MINUTE


def _rate_limit():
    """Simple rate limiter for CoinGecko free tier."""
    time.sleep(_MIN_DELAY)


def fetch_crypto_ohlcv(
    coin_id: str,
    vs_currency: str = "usd",
    days: int | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch daily OHLC data for a cryptocurrency.
    Returns DataFrame with columns: Open, High, Low, Close.
    """
    days = days or (DEFAULT_LOOKBACK_YEARS * 365)
    path = cache_key("crypto_ohlcv", coin_id)

    if use_cache and is_cache_fresh(path, max_age_hours=CRYPTO_CACHE_MINUTES / 60):
        log.info(f"Cache hit: {coin_id}")
        return load_parquet(path)

    log.info(f"Downloading crypto: {coin_id} ({days} days)")
    try:
        _rate_limit()
        ohlc = cg.get_coin_ohlc_by_id(id=coin_id, vs_currency=vs_currency, days=str(days))

        if not ohlc:
            log.warning(f"No OHLC data for {coin_id}")
            return pd.DataFrame()

        df = pd.DataFrame(ohlc, columns=["Timestamp", "Open", "High", "Low", "Close"])
        df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
        df = df.set_index("Date").drop(columns=["Timestamp"])
        df = df[~df.index.duplicated(keep="last")]

        save_parquet(df, path)
        log.info(f"Cached: {coin_id} ({len(df)} rows)")
        return df

    except Exception as e:
        log.error(f"Failed to fetch {coin_id}: {e}")
        return pd.DataFrame()


def fetch_crypto_market_data(
    coin_id: str,
    vs_currency: str = "usd",
    days: int | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch daily price, market cap, and volume for a cryptocurrency.
    Returns DataFrame with columns: Price, MarketCap, Volume.
    """
    days = days or (DEFAULT_LOOKBACK_YEARS * 365)
    path = cache_key("crypto_market", coin_id)

    if use_cache and is_cache_fresh(path, max_age_hours=CRYPTO_CACHE_MINUTES / 60):
        log.info(f"Cache hit (market): {coin_id}")
        return load_parquet(path)

    log.info(f"Downloading market data: {coin_id}")
    try:
        _rate_limit()
        data = cg.get_coin_market_chart_by_id(
            id=coin_id, vs_currency=vs_currency, days=days
        )

        prices = pd.DataFrame(data["prices"], columns=["Timestamp", "Price"])
        mcaps = pd.DataFrame(data["market_caps"], columns=["Timestamp", "MarketCap"])
        volumes = pd.DataFrame(data["total_volumes"], columns=["Timestamp", "Volume"])

        for frame in [prices, mcaps, volumes]:
            frame["Date"] = pd.to_datetime(frame["Timestamp"], unit="ms")
            frame.set_index("Date", inplace=True)
            frame.drop(columns=["Timestamp"], inplace=True)

        df = prices.join(mcaps).join(volumes)
        df = df[~df.index.duplicated(keep="last")]

        save_parquet(df, path)
        log.info(f"Cached market data: {coin_id} ({len(df)} rows)")
        return df

    except Exception as e:
        log.error(f"Failed to fetch market data for {coin_id}: {e}")
        return pd.DataFrame()


def fetch_all_crypto(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for all cryptos in the universe."""
    results = {}
    for coin_id in CRYPTO_ALL:
        df = fetch_crypto_ohlcv(coin_id, use_cache=use_cache)
        if not df.empty:
            results[coin_id] = df
    return results


def get_crypto_close_prices(
    coin_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Return DataFrame of close prices, one column per crypto."""
    coin_ids = coin_ids or list(CRYPTO_MAJOR.keys())
    closes = {}
    for coin_id in coin_ids:
        df = fetch_crypto_ohlcv(coin_id)
        if not df.empty and "Close" in df.columns:
            symbol = CRYPTO_ALL.get(coin_id, {}).get("symbol", coin_id)
            closes[symbol] = df["Close"]
    return pd.DataFrame(closes).dropna(how="all")


def get_current_prices() -> dict[str, dict]:
    """Get current price snapshot for all tracked cryptos."""
    log.info("Fetching current crypto prices")
    try:
        _rate_limit()
        ids = ",".join(CRYPTO_ALL.keys())
        data = cg.get_price(
            ids=ids,
            vs_currencies="usd",
            include_market_cap=True,
            include_24hr_vol=True,
            include_24hr_change=True,
        )
        return data
    except Exception as e:
        log.error(f"Failed to fetch current prices: {e}")
        return {}

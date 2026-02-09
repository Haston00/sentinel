"""
SENTINEL — Stock data pipeline.
Fetches OHLCV + fundamental data via yfinance, with Parquet caching.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from config.settings import DEFAULT_LOOKBACK_YEARS, STOCK_CACHE_MINUTES
from utils.helpers import (
    cache_key,
    get_lookback_date,
    is_cache_fresh,
    load_parquet,
    save_parquet,
    today_str,
)
from utils.logger import get_logger

log = get_logger("data.stocks")


def fetch_ohlcv(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a single ticker.
    Returns DataFrame with columns: Open, High, Low, Close, Volume.
    Index is DatetimeIndex.
    """
    start = start or get_lookback_date(DEFAULT_LOOKBACK_YEARS)
    end = end or today_str()

    path = cache_key("ohlcv", ticker)
    if use_cache and is_cache_fresh(path, max_age_hours=STOCK_CACHE_MINUTES / 60):
        log.info(f"Cache hit: {ticker}")
        return load_parquet(path)

    log.info(f"Downloading: {ticker} ({start} to {end})")
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            log.warning(f"No data returned for {ticker}")
            return pd.DataFrame()

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Date"
        save_parquet(df, path)
        log.info(f"Cached: {ticker} ({len(df)} rows)")
        return df

    except Exception as e:
        log.error(f"Failed to fetch {ticker}: {e}")
        return pd.DataFrame()


def fetch_multiple(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for multiple tickers. Returns dict of ticker -> DataFrame."""
    results = {}
    for ticker in tickers:
        df = fetch_ohlcv(ticker, start=start, end=end, use_cache=use_cache)
        if not df.empty:
            results[ticker] = df
    return results


def get_close_prices(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of adjusted close prices, one column per ticker."""
    data = fetch_multiple(tickers, start=start, end=end)
    closes = {}
    for ticker, df in data.items():
        if "Close" in df.columns:
            closes[ticker] = df["Close"]
    return pd.DataFrame(closes).dropna(how="all")


def fetch_fundamentals(ticker: str) -> dict:
    """
    Fetch fundamental data for a single stock.
    Returns dict with key financial metrics.
    """
    log.info(f"Fetching fundamentals: {ticker}")
    try:
        info = yf.Ticker(ticker).info
        return {
            "ticker": ticker,
            "name": info.get("shortName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap", None),
            "pe_ratio": info.get("trailingPE", None),
            "forward_pe": info.get("forwardPE", None),
            "pb_ratio": info.get("priceToBook", None),
            "dividend_yield": info.get("dividendYield", None),
            "beta": info.get("beta", None),
            "52w_high": info.get("fiftyTwoWeekHigh", None),
            "52w_low": info.get("fiftyTwoWeekLow", None),
            "avg_volume": info.get("averageVolume", None),
            "earnings_growth": info.get("earningsGrowth", None),
            "revenue_growth": info.get("revenueGrowth", None),
            "profit_margin": info.get("profitMargins", None),
            "roe": info.get("returnOnEquity", None),
            "debt_to_equity": info.get("debtToEquity", None),
        }
    except Exception as e:
        log.error(f"Failed to fetch fundamentals for {ticker}: {e}")
        return {"ticker": ticker}


def get_returns(
    ticker: str,
    period: int = 1,
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Get periodic returns for a ticker."""
    df = fetch_ohlcv(ticker, start=start, end=end)
    if df.empty:
        return pd.Series(dtype=float)
    return df["Close"].pct_change(periods=period).dropna()

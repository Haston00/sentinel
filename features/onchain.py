"""
SENTINEL — On-chain crypto metrics.
Computes crypto-specific features from market data available via CoinGecko free API.
"""

import numpy as np
import pandas as pd

from data.crypto import fetch_crypto_market_data, fetch_crypto_ohlcv
from utils.logger import get_logger

log = get_logger("features.onchain")


def compute_nvt_ratio(coin_id: str) -> pd.Series:
    """
    Network Value to Transactions (NVT) approximation.
    Uses market cap / volume as a proxy (true NVT needs on-chain tx volume).
    High NVT = overvalued relative to usage. Low NVT = undervalued.
    """
    df = fetch_crypto_market_data(coin_id)
    if df.empty or "MarketCap" not in df.columns or "Volume" not in df.columns:
        return pd.Series(dtype=float, name=f"{coin_id}_NVT")

    # Smooth volume to avoid daily spikes
    smoothed_vol = df["Volume"].rolling(14).mean()
    nvt = df["MarketCap"] / smoothed_vol.replace(0, np.nan)
    nvt.name = f"{coin_id}_NVT"
    return nvt


def compute_mvrv_proxy(coin_id: str) -> pd.Series:
    """
    Market Value to Realized Value proxy.
    Approximation using market cap vs 200-day moving average of market cap.
    """
    df = fetch_crypto_market_data(coin_id)
    if df.empty or "MarketCap" not in df.columns:
        return pd.Series(dtype=float, name=f"{coin_id}_MVRV")

    realized_proxy = df["MarketCap"].rolling(200).mean()
    mvrv = df["MarketCap"] / realized_proxy.replace(0, np.nan)
    mvrv.name = f"{coin_id}_MVRV"
    return mvrv


def compute_crypto_features(coin_id: str) -> pd.DataFrame:
    """
    Compute all available crypto-specific features for a given coin.
    Returns DataFrame with feature columns.
    """
    features = {}

    # Price data features
    ohlcv = fetch_crypto_ohlcv(coin_id)
    market = fetch_crypto_market_data(coin_id)

    if ohlcv.empty:
        log.warning(f"No OHLCV data for {coin_id}")
        return pd.DataFrame()

    c = ohlcv["Close"]

    # Volatility metrics
    returns = c.pct_change()
    features["Volatility_7d"] = returns.rolling(7).std()
    features["Volatility_30d"] = returns.rolling(30).std()
    features["Volatility_90d"] = returns.rolling(90).std()

    # Volatility ratio (short-term vs long-term)
    features["Vol_Ratio_7_30"] = features["Volatility_7d"] / features["Volatility_30d"].replace(0, np.nan)

    # Log returns (more normal distribution)
    features["LogReturn_1d"] = np.log(c / c.shift(1))
    features["LogReturn_7d"] = np.log(c / c.shift(7))
    features["LogReturn_30d"] = np.log(c / c.shift(30))

    # Drawdown from ATH
    running_max = c.cummax()
    features["Drawdown_From_ATH"] = (c - running_max) / running_max

    # Price momentum
    features["Momentum_7d"] = c.pct_change(7)
    features["Momentum_30d"] = c.pct_change(30)
    features["Momentum_90d"] = c.pct_change(90)

    # Moving average signals
    sma_50 = c.rolling(50).mean()
    sma_200 = c.rolling(200).mean()
    features["Price_vs_SMA50"] = (c / sma_50 - 1) * 100
    features["Price_vs_SMA200"] = (c / sma_200 - 1) * 100
    features["Golden_Cross"] = (sma_50 > sma_200).astype(int)

    # Market data features
    if not market.empty:
        if "Volume" in market.columns:
            vol = market["Volume"]
            features["Volume_MA7"] = vol.rolling(7).mean()
            features["Volume_MA30"] = vol.rolling(30).mean()
            features["Volume_Ratio"] = vol / vol.rolling(30).mean().replace(0, np.nan)
            features["Volume_Trend"] = vol.rolling(7).mean() / vol.rolling(30).mean().replace(0, np.nan)

        if "MarketCap" in market.columns:
            features["MarketCap"] = market["MarketCap"]
            features["MarketCap_Change_7d"] = market["MarketCap"].pct_change(7)

    # NVT and MVRV proxies
    nvt = compute_nvt_ratio(coin_id)
    if not nvt.empty:
        features["NVT_Ratio"] = nvt

    mvrv = compute_mvrv_proxy(coin_id)
    if not mvrv.empty:
        features["MVRV_Proxy"] = mvrv

    if not features:
        return pd.DataFrame()

    df = pd.DataFrame(features)
    log.info(f"Computed {len(df.columns)} on-chain features for {coin_id}")
    return df

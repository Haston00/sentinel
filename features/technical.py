"""
SENTINEL — Technical analysis feature engineering.
Uses the `ta` library to compute 20+ indicators from OHLCV data.
"""

import pandas as pd
import ta

from utils.logger import get_logger

log = get_logger("features.technical")


def compute_all_technical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a full suite of technical indicators from OHLCV data.
    Input: DataFrame with Open, High, Low, Close, Volume columns.
    Output: DataFrame with all original + indicator columns.
    """
    if df.empty or len(df) < 50:
        log.warning("Insufficient data for technical indicators")
        return df

    out = df.copy()
    o, h, l, c, v = out["Open"], out["High"], out["Low"], out["Close"], out["Volume"]

    # ── Trend Indicators ──────────────────────────────────────
    # Moving averages
    out["SMA_20"] = ta.trend.sma_indicator(c, window=20)
    out["SMA_50"] = ta.trend.sma_indicator(c, window=50)
    out["SMA_200"] = ta.trend.sma_indicator(c, window=200)
    out["EMA_12"] = ta.trend.ema_indicator(c, window=12)
    out["EMA_26"] = ta.trend.ema_indicator(c, window=26)

    # MACD
    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    out["MACD"] = macd.macd()
    out["MACD_Signal"] = macd.macd_signal()
    out["MACD_Hist"] = macd.macd_diff()

    # ADX (Average Directional Index)
    adx = ta.trend.ADXIndicator(h, l, c, window=14)
    out["ADX"] = adx.adx()
    out["ADX_Pos"] = adx.adx_pos()
    out["ADX_Neg"] = adx.adx_neg()

    # Aroon
    aroon = ta.trend.AroonIndicator(h, l, window=25)
    out["Aroon_Up"] = aroon.aroon_up()
    out["Aroon_Down"] = aroon.aroon_down()

    # ── Momentum Indicators ───────────────────────────────────
    # RSI
    out["RSI_14"] = ta.momentum.rsi(c, window=14)

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
    out["Stoch_K"] = stoch.stoch()
    out["Stoch_D"] = stoch.stoch_signal()

    # Williams %R
    out["Williams_R"] = ta.momentum.williams_r(h, l, c, lbp=14)

    # Rate of Change
    out["ROC_10"] = ta.momentum.roc(c, window=10)

    # CCI (Commodity Channel Index)
    out["CCI_20"] = ta.trend.cci(h, l, c, window=20)

    # ── Volatility Indicators ─────────────────────────────────
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    out["BB_Upper"] = bb.bollinger_hband()
    out["BB_Lower"] = bb.bollinger_lband()
    out["BB_Width"] = bb.bollinger_wband()
    out["BB_Pct"] = bb.bollinger_pband()

    # ATR (Average True Range)
    out["ATR_14"] = ta.volatility.average_true_range(h, l, c, window=14)

    # Keltner Channel
    kc = ta.volatility.KeltnerChannel(h, l, c, window=20)
    out["KC_Upper"] = kc.keltner_channel_hband()
    out["KC_Lower"] = kc.keltner_channel_lband()

    # ── Volume Indicators ─────────────────────────────────────
    # OBV (On-Balance Volume)
    out["OBV"] = ta.volume.on_balance_volume(c, v)

    # VWAP approximation (daily)
    out["VWAP"] = ta.volume.volume_weighted_average_price(h, l, c, v, window=14)

    # Chaikin Money Flow
    out["CMF"] = ta.volume.chaikin_money_flow(h, l, c, v, window=20)

    # Force Index
    out["Force_Index"] = ta.volume.force_index(c, v, window=13)

    # ── Derived Features ──────────────────────────────────────
    # Price relative to moving averages
    out["Price_vs_SMA20"] = (c / out["SMA_20"] - 1) * 100
    out["Price_vs_SMA50"] = (c / out["SMA_50"] - 1) * 100
    out["Price_vs_SMA200"] = (c / out["SMA_200"] - 1) * 100

    # Golden/Death cross signal
    out["SMA50_vs_SMA200"] = (out["SMA_50"] / out["SMA_200"] - 1) * 100

    # Returns at various lookbacks
    out["Return_1d"] = c.pct_change(1)
    out["Return_5d"] = c.pct_change(5)
    out["Return_21d"] = c.pct_change(21)
    out["Return_63d"] = c.pct_change(63)

    # Volatility (rolling std of returns)
    out["Vol_10d"] = out["Return_1d"].rolling(10).std()
    out["Vol_21d"] = out["Return_1d"].rolling(21).std()

    # Volume relative to average
    out["Vol_Ratio_20d"] = v / v.rolling(20).mean()

    log.info(f"Computed {len(out.columns) - len(df.columns)} technical features")
    return out


def compute_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical features and drop raw price columns.
    Returns only the feature columns suitable for ML models.
    """
    full = compute_all_technical(df)
    # Drop raw OHLCV — keep only derived features
    feature_cols = [c for c in full.columns if c not in ["Open", "High", "Low", "Close", "Volume"]]
    return full[feature_cols]


def get_feature_names() -> list[str]:
    """Return list of all technical feature column names."""
    dummy = pd.DataFrame({
        "Open": range(300),
        "High": range(300),
        "Low": range(300),
        "Close": range(300),
        "Volume": range(300),
    }, dtype=float)
    # Add slight variation so indicators can compute
    dummy["Close"] = dummy["Close"] + 100
    dummy["High"] = dummy["Close"] + 1
    dummy["Low"] = dummy["Close"] - 1
    dummy["Open"] = dummy["Close"] - 0.5
    dummy["Volume"] = 1000000

    features = compute_for_ml(dummy)
    return list(features.columns)

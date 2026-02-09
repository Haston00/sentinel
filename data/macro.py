"""
SENTINEL — Macro economic data pipeline.
Fetches indicators from FRED (Federal Reserve Economic Data).
"""
from __future__ import annotations

import pandas as pd

from config.assets import MACRO_INDICATORS
from config.settings import DEFAULT_LOOKBACK_YEARS, FRED_API_KEY
from utils.helpers import (
    cache_key,
    get_lookback_date,
    is_cache_fresh,
    load_parquet,
    save_parquet,
    today_str,
)
from utils.logger import get_logger

log = get_logger("data.macro")


def _get_fred():
    """Lazy-load FRED API client."""
    if not FRED_API_KEY:
        log.warning("FRED_API_KEY not set — macro data will be unavailable")
        return None
    from fredapi import Fred
    return Fred(api_key=FRED_API_KEY)


def fetch_series(
    series_id: str,
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.Series:
    """Fetch a single FRED series. Returns a named pd.Series with DatetimeIndex."""
    start = start or get_lookback_date(DEFAULT_LOOKBACK_YEARS + 2)  # Extra buffer for lagged indicators
    end = end or today_str()

    path = cache_key("fred", series_id)
    if use_cache and is_cache_fresh(path):
        log.info(f"Cache hit: {series_id}")
        df = load_parquet(path)
        return df.iloc[:, 0]

    fred = _get_fred()
    if fred is None:
        return pd.Series(dtype=float, name=series_id)

    log.info(f"Downloading FRED: {series_id}")
    try:
        data = fred.get_series(series_id, observation_start=start, observation_end=end)
        data.name = series_id
        data.index.name = "Date"

        # Save as single-column DataFrame for Parquet compatibility
        df = data.to_frame()
        save_parquet(df, path)
        log.info(f"Cached: {series_id} ({len(data)} obs)")
        return data

    except Exception as e:
        log.error(f"Failed to fetch {series_id}: {e}")
        return pd.Series(dtype=float, name=series_id)


def fetch_all_macro(use_cache: bool = True) -> dict[str, pd.Series]:
    """Fetch all configured macro indicators. Returns dict of series_id -> pd.Series."""
    results = {}
    for key, info in MACRO_INDICATORS.items():
        series = fetch_series(info["series"], use_cache=use_cache)
        if not series.empty:
            results[key] = series
    return results


def get_macro_dataframe(
    indicators: list[str] | None = None,
    resample_to: str = "B",  # Business day
) -> pd.DataFrame:
    """
    Build a single DataFrame with all macro indicators resampled to business-day frequency.
    Forward-fills lower-frequency data.
    """
    indicators = indicators or list(MACRO_INDICATORS.keys())

    series_list = []
    for ind in indicators:
        info = MACRO_INDICATORS.get(ind)
        if info is None:
            log.warning(f"Unknown indicator: {ind}")
            continue
        s = fetch_series(info["series"])
        if not s.empty:
            s.name = ind
            series_list.append(s)

    if not series_list:
        return pd.DataFrame()

    df = pd.concat(series_list, axis=1)

    # Resample to business day and forward-fill
    df = df.resample(resample_to).last().ffill()
    return df


def get_yield_curve() -> pd.DataFrame:
    """Fetch yield curve data: 2Y, 10Y, 30Y yields and 10Y-2Y spread."""
    rate_series = ["DGS2", "DGS10", "DGS30", "T10Y2Y"]
    data = {}
    for sid in rate_series:
        s = fetch_series(sid)
        if not s.empty:
            data[sid] = s
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df.resample("B").last().ffill()
    return df


def get_latest_values() -> dict[str, float]:
    """Get the most recent value for each macro indicator."""
    latest = {}
    for key, info in MACRO_INDICATORS.items():
        s = fetch_series(info["series"])
        if not s.empty:
            latest[key] = s.dropna().iloc[-1]
    return latest

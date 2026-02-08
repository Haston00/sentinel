"""
SENTINEL — Utility helpers.
Date utilities, data validation, caching helpers.
"""

import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from config.settings import CACHE_DIR, CACHE_EXPIRY_HOURS, DEFAULT_LOOKBACK_YEARS


def get_lookback_date(years: int = DEFAULT_LOOKBACK_YEARS) -> str:
    """Return start date string (YYYY-MM-DD) for N years ago."""
    dt = datetime.now() - timedelta(days=years * 365)
    return dt.strftime("%Y-%m-%d")


def today_str() -> str:
    """Return today's date as YYYY-MM-DD."""
    return datetime.now().strftime("%Y-%m-%d")


def cache_key(prefix: str, identifier: str) -> Path:
    """Generate a deterministic cache file path."""
    safe_id = identifier.replace("^", "_").replace("/", "_").replace(" ", "_")
    return CACHE_DIR / f"{prefix}_{safe_id}.parquet"


def is_cache_fresh(path: Path, max_age_hours: float = CACHE_EXPIRY_HOURS) -> bool:
    """Check if a cached file exists and is within the expiry window."""
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    age = datetime.now() - mtime
    return age < timedelta(hours=max_age_hours)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to Parquet with directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow")


def load_parquet(path: Path) -> pd.DataFrame:
    """Load DataFrame from Parquet."""
    return pd.read_parquet(path, engine="pyarrow")


def safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """Percentage change that handles zeros gracefully."""
    return series.pct_change(periods=periods).replace([float("inf"), float("-inf")], 0.0)


def forward_fill_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill NaNs, then drop any remaining leading NaN rows."""
    df = df.ffill()
    first_valid = df.apply(lambda col: col.first_valid_index()).max()
    if first_valid is not None:
        df = df.loc[first_valid:]
    return df


def annualize_returns(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualize a series of periodic returns."""
    total = (1 + returns).prod()
    n_periods = len(returns)
    if n_periods == 0:
        return 0.0
    return total ** (periods_per_year / n_periods) - 1


def annualize_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualize volatility from periodic returns."""
    return returns.std() * (periods_per_year ** 0.5)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.04, periods_per_year: int = 252) -> float:
    """Calculate annualized Sharpe ratio."""
    ann_ret = annualize_returns(returns, periods_per_year)
    ann_vol = annualize_volatility(returns, periods_per_year)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / ann_vol


def max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from a return series."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

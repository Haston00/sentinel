"""
SENTINEL — Macro-economic feature engineering.
Transforms raw FRED data into predictive signals.
"""

import pandas as pd
import numpy as np

from data.macro import fetch_series, get_yield_curve, get_macro_dataframe
from utils.logger import get_logger

log = get_logger("features.macro")


def compute_yield_curve_features() -> pd.DataFrame:
    """
    Compute yield curve signals:
    - 10Y-2Y spread level and direction
    - Inversion indicator
    - Steepening/flattening momentum
    """
    yc = get_yield_curve()
    if yc.empty:
        return pd.DataFrame()

    features = pd.DataFrame(index=yc.index)

    if "T10Y2Y" in yc.columns:
        spread = yc["T10Y2Y"]
        features["YC_Spread"] = spread
        features["YC_Inverted"] = (spread < 0).astype(int)
        features["YC_Spread_MA20"] = spread.rolling(20).mean()
        features["YC_Spread_Change_21d"] = spread.diff(21)
        features["YC_Steepening"] = (spread.diff(5) > 0).astype(int)

    if "DGS10" in yc.columns and "DGS2" in yc.columns:
        features["Rate_10Y"] = yc["DGS10"]
        features["Rate_2Y"] = yc["DGS2"]
        features["Rate_10Y_Change_21d"] = yc["DGS10"].diff(21)
        features["Rate_2Y_Change_21d"] = yc["DGS2"].diff(21)

    return features.dropna(how="all")


def compute_inflation_features() -> pd.DataFrame:
    """
    Compute inflation-derived signals:
    - CPI YoY change and momentum
    - Core CPI
    - Breakeven inflation
    - Real rates
    """
    features = {}

    cpi = fetch_series("CPIAUCSL")
    if not cpi.empty:
        features["CPI_YoY"] = cpi.pct_change(12) * 100  # 12-month pct change
        features["CPI_MoM"] = cpi.pct_change(1) * 100
        features["CPI_Accel"] = features["CPI_YoY"].diff(3)  # Acceleration

    core_cpi = fetch_series("CPILFESL")
    if not core_cpi.empty:
        features["CoreCPI_YoY"] = core_cpi.pct_change(12) * 100

    breakeven = fetch_series("T5YIE")
    if not breakeven.empty:
        features["Breakeven_5Y"] = breakeven
        features["Breakeven_Change_21d"] = breakeven.diff(21)

    # Real rate = 10Y nominal - 5Y breakeven
    nom_10y = fetch_series("DGS10")
    if not nom_10y.empty and not breakeven.empty:
        aligned = pd.DataFrame({"Nominal_10Y": nom_10y, "Breakeven": breakeven}).dropna()
        features["Real_Rate"] = aligned["Nominal_10Y"] - aligned["Breakeven"]

    if not features:
        return pd.DataFrame()

    df = pd.DataFrame(features)
    return df.resample("B").last().ffill().dropna(how="all")


def compute_labor_features() -> pd.DataFrame:
    """Labor market features: unemployment, claims, payrolls."""
    features = {}

    unrate = fetch_series("UNRATE")
    if not unrate.empty:
        features["Unemployment"] = unrate
        features["Unemployment_Change_3m"] = unrate.diff(3)
        # Sahm Rule indicator: 3-month avg rise >= 0.5% from 12-month low
        rolling_min = unrate.rolling(12).min()
        rolling_avg = unrate.rolling(3).mean()
        features["Sahm_Indicator"] = rolling_avg - rolling_min

    claims = fetch_series("ICSA")
    if not claims.empty:
        features["Initial_Claims"] = claims
        features["Claims_4wk_Avg"] = claims.rolling(4).mean()
        features["Claims_Change_4wk"] = claims.pct_change(4) * 100

    payrolls = fetch_series("PAYEMS")
    if not payrolls.empty:
        features["Payrolls_MoM_Change"] = payrolls.diff(1)
        features["Payrolls_3m_Avg_Change"] = payrolls.diff(1).rolling(3).mean()

    if not features:
        return pd.DataFrame()

    df = pd.DataFrame(features)
    return df.resample("B").last().ffill().dropna(how="all")


def compute_financial_conditions() -> pd.DataFrame:
    """Financial conditions signals: HY spread, VIX, M2."""
    features = {}

    hy_spread = fetch_series("BAMLH0A0HYM2")
    if not hy_spread.empty:
        features["HY_Spread"] = hy_spread
        features["HY_Spread_Change_21d"] = hy_spread.diff(21)
        features["HY_Spread_Zscore"] = (
            (hy_spread - hy_spread.rolling(252).mean()) / hy_spread.rolling(252).std()
        )

    vix = fetch_series("VIXCLS")
    if not vix.empty:
        features["VIX"] = vix
        features["VIX_MA20"] = vix.rolling(20).mean()
        features["VIX_Above_20"] = (vix > 20).astype(int)
        features["VIX_Above_30"] = (vix > 30).astype(int)

    m2 = fetch_series("M2SL")
    if not m2.empty:
        features["M2_YoY"] = m2.pct_change(12) * 100
        features["M2_Accel"] = features["M2_YoY"].diff(3)

    fedfunds = fetch_series("FEDFUNDS")
    if not fedfunds.empty:
        features["FedFunds"] = fedfunds
        features["FedFunds_Change_3m"] = fedfunds.diff(3)

    if not features:
        return pd.DataFrame()

    df = pd.DataFrame(features)
    return df.resample("B").last().ffill().dropna(how="all")


def compute_housing_features() -> pd.DataFrame:
    """Housing market signals."""
    features = {}

    starts = fetch_series("HOUST")
    if not starts.empty:
        features["Housing_Starts"] = starts
        features["Housing_Starts_YoY"] = starts.pct_change(12) * 100

    cs = fetch_series("CSUSHPINSA")
    if not cs.empty:
        features["CaseShiller_YoY"] = cs.pct_change(12) * 100
        features["CaseShiller_Accel"] = features["CaseShiller_YoY"].diff(3)

    if not features:
        return pd.DataFrame()

    df = pd.DataFrame(features)
    return df.resample("B").last().ffill().dropna(how="all")


def compute_all_macro_features() -> pd.DataFrame:
    """
    Combine all macro feature sets into a single DataFrame.
    Resampled to business-day frequency with forward-fill.
    """
    feature_sets = [
        compute_yield_curve_features(),
        compute_inflation_features(),
        compute_labor_features(),
        compute_financial_conditions(),
        compute_housing_features(),
    ]

    non_empty = [f for f in feature_sets if not f.empty]
    if not non_empty:
        log.warning("No macro features computed — check FRED API key")
        return pd.DataFrame()

    combined = pd.concat(non_empty, axis=1)
    combined = combined.resample("B").last().ffill()

    log.info(f"Computed {len(combined.columns)} macro features")
    return combined

"""
SENTINEL — Multi-factor model.
Fama-French style factor analysis for sector and stock evaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config.assets import BENCHMARKS, SECTORS
from data.stocks import fetch_ohlcv, get_close_prices
from utils.helpers import safe_pct_change
from utils.logger import get_logger

log = get_logger("models.factors")


class FactorModel:
    """
    Multi-factor model for stock/sector expected return estimation.
    Factors: Market, Size, Value, Momentum, Quality (approximated from available data).
    """

    FACTOR_NAMES = ["Market", "Size", "Value", "Momentum", "Quality"]

    def __init__(self):
        self.factor_returns: pd.DataFrame | None = None
        self.betas: dict[str, dict] = {}

    def build_factors(self) -> pd.DataFrame:
        """
        Construct factor return series from available ETF data.
        Uses long/short proxies where possible.
        """
        log.info("Building factor return series")

        # Market factor: SPY returns
        spy = fetch_ohlcv("SPY")
        if spy.empty:
            log.error("Cannot build factors: no SPY data")
            return pd.DataFrame()

        mkt_returns = spy["Close"].pct_change().dropna()
        mkt_returns.name = "Market"

        # Size factor proxy: IWM (small cap) - SPY (large cap)
        iwm = fetch_ohlcv("IWM")
        if not iwm.empty:
            size_factor = iwm["Close"].pct_change() - spy["Close"].pct_change()
            size_factor.name = "Size"
        else:
            size_factor = pd.Series(dtype=float, name="Size")

        # Value factor proxy: XLF (financials, value-heavy) - XLK (tech, growth-heavy)
        xlf = fetch_ohlcv("XLF")
        xlk = fetch_ohlcv("XLK")
        if not xlf.empty and not xlk.empty:
            value_factor = xlf["Close"].pct_change() - xlk["Close"].pct_change()
            value_factor.name = "Value"
        else:
            value_factor = pd.Series(dtype=float, name="Value")

        # Momentum factor proxy: MTUM ETF returns (or construct from winners-losers)
        mtum = fetch_ohlcv("MTUM")
        if not mtum.empty:
            mom_factor = mtum["Close"].pct_change() - spy["Close"].pct_change()
            mom_factor.name = "Momentum"
        else:
            mom_factor = pd.Series(dtype=float, name="Momentum")

        # Quality factor proxy: QUAL ETF returns
        qual = fetch_ohlcv("QUAL")
        if not qual.empty:
            quality_factor = qual["Close"].pct_change() - spy["Close"].pct_change()
            quality_factor.name = "Quality"
        else:
            quality_factor = pd.Series(dtype=float, name="Quality")

        factors = pd.concat(
            [mkt_returns, size_factor, value_factor, mom_factor, quality_factor],
            axis=1,
        ).dropna()

        self.factor_returns = factors
        log.info(f"Built {len(factors.columns)} factors over {len(factors)} days")
        return factors

    def estimate_betas(self, ticker: str) -> dict:
        """
        Estimate factor betas (loadings) for a given ticker via OLS regression.
        Returns dict of factor_name -> beta.
        """
        if self.factor_returns is None:
            self.build_factors()
        if self.factor_returns is None or self.factor_returns.empty:
            return {}

        asset = fetch_ohlcv(ticker)
        if asset.empty:
            return {}

        asset_returns = asset["Close"].pct_change().dropna()

        # Align dates
        aligned = pd.concat([asset_returns.rename("Asset"), self.factor_returns], axis=1).dropna()
        if len(aligned) < 60:
            log.warning(f"Insufficient overlapping data for {ticker}")
            return {}

        y = aligned["Asset"].values
        X = aligned[self.factor_returns.columns].values

        reg = LinearRegression().fit(X, y)
        betas = dict(zip(self.factor_returns.columns, reg.coef_))
        betas["Alpha"] = reg.intercept_
        betas["R_Squared"] = reg.score(X, y)

        self.betas[ticker] = betas
        log.info(f"{ticker} betas: {', '.join(f'{k}={v:.3f}' for k, v in betas.items())}")
        return betas

    def expected_return(self, ticker: str, factor_forecast: dict | None = None) -> float:
        """
        Calculate expected return based on factor betas and forecasted factor returns.
        If no factor forecast provided, uses historical mean factor returns.
        """
        betas = self.betas.get(ticker)
        if betas is None:
            betas = self.estimate_betas(ticker)
        if not betas:
            return 0.0

        if factor_forecast is None and self.factor_returns is not None:
            # Use trailing 21-day mean factor returns, annualized
            factor_forecast = (self.factor_returns.tail(21).mean() * 252).to_dict()

        if factor_forecast is None:
            return 0.0

        expected = betas.get("Alpha", 0.0) * 252  # Annualize alpha
        for factor_name, beta in betas.items():
            if factor_name in ("Alpha", "R_Squared"):
                continue
            expected += beta * factor_forecast.get(factor_name, 0.0)

        return expected

    def sector_factor_exposure(self) -> pd.DataFrame:
        """Calculate factor exposures for all sector ETFs."""
        rows = []
        for sector_name, info in SECTORS.items():
            betas = self.estimate_betas(info["etf"])
            if betas:
                betas["Sector"] = sector_name
                betas["ETF"] = info["etf"]
                rows.append(betas)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.set_index("Sector")
        return df

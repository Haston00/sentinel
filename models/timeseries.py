"""
SENTINEL — Time series forecasting models.
ARIMA, SARIMAX for individual assets. VAR for cross-asset relationships.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR

from utils.logger import get_logger

log = get_logger("models.timeseries")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class ARIMAForecaster:
    """ARIMA model for univariate price forecasting."""

    def __init__(self, order: tuple = (2, 1, 2)):
        self.order = order
        self.model = None
        self.result = None

    def fit(self, series: pd.Series) -> "ARIMAForecaster":
        """Fit ARIMA model on a price or return series."""
        clean = series.dropna()
        if len(clean) < 100:
            log.warning("Insufficient data for ARIMA — need at least 100 observations")
            return self

        try:
            self.model = ARIMA(clean, order=self.order)
            self.result = self.model.fit()
            log.info(f"ARIMA{self.order} fitted — AIC: {self.result.aic:.1f}")
        except Exception as e:
            log.error(f"ARIMA fit failed: {e}")
            self.result = None
        return self

    def forecast(self, steps: int = 21) -> pd.DataFrame:
        """
        Generate point forecast with confidence intervals.
        Returns DataFrame with columns: Forecast, Lower, Upper.
        """
        if self.result is None:
            return pd.DataFrame()

        try:
            fc = self.result.get_forecast(steps=steps)
            mean = fc.predicted_mean
            ci = fc.conf_int(alpha=0.05)

            result = pd.DataFrame({
                "Forecast": mean.values,
                "Lower": ci.iloc[:, 0].values,
                "Upper": ci.iloc[:, 1].values,
            }, index=mean.index)
            return result

        except Exception as e:
            log.error(f"ARIMA forecast failed: {e}")
            return pd.DataFrame()


class SARIMAXForecaster:
    """SARIMAX model with optional exogenous variables."""

    def __init__(
        self,
        order: tuple = (1, 1, 1),
        seasonal_order: tuple = (1, 1, 1, 5),  # Weekly seasonality for daily data
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.result = None

    def fit(self, endog: pd.Series, exog: pd.DataFrame | None = None) -> "SARIMAXForecaster":
        """Fit SARIMAX model."""
        clean_endog = endog.dropna()
        if len(clean_endog) < 200:
            log.warning("Insufficient data for SARIMAX")
            return self

        clean_exog = None
        if exog is not None:
            clean_exog = exog.reindex(clean_endog.index).ffill().dropna()
            clean_endog = clean_endog.reindex(clean_exog.index)

        try:
            self.model = SARIMAX(
                clean_endog,
                exog=clean_exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.result = self.model.fit(disp=False, maxiter=200)
            log.info(f"SARIMAX fitted — AIC: {self.result.aic:.1f}")
        except Exception as e:
            log.error(f"SARIMAX fit failed: {e}")
            self.result = None
        return self

    def forecast(self, steps: int = 21, exog_future: pd.DataFrame | None = None) -> pd.DataFrame:
        """Generate forecast with confidence intervals."""
        if self.result is None:
            return pd.DataFrame()

        try:
            fc = self.result.get_forecast(steps=steps, exog=exog_future)
            mean = fc.predicted_mean
            ci = fc.conf_int(alpha=0.05)

            return pd.DataFrame({
                "Forecast": mean.values,
                "Lower": ci.iloc[:, 0].values,
                "Upper": ci.iloc[:, 1].values,
            }, index=mean.index)

        except Exception as e:
            log.error(f"SARIMAX forecast failed: {e}")
            return pd.DataFrame()


class VARForecaster:
    """
    Vector Autoregression for cross-asset relationship modeling.
    Captures how assets influence each other.
    """

    def __init__(self, maxlags: int = 10):
        self.maxlags = maxlags
        self.model = None
        self.result = None
        self.columns = None

    def fit(self, returns_df: pd.DataFrame) -> "VARForecaster":
        """
        Fit VAR model on multi-asset return series.
        Input: DataFrame where each column is an asset's return series.
        """
        clean = returns_df.dropna()
        if len(clean) < 200:
            log.warning("Insufficient data for VAR model")
            return self

        self.columns = clean.columns.tolist()

        try:
            self.model = VAR(clean)
            self.result = self.model.fit(maxlags=self.maxlags, ic="aic")
            log.info(f"VAR fitted with {self.result.k_ar} lags on {len(self.columns)} series")
        except Exception as e:
            log.error(f"VAR fit failed: {e}")
            self.result = None
        return self

    def forecast(self, steps: int = 21) -> pd.DataFrame:
        """Forecast all series forward."""
        if self.result is None:
            return pd.DataFrame()

        try:
            fc = self.result.forecast(self.result.endog[-self.result.k_ar:], steps=steps)
            return pd.DataFrame(fc, columns=self.columns)
        except Exception as e:
            log.error(f"VAR forecast failed: {e}")
            return pd.DataFrame()

    def impulse_response(self, periods: int = 21) -> dict:
        """
        Compute impulse response functions.
        Shows how a shock to one asset propagates to others.
        """
        if self.result is None:
            return {}

        try:
            irf = self.result.irf(periods)
            return {
                "irf": irf.irfs,
                "columns": self.columns,
                "periods": periods,
            }
        except Exception as e:
            log.error(f"IRF computation failed: {e}")
            return {}

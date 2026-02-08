"""
SENTINEL — Volatility forecasting models.
GARCH(1,1) and EGARCH for conditional volatility estimation.
"""

import warnings

import numpy as np
import pandas as pd
from arch import arch_model

from config.settings import GARCH_P, GARCH_Q
from utils.logger import get_logger

log = get_logger("models.volatility")

warnings.filterwarnings("ignore", category=FutureWarning)


class GARCHForecaster:
    """GARCH family models for volatility forecasting."""

    def __init__(
        self,
        p: int = GARCH_P,
        q: int = GARCH_Q,
        model_type: str = "GARCH",
        dist: str = "t",  # Student-t for fat tails
    ):
        self.p = p
        self.q = q
        self.model_type = model_type
        self.dist = dist
        self.model = None
        self.result = None

    def fit(self, returns: pd.Series, scale: float = 100.0) -> "GARCHForecaster":
        """
        Fit GARCH model on return series.
        Returns are scaled by `scale` for numerical stability.
        """
        clean = returns.dropna()
        if len(clean) < 200:
            log.warning("Insufficient data for GARCH")
            return self

        scaled = clean * scale

        try:
            vol_type = "EGARCH" if self.model_type == "EGARCH" else "GARCH"
            self.model = arch_model(
                scaled,
                vol=vol_type,
                p=self.p,
                q=self.q,
                dist=self.dist,
                mean="AR",
                lags=1,
            )
            self.result = self.model.fit(disp="off", show_warning=False)
            log.info(
                f"{self.model_type}({self.p},{self.q}) fitted — "
                f"Log-likelihood: {self.result.loglikelihood:.1f}"
            )
        except Exception as e:
            log.error(f"GARCH fit failed: {e}")
            self.result = None
        return self

    def forecast_volatility(
        self,
        steps: int = 21,
        scale: float = 100.0,
    ) -> pd.DataFrame:
        """
        Forecast conditional volatility forward.
        Returns DataFrame with: Variance, Volatility (annualized), and Mean forecast.
        """
        if self.result is None:
            return pd.DataFrame()

        try:
            fc = self.result.forecast(horizon=steps)

            variance = fc.variance.iloc[-1].values / (scale ** 2)
            vol_daily = np.sqrt(variance)
            vol_annual = vol_daily * np.sqrt(252)

            mean_fc = fc.mean.iloc[-1].values / scale if fc.mean is not None else np.zeros(steps)

            return pd.DataFrame({
                "Variance": variance,
                "Vol_Daily": vol_daily,
                "Vol_Annual": vol_annual,
                "Mean_Forecast": mean_fc,
            }, index=range(1, steps + 1))

        except Exception as e:
            log.error(f"GARCH forecast failed: {e}")
            return pd.DataFrame()

    def current_volatility(self, scale: float = 100.0) -> dict:
        """Get the current (most recent) conditional volatility estimate."""
        if self.result is None:
            return {}

        cond_vol = self.result.conditional_volatility
        latest = cond_vol.iloc[-1] / scale
        annual = latest * np.sqrt(252)

        return {
            "daily_vol": round(latest, 6),
            "annual_vol": round(annual, 4),
            "annual_vol_pct": round(annual * 100, 2),
        }


def fit_garch(returns: pd.Series) -> GARCHForecaster:
    """Convenience: fit a standard GARCH(1,1) with Student-t."""
    model = GARCHForecaster(model_type="GARCH")
    return model.fit(returns)


def fit_egarch(returns: pd.Series) -> GARCHForecaster:
    """Convenience: fit EGARCH(1,1) — captures asymmetric volatility."""
    model = GARCHForecaster(model_type="EGARCH")
    return model.fit(returns)


def compare_volatility_models(returns: pd.Series) -> dict:
    """
    Fit both GARCH and EGARCH, return the better model based on BIC.
    """
    garch = fit_garch(returns)
    egarch = fit_egarch(returns)

    results = {}
    if garch.result is not None:
        results["GARCH"] = {
            "model": garch,
            "aic": garch.result.aic,
            "bic": garch.result.bic,
        }
    if egarch.result is not None:
        results["EGARCH"] = {
            "model": egarch,
            "aic": egarch.result.aic,
            "bic": egarch.result.bic,
        }

    if not results:
        return {"best": None, "models": results}

    best_name = min(results, key=lambda k: results[k]["bic"])
    log.info(f"Best volatility model: {best_name} (BIC: {results[best_name]['bic']:.1f})")

    return {"best": results[best_name]["model"], "models": results}

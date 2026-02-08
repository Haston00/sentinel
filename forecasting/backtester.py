"""
SENTINEL — Walk-forward backtesting engine.
Tests forecast models on historical data with no lookahead bias.
"""

import numpy as np
import pandas as pd

from utils.helpers import annualize_returns, annualize_volatility, max_drawdown, sharpe_ratio
from utils.logger import get_logger

log = get_logger("forecasting.backtester")


class WalkForwardBacktester:
    """
    Walk-forward out-of-sample backtesting.
    At each step: train on past data, forecast, record result, move forward.
    """

    def __init__(
        self,
        min_train_window: int = 252,
        step_size: int = 21,
        forecast_horizon: int = 21,
    ):
        self.min_train_window = min_train_window
        self.step_size = step_size
        self.forecast_horizon = forecast_horizon
        self.results: list[dict] = []

    def run(
        self,
        prices: pd.Series,
        model_factory,
        feature_factory=None,
    ) -> pd.DataFrame:
        """
        Execute walk-forward backtest.

        Args:
            prices: Historical price series.
            model_factory: Callable that takes (train_prices, train_features) and returns
                          a dict with 'direction' (-1/0/1) and 'confidence'.
            feature_factory: Optional callable that takes prices and returns feature DataFrame.

        Returns: DataFrame with backtest results.
        """
        self.results = []
        n = len(prices)
        log.info(
            f"Starting walk-forward backtest: {n} observations, "
            f"min_train={self.min_train_window}, step={self.step_size}, "
            f"horizon={self.forecast_horizon}"
        )

        t = self.min_train_window
        while t + self.forecast_horizon <= n:
            # Training data: everything up to t
            train_prices = prices.iloc[:t]

            # Build features if factory provided
            train_features = None
            if feature_factory is not None:
                train_features = feature_factory(train_prices)

            # Get forecast
            try:
                forecast = model_factory(train_prices, train_features)
            except Exception as e:
                log.warning(f"Model failed at step {t}: {e}")
                t += self.step_size
                continue

            # Actual forward return
            actual_return = (prices.iloc[t + self.forecast_horizon - 1] / prices.iloc[t - 1]) - 1
            actual_direction = 1 if actual_return > 0.005 else (-1 if actual_return < -0.005 else 0)

            predicted_direction = forecast.get("direction", 0)
            correct = predicted_direction == actual_direction

            self.results.append({
                "date": prices.index[t],
                "predicted_direction": predicted_direction,
                "actual_direction": actual_direction,
                "actual_return": actual_return,
                "confidence": forecast.get("confidence", 0.5),
                "correct": correct,
            })

            t += self.step_size

        if not self.results:
            log.warning("No backtest results produced")
            return pd.DataFrame()

        df = pd.DataFrame(self.results)
        df = df.set_index("date")
        log.info(f"Backtest complete: {len(df)} forecast periods")
        return df

    def compute_metrics(self, results_df: pd.DataFrame | None = None) -> dict:
        """
        Compute performance metrics from backtest results.
        Returns dict with hit rate, Sharpe, max drawdown, etc.
        """
        df = results_df if results_df is not None else pd.DataFrame(self.results)
        if df.empty:
            return {}

        if "date" in df.columns:
            df = df.set_index("date")

        n = len(df)
        hit_rate = df["correct"].mean()

        # Strategy returns: if predicted up, go long; if down, go short; if flat, stay out
        strategy_returns = df["actual_return"] * df["predicted_direction"]

        metrics = {
            "n_forecasts": n,
            "hit_rate": round(hit_rate, 4),
            "hit_rate_pct": f"{hit_rate:.1%}",
            "total_return": round((1 + strategy_returns).prod() - 1, 4),
            "annualized_return": round(annualize_returns(strategy_returns), 4),
            "annualized_vol": round(annualize_volatility(strategy_returns), 4),
            "sharpe_ratio": round(sharpe_ratio(strategy_returns), 4),
            "max_drawdown": round(max_drawdown(strategy_returns), 4),
            "avg_return_correct": round(
                strategy_returns[df["correct"]].mean() if df["correct"].sum() > 0 else 0, 4
            ),
            "avg_return_incorrect": round(
                strategy_returns[~df["correct"]].mean() if (~df["correct"]).sum() > 0 else 0, 4
            ),
            "long_accuracy": round(
                df[df["predicted_direction"] == 1]["correct"].mean()
                if (df["predicted_direction"] == 1).sum() > 0 else 0, 4
            ),
            "short_accuracy": round(
                df[df["predicted_direction"] == -1]["correct"].mean()
                if (df["predicted_direction"] == -1).sum() > 0 else 0, 4
            ),
        }

        # Calibration: does higher confidence correlate with higher accuracy?
        if "confidence" in df.columns:
            high_conf = df[df["confidence"] > 0.6]
            low_conf = df[df["confidence"] <= 0.6]
            metrics["high_conf_accuracy"] = round(
                high_conf["correct"].mean() if len(high_conf) > 0 else 0, 4
            )
            metrics["low_conf_accuracy"] = round(
                low_conf["correct"].mean() if len(low_conf) > 0 else 0, 4
            )

        log.info(f"Backtest metrics: hit_rate={metrics['hit_rate_pct']}, sharpe={metrics['sharpe_ratio']}")
        return metrics

    def compute_regime_metrics(
        self,
        results_df: pd.DataFrame,
        regime_series: pd.Series,
    ) -> pd.DataFrame:
        """Break down performance by regime."""
        if results_df.empty:
            return pd.DataFrame()

        df = results_df.copy()
        if "date" in df.columns:
            df = df.set_index("date")

        # Align regimes to backtest dates
        aligned_regimes = regime_series.reindex(df.index, method="ffill")
        df["Regime"] = aligned_regimes

        rows = []
        for regime in df["Regime"].dropna().unique():
            regime_df = df[df["Regime"] == regime]
            strategy_returns = regime_df["actual_return"] * regime_df["predicted_direction"]
            rows.append({
                "Regime": regime,
                "N_Forecasts": len(regime_df),
                "Hit_Rate": regime_df["correct"].mean(),
                "Avg_Return": strategy_returns.mean(),
                "Sharpe": sharpe_ratio(strategy_returns) if len(strategy_returns) > 10 else 0,
            })

        return pd.DataFrame(rows).set_index("Regime") if rows else pd.DataFrame()

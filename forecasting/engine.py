"""
SENTINEL — Master forecast engine.
Orchestrates: data → features → regime detection → model ensemble → news adjustment → output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import HORIZONS
from catalysts.calendar import get_uncertainty_multiplier
from data.stocks import fetch_ohlcv
from data.crypto import fetch_crypto_ohlcv
from features.technical import compute_all_technical, compute_for_ml
from features.macro_features import compute_all_macro_features
from features.sentiment import analyze_news_batch, score_text
from models.regime import RegimeDetector
from models.timeseries import ARIMAForecaster
from models.volatility import fit_garch
from models.ml_models import MLForecaster
from models.ensemble import AdaptiveEnsemble
from forecasting.explainer import explain_forecast
from utils.logger import get_logger

log = get_logger("forecasting.engine")


class ForecastEngine:
    """
    Master orchestrator that runs the full forecasting pipeline for any asset.
    """

    def __init__(self):
        self.ensemble = AdaptiveEnsemble()
        self.regime_detector: RegimeDetector | None = None
        self._ml_models: dict[str, MLForecaster] = {}

    def _load_regime_detector(self, asset_type: str = "equity") -> RegimeDetector:
        """Load or build regime detector."""
        if self.regime_detector is not None:
            return self.regime_detector

        detector = RegimeDetector()
        model_name = "equity_regime" if asset_type == "equity" else "crypto_regime"
        try:
            detector.load(model_name)
        except Exception:
            log.info(f"Building fresh {asset_type} regime model")
            if asset_type == "equity":
                from models.regime import build_equity_regime_model
                detector = build_equity_regime_model()
            else:
                from models.regime import build_crypto_regime_model
                detector = build_crypto_regime_model()

        self.regime_detector = detector
        return detector

    def forecast_equity(
        self,
        ticker: str,
        horizons: dict[str, int] | None = None,
        include_news: bool = True,
    ) -> dict:
        """
        Full forecast pipeline for a stock or ETF.
        Returns dict with forecasts for each horizon, regime info, and explanations.
        """
        horizons = horizons or HORIZONS
        log.info(f"Generating forecast for {ticker}")

        # 1. Fetch data
        ohlcv = fetch_ohlcv(ticker)
        if ohlcv.empty:
            return {"error": f"No data available for {ticker}"}

        close = ohlcv["Close"]
        returns = close.pct_change().dropna()

        # 2. Compute technical features
        tech_features = compute_all_technical(ohlcv)

        # 3. Detect regime
        regime_detector = self._load_regime_detector("equity")
        spy_data = fetch_ohlcv("SPY")
        if not spy_data.empty:
            regime_info = regime_detector.current_regime(spy_data["Close"])
        else:
            regime_info = {"regime": "Unknown", "probabilities": {}}

        current_regime = regime_info["regime"]

        # 4. Run models for each horizon
        forecast_results = {}
        for horizon_name, horizon_days in horizons.items():
            log.info(f"  Horizon: {horizon_name} ({horizon_days} days)")
            model_forecasts = {}

            # 4a. ARIMA
            try:
                arima = ARIMAForecaster(order=(2, 1, 2))
                arima.fit(close)
                arima_fc = arima.forecast(steps=horizon_days)
                if not arima_fc.empty:
                    last_price = close.iloc[-1]
                    fc_price = arima_fc["Forecast"].iloc[-1]
                    expected_return = (fc_price / last_price) - 1
                    model_forecasts["ARIMA"] = {
                        "point": expected_return,
                        "lower": (arima_fc["Lower"].iloc[-1] / last_price) - 1,
                        "upper": (arima_fc["Upper"].iloc[-1] / last_price) - 1,
                        "direction": 1 if expected_return > 0.005 else (-1 if expected_return < -0.005 else 0),
                        "confidence": 0.6,
                    }
            except Exception as e:
                log.warning(f"ARIMA failed for {ticker}: {e}")

            # 4b. GARCH volatility forecast
            try:
                garch = fit_garch(returns)
                vol_fc = garch.forecast_volatility(steps=horizon_days)
                if not vol_fc.empty:
                    vol_estimate = vol_fc["Vol_Annual"].mean()
                    model_forecasts["GARCH"] = {
                        "point": 0.0,  # GARCH forecasts volatility, not direction
                        "lower": -vol_estimate * 1.96 / np.sqrt(252 / horizon_days),
                        "upper": vol_estimate * 1.96 / np.sqrt(252 / horizon_days),
                        "direction": 0,
                        "confidence": 0.5,
                        "volatility_annual": vol_estimate,
                    }
            except Exception as e:
                log.warning(f"GARCH failed for {ticker}: {e}")

            # 4c. ML model (XGBoost)
            try:
                ml_features = compute_for_ml(ohlcv).dropna()
                forward_returns = close.pct_change(horizon_days).shift(-horizon_days)

                if len(ml_features) > 300:
                    xgb = MLForecaster(model_type="xgboost")
                    xgb.fit(ml_features.iloc[:-horizon_days], forward_returns.iloc[:-horizon_days])
                    latest_features = ml_features.iloc[[-1]]
                    pred = xgb.predict(latest_features)

                    if not pred.empty:
                        direction = int(pred["Prediction"].iloc[0])
                        # Get probability of predicted direction
                        prob_col = f"Prob_{direction}"
                        confidence = pred[prob_col].iloc[0] if prob_col in pred.columns else 0.5

                        model_forecasts["XGBoost"] = {
                            "point": direction * 0.02 * confidence,
                            "lower": -0.05,
                            "upper": 0.05,
                            "direction": direction,
                            "confidence": confidence,
                        }
            except Exception as e:
                log.warning(f"XGBoost failed for {ticker}: {e}")

            # 5. Combine via ensemble
            if model_forecasts:
                for name in model_forecasts:
                    self.ensemble.register_model(name)

                combined = self.ensemble.combine_forecasts(model_forecasts, regime=current_regime)

                # 6. Catalyst adjustment
                forecast_date = pd.Timestamp.now() + pd.Timedelta(days=horizon_days)
                catalyst_mult = get_uncertainty_multiplier(forecast_date)
                if catalyst_mult > 1.0:
                    combined["lower"] *= catalyst_mult
                    combined["upper"] *= catalyst_mult
                    combined["catalyst_adjustment"] = catalyst_mult

                forecast_results[horizon_name] = combined
            else:
                forecast_results[horizon_name] = {
                    "point": 0.0, "direction": 0, "confidence": 0.0,
                    "note": "No models produced valid forecasts",
                }

        # 7. News sentiment overlay
        news_sentiment = 0.0
        if include_news:
            try:
                from data.news import fetch_all_news
                news = fetch_all_news()
                if not news.empty:
                    analyzed = analyze_news_batch(news)
                    # Filter to relevant news for this ticker
                    relevant = analyzed[
                        analyzed["Entities_Tickers"].apply(lambda t: ticker in t)
                    ]
                    if not relevant.empty:
                        news_sentiment = relevant["Sentiment"].mean()
            except Exception as e:
                log.warning(f"News sentiment failed: {e}")

        # 8. Build explanation
        explanation = explain_forecast(
            ticker=ticker,
            forecasts=forecast_results,
            regime=regime_info,
            news_sentiment=news_sentiment,
            tech_features=tech_features,
        )

        return {
            "ticker": ticker,
            "asset_type": "equity",
            "timestamp": pd.Timestamp.now().isoformat(),
            "current_price": float(close.iloc[-1]),
            "regime": regime_info,
            "forecasts": forecast_results,
            "news_sentiment": news_sentiment,
            "explanation": explanation,
        }

    def forecast_crypto(
        self,
        coin_id: str,
        horizons: dict[str, int] | None = None,
    ) -> dict:
        """Full forecast pipeline for a cryptocurrency."""
        from config.assets import CRYPTO_ALL
        horizons = horizons or HORIZONS

        symbol = CRYPTO_ALL.get(coin_id, {}).get("symbol", coin_id.upper())
        log.info(f"Generating crypto forecast for {symbol}")

        ohlcv = fetch_crypto_ohlcv(coin_id)
        if ohlcv.empty:
            return {"error": f"No data for {coin_id}"}

        close = ohlcv["Close"]
        returns = close.pct_change().dropna()

        # Regime detection (crypto-specific)
        regime_detector = self._load_regime_detector("crypto")
        btc = fetch_crypto_ohlcv("bitcoin")
        if not btc.empty:
            regime_info = regime_detector.current_regime(btc["Close"])
        else:
            regime_info = {"regime": "Unknown", "probabilities": {}}

        forecast_results = {}
        for horizon_name, horizon_days in horizons.items():
            model_forecasts = {}

            # ARIMA on price
            try:
                arima = ARIMAForecaster(order=(1, 1, 1))
                arima.fit(close)
                arima_fc = arima.forecast(steps=horizon_days)
                if not arima_fc.empty:
                    last_price = close.iloc[-1]
                    fc_price = arima_fc["Forecast"].iloc[-1]
                    expected_return = (fc_price / last_price) - 1
                    model_forecasts["ARIMA"] = {
                        "point": expected_return,
                        "lower": (arima_fc["Lower"].iloc[-1] / last_price) - 1,
                        "upper": (arima_fc["Upper"].iloc[-1] / last_price) - 1,
                        "direction": 1 if expected_return > 0.01 else (-1 if expected_return < -0.01 else 0),
                        "confidence": 0.5,
                    }
            except Exception as e:
                log.warning(f"ARIMA failed for {coin_id}: {e}")

            # GARCH
            try:
                garch = fit_garch(returns)
                vol_fc = garch.forecast_volatility(steps=horizon_days)
                if not vol_fc.empty:
                    vol_est = vol_fc["Vol_Annual"].mean()
                    model_forecasts["GARCH"] = {
                        "point": 0.0,
                        "lower": -vol_est * 1.96 / np.sqrt(252 / horizon_days),
                        "upper": vol_est * 1.96 / np.sqrt(252 / horizon_days),
                        "direction": 0,
                        "confidence": 0.5,
                        "volatility_annual": vol_est,
                    }
            except Exception as e:
                log.warning(f"GARCH failed for {coin_id}: {e}")

            if model_forecasts:
                for name in model_forecasts:
                    self.ensemble.register_model(name)
                combined = self.ensemble.combine_forecasts(
                    model_forecasts, regime=regime_info["regime"]
                )
                forecast_results[horizon_name] = combined

        return {
            "ticker": symbol,
            "coin_id": coin_id,
            "asset_type": "crypto",
            "timestamp": pd.Timestamp.now().isoformat(),
            "current_price": float(close.iloc[-1]),
            "regime": regime_info,
            "forecasts": forecast_results,
        }

    def forecast_sector(self, sector_name: str) -> dict:
        """Forecast a GICS sector using its ETF proxy."""
        from config.assets import SECTORS
        sector_info = SECTORS.get(sector_name)
        if not sector_info:
            return {"error": f"Unknown sector: {sector_name}"}

        result = self.forecast_equity(sector_info["etf"])
        result["sector"] = sector_name
        result["holdings"] = sector_info["holdings"]
        return result

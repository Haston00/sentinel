"""
SENTINEL — Prediction Tracker.
Persistent storage for every prediction the system makes.
When actuals come in, scores them and feeds data to the optimizer.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import CACHE_DIR
from utils.logger import get_logger

log = get_logger("learning.tracker")

LEARNING_DIR = Path(CACHE_DIR).parent.parent / "learning" / "data"
LEARNING_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = LEARNING_DIR / "predictions.parquet"
SCORES_FILE = LEARNING_DIR / "scores.parquet"
WEIGHTS_HISTORY_FILE = LEARNING_DIR / "weight_history.parquet"

# Schema for prediction records
PREDICTION_COLS = [
    "prediction_id", "timestamp", "asset_ticker", "asset_type",
    "horizon", "horizon_days", "maturity_date",
    "predicted_direction", "predicted_return", "predicted_lower", "predicted_upper",
    "confidence", "regime", "n_models",
    "actual_return", "actual_direction", "scored",
    "correct_direction", "abs_error", "within_ci",
]


class PredictionTracker:
    """
    Logs every forecast, matches with actuals, and maintains the learning database.
    All data persists to Parquet files.
    """

    def __init__(self):
        self._predictions = self._load_predictions()

    def _load_predictions(self) -> pd.DataFrame:
        """Load existing predictions from disk."""
        if PREDICTIONS_FILE.exists():
            try:
                df = pd.read_parquet(PREDICTIONS_FILE)
                log.info(f"Loaded {len(df)} historical predictions")
                return df
            except Exception as e:
                log.warning(f"Failed to load predictions: {e}")
        return pd.DataFrame(columns=PREDICTION_COLS)

    def _save_predictions(self) -> None:
        """Persist predictions to disk."""
        self._predictions.to_parquet(PREDICTIONS_FILE, index=False)

    def log_prediction(
        self,
        asset_ticker: str,
        asset_type: str,
        horizon: str,
        horizon_days: int,
        predicted_direction: int,
        predicted_return: float,
        predicted_lower: float,
        predicted_upper: float,
        confidence: float,
        regime: str,
        n_models: int,
        model_details: dict | None = None,
    ) -> str:
        """
        Log a new prediction. Returns prediction_id.
        Called automatically by the forecast engine after every forecast.
        """
        now = pd.Timestamp.now()
        maturity = now + pd.Timedelta(days=horizon_days)
        pred_id = str(uuid.uuid4())[:12]

        record = {
            "prediction_id": pred_id,
            "timestamp": now,
            "asset_ticker": asset_ticker,
            "asset_type": asset_type,
            "horizon": horizon,
            "horizon_days": horizon_days,
            "maturity_date": maturity,
            "predicted_direction": predicted_direction,
            "predicted_return": predicted_return,
            "predicted_lower": predicted_lower,
            "predicted_upper": predicted_upper,
            "confidence": confidence,
            "regime": regime,
            "n_models": n_models,
            "actual_return": np.nan,
            "actual_direction": np.nan,
            "scored": False,
            "correct_direction": np.nan,
            "abs_error": np.nan,
            "within_ci": np.nan,
        }

        new_row = pd.DataFrame([record])
        self._predictions = pd.concat([self._predictions, new_row], ignore_index=True)
        self._save_predictions()

        log.info(
            f"Logged prediction {pred_id}: {asset_ticker} {horizon} "
            f"dir={predicted_direction} conf={confidence:.1%} regime={regime}"
        )
        return pred_id

    def score_matured_predictions(self) -> int:
        """
        Find predictions whose maturity_date has passed, fetch actual prices,
        compute accuracy, and update records. Returns count of newly scored.
        """
        now = pd.Timestamp.now()
        unscored = self._predictions[
            (~self._predictions["scored"]) &
            (self._predictions["maturity_date"] <= now)
        ]

        if unscored.empty:
            log.info("No matured predictions to score")
            return 0

        scored_count = 0
        from data.stocks import fetch_ohlcv
        from data.crypto import fetch_crypto_ohlcv

        for idx, row in unscored.iterrows():
            try:
                ticker = row["asset_ticker"]
                pred_date = pd.Timestamp(row["timestamp"])
                maturity = pd.Timestamp(row["maturity_date"])

                # Fetch actual price data
                if row["asset_type"] == "crypto":
                    from config.assets import CRYPTO_ALL
                    coin_id = None
                    for cid, info in CRYPTO_ALL.items():
                        if info["symbol"] == ticker:
                            coin_id = cid
                            break
                    if coin_id:
                        ohlcv = fetch_crypto_ohlcv(coin_id)
                    else:
                        continue
                else:
                    ohlcv = fetch_ohlcv(ticker)

                if ohlcv.empty:
                    continue

                close = ohlcv["Close"]

                # Find prices closest to prediction date and maturity date
                start_prices = close[close.index >= pred_date]
                end_prices = close[close.index >= maturity]

                if start_prices.empty or end_prices.empty:
                    continue

                start_price = float(start_prices.iloc[0])
                end_price = float(end_prices.iloc[0])
                actual_return = (end_price / start_price) - 1
                actual_direction = 1 if actual_return > 0.005 else (-1 if actual_return < -0.005 else 0)

                # Score it
                pred_dir = int(row["predicted_direction"])
                correct = pred_dir == actual_direction
                abs_error = abs(float(row["predicted_return"]) - actual_return)
                within_ci = float(row["predicted_lower"]) <= actual_return <= float(row["predicted_upper"])

                self._predictions.at[idx, "actual_return"] = actual_return
                self._predictions.at[idx, "actual_direction"] = actual_direction
                self._predictions.at[idx, "scored"] = True
                self._predictions.at[idx, "correct_direction"] = correct
                self._predictions.at[idx, "abs_error"] = abs_error
                self._predictions.at[idx, "within_ci"] = within_ci
                scored_count += 1

                log.info(
                    f"Scored {row['prediction_id']}: {ticker} {row['horizon']} "
                    f"pred={pred_dir} actual={actual_direction} "
                    f"correct={correct} error={abs_error:.4f}"
                )

            except Exception as e:
                log.warning(f"Failed to score prediction {row['prediction_id']}: {e}")

        if scored_count > 0:
            self._save_predictions()
            log.info(f"Scored {scored_count} matured predictions")

        return scored_count

    def get_all_predictions(self) -> pd.DataFrame:
        """Return all predictions."""
        return self._predictions.copy()

    def get_scored_predictions(self) -> pd.DataFrame:
        """Return only scored predictions."""
        return self._predictions[self._predictions["scored"]].copy()

    def get_pending_predictions(self) -> pd.DataFrame:
        """Return predictions still awaiting actuals."""
        return self._predictions[~self._predictions["scored"]].copy()

    def get_accuracy_summary(self) -> dict:
        """Compute overall accuracy metrics from scored predictions."""
        scored = self.get_scored_predictions()
        if scored.empty:
            return {"total_predictions": len(self._predictions), "scored": 0}

        n = len(scored)
        correct = scored["correct_direction"].sum()
        hit_rate = correct / n if n > 0 else 0
        avg_error = scored["abs_error"].mean()
        ci_rate = scored["within_ci"].mean()

        # Confidence calibration: bin by confidence, check actual accuracy
        calibration = {}
        for bucket_name, low, high in [("Low", 0, 0.4), ("Medium", 0.4, 0.65), ("High", 0.65, 1.01)]:
            bucket = scored[(scored["confidence"] >= low) & (scored["confidence"] < high)]
            if len(bucket) > 0:
                calibration[bucket_name] = {
                    "count": len(bucket),
                    "avg_confidence": float(bucket["confidence"].mean()),
                    "actual_accuracy": float(bucket["correct_direction"].mean()),
                }

        # By regime
        regime_accuracy = {}
        for regime in scored["regime"].dropna().unique():
            regime_df = scored[scored["regime"] == regime]
            regime_accuracy[regime] = {
                "count": len(regime_df),
                "hit_rate": float(regime_df["correct_direction"].mean()),
                "avg_error": float(regime_df["abs_error"].mean()),
            }

        # By horizon
        horizon_accuracy = {}
        for horizon in scored["horizon"].unique():
            h_df = scored[scored["horizon"] == horizon]
            horizon_accuracy[horizon] = {
                "count": len(h_df),
                "hit_rate": float(h_df["correct_direction"].mean()),
                "avg_error": float(h_df["abs_error"].mean()),
            }

        # By asset type
        asset_accuracy = {}
        for atype in scored["asset_type"].unique():
            a_df = scored[scored["asset_type"] == atype]
            asset_accuracy[atype] = {
                "count": len(a_df),
                "hit_rate": float(a_df["correct_direction"].mean()),
                "avg_error": float(a_df["abs_error"].mean()),
            }

        return {
            "total_predictions": len(self._predictions),
            "scored": n,
            "pending": len(self._predictions) - n,
            "hit_rate": round(hit_rate, 4),
            "hit_rate_pct": f"{hit_rate:.1%}",
            "avg_abs_error": round(avg_error, 6),
            "ci_capture_rate": round(ci_rate, 4),
            "calibration": calibration,
            "by_regime": regime_accuracy,
            "by_horizon": horizon_accuracy,
            "by_asset_type": asset_accuracy,
        }

    def get_accuracy_over_time(self, window: int = 20) -> pd.DataFrame:
        """
        Compute rolling accuracy over time to visualize learning progress.
        """
        scored = self.get_scored_predictions()
        if len(scored) < window:
            return pd.DataFrame()

        scored = scored.sort_values("timestamp")
        scored["rolling_hit_rate"] = (
            scored["correct_direction"].rolling(window, min_periods=5).mean()
        )
        scored["rolling_error"] = (
            scored["abs_error"].rolling(window, min_periods=5).mean()
        )
        scored["rolling_ci_rate"] = (
            scored["within_ci"].astype(float).rolling(window, min_periods=5).mean()
        )

        return scored[["timestamp", "rolling_hit_rate", "rolling_error", "rolling_ci_rate"]].dropna()

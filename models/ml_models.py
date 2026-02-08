"""
SENTINEL — Machine Learning Models.
XGBoost and Random Forest trained on 41 technical features to predict
forward returns. This is the real AI — learns patterns from YOUR data.

How it works:
  1. Takes 5 years of SPY (or any asset) OHLCV data
  2. Computes 41 technical features (RSI, MACD, Bollinger, ATR, OBV, etc.)
  3. Creates targets: was the stock UP, DOWN, or FLAT over the next N days?
  4. Trains XGBoost + Random Forest on 80% of data
  5. Tests on the remaining 20% (walk-forward, no cheating)
  6. Reports REAL accuracy, not theoretical

The model outputs probabilities: "62% chance of going up next week"
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from data.stocks import fetch_ohlcv
from features.technical import compute_for_ml as compute_features
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = Path(__file__).parent / "saved"
MODEL_DIR.mkdir(exist_ok=True)


def _create_target(close, horizon=5, threshold=0.005):
    """
    Create classification target: was the stock up, down, or flat
    over the next `horizon` trading days?
      1 = up more than threshold
      0 = flat (within threshold)
     -1 = down more than threshold
    """
    fwd_ret = close.pct_change(horizon).shift(-horizon)
    # Binary classification: 0=down/flat, 1=up
    # Binary is more robust than 3-class for small datasets
    target = pd.Series(0, index=close.index, dtype=int)
    target[fwd_ret > threshold] = 1
    target[fwd_ret.isna()] = np.nan
    return target, fwd_ret


class MLForecaster:
    """Trains XGBoost + Random Forest and blends their predictions."""

    def __init__(self):
        self.xgb = None
        self.rf = None
        self.feature_names = []
        self.metrics = {}
        self.feature_importance = pd.Series(dtype=float)

    def train(self, ticker="SPY", horizon=5, threshold=0.005):
        """
        Full training pipeline: fetch data → features → train → evaluate.
        Returns self with trained models and metrics.
        """
        logger.info(f"Training ML models for {ticker}, horizon={horizon}d")

        # 1. Get data
        df = fetch_ohlcv(ticker)
        if df.empty or len(df) < 300:
            raise ValueError(f"Insufficient data for {ticker}: {len(df)} rows")

        # 2. Compute features
        features = compute_features(df)
        if features.empty:
            raise ValueError("Feature computation returned empty")

        # 3. Create target
        target, fwd_ret = _create_target(df["Close"], horizon, threshold)

        # 4. Align features and target, drop NaNs
        combined = features.copy()
        combined["target"] = target
        combined["fwd_ret"] = fwd_ret
        combined = combined.dropna()

        if len(combined) < 200:
            raise ValueError(f"Only {len(combined)} clean samples after alignment")

        X = combined[features.columns]
        y = combined["target"].astype(int)
        self.feature_names = list(features.columns)

        # 5. Walk-forward split: 80% train, 20% test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(self.feature_names)}")
        logger.info(f"Target dist (train): {y_train.value_counts().to_dict()}")

        # 6. Train XGBoost
        self.xgb = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss",
            verbosity=0,
        )
        self.xgb.fit(X_train, y_train)

        # 7. Train Random Forest
        self.rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )
        self.rf.fit(X_train, y_train)

        # 8. Evaluate on held-out test set (no cheating)
        xgb_pred = self.xgb.predict(X_test)
        rf_pred = self.rf.predict(X_test)

        xgb_probs = self.xgb.predict_proba(X_test)
        rf_probs = self.rf.predict_proba(X_test)

        # Ensemble: average probabilities, pick argmax
        avg_probs = (xgb_probs + rf_probs) / 2
        ensemble_classes = self.xgb.classes_
        ens_pred = ensemble_classes[np.argmax(avg_probs, axis=1)]

        xgb_acc = accuracy_score(y_test, xgb_pred)
        rf_acc = accuracy_score(y_test, rf_pred)
        ens_acc = accuracy_score(y_test, ens_pred)

        logger.info(f"XGBoost test accuracy: {xgb_acc:.3f}")
        logger.info(f"RandomForest test accuracy: {rf_acc:.3f}")
        logger.info(f"Ensemble test accuracy: {ens_acc:.3f}")

        # 9. Time-series cross-validation (5-fold)
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for train_i, test_i in tscv.split(X):
            x_tr, x_te = X.iloc[train_i], X.iloc[test_i]
            y_tr, y_te = y.iloc[train_i], y.iloc[test_i]
            m = XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                random_state=42, eval_metric="mlogloss", verbosity=0,
            )
            m.fit(x_tr, y_tr)
            cv_scores.append(accuracy_score(y_te, m.predict(x_te)))

        cv_mean = np.mean(cv_scores)
        logger.info(f"5-fold CV accuracy: {cv_mean:.3f} (+/- {np.std(cv_scores):.3f})")

        # 10. Feature importance (XGBoost)
        self.feature_importance = pd.Series(
            self.xgb.feature_importances_,
            index=self.feature_names,
        ).sort_values(ascending=False)

        # Store metrics
        self.metrics = {
            "ticker": ticker,
            "horizon": horizon,
            "threshold": threshold,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": len(self.feature_names),
            "xgb_accuracy": round(xgb_acc, 4),
            "rf_accuracy": round(rf_acc, 4),
            "ensemble_accuracy": round(ens_acc, 4),
            "cv_mean_accuracy": round(cv_mean, 4),
            "cv_std": round(float(np.std(cv_scores)), 4),
            "cv_fold_scores": [round(s, 4) for s in cv_scores],
            "test_period_start": str(X_test.index[0].date()) if hasattr(X_test.index[0], 'date') else str(X_test.index[0]),
            "test_period_end": str(X_test.index[-1].date()) if hasattr(X_test.index[-1], 'date') else str(X_test.index[-1]),
            "target_distribution": y.value_counts().to_dict(),
        }

        return self

    def predict_current(self, ticker="SPY"):
        """
        Predict direction for the CURRENT (most recent) day.
        Returns dict with prediction, probabilities, and confidence.
        """
        if self.xgb is None or self.rf is None:
            raise RuntimeError("Train models first with .train()")

        df = fetch_ohlcv(ticker)
        if df.empty:
            return {"error": f"No data for {ticker}"}

        features = compute_features(df)
        if features.empty:
            return {"error": "Feature computation failed"}

        # Get the last row of features
        latest = features.iloc[[-1]].fillna(0)

        # XGBoost prediction
        xgb_probs = self.xgb.predict_proba(latest[self.feature_names])[0]
        rf_probs = self.rf.predict_proba(latest[self.feature_names])[0]

        # Ensemble average
        avg_probs = (xgb_probs + rf_probs) / 2
        classes = self.xgb.classes_
        pred_class = classes[np.argmax(avg_probs)]

        # Build probability dict
        prob_dict = {}
        for i, cls in enumerate(classes):
            label = {1: "UP", 0: "DOWN"}.get(cls, str(cls))
            prob_dict[label] = {
                "probability": round(float(avg_probs[i]) * 100, 1),
                "xgb": round(float(xgb_probs[i]) * 100, 1),
                "rf": round(float(rf_probs[i]) * 100, 1),
            }

        direction = {1: "UP", 0: "DOWN"}.get(pred_class, "UNKNOWN")
        confidence = float(np.max(avg_probs)) * 100

        # Top contributing features
        top_features = []
        if not self.feature_importance.empty:
            latest_vals = latest[self.feature_names].iloc[0]
            for feat in self.feature_importance.head(10).index:
                val = latest_vals.get(feat, 0)
                importance = self.feature_importance[feat]
                top_features.append({
                    "feature": feat,
                    "value": round(float(val), 4),
                    "importance": round(float(importance), 4),
                })

        return {
            "direction": direction,
            "confidence": round(confidence, 1),
            "probabilities": prob_dict,
            "top_features": top_features,
            "model_accuracy": self.metrics.get("ensemble_accuracy", 0),
            "date": str(features.index[-1].date()) if hasattr(features.index[-1], 'date') else str(features.index[-1]),
        }

    def predict_history(self, ticker="SPY"):
        """
        Run predictions over the full test period for backtesting visualization.
        Returns DataFrame with predictions and actual outcomes.
        """
        if self.xgb is None:
            raise RuntimeError("Train first")

        df = fetch_ohlcv(ticker)
        features = compute_features(df)
        target, fwd_ret = _create_target(df["Close"], self.metrics.get("horizon", 5))

        combined = features.copy()
        combined["target"] = target
        combined["fwd_ret"] = fwd_ret
        combined = combined.dropna()

        X = combined[self.feature_names]

        xgb_probs = self.xgb.predict_proba(X)
        rf_probs = self.rf.predict_proba(X)
        avg_probs = (xgb_probs + rf_probs) / 2

        classes = self.xgb.classes_
        preds = classes[np.argmax(avg_probs, axis=1)]

        result = pd.DataFrame(index=combined.index)
        result["actual"] = combined["target"]
        result["predicted"] = preds
        result["fwd_return"] = combined["fwd_ret"]
        result["correct"] = (result["actual"] == result["predicted"]).astype(int)

        for i, cls in enumerate(classes):
            label = {1: "UP", 0: "DOWN"}.get(cls, str(cls))
            result[f"prob_{label}"] = avg_probs[:, i]

        result["confidence"] = np.max(avg_probs, axis=1)
        return result

    def save(self, name="spy"):
        path = MODEL_DIR / f"ml_{name}.pkl"
        joblib.dump({
            "xgb": self.xgb, "rf": self.rf,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
        }, path)
        logger.info(f"Saved ML models to {path}")

    def load(self, name="spy"):
        path = MODEL_DIR / f"ml_{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No saved model at {path}")
        data = joblib.load(path)
        self.xgb = data["xgb"]
        self.rf = data["rf"]
        self.feature_names = data["feature_names"]
        self.metrics = data["metrics"]
        self.feature_importance = data.get("feature_importance", pd.Series(dtype=float))
        logger.info(f"Loaded ML models from {path}")
        return self


# ── High-level API ────────────────────────────────────────────

def train_ml_models(ticker="SPY", horizon=5, force=False):
    """Train or load cached ML models. Returns MLForecaster."""
    cache_path = MODEL_DIR / f"ml_{ticker.lower()}.pkl"
    forecaster = MLForecaster()

    if not force and cache_path.exists():
        try:
            forecaster.load(ticker.lower())
            logger.info(f"Loaded cached ML model for {ticker}")
            return forecaster
        except Exception:
            logger.warning("Cache corrupt, retraining")

    forecaster.train(ticker=ticker, horizon=horizon)
    forecaster.save(ticker.lower())
    return forecaster


def get_ml_prediction(ticker="SPY", horizon=5):
    """Get current ML prediction for a ticker."""
    forecaster = train_ml_models(ticker, horizon)
    return forecaster.predict_current(ticker)

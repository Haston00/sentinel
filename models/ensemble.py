"""
SENTINEL — Adaptive ensemble meta-learner.
Combines forecasts from multiple model families, weighting by recent performance.
"""

import numpy as np
import pandas as pd

from config.settings import ENSEMBLE_EVAL_WINDOW
from utils.logger import get_logger

log = get_logger("models.ensemble")


class AdaptiveEnsemble:
    """
    Combines forecasts from multiple models using adaptive weighting.
    Models that performed better recently get more weight.
    Regime-aware: weights shift based on current market regime.
    """

    def __init__(self, eval_window: int = ENSEMBLE_EVAL_WINDOW):
        self.eval_window = eval_window
        self.model_weights: dict[str, float] = {}
        self.model_history: dict[str, list[dict]] = {}

    def register_model(self, name: str, initial_weight: float = 1.0) -> None:
        """Register a model in the ensemble."""
        self.model_weights[name] = initial_weight
        self.model_history[name] = []
        log.info(f"Registered model: {name} (weight={initial_weight})")

    def record_forecast(
        self,
        model_name: str,
        date: str,
        predicted: float,
        actual: float | None = None,
    ) -> None:
        """Record a forecast for later evaluation."""
        if model_name not in self.model_history:
            self.register_model(model_name)

        self.model_history[model_name].append({
            "date": date,
            "predicted": predicted,
            "actual": actual,
        })

    def update_weights(self) -> dict[str, float]:
        """
        Update model weights based on recent forecast accuracy.
        Uses inverse MAE weighting over the evaluation window.
        """
        model_errors = {}

        for name, history in self.model_history.items():
            # Filter to records with actual values
            evaluated = [h for h in history if h["actual"] is not None]
            recent = evaluated[-self.eval_window:]

            if len(recent) < 10:
                model_errors[name] = float("inf")
                continue

            errors = [abs(h["predicted"] - h["actual"]) for h in recent]
            mae = np.mean(errors)
            model_errors[name] = max(mae, 1e-10)  # Avoid division by zero

        if not model_errors:
            return self.model_weights

        # Inverse MAE weighting
        inv_errors = {name: 1.0 / err for name, err in model_errors.items() if err != float("inf")}
        total_inv = sum(inv_errors.values())

        if total_inv > 0:
            self.model_weights = {name: inv / total_inv for name, inv in inv_errors.items()}
        else:
            # Equal weights if no valid evaluations
            n = len(self.model_weights)
            self.model_weights = {name: 1.0 / n for name in self.model_weights}

        log.info(f"Updated weights: {', '.join(f'{k}={v:.3f}' for k, v in self.model_weights.items())}")
        return self.model_weights

    def combine_forecasts(
        self,
        forecasts: dict[str, dict],
        regime: str | None = None,
    ) -> dict:
        """
        Combine multiple model forecasts into a single ensemble forecast.

        forecasts: dict of model_name -> {
            "point": float (point forecast),
            "lower": float (CI lower),
            "upper": float (CI upper),
            "direction": int (-1, 0, 1),
            "confidence": float (0-1),
        }

        Returns combined forecast with weighted average and uncertainty.
        """
        if not forecasts:
            return {"point": 0.0, "lower": 0.0, "upper": 0.0, "direction": 0, "confidence": 0.0}

        # Ensure all models have weights
        for name in forecasts:
            if name not in self.model_weights:
                self.register_model(name)

        # Normalize weights for only the models that produced forecasts
        active_weights = {name: self.model_weights.get(name, 0.0) for name in forecasts}
        total_weight = sum(active_weights.values())
        if total_weight == 0:
            total_weight = len(active_weights)
            active_weights = {name: 1.0 / total_weight for name in active_weights}
        else:
            active_weights = {name: w / total_weight for name, w in active_weights.items()}

        # Regime adjustment: in transition regime, increase uncertainty
        regime_multiplier = 1.0
        if regime == "Transition":
            regime_multiplier = 1.3  # Widen confidence intervals
        elif regime == "Bear":
            regime_multiplier = 1.15

        # Weighted point forecast
        point = sum(
            active_weights[name] * fc["point"]
            for name, fc in forecasts.items()
        )

        # Weighted confidence intervals (use widest bounds, adjusted by weight)
        lowers = [fc.get("lower", fc["point"]) for fc in forecasts.values()]
        uppers = [fc.get("upper", fc["point"]) for fc in forecasts.values()]
        lower = sum(
            active_weights[name] * fc.get("lower", fc["point"])
            for name, fc in forecasts.items()
        ) * regime_multiplier
        upper = sum(
            active_weights[name] * fc.get("upper", fc["point"])
            for name, fc in forecasts.items()
        ) * regime_multiplier

        # Weighted direction consensus
        direction_votes = {}
        for name, fc in forecasts.items():
            d = fc.get("direction", 0)
            direction_votes[d] = direction_votes.get(d, 0) + active_weights[name]
        direction = max(direction_votes, key=direction_votes.get)

        # Consensus confidence = weight of winning direction
        confidence = direction_votes[direction]

        # Agreement bonus: if all models agree, boost confidence
        if len(set(fc.get("direction", 0) for fc in forecasts.values())) == 1:
            confidence = min(confidence * 1.2, 1.0)

        return {
            "point": round(point, 6),
            "lower": round(lower, 6),
            "upper": round(upper, 6),
            "direction": direction,
            "direction_label": {-1: "Bearish", 0: "Neutral", 1: "Bullish"}.get(direction, "Neutral"),
            "confidence": round(confidence, 4),
            "model_weights": active_weights,
            "regime_adjustment": regime_multiplier,
            "n_models": len(forecasts),
        }

    def get_model_performance(self) -> pd.DataFrame:
        """Get performance summary for each registered model."""
        rows = []
        for name, history in self.model_history.items():
            evaluated = [h for h in history if h["actual"] is not None]
            if not evaluated:
                rows.append({"Model": name, "N_Forecasts": 0})
                continue

            errors = [abs(h["predicted"] - h["actual"]) for h in evaluated]
            correct_direction = sum(
                1 for h in evaluated
                if (h["predicted"] > 0 and h["actual"] > 0)
                or (h["predicted"] < 0 and h["actual"] < 0)
                or (h["predicted"] == 0 and abs(h["actual"]) < 0.005)
            )

            rows.append({
                "Model": name,
                "N_Forecasts": len(evaluated),
                "MAE": np.mean(errors),
                "Hit_Rate": correct_direction / len(evaluated),
                "Weight": self.model_weights.get(name, 0.0),
            })

        return pd.DataFrame(rows).set_index("Model") if rows else pd.DataFrame()

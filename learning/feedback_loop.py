"""
SENTINEL — Feedback Loop Controller.
Orchestrates the full learning cycle: score matured predictions → optimize → update weights.
Can run on-demand or scheduled.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.settings import CACHE_DIR
from learning.prediction_tracker import PredictionTracker
from learning.optimizer import LearningOptimizer
from utils.logger import get_logger

log = get_logger("learning.feedback")

LEARNING_DIR = Path(CACHE_DIR).parent.parent / "learning" / "data"
FEEDBACK_LOG_FILE = LEARNING_DIR / "feedback_log.json"


class FeedbackLoop:
    """
    Runs the complete learning cycle:
    1. Score any matured predictions (fetch actuals, compare to predicted)
    2. Run the optimizer on all scored data
    3. Update model weights and bias corrections
    4. Log what was learned
    """

    def __init__(self):
        self.tracker = PredictionTracker()
        self.optimizer = LearningOptimizer()

    def run_cycle(self) -> dict:
        """
        Execute one full feedback cycle.
        Call this daily or whenever you want the system to learn.
        Returns summary of what happened.
        """
        log.info("Starting feedback cycle...")
        cycle_start = datetime.now()

        # Step 1: Score matured predictions
        newly_scored = self.tracker.score_matured_predictions()
        log.info(f"Step 1: Scored {newly_scored} matured predictions")

        # Step 2: Get all scored data
        scored_df = self.tracker.get_scored_predictions()
        total_scored = len(scored_df)
        log.info(f"Step 2: {total_scored} total scored predictions available")

        # Step 3: Run optimizer
        if total_scored >= 10:
            optimization_results = self.optimizer.optimize(scored_df)
            log.info("Step 3: Optimization complete")
        else:
            optimization_results = {"status": "waiting_for_more_data", "need": 10 - total_scored}
            log.info(f"Step 3: Need {10 - total_scored} more scored predictions before optimization")

        # Step 4: Get accuracy summary
        accuracy = self.tracker.get_accuracy_summary()

        # Step 5: Log this cycle
        cycle_result = {
            "timestamp": cycle_start.isoformat(),
            "newly_scored": newly_scored,
            "total_scored": total_scored,
            "total_pending": accuracy.get("pending", 0),
            "hit_rate": accuracy.get("hit_rate_pct", "N/A"),
            "avg_error": accuracy.get("avg_abs_error", 0),
            "optimization": optimization_results,
        }

        self._log_cycle(cycle_result)
        log.info(
            f"Feedback cycle complete: scored={newly_scored}, "
            f"total={total_scored}, hit_rate={accuracy.get('hit_rate_pct', 'N/A')}"
        )

        return cycle_result

    def get_dashboard_data(self) -> dict:
        """
        Get all data needed for the learning dashboard page.
        """
        accuracy = self.tracker.get_accuracy_summary()
        learning_curve = self.tracker.get_accuracy_over_time(window=20)
        pending = self.tracker.get_pending_predictions()
        scored = self.tracker.get_scored_predictions()
        learning_summary = self.optimizer.get_learning_summary()
        cycle_log = self._load_cycle_log()

        return {
            "accuracy": accuracy,
            "learning_curve": learning_curve,
            "pending_predictions": pending,
            "scored_predictions": scored,
            "learning_summary": learning_summary,
            "cycle_history": cycle_log,
        }

    def _log_cycle(self, result: dict) -> None:
        """Append cycle result to the feedback log."""
        existing = self._load_cycle_log()
        existing.append(result)
        # Keep last 100 cycles
        existing = existing[-100:]
        with open(FEEDBACK_LOG_FILE, "w") as f:
            json.dump(existing, f, indent=2, default=str)

    @staticmethod
    def _load_cycle_log() -> list:
        if FEEDBACK_LOG_FILE.exists():
            try:
                with open(FEEDBACK_LOG_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

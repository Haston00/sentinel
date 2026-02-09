"""
SENTINEL — Scheduled Auto-Learning.
Automated nightly feedback cycle: score matured predictions,
run optimizer, update pattern memory, log results.

Runs as a background thread within the Streamlit app.
Can also be run standalone via command line.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.settings import CACHE_DIR
from utils.logger import get_logger

log = get_logger("learning.scheduler")

SCHEDULER_DIR = Path(CACHE_DIR).parent.parent / "learning" / "data"
SCHEDULER_DIR.mkdir(parents=True, exist_ok=True)
SCHEDULER_LOG_FILE = SCHEDULER_DIR / "scheduler_log.json"
SCHEDULER_STATE_FILE = SCHEDULER_DIR / "scheduler_state.json"


class LearningScheduler:
    """
    Automated learning cycle that runs on a schedule.
    Default: every 6 hours during market hours, full cycle at market close.
    """

    def __init__(self, interval_hours: float = 6):
        self.interval_hours = interval_hours
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_run = self._load_last_run()

    def _load_last_run(self) -> str | None:
        if SCHEDULER_STATE_FILE.exists():
            try:
                with open(SCHEDULER_STATE_FILE) as f:
                    state = json.load(f)
                    return state.get("last_run")
            except Exception:
                pass
        return None

    def _save_state(self, last_run: str, status: str) -> None:
        state = {
            "last_run": last_run,
            "status": status,
            "interval_hours": self.interval_hours,
        }
        with open(SCHEDULER_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    def run_cycle(self) -> dict:
        """
        Run one complete learning cycle.
        1. Score matured predictions against actual results
        2. Run the optimizer on all scored data
        3. Update pattern memory with latest market data
        4. Log everything
        """
        log.info("Starting learning cycle...")
        cycle_start = datetime.now()
        results = {"timestamp": cycle_start.isoformat(), "steps": {}}

        # ── Step 1: Score Matured Predictions ─────────────────────
        try:
            from learning.prediction_tracker import PredictionTracker
            tracker = PredictionTracker()
            scored = tracker.score_matured_predictions()
            results["steps"]["score_predictions"] = {
                "status": "success",
                "new_scored": scored,
            }
            log.info(f"Scored {scored} matured predictions")
        except Exception as e:
            results["steps"]["score_predictions"] = {
                "status": "error",
                "error": str(e),
            }
            log.warning(f"Prediction scoring failed: {e}")

        # ── Step 2: Run Optimizer ──────────────────────────────────
        try:
            from learning.feedback_loop import FeedbackLoop
            loop = FeedbackLoop()
            cycle_results = loop.run_cycle()
            results["steps"]["optimizer"] = {
                "status": "success",
                "details": {
                    k: v for k, v in cycle_results.items()
                    if k in ("improvement", "n_alerts", "system_healthy")
                },
            }
            log.info("Optimizer cycle complete")
        except Exception as e:
            results["steps"]["optimizer"] = {
                "status": "error",
                "error": str(e),
            }
            log.warning(f"Optimizer failed: {e}")

        # ── Step 3: Update Pattern Memory ─────────────────────────
        try:
            from learning.pattern_memory import PatternMemory
            memory = PatternMemory()
            if memory.get_memory_stats()["history_days"] > 0:
                new_days = memory.update_history(max_years=1)  # Just latest data
                results["steps"]["pattern_memory"] = {
                    "status": "success",
                    "new_days": new_days,
                }
                log.info(f"Pattern memory updated: {new_days} new days")
            else:
                results["steps"]["pattern_memory"] = {
                    "status": "skipped",
                    "reason": "No existing history — needs initial build",
                }
        except Exception as e:
            results["steps"]["pattern_memory"] = {
                "status": "error",
                "error": str(e),
            }
            log.warning(f"Pattern memory update failed: {e}")

        # ── Step 4: Log Results ───────────────────────────────────
        elapsed = (datetime.now() - cycle_start).total_seconds()
        results["elapsed_seconds"] = round(elapsed, 1)
        results["status"] = "complete"

        self._log_cycle(results)
        self._save_state(cycle_start.isoformat(), "complete")
        self._last_run = cycle_start.isoformat()

        log.info(f"Learning cycle complete in {elapsed:.1f}s")
        return results

    def _log_cycle(self, results: dict) -> None:
        """Append cycle results to log file."""
        existing = []
        if SCHEDULER_LOG_FILE.exists():
            try:
                with open(SCHEDULER_LOG_FILE) as f:
                    existing = json.load(f)
            except Exception:
                pass

        existing.append(results)
        existing = existing[-100:]  # Keep last 100 cycles

        with open(SCHEDULER_LOG_FILE, "w") as f:
            json.dump(existing, f, indent=2, default=str)

    def start_background(self) -> None:
        """Start the scheduler as a background thread."""
        if self._running:
            log.info("Scheduler already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        log.info(f"Learning scheduler started (interval: {self.interval_hours}h)")

    def stop(self) -> None:
        """Stop the background scheduler."""
        self._running = False
        log.info("Learning scheduler stopped")

    def _run_loop(self) -> None:
        """Background loop that runs learning cycles on schedule."""
        while self._running:
            try:
                # Check if it's time to run
                if self._should_run():
                    self.run_cycle()
            except Exception as e:
                log.error(f"Scheduler error: {e}")

            # Sleep in small increments so we can stop quickly
            for _ in range(int(self.interval_hours * 3600 / 10)):
                if not self._running:
                    break
                time.sleep(10)

    def _should_run(self) -> bool:
        """Check if enough time has passed since last run."""
        if self._last_run is None:
            return True

        try:
            last = pd.Timestamp(self._last_run)
            elapsed_hours = (pd.Timestamp.now() - last).total_seconds() / 3600
            return elapsed_hours >= self.interval_hours
        except Exception:
            return True

    def get_status(self) -> dict:
        """Get current scheduler status."""
        cycle_log = []
        if SCHEDULER_LOG_FILE.exists():
            try:
                with open(SCHEDULER_LOG_FILE) as f:
                    cycle_log = json.load(f)
            except Exception:
                pass

        return {
            "running": self._running,
            "interval_hours": self.interval_hours,
            "last_run": self._last_run,
            "total_cycles": len(cycle_log),
            "recent_cycles": cycle_log[-5:] if cycle_log else [],
        }


# Singleton instance for use across the app
_scheduler_instance: LearningScheduler | None = None


def get_scheduler() -> LearningScheduler:
    """Get or create the global scheduler instance."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = LearningScheduler(interval_hours=6)
    return _scheduler_instance

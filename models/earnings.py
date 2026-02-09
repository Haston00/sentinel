"""
SENTINEL — Earnings Surprise Model.
Tracks actual vs estimated earnings, learns stock-specific reaction patterns
to beats/misses, and generates predictive signals for earnings season.

Hedge fund edge: Stocks don't react linearly to earnings. A 10% beat might
move a stock 2% if expectations were already priced in. This model learns
each stock's specific sensitivity to surprises.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import CACHE_DIR
from utils.logger import get_logger

log = get_logger("models.earnings")

EARNINGS_DIR = Path(CACHE_DIR).parent.parent / "learning" / "data" / "earnings"
EARNINGS_DIR.mkdir(parents=True, exist_ok=True)
EARNINGS_DB_FILE = EARNINGS_DIR / "earnings_history.parquet"
REACTION_PATTERNS_FILE = EARNINGS_DIR / "reaction_patterns.json"


def fetch_earnings_history(ticker: str, n_quarters: int = 20) -> pd.DataFrame:
    """
    Fetch historical earnings data (actual vs estimate) for a ticker.
    Uses yfinance earnings calendar.
    """
    import yfinance as yf

    try:
        stock = yf.Ticker(ticker)

        # Get earnings dates and surprise data
        earnings = stock.earnings_dates
        if earnings is None or earnings.empty:
            return pd.DataFrame()

        # Clean up the DataFrame
        df = earnings.head(n_quarters).copy()
        df = df.reset_index()

        # Standardize column names
        rename_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if "eps estimate" in col_lower or "estimated" in col_lower:
                rename_map[col] = "eps_estimate"
            elif "reported" in col_lower or "actual" in col_lower:
                rename_map[col] = "eps_actual"
            elif "surprise" in col_lower and "%" in col_lower:
                rename_map[col] = "surprise_pct"
            elif "surprise" in col_lower:
                rename_map[col] = "surprise"
            elif "date" in col_lower or "earning" in col_lower:
                rename_map[col] = "date"

        df = df.rename(columns=rename_map)

        # Compute surprise if not available
        if "surprise_pct" not in df.columns and "eps_actual" in df.columns and "eps_estimate" in df.columns:
            df["eps_actual"] = pd.to_numeric(df["eps_actual"], errors="coerce")
            df["eps_estimate"] = pd.to_numeric(df["eps_estimate"], errors="coerce")
            df = df.dropna(subset=["eps_actual", "eps_estimate"])
            df["surprise"] = df["eps_actual"] - df["eps_estimate"]
            df["surprise_pct"] = df["surprise"] / df["eps_estimate"].abs().clip(lower=0.01) * 100

        df["ticker"] = ticker
        return df

    except Exception as e:
        log.warning(f"Earnings fetch failed for {ticker}: {e}")
        return pd.DataFrame()


def compute_price_reaction(ticker: str, earnings_date: str, window: int = 5) -> dict:
    """
    Compute the stock price reaction around an earnings date.
    Measures: gap (open vs prior close), drift (close vs open), and total move.
    """
    import yfinance as yf

    try:
        # Get price data around earnings
        start = pd.Timestamp(earnings_date) - pd.Timedelta(days=10)
        end = pd.Timestamp(earnings_date) + pd.Timedelta(days=window + 5)

        stock = yf.Ticker(ticker)
        hist = stock.history(start=start, end=end)
        if hist.empty or len(hist) < 3:
            return {}

        # Find the earnings date in the price data
        earn_ts = pd.Timestamp(earnings_date)
        hist.index = hist.index.tz_localize(None)

        # Find nearest trading day on or after earnings
        post_dates = hist.index[hist.index >= earn_ts]
        if post_dates.empty:
            return {}

        earn_idx = hist.index.get_loc(post_dates[0])
        if earn_idx < 1:
            return {}

        pre_close = float(hist["Close"].iloc[earn_idx - 1])
        post_open = float(hist["Open"].iloc[earn_idx])
        post_close = float(hist["Close"].iloc[earn_idx])

        # Gap: opening gap on earnings day
        gap_pct = (post_open / pre_close - 1) * 100

        # Intraday: from open to close on earnings day
        intraday_pct = (post_close / post_open - 1) * 100

        # Total day 1 reaction
        day1_pct = (post_close / pre_close - 1) * 100

        # N-day drift after earnings
        drift_end_idx = min(earn_idx + window, len(hist) - 1)
        drift_close = float(hist["Close"].iloc[drift_end_idx])
        drift_pct = (drift_close / pre_close - 1) * 100

        return {
            "pre_close": round(pre_close, 2),
            "post_open": round(post_open, 2),
            "post_close": round(post_close, 2),
            "gap_pct": round(gap_pct, 2),
            "intraday_pct": round(intraday_pct, 2),
            "day1_pct": round(day1_pct, 2),
            "drift_pct": round(drift_pct, 2),
            "drift_days": window,
        }
    except Exception as e:
        log.warning(f"Price reaction calc failed for {ticker} on {earnings_date}: {e}")
        return {}


class EarningsSurpriseModel:
    """
    Learns stock-specific earnings reaction patterns.
    Predicts expected price move based on:
    - Historical reaction to beats/misses of similar magnitude
    - Sector comparison (how this stock reacts vs sector peers)
    - Drift patterns (does the stock continue moving or reverse?)
    """

    def __init__(self):
        self._patterns = self._load_patterns()
        self._history = self._load_history()

    def _load_patterns(self) -> dict:
        if REACTION_PATTERNS_FILE.exists():
            try:
                with open(REACTION_PATTERNS_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _load_history(self) -> pd.DataFrame:
        if EARNINGS_DB_FILE.exists():
            try:
                return pd.read_parquet(EARNINGS_DB_FILE)
            except Exception:
                pass
        return pd.DataFrame()

    def build_history(self, tickers: list[str]) -> int:
        """
        Build earnings history database for a list of tickers.
        Downloads earnings data and computes price reactions.
        """
        all_records = []

        for ticker in tickers:
            log.info(f"Fetching earnings for {ticker}")
            earnings = fetch_earnings_history(ticker)
            if earnings.empty:
                continue

            for _, row in earnings.iterrows():
                date_val = row.get("date", None)
                if date_val is None:
                    continue

                date_str = str(date_val)[:10]
                reaction = compute_price_reaction(ticker, date_str)

                record = {
                    "ticker": ticker,
                    "date": date_str,
                    "eps_estimate": float(row.get("eps_estimate", 0)) if pd.notna(row.get("eps_estimate")) else None,
                    "eps_actual": float(row.get("eps_actual", 0)) if pd.notna(row.get("eps_actual")) else None,
                    "surprise_pct": float(row.get("surprise_pct", 0)) if pd.notna(row.get("surprise_pct")) else None,
                }
                record.update(reaction)
                all_records.append(record)

        if not all_records:
            return 0

        new_df = pd.DataFrame(all_records)

        # Merge with existing
        if not self._history.empty:
            combined = pd.concat([self._history, new_df]).drop_duplicates(
                subset=["ticker", "date"], keep="last"
            )
        else:
            combined = new_df

        combined.to_parquet(EARNINGS_DB_FILE, index=False)
        self._history = combined
        log.info(f"Earnings history: {len(combined)} records for {combined['ticker'].nunique()} tickers")
        return len(all_records)

    def learn_patterns(self) -> dict:
        """
        Analyze all earnings history and learn reaction patterns.
        Groups by ticker and surprise magnitude to find predictive patterns.
        """
        if self._history.empty:
            return {}

        df = self._history.dropna(subset=["surprise_pct", "day1_pct"]).copy()
        if df.empty:
            return {}

        patterns = {}

        for ticker in df["ticker"].unique():
            t_df = df[df["ticker"] == ticker]
            if len(t_df) < 3:
                continue

            # Classify surprises
            beats = t_df[t_df["surprise_pct"] > 1]  # > 1% beat
            misses = t_df[t_df["surprise_pct"] < -1]  # > 1% miss
            inline = t_df[(t_df["surprise_pct"] >= -1) & (t_df["surprise_pct"] <= 1)]

            ticker_pattern = {
                "n_quarters": len(t_df),
                "avg_surprise_pct": round(float(t_df["surprise_pct"].mean()), 2),
            }

            if len(beats) >= 2:
                ticker_pattern["on_beat"] = {
                    "n": len(beats),
                    "avg_gap_pct": round(float(beats["gap_pct"].mean()), 2) if "gap_pct" in beats.columns else None,
                    "avg_day1_pct": round(float(beats["day1_pct"].mean()), 2),
                    "avg_drift_pct": round(float(beats["drift_pct"].mean()), 2) if "drift_pct" in beats.columns else None,
                    "pct_positive_day1": round(float((beats["day1_pct"] > 0).mean()), 2),
                    "surprise_sensitivity": self._compute_sensitivity(beats),
                }

            if len(misses) >= 2:
                ticker_pattern["on_miss"] = {
                    "n": len(misses),
                    "avg_gap_pct": round(float(misses["gap_pct"].mean()), 2) if "gap_pct" in misses.columns else None,
                    "avg_day1_pct": round(float(misses["day1_pct"].mean()), 2),
                    "avg_drift_pct": round(float(misses["drift_pct"].mean()), 2) if "drift_pct" in misses.columns else None,
                    "pct_negative_day1": round(float((misses["day1_pct"] < 0).mean()), 2),
                    "surprise_sensitivity": self._compute_sensitivity(misses),
                }

            if len(inline) >= 2:
                ticker_pattern["on_inline"] = {
                    "n": len(inline),
                    "avg_day1_pct": round(float(inline["day1_pct"].mean()), 2),
                }

            # Drift analysis: does the stock continue or reverse after earnings?
            if "drift_pct" in t_df.columns and "day1_pct" in t_df.columns:
                drift_continuation = (
                    (t_df["drift_pct"] > 0) == (t_df["day1_pct"] > 0)
                ).mean()
                ticker_pattern["drift_continuation_rate"] = round(float(drift_continuation), 2)

            patterns[ticker] = ticker_pattern

        self._patterns = patterns
        with open(REACTION_PATTERNS_FILE, "w") as f:
            json.dump(patterns, f, indent=2)

        log.info(f"Learned earnings patterns for {len(patterns)} tickers")
        return patterns

    def _compute_sensitivity(self, df: pd.DataFrame) -> float | None:
        """
        Compute surprise sensitivity: how much does stock move per 1% surprise?
        Uses linear regression of day1_pct on surprise_pct.
        """
        if len(df) < 3 or "surprise_pct" not in df.columns or "day1_pct" not in df.columns:
            return None

        try:
            x = df["surprise_pct"].values
            y = df["day1_pct"].values
            # Simple OLS slope
            x_mean = x.mean()
            y_mean = y.mean()
            num = ((x - x_mean) * (y - y_mean)).sum()
            den = ((x - x_mean) ** 2).sum()
            if abs(den) < 1e-8:
                return None
            slope = num / den
            return round(float(slope), 4)
        except Exception:
            return None

    def predict_reaction(self, ticker: str, expected_surprise_pct: float) -> dict:
        """
        Predict expected price reaction given an expected earnings surprise.
        Uses learned patterns for this specific ticker.
        """
        if ticker not in self._patterns:
            return {"error": f"No learned pattern for {ticker}"}

        pattern = self._patterns[ticker]

        if expected_surprise_pct > 1:
            # Predict based on beat pattern
            beat_data = pattern.get("on_beat", {})
            if not beat_data:
                return {"prediction": "insufficient_data"}

            sensitivity = beat_data.get("surprise_sensitivity", 0.3)
            base_move = beat_data.get("avg_day1_pct", 2.0)
            predicted_move = base_move if sensitivity is None else sensitivity * expected_surprise_pct
            drift = beat_data.get("avg_drift_pct", predicted_move * 0.5)

            return {
                "direction": "up",
                "predicted_day1_pct": round(predicted_move, 2),
                "predicted_drift_5d_pct": round(drift, 2) if drift else None,
                "confidence": round(beat_data.get("pct_positive_day1", 0.6), 2),
                "based_on_n": beat_data.get("n", 0),
                "sensitivity": sensitivity,
            }

        elif expected_surprise_pct < -1:
            miss_data = pattern.get("on_miss", {})
            if not miss_data:
                return {"prediction": "insufficient_data"}

            sensitivity = miss_data.get("surprise_sensitivity", -0.3)
            base_move = miss_data.get("avg_day1_pct", -2.0)
            predicted_move = base_move if sensitivity is None else sensitivity * expected_surprise_pct
            drift = miss_data.get("avg_drift_pct", predicted_move * 0.5)

            return {
                "direction": "down",
                "predicted_day1_pct": round(predicted_move, 2),
                "predicted_drift_5d_pct": round(drift, 2) if drift else None,
                "confidence": round(miss_data.get("pct_negative_day1", 0.6), 2),
                "based_on_n": miss_data.get("n", 0),
                "sensitivity": sensitivity,
            }

        else:
            inline_data = pattern.get("on_inline", {})
            return {
                "direction": "neutral",
                "predicted_day1_pct": round(inline_data.get("avg_day1_pct", 0), 2) if inline_data else 0,
                "confidence": 0.4,
                "based_on_n": inline_data.get("n", 0) if inline_data else 0,
            }

    def get_upcoming_earnings(self, tickers: list[str]) -> list[dict]:
        """Get upcoming earnings dates for watched tickers."""
        import yfinance as yf

        upcoming = []
        today = pd.Timestamp.now().normalize()

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                dates = stock.earnings_dates
                if dates is None or dates.empty:
                    continue

                dates.index = dates.index.tz_localize(None)
                future = dates[dates.index >= today]
                if not future.empty:
                    next_date = future.index[0]
                    days_until = (next_date - today).days

                    # Get analyst estimate if available
                    estimate = None
                    for col in future.columns:
                        if "estimate" in col.lower():
                            val = future.iloc[0][col]
                            if pd.notna(val):
                                estimate = float(val)
                                break

                    upcoming.append({
                        "ticker": ticker,
                        "date": str(next_date)[:10],
                        "days_until": int(days_until),
                        "eps_estimate": estimate,
                        "has_pattern": ticker in self._patterns,
                    })
            except Exception:
                continue

        upcoming.sort(key=lambda x: x["days_until"])
        return upcoming

    def get_stats(self) -> dict:
        """Get summary statistics about the earnings database."""
        return {
            "total_records": len(self._history),
            "tickers_tracked": self._history["ticker"].nunique() if not self._history.empty else 0,
            "patterns_learned": len(self._patterns),
            "avg_quarters_per_ticker": round(
                len(self._history) / max(self._history["ticker"].nunique(), 1), 1
            ) if not self._history.empty else 0,
        }

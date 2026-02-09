"""
SENTINEL — Cross-Asset Pattern Memory.
Permanent institutional memory of how assets interact across decades.

When gold drops 3%, what did equities do next? Which sectors led?
When the yield curve inverted, what happened to tech over the next month?
When BTC crashed 20%, did altcoins follow or diverge?

This module builds and maintains a massive pattern database from decades
of historical data. Every pattern is scored by frequency, reliability,
and recency. The forecast engine queries this memory before every prediction.

Data is stored in Parquet files — can hold 50+ years of daily data
(~12,500 rows per asset × 50 assets = 625,000 rows, trivial for Parquet).
Nothing is ever deleted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import CACHE_DIR
from utils.logger import get_logger

log = get_logger("learning.patterns")

LEARNING_DIR = Path(CACHE_DIR).parent.parent / "learning" / "data"
LEARNING_DIR.mkdir(parents=True, exist_ok=True)

PATTERN_DB_FILE = LEARNING_DIR / "pattern_database.parquet"
PATTERN_STATS_FILE = LEARNING_DIR / "pattern_stats.json"
CROSS_ASSET_FILE = LEARNING_DIR / "cross_asset_history.parquet"

# Asset groups for cross-asset analysis
EQUITY_INDICES = {"SPY": "S&P 500", "QQQ": "Nasdaq", "DIA": "Dow", "IWM": "Russell 2000"}
SECTOR_ETFS = {
    "XLK": "Technology", "XLV": "Healthcare", "XLF": "Financials",
    "XLY": "Consumer Disc", "XLC": "Communications", "XLI": "Industrials",
    "XLP": "Consumer Staples", "XLE": "Energy", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLB": "Materials",
}
COMMODITIES = {"GLD": "Gold", "SLV": "Silver", "USO": "Oil", "UNG": "Natural Gas"}
BONDS = {"TLT": "20Y+ Treasury", "IEF": "7-10Y Treasury", "HYG": "High Yield", "LQD": "Inv Grade"}
DOLLAR = {"UUP": "US Dollar"}
CRYPTO_PROXIES = {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana"}

ALL_ASSETS = {**EQUITY_INDICES, **SECTOR_ETFS, **COMMODITIES, **BONDS, **DOLLAR, **CRYPTO_PROXIES}

# Pattern trigger thresholds
MOVE_THRESHOLDS = {
    "large_drop": -0.03,      # -3% or worse
    "moderate_drop": -0.015,  # -1.5% to -3%
    "small_drop": -0.005,     # -0.5% to -1.5%
    "flat": 0.005,            # -0.5% to +0.5%
    "small_rally": 0.015,     # +0.5% to +1.5%
    "moderate_rally": 0.03,   # +1.5% to +3%
    "large_rally": 1.0,       # +3% or more
}

# Forward lookback windows (trading days)
FORWARD_WINDOWS = {"1D": 1, "1W": 5, "2W": 10, "1M": 21, "3M": 63}


class PatternMemory:
    """
    Builds and queries a massive cross-asset pattern database.
    Remembers every significant market move and what followed.
    """

    def __init__(self):
        self._history = self._load_history()
        self._patterns = self._load_patterns()

    def _load_history(self) -> pd.DataFrame:
        """Load cached cross-asset price history."""
        if CROSS_ASSET_FILE.exists():
            try:
                df = pd.read_parquet(CROSS_ASSET_FILE)
                log.info(f"Loaded cross-asset history: {len(df)} days, {len(df.columns)} assets")
                return df
            except Exception as e:
                log.warning(f"Failed to load history: {e}")
        return pd.DataFrame()

    def _save_history(self) -> None:
        self._history.to_parquet(CROSS_ASSET_FILE)

    def _load_patterns(self) -> list:
        """Load pre-computed pattern database."""
        if PATTERN_DB_FILE.exists():
            try:
                df = pd.read_parquet(PATTERN_DB_FILE)
                return df.to_dict("records")
            except Exception:
                pass
        return []

    def _save_patterns(self) -> None:
        if self._patterns:
            pd.DataFrame(self._patterns).to_parquet(PATTERN_DB_FILE, index=False)

    def update_history(self, max_years: int = 20) -> int:
        """
        Fetch/update full price history for all tracked assets.
        Downloads up to 20 years of daily data. Appends new data to existing.
        Returns number of new days added.
        """
        import yfinance as yf

        tickers = list(ALL_ASSETS.keys())
        period = f"{max_years}y"

        log.info(f"Downloading {len(tickers)} assets, {max_years} years of history...")

        try:
            raw = yf.download(tickers, period=period, group_by="ticker", progress=False)
        except Exception as e:
            log.error(f"Download failed: {e}")
            return 0

        # Extract close prices into a clean DataFrame
        closes = pd.DataFrame()
        for ticker in tickers:
            try:
                if len(tickers) > 1:
                    col = raw[ticker]["Close"].dropna()
                else:
                    col = raw["Close"].dropna()
                closes[ticker] = col
            except Exception:
                pass

        if closes.empty:
            log.warning("No price data downloaded")
            return 0

        # Merge with existing history (keep all dates, don't lose old data)
        if not self._history.empty:
            # Combine old + new, preferring new data where dates overlap
            combined = pd.concat([self._history, closes])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            new_days = len(combined) - len(self._history)
            self._history = combined
        else:
            self._history = closes
            new_days = len(closes)

        self._save_history()
        log.info(f"History updated: {len(self._history)} total days, {new_days} new, {len(closes.columns)} assets")
        return new_days

    def build_pattern_database(self) -> int:
        """
        Scan all history and build the pattern database.
        For every significant move in every asset, record what every other asset did
        over the next 1D, 1W, 2W, 1M, and 3M.

        This is the core intelligence — what has historically followed every type of move.
        """
        if self._history.empty:
            log.warning("No history data — run update_history() first")
            return 0

        returns = self._history.pct_change().dropna()
        patterns = []

        for trigger_ticker, trigger_name in ALL_ASSETS.items():
            if trigger_ticker not in returns.columns:
                continue

            trigger_returns = returns[trigger_ticker]

            for i in range(len(trigger_returns)):
                trigger_ret = trigger_returns.iloc[i]
                trigger_date = trigger_returns.index[i]

                # Classify the move
                move_type = self._classify_move(trigger_ret)
                if move_type == "flat":
                    continue  # Only track significant moves

                # Record what every other asset did after this move
                for response_ticker, response_name in ALL_ASSETS.items():
                    if response_ticker not in returns.columns:
                        continue
                    if response_ticker == trigger_ticker:
                        continue

                    for window_name, window_days in FORWARD_WINDOWS.items():
                        end_idx = i + window_days
                        if end_idx >= len(returns):
                            continue

                        # Forward return of the response asset
                        response_col = returns[response_ticker]
                        forward_return = float(
                            (1 + response_col.iloc[i + 1:end_idx + 1]).prod() - 1
                        )

                        patterns.append({
                            "trigger_ticker": trigger_ticker,
                            "trigger_name": trigger_name,
                            "trigger_date": trigger_date,
                            "trigger_return": float(trigger_ret),
                            "move_type": move_type,
                            "response_ticker": response_ticker,
                            "response_name": response_name,
                            "window": window_name,
                            "window_days": window_days,
                            "forward_return": forward_return,
                            "forward_direction": 1 if forward_return > 0.005 else (-1 if forward_return < -0.005 else 0),
                        })

        self._patterns = patterns
        self._save_patterns()

        # Compute and save aggregated stats
        self._compute_stats()

        log.info(f"Pattern database built: {len(patterns):,} total patterns across {len(ALL_ASSETS)} assets")
        return len(patterns)

    def _classify_move(self, ret: float) -> str:
        """Classify a return into a move type."""
        if ret <= MOVE_THRESHOLDS["large_drop"]:
            return "large_drop"
        elif ret <= MOVE_THRESHOLDS["moderate_drop"]:
            return "moderate_drop"
        elif ret <= MOVE_THRESHOLDS["small_drop"]:
            return "small_drop"
        elif ret <= MOVE_THRESHOLDS["flat"]:
            return "flat"
        elif ret <= MOVE_THRESHOLDS["small_rally"]:
            return "small_rally"
        elif ret <= MOVE_THRESHOLDS["moderate_rally"]:
            return "moderate_rally"
        else:
            return "large_rally"

    def _compute_stats(self) -> None:
        """Compute aggregated pattern statistics."""
        if not self._patterns:
            return

        df = pd.DataFrame(self._patterns)
        stats = {}

        # For each trigger → response pair and move type
        for (trigger, response, move, window), group in df.groupby(
            ["trigger_ticker", "response_ticker", "move_type", "window"]
        ):
            n = len(group)
            if n < 5:
                continue

            avg_return = float(group["forward_return"].mean())
            median_return = float(group["forward_return"].median())
            pct_positive = float((group["forward_return"] > 0.005).mean())
            pct_negative = float((group["forward_return"] < -0.005).mean())
            std = float(group["forward_return"].std())

            key = f"{trigger}|{response}|{move}|{window}"
            stats[key] = {
                "trigger": trigger,
                "trigger_name": ALL_ASSETS.get(trigger, trigger),
                "response": response,
                "response_name": ALL_ASSETS.get(response, response),
                "move_type": move,
                "window": window,
                "n_occurrences": n,
                "avg_forward_return": round(avg_return, 6),
                "median_forward_return": round(median_return, 6),
                "pct_positive": round(pct_positive, 4),
                "pct_negative": round(pct_negative, 4),
                "std": round(std, 6),
                "reliability": round(max(pct_positive, pct_negative), 4),
                "edge": round(abs(avg_return) / max(std, 1e-8), 4),
            }

        with open(PATTERN_STATS_FILE, "w") as f:
            json.dump(stats, f, indent=2, default=str)

        log.info(f"Computed {len(stats)} pattern statistics")

    def query(
        self,
        trigger_ticker: str,
        move_type: str | None = None,
        response_ticker: str | None = None,
        window: str = "1M",
        min_occurrences: int = 10,
        min_reliability: float = 0.55,
    ) -> list[dict]:
        """
        Query the pattern database.

        Example: "When gold (GLD) has a large_drop, what happens to each sector over 1M?"
        query("GLD", move_type="large_drop", window="1M")

        Returns list of pattern stats sorted by reliability.
        """
        if not PATTERN_STATS_FILE.exists():
            return []

        with open(PATTERN_STATS_FILE) as f:
            all_stats = json.load(f)

        results = []
        for key, stat in all_stats.items():
            if stat["trigger"] != trigger_ticker:
                continue
            if move_type and stat["move_type"] != move_type:
                continue
            if response_ticker and stat["response"] != response_ticker:
                continue
            if stat["window"] != window:
                continue
            if stat["n_occurrences"] < min_occurrences:
                continue
            if stat["reliability"] < min_reliability:
                continue

            results.append(stat)

        results.sort(key=lambda x: x["reliability"], reverse=True)
        return results

    def query_current_conditions(self, window: str = "1M", top_n: int = 10) -> list[dict]:
        """
        Look at TODAY's market moves and query what historically followed.
        Returns the most reliable patterns triggered by today's conditions.
        """
        if self._history.empty:
            return []

        returns = self._history.pct_change()
        if returns.empty:
            return []

        latest = returns.iloc[-1]
        triggered = []

        for ticker in latest.index:
            ret = latest[ticker]
            if pd.isna(ret):
                continue

            move = self._classify_move(ret)
            if move == "flat":
                continue

            patterns = self.query(ticker, move_type=move, window=window)
            for p in patterns:
                p["triggered_by_return"] = round(float(ret), 4)
                triggered.append(p)

        # Sort by reliability × edge, return top N
        triggered.sort(key=lambda x: x["reliability"] * x["edge"], reverse=True)
        return triggered[:top_n]

    def get_relationship(
        self,
        asset_a: str,
        asset_b: str,
        window: str = "1M",
    ) -> dict:
        """
        Get the full historical relationship between two assets.
        "When A drops big, what does B do? When A rallies, what about B?"
        """
        move_types = ["large_drop", "moderate_drop", "small_drop",
                      "small_rally", "moderate_rally", "large_rally"]

        relationship = {
            "asset_a": asset_a,
            "asset_a_name": ALL_ASSETS.get(asset_a, asset_a),
            "asset_b": asset_b,
            "asset_b_name": ALL_ASSETS.get(asset_b, asset_b),
            "window": window,
            "patterns": {},
        }

        for move in move_types:
            results = self.query(asset_a, move_type=move, response_ticker=asset_b,
                                 window=window, min_occurrences=3, min_reliability=0.0)
            if results:
                r = results[0]
                relationship["patterns"][move] = {
                    "n": r["n_occurrences"],
                    "avg_response": r["avg_forward_return"],
                    "pct_positive": r["pct_positive"],
                    "pct_negative": r["pct_negative"],
                    "reliability": r["reliability"],
                }

        return relationship

    def get_strongest_signals_for(
        self,
        ticker: str,
        window: str = "1M",
        min_occurrences: int = 20,
    ) -> list[dict]:
        """
        What OTHER assets' moves are the best predictors of THIS ticker?
        "What signals best predict where gold is headed next month?"
        """
        if not PATTERN_STATS_FILE.exists():
            return []

        with open(PATTERN_STATS_FILE) as f:
            all_stats = json.load(f)

        signals = []
        for key, stat in all_stats.items():
            if stat["response"] != ticker:
                continue
            if stat["window"] != window:
                continue
            if stat["n_occurrences"] < min_occurrences:
                continue
            if stat["reliability"] < 0.6:
                continue

            signals.append(stat)

        signals.sort(key=lambda x: x["reliability"] * abs(x["avg_forward_return"]), reverse=True)
        return signals[:20]

    def get_memory_stats(self) -> dict:
        """Get statistics about the pattern memory."""
        stats = {
            "history_days": len(self._history),
            "history_assets": len(self._history.columns) if not self._history.empty else 0,
            "total_patterns": len(self._patterns),
            "history_start": str(self._history.index[0])[:10] if not self._history.empty else "N/A",
            "history_end": str(self._history.index[-1])[:10] if not self._history.empty else "N/A",
        }

        if self._history.empty:
            stats["years_of_data"] = 0
        else:
            days = (self._history.index[-1] - self._history.index[0]).days
            stats["years_of_data"] = round(days / 365.25, 1)

        if PATTERN_STATS_FILE.exists():
            with open(PATTERN_STATS_FILE) as f:
                pattern_stats = json.load(f)
            stats["unique_patterns"] = len(pattern_stats)

        return stats

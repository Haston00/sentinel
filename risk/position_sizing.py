"""
SENTINEL — Position Sizing Engine.
Translates forecast conviction into optimal position sizes.
Uses Kelly criterion, volatility scaling, and risk limits.

Hedge fund standard: Never risk more than you can afford to lose.
Position size = f(conviction, volatility, correlation, drawdown state).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import CACHE_DIR
from utils.logger import get_logger

log = get_logger("risk.position_sizing")

RISK_DIR = Path(CACHE_DIR).parent.parent / "risk" / "data"
RISK_DIR.mkdir(parents=True, exist_ok=True)
POSITION_LOG_FILE = RISK_DIR / "position_log.parquet"
RISK_CONFIG_FILE = RISK_DIR / "risk_config.json"

# Default risk parameters (conservative institutional defaults)
DEFAULT_RISK_CONFIG = {
    "total_portfolio_value": 100_000,      # Default $100K portfolio
    "max_single_position_pct": 0.10,       # Max 10% in any one position
    "max_sector_exposure_pct": 0.30,       # Max 30% in any sector
    "max_asset_class_pct": 0.50,           # Max 50% in any asset class
    "max_total_exposure_pct": 1.00,        # Max 100% invested (no leverage)
    "max_daily_loss_pct": 0.02,            # Max 2% daily loss trigger
    "kelly_fraction": 0.5,                 # Half-Kelly (institutional standard)
    "vol_target_annual": 0.15,             # 15% annualized vol target
    "min_position_pct": 0.01,              # Min 1% position (below this = skip)
    "drawdown_scale_threshold": 0.05,      # Start reducing at 5% drawdown
    "drawdown_max_reduction": 0.5,         # Reduce to 50% of normal at max drawdown
}


class PositionSizer:
    """
    Institutional-grade position sizing engine.

    Three independent sizing methods, take the minimum:
    1. Kelly-based (conviction-driven)
    2. Volatility-targeted (risk-driven)
    3. Risk-limit constrained (compliance-driven)
    """

    def __init__(self, config: dict | None = None):
        self.config = config or self._load_config()
        self._position_log = self._load_log()

    def _load_config(self) -> dict:
        if RISK_CONFIG_FILE.exists():
            try:
                with open(RISK_CONFIG_FILE) as f:
                    saved = json.load(f)
                    return {**DEFAULT_RISK_CONFIG, **saved}
            except Exception:
                pass
        return DEFAULT_RISK_CONFIG.copy()

    def save_config(self, config: dict) -> None:
        self.config = {**DEFAULT_RISK_CONFIG, **config}
        with open(RISK_CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=2)

    def _load_log(self) -> pd.DataFrame:
        if POSITION_LOG_FILE.exists():
            try:
                return pd.read_parquet(POSITION_LOG_FILE)
            except Exception:
                pass
        return pd.DataFrame()

    def size_position(
        self,
        ticker: str,
        forecast: dict,
        asset_type: str = "equity",
        sector: str | None = None,
        current_positions: dict | None = None,
    ) -> dict:
        """
        Compute optimal position size for a trade.

        Args:
            ticker: Asset ticker
            forecast: Forecast dict with 'confidence', 'point' (expected return),
                     'lower', 'upper', 'direction'
            asset_type: 'equity' or 'crypto'
            sector: GICS sector if equity
            current_positions: Dict of {ticker: position_value} for existing portfolio

        Returns:
            Dict with recommended position size and reasoning.
        """
        portfolio_value = self.config["total_portfolio_value"]
        current_positions = current_positions or {}

        confidence = forecast.get("confidence", 0.5)
        expected_return = forecast.get("point", 0)
        direction = forecast.get("direction", 0)
        lower = forecast.get("lower", -0.05)
        upper = forecast.get("upper", 0.05)

        if direction == 0 or abs(expected_return) < 0.001:
            return {
                "ticker": ticker,
                "action": "NO_TRADE",
                "reason": "No directional signal",
                "position_pct": 0,
                "position_value": 0,
            }

        # ── Method 1: Kelly-Based Sizing ──────────────────────────
        kelly_pct = self._kelly_size(confidence, expected_return, lower, upper)

        # ── Method 2: Volatility-Targeted Sizing ──────────────────
        vol_pct = self._vol_target_size(ticker, asset_type)

        # ── Method 3: Risk-Limit Sizing ───────────────────────────
        limit_pct = self._risk_limit_size(
            ticker, asset_type, sector, current_positions
        )

        # ── Take the minimum of all three methods ─────────────────
        raw_pct = min(kelly_pct, vol_pct, limit_pct)

        # ── Apply drawdown scaling ────────────────────────────────
        drawdown_scale = self._drawdown_adjustment()
        final_pct = raw_pct * drawdown_scale

        # ── Apply minimum position filter ─────────────────────────
        if final_pct < self.config["min_position_pct"]:
            return {
                "ticker": ticker,
                "action": "SKIP",
                "reason": f"Position too small ({final_pct:.1%} < {self.config['min_position_pct']:.1%} minimum)",
                "position_pct": 0,
                "position_value": 0,
                "kelly_pct": round(kelly_pct, 4),
                "vol_pct": round(vol_pct, 4),
                "limit_pct": round(limit_pct, 4),
            }

        position_value = round(portfolio_value * final_pct, 2)
        action = "BUY" if direction > 0 else "SHORT"

        result = {
            "ticker": ticker,
            "action": action,
            "direction": direction,
            "position_pct": round(final_pct, 4),
            "position_value": round(position_value, 2),
            "shares_approx": None,  # Filled if we have price
            "kelly_pct": round(kelly_pct, 4),
            "vol_pct": round(vol_pct, 4),
            "limit_pct": round(limit_pct, 4),
            "drawdown_scale": round(drawdown_scale, 4),
            "confidence": round(confidence, 4),
            "expected_return": round(expected_return, 4),
            "risk_reward_ratio": round(abs(upper) / max(abs(lower), 0.001), 2),
            "sizing_method": (
                "kelly" if kelly_pct == raw_pct
                else "volatility" if vol_pct == raw_pct
                else "risk_limit"
            ),
        }

        # Estimate share count
        try:
            import yfinance as yf
            hist = yf.Ticker(ticker).history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
                result["shares_approx"] = int(position_value / price)
                result["current_price"] = round(price, 2)
        except Exception:
            pass

        return result

    def _kelly_size(
        self,
        confidence: float,
        expected_return: float,
        lower: float,
        upper: float,
    ) -> float:
        """
        Kelly criterion position sizing.
        f* = (p × b - q) / b
        where p = win probability, q = loss probability, b = win/loss ratio.
        We use half-Kelly for institutional risk management.
        """
        p = confidence
        q = 1 - p

        # Estimate win/loss ratio from forecast bounds
        expected_win = abs(expected_return)
        expected_loss = abs(lower) if expected_return > 0 else abs(upper)
        b = expected_win / max(expected_loss, 0.001)

        kelly_full = (p * b - q) / max(b, 0.001)
        kelly_half = kelly_full * self.config["kelly_fraction"]

        # Clamp to reasonable bounds
        return float(np.clip(kelly_half, 0, self.config["max_single_position_pct"]))

    def _vol_target_size(self, ticker: str, asset_type: str) -> float:
        """
        Volatility-targeted sizing.
        Position size = vol_target / asset_vol
        If an asset has 30% annual vol and we target 15%, position = 50%.
        """
        # Estimate asset volatility
        try:
            import yfinance as yf
            hist = yf.Ticker(ticker).history(period="120d")
            if hist.empty or len(hist) < 20:
                # Conservative default vol estimates
                default_vol = 0.30 if asset_type == "crypto" else 0.20
                return self.config["vol_target_annual"] / default_vol

            returns = hist["Close"].pct_change().dropna()
            asset_vol = float(returns.std() * np.sqrt(252))
        except Exception:
            asset_vol = 0.80 if asset_type == "crypto" else 0.25

        if asset_vol < 0.01:
            asset_vol = 0.25  # Floor

        vol_target = self.config["vol_target_annual"]
        vol_sized = vol_target / asset_vol

        return float(np.clip(vol_sized, 0, self.config["max_single_position_pct"]))

    def _risk_limit_size(
        self,
        ticker: str,
        asset_type: str,
        sector: str | None,
        current_positions: dict,
    ) -> float:
        """
        Risk-limit constrained sizing.
        Enforces: single position max, sector max, asset class max, total max.
        """
        portfolio = self.config["total_portfolio_value"]
        max_single = self.config["max_single_position_pct"]
        max_sector = self.config["max_sector_exposure_pct"]
        max_asset_class = self.config["max_asset_class_pct"]
        max_total = self.config["max_total_exposure_pct"]

        # Check existing exposure
        total_exposure = sum(abs(v) for v in current_positions.values()) / max(portfolio, 1)
        remaining_total = max(max_total - total_exposure, 0)

        # Sector exposure check
        if sector and current_positions:
            sector_exposure = sum(
                abs(v) for t, v in current_positions.items()
                if self._get_sector(t) == sector
            ) / max(portfolio, 1)
            remaining_sector = max(max_sector - sector_exposure, 0)
        else:
            remaining_sector = max_sector

        # Asset class check
        if current_positions:
            class_exposure = sum(
                abs(v) for t, v in current_positions.items()
                if self._get_asset_class(t) == asset_type
            ) / max(portfolio, 1)
            remaining_class = max(max_asset_class - class_exposure, 0)
        else:
            remaining_class = max_asset_class

        # Take the most restrictive limit
        return min(max_single, remaining_total, remaining_sector, remaining_class)

    def _drawdown_adjustment(self) -> float:
        """
        Scale down position sizes during drawdowns.
        Start reducing at threshold, linearly decrease to max_reduction.
        """
        # Use learning optimizer's drawdown data if available
        try:
            from learning.optimizer import LearningOptimizer
            optimizer = LearningOptimizer()
            reliability = optimizer.model_reliability.get("system", {})
            recent_accuracy = reliability.get("reliability_mean", 0.5)

            if recent_accuracy < 0.4:
                return self.config["drawdown_max_reduction"]
            elif recent_accuracy < 0.5:
                # Linear interpolation between full and max reduction
                scale = (recent_accuracy - 0.4) / 0.1
                return self.config["drawdown_max_reduction"] + scale * (1.0 - self.config["drawdown_max_reduction"])
        except Exception:
            pass

        return 1.0  # No adjustment

    @staticmethod
    def _get_sector(ticker: str) -> str | None:
        try:
            from config.assets import get_sector_for_ticker
            return get_sector_for_ticker(ticker)
        except Exception:
            return None

    @staticmethod
    def _get_asset_class(ticker: str) -> str:
        if ticker.endswith("-USD") or ticker in ("BTC", "ETH", "SOL"):
            return "crypto"
        return "equity"

    def size_portfolio(
        self,
        forecasts: dict[str, dict],
        current_positions: dict | None = None,
    ) -> dict:
        """
        Size an entire portfolio of trades simultaneously.
        Accounts for correlations and total exposure limits.

        Args:
            forecasts: Dict of {ticker: forecast_dict}
            current_positions: Dict of {ticker: position_value}

        Returns:
            Dict with all position recommendations and portfolio summary.
        """
        current_positions = current_positions or {}
        positions = []
        total_allocated = 0

        # Sort by conviction (highest first)
        sorted_tickers = sorted(
            forecasts.keys(),
            key=lambda t: abs(forecasts[t].get("confidence", 0) * forecasts[t].get("point", 0)),
            reverse=True,
        )

        for ticker in sorted_tickers:
            remaining_pct = self.config["max_total_exposure_pct"] - total_allocated
            if remaining_pct <= self.config["min_position_pct"]:
                break

            result = self.size_position(
                ticker,
                forecasts[ticker],
                current_positions=current_positions,
            )

            if result["action"] not in ("NO_TRADE", "SKIP"):
                # Don't exceed remaining allocation
                actual_pct = min(result["position_pct"], remaining_pct)
                result["position_pct"] = round(actual_pct, 4)
                result["position_value"] = round(
                    self.config["total_portfolio_value"] * actual_pct, 2
                )
                total_allocated += actual_pct

            positions.append(result)

        # Portfolio summary
        active = [p for p in positions if p["action"] not in ("NO_TRADE", "SKIP")]
        long_exposure = sum(p["position_pct"] for p in active if p.get("direction", 0) > 0)
        short_exposure = sum(p["position_pct"] for p in active if p.get("direction", 0) < 0)

        return {
            "positions": positions,
            "n_active": len(active),
            "n_skipped": len(positions) - len(active),
            "total_exposure_pct": round(total_allocated, 4),
            "long_exposure_pct": round(long_exposure, 4),
            "short_exposure_pct": round(short_exposure, 4),
            "net_exposure_pct": round(long_exposure - short_exposure, 4),
            "cash_pct": round(1.0 - total_allocated, 4),
            "portfolio_value": self.config["total_portfolio_value"],
        }

    def get_config(self) -> dict:
        return self.config.copy()

"""
SENTINEL — Risk Management Overlay.
Portfolio-level risk monitoring: VaR, stress testing, correlation analysis,
concentration limits, and drawdown protection.

Hedge fund standard: Know your risk before you know your P&L.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import CACHE_DIR
from utils.logger import get_logger

log = get_logger("risk.risk_manager")

RISK_DIR = Path(CACHE_DIR).parent.parent / "risk" / "data"
RISK_DIR.mkdir(parents=True, exist_ok=True)
RISK_REPORT_FILE = RISK_DIR / "latest_risk_report.json"


class RiskManager:
    """
    Portfolio-level risk management engine.
    Monitors all risk dimensions and generates alerts.
    """

    def __init__(self, portfolio_value: float = 100_000):
        self.portfolio_value = portfolio_value

    def compute_var(
        self,
        tickers: list[str],
        weights: list[float],
        confidence: float = 0.95,
        horizon_days: int = 1,
        lookback_days: int = 252,
    ) -> dict:
        """
        Compute Value at Risk using three methods:
        1. Historical VaR — uses actual return distribution
        2. Parametric VaR — assumes normal distribution
        3. Monte Carlo VaR — simulates 10,000 scenarios

        Returns all three for robustness. Use the most conservative.
        """
        import yfinance as yf

        tickers_clean = [t for t in tickers]
        if not tickers_clean or not weights:
            return {"error": "No positions provided"}

        # Fetch returns
        try:
            data = yf.download(
                tickers_clean, period=f"{lookback_days + 30}d",
                group_by="ticker", progress=False,
            )
        except Exception as e:
            return {"error": f"Data fetch failed: {e}"}

        returns_dict = {}
        for t in tickers_clean:
            try:
                if len(tickers_clean) > 1:
                    close = data[t]["Close"].dropna()
                else:
                    close = data["Close"].dropna()
                returns_dict[t] = close.pct_change().dropna()
            except Exception:
                continue

        if not returns_dict:
            return {"error": "No return data available"}

        returns_df = pd.DataFrame(returns_dict).dropna()
        if returns_df.empty or len(returns_df) < 20:
            return {"error": "Insufficient historical data"}

        # Normalize weights
        w = np.array(weights[:len(returns_df.columns)])
        w = w / w.sum()

        # Portfolio returns
        port_returns = returns_df.values @ w

        # ── Method 1: Historical VaR ─────────────────────────────
        alpha = 1 - confidence
        hist_var_1d = float(-np.percentile(port_returns, alpha * 100))
        hist_var = hist_var_1d * np.sqrt(horizon_days)

        # ── Method 2: Parametric (Normal) VaR ─────────────────────
        from scipy.stats import norm
        mu = float(port_returns.mean())
        sigma = float(port_returns.std())
        z = norm.ppf(confidence)
        param_var_1d = float(-(mu - z * sigma))
        param_var = param_var_1d * np.sqrt(horizon_days)

        # ── Method 3: Monte Carlo VaR ────────────────────────────
        n_sims = 10_000
        cov_matrix = returns_df.cov().values
        mean_returns = returns_df.mean().values

        simulated = np.random.multivariate_normal(
            mean_returns * horizon_days,
            cov_matrix * horizon_days,
            size=n_sims,
        )
        sim_port_returns = simulated @ w
        mc_var = float(-np.percentile(sim_port_returns, alpha * 100))

        # Use most conservative (highest VaR)
        max_var = max(hist_var, param_var, mc_var)

        # Conditional VaR (Expected Shortfall) — average loss beyond VaR
        tail = port_returns[port_returns <= -hist_var_1d]
        cvar = float(-tail.mean() * np.sqrt(horizon_days)) if len(tail) > 0 else hist_var * 1.4

        return {
            "confidence": confidence,
            "horizon_days": horizon_days,
            "historical_var": round(hist_var, 6),
            "parametric_var": round(param_var, 6),
            "monte_carlo_var": round(mc_var, 6),
            "conservative_var": round(max_var, 6),
            "var_dollars": round(max_var * self.portfolio_value, 2),
            "cvar": round(cvar, 6),
            "cvar_dollars": round(cvar * self.portfolio_value, 2),
            "interpretation": (
                f"At {confidence:.0%} confidence, max {horizon_days}-day loss: "
                f"${max_var * self.portfolio_value:,.0f} "
                f"({max_var:.2%} of portfolio)"
            ),
            "n_observations": len(returns_df),
        }

    def correlation_matrix(
        self,
        tickers: list[str],
        lookback_days: int = 120,
    ) -> dict:
        """
        Compute correlation matrix for portfolio holdings.
        High correlation = concentrated risk.
        """
        import yfinance as yf

        try:
            data = yf.download(
                tickers, period=f"{lookback_days + 10}d",
                group_by="ticker", progress=False,
            )
        except Exception as e:
            return {"error": str(e)}

        returns_dict = {}
        for t in tickers:
            try:
                if len(tickers) > 1:
                    close = data[t]["Close"].dropna()
                else:
                    close = data["Close"].dropna()
                returns_dict[t] = close.pct_change().dropna()
            except Exception:
                continue

        if len(returns_dict) < 2:
            return {"error": "Need at least 2 assets for correlation"}

        returns_df = pd.DataFrame(returns_dict).dropna()
        corr = returns_df.corr()

        # Find highly correlated pairs
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                c = float(corr.iloc[i, j])
                if abs(c) > 0.7:
                    high_corr.append({
                        "pair": f"{corr.columns[i]}/{corr.columns[j]}",
                        "correlation": round(c, 3),
                        "risk": "high" if abs(c) > 0.85 else "moderate",
                    })

        # Average pairwise correlation (diversification measure)
        upper_triangle = corr.values[np.triu_indices_from(corr.values, k=1)]
        avg_corr = float(upper_triangle.mean()) if len(upper_triangle) > 0 else 0

        return {
            "matrix": {
                col: {row: round(float(corr.loc[row, col]), 3) for row in corr.index}
                for col in corr.columns
            },
            "high_correlation_pairs": sorted(high_corr, key=lambda x: abs(x["correlation"]), reverse=True),
            "avg_correlation": round(avg_corr, 3),
            "diversification_score": round(1 - abs(avg_corr), 3),
            "n_high_corr_pairs": len(high_corr),
            "risk_assessment": (
                "POOR diversification" if avg_corr > 0.5
                else "MODERATE diversification" if avg_corr > 0.3
                else "GOOD diversification"
            ),
        }

    def stress_test(
        self,
        tickers: list[str],
        weights: list[float],
    ) -> dict:
        """
        Run stress tests against historical crisis scenarios.
        Shows portfolio performance during known market shocks.
        """
        import yfinance as yf

        # Known crisis periods
        scenarios = {
            "COVID Crash (Feb-Mar 2020)": ("2020-02-19", "2020-03-23"),
            "2022 Bear Market (Jan-Oct)": ("2022-01-03", "2022-10-12"),
            "Aug 2024 Carry Unwind": ("2024-07-31", "2024-08-05"),
            "SVB Crisis (Mar 2023)": ("2023-03-08", "2023-03-13"),
            "Dec 2018 Sell-off": ("2018-12-03", "2018-12-24"),
            "Fed Taper Tantrum (2013)": ("2013-05-22", "2013-06-24"),
        }

        results = {}

        try:
            data = yf.download(
                tickers, start="2013-01-01", group_by="ticker", progress=False,
            )
        except Exception as e:
            return {"error": str(e)}

        w = np.array(weights[:len(tickers)])
        w = w / w.sum()

        for scenario_name, (start, end) in scenarios.items():
            try:
                scenario_returns = {}
                for i, t in enumerate(tickers):
                    try:
                        if len(tickers) > 1:
                            close = data[t]["Close"]
                        else:
                            close = data["Close"]

                        close = close.dropna()
                        mask = (close.index >= start) & (close.index <= end)
                        period = close[mask]

                        if len(period) >= 2:
                            ret = float(period.iloc[-1] / period.iloc[0] - 1)
                            scenario_returns[t] = ret
                    except Exception:
                        scenario_returns[t] = 0

                if scenario_returns:
                    port_return = sum(
                        scenario_returns.get(t, 0) * w[i]
                        for i, t in enumerate(tickers)
                    )

                    worst_asset = min(scenario_returns, key=scenario_returns.get)
                    best_asset = max(scenario_returns, key=scenario_returns.get)

                    results[scenario_name] = {
                        "portfolio_return": round(float(port_return), 4),
                        "portfolio_loss_dollars": round(float(port_return) * self.portfolio_value, 2),
                        "worst_asset": worst_asset,
                        "worst_return": round(scenario_returns[worst_asset], 4),
                        "best_asset": best_asset,
                        "best_return": round(scenario_returns[best_asset], 4),
                        "asset_returns": {
                            t: round(r, 4) for t, r in scenario_returns.items()
                        },
                    }
            except Exception:
                continue

        return {
            "scenarios": results,
            "worst_case": min(
                results.values(), key=lambda x: x["portfolio_return"]
            ) if results else None,
            "avg_crisis_loss": round(
                float(np.mean([r["portfolio_return"] for r in results.values()])), 4
            ) if results else None,
        }

    def concentration_check(
        self,
        positions: dict[str, float],
    ) -> dict:
        """
        Check portfolio concentration across sectors, asset classes, and single names.
        Returns alerts for any limit breaches.
        """
        total = sum(abs(v) for v in positions.values())
        if total == 0:
            return {"alerts": [], "status": "empty_portfolio"}

        alerts = []

        # Single position concentration
        for ticker, value in positions.items():
            pct = abs(value) / total
            if pct > 0.15:
                alerts.append({
                    "type": "SINGLE_NAME",
                    "severity": "HIGH" if pct > 0.20 else "MEDIUM",
                    "ticker": ticker,
                    "pct": round(pct, 3),
                    "message": f"{ticker} is {pct:.1%} of portfolio (max recommended: 10-15%)",
                })

        # Sector concentration
        from config.assets import get_sector_for_ticker
        sector_exposure = {}
        for ticker, value in positions.items():
            sector = get_sector_for_ticker(ticker) or "Other"
            sector_exposure[sector] = sector_exposure.get(sector, 0) + abs(value)

        for sector, value in sector_exposure.items():
            pct = value / total
            if pct > 0.35:
                alerts.append({
                    "type": "SECTOR",
                    "severity": "HIGH" if pct > 0.45 else "MEDIUM",
                    "sector": sector,
                    "pct": round(pct, 3),
                    "message": f"{sector} sector is {pct:.1%} of portfolio (max recommended: 30%)",
                })

        # Crypto concentration
        crypto_value = sum(
            abs(v) for t, v in positions.items()
            if t.endswith("-USD") or t in ("BTC", "ETH", "SOL")
        )
        crypto_pct = crypto_value / total
        if crypto_pct > 0.25:
            alerts.append({
                "type": "ASSET_CLASS",
                "severity": "HIGH" if crypto_pct > 0.40 else "MEDIUM",
                "asset_class": "crypto",
                "pct": round(crypto_pct, 3),
                "message": f"Crypto is {crypto_pct:.1%} of portfolio (high volatility asset class)",
            })

        return {
            "n_positions": len(positions),
            "total_value": round(total, 2),
            "sector_breakdown": {
                s: round(v / total, 3) for s, v in sector_exposure.items()
            },
            "crypto_pct": round(crypto_pct, 3),
            "alerts": alerts,
            "n_alerts": len(alerts),
            "status": "BREACH" if any(a["severity"] == "HIGH" for a in alerts) else (
                "WARNING" if alerts else "CLEAR"
            ),
        }

    def max_drawdown(
        self,
        tickers: list[str],
        weights: list[float],
        lookback_days: int = 252,
    ) -> dict:
        """
        Compute maximum drawdown of portfolio over lookback period.
        """
        import yfinance as yf

        try:
            data = yf.download(
                tickers, period=f"{lookback_days + 10}d",
                group_by="ticker", progress=False,
            )
        except Exception as e:
            return {"error": str(e)}

        returns_dict = {}
        for t in tickers:
            try:
                if len(tickers) > 1:
                    close = data[t]["Close"].dropna()
                else:
                    close = data["Close"].dropna()
                returns_dict[t] = close.pct_change().dropna()
            except Exception:
                continue

        if not returns_dict:
            return {"error": "No data"}

        returns_df = pd.DataFrame(returns_dict).dropna()
        w = np.array(weights[:len(returns_df.columns)])
        w = w / w.sum()

        port_returns = returns_df.values @ w
        cumulative = np.cumprod(1 + port_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative / running_max - 1

        max_dd = float(drawdowns.min())
        max_dd_idx = np.argmin(drawdowns)

        # Find peak before max drawdown
        peak_idx = np.argmax(cumulative[:max_dd_idx + 1]) if max_dd_idx > 0 else 0

        # Current drawdown
        current_dd = float(drawdowns[-1])

        return {
            "max_drawdown": round(max_dd, 4),
            "max_drawdown_dollars": round(max_dd * self.portfolio_value, 2),
            "max_drawdown_duration_days": int(max_dd_idx - peak_idx),
            "current_drawdown": round(current_dd, 4),
            "current_drawdown_dollars": round(current_dd * self.portfolio_value, 2),
            "in_drawdown": current_dd < -0.02,
            "total_return": round(float(cumulative[-1] - 1), 4),
            "lookback_days": lookback_days,
        }

    def full_risk_report(
        self,
        tickers: list[str],
        weights: list[float],
        positions: dict[str, float] | None = None,
    ) -> dict:
        """
        Generate comprehensive risk report.
        This is the daily risk report a hedge fund PM would review.
        """
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "portfolio_value": self.portfolio_value,
            "n_positions": len(tickers),
        }

        # VaR (1-day and 10-day at 95% and 99%)
        report["var_95_1d"] = self.compute_var(tickers, weights, 0.95, 1)
        report["var_99_1d"] = self.compute_var(tickers, weights, 0.99, 1)
        report["var_95_10d"] = self.compute_var(tickers, weights, 0.95, 10)

        # Correlation
        report["correlation"] = self.correlation_matrix(tickers)

        # Stress test
        report["stress_test"] = self.stress_test(tickers, weights)

        # Max drawdown
        report["drawdown"] = self.max_drawdown(tickers, weights)

        # Concentration
        if positions:
            report["concentration"] = self.concentration_check(positions)

        # Overall risk assessment
        alerts = []
        var_95 = report["var_95_1d"].get("conservative_var", 0)
        if var_95 > 0.03:
            alerts.append("HIGH: 1-day VaR exceeds 3%")
        if report.get("correlation", {}).get("avg_correlation", 0) > 0.5:
            alerts.append("HIGH: Portfolio poorly diversified (avg correlation > 50%)")
        dd = report.get("drawdown", {}).get("current_drawdown", 0)
        if dd < -0.10:
            alerts.append(f"HIGH: Currently in {abs(dd):.1%} drawdown")

        report["risk_alerts"] = alerts
        report["overall_risk_level"] = (
            "CRITICAL" if len(alerts) >= 2
            else "ELEVATED" if alerts
            else "NORMAL"
        )

        # Save report
        try:
            with open(RISK_REPORT_FILE, "w") as f:
                json.dump(report, f, indent=2, default=str)
        except Exception:
            pass

        return report

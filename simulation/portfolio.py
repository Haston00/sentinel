"""
SENTINEL — Paper Trading Portfolio Engine.
Manages a virtual $100K portfolio: positions, cash, P&L, trade history.
Persists state to JSON so it survives restarts.
"""

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from data.crypto import get_current_prices as get_crypto_prices
from config.assets import CRYPTO_ALL
from utils.logger import get_logger

logger = get_logger(__name__)

PORTFOLIO_DIR = Path("simulation/data")
PORTFOLIO_FILE = PORTFOLIO_DIR / "portfolio.json"
STARTING_CASH = 100_000.00

ET = ZoneInfo("America/New_York")


def _now():
    return datetime.now(ET).isoformat()


def _load_state() -> dict:
    """Load portfolio state from disk."""
    if PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    return _new_state()


def _save_state(state: dict):
    """Persist portfolio state to disk."""
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _new_state() -> dict:
    return {
        "cash": STARTING_CASH,
        "starting_cash": STARTING_CASH,
        "positions": {},
        "trade_history": [],
        "daily_snapshots": [],
        "created": _now(),
        "last_updated": _now(),
    }


def reset_portfolio() -> dict:
    """Reset portfolio to $100K starting state."""
    state = _new_state()
    _save_state(state)
    return state


def get_portfolio() -> dict:
    """Return current portfolio state with live prices."""
    state = _load_state()
    positions = state.get("positions", {})

    if not positions:
        state["total_value"] = state["cash"]
        state["total_pnl"] = 0.0
        state["total_pnl_pct"] = 0.0
        state["positions_detail"] = []
        return state

    # Fetch live prices for all held tickers
    stock_tickers = [t for t in positions if not positions[t].get("is_crypto")]
    crypto_tickers = [t for t in positions if positions[t].get("is_crypto")]

    live_prices = {}

    # Stocks
    if stock_tickers:
        try:
            data = yf.download(stock_tickers, period="2d", progress=False, threads=True)
            for ticker in stock_tickers:
                try:
                    if len(stock_tickers) == 1:
                        closes = data["Close"].dropna()
                    else:
                        closes = data[ticker]["Close"].dropna() if ticker in data.columns.get_level_values(0) else pd.Series()
                    if not closes.empty:
                        live_prices[ticker] = float(closes.iloc[-1])
                except Exception:
                    pass
        except Exception:
            pass

    # Crypto
    if crypto_tickers:
        try:
            cprices = get_crypto_prices()
            if cprices:
                for symbol in crypto_tickers:
                    for coin_id, info in CRYPTO_ALL.items():
                        if info["symbol"] == symbol:
                            cd = cprices.get(coin_id, {})
                            if cd:
                                live_prices[symbol] = cd.get("usd", 0)
                            break
        except Exception:
            pass

    # Build positions detail
    details = []
    total_invested = 0
    total_market = 0

    for ticker, pos in positions.items():
        shares = pos["shares"]
        avg_cost = pos["avg_cost"]
        invested = shares * avg_cost
        current_price = live_prices.get(ticker, avg_cost)
        market_value = shares * current_price
        pnl = market_value - invested
        pnl_pct = (pnl / invested * 100) if invested > 0 else 0

        total_invested += invested
        total_market += market_value

        details.append({
            "ticker": ticker,
            "shares": shares,
            "avg_cost": round(avg_cost, 4),
            "current_price": round(current_price, 4),
            "market_value": round(market_value, 2),
            "cost_basis": round(invested, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "weight": 0,  # filled below
            "is_crypto": pos.get("is_crypto", False),
            "sector": pos.get("sector", ""),
            "date_opened": pos.get("date_opened", ""),
            "thesis": pos.get("thesis", ""),
        })

    total_value = state["cash"] + total_market

    # Calculate weights
    for d in details:
        d["weight"] = round(d["market_value"] / total_value * 100, 1) if total_value > 0 else 0

    # Sort by market value
    details.sort(key=lambda x: x["market_value"], reverse=True)

    state["positions_detail"] = details
    state["total_value"] = round(total_value, 2)
    state["total_pnl"] = round(total_value - state["starting_cash"], 2)
    state["total_pnl_pct"] = round((total_value / state["starting_cash"] - 1) * 100, 2)
    state["total_invested"] = round(total_invested, 2)
    state["total_market"] = round(total_market, 2)
    state["last_updated"] = _now()

    return state


def execute_trade(ticker: str, action: str, shares: float, thesis: str = "",
                  is_crypto: bool = False, sector: str = "") -> dict:
    """
    Execute a buy or sell trade.
    Returns dict with success/failure and details.
    """
    state = _load_state()

    # Get current price
    current_price = _get_live_price(ticker, is_crypto)
    if current_price is None or current_price <= 0:
        return {"success": False, "error": f"Could not get price for {ticker}"}

    if action == "BUY":
        cost = shares * current_price
        if cost > state["cash"]:
            max_shares = int(state["cash"] / current_price)
            return {
                "success": False,
                "error": f"Not enough cash. Need ${cost:,.2f} but only have ${state['cash']:,.2f}. Max shares: {max_shares}",
            }

        # Update or create position
        if ticker in state["positions"]:
            pos = state["positions"][ticker]
            total_cost = pos["shares"] * pos["avg_cost"] + cost
            total_shares = pos["shares"] + shares
            pos["avg_cost"] = total_cost / total_shares
            pos["shares"] = total_shares
        else:
            state["positions"][ticker] = {
                "shares": shares,
                "avg_cost": current_price,
                "date_opened": _now(),
                "is_crypto": is_crypto,
                "sector": sector,
                "thesis": thesis,
            }

        state["cash"] -= cost
        state["cash"] = round(state["cash"], 2)

        trade = {
            "ticker": ticker,
            "action": "BUY",
            "shares": shares,
            "price": round(current_price, 4),
            "total": round(cost, 2),
            "thesis": thesis,
            "timestamp": _now(),
            "cash_after": state["cash"],
        }

    elif action == "SELL":
        if ticker not in state["positions"]:
            return {"success": False, "error": f"You don't own any {ticker}"}

        pos = state["positions"][ticker]
        if shares > pos["shares"]:
            return {
                "success": False,
                "error": f"You only own {pos['shares']} shares of {ticker}",
            }

        proceeds = shares * current_price
        cost_basis = shares * pos["avg_cost"]
        realized_pnl = proceeds - cost_basis
        realized_pnl_pct = (realized_pnl / cost_basis * 100) if cost_basis > 0 else 0

        pos["shares"] -= shares
        if pos["shares"] <= 0.0001:  # Float cleanup
            del state["positions"][ticker]
        else:
            state["positions"][ticker] = pos

        state["cash"] += proceeds
        state["cash"] = round(state["cash"], 2)

        trade = {
            "ticker": ticker,
            "action": "SELL",
            "shares": shares,
            "price": round(current_price, 4),
            "total": round(proceeds, 2),
            "cost_basis_per_share": round(cost_basis / shares, 4),
            "realized_pnl": round(realized_pnl, 2),
            "realized_pnl_pct": round(realized_pnl_pct, 2),
            "thesis": thesis,
            "timestamp": _now(),
            "cash_after": state["cash"],
        }
    else:
        return {"success": False, "error": f"Unknown action: {action}"}

    state["trade_history"].append(trade)
    state["last_updated"] = _now()
    _save_state(state)

    return {"success": True, "trade": trade, "price": current_price}


def take_daily_snapshot():
    """Save a daily portfolio value snapshot for performance tracking."""
    state = _load_state()
    port = get_portfolio()

    snapshot = {
        "date": datetime.now(ET).strftime("%Y-%m-%d"),
        "total_value": port["total_value"],
        "cash": state["cash"],
        "invested": port.get("total_market", 0),
        "num_positions": len(state["positions"]),
    }

    # Avoid duplicate for same day
    snaps = state.get("daily_snapshots", [])
    if snaps and snaps[-1]["date"] == snapshot["date"]:
        snaps[-1] = snapshot
    else:
        snaps.append(snapshot)

    state["daily_snapshots"] = snaps
    _save_state(state)
    return snapshot


def get_trade_history() -> list:
    """Return full trade history."""
    state = _load_state()
    return list(reversed(state.get("trade_history", [])))


def get_performance_stats() -> dict:
    """Calculate performance metrics like a real fund."""
    state = _load_state()
    port = get_portfolio()
    history = state.get("trade_history", [])
    snapshots = state.get("daily_snapshots", [])

    total_value = port["total_value"]
    starting = state["starting_cash"]
    total_return = (total_value / starting - 1) * 100

    # Win/loss on closed trades
    sells = [t for t in history if t["action"] == "SELL"]
    wins = [t for t in sells if t.get("realized_pnl", 0) > 0]
    losses = [t for t in sells if t.get("realized_pnl", 0) < 0]
    win_rate = len(wins) / len(sells) * 100 if sells else 0
    avg_win = np.mean([t["realized_pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t["realized_pnl"]) for t in losses]) if losses else 0
    profit_factor = (avg_win * len(wins)) / (avg_loss * len(losses)) if losses and avg_loss > 0 else float("inf") if wins else 0
    total_realized = sum(t.get("realized_pnl", 0) for t in sells)

    # Unrealized P&L
    total_unrealized = sum(p["pnl"] for p in port.get("positions_detail", []))

    # Daily returns from snapshots
    daily_returns = []
    if len(snapshots) >= 2:
        for i in range(1, len(snapshots)):
            prev_val = snapshots[i - 1]["total_value"]
            curr_val = snapshots[i]["total_value"]
            if prev_val > 0:
                daily_returns.append(curr_val / prev_val - 1)

    # Risk metrics
    sharpe = 0
    sortino = 0
    max_dd = 0
    volatility = 0

    if daily_returns:
        avg_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns)
        volatility = std_ret * np.sqrt(252) * 100  # Annualized
        sharpe = (avg_ret * 252) / (std_ret * np.sqrt(252)) if std_ret > 0 else 0

        downside = [r for r in daily_returns if r < 0]
        downside_std = np.std(downside) if downside else 0.001
        sortino = (avg_ret * 252) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0

        # Max drawdown
        values = [s["total_value"] for s in snapshots]
        peak = values[0]
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100
            if dd > max_dd:
                max_dd = dd

    # Sector allocation
    sector_alloc = {}
    for p in port.get("positions_detail", []):
        sec = p.get("sector", "Other") or "Other"
        sector_alloc[sec] = sector_alloc.get(sec, 0) + p["weight"]

    # Diversification score (0-100 based on HHI)
    weights = [p["weight"] / 100 for p in port.get("positions_detail", []) if p["weight"] > 0]
    if weights:
        hhi = sum(w ** 2 for w in weights)
        # HHI of 1 = fully concentrated, 1/n = perfectly diversified
        n = len(weights)
        min_hhi = 1 / n if n > 0 else 1
        # Scale: 0 = fully concentrated, 100 = perfectly diversified
        diversification = max(0, min(100, (1 - hhi) / (1 - min_hhi) * 100)) if min_hhi < 1 else 0
    else:
        diversification = 0

    return {
        "total_value": round(total_value, 2),
        "cash": round(state["cash"], 2),
        "total_return_pct": round(total_return, 2),
        "total_realized_pnl": round(total_realized, 2),
        "total_unrealized_pnl": round(total_unrealized, 2),
        "num_trades": len(history),
        "num_positions": len(state["positions"]),
        "win_rate": round(win_rate, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "∞",
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "volatility_annual_pct": round(volatility, 2),
        "diversification_score": round(diversification, 0),
        "sector_allocation": sector_alloc,
        "days_active": len(snapshots),
    }


def _get_live_price(ticker: str, is_crypto: bool = False) -> float:
    """Get current price for a single ticker."""
    if is_crypto:
        try:
            cprices = get_crypto_prices()
            if cprices:
                for coin_id, info in CRYPTO_ALL.items():
                    if info["symbol"] == ticker:
                        cd = cprices.get(coin_id, {})
                        return cd.get("usd", None)
        except Exception:
            pass
        return None

    try:
        data = yf.download(ticker, period="2d", progress=False)
        if not data.empty:
            return float(data["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return None

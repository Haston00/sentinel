"""
SENTINEL — API Blueprint.
All JSON endpoints for the Flask dashboard.
"""

import json
import sys
import time
import traceback
from pathlib import Path
from flask import Blueprint, jsonify, request, session

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

api_bp = Blueprint("api", __name__)

# Briefing cache — avoids re-generating on every page load
_briefing_cache = {"data": None, "timestamp": 0}
BRIEFING_TTL = 600  # 10 minutes


def _error(msg, status=500):
    return jsonify({"error": str(msg)}), status


# ═══════════════════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════════════════

@api_bp.route("/auth/check")
def auth_check():
    return jsonify({"authenticated": bool(session.get("authenticated"))})


# ═══════════════════════════════════════════════════════════════
# MARKET OVERVIEW
# ═══════════════════════════════════════════════════════════════

@api_bp.route("/market/overview")
def market_overview():
    """Composite score, key benchmark metrics, sector snapshot."""
    try:
        from data.stocks import fetch_ohlcv
        from config.assets import BENCHMARKS, SECTORS, VIX_TICKER
        from features.signals import compute_composite_score

        benchmarks = {}
        for name, ticker in BENCHMARKS.items():
            try:
                df = fetch_ohlcv(ticker)
                if not df.empty:
                    close = df["Close"]
                    price = float(close.iloc[-1])
                    day_chg = float(close.pct_change().iloc[-1]) * 100
                    wk_chg = float(close.pct_change(5).iloc[-1]) * 100 if len(close) > 5 else 0
                    mo_chg = float(close.pct_change(21).iloc[-1]) * 100 if len(close) > 21 else 0
                    benchmarks[name] = {
                        "ticker": ticker, "price": round(price, 2),
                        "day_change": round(day_chg, 2),
                        "week_change": round(wk_chg, 2),
                        "month_change": round(mo_chg, 2),
                    }
            except Exception:
                pass

        # Composite score on SPY
        composite = {"score": 50, "label": "N/A"}
        try:
            spy = fetch_ohlcv("SPY")
            if not spy.empty:
                result = compute_composite_score(spy)
                composite = {"score": round(result["score"], 1), "label": result["label"]}
        except Exception:
            pass

        # VIX
        vix = None
        try:
            vdf = fetch_ohlcv(VIX_TICKER)
            if not vdf.empty:
                vix = round(float(vdf["Close"].iloc[-1]), 2)
        except Exception:
            pass

        # Sector snapshot (ETF prices)
        sectors = {}
        for sname, info in SECTORS.items():
            try:
                df = fetch_ohlcv(info["etf"])
                if not df.empty:
                    c = df["Close"]
                    sectors[sname] = {
                        "etf": info["etf"],
                        "price": round(float(c.iloc[-1]), 2),
                        "day_change": round(float(c.pct_change().iloc[-1]) * 100, 2),
                        "month_change": round(float(c.pct_change(21).iloc[-1]) * 100, 2) if len(c) > 21 else 0,
                    }
            except Exception:
                pass

        return jsonify({
            "benchmarks": benchmarks,
            "composite": composite,
            "vix": vix,
            "sectors": sectors,
        })
    except Exception as e:
        return _error(e)


@api_bp.route("/market/chart/<ticker>")
def market_chart(ticker):
    """Return Plotly figure JSON for a ticker chart.
    Short timeframes (<=90 days) get candlestick; longer gets area chart.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from data.stocks import fetch_ohlcv
        from config.settings import COLORS

        days = int(request.args.get("days", 252))
        df = fetch_ohlcv(ticker)
        if df.empty:
            return _error("No data for ticker", 404)
        trimmed = df.tail(days)
        close = trimmed["Close"]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03, row_heights=[0.75, 0.25])

        if days <= 90:
            # Short timeframe — candlestick shows intraday action
            fig.add_trace(go.Candlestick(
                x=trimmed.index, open=trimmed["Open"], high=trimmed["High"],
                low=trimmed["Low"], close=trimmed["Close"], name="Price",
                increasing_line_color=COLORS["bull"],
                decreasing_line_color=COLORS["bear"],
            ), row=1, col=1)
        else:
            # Longer timeframe — area chart like Bloomberg/CNBC
            # Color the line green or red based on overall trend
            first_close = float(close.iloc[0])
            last_close = float(close.iloc[-1])
            trend_color = COLORS["bull"] if last_close >= first_close else COLORS["bear"]
            fill_color = "rgba(0,200,83,0.12)" if last_close >= first_close else "rgba(255,23,68,0.12)"

            fig.add_trace(go.Scatter(
                x=close.index.tolist(), y=close.values.tolist(),
                name="Price", mode="lines",
                line=dict(color=trend_color, width=2),
                fill="tozeroy", fillcolor=fill_color,
            ), row=1, col=1)

        # SMAs
        for period, color in [(20, "#FFD600"), (50, "#FF9100"), (200, "#2962FF")]:
            if len(trimmed) >= period:
                sma = close.rolling(period).mean()
                fig.add_trace(go.Scatter(
                    x=trimmed.index, y=sma, name=f"SMA {period}",
                    line=dict(width=1, color=color),
                ), row=1, col=1)

        # Volume
        if "Volume" in trimmed.columns:
            colors = [COLORS["bull"] if c >= o else COLORS["bear"]
                      for c, o in zip(trimmed["Close"], trimmed["Open"])]
            fig.add_trace(go.Bar(
                x=trimmed.index, y=trimmed["Volume"], name="Volume",
                marker_color=colors, opacity=0.5,
            ), row=2, col=1)

        # Calculate YTD / period return for subtitle
        period_return = (float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100
        sign = "+" if period_return >= 0 else ""
        period_label = f"{days}D" if days <= 60 else "1Y" if days >= 200 else f"{days}D"

        fig.update_layout(
            title=f"{ticker}  {sign}{period_return:.1f}% ({period_label})",
            template="plotly_dark",
            paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"]), height=500,
            margin=dict(l=60, r=30, t=60, b=40),
            xaxis_rangeslider_visible=False, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(size=10, color="#94a3b8")),
        )
        return jsonify(json.loads(fig.to_json()))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════
# GENIUS BRIEFING
# ═══════════════════════════════════════════════════════════════

def _generate_briefing_with_news():
    """Generate briefing + attach news. Result is cached."""
    from features.analyst import generate_market_briefing
    briefing = generate_market_briefing()

    # Add news headlines
    try:
        import pandas as pd
        from data.news import fetch_rss_feeds
        articles = []
        rss = fetch_rss_feeds()
        if isinstance(rss, pd.DataFrame) and not rss.empty:
            for _, row in rss.head(15).iterrows():
                articles.append({
                    "title": str(row.get("Title", "")),
                    "source": str(row.get("Source", row.get("domain", ""))),
                    "url": str(row.get("URL", row.get("link", "#"))),
                })
        seen = set()
        unique = []
        for a in articles:
            if a["title"] and a["title"] not in seen:
                seen.add(a["title"])
                unique.append(a)
        briefing["news"] = unique[:15]
    except Exception:
        briefing["news"] = []

    return briefing


@api_bp.route("/briefing/generate", methods=["POST"])
def briefing_generate():
    """Generate full market briefing. Returns cached version if fresh."""
    global _briefing_cache
    try:
        force = (request.get_json() or {}).get("force", False)
        age = time.time() - _briefing_cache["timestamp"]

        # Return cache if fresh (< 10 min) and not forced
        if _briefing_cache["data"] and age < BRIEFING_TTL and not force:
            _briefing_cache["data"]["_cached"] = True
            _briefing_cache["data"]["_age_seconds"] = int(age)
            return jsonify(_briefing_cache["data"])

        briefing = _generate_briefing_with_news()
        _briefing_cache = {"data": briefing, "timestamp": time.time()}
        briefing["_cached"] = False
        return jsonify(briefing)
    except Exception as e:
        return _error(e)


@api_bp.route("/briefing/latest")
def briefing_latest():
    """Return cached briefing if available, otherwise generate."""
    global _briefing_cache
    try:
        if _briefing_cache["data"]:
            return jsonify(_briefing_cache["data"])
        briefing = _generate_briefing_with_news()
        _briefing_cache = {"data": briefing, "timestamp": time.time()}
        return jsonify(briefing)
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════
# MACRO
# ═══════════════════════════════════════════════════════════════

@api_bp.route("/macro/indicators")
def macro_indicators():
    """Return macro indicator data from FRED."""
    try:
        from config.assets import MACRO_INDICATORS
        from data.macro import fetch_series

        results = {}
        for key, info in MACRO_INDICATORS.items():
            try:
                series = fetch_series(info["series"])
                if series is not None and not series.empty:
                    latest = float(series.iloc[-1])
                    prev = float(series.iloc[-2]) if len(series) > 1 else latest
                    change = latest - prev
                    results[key] = {
                        "name": info["name"],
                        "value": round(latest, 4),
                        "change": round(change, 4),
                        "frequency": info["frequency"],
                        "date": str(series.index[-1].date()) if hasattr(series.index[-1], 'date') else str(series.index[-1]),
                    }
            except Exception:
                pass
        return jsonify(results)
    except Exception as e:
        return _error(e)


@api_bp.route("/macro/chart/<indicator>")
def macro_chart(indicator):
    """Return time series chart for a FRED indicator."""
    try:
        import plotly.graph_objects as go
        from config.assets import MACRO_INDICATORS
        from config.settings import COLORS
        from data.macro import fetch_series

        info = MACRO_INDICATORS.get(indicator)
        if not info:
            return _error("Unknown indicator", 404)

        series = fetch_series(info["series"])
        if series is None or series.empty:
            return _error("No data", 404)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series.index.tolist(), y=series.values.tolist(),
            name=info["name"],
            line=dict(color=COLORS["primary"], width=2),
        ))
        fig.update_layout(
            title=info["name"], template="plotly_dark",
            paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"]), height=400,
            margin=dict(l=50, r=20, t=40, b=40),
        )
        return jsonify(json.loads(fig.to_json()))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════
# REGIME
# ═══════════════════════════════════════════════════════════════

@api_bp.route("/regime/current")
def regime_current():
    """Current HMM regime + probabilities."""
    try:
        from data.stocks import fetch_ohlcv
        from models.regime import RegimeDetector

        spy = fetch_ohlcv("SPY")
        if spy.empty:
            return _error("Cannot load SPY data")

        vix = fetch_ohlcv("^VIX")
        vix_df = vix if not vix.empty else None

        detector = RegimeDetector()
        try:
            detector.load("equity_regime")
        except Exception:
            detector.fit(spy, vix=vix_df)
            detector.save("equity_regime")

        regime_info = detector.current_regime(spy, vix=vix_df)
        # Convert numpy types to Python types
        result = {}
        for k, v in regime_info.items():
            if hasattr(v, 'item'):
                result[k] = v.item()
            elif isinstance(v, dict):
                result[k] = {kk: vv.item() if hasattr(vv, 'item') else vv for kk, vv in v.items()}
            else:
                result[k] = v
        return jsonify(result)
    except Exception as e:
        return _error(e)


@api_bp.route("/regime/history")
def regime_history():
    """Regime history timeline as Plotly chart JSON."""
    try:
        import plotly.graph_objects as go
        from data.stocks import fetch_ohlcv
        from models.regime import RegimeDetector
        from config.settings import COLORS

        spy = fetch_ohlcv("SPY")
        if spy.empty:
            return _error("Cannot load SPY data")

        vix = fetch_ohlcv("^VIX")
        vix_df = vix if not vix.empty else None

        detector = RegimeDetector()
        try:
            detector.load("equity_regime")
        except Exception:
            detector.fit(spy, vix=vix_df)
            detector.save("equity_regime")

        # Get full state sequence
        states, probs, idx = detector.predict()
        label_map = detector._labels

        fig = go.Figure()
        regime_colors = {"BULL": COLORS["bull"], "BEAR": COLORS["bear"], "TRANSITION": COLORS["neutral"]}

        for state_num, info in label_map.items():
            mask = states == state_num
            dates = [str(idx[i].date()) if hasattr(idx[i], 'date') else str(idx[i]) for i in range(len(idx)) if mask[i]]
            if dates:
                fig.add_trace(go.Scatter(
                    x=dates, y=[info["name"]] * len(dates),
                    mode="markers", marker=dict(color=regime_colors.get(info["name"], "#888"), size=3),
                    name=info["name"],
                ))

        fig.update_layout(
            title="Regime History", template="plotly_dark",
            paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"]), height=300,
            margin=dict(l=80, r=20, t=40, b=40),
        )
        return jsonify(json.loads(fig.to_json()))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════
# ALPHA SCREENER
# ═══════════════════════════════════════════════════════════════

@api_bp.route("/screener/run", methods=["POST"])
def screener_run():
    """Run alpha screener on specified tickers."""
    try:
        from data.stocks import fetch_ohlcv
        from features.signals import compute_composite_score, score_multiple

        data = request.get_json() or {}
        mode = data.get("mode", "etf")
        custom = data.get("tickers", "")

        from config.assets import SECTORS
        if mode == "all":
            tickers = set()
            for s in SECTORS.values():
                tickers.add(s["etf"])
                tickers.update(s["holdings"])
            tickers = sorted(tickers)
        elif mode == "custom" and custom:
            tickers = [t.strip().upper() for t in custom.split(",") if t.strip()]
        else:
            tickers = [s["etf"] for s in SECTORS.values()]

        ohlcv_data = {}
        for t in tickers:
            df = fetch_ohlcv(t)
            if not df.empty:
                ohlcv_data[t] = df

        results = score_multiple(ohlcv_data)
        if results.empty:
            return jsonify({"results": []})

        rows = []
        for _, row in results.iterrows():
            rows.append({
                "ticker": row.name if hasattr(row, 'name') else row.get("Ticker", ""),
                "score": round(float(row["Score"]), 1),
                "label": row.get("Label", ""),
                "trend": round(float(row.get("Trend", 50)), 1),
                "momentum": round(float(row.get("Momentum", 50)), 1),
                "rsi": round(float(row.get("RSI", 50)), 1),
                "volume": round(float(row.get("Volume", 50)), 1),
            })
        return jsonify({"results": rows})
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════
# FORECASTER
# ═══════════════════════════════════════════════════════════════

@api_bp.route("/forecast/run", methods=["POST"])
def forecast_run():
    """Run XGBoost forecast for a ticker."""
    try:
        import plotly.graph_objects as go
        from data.stocks import fetch_ohlcv
        from config.settings import COLORS

        data = request.get_json() or {}
        ticker = data.get("ticker", "SPY").upper()
        horizon = data.get("horizon", "1M")

        df = fetch_ohlcv(ticker)
        if df.empty:
            return _error("No data for ticker", 404)

        # Try to run the forecast engine
        try:
            from forecasting.engine import ForecastEngine
            engine = ForecastEngine()
            result = engine.forecast(ticker)

            close = df["Close"].tail(252)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=close.index.tolist(), y=close.values.tolist(),
                name="Price", line=dict(color=COLORS["primary"], width=2),
            ))
            fig.update_layout(
                title=f"{ticker} Forecast", template="plotly_dark",
                paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["surface"],
                font=dict(color=COLORS["text"]), height=450,
                margin=dict(l=60, r=30, t=60, b=40),
            )
            return jsonify({
                "chart": json.loads(fig.to_json()),
                "forecast": {
                    h: {k: round(float(v), 4) if isinstance(v, (int, float)) else v
                        for k, v in fc.items()}
                    for h, fc in result.items() if isinstance(fc, dict)
                }
            })
        except Exception:
            # Fallback: price chart with SMAs
            close = df["Close"].tail(252)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=close.index.tolist(), y=close.values.tolist(),
                name="Price", line=dict(color=COLORS["primary"], width=2),
            ))
            for period, color in [(20, "#FFD600"), (50, "#FF9100")]:
                if len(close) >= period:
                    sma = close.rolling(period).mean()
                    fig.add_trace(go.Scatter(
                        x=close.index.tolist(), y=sma.values.tolist(),
                        name=f"SMA {period}", line=dict(width=1, color=color),
                    ))
            fig.update_layout(
                title=f"{ticker} — Price History", template="plotly_dark",
                paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["surface"],
                font=dict(color=COLORS["text"]), height=450,
                margin=dict(l=60, r=30, t=60, b=40),
            )
            return jsonify({"chart": json.loads(fig.to_json()), "forecast": {}})
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════
# SCENARIO LAB
# ═══════════════════════════════════════════════════════════════

@api_bp.route("/scenario/run", methods=["POST"])
def scenario_run():
    """Run a what-if scenario."""
    try:
        data = request.get_json() or {}
        scenario_type = data.get("type", "rate_hike")
        magnitude = float(data.get("magnitude", 0.25))

        # Simple scenario analysis
        impacts = {
            "rate_hike": {
                "equities": -magnitude * 3,
                "bonds": -magnitude * 5,
                "gold": -magnitude * 2,
                "dollar": magnitude * 4,
                "description": f"Fed raises rates by {magnitude*100:.0f}bps",
            },
            "rate_cut": {
                "equities": magnitude * 4,
                "bonds": magnitude * 6,
                "gold": magnitude * 3,
                "dollar": -magnitude * 4,
                "description": f"Fed cuts rates by {magnitude*100:.0f}bps",
            },
            "recession": {
                "equities": -20 * magnitude,
                "bonds": 8 * magnitude,
                "gold": 10 * magnitude,
                "dollar": -5 * magnitude,
                "description": f"Recession scenario (severity: {magnitude:.1f}x)",
            },
            "inflation_shock": {
                "equities": -10 * magnitude,
                "bonds": -12 * magnitude,
                "gold": 15 * magnitude,
                "dollar": -3 * magnitude,
                "description": f"Inflation shock (severity: {magnitude:.1f}x)",
            },
        }
        result = impacts.get(scenario_type, impacts["rate_hike"])
        return jsonify(result)
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════
# CORRELATION & SECTORS
# ═══════════════════════════════════════════════════════════════

@api_bp.route("/correlation/matrix")
def correlation_matrix():
    """Return correlation matrix heatmap as Plotly JSON."""
    try:
        import plotly.graph_objects as go
        from data.stocks import fetch_ohlcv
        from config.assets import SECTORS, BENCHMARKS
        from config.settings import COLORS
        import pandas as pd

        tickers = list(BENCHMARKS.values()) + [s["etf"] for s in SECTORS.values()]
        closes = {}
        for t in tickers:
            df = fetch_ohlcv(t)
            if not df.empty:
                closes[t] = df["Close"]
        if not closes:
            return _error("No data")

        close_df = pd.DataFrame(closes).dropna()
        corr = close_df.pct_change().dropna().corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}", textfont=dict(size=9),
        ))
        fig.update_layout(
            title="Asset Correlation Matrix", template="plotly_dark",
            paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"]), height=600,
            margin=dict(l=80, r=30, t=60, b=80),
        )
        return jsonify(json.loads(fig.to_json()))
    except Exception as e:
        return _error(e)


@api_bp.route("/sectors/heatmap")
def sectors_heatmap():
    """Return sector performance data for heatmap."""
    try:
        from data.stocks import fetch_ohlcv
        from config.assets import SECTORS

        results = []
        for name, info in SECTORS.items():
            try:
                df = fetch_ohlcv(info["etf"])
                if not df.empty:
                    c = df["Close"]
                    results.append({
                        "sector": name,
                        "etf": info["etf"],
                        "price": round(float(c.iloc[-1]), 2),
                        "day": round(float(c.pct_change().iloc[-1]) * 100, 2),
                        "week": round(float(c.pct_change(5).iloc[-1]) * 100, 2) if len(c) > 5 else 0,
                        "month": round(float(c.pct_change(21).iloc[-1]) * 100, 2) if len(c) > 21 else 0,
                        "quarter": round(float(c.pct_change(63).iloc[-1]) * 100, 2) if len(c) > 63 else 0,
                        "ytd": round(float((c.iloc[-1] / c.iloc[0] - 1) * 100), 2),
                    })
            except Exception:
                pass
        return jsonify({"sectors": results})
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════
# FACTOR EXPLORER
# ═══════════════════════════════════════════════════════════════

@api_bp.route("/factors/<factor>")
def factor_data(factor):
    """Return factor analysis data."""
    try:
        factor_etfs = {
            "value": {"ticker": "VLUE", "name": "Value (VLUE)"},
            "momentum": {"ticker": "MTUM", "name": "Momentum (MTUM)"},
            "quality": {"ticker": "QUAL", "name": "Quality (QUAL)"},
            "size": {"ticker": "SIZE", "name": "Size (SIZE)"},
            "min_vol": {"ticker": "USMV", "name": "Min Volatility (USMV)"},
            "growth": {"ticker": "IWF", "name": "Growth (IWF)"},
        }

        info = factor_etfs.get(factor)
        if not info:
            return _error("Unknown factor", 404)

        import plotly.graph_objects as go
        from data.stocks import fetch_ohlcv
        from config.settings import COLORS

        df = fetch_ohlcv(info["ticker"])
        if df.empty:
            return _error("No data")

        trimmed = df.tail(252)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=trimmed.index, open=trimmed["Open"], high=trimmed["High"],
            low=trimmed["Low"], close=trimmed["Close"], name="Price",
            increasing_line_color=COLORS["bull"],
            decreasing_line_color=COLORS["bear"],
        ))
        for period, color in [(20, "#FFD600"), (50, "#FF9100")]:
            if len(trimmed) >= period:
                sma = trimmed["Close"].rolling(period).mean()
                fig.add_trace(go.Scatter(
                    x=trimmed.index, y=sma, name=f"SMA {period}",
                    line=dict(width=1, color=color),
                ))
        fig.update_layout(
            title=info["name"], template="plotly_dark",
            paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"]), height=450,
            margin=dict(l=60, r=30, t=60, b=40),
            xaxis_rangeslider_visible=False,
        )
        c = df["Close"]
        return jsonify({
            "chart": json.loads(fig.to_json()),
            "stats": {
                "price": round(float(c.iloc[-1]), 2),
                "day": round(float(c.pct_change().iloc[-1]) * 100, 2),
                "month": round(float(c.pct_change(21).iloc[-1]) * 100, 2) if len(c) > 21 else 0,
                "quarter": round(float(c.pct_change(63).iloc[-1]) * 100, 2) if len(c) > 63 else 0,
                "year": round(float(c.pct_change(252).iloc[-1]) * 100, 2) if len(c) > 252 else 0,
            }
        })
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════
# CROSS-ASSET
# ═══════════════════════════════════════════════════════════════

@api_bp.route("/commodities/data")
def commodities_data():
    """Return commodity prices and charts."""
    try:
        from data.stocks import fetch_ohlcv
        from config.assets import COMMODITY_TICKERS

        results = {}
        for name, ticker in COMMODITY_TICKERS.items():
            try:
                df = fetch_ohlcv(ticker)
                if not df.empty:
                    c = df["Close"]
                    results[name] = {
                        "ticker": ticker,
                        "price": round(float(c.iloc[-1]), 2),
                        "day": round(float(c.pct_change().iloc[-1]) * 100, 2),
                        "week": round(float(c.pct_change(5).iloc[-1]) * 100, 2) if len(c) > 5 else 0,
                        "month": round(float(c.pct_change(21).iloc[-1]) * 100, 2) if len(c) > 21 else 0,
                    }
            except Exception:
                pass
        return jsonify(results)
    except Exception as e:
        return _error(e)


@api_bp.route("/commodities/chart/<ticker>")
def commodities_chart(ticker):
    """Return commodity chart."""
    try:
        import plotly.graph_objects as go
        from data.stocks import fetch_ohlcv
        from config.settings import COLORS

        df = fetch_ohlcv(ticker)
        if df.empty:
            return _error("No data", 404)
        trimmed = df.tail(252)
        close = trimmed["Close"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=close.index.tolist(), y=close.values.tolist(),
            name=ticker, line=dict(color=COLORS["primary"], width=2),
            fill="tozeroy", fillcolor="rgba(41,98,255,0.08)",
        ))
        fig.update_layout(
            title=ticker, template="plotly_dark",
            paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"]), height=400,
            margin=dict(l=60, r=30, t=60, b=40),
        )
        return jsonify(json.loads(fig.to_json()))
    except Exception as e:
        return _error(e)


@api_bp.route("/crypto/data")
def crypto_data():
    """Return crypto prices from CoinGecko."""
    try:
        from data.crypto import get_current_prices
        from config.assets import CRYPTO_ALL

        prices = get_current_prices()
        results = {}
        if prices:
            for coin_id, info in CRYPTO_ALL.items():
                cd = prices.get(coin_id, {})
                if cd:
                    results[info["symbol"]] = {
                        "name": info["name"],
                        "price": cd.get("usd", 0),
                        "change_24h": round(cd.get("usd_24h_change", 0) or 0, 2),
                        "market_cap": cd.get("usd_market_cap", 0),
                        "volume_24h": cd.get("usd_24h_vol", 0),
                    }
        return jsonify(results)
    except Exception as e:
        return _error(e)


@api_bp.route("/fixed-income/curves")
def fixed_income():
    """Return yield curve data."""
    try:
        from data.stocks import fetch_ohlcv
        from config.assets import BOND_TICKERS
        import plotly.graph_objects as go
        from config.settings import COLORS

        results = {}
        for name, ticker in BOND_TICKERS.items():
            try:
                df = fetch_ohlcv(ticker)
                if not df.empty:
                    c = df["Close"]
                    results[name] = {
                        "ticker": ticker,
                        "price": round(float(c.iloc[-1]), 2),
                        "day": round(float(c.pct_change().iloc[-1]) * 100, 2),
                        "month": round(float(c.pct_change(21).iloc[-1]) * 100, 2) if len(c) > 21 else 0,
                    }
            except Exception:
                pass

        # Yield curve from FRED
        try:
            from data.macro import fetch_series
            maturities = ["DGS2", "DGS10", "DGS30"]
            mat_labels = ["2Y", "10Y", "30Y"]
            yields = []
            for s in maturities:
                series = fetch_series(s)
                if series is not None and not series.empty:
                    yields.append(float(series.iloc[-1]))
                else:
                    yields.append(None)

            fig = go.Figure()
            valid_x = [l for l, y in zip(mat_labels, yields) if y is not None]
            valid_y = [y for y in yields if y is not None]
            if valid_y:
                fig.add_trace(go.Scatter(
                    x=valid_x, y=valid_y, mode="lines+markers",
                    line=dict(color=COLORS["primary"], width=2),
                    marker=dict(size=8),
                    name="Yield Curve"
                ))
                fig.update_layout(
                    title="Treasury Yield Curve", template="plotly_dark",
                    paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["surface"],
                    font=dict(color=COLORS["text"]), height=350,
                    yaxis_title="Yield (%)",
                )
                results["yield_curve_chart"] = json.loads(fig.to_json())
        except Exception:
            pass

        return jsonify(results)
    except Exception as e:
        return _error(e)


@api_bp.route("/forex/pairs")
def forex_pairs():
    """Return major FX pair data."""
    try:
        from data.stocks import fetch_ohlcv
        import plotly.graph_objects as go
        from config.settings import COLORS

        pairs = {
            "EUR/USD": "EURUSD=X",
            "GBP/USD": "GBPUSD=X",
            "USD/JPY": "USDJPY=X",
            "USD/CHF": "USDCHF=X",
            "AUD/USD": "AUDUSD=X",
            "USD/CAD": "USDCAD=X",
            "DXY (Dollar)": "UUP",
        }

        results = {}
        for name, ticker in pairs.items():
            try:
                df = fetch_ohlcv(ticker)
                if not df.empty:
                    c = df["Close"]
                    results[name] = {
                        "ticker": ticker,
                        "price": round(float(c.iloc[-1]), 4),
                        "day": round(float(c.pct_change().iloc[-1]) * 100, 2),
                        "week": round(float(c.pct_change(5).iloc[-1]) * 100, 2) if len(c) > 5 else 0,
                    }
            except Exception:
                pass

        return jsonify(results)
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════
# TRADING / PORTFOLIO
# ═══════════════════════════════════════════════════════════════

@api_bp.route("/portfolio/holdings")
def portfolio_holdings():
    """Return current portfolio with live prices."""
    try:
        from simulation.portfolio import get_portfolio, get_performance_stats
        port = get_portfolio()
        stats = get_performance_stats()
        return jsonify({"portfolio": port, "stats": stats})
    except Exception as e:
        return _error(e)


@api_bp.route("/portfolio/trade", methods=["POST"])
def portfolio_trade():
    """Execute a paper trade."""
    try:
        from simulation.portfolio import execute_trade
        from config.assets import get_sector_for_ticker, CRYPTO_ALL

        data = request.get_json() or {}
        ticker = data.get("ticker", "").upper()
        action = data.get("action", "buy").lower()
        shares = float(data.get("shares", 0))
        thesis = data.get("thesis", "")

        if not ticker or shares <= 0:
            return _error("Invalid ticker or shares", 400)

        is_crypto = ticker in [c["symbol"] for c in CRYPTO_ALL.values()]
        sector = get_sector_for_ticker(ticker) or ""

        result = execute_trade(ticker, action, shares, thesis=thesis,
                             is_crypto=is_crypto, sector=sector)
        return jsonify(result)
    except Exception as e:
        return _error(e)


@api_bp.route("/portfolio/history")
def portfolio_history():
    """Return trade history."""
    try:
        from simulation.portfolio import get_trade_history
        history = get_trade_history()
        return jsonify({"trades": history})
    except Exception as e:
        return _error(e)


@api_bp.route("/portfolio/reset", methods=["POST"])
def portfolio_reset():
    """Reset portfolio to starting state."""
    try:
        from simulation.portfolio import reset_portfolio
        reset_portfolio()
        return jsonify({"success": True, "message": "Portfolio reset to $100K"})
    except Exception as e:
        return _error(e)


@api_bp.route("/risk/analytics")
def risk_analytics():
    """Return risk analytics: VaR, drawdown, Sharpe."""
    try:
        from simulation.portfolio import get_performance_stats
        stats = get_performance_stats()
        return jsonify(stats)
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════
# ADVANCED
# ═══════════════════════════════════════════════════════════════

@api_bp.route("/options/flow")
def options_flow():
    """Return unusual options activity."""
    try:
        from data.options import detect_unusual_activity
        # detect_unusual_activity needs a ticker — scan a few popular ones
        all_unusual = []
        for ticker in ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "MSFT"]:
            try:
                result = detect_unusual_activity(ticker)
                if result:
                    all_unusual.extend(result if isinstance(result, list) else [result])
            except Exception:
                pass
        return jsonify({"options": all_unusual[:20]})
    except Exception as e:
        return jsonify({"options": [], "note": "Options data temporarily unavailable"})


@api_bp.route("/earnings/calendar")
def earnings_calendar():
    """Return upcoming earnings calendar."""
    try:
        from models.earnings import fetch_earnings_history
        # Get earnings for a few major tickers
        all_earnings = []
        for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]:
            try:
                result = fetch_earnings_history(ticker)
                if result is not None and not result.empty:
                    latest = result.tail(1).to_dict('records')
                    for r in latest:
                        r['ticker'] = ticker
                    all_earnings.extend(latest)
            except Exception:
                pass
        return jsonify({"earnings": all_earnings})
    except Exception as e:
        return jsonify({"earnings": [], "note": "Earnings data temporarily unavailable"})


@api_bp.route("/insiders/recent")
def insiders_recent():
    """Return recent insider trades from SEC Edgar."""
    try:
        from data.sec_edgar import get_company_filings
        # Get recent filings for major companies
        all_filings = []
        for ticker in ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]:
            try:
                result = get_company_filings(ticker, form_type="4")
                if result:
                    all_filings.extend(result if isinstance(result, list) else [result])
            except Exception:
                pass
        return jsonify({"trades": all_filings[:20]})
    except Exception as e:
        return jsonify({"trades": [], "note": "Insider data temporarily unavailable"})


@api_bp.route("/darkpool/volume")
def darkpool_volume():
    """Return dark pool volume data."""
    try:
        return jsonify({
            "data": [],
            "note": "Dark pool data requires premium data feeds. Coming soon."
        })
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════
# NEWS
# ═══════════════════════════════════════════════════════════════

@api_bp.route("/market/news")
def market_news():
    """Return recent market news."""
    try:
        import pandas as pd
        from data.news import fetch_rss_feeds
        news = fetch_rss_feeds()
        # Convert DataFrame to list of dicts
        if isinstance(news, pd.DataFrame):
            if not news.empty:
                news = news.head(20).to_dict("records")
            else:
                news = []
        elif isinstance(news, list):
            news = news[:20]
        else:
            news = []
        return jsonify({"articles": news})
    except Exception as e:
        return jsonify({"articles": [], "note": "News temporarily unavailable"})


# ═══════════════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════════════

@api_bp.route("/settings", methods=["GET"])
def get_settings():
    return jsonify({
        "theme": "dark",
        "version": "2.0",
        "data_sources": ["Yahoo Finance", "CoinGecko", "FRED", "GDELT", "NewsAPI"],
    })


@api_bp.route("/settings", methods=["POST"])
def update_settings():
    data = request.get_json() or {}
    return jsonify({"success": True, "updated": data})


# ═══════════════════════════════════════════════════════════════
# TICKER BAR
# ═══════════════════════════════════════════════════════════════

@api_bp.route("/ticker-bar")
def ticker_bar():
    """Return ticker bar data for the scrolling display."""
    try:
        import yfinance as yf

        symbols = {
            "SPY": "S&P 500", "QQQ": "NASDAQ", "DIA": "DOW", "IWM": "RUSSELL",
            "AAPL": "AAPL", "MSFT": "MSFT", "NVDA": "NVDA", "TSLA": "TSLA",
            "BTC-USD": "BTC", "ETH-USD": "ETH", "GLD": "GOLD", "^VIX": "VIX",
        }
        tickers = list(symbols.keys())
        data = yf.download(tickers, period="5d", group_by="ticker", progress=False)
        results = []
        for t in tickers:
            try:
                close = data[t]["Close"].dropna()
                if len(close) >= 2:
                    price = float(close.iloc[-1])
                    change = float(close.iloc[-1] / close.iloc[-2] - 1) * 100
                    results.append({
                        "symbol": symbols[t],
                        "price": round(price, 2),
                        "change": round(change, 2),
                    })
            except Exception:
                pass
        return jsonify(results)
    except Exception as e:
        return jsonify([])

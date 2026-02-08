"""
SENTINEL — Catalyst calendar.
Tracks known upcoming market-moving events.
Fed meetings, CPI releases, GDP prints, earnings seasons, crypto halvings.
"""

from datetime import datetime, timedelta

import pandas as pd

from utils.logger import get_logger

log = get_logger("catalysts.calendar")


# ── Recurring Event Templates ─────────────────────────────────
# These get populated dynamically; static examples for 2025-2026

FOMC_MEETINGS_2025_2026 = [
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
]

CPI_RELEASES_2025_2026 = [
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-11", "2025-08-12",
    "2025-09-10", "2025-10-14", "2025-11-12", "2025-12-10",
    "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-14",
    "2026-05-12", "2026-06-10", "2026-07-14", "2026-08-12",
    "2026-09-10", "2026-10-13", "2026-11-12", "2026-12-10",
]

GDP_RELEASES_2025_2026 = [
    "2025-01-30", "2025-03-27", "2025-04-30", "2025-06-26",
    "2025-07-30", "2025-09-25", "2025-10-30", "2025-12-23",
    "2026-01-29", "2026-03-26", "2026-04-29", "2026-06-25",
]

EARNINGS_SEASONS_2025_2026 = [
    {"start": "2025-01-13", "end": "2025-02-14", "label": "Q4 2024 Earnings"},
    {"start": "2025-04-14", "end": "2025-05-16", "label": "Q1 2025 Earnings"},
    {"start": "2025-07-14", "end": "2025-08-15", "label": "Q2 2025 Earnings"},
    {"start": "2025-10-13", "end": "2025-11-14", "label": "Q3 2025 Earnings"},
    {"start": "2026-01-12", "end": "2026-02-13", "label": "Q4 2025 Earnings"},
    {"start": "2026-04-13", "end": "2026-05-15", "label": "Q1 2026 Earnings"},
]

CRYPTO_EVENTS = [
    {"date": "2028-04-01", "label": "Bitcoin Halving (est.)", "impact": "high"},
    # Add ETH/SOL upgrades as they are announced
]


def build_catalyst_calendar() -> pd.DataFrame:
    """
    Build a comprehensive calendar of upcoming market catalysts.
    Returns DataFrame with: Date, Event, Category, Impact_Level.
    """
    events = []

    # FOMC meetings
    for date_str in FOMC_MEETINGS_2025_2026:
        events.append({
            "Date": date_str,
            "Event": "FOMC Meeting Decision",
            "Category": "Monetary Policy",
            "Impact": "high",
            "Uncertainty_Multiplier": 1.5,
        })

    # CPI releases
    for date_str in CPI_RELEASES_2025_2026:
        events.append({
            "Date": date_str,
            "Event": "CPI Report",
            "Category": "Inflation",
            "Impact": "high",
            "Uncertainty_Multiplier": 1.3,
        })

    # GDP releases
    for date_str in GDP_RELEASES_2025_2026:
        events.append({
            "Date": date_str,
            "Event": "GDP Report",
            "Category": "Growth",
            "Impact": "medium",
            "Uncertainty_Multiplier": 1.2,
        })

    # Earnings seasons
    for season in EARNINGS_SEASONS_2025_2026:
        events.append({
            "Date": season["start"],
            "Event": f"{season['label']} Begins",
            "Category": "Earnings",
            "Impact": "high",
            "Uncertainty_Multiplier": 1.3,
        })

    # Crypto events
    for evt in CRYPTO_EVENTS:
        events.append({
            "Date": evt["date"],
            "Event": evt["label"],
            "Category": "Crypto",
            "Impact": evt["impact"],
            "Uncertainty_Multiplier": 1.5,
        })

    df = pd.DataFrame(events)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    log.info(f"Built catalyst calendar: {len(df)} events")
    return df


def get_upcoming_catalysts(days_ahead: int = 30) -> pd.DataFrame:
    """Get catalysts within the next N days."""
    calendar = build_catalyst_calendar()
    now = pd.Timestamp.now()
    cutoff = now + pd.Timedelta(days=days_ahead)
    upcoming = calendar[(calendar["Date"] >= now) & (calendar["Date"] <= cutoff)]
    return upcoming


def get_uncertainty_multiplier(forecast_date: pd.Timestamp) -> float:
    """
    Check if a forecast date is near any catalyst.
    Returns a multiplier (>= 1.0) to widen confidence intervals.
    """
    calendar = build_catalyst_calendar()
    window = pd.Timedelta(days=3)

    nearby = calendar[
        (calendar["Date"] >= forecast_date - window)
        & (calendar["Date"] <= forecast_date + window)
    ]

    if nearby.empty:
        return 1.0

    # Use the maximum multiplier from nearby events
    return nearby["Uncertainty_Multiplier"].max()


def is_earnings_season(date: pd.Timestamp | None = None) -> bool:
    """Check if a given date falls within earnings season."""
    date = date or pd.Timestamp.now()
    for season in EARNINGS_SEASONS_2025_2026:
        start = pd.Timestamp(season["start"])
        end = pd.Timestamp(season["end"])
        if start <= date <= end:
            return True
    return False

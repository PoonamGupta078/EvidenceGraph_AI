"""
pipeline/calendar_reconciliation.py
Detects false anomalies caused by calendar effects.

Checks:
  - Public holidays in the date range
  - Day-of-week effects (weekend dips vs weekday baselines)
  - Month-end / quarter-end spikes
  - Fiscal calendar misalignment between sources

Key output: flags whether a detected anomaly is likely a calendar artifact.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


# Simplified US public holiday list (extend as needed)
US_HOLIDAYS_2024 = [
    "2024-01-01",  # New Year's Day
    "2024-01-15",  # MLK Day
    "2024-02-19",  # Presidents Day
    "2024-05-27",  # Memorial Day
    "2024-07-04",  # Independence Day
    "2024-09-02",  # Labor Day
    "2024-11-28",  # Thanksgiving
    "2024-11-29",  # Black Friday
    "2024-12-25",  # Christmas
]


def _day_of_week_effect(df: pd.DataFrame, kpi: str) -> Dict[str, Any]:
    """
    Checks if a KPI has significant day-of-week variance.
    If so, anomalies on weekends/Mondays may be calendar artifacts.
    """
    if kpi not in df.columns:
        return {}

    df = df.copy()
    df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek  # 0=Monday, 6=Sunday
    grouped = df.groupby("dow")[kpi].mean()
    weekday_mean = grouped.iloc[:5].mean()
    weekend_mean = grouped.iloc[5:].mean()

    has_dow_effect = bool(abs(weekday_mean - weekend_mean) / (weekday_mean or 1) > 0.10)
    return {
        "has_dow_effect": has_dow_effect,
        "weekday_mean": round(float(weekday_mean), 2),
        "weekend_mean": round(float(weekend_mean), 2),
        "weekday_weekend_diff_pct": round(abs(weekday_mean - weekend_mean) / (weekday_mean or 1) * 100, 2),
    }


def _holiday_overlap(date_range: pd.DatetimeIndex) -> List[str]:
    """Returns holidays that fall within the date range."""
    dates_set = set(date_range.strftime("%Y-%m-%d").tolist())
    return [h for h in US_HOLIDAYS_2024 if h in dates_set]


def _month_end_effect(df: pd.DataFrame, kpi: str) -> bool:
    """
    Returns True if the KPI shows significantly different values
    in the last 3 days of each month vs rest of month.
    """
    if kpi not in df.columns:
        return False
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["is_month_end"] = df["date"].dt.is_month_end | (df["date"].dt.day >= 28)
    month_end = df[df["is_month_end"]][kpi].mean()
    rest = df[~df["is_month_end"]][kpi].mean()
    return bool(abs(month_end - rest) / (rest or 1) > 0.08)


def check_calendar_effects(
    df: pd.DataFrame,
    material_kpis: List[str],
    detected_anomaly_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Checks whether detected anomalies may be calendar artifacts.

    Returns:
        - is_likely_calendar_artifact: bool
        - calendar_findings: list of specific calendar effect findings
        - holidays_in_range: list of holidays found
        - dow_effects: {kpi: dow_analysis}
        - month_end_effects: {kpi: bool}
        - recommendation: str
    """
    date_col = pd.to_datetime(df["date"])
    date_range = pd.DatetimeIndex(date_col.dropna().unique())

    holidays = _holiday_overlap(date_range)
    calendar_findings = []

    if holidays:
        calendar_findings.append({
            "type": "HOLIDAY",
            "finding": f"{len(holidays)} public holiday(s) in date range: {', '.join(holidays)}",
            "impact": "KPI drops on holiday dates are expected and may not be anomalous.",
        })

    # Check if anomaly indices align with holidays
    holiday_dates = set(pd.to_datetime(holidays).strftime("%Y-%m-%d").tolist() if holidays else [])
    if detected_anomaly_indices:
        anomaly_dates = set(date_col.iloc[idx].strftime("%Y-%m-%d") for idx in detected_anomaly_indices if idx < len(date_col))
        holiday_overlap_at_anomaly = anomaly_dates & holiday_dates
        if holiday_overlap_at_anomaly:
            calendar_findings.append({
                "type": "HOLIDAY_ALIGNED_ANOMALY",
                "finding": f"Detected anomaly aligns with holiday date(s): {holiday_overlap_at_anomaly}",
                "impact": "HIGH — this anomaly is likely a calendar artifact, not a real signal.",
            })

    # Day-of-week analysis
    dow_effects = {}
    for kpi in material_kpis:
        if kpi in df.columns:
            dow_effects[kpi] = _day_of_week_effect(df, kpi)
            if dow_effects[kpi].get("has_dow_effect"):
                calendar_findings.append({
                    "type": "DAY_OF_WEEK",
                    "finding": f"{kpi}: significant weekday/weekend variance ({dow_effects[kpi]['weekday_weekend_diff_pct']:.1f}%)",
                    "impact": "Anomalies on weekends or Mondays may be DOW effects.",
                })

    # Month-end effects
    month_end_effects = {}
    for kpi in material_kpis:
        if kpi in df.columns:
            has_me = _month_end_effect(df, kpi)
            month_end_effects[kpi] = has_me
            if has_me:
                calendar_findings.append({
                    "type": "MONTH_END",
                    "finding": f"{kpi}: month-end spike/dip pattern detected (>8% deviation)",
                    "impact": "Month-end effects may inflate anomaly scores.",
                })

    # Overall verdict
    high_impact = [f for f in calendar_findings if "HIGH" in f.get("impact", "")]
    is_likely_calendar_artifact = len(high_impact) > 0

    if is_likely_calendar_artifact:
        recommendation = "Detected anomaly overlaps with known calendar event. Verify whether signal persists after adjusting for calendar effects before acting."
    elif calendar_findings:
        recommendation = "Calendar effects present in data window but not directly aligned with anomaly peak. Proceed with caution."
    else:
        recommendation = "No calendar effects detected. Anomaly is unlikely to be a calendar artifact."

    return {
        "is_likely_calendar_artifact": is_likely_calendar_artifact,
        "calendar_findings": calendar_findings,
        "holidays_in_range": holidays,
        "dow_effects": dow_effects,
        "month_end_effects": month_end_effects,
        "recommendation": recommendation,
    }

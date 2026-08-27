"""
pipeline/calendar_reconciliation.py

Detects potential false anomalies caused by calendar effects.

Checks:
  - Indian public holidays in the date range
  - Day-of-week effects
  - Month-end effects
  - Quarter-end effects

Important:
  - Calendar effects are treated as supporting evidence, NOT ground truth.
  - A holiday is only considered relevant when an anomaly actually overlaps it.
  - Region E's seasonal_index is a business-seasonality signal and is NOT
    treated as an Indian holiday effect.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------------
# Indian public / major national holidays relevant to the synthetic 2024 data
# ---------------------------------------------------------------------------
#
# The synthetic dataset covers January 2024 through March 2024.
# Therefore only holidays inside this period are relevant.
#
# These are calendar reference dates only. They are NOT scenario ground truth.
#
INDIAN_HOLIDAYS_2024 = {
    "2024-01-26": "Republic Day",
    "2024-03-08": "Maha Shivratri",
    "2024-03-25": "Holi",
}


def _day_of_week_effect(
    df: pd.DataFrame,
    kpi: str,
) -> Dict[str, Any]:
    """
    Check whether the KPI has a meaningful weekday/weekend pattern.

    This does NOT say that a particular anomaly is caused by the
    day-of-week effect. It only establishes that such a pattern exists.
    """

    if kpi not in df.columns or "date" not in df.columns:
        return {}

    temp = df[["date", kpi]].copy()
    temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
    temp[kpi] = pd.to_numeric(temp[kpi], errors="coerce")

    temp = temp.dropna(subset=["date", kpi])

    if len(temp) < 7:
        return {
            "has_dow_effect": False,
            "status": "INSUFFICIENT_DATA",
        }

    temp["dow"] = temp["date"].dt.dayofweek

    weekday_values = temp.loc[temp["dow"] < 5, kpi]
    weekend_values = temp.loc[temp["dow"] >= 5, kpi]

    if len(weekday_values) == 0 or len(weekend_values) == 0:
        return {
            "has_dow_effect": False,
            "status": "INSUFFICIENT_WEEKEND_WEEKDAY_DATA",
        }

    weekday_mean = float(weekday_values.mean())
    weekend_mean = float(weekend_values.mean())

    denominator = abs(weekday_mean) if abs(weekday_mean) > 1e-12 else 1.0

    diff_pct = abs(weekend_mean - weekday_mean) / denominator * 100

    return {
        "has_dow_effect": bool(diff_pct > 10.0),
        "weekday_mean": round(weekday_mean, 2),
        "weekend_mean": round(weekend_mean, 2),
        "weekday_weekend_diff_pct": round(diff_pct, 2),
        "status": "OK",
    }


def _holiday_overlap(
    date_range: pd.DatetimeIndex,
) -> List[Dict[str, str]]:
    """
    Return configured Indian holidays that actually occur in the
    supplied date range.
    """

    holidays = []

    for date in date_range:
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")

        if date_str in INDIAN_HOLIDAYS_2024:
            holidays.append({
                "date": date_str,
                "name": INDIAN_HOLIDAYS_2024[date_str],
            })

    return holidays


def _holiday_anomaly_overlap(
    df: pd.DataFrame,
    detected_anomaly_indices: Optional[List[int]],
    holidays: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """
    Determine whether supplied anomaly indices overlap configured holidays.
    """

    if not detected_anomaly_indices or not holidays:
        return []

    if "date" not in df.columns:
        return []

    dates = pd.to_datetime(df["date"], errors="coerce")

    holiday_lookup = {
        item["date"]: item["name"]
        for item in holidays
    }

    overlaps = []

    for idx in detected_anomaly_indices:
        if idx < 0 or idx >= len(df):
            continue

        anomaly_date = dates.iloc[idx]

        if pd.isna(anomaly_date):
            continue

        date_str = anomaly_date.strftime("%Y-%m-%d")

        if date_str in holiday_lookup:
            overlaps.append({
                "index": int(idx),
                "date": date_str,
                "holiday": holiday_lookup[date_str],
            })

    return overlaps


def _month_end_effect(
    df: pd.DataFrame,
    kpi: str,
) -> Dict[str, Any]:
    """
    Check whether values during the final three calendar days of a month
    differ materially from the rest of the month.
    """

    if kpi not in df.columns or "date" not in df.columns:
        return {
            "has_month_end_effect": False,
            "status": "MISSING_COLUMN",
        }

    temp = df[["date", kpi]].copy()
    temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
    temp[kpi] = pd.to_numeric(temp[kpi], errors="coerce")
    temp = temp.dropna(subset=["date", kpi])

    if len(temp) < 10:
        return {
            "has_month_end_effect": False,
            "status": "INSUFFICIENT_DATA",
        }

    temp["is_month_end"] = temp["date"].dt.day >= 28

    month_end_values = temp.loc[temp["is_month_end"], kpi]
    rest_values = temp.loc[~temp["is_month_end"], kpi]

    if len(month_end_values) == 0 or len(rest_values) == 0:
        return {
            "has_month_end_effect": False,
            "status": "INSUFFICIENT_DATA",
        }

    month_end_mean = float(month_end_values.mean())
    rest_mean = float(rest_values.mean())

    denominator = abs(rest_mean) if abs(rest_mean) > 1e-12 else 1.0

    diff_pct = abs(month_end_mean - rest_mean) / denominator * 100

    return {
        "has_month_end_effect": bool(diff_pct > 8.0),
        "month_end_mean": round(month_end_mean, 2),
        "rest_mean": round(rest_mean, 2),
        "month_end_diff_pct": round(diff_pct, 2),
        "status": "OK",
    }


def _quarter_end_effect(
    df: pd.DataFrame,
    kpi: str,
) -> Dict[str, Any]:
    """
    Check whether the final three days of a quarter behave differently
    from the rest of the observed period.

    The synthetic data ends on 2024-03-31, so this can identify a
    quarter-end pattern without assuming that it is causal.
    """

    if kpi not in df.columns or "date" not in df.columns:
        return {
            "has_quarter_end_effect": False,
            "status": "MISSING_COLUMN",
        }

    temp = df[["date", kpi]].copy()
    temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
    temp[kpi] = pd.to_numeric(temp[kpi], errors="coerce")
    temp = temp.dropna(subset=["date", kpi])

    if len(temp) < 10:
        return {
            "has_quarter_end_effect": False,
            "status": "INSUFFICIENT_DATA",
        }

    temp["is_quarter_end"] = (
        temp["date"].dt.is_quarter_end
        | (
            temp["date"].dt.month.isin([3, 6, 9, 12])
            & (temp["date"].dt.day >= 29)
        )
    )

    quarter_end_values = temp.loc[temp["is_quarter_end"], kpi]
    rest_values = temp.loc[~temp["is_quarter_end"], kpi]

    if len(quarter_end_values) == 0 or len(rest_values) == 0:
        return {
            "has_quarter_end_effect": False,
            "status": "INSUFFICIENT_DATA",
        }

    quarter_end_mean = float(quarter_end_values.mean())
    rest_mean = float(rest_values.mean())

    denominator = abs(rest_mean) if abs(rest_mean) > 1e-12 else 1.0

    diff_pct = abs(quarter_end_mean - rest_mean) / denominator * 100

    return {
        "has_quarter_end_effect": bool(diff_pct > 8.0),
        "quarter_end_mean": round(quarter_end_mean, 2),
        "rest_mean": round(rest_mean, 2),
        "quarter_end_diff_pct": round(diff_pct, 2),
        "status": "OK",
    }


def check_calendar_effects(
    df: pd.DataFrame,
    material_kpis: List[str],
    detected_anomaly_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Check whether detected anomalies may be calendar artifacts.

    Returns:
        is_likely_calendar_artifact:
            True only when an explicitly configured holiday overlaps
            a detected anomaly.

        calendar_findings:
            Detailed calendar findings.

        holidays_in_range:
            Indian holidays occurring inside the observed date range.

        dow_effects:
            Day-of-week analysis per KPI.

        month_end_effects:
            Month-end analysis per KPI.

        quarter_end_effects:
            Quarter-end analysis per KPI.

        recommendation:
            Human-readable recommendation.
    """

    if "date" not in df.columns:
        return {
            "is_likely_calendar_artifact": False,
            "calendar_findings": [],
            "holidays_in_range": [],
            "dow_effects": {},
            "month_end_effects": {},
            "quarter_end_effects": {},
            "recommendation": "Calendar check skipped: date column is missing.",
        }

    date_series = pd.to_datetime(df["date"], errors="coerce").dropna()

    if date_series.empty:
        return {
            "is_likely_calendar_artifact": False,
            "calendar_findings": [],
            "holidays_in_range": [],
            "dow_effects": {},
            "month_end_effects": {},
            "quarter_end_effects": {},
            "recommendation": "Calendar check skipped: no valid dates available.",
        }

    date_range = pd.DatetimeIndex(date_series.unique())

    calendar_findings: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 1. Indian holiday analysis
    # ------------------------------------------------------------------

    holidays = _holiday_overlap(date_range)

    if holidays:
        holiday_names = ", ".join(
            f"{h['name']} ({h['date']})"
            for h in holidays
        )

        calendar_findings.append({
            "type": "HOLIDAY_IN_RANGE",
            "finding": f"Configured Indian holiday(s) present: {holiday_names}",
            "impact": (
                "Calendar context only. A holiday should only be treated "
                "as an anomaly explanation if the detected anomaly overlaps it."
            ),
        })

    holiday_anomaly_overlaps = _holiday_anomaly_overlap(
        df,
        detected_anomaly_indices,
        holidays,
    )

    if holiday_anomaly_overlaps:
        for overlap in holiday_anomaly_overlaps:
            calendar_findings.append({
                "type": "HOLIDAY_ALIGNED_ANOMALY",
                "finding": (
                    f"Detected anomaly on {overlap['date']} overlaps "
                    f"{overlap['holiday']}."
                ),
                "impact": (
                    "HIGH — calendar effect is a plausible explanation "
                    "and should be checked before taking action."
                ),
            })

    # ------------------------------------------------------------------
    # 2. Day-of-week analysis
    # ------------------------------------------------------------------

    dow_effects: Dict[str, Any] = {}

    for kpi in material_kpis:
        if kpi not in df.columns:
            continue

        result = _day_of_week_effect(df, kpi)
        dow_effects[kpi] = result

        if result.get("has_dow_effect"):
            calendar_findings.append({
                "type": "DAY_OF_WEEK",
                "kpi": kpi,
                "finding": (
                    f"{kpi}: weekday/weekend difference is "
                    f"{result['weekday_weekend_diff_pct']:.1f}%."
                ),
                "impact": (
                    "Calendar pattern exists. Individual anomalies should "
                    "be compared against the same day-of-week baseline."
                ),
            })

    # ------------------------------------------------------------------
    # 3. Month-end analysis
    # ------------------------------------------------------------------

    month_end_effects: Dict[str, Any] = {}

    for kpi in material_kpis:
        if kpi not in df.columns:
            continue

        result = _month_end_effect(df, kpi)
        month_end_effects[kpi] = result

        if result.get("has_month_end_effect"):
            calendar_findings.append({
                "type": "MONTH_END",
                "kpi": kpi,
                "finding": (
                    f"{kpi}: month-end values differ from the rest "
                    f"of the month by {result['month_end_diff_pct']:.1f}%."
                ),
                "impact": (
                    "Month-end behavior may contribute to an apparent "
                    "anomaly."
                ),
            })

    # ------------------------------------------------------------------
    # 4. Quarter-end analysis
    # ------------------------------------------------------------------

    quarter_end_effects: Dict[str, Any] = {}

    for kpi in material_kpis:
        if kpi not in df.columns:
            continue

        result = _quarter_end_effect(df, kpi)
        quarter_end_effects[kpi] = result

        if result.get("has_quarter_end_effect"):
            calendar_findings.append({
                "type": "QUARTER_END",
                "kpi": kpi,
                "finding": (
                    f"{kpi}: quarter-end values differ from the rest "
                    f"of the period by "
                    f"{result['quarter_end_diff_pct']:.1f}%."
                ),
                "impact": (
                    "Quarter-end behavior may contribute to an apparent "
                    "anomaly."
                ),
            })

    # ------------------------------------------------------------------
    # 5. Final verdict
    # ------------------------------------------------------------------

    # IMPORTANT:
    # Merely having a holiday, DOW effect, month-end effect, or quarter-end
    # effect does NOT make the anomaly a calendar artifact.
    #
    # We only mark it likely when the supplied anomaly index actually
    # overlaps an explicitly configured holiday.
    #
    is_likely_calendar_artifact = bool(holiday_anomaly_overlaps)

    if is_likely_calendar_artifact:
        recommendation = (
            "Detected anomaly overlaps an Indian holiday. "
            "Verify whether the signal persists after controlling for "
            "the holiday before taking action."
        )

    elif calendar_findings:
        recommendation = (
            "Calendar patterns are present in the observed data, but no "
            "detected anomaly was directly linked to a configured holiday. "
            "Compare anomalies against day-of-week, month-end, and "
            "quarter-end baselines before acting."
        )

    else:
        recommendation = (
            "No material calendar pattern was detected. "
            "The anomaly is unlikely to be explained by the checked "
            "calendar effects."
        )

    return {
        "is_likely_calendar_artifact": is_likely_calendar_artifact,
        "calendar_findings": calendar_findings,
        "holidays_in_range": holidays,
        "holiday_anomaly_overlaps": holiday_anomaly_overlaps,
        "dow_effects": dow_effects,
        "month_end_effects": month_end_effects,
        "quarter_end_effects": quarter_end_effects,
        "recommendation": recommendation,
    }
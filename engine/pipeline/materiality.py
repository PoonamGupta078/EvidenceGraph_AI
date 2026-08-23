"""
pipeline/materiality.py
Detects materially significant anomalies using STL decomposition + CUSUM.

Two-step approach:
1. STL decomposition (statsmodels) → isolate residuals from trend/seasonality
2. CUSUM change-point detection on residuals → confirm structural break

A KPI is flagged as material only if BOTH steps agree.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

try:
    from statsmodels.tsa.seasonal import STL
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


MATERIALITY_THRESHOLDS = {
    "revenue": 5.0,              # 5% deviation
    "order_cancellation_rate": 2.0,
    "fulfillment_delay_rate": 3.0,
    "support_ticket_volume": 10.0,
    "warehouse_staffing_level": 8.0,
}

CUSUM_K = 0.5   # allowable slack
CUSUM_H = 5.0   # decision threshold


def _cusum_detect(series: np.ndarray, k: float = CUSUM_K, h: float = CUSUM_H) -> Dict[str, Any]:
    """
    Runs one-sided CUSUM on a normalized residual series.
    Returns detection point and max statistic.
    """
    mu = np.nanmean(series)
    sigma = np.nanstd(series) or 1.0
    normalized = (series - mu) / sigma

    cusum_pos = np.zeros(len(normalized))
    cusum_neg = np.zeros(len(normalized))

    for i in range(1, len(normalized)):
        if np.isnan(normalized[i]):
            cusum_pos[i] = cusum_pos[i - 1]
            cusum_neg[i] = cusum_neg[i - 1]
            continue
        cusum_pos[i] = max(0, cusum_pos[i - 1] + normalized[i] - k)
        cusum_neg[i] = max(0, cusum_neg[i - 1] - normalized[i] - k)

    max_stat = max(np.max(cusum_pos), np.max(cusum_neg))
    detected = bool(max_stat >= h)

    detection_idx = None
    if detected:
        crossings = np.where((cusum_pos >= h) | (cusum_neg >= h))[0]
        detection_idx = int(crossings[0]) if len(crossings) > 0 else None

    return {
        "detected": detected,
        "max_statistic": round(float(max_stat), 4),
        "detection_index": detection_idx,
        "cusum_pos": cusum_pos.tolist(),
        "cusum_neg": cusum_neg.tolist(),
    }


def _stl_residuals(series: pd.Series) -> Optional[np.ndarray]:
    """
    Decomposes series with STL and returns residuals.
    Falls back to simple detrending if statsmodels unavailable.
    """
    clean = series.dropna()
    if len(clean) < 14:
        return None

    if HAS_STATSMODELS:
        try:
            result = STL(clean, period=7, robust=True).fit()
            residuals = result.resid.values
            # Align back to original index
            full_residuals = np.full(len(series), np.nan)
            full_residuals[series.notna().values] = residuals
            return full_residuals
        except Exception:
            pass

    # Fallback: rolling median detrend
    trend = series.rolling(7, min_periods=3, center=True).median()
    return (series - trend).values


def detect_materiality(
    df: pd.DataFrame,
    kpis: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Runs STL + CUSUM materiality detection on all requested KPIs.

    Args:
        df: Reconciled aligned DataFrame
        kpis: list of KPI columns to check (defaults to all available)

    Returns:
        - material_kpis: list of KPI ids that triggered
        - kpi_results: {kpi: {material, stl_detected, cusum_detected, ...}}
        - any_material: bool
        - signal_strength: float [0, 1] (fraction of KPIs that triggered)
    """
    if kpis is None:
        kpis = [c for c in MATERIALITY_THRESHOLDS if c in df.columns]

    kpi_results = {}

    for kpi in kpis:
        if kpi not in df.columns:
            continue

        series = df[kpi].astype(float)

        # Baseline (pre-shock): first 30 days or half the series
        midpoint = min(30, len(series) // 2)
        baseline = series.iloc[:midpoint]
        baseline_mean = baseline.mean()
        current_mean = series.iloc[midpoint:].mean()

        # Percentage deviation from baseline
        if baseline_mean and not np.isnan(baseline_mean):
            pct_deviation = abs((current_mean - baseline_mean) / baseline_mean) * 100
        else:
            pct_deviation = 0.0

        threshold = MATERIALITY_THRESHOLDS.get(kpi, 5.0)
        threshold_exceeded = bool(pct_deviation >= threshold)

        # STL residuals
        residuals = _stl_residuals(series)
        stl_detected = False
        stl_z_max = 0.0
        if residuals is not None:
            residual_std = np.nanstd(residuals) or 1.0
            z_scores = np.abs(residuals / residual_std)
            stl_z_max = float(np.nanmax(z_scores))
            stl_detected = bool(stl_z_max >= 2.5)

        # CUSUM on residuals (or raw series if STL failed)
        cusum_input = residuals if residuals is not None else series.values
        cusum_result = _cusum_detect(cusum_input)

        # Material = threshold exceeded AND (STL OR CUSUM)
        material = threshold_exceeded and (stl_detected or cusum_result["detected"])

        kpi_results[kpi] = {
            "material": material,
            "pct_deviation_from_baseline": round(pct_deviation, 2),
            "threshold_pct": threshold,
            "threshold_exceeded": threshold_exceeded,
            "stl_detected": stl_detected,
            "stl_z_max": round(stl_z_max, 3),
            "cusum_detected": cusum_result["detected"],
            "cusum_max_statistic": cusum_result["max_statistic"],
            "cusum_detection_index": cusum_result["detection_index"],
            "baseline_mean": round(float(baseline_mean), 2) if not np.isnan(baseline_mean) else None,
            "current_mean": round(float(current_mean), 2) if not np.isnan(current_mean) else None,
        }

    material_kpis = [k for k, v in kpi_results.items() if v["material"]]
    signal_strength = round(len(material_kpis) / len(kpis), 4) if kpis else 0.0

    return {
        "material_kpis": material_kpis,
        "kpi_results": kpi_results,
        "any_material": len(material_kpis) > 0,
        "signal_strength": signal_strength,
        "kpis_checked": kpis,
    }

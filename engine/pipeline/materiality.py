"""
pipeline/materiality.py

Detects materially significant anomalies using:
1. Business materiality threshold
2. STL residual anomaly detection
3. CUSUM change-point detection

A KPI is considered material only when:
    business materiality threshold is exceeded
    AND
    (STL anomaly OR CUSUM structural-break signal)

This module detects statistical/business signals only.
It does not make causal claims.
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
    "revenue": 5.0,
    "order_cancellation_rate": 2.0,
    "fulfillment_delay_rate": 3.0,
    "support_ticket_volume": 10.0,
    "warehouse_staffing_level": 8.0,
    "unit_price": 3.0,
    "marketing_spend": 15.0,
}

CUSUM_K = 0.5
CUSUM_H = 5.0


def _cusum_detect(
    series: np.ndarray,
    k: float = CUSUM_K,
    h: float = CUSUM_H,
) -> Dict[str, Any]:
    """
    Runs one-sided CUSUM on a normalized series.

    NaN observations are ignored while preserving the cumulative state.

    Returns:
        detected:
            Whether the CUSUM statistic crossed the decision threshold.

        detection_index:
            First observation where the threshold was crossed.

        max_statistic:
            Maximum positive/negative CUSUM statistic.
    """

    values = np.asarray(series, dtype=float)

    valid = np.isfinite(values)

    if valid.sum() < 5:
        return {
            "detected": False,
            "max_statistic": 0.0,
            "detection_index": None,
            "cusum_pos": np.zeros(len(values)).tolist(),
            "cusum_neg": np.zeros(len(values)).tolist(),
        }

    mu = float(np.nanmean(values))
    sigma = float(np.nanstd(values))

    if not np.isfinite(sigma) or sigma == 0:
        sigma = 1.0

    normalized = (values - mu) / sigma

    cusum_pos = np.zeros(len(normalized), dtype=float)
    cusum_neg = np.zeros(len(normalized), dtype=float)

    for i in range(1, len(normalized)):

        if not np.isfinite(normalized[i]):
            cusum_pos[i] = cusum_pos[i - 1]
            cusum_neg[i] = cusum_neg[i - 1]
            continue

        cusum_pos[i] = max(
            0.0,
            cusum_pos[i - 1] + normalized[i] - k,
        )

        cusum_neg[i] = max(
            0.0,
            cusum_neg[i - 1] - normalized[i] - k,
        )

    max_stat = max(
        float(np.max(cusum_pos)),
        float(np.max(cusum_neg)),
    )

    detected = bool(max_stat >= h)

    detection_idx = None

    if detected:
        crossings = np.where(
            (cusum_pos >= h)
            | (cusum_neg >= h)
        )[0]

        if len(crossings) > 0:
            detection_idx = int(crossings[0])

    return {
        "detected": detected,
        "max_statistic": round(max_stat, 4),
        "detection_index": detection_idx,
        "cusum_pos": cusum_pos.tolist(),
        "cusum_neg": cusum_neg.tolist(),
    }


def _stl_residuals(series: pd.Series) -> Optional[np.ndarray]:
    """
    Decomposes a time series and returns residuals.

    Uses STL with weekly seasonality when statsmodels is available.

    Falls back to a rolling-median detrend when STL cannot be used.

    Returns:
        Residual array aligned with the original series.
    """

    values = pd.to_numeric(series, errors="coerce")

    if values.notna().sum() < 14:
        return None

    if HAS_STATSMODELS:
        try:
            clean = values.dropna()

            result = STL(
                clean,
                period=7,
                robust=True,
            ).fit()

            residuals = result.resid.to_numpy(dtype=float)

            full_residuals = np.full(
                len(values),
                np.nan,
                dtype=float,
            )

            full_residuals[values.notna().to_numpy()] = residuals

            return full_residuals

        except Exception:
            pass

    # Deterministic fallback.
    trend = values.rolling(
        window=7,
        min_periods=3,
        center=True,
    ).median()

    return (values - trend).to_numpy(dtype=float)


def _stl_anomaly_detect(
    residuals: Optional[np.ndarray],
    threshold: float = 2.5,
) -> Dict[str, Any]:
    """
    Detects unusually large STL residuals.

    This is an anomaly signal, not proof of a structural break
    and not proof of causality.
    """

    if residuals is None:
        return {
            "detected": False,
            "z_max": 0.0,
            "detection_index": None,
        }

    values = np.asarray(residuals, dtype=float)

    valid = np.isfinite(values)

    if valid.sum() < 5:
        return {
            "detected": False,
            "z_max": 0.0,
            "detection_index": None,
        }

    valid_values = values[valid]

    residual_std = float(np.std(valid_values))

    if not np.isfinite(residual_std) or residual_std == 0:
        return {
            "detected": False,
            "z_max": 0.0,
            "detection_index": None,
        }

    z_scores = np.full(len(values), np.nan)

    z_scores[valid] = np.abs(
        valid_values / residual_std
    )

    z_max = float(np.nanmax(z_scores))

    detected = bool(z_max >= threshold)

    detection_index = None

    if detected:
        crossings = np.where(z_scores >= threshold)[0]

        if len(crossings) > 0:
            detection_index = int(crossings[0])

    return {
        "detected": detected,
        "z_max": round(z_max, 3),
        "detection_index": detection_index,
    }


def detect_materiality(
    df: pd.DataFrame,
    kpis: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Runs business materiality + STL + CUSUM detection.

    A KPI is material only when:

        threshold_exceeded
        AND
        (stl_detected OR cusum_detected)

    This module does not claim causality.
    """

    if kpis is None:
        kpis = [
            c
            for c in MATERIALITY_THRESHOLDS
            if c in df.columns
        ]

    kpi_results: Dict[str, Dict[str, Any]] = {}

    for kpi in kpis:

        if kpi not in df.columns:
            continue

        series = pd.to_numeric(
            df[kpi],
            errors="coerce",
        )

        if series.notna().sum() < 14:
            kpi_results[kpi] = {
                "material": False,
                "status": "INSUFFICIENT_DATA",
            }
            continue

        # ---------------------------------------------------------
        # Baseline / current comparison
        # ---------------------------------------------------------

        midpoint = min(
            30,
            len(series) // 2,
        )

        if midpoint < 1 or len(series) - midpoint < 1:
            kpi_results[kpi] = {
                "material": False,
                "status": "INSUFFICIENT_DATA",
            }
            continue

        baseline = series.iloc[:midpoint]
        current = series.iloc[midpoint:]

        baseline_mean = baseline.mean()
        current_mean = current.mean()

        if (
            not np.isfinite(baseline_mean)
            or not np.isfinite(current_mean)
        ):
            kpi_results[kpi] = {
                "material": False,
                "status": "INSUFFICIENT_DATA",
            }
            continue

        absolute_impact = (
            float(current_mean - baseline_mean)
        )

        if baseline_mean != 0:
            pct_deviation = (
                abs(absolute_impact / baseline_mean)
                * 100.0
            )
        else:
            pct_deviation = 0.0

        if absolute_impact > 0:
            direction = "increase"
        elif absolute_impact < 0:
            direction = "decrease"
        else:
            direction = "flat"

        threshold = MATERIALITY_THRESHOLDS.get(
            kpi,
            5.0,
        )

        threshold_exceeded = bool(
            pct_deviation >= threshold
        )

        # ---------------------------------------------------------
        # STL
        # ---------------------------------------------------------

        residuals = _stl_residuals(series)

        stl_result = _stl_anomaly_detect(
            residuals
        )

        # ---------------------------------------------------------
        # CUSUM
        # ---------------------------------------------------------

        if residuals is not None:
            cusum_input = residuals
        else:
            cusum_input = series.to_numpy(dtype=float)

        cusum_result = _cusum_detect(
            cusum_input
        )

        # ---------------------------------------------------------
        # Final materiality decision
        # ---------------------------------------------------------

        material = bool(
            threshold_exceeded
            and (
                stl_result["detected"]
                or cusum_result["detected"]
            )
        )

        statistical_method = (
            "STL + CUSUM"
            if HAS_STATSMODELS
            else "ROBUST_ROLLING_DETREND + CUSUM"
        )

        kpi_results[kpi] = {
            "material": material,
            "status": "OK",

            "pct_deviation_from_baseline": round(
                pct_deviation,
                2,
            ),

            "absolute_impact": round(
                absolute_impact,
                2,
            ),

            "direction": direction,

            "threshold_pct": threshold,
            "threshold_exceeded": threshold_exceeded,

            "statistical_method": [
                statistical_method,
                "CUSUM",
            ],

            "stl_detected": stl_result["detected"],
            "stl_z_max": stl_result["z_max"],
            "stl_detection_index": (
                stl_result["detection_index"]
            ),

            "cusum_detected": cusum_result["detected"],
            "cusum_max_statistic": (
                cusum_result["max_statistic"]
            ),
            "cusum_detection_index": (
                cusum_result["detection_index"]
            ),

            "baseline_mean": round(
                float(baseline_mean),
                2,
            ),

            "current_mean": round(
                float(current_mean),
                2,
            ),
        }

    material_kpis = [
        k
        for k, result in kpi_results.items()
        if result.get("material", False)
    ]

    material_kpi_ratio = round(
        len(material_kpis) / len(kpis),
        4,
    ) if kpis else 0.0

    return {
        "material_kpis": material_kpis,
        "kpi_results": kpi_results,
        "any_material": bool(
            len(material_kpis) > 0
        ),
        "material_kpi_ratio": material_kpi_ratio,

        # Backward compatibility
        "signal_strength": material_kpi_ratio,

        "kpis_checked": kpis,
    }
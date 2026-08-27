"""
pipeline/intervention_sandbox.py

Counterfactual simulation for the Decision Workspace.

Given:
  - A supported intervention lever
  - A target value for that lever
  - Historical aligned KPI data

The sandbox propagates the intervention through the configured
Evidence Graph causal chain using lag-aware historical regressions.

Current causal chain:
    warehouse_staffing_level
        ↓ 2 days
    fulfillment_delay_rate
        ↓ 1 day
    support_ticket_volume
        ↓ 1 day
    order_cancellation_rate
        ↓ 1 day
    revenue

Important:
  - These regressions are observational and are used for scenario
    estimation, not proof of causal effect.
  - Confidence is based on model fit (R²), sample size, and whether
    the intervention remains within the historical training range.
  - Revenue uncertainty is derived from historical model residuals,
    not an arbitrary percentage.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Evidence Graph causal chain
# ---------------------------------------------------------------------------

CAUSAL_CHAIN = [
    ("warehouse_staffing_level", "fulfillment_delay_rate", 2),
    ("fulfillment_delay_rate", "support_ticket_volume", 1),
    ("support_ticket_volume", "order_cancellation_rate", 1),
    ("order_cancellation_rate", "revenue", 1),
]


CAUSAL_CHAIN_ORDER = [
    "warehouse_staffing_level",
    "fulfillment_delay_rate",
    "support_ticket_volume",
    "order_cancellation_rate",
    "revenue",
]


# ---------------------------------------------------------------------------
# Supported intervention levers
# ---------------------------------------------------------------------------

LEVER_TARGETS = {
    "warehouse_staffing_level": {
        "min": 60,
        "max": 100,
        "default_recovery": 90,
    },
    "fulfillment_delay_rate": {
        "min": 0,
        "max": 40,
        "default_recovery": 10,
    },
    "support_ticket_volume": {
        "min": 0,
        "max": 500,
        "default_recovery": 55,
    },
}


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------

def _fit_regression(
    x: np.ndarray,
    y: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    Fit a one-dimensional historical regression.

    Returns:
        Dictionary containing the fitted model and diagnostics.

    sklearn LinearRegression is preferred. If sklearn is unavailable,
    a deterministic least-squares linear fallback is used.
    """

    valid = np.isfinite(x) & np.isfinite(y)

    x_valid = np.asarray(x)[valid].astype(float)
    y_valid = np.asarray(y)[valid].astype(float)

    if len(x_valid) < 5:
        return None

    x_matrix = x_valid.reshape(-1, 1)

    # ------------------------------------------------------------------
    # sklearn implementation
    # ------------------------------------------------------------------
    if HAS_SKLEARN:
        model = LinearRegression()
        model.fit(x_matrix, y_valid)

        predictions = model.predict(x_matrix)

        r2 = float(r2_score(y_valid, predictions))

        residuals = y_valid - predictions
        residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

        return {
            "model": model,
            "r2": round(max(0.0, r2), 4),
            "residual_std": round(max(0.0, residual_std), 4),
            "n_samples": int(len(x_valid)),
            "x_min": float(np.min(x_valid)),
            "x_max": float(np.max(x_valid)),
            "y_min": float(np.min(y_valid)),
            "y_max": float(np.max(y_valid)),
            "method": "LinearRegression",
        }

    # ------------------------------------------------------------------
    # Deterministic NumPy least-squares fallback
    # ------------------------------------------------------------------
    design = np.column_stack([
        np.ones(len(x_valid)),
        x_valid,
    ])

    try:
        coefficients, _, _, _ = np.linalg.lstsq(
            design,
            y_valid,
            rcond=None,
        )
    except Exception:
        return None

    intercept = float(coefficients[0])
    slope = float(coefficients[1])

    predictions = intercept + slope * x_valid

    ss_res = float(np.sum((y_valid - predictions) ** 2))
    ss_tot = float(np.sum((y_valid - np.mean(y_valid)) ** 2))

    r2 = 0.0 if ss_tot == 0 else 1.0 - (ss_res / ss_tot)

    residuals = y_valid - predictions
    residual_std = (
        float(np.std(residuals, ddof=1))
        if len(residuals) > 1
        else 0.0
    )

    class NumpyLinearModel:
        def __init__(self, intercept: float, slope: float):
            self.intercept = intercept
            self.slope = slope

        def predict(self, X: np.ndarray) -> np.ndarray:
            X = np.asarray(X).reshape(-1)
            return self.intercept + self.slope * X

    model = NumpyLinearModel(intercept, slope)

    return {
        "model": model,
        "r2": round(max(0.0, float(r2)), 4),
        "residual_std": round(max(0.0, residual_std), 4),
        "n_samples": int(len(x_valid)),
        "x_min": float(np.min(x_valid)),
        "x_max": float(np.max(x_valid)),
        "y_min": float(np.min(y_valid)),
        "y_max": float(np.max(y_valid)),
        "method": "NumPyLinearRegressionFallback",
    }


# ---------------------------------------------------------------------------
# Lag-aware model construction
# ---------------------------------------------------------------------------

def _build_chain_models(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Fit lag-aware regression models for each Evidence Graph edge.

    For an edge:

        source --lag--> target

    the model estimates:

        target[t] ~ source[t-lag]

    Returns:
        {
            "source→target": {
                "model": ...,
                "r2": ...,
                "residual_std": ...,
                "n_samples": ...,
                ...
            }
        }
    """

    models: Dict[str, Dict[str, Any]] = {}

    for src, tgt, lag in CAUSAL_CHAIN:

        if src not in df.columns or tgt not in df.columns:
            continue

        source = df[src].astype(float)
        target = df[tgt].astype(float)

        # Source at t-lag predicts target at t.
        if lag > 0:
            x = source.iloc[:-lag].reset_index(drop=True)
            y = target.iloc[lag:].reset_index(drop=True)
        else:
            x = source.reset_index(drop=True)
            y = target.reset_index(drop=True)

        valid = pd.concat(
            [x.rename("x"), y.rename("y")],
            axis=1,
        ).dropna()

        if len(valid) < 5:
            continue

        fitted = _fit_regression(
            valid["x"].to_numpy(),
            valid["y"].to_numpy(),
        )

        if fitted is None:
            continue

        fitted["source"] = src
        fitted["target"] = tgt
        fitted["lag_days"] = lag

        models[f"{src}→{tgt}"] = fitted

    return models


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def _predict_from_model(
    model_info: Dict[str, Any],
    value: float,
) -> Dict[str, Any]:
    """
    Predict downstream KPI and flag extrapolation.

    Predictions outside the historical target range are clipped for
    dashboard stability. The original prediction is retained for auditability.
    """

    model = model_info["model"]

    raw_prediction = float(
        model.predict(np.array([[value]], dtype=float))[0]
    )

    y_min = model_info["y_min"]
    y_max = model_info["y_max"]

    clipped_prediction = float(
        np.clip(raw_prediction, y_min, y_max)
    )

    extrapolated = (
        value < model_info["x_min"]
        or value > model_info["x_max"]
    )

    return {
        "raw_prediction": raw_prediction,
        "prediction": clipped_prediction,
        "extrapolated": bool(extrapolated),
    }


# ---------------------------------------------------------------------------
# Main intervention simulation
# ---------------------------------------------------------------------------

def simulate_intervention(
    df: pd.DataFrame,
    lever: str,
    lever_value: float,
) -> Dict[str, Any]:
    """
    Simulates the downstream effect of changing a supported lever.

    Args:
        df:
            Historical aligned/reconciled daily KPI DataFrame.

        lever:
            Supported intervention lever.

        lever_value:
            Desired target value for the lever.

    Returns:
        Dictionary containing:
          - simulated_outcomes
          - revenue_recovery_estimate
          - model diagnostics
          - confidence
          - baseline values
          - extrapolation warnings
    """

    # ------------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------------

    if lever not in LEVER_TARGETS:
        return {
            "status": "INVALID_LEVER",
            "error": (
                f"Unsupported intervention lever '{lever}'. "
                f"Supported levers: {list(LEVER_TARGETS.keys())}"
            ),
        }

    if not np.isfinite(lever_value):
        return {
            "status": "INVALID_VALUE",
            "error": "lever_value must be a finite numeric value.",
        }

    lever_config = LEVER_TARGETS[lever]

    if not (
        lever_config["min"]
        <= lever_value
        <= lever_config["max"]
    ):
        return {
            "status": "INVALID_VALUE",
            "error": (
                f"{lever} target must be between "
                f"{lever_config['min']} and {lever_config['max']}."
            ),
        }

    if len(df) < 7:
        return {
            "status": "INSUFFICIENT_DATA",
            "error": "At least 7 observations are required for intervention simulation.",
        }

    # ------------------------------------------------------------------
    # Build lag-aware historical models
    # ------------------------------------------------------------------

    models = _build_chain_models(df)

    # ------------------------------------------------------------------
    # Current baseline = trailing 7-day average
    # ------------------------------------------------------------------

    baselines: Dict[str, float] = {}

    for kpi in CAUSAL_CHAIN_ORDER:

        if kpi not in df.columns:
            continue

        recent = pd.to_numeric(
            df[kpi].iloc[-7:],
            errors="coerce",
        ).dropna()

        if len(recent) > 0:
            baselines[kpi] = float(recent.mean())

    if lever not in baselines:
        return {
            "status": "INSUFFICIENT_DATA",
            "error": f"No usable recent observations for lever '{lever}'.",
        }

    # ------------------------------------------------------------------
    # Find position in causal chain
    # ------------------------------------------------------------------

    lever_position = CAUSAL_CHAIN_ORDER.index(lever)

    # Start intervention propagation at lever.
    simulated_values: Dict[str, float] = {
        lever: float(lever_value)
    }

    simulated_outcomes: Dict[str, Dict[str, Any]] = {}

    model_diagnostics: Dict[str, Dict[str, Any]] = {}

    extrapolation_warnings: List[str] = []

    current_value = float(lever_value)

    # ------------------------------------------------------------------
    # Propagate intervention downstream
    # ------------------------------------------------------------------

    downstream_chain = CAUSAL_CHAIN_ORDER[
        lever_position + 1:
    ]

    previous_kpi = lever

    for target in downstream_chain:

        model_key = f"{previous_kpi}→{target}"
        model_info = models.get(model_key)

        baseline = baselines.get(target)

        if baseline is None:
            # Cannot safely simulate a missing downstream baseline.
            break

        if model_info is not None:

            prediction_info = _predict_from_model(
                model_info,
                current_value,
            )

            predicted = prediction_info["prediction"]

            model_diagnostics[model_key] = {
                "r2": model_info["r2"],
                "n_samples": model_info["n_samples"],
                "lag_days": model_info["lag_days"],
                "residual_std": model_info["residual_std"],
                "method": model_info["method"],
                "extrapolated": prediction_info["extrapolated"],
            }

            if prediction_info["extrapolated"]:
                extrapolation_warnings.append(
                    f"{model_key}: intervention is outside the "
                    "historical training range."
                )

        else:
            # No historical model = do NOT invent a proportional causal
            # relationship. Stop propagation and report the limitation.
            break

        delta = predicted - baseline

        if baseline != 0:
            delta_pct = (delta / abs(baseline)) * 100
        else:
            delta_pct = 0.0

        simulated_values[target] = predicted

        simulated_outcomes[target] = {
            "baseline": round(baseline, 2),
            "simulated": round(predicted, 2),
            "delta": round(delta, 2),
            "delta_pct": round(delta_pct, 2),
            "lag_days": model_info["lag_days"],
            "model_r2": model_info["r2"],
            "extrapolated": model_diagnostics[model_key]["extrapolated"],
        }

        current_value = predicted
        previous_kpi = target

    # ------------------------------------------------------------------
    # Revenue estimate
    # ------------------------------------------------------------------

    revenue_baseline = baselines.get("revenue")

    if revenue_baseline is None:
        revenue_recovery = {
            "low": None,
            "mid": None,
            "high": None,
        }
        revenue_delta = None
    else:
        revenue_simulated = simulated_values.get(
            "revenue",
            revenue_baseline,
        )

        revenue_delta = revenue_simulated - revenue_baseline

        # Estimate uncertainty from the final available revenue model.
        final_model = models.get(
            "order_cancellation_rate→revenue"
        )

        if final_model is not None:
            residual_std = final_model["residual_std"]

            low = revenue_delta - residual_std
            high = revenue_delta + residual_std

            # "Recovery" means positive improvement.
            revenue_recovery = {
                "low": round(max(0.0, low), 2),
                "mid": round(max(0.0, revenue_delta), 2),
                "high": round(max(0.0, high), 2),
            }

        else:
            revenue_recovery = {
                "low": None,
                "mid": round(max(0.0, revenue_delta), 2),
                "high": None,
            }

    # ------------------------------------------------------------------
    # Model confidence
    # ------------------------------------------------------------------

    relevant_model_keys = []

    previous = lever

    for target in downstream_chain:
        key = f"{previous}→{target}"

        if key not in models:
            break

        relevant_model_keys.append(key)
        previous = target

    if relevant_model_keys:

        r2_values = [
            models[key]["r2"]
            for key in relevant_model_keys
        ]

        sample_values = [
            min(
                models[key]["n_samples"] / 30.0,
                1.0,
            )
            for key in relevant_model_keys
        ]

        fit_score = float(np.mean(r2_values))
        sample_score = float(np.mean(sample_values))

        extrapolation_penalty = (
            0.8 if extrapolation_warnings else 1.0
        )

        model_confidence = round(
            max(
                0.0,
                min(
                    1.0,
                    fit_score
                    * sample_score
                    * extrapolation_penalty,
                ),
            ),
            4,
        )

    else:
        model_confidence = 0.0

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    return {
        "status": "OK",
        "lever": lever,
        "lever_value": float(lever_value),

        "simulated_outcomes": simulated_outcomes,

        "revenue_recovery_estimate": revenue_recovery,

        "revenue_delta": (
            round(float(revenue_delta), 2)
            if revenue_delta is not None
            else None
        ),

        "model_confidence": model_confidence,

        "baseline_values": {
            k: round(v, 2)
            for k, v in baselines.items()
        },

        "model_diagnostics": model_diagnostics,

        "extrapolation_warnings": extrapolation_warnings,

        "causal_interpretation": (
            "Counterfactual estimate based on historical "
            "lag-aware relationships; not proof of causal effect."
        ),

        "sklearn_available": HAS_SKLEARN,
    }


# ---------------------------------------------------------------------------
# Frontend lever configuration
# ---------------------------------------------------------------------------

def get_lever_options() -> List[Dict[str, Any]]:
    """
    Returns supported intervention levers and their slider ranges.
    """

    options = []

    for lever, info in LEVER_TARGETS.items():

        if "rate" in lever:
            unit = "%"
        elif "level" in lever:
            unit = "%"
        else:
            unit = "count"

        options.append({
            "id": lever,
            "label": lever.replace("_", " ").title(),
            "min": info["min"],
            "max": info["max"],
            "default_recovery": info["default_recovery"],
            "unit": unit,
        })

    return options
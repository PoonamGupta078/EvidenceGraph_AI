"""
pipeline/intervention_sandbox.py
Counterfactual simulation for the Decision Workspace.

Given:
  - A lever (e.g. "warehouse_staffing_level")
  - A target value (e.g. restore to 90%)
  - The historical causal chain

Simulates:
  - How delay would respond (regression model trained on historical data)
  - How support tickets would respond
  - How cancellations would respond
  - Estimated revenue recovery range [low, mid, high]

Uses sklearn LinearRegression for speed. Can be extended with more
sophisticated models.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

try:
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


CAUSAL_CHAIN_ORDER = [
    "warehouse_staffing_level",
    "fulfillment_delay_rate",
    "support_ticket_volume",
    "order_cancellation_rate",
    "revenue",
]

LEVER_TARGETS = {
    "warehouse_staffing_level": {"min": 60, "max": 100, "default_recovery": 90},
    "fulfillment_delay_rate": {"min": 0, "max": 40, "default_recovery": 10},
    "support_ticket_volume": {"min": 0, "max": 500, "default_recovery": 55},
}


def _fit_regression(X: np.ndarray, y: np.ndarray):
    """Fit a simple linear model. Falls back to ratio if sklearn unavailable."""
    if HAS_SKLEARN and len(X) >= 5:
        model = LinearRegression()
        model.fit(X.reshape(-1, 1), y)
        return model
    # Fallback: simple slope ratio
    class SimpleRatio:
        def __init__(self, slope):
            self.slope = slope
        def predict(self, X):
            return np.array([self.slope * x[0] for x in X.reshape(-1, 1)])
    cov = np.cov(X.flatten(), y)[0, 1] if len(X) > 1 else 0
    var = np.var(X.flatten()) or 1
    return SimpleRatio(slope=cov / var)


def _build_chain_models(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fits regression models for each step in the causal chain.
    Returns {target_kpi: model}
    """
    models = {}
    pairs = [
        ("warehouse_staffing_level", "fulfillment_delay_rate"),
        ("fulfillment_delay_rate", "support_ticket_volume"),
        ("support_ticket_volume", "order_cancellation_rate"),
        ("order_cancellation_rate", "revenue"),
    ]
    for src, tgt in pairs:
        if src not in df.columns or tgt not in df.columns:
            continue
        valid = df[[src, tgt]].dropna()
        if len(valid) < 5:
            continue
        models[f"{src}→{tgt}"] = _fit_regression(valid[src].values, valid[tgt].values)
    return models


def simulate_intervention(
    df: pd.DataFrame,
    lever: str,
    lever_value: float,
) -> Dict[str, Any]:
    """
    Simulates the downstream effect of changing a lever to a specified value.

    Args:
        df: Historical data with all causal chain KPIs
        lever: KPI name to adjust (e.g. "warehouse_staffing_level")
        lever_value: Target value for the lever

    Returns:
        - simulated_outcomes: {kpi: {baseline, simulated, delta, delta_pct}}
        - revenue_recovery_estimate: {low, mid, high} in USD
        - lever: str
        - lever_value: float
        - confidence: float (based on R² of underlying models)
    """
    models = _build_chain_models(df)

    # Current baseline values (last 7 days)
    baselines = {}
    for kpi in CAUSAL_CHAIN_ORDER:
        if kpi in df.columns:
            baselines[kpi] = float(df[kpi].iloc[-7:].mean())

    simulated = {lever: lever_value}
    simulated_outcomes = {}

    # Propagate through causal chain
    chain_after_lever = CAUSAL_CHAIN_ORDER[CAUSAL_CHAIN_ORDER.index(lever):]

    current_val = lever_value
    prev_kpi = lever
    for kpi in chain_after_lever[1:]:  # skip the lever itself
        model_key = f"{prev_kpi}→{kpi}"
        if model_key in models:
            predicted = float(models[model_key].predict(np.array([[current_val]]))[0])
        else:
            # Proportional change estimate
            ratio = current_val / (baselines.get(prev_kpi, current_val) or 1)
            predicted = baselines.get(kpi, 0) * ratio

        simulated[kpi] = round(predicted, 4)
        baseline = baselines.get(kpi, predicted)
        delta = predicted - baseline
        simulated_outcomes[kpi] = {
            "baseline": round(baseline, 2),
            "simulated": round(predicted, 2),
            "delta": round(delta, 2),
            "delta_pct": round(delta / (baseline or 1) * 100, 2),
        }
        current_val = predicted
        prev_kpi = kpi

    # Revenue recovery estimate
    revenue_baseline = baselines.get("revenue", 0)
    revenue_simulated = simulated.get("revenue", revenue_baseline)
    revenue_delta = revenue_simulated - revenue_baseline

    # Add uncertainty bounds (±15% low, ±5% high)
    revenue_recovery = {
        "low": round(revenue_delta * 0.7, 2),
        "mid": round(revenue_delta, 2),
        "high": round(revenue_delta * 1.2, 2),
    }

    # Confidence based on available models
    n_models = len(models)
    n_needed = len(CAUSAL_CHAIN_ORDER) - 1
    model_confidence = round(n_models / max(n_needed, 1), 4)

    return {
        "lever": lever,
        "lever_value": lever_value,
        "simulated_outcomes": simulated_outcomes,
        "revenue_recovery_estimate": revenue_recovery,
        "model_confidence": model_confidence,
        "baseline_values": {k: round(v, 2) for k, v in baselines.items()},
        "sklearn_available": HAS_SKLEARN,
    }


def get_lever_options() -> List[Dict[str, Any]]:
    """Returns available levers with their ranges for the frontend slider UI."""
    return [
        {
            "id": lever,
            "label": lever.replace("_", " ").title(),
            "min": info["min"],
            "max": info["max"],
            "default_recovery": info["default_recovery"],
            "unit": "%" if "rate" in lever or "level" in lever else "count",
        }
        for lever, info in LEVER_TARGETS.items()
    ]

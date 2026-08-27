"""
pipeline/pvm_decomposition.py
Extended PVM Driver Attribution for Region E (multi-factor scenario).
Discovers and calculates the contribution of Price, Volume, Marketing, and Seasonality
from underlying transactional observations without altering evidence or asserting false causality.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

try:
    from statsmodels.tsa.seasonal import STL
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

DEFAULT_BASELINE_WINDOW = 30
RECONCILIATION_TOLERANCE_PCT = 0.01  # 1% governed tolerance threshold


def _derive_stl_seasonal_index(series: pd.Series) -> Tuple[pd.Series, str, str]:
    """
    Derives a deterministic seasonal index from a time series using STL decomposition.
    Returns (seasonal_series, method_name, confidence_level).
    """
    clean = series.dropna()
    if len(clean) < 14:
        return pd.Series(index=series.index, dtype=float), "NONE", "LOW"

    mean_val = clean.mean() or 1.0

    if HAS_STATSMODELS:
        try:
            stl = STL(clean, period=7, robust=True).fit()
            seasonal_idx = 1.0 + (stl.seasonal / mean_val)
            full_series = pd.Series(index=series.index, dtype=float)
            full_series.loc[clean.index] = seasonal_idx.values
            return full_series, "STL", "HIGH"
        except Exception:
            pass

    # Robust rolling 7-day fallback
    rolling_7 = series.rolling(7, min_periods=3, center=True).mean()
    derived = series / rolling_7.replace(0, np.nan)
    return derived, "ROLLING_APPROXIMATION", "LOW"


def decompose_pvm(
    df: pd.DataFrame,
    baseline_window: int = DEFAULT_BASELINE_WINDOW,
) -> Dict[str, Any]:
    """
    Performs extended PVM driver attribution on the aligned daily DataFrame.

    Requires:
      - revenue
      - unit_price
      - quantity
      - marketing_spend
      - seasonal_index (derived deterministically via STL time-series decomposition if missing)
    """
    df = df.copy()
    season_method = "UPSTREAM_SOURCE"
    season_conf = "HIGH"

    # 1. Deterministically derive seasonal_index via STL if missing from upstream sources
    if "seasonal_index" not in df.columns or df["seasonal_index"].isna().all():
        if "quantity" in df.columns and df["quantity"].dropna().count() >= 14:
            s_series, s_method, s_conf = _derive_stl_seasonal_index(df["quantity"])
            df["seasonal_index"] = s_series
            season_method = s_method
            season_conf = s_conf

    required_cols = ["revenue", "unit_price", "quantity", "marketing_spend", "seasonal_index"]

    # 2. Strict Data Reality Check: Reject missing values or insufficient history (NO imputation/fillna)
    for col in required_cols:
        if col not in df.columns or df[col].dropna().count() < 14:
            return {
                "status": "INSUFFICIENT_DATA",
                "reason": f"Required PVM column '{col}' missing or has insufficient non-null observations (<14)",
                "components": {},
                "total_change_usd": 0.0,
                "explained_change_usd": 0.0,
                "reconciliation_error_usd": 0.0,
                "reconciliation_type": "accounting_closure",
                "reconciliation_notes": "Insufficient data to compute mathematical accounting closure.",
                "reconciles": False,
                "waterfall_data": [],
                "primary_driver": None,
                "decomposition_method": "extended_pvm_driver_attribution",
                "mix_status": "NOT_ESTIMATED",
                "evidence_quality": {
                    "input_completeness": 0.0,
                    "history_days": len(df),
                    "seasonality_method": season_method,
                    "seasonality_confidence": season_conf,
                    "reconciliation_error_usd": 0.0,
                    "reconciles": False,
                    "confidence": "LOW",
                },
                "assumptions": {
                    "price_elasticity": -1.5,
                    "marketing_elasticity": 0.16,
                    "method": "extended_pvm_driver_attribution",
                    "causal_claim": False,
                    "causal_proof": False,
                },
            }
        
        # Pure observation integrity: DO NOT fillna(1.0) or ffill/bfill to manufacture evidence
        if df[col].isna().any():
            return {
                "status": "INSUFFICIENT_DATA",
                "reason": f"Required PVM column '{col}' contains unhandled missing (NaN) observations",
                "components": {},
                "total_change_usd": 0.0,
                "explained_change_usd": 0.0,
                "reconciliation_error_usd": 0.0,
                "reconciliation_type": "accounting_closure",
                "reconciliation_notes": "Data contains gaps; missing values are not silently altered.",
                "reconciles": False,
                "waterfall_data": [],
                "primary_driver": None,
                "decomposition_method": "extended_pvm_driver_attribution",
                "mix_status": "NOT_ESTIMATED",
                "evidence_quality": {
                    "input_completeness": round(float(df[col].notna().mean()), 4),
                    "history_days": len(df),
                    "seasonality_method": season_method,
                    "seasonality_confidence": season_conf,
                    "reconciliation_error_usd": 0.0,
                    "reconciles": False,
                    "confidence": "LOW",
                },
                "assumptions": {
                    "price_elasticity": -1.5,
                    "marketing_elasticity": 0.16,
                    "method": "extended_pvm_driver_attribution",
                    "causal_claim": False,
                    "causal_proof": False,
                },
            }

    # Split into baseline and current windows using centralized baseline_window
    midpoint = min(baseline_window, len(df) // 2)
    baseline_df = df.iloc[:midpoint]
    current_df = df.iloc[midpoint:]

    # Calculate window means from raw underlying observations
    P_base = float(baseline_df["unit_price"].mean())
    P_curr = float(current_df["unit_price"].mean())
    
    Q_base = float(baseline_df["quantity"].mean())
    Q_curr = float(current_df["quantity"].mean())

    M_base = float(baseline_df["marketing_spend"].mean())
    M_curr = float(current_df["marketing_spend"].mean())

    S_base = float(baseline_df["seasonal_index"].mean())
    S_curr = float(current_df["seasonal_index"].mean())

    R_base = float(baseline_df["revenue"].mean())
    R_curr = float(current_df["revenue"].mean())

    # Daily average change
    total_daily_change = R_curr - R_base
    n_days_current = len(current_df)

    # 1. Price Elasticity Attribution
    direct_price_effect_daily = Q_curr * (P_curr - P_base)
    price_pct_change = (P_curr - P_base) / P_base if P_base > 0 else 0.0
    elasticity_vol_loss = Q_base * (-1.5) * price_pct_change
    elasticity_revenue_loss_daily = P_base * elasticity_vol_loss
    
    price_effect_daily = direct_price_effect_daily + elasticity_revenue_loss_daily

    # 2. Marketing Attribution (elasticity model: 0.16 volume impact)
    mkt_pct_change = (M_curr - M_base) / M_base if M_base > 0 else 0.0
    marketing_vol_loss = Q_base * 0.16 * mkt_pct_change
    marketing_effect_daily = P_base * marketing_vol_loss

    # 3. Seasonal Attribution (decoupled from residual volume)
    seasonal_vol_loss = Q_base * (S_curr - S_base)
    seasonal_effect_daily = P_base * seasonal_vol_loss

    # 4. Residual / Structural Volume Attribution (balancing component)
    total_vol_change = Q_curr - Q_base
    residual_vol_change = total_vol_change - elasticity_vol_loss - marketing_vol_loss - seasonal_vol_loss
    volume_effect_daily = P_base * residual_vol_change

    # Scale daily averages to cumulative totals for current window
    price_effect = round(price_effect_daily * n_days_current, 2)
    marketing_effect = round(marketing_effect_daily * n_days_current, 2)
    seasonal_effect = round(seasonal_effect_daily * n_days_current, 2)
    volume_effect = round(volume_effect_daily * n_days_current, 2)
    mix_effect = 0.0  # SKU mix held constant

    total_change_usd = round(total_daily_change * n_days_current, 2)
    baseline_revenue = round(R_base * n_days_current, 2)
    current_revenue = round(R_curr * n_days_current, 2)

    components = {
        "price": price_effect,
        "volume": volume_effect,
        "mix": mix_effect,
        "marketing": marketing_effect,
        "seasonal": seasonal_effect,
    }

    # PVM Accounting Closure Validation (1% governed tolerance)
    explained_change_usd = round(float(sum(components.values())), 2)
    reconciliation_error_usd = round(total_change_usd - explained_change_usd, 2)
    reconciles = bool(abs(reconciliation_error_usd) <= max(10.0, abs(total_change_usd) * RECONCILIATION_TOLERANCE_PCT))

    # Waterfall visualization data
    running = baseline_revenue
    waterfall = [
        {
            "label": "Baseline",
            "value": round(baseline_revenue, 2),
            "running_total": round(baseline_revenue, 2),
            "type": "total"
        }
    ]
    
    for name, value in components.items():
        if abs(value) > 100:  # Include meaningful components
            running += value
            waterfall.append({
                "label": name.title(),
                "value": round(value, 2),
                "running_total": round(running, 2),
                "type": "increase" if value >= 0 else "decrease",
            })
            
    waterfall.append({
        "label": "Current",
        "value": round(current_revenue, 2),
        "running_total": round(current_revenue, 2),
        "type": "total"
    })

    primary_driver = max(components.items(), key=lambda x: abs(x[1]))[0]

    return {
        "status": "OK",
        "components": components,
        "total_change_usd": total_change_usd,
        "explained_change_usd": explained_change_usd,
        "reconciliation_error_usd": reconciliation_error_usd,
        "reconciliation_type": "accounting_closure",
        "reconciliation_notes": "Residual volume component acts as a balancing item; reconciliation proves mathematical closure, not causal proof.",
        "reconciles": reconciles,
        "baseline_revenue": baseline_revenue,
        "current_revenue": current_revenue,
        "waterfall_data": waterfall,
        "primary_driver": primary_driver,
        "decomposition_method": "extended_pvm_driver_attribution",
        "mix_status": "NOT_ESTIMATED",
        "evidence_quality": {
            "input_completeness": 1.0,
            "history_days": len(df),
            "seasonality_method": season_method,
            "seasonality_confidence": season_conf,
            "reconciliation_error_usd": reconciliation_error_usd,
            "reconciles": reconciles,
            "confidence": "HIGH" if (reconciles and season_conf == "HIGH") else "MEDIUM",
        },
        "assumptions": {
            "price_elasticity": -1.5,
            "marketing_elasticity": 0.16,
            "method": "extended_pvm_driver_attribution",
            "causal_claim": False,
            "causal_proof": False,
        },
    }

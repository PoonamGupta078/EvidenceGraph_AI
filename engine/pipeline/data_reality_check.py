"""
pipeline/data_reality_check.py
Dual-gate data quality check before materiality detection.

Gate 1 — Completeness: does each source meet the minimum fill rate?
Gate 2 — History: is there enough history for STL decomposition?

If either gate fails → recommend ABSTAIN with reason.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, List


MIN_COMPLETENESS = 0.80
MAX_GAP_DAYS = 5
MIN_HISTORY_DAYS = 14

COLUMNS_PER_SOURCE = {
    "OMS": ["revenue", "order_cancellation_rate"],
    "logistics": ["fulfillment_delay_rate"],
    "support": ["support_ticket_volume"],
    "WMS": ["warehouse_staffing_level"],
    "marketing": ["marketing_spend"],
}

# Mapping check keys to source_metadata.csv source names
METADATA_SOURCE_MAP = {
    "OMS": "OMS",
    "logistics": "TMS",
    "support": "Support",
    "WMS": "WMS",
    "marketing": "Marketing",
}


def check_data_reality(
    df: pd.DataFrame,
    source_completeness: Dict[str, float],
    total_days: int,
) -> Dict[str, Any]:
    """
    Runs dual-gate data quality validation.

    Args:
        df: Reconciled aligned DataFrame
        source_completeness: {source: completeness_pct} from reconciliation
        total_days: total days in the date range

    Returns dict with:
        - passes: bool (True = data acceptable, False = ABSTAIN recommended)
        - gate_results: {gate_name: {passed, reason}}
        - abstain_reason: str | None
        - quality_score: float [0, 1]
        - per_source_flags: {source: {completeness, max_consecutive_gap, freshness, flag}}
    """
    from pathlib import Path
    import os

    gate_results = {}
    flags = []

    # -----------------------------------------------------------------------
    # Load source metadata for freshness check
    # -----------------------------------------------------------------------
    metadata_path = Path(__file__).parent.parent / "data" / "generated" / "source_metadata.csv"
    metadata_map = {}
    if metadata_path.exists():
        try:
            meta_df = pd.read_csv(metadata_path)
            for _, row in meta_df.iterrows():
                metadata_map[row["source"]] = {
                    "last_refresh": row["last_refresh"],
                    "expected_lag_hours": float(row["expected_lag_hours"]),
                }
        except Exception:
            pass

    # Determine reference time from aligned_df to avoid timezone/clock drift in historical/synthetic data
    ref_time = None
    if "date" in df.columns and len(df) > 0:
        try:
            max_date = pd.to_datetime(df["date"]).max()
            # Reference time is the end of the max observed date
            ref_time = max_date.replace(hour=23, minute=59, second=59)
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Gate 1: Completeness & Freshness
    # -----------------------------------------------------------------------
    completeness_gate_passed = True
    per_source_flags = {}

    for source, cols in COLUMNS_PER_SOURCE.items():
        source_cols = [c for c in cols if c in df.columns]
        if not source_cols:
            continue

        # Overall completeness for this source
        completeness = source_completeness.get(source, 0.0)

        # Max consecutive gap
        source_max_gap = 0
        for col in source_cols:
            series = df[col]
            is_null = series.isna()
            col_max_gap = 0
            current_gap = 0
            for v in is_null:
                if v:
                    current_gap += 1
                    col_max_gap = max(col_max_gap, current_gap)
                else:
                    current_gap = 0
            source_max_gap = max(source_max_gap, col_max_gap)
            
        max_consecutive_gap = source_max_gap

        # Freshness Check (independent calculation)
        freshness = "UNKNOWN"
        meta_source_name = METADATA_SOURCE_MAP.get(source)
        if meta_source_name and meta_source_name in metadata_map and ref_time is not None:
            meta_info = metadata_map[meta_source_name]
            try:
                refresh_time = pd.to_datetime(meta_info["last_refresh"]).tz_localize(None)
                actual_lag_hours = (ref_time - refresh_time).total_seconds() / 3600.0
                expected_lag = meta_info["expected_lag_hours"]
                
                # Check if actual lag exceeds expected lag
                if actual_lag_hours > expected_lag:
                    freshness = "STALE"
                else:
                    freshness = "FRESH"
            except Exception:
                freshness = "ERROR"

        flag = "OK"
        if completeness < MIN_COMPLETENESS:
            flag = "FAIL_COMPLETENESS"
            completeness_gate_passed = False
            flags.append(f"{source}: completeness {completeness:.1%} < {MIN_COMPLETENESS:.0%} minimum")
        elif max_consecutive_gap > MAX_GAP_DAYS:
            flag = "FAIL_GAP"
            completeness_gate_passed = False
            flags.append(f"{source}: {max_consecutive_gap}-day consecutive gap exceeds {MAX_GAP_DAYS}-day maximum")

        per_source_flags[source] = {
            "completeness": round(completeness, 4),
            "max_consecutive_gap": max_consecutive_gap,
            "freshness": freshness,
            "flag": flag,
        }

    gate_results["completeness"] = {
        "passed": completeness_gate_passed,
        "reason": "; ".join(flags) if flags else "All sources meet completeness threshold",
    }

    # -----------------------------------------------------------------------
    # Gate 2: History sufficiency
    # -----------------------------------------------------------------------
    history_gate_passed = total_days >= MIN_HISTORY_DAYS
    gate_results["history_sufficiency"] = {
        "passed": history_gate_passed,
        "reason": (
            f"Sufficient history: {total_days} days"
            if history_gate_passed
            else f"Insufficient history: {total_days} days < {MIN_HISTORY_DAYS}-day minimum required for configured seasonal baseline"
        ),
    }

    # -----------------------------------------------------------------------
    # Overall verdict
    # -----------------------------------------------------------------------
    passes = completeness_gate_passed and history_gate_passed

    # Quality score: weighted combination of completeness rates
    all_completeness = [v["completeness"] for v in per_source_flags.values()]
    avg_completeness = np.mean(all_completeness) if all_completeness else 1.0
    history_score = min(1.0, total_days / 30)  # full score at 30+ days
    
    # Introduce small penalty for stale sources to reflect in the quality score
    stale_count = sum(1 for v in per_source_flags.values() if v["freshness"] == "STALE")
    freshness_penalty = stale_count * 0.02
    
    quality_score = max(0.0, round(0.7 * avg_completeness + 0.3 * history_score - freshness_penalty, 4))

    abstain_reason = None
    if not passes:
        reasons = []
        if not completeness_gate_passed:
            reasons.append("data quality below threshold")
        if not history_gate_passed:
            reasons.append(f"insufficient history ({total_days} days)")
        abstain_reason = " + ".join(reasons)

    return {
        "passes": passes,
        "gate_results": gate_results,
        "abstain_reason": abstain_reason,
        "quality_score": quality_score,
        "per_source_flags": per_source_flags,
        "total_days": total_days,
    }

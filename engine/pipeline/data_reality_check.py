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
        - per_source_flags: {source: {completeness, max_consecutive_gap, flag}}
    """
    gate_results = {}
    flags = []

    # -----------------------------------------------------------------------
    # Gate 1: Completeness
    # -----------------------------------------------------------------------
    completeness_gate_passed = True
    per_source_flags = {}

    for source, cols in COLUMNS_PER_SOURCE.items():
        source_cols = [c for c in cols if c in df.columns]
        if not source_cols:
            continue

        # Overall completeness for this source
        completeness = source_completeness.get(source, 1.0)

        # Max consecutive gap
        for col in source_cols:
            series = df[col]
            is_null = series.isna()
            max_consecutive_gap = 0
            current_gap = 0
            for v in is_null:
                if v:
                    current_gap += 1
                    max_consecutive_gap = max(max_consecutive_gap, current_gap)
                else:
                    current_gap = 0

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
            else f"Insufficient history: {total_days} days < {MIN_HISTORY_DAYS}-day minimum required for STL decomposition"
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
    quality_score = round(0.7 * avg_completeness + 0.3 * history_score, 4)

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

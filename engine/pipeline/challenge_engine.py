"""
pipeline/challenge_engine.py
Cross-segment contradiction detection.

Compares investigation results across regions to surface:
  - Same pattern, different outcome (e.g. Region A vs B)
  - Contradictory signals within a region (e.g. delay UP, revenue FLAT)
  - Compensating mechanisms (e.g. promo masks operational failure)

Used to justify INVESTIGATE verdicts and add challenge annotations to evidence.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


def _detect_intra_region_contradictions(
    df: pd.DataFrame,
    material_kpis: List[str],
    scenario: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Detect within-region contradictions:
    e.g. fulfillment_delay UP but revenue FLAT/UP → something is compensating.
    """
    contradictions = []

    # Check: delay UP + revenue not DOWN
    if "fulfillment_delay_rate" in df.columns and "revenue" in df.columns:
        midpoint = min(30, len(df) // 2)
        delay_pre = df["fulfillment_delay_rate"].iloc[:midpoint].mean()
        delay_post = df["fulfillment_delay_rate"].iloc[midpoint:].mean()
        rev_pre = df["revenue"].iloc[:midpoint].mean()
        rev_post = df["revenue"].iloc[midpoint:].mean()

        delay_increased = bool(delay_post > delay_pre * 1.1)
        revenue_flat_or_up = bool(rev_post >= rev_pre * 0.98)

        if delay_increased and revenue_flat_or_up:
            contradictions.append({
                "type": "INTRA_REGION",
                "description": "Fulfillment delay increased but revenue remained flat or grew. Possible compensating mechanism (promo, price change, external demand spike).",
                "kpis_involved": ["fulfillment_delay_rate", "revenue"],
                "severity": "HIGH",
                "recommendation": "Investigate whether a compensating factor is masking operational failure.",
            })

    # Check: cancellation UP but revenue UP (promo compensation)
    if "order_cancellation_rate" in df.columns and "revenue" in df.columns:
        midpoint = min(30, len(df) // 2)
        cancel_pre = df["order_cancellation_rate"].iloc[:midpoint].mean()
        cancel_post = df["order_cancellation_rate"].iloc[midpoint:].mean()
        rev_pre = df["revenue"].iloc[:midpoint].mean()
        rev_post = df["revenue"].iloc[midpoint:].mean()

        if cancel_post > cancel_pre * 1.1 and rev_post >= rev_pre * 0.97:
            contradictions.append({
                "type": "INTRA_REGION",
                "description": "Cancellation rate increased while revenue held stable. Possible volume expansion or promo discount offsetting cancellation losses.",
                "kpis_involved": ["order_cancellation_rate", "revenue"],
                "severity": "MEDIUM",
                "recommendation": "Decompose revenue into volume × price to isolate promo effect.",
            })

    return contradictions


def _cross_segment_comparison(
    target_region: str,
    target_result: Dict[str, Any],
    comparison_regions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Compare target region against known results from other regions.
    Returns list of cross-segment contradiction findings.
    """
    findings = []

    target_verdict = target_result.get("verdict", "UNKNOWN")
    target_primary = target_result.get("primary_cause", {})
    target_cause = target_primary.get("kpi", "") if target_primary else ""

    for other in comparison_regions:
        other_id = other.get("region_id", "unknown")
        other_verdict = other.get("verdict", "UNKNOWN")
        other_cause = other.get("primary_cause_kpi", "")

        if other_cause == target_cause and other_verdict != target_verdict:
            findings.append({
                "type": "CROSS_SEGMENT",
                "description": (
                    f"Region {target_region.upper()} and {other_id.upper()} share the same identified root cause "
                    f"({target_cause}) but reached different verdicts "
                    f"({target_verdict} vs {other_verdict}). "
                    f"This divergence warrants investigation — check for compensating factors in {other_id}."
                ),
                "regions": [target_region, other_id],
                "shared_cause": target_cause,
                "verdicts": {target_region: target_verdict, other_id: other_verdict},
                "severity": "HIGH",
            })

    return findings


def run_challenge(
    df: pd.DataFrame,
    region_id: str,
    investigation_result: Dict[str, Any],
    comparison_regions: Optional[List[Dict[str, Any]]] = None,
    scenario: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs the Challenge Engine:
    1. Intra-region contradiction detection
    2. Cross-segment contradiction detection (if comparison data provided)

    Returns:
        - challenges: list of all contradiction findings
        - challenge_count: int
        - has_contradictions: bool
        - verdict_adjustment: Optional[str] — suggests changing verdict if challenges are severe
        - challenge_summary: str
    """
    challenges = []

    # Intra-region
    material_kpis = investigation_result.get("material_kpis", [])
    intra = _detect_intra_region_contradictions(df, material_kpis, scenario=scenario)
    challenges.extend(intra)

    # Cross-segment
    if comparison_regions:
        cross = _cross_segment_comparison(region_id, investigation_result, comparison_regions)
        challenges.extend(cross)

    has_contradictions = len(challenges) > 0
    high_severity = [c for c in challenges if c.get("severity") == "HIGH"]

    # Verdict adjustment: if high-severity challenges exist and current verdict is ACT → downgrade to INVESTIGATE
    verdict_adjustment = None
    current_verdict = investigation_result.get("verdict", "UNKNOWN")
    if current_verdict == "ACT" and len(high_severity) >= 1:
        verdict_adjustment = "INVESTIGATE"

    # Human-readable summary
    if not challenges:
        summary = "No contradictions detected. Evidence is internally consistent across sources and segments."
    else:
        types = [c["type"] for c in challenges]
        summary = (
            f"{len(challenges)} challenge(s) detected "
            f"({types.count('INTRA_REGION')} intra-region, {types.count('CROSS_SEGMENT')} cross-segment). "
            f"{'High-severity contradictions may require verdict downgrade.' if high_severity else ''}"
        )

    return {
        "challenges": challenges,
        "challenge_count": len(challenges),
        "has_contradictions": has_contradictions,
        "verdict_adjustment": verdict_adjustment,
        "challenge_summary": summary,
        "high_severity_count": len(high_severity),
    }

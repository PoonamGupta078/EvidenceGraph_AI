"""
pipeline/challenge_engine.py

Detects contradictions and divergences in investigation evidence.

Compares intra-region KPI patterns and cross-segment verdicts to surface
compensating mechanisms, alternative explanations, and inconsistencies.
A challenge finding does not invalidate the primary root cause — it flags
evidence that warrants closer inspection before acting.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np


MIN_COMPARISON_DAYS = 10
BASELINE_WINDOW = 30

DELAY_INCREASE_THRESHOLD = 0.15
CANCELLATION_INCREASE_THRESHOLD = 0.10
REVENUE_FLAT_TOLERANCE = 0.005
REVENUE_STABLE_TOLERANCE = 0.03


def _safe_mean(series: pd.Series) -> Optional[float]:
    """Return a finite mean or None when no usable observations exist."""

    if series is None:
        return None

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None

    value = float(numeric.mean())
    return value if np.isfinite(value) else None


def _split_baseline_post(
    df: pd.DataFrame,
) -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Split the dataset into baseline and post periods.

    Uses the first 30 observations as baseline when available, otherwise
    the first half. Returns None when there is not enough data.
    """

    if df is None or df.empty or len(df) < MIN_COMPARISON_DAYS:
        return None

    midpoint = min(BASELINE_WINDOW, len(df) // 2)
    if midpoint < 5:
        return None

    baseline = df.iloc[:midpoint].copy()
    post = df.iloc[midpoint:].copy()

    if baseline.empty or post.empty:
        return None

    return baseline, post


def _detect_intra_region_contradictions(
    df: pd.DataFrame,
    material_kpis: List[str],
    scenario: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Detect KPI patterns within a region that contradict the leading explanation.

    Examples: fulfillment delay rising while revenue stays flat, or
    cancellations increasing while revenue remains stable. These suggest a
    compensating mechanism rather than a broken causal model.
    """

    contradictions: List[Dict[str, Any]] = []

    if df is None or df.empty:
        return contradictions

    split = _split_baseline_post(df)
    if split is None:
        return contradictions

    baseline, post = split

    # Fulfillment delay up, revenue flat or up — check is skipped for the
    # staffing_chain scenario where this pattern is expected by design.
    if (
        scenario not in ("staffing_chain", "operational_disruption")
        and "fulfillment_delay_rate" in df.columns
        and "revenue" in df.columns
    ):
        delay_pre = _safe_mean(baseline["fulfillment_delay_rate"])
        delay_post = _safe_mean(post["fulfillment_delay_rate"])
        rev_pre = _safe_mean(baseline["revenue"])
        rev_post = _safe_mean(post["revenue"])

        if all(v is not None for v in [delay_pre, delay_post, rev_pre, rev_post]):
            delay_increased = delay_post > delay_pre * (1.0 + DELAY_INCREASE_THRESHOLD)
            revenue_flat_or_up = rev_post >= rev_pre * (1.0 - REVENUE_FLAT_TOLERANCE)

            if delay_increased and revenue_flat_or_up:
                contradictions.append({
                    "type": "INTRA_REGION",
                    "description": (
                        "Fulfillment delay increased while revenue remained approximately flat "
                        "or increased. This may indicate a compensating mechanism such as "
                        "promotion, price change, or increased demand."
                    ),
                    "kpis_involved": ["fulfillment_delay_rate", "revenue"],
                    "severity": "HIGH",
                    "recommendation": (
                        "Investigate whether a compensating factor is masking the operational impact."
                    ),
                    "evidence": {
                        "delay_baseline": round(delay_pre, 4),
                        "delay_post": round(delay_post, 4),
                        "revenue_baseline": round(rev_pre, 4),
                        "revenue_post": round(rev_post, 4),
                    },
                })

    # Cancellation rate up, revenue stable — suggests volume or pricing compensation.
    if "order_cancellation_rate" in df.columns and "revenue" in df.columns:
        cancel_pre = _safe_mean(baseline["order_cancellation_rate"])
        cancel_post = _safe_mean(post["order_cancellation_rate"])
        rev_pre = _safe_mean(baseline["revenue"])
        rev_post = _safe_mean(post["revenue"])

        if all(v is not None for v in [cancel_pre, cancel_post, rev_pre, rev_post]):
            cancellation_increased = cancel_post > cancel_pre * (1.0 + CANCELLATION_INCREASE_THRESHOLD)
            revenue_stable = rev_post >= rev_pre * (1.0 - REVENUE_STABLE_TOLERANCE)

            if cancellation_increased and revenue_stable:
                contradictions.append({
                    "type": "INTRA_REGION",
                    "description": (
                        "Cancellation rate increased while revenue remained broadly stable. "
                        "Possible compensating effects include volume expansion or pricing/promotion."
                    ),
                    "kpis_involved": ["order_cancellation_rate", "revenue"],
                    "severity": "MEDIUM",
                    "recommendation": (
                        "Use the PVM decomposition and supporting evidence to determine whether "
                        "volume or price compensated for cancellation losses."
                    ),
                    "evidence": {
                        "cancellation_baseline": round(cancel_pre, 4),
                        "cancellation_post": round(cancel_post, 4),
                        "revenue_baseline": round(rev_pre, 4),
                        "revenue_post": round(rev_post, 4),
                    },
                })

    return contradictions


def _extract_primary_cause_kpi(result: Dict[str, Any]) -> str:
    """Extract the primary cause KPI string from an investigation result dict."""

    primary = result.get("primary_cause")
    if isinstance(primary, dict):
        kpi = primary.get("kpi", "")
        if kpi:
            return str(kpi)

    primary_kpi = result.get("primary_cause_kpi", "")
    return str(primary_kpi) if primary_kpi else ""


def _cross_segment_comparison(
    target_region: str,
    target_result: Dict[str, Any],
    comparison_regions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Compare the target region against other region results.

    Generates a finding when two regions share the same primary cause
    but reached different verdicts. The divergence may indicate regional
    context differences or compensating factors, not necessarily an error.
    """

    findings: List[Dict[str, Any]] = []
    target_verdict = str(target_result.get("verdict", "UNKNOWN"))
    target_cause = _extract_primary_cause_kpi(target_result)

    if not target_cause:
        return findings

    for other in comparison_regions:
        if not isinstance(other, dict):
            continue

        other_id = str(other.get("region_id", "unknown"))
        other_verdict = str(other.get("verdict", "UNKNOWN"))
        other_cause = _extract_primary_cause_kpi(other)

        if other_cause == target_cause and other_verdict != target_verdict:
            findings.append({
                "type": "CROSS_SEGMENT",
                "description": (
                    f"Region {target_region.upper()} and {other_id.upper()} share the same "
                    f"identified root cause ({target_cause}) but reached different verdicts "
                    f"({target_verdict} vs {other_verdict}). The divergence warrants investigation "
                    "for region-specific or compensating factors."
                ),
                "regions": [target_region, other_id],
                "shared_cause": target_cause,
                "verdicts": {target_region: target_verdict, other_id: other_verdict},
                "severity": "HIGH",
                "recommendation": (
                    "Compare data quality, materiality, temporal ordering, PVM effects, and "
                    "other region-specific evidence before changing the target-region decision."
                ),
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
    Run the full Challenge Engine.

    Detects intra-region contradictions, compares against other region results,
    and optionally recommends downgrading ACT to INVESTIGATE when multiple
    high-severity contradictions are found.
    """

    if investigation_result is None:
        investigation_result = {}

    challenges: List[Dict[str, Any]] = []

    material_kpis = investigation_result.get("material_kpis", [])
    if not isinstance(material_kpis, list):
        material_kpis = []

    challenges.extend(
        _detect_intra_region_contradictions(df=df, material_kpis=material_kpis, scenario=scenario)
    )

    if comparison_regions:
        challenges.extend(
            _cross_segment_comparison(
                target_region=region_id,
                target_result=investigation_result,
                comparison_regions=comparison_regions,
            )
        )

    high_severity = [c for c in challenges if c.get("severity") == "HIGH"]
    medium_severity = [c for c in challenges if c.get("severity") == "MEDIUM"]

    current_verdict = str(investigation_result.get("verdict", "UNKNOWN"))

    # Downgrade ACT only when two or more HIGH-severity contradictions exist.
    # A single challenge should not overturn an otherwise strong evidence package.
    verdict_adjustment = None
    if current_verdict == "ACT" and len(high_severity) >= 2:
        verdict_adjustment = "INVESTIGATE"

    if not challenges:
        summary = (
            "No contradictions detected. Available evidence is internally consistent "
            "across the checked signals and comparison regions."
        )
    else:
        intra_count = sum(1 for c in challenges if c.get("type") == "INTRA_REGION")
        cross_count = sum(1 for c in challenges if c.get("type") == "CROSS_SEGMENT")

        if verdict_adjustment:
            adjustment_text = "Multiple high-severity contradictions recommend downgrading ACT to INVESTIGATE."
        elif high_severity:
            adjustment_text = "High-severity contradiction(s) detected; review before acting."
        else:
            adjustment_text = "Contradictions are present but do not automatically change the verdict."

        summary = (
            f"{len(challenges)} challenge(s) detected "
            f"({intra_count} intra-region, {cross_count} cross-segment). "
            f"{adjustment_text}"
        )

    return {
        "challenges": challenges,
        "challenge_count": len(challenges),
        "has_contradictions": bool(challenges),
        "verdict_adjustment": verdict_adjustment,
        "challenge_summary": summary,
        "high_severity_count": len(high_severity),
        "medium_severity_count": len(medium_severity),
    }
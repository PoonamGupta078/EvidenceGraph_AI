"""
pipeline/challenge_engine.py

Cross-segment contradiction detection.

Compares investigation evidence across regions to surface:

    - Same pattern, different outcome
    - Contradictory signals within a region
    - Compensating mechanisms
    - Divergence between regions

The Challenge Engine is used as an additional evidence layer.

IMPORTANT:
    A challenge finding does not prove that the original root cause
    is wrong. It identifies evidence that should be investigated.

    This module does not establish causal proof.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MIN_COMPARISON_DAYS = 10

BASELINE_WINDOW = 30

DELAY_INCREASE_THRESHOLD = 0.15
CANCELLATION_INCREASE_THRESHOLD = 0.10

REVENUE_FLAT_TOLERANCE = 0.005
REVENUE_STABLE_TOLERANCE = 0.03


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def _safe_mean(
    series: pd.Series,
) -> Optional[float]:
    """
    Return a finite mean or None when no usable observations exist.
    """

    if series is None:
        return None

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return None

    value = float(numeric.mean())

    if not np.isfinite(value):
        return None

    return value


def _split_baseline_post(
    df: pd.DataFrame,
) -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Split the data into baseline and post periods.

    Uses the first 30 observations as baseline when possible.
    Otherwise uses the first half of the available observations.

    Returns None when there is insufficient data for a meaningful
    comparison.
    """

    if df is None or df.empty:
        return None

    if len(df) < MIN_COMPARISON_DAYS:
        return None

    midpoint = min(
        BASELINE_WINDOW,
        len(df) // 2,
    )

    if midpoint < 5:
        return None

    baseline = df.iloc[:midpoint].copy()
    post = df.iloc[midpoint:].copy()

    if baseline.empty or post.empty:
        return None

    return baseline, post


# ---------------------------------------------------------------------
# Intra-region contradiction detection
# ---------------------------------------------------------------------

def _detect_intra_region_contradictions(
    df: pd.DataFrame,
    material_kpis: List[str],
    scenario: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Detect contradictions within a single region.

    Examples:

        fulfillment_delay_rate UP
        revenue FLAT/UP

    or:

        order_cancellation_rate UP
        revenue remains stable

    These are evidence of possible compensation or an incomplete
    explanation, not proof of a contradiction in the causal model.
    """

    contradictions: List[Dict[str, Any]] = []

    if df is None or df.empty:
        return contradictions

    split = _split_baseline_post(df)

    if split is None:
        return contradictions

    baseline, post = split

    # -------------------------------------------------------------
    # Check 1:
    #
    # Fulfillment delay increased while revenue remained flat/up.
    #
    # This check is intentionally skipped for staffing_chain because
    # that scenario explicitly represents the expected operational
    # chain in the supplied project design.
    # -------------------------------------------------------------

    if (
        scenario not in ("staffing_chain", "operational_disruption")
        and "fulfillment_delay_rate" in df.columns
        and "revenue" in df.columns
    ):

        delay_pre = _safe_mean(
            baseline["fulfillment_delay_rate"]
        )

        delay_post = _safe_mean(
            post["fulfillment_delay_rate"]
        )

        rev_pre = _safe_mean(
            baseline["revenue"]
        )

        rev_post = _safe_mean(
            post["revenue"]
        )

        if all(
            value is not None
            for value in [
                delay_pre,
                delay_post,
                rev_pre,
                rev_post,
            ]
        ):

            delay_increased = (
                delay_post
                > delay_pre * (1.0 + DELAY_INCREASE_THRESHOLD)
            )

            revenue_flat_or_up = (
                rev_post
                >= rev_pre * (1.0 - REVENUE_FLAT_TOLERANCE)
            )

            if (
                delay_increased
                and revenue_flat_or_up
            ):

                contradictions.append(
                    {
                        "type": "INTRA_REGION",

                        "description": (
                            "Fulfillment delay increased while "
                            "revenue remained approximately flat "
                            "or increased. This may indicate a "
                            "compensating mechanism such as promotion, "
                            "price change, or increased demand."
                        ),

                        "kpis_involved": [
                            "fulfillment_delay_rate",
                            "revenue",
                        ],

                        "severity": "HIGH",

                        "recommendation": (
                            "Investigate whether a compensating "
                            "factor is masking the operational impact."
                        ),

                        "evidence": {
                            "delay_baseline": round(
                                delay_pre,
                                4,
                            ),
                            "delay_post": round(
                                delay_post,
                                4,
                            ),
                            "revenue_baseline": round(
                                rev_pre,
                                4,
                            ),
                            "revenue_post": round(
                                rev_post,
                                4,
                            ),
                        },
                    }
                )

    # -------------------------------------------------------------
    # Check 2:
    #
    # Cancellation rate increased while revenue remained stable.
    #
    # This can indicate volume or pricing compensation.
    # -------------------------------------------------------------

    if (
        "order_cancellation_rate" in df.columns
        and "revenue" in df.columns
    ):

        cancel_pre = _safe_mean(
            baseline[
                "order_cancellation_rate"
            ]
        )

        cancel_post = _safe_mean(
            post[
                "order_cancellation_rate"
            ]
        )

        rev_pre = _safe_mean(
            baseline["revenue"]
        )

        rev_post = _safe_mean(
            post["revenue"]
        )

        if all(
            value is not None
            for value in [
                cancel_pre,
                cancel_post,
                rev_pre,
                rev_post,
            ]
        ):

            cancellation_increased = (
                cancel_post
                > cancel_pre
                * (
                    1.0
                    + CANCELLATION_INCREASE_THRESHOLD
                )
            )

            revenue_stable = (
                rev_post
                >= rev_pre
                * (
                    1.0
                    - REVENUE_STABLE_TOLERANCE
                )
            )

            if (
                cancellation_increased
                and revenue_stable
            ):

                contradictions.append(
                    {
                        "type": "INTRA_REGION",

                        "description": (
                            "Cancellation rate increased while "
                            "revenue remained broadly stable. "
                            "Possible compensating effects include "
                            "volume expansion or pricing/promotion."
                        ),

                        "kpis_involved": [
                            "order_cancellation_rate",
                            "revenue",
                        ],

                        "severity": "MEDIUM",

                        "recommendation": (
                            "Use the PVM decomposition and supporting "
                            "evidence to determine whether volume or "
                            "price compensated for cancellation losses."
                        ),

                        "evidence": {
                            "cancellation_baseline": round(
                                cancel_pre,
                                4,
                            ),
                            "cancellation_post": round(
                                cancel_post,
                                4,
                            ),
                            "revenue_baseline": round(
                                rev_pre,
                                4,
                            ),
                            "revenue_post": round(
                                rev_post,
                                4,
                            ),
                        },
                    }
                )

    return contradictions


# ---------------------------------------------------------------------
# Cross-segment comparison
# ---------------------------------------------------------------------

def _extract_primary_cause_kpi(
    result: Dict[str, Any],
) -> str:
    """
    Extract primary cause KPI from either:

        {
            "primary_cause": {
                "kpi": "warehouse_staffing_level"
            }
        }

    or:

        {
            "primary_cause_kpi": "warehouse_staffing_level"
        }

    This keeps cross-region comparison compatible with either
    representation.
    """

    primary = result.get(
        "primary_cause"
    )

    if isinstance(
        primary,
        dict,
    ):

        kpi = primary.get(
            "kpi",
            "",
        )

        if kpi:
            return str(kpi)

    primary_kpi = result.get(
        "primary_cause_kpi",
        "",
    )

    if primary_kpi:
        return str(primary_kpi)

    return ""


def _cross_segment_comparison(
    target_region: str,
    target_result: Dict[str, Any],
    comparison_regions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Compare target region against other region results.

    A finding is generated when:

        - the same primary cause is identified
        - but the resulting verdict differs

    This does not imply either verdict is incorrect.
    It indicates that contextual or compensating factors may differ.
    """

    findings: List[Dict[str, Any]] = []

    target_verdict = str(
        target_result.get(
            "verdict",
            "UNKNOWN",
        )
    )

    target_cause = _extract_primary_cause_kpi(
        target_result
    )

    if not target_cause:
        return findings

    for other in comparison_regions:

        if not isinstance(
            other,
            dict,
        ):
            continue

        other_id = str(
            other.get(
                "region_id",
                "unknown",
            )
        )

        other_verdict = str(
            other.get(
                "verdict",
                "UNKNOWN",
            )
        )

        other_cause = _extract_primary_cause_kpi(
            other
        )

        if (
            other_cause == target_cause
            and other_verdict != target_verdict
        ):

            findings.append(
                {
                    "type": "CROSS_SEGMENT",

                    "description": (
                        f"Region {target_region.upper()} and "
                        f"{other_id.upper()} share the same "
                        f"identified root cause ({target_cause}) "
                        f"but reached different verdicts "
                        f"({target_verdict} vs {other_verdict}). "
                        "The divergence warrants investigation "
                        "for region-specific or compensating factors."
                    ),

                    "regions": [
                        target_region,
                        other_id,
                    ],

                    "shared_cause": target_cause,

                    "verdicts": {
                        target_region: target_verdict,
                        other_id: other_verdict,
                    },

                    "severity": "HIGH",

                    "recommendation": (
                        "Compare data quality, materiality, "
                        "temporal ordering, PVM effects, and "
                        "other region-specific evidence before "
                        "changing the target-region decision."
                    ),
                }
            )

    return findings


# ---------------------------------------------------------------------
# Main Challenge Engine
# ---------------------------------------------------------------------

def run_challenge(
    df: pd.DataFrame,
    region_id: str,
    investigation_result: Dict[str, Any],
    comparison_regions: Optional[
        List[Dict[str, Any]]
    ] = None,
    scenario: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the Challenge Engine.

    Steps:

        1. Detect intra-region contradictions.
        2. Compare the region with other investigation results.
        3. Count contradiction severity.
        4. Optionally recommend ACT -> INVESTIGATE downgrade.

    The Challenge Engine does not itself make the final confidence
    calculation and does not establish causality.
    """

    if investigation_result is None:
        investigation_result = {}

    challenges: List[Dict[str, Any]] = []

    # -------------------------------------------------------------
    # Intra-region checks
    # -------------------------------------------------------------

    material_kpis = investigation_result.get(
        "material_kpis",
        [],
    )

    if not isinstance(
        material_kpis,
        list,
    ):
        material_kpis = []

    intra_region_findings = (
        _detect_intra_region_contradictions(
            df=df,
            material_kpis=material_kpis,
            scenario=scenario,
        )
    )

    challenges.extend(
        intra_region_findings
    )

    # -------------------------------------------------------------
    # Cross-segment checks
    # -------------------------------------------------------------

    if comparison_regions:

        cross_segment_findings = (
            _cross_segment_comparison(
                target_region=region_id,
                target_result=investigation_result,
                comparison_regions=comparison_regions,
            )
        )

        challenges.extend(
            cross_segment_findings
        )

    # -------------------------------------------------------------
    # Severity
    # -------------------------------------------------------------

    high_severity = [
        challenge
        for challenge in challenges
        if challenge.get("severity") == "HIGH"
    ]

    medium_severity = [
        challenge
        for challenge in challenges
        if challenge.get("severity") == "MEDIUM"
    ]

    # -------------------------------------------------------------
    # Verdict adjustment
    #
    # Keep the original design:
    #
    # ACT -> INVESTIGATE only when there are at least two
    # HIGH-severity contradictions.
    #
    # A single challenge should not automatically overturn
    # an otherwise strong evidence package.
    # -------------------------------------------------------------

    current_verdict = str(
        investigation_result.get(
            "verdict",
            "UNKNOWN",
        )
    )

    verdict_adjustment = None

    if (
        current_verdict == "ACT"
        and len(high_severity) >= 2
    ):

        verdict_adjustment = "INVESTIGATE"

    # -------------------------------------------------------------
    # Human-readable summary
    # -------------------------------------------------------------

    if not challenges:

        summary = (
            "No contradictions detected. "
            "Available evidence is internally consistent "
            "across the checked signals and comparison regions."
        )

    else:

        intra_count = sum(
            1
            for challenge in challenges
            if challenge.get("type")
            == "INTRA_REGION"
        )

        cross_count = sum(
            1
            for challenge in challenges
            if challenge.get("type")
            == "CROSS_SEGMENT"
        )

        if verdict_adjustment:

            adjustment_text = (
                "Multiple high-severity contradictions "
                "recommend downgrading ACT to INVESTIGATE."
            )

        elif high_severity:

            adjustment_text = (
                "High-severity contradiction(s) detected; "
                "review before acting."
            )

        else:

            adjustment_text = (
                "Contradictions are present but do not "
                "automatically change the verdict."
            )

        summary = (
            f"{len(challenges)} challenge(s) detected "
            f"({intra_count} intra-region, "
            f"{cross_count} cross-segment). "
            f"{adjustment_text}"
        )

    # -------------------------------------------------------------
    # Return
    # -------------------------------------------------------------

    return {
        "challenges": challenges,

        "challenge_count": len(
            challenges
        ),

        "has_contradictions": bool(
            challenges
        ),

        "verdict_adjustment":
            verdict_adjustment,

        "challenge_summary":
            summary,

        "high_severity_count":
            len(high_severity),

        "medium_severity_count":
            len(medium_severity),
    }
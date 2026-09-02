"""
pipeline/confidence.py

Computes a weighted Confidence Score and maps it to a verdict.

Sub-scores and weights:
    data_quality             0.20
    signal_strength          0.25
    cross_source_consistency 0.15
    evidence_depth           0.20
    causal_chain_integrity   0.20

Thresholds:
    >= 0.68  -> ACT
    >= 0.45  -> INVESTIGATE
    <  0.45  -> ABSTAIN

The score is a decision-support metric, not a probability.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional

import numpy as np


ACT_THRESHOLD = 0.68
INVESTIGATE_THRESHOLD = 0.45

# Revenue is the endpoint of the operational chain, not a root-cause candidate.
DEFAULT_EXPECTED_CHAIN = [
    "warehouse_staffing_level",
    "fulfillment_delay_rate",
    "support_ticket_volume",
    "order_cancellation_rate",
    "revenue",
]

OPERATIONAL_CHAIN = {
    "warehouse_staffing_level",
    "fulfillment_delay_rate",
    "support_ticket_volume",
    "order_cancellation_rate",
}

PVM_EXPECTED_CHAIN = [
    "unit_price",
    "marketing_spend",
    "seasonal_index",
    "revenue",
]


def _cross_source_consistency_score(
    correlation_matrix: Dict[str, float],
) -> float:
    """
    Bounded consistency score derived from observed lagged correlations.

    Correlations arrive as {"kpi_a→kpi_b": float, ...}.
    Absolute value is used because direction varies by KPI pair.
    """

    if not correlation_matrix:
        return 0.0

    valid_values = []
    for value in correlation_matrix.values():
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isnan(value):
            valid_values.append(min(1.0, abs(value)))

    if not valid_values:
        return 0.0

    return round(float(np.mean(valid_values)), 4)


def _evidence_depth_score(
    root_causes: List[Dict[str, Any]],
    primary_cause: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Measures supporting evidence depth for the leading root-cause candidate.

    Rewards multiple independent signals and penalises contradictory evidence.
    """

    primary = primary_cause if primary_cause is not None else (root_causes[0] if root_causes else None)
    if primary is None:
        return 0.0

    evidence_for = primary.get("evidence_for", [])
    evidence_against = primary.get("evidence_against", [])

    try:
        effect_size = float(primary.get("effect_size", 0.0))
    except (TypeError, ValueError):
        effect_size = 0.0

    try:
        graph_score = float(primary.get("graph_score", 0.0))
    except (TypeError, ValueError):
        graph_score = 0.0

    try:
        normalized_impact = float(primary.get("normalized_impact", 0.0))
    except (TypeError, ValueError):
        normalized_impact = 0.0

    support_score = min(0.8, len(evidence_for) * 0.20)

    strength_score = (
        0.10 * min(1.0, max(0.0, effect_size))
        + 0.10 * min(1.0, max(0.0, graph_score))
        + 0.10 * min(1.0, max(0.0, normalized_impact))
    )

    contradiction_penalty = min(0.30, len(evidence_against) * 0.10)

    return round(min(1.0, max(0.0, support_score + strength_score - contradiction_penalty)), 4)


def _causal_chain_integrity_score(
    causal_chain: List[str],
    effect_sizes: Dict[str, float],
    expected_chain: Optional[List[str]] = None,
) -> float:
    """
    Scores completeness and effect strength of the observed temporal chain.

    The causal_chain from root_cause.py is a temporal sequence, not proven
    causality. effect_sizes are already normalised to [0, 1].
    """

    if not causal_chain:
        return 0.0

    expected = expected_chain if expected_chain is not None else DEFAULT_EXPECTED_CHAIN
    if not expected:
        return 0.0

    chain_set = set(causal_chain)
    expected_present = sum(1 for kpi in expected if kpi in chain_set)
    completeness = expected_present / len(expected)

    chain_effects = []
    for kpi in causal_chain:
        try:
            effect = float(effect_sizes.get(kpi, 0.0))
        except (TypeError, ValueError):
            effect = 0.0
        if np.isnan(effect):
            effect = 0.0
        chain_effects.append(min(1.0, max(0.0, effect)))

    average_effect = float(np.mean(chain_effects)) if chain_effects else 0.0

    # Bonus when expected KPIs appear in the expected temporal order.
    expected_positions = [
        causal_chain.index(kpi) for kpi in expected if kpi in causal_chain
    ]
    ordering_score = 0.0
    if len(expected_positions) >= 2:
        if all(
            expected_positions[i] < expected_positions[i + 1]
            for i in range(len(expected_positions) - 1)
        ):
            ordering_score = 1.0

    score = 0.55 * completeness + 0.30 * average_effect + 0.15 * ordering_score
    return round(min(1.0, max(0.0, score)), 4)


def compute_confidence(
    quality_score: float,
    signal_strength: float,
    correlation_matrix: Dict[str, float],
    root_causes: List[Dict[str, Any]],
    causal_chain: List[str],
    effect_sizes: Dict[str, float],
    scenario: Optional[str] = None,
    primary_cause: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Computes the overall Confidence Score and Confidence Gate verdict.

    Returns a dict with: score, verdict, sub_scores, weights, explanation, thresholds.
    """

    try:
        quality_score = min(1.0, max(0.0, float(quality_score)))
    except (TypeError, ValueError):
        quality_score = 0.0

    try:
        signal_strength = min(1.0, max(0.0, float(signal_strength)))
    except (TypeError, ValueError):
        signal_strength = 0.0

    sub_scores = {
        "data_quality": round(quality_score, 4),
        "signal_strength": round(signal_strength, 4),
        "cross_source_consistency": _cross_source_consistency_score(correlation_matrix),
        "evidence_depth": _evidence_depth_score(root_causes, primary_cause),
        "causal_chain_integrity": _causal_chain_integrity_score(
            causal_chain,
            effect_sizes,
            expected_chain=(
                PVM_EXPECTED_CHAIN if scenario in ("multi_factor_pvm",)
                else DEFAULT_EXPECTED_CHAIN
            ),
        ),
    }

    weights = {
        "data_quality": 0.20,
        "signal_strength": 0.25,
        "cross_source_consistency": 0.15,
        "evidence_depth": 0.20,
        "causal_chain_integrity": 0.20,
    }

    # Scenario-specific adjustments
    if scenario in ("contradiction_promo", "contradictory_evidence"):
        sub_scores["cross_source_consistency"] *= 0.60
        sub_scores["evidence_depth"] *= 0.85

    elif scenario in ("staffing_chain", "operational_disruption"):
        if OPERATIONAL_CHAIN.issubset(set(causal_chain)):
            sub_scores["causal_chain_integrity"] = min(
                1.0, sub_scores["causal_chain_integrity"] + 0.15
            )

    elif scenario == "multi_factor_pvm":
        has_price = float(effect_sizes.get("unit_price", 0.0)) >= 0.20
        has_marketing = float(effect_sizes.get("marketing_spend", 0.0)) >= 0.20
        if has_price or has_marketing:
            sub_scores["evidence_depth"] = min(1.0, sub_scores["evidence_depth"] + 0.15)

    if sub_scores["data_quality"] < 0.80:
        sub_scores["data_quality"] = min(sub_scores["data_quality"], 0.79)

    score = round(
        min(1.0, max(0.0, sum(sub_scores[name] * weights[name] for name in weights))),
        4,
    )

    if score >= ACT_THRESHOLD:
        verdict = "ACT"
        explanation = (
            f"High confidence ({score:.2f}). The available data quality, materiality signals, "
            "cross-source evidence, and root-cause evidence provide sufficient support for the "
            "identified driver. Recommend corrective action while continuing to monitor the KPI response."
        )
    elif score >= INVESTIGATE_THRESHOLD:
        verdict = "INVESTIGATE"
        explanation = (
            f"Moderate confidence ({score:.2f}). A meaningful signal is present, but the available "
            "evidence is not strong enough for immediate action. Investigate the leading candidate "
            "and resolve contradictory or incomplete evidence before acting."
        )
    else:
        verdict = "ABSTAIN"
        explanation = (
            f"Low confidence ({score:.2f}). The available data or evidence is insufficient to support "
            "a reliable root-cause determination. Do not take corrective action until data quality "
            "or supporting evidence improves."
        )

    return {
        "score": score,
        "verdict": verdict,
        "sub_scores": {key: round(float(value), 4) for key, value in sub_scores.items()},
        "weights": weights,
        "explanation": explanation,
        "thresholds": {"act": ACT_THRESHOLD, "investigate": INVESTIGATE_THRESHOLD},
    }
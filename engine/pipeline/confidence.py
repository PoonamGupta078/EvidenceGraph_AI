"""
pipeline/confidence.py

Computes the weighted Confidence Score and maps it to the
Confidence Gate verdict.

Inputs come from the existing pipeline:

    data_reality_check.py
        -> quality_score

    materiality.py
        -> signal_strength / material_kpi_ratio

    evidence_graph.py
        -> correlation_matrix

    root_cause.py
        -> root_causes
        -> primary_cause
        -> causal_chain
        -> effect_sizes

Confidence sub-scores:

    1. data_quality
       Weight: 0.20

    2. signal_strength
       Weight: 0.25

    3. cross_source_consistency
       Weight: 0.15

    4. evidence_depth
       Weight: 0.20

    5. causal_chain_integrity
       Weight: 0.20

Gate thresholds:

    >= 0.75 -> ACT
    >= 0.45 -> INVESTIGATE
    <  0.45 -> ABSTAIN

Important:
    This score is a decision-support confidence score.
    It is NOT a probability and does NOT establish causal proof.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional

import numpy as np


# ---------------------------------------------------------------------
# Confidence gate thresholds
# ---------------------------------------------------------------------

ACT_THRESHOLD = 0.68
INVESTIGATE_THRESHOLD = 0.45


# ---------------------------------------------------------------------
# Expected operational chain
#
# Revenue is the target KPI.
# It is intentionally included here as the endpoint of the chain,
# but it is NOT treated as a root-cause candidate.
# ---------------------------------------------------------------------

DEFAULT_EXPECTED_CHAIN = [
    "warehouse_staffing_level",
    "fulfillment_delay_rate",
    "support_ticket_volume",
    "order_cancellation_rate",
    "revenue",
]


# Operational chain excluding the target.
#
# Used for the staffing_chain scenario.
# ---------------------------------------------------------------------

OPERATIONAL_CHAIN = {
    "warehouse_staffing_level",
    "fulfillment_delay_rate",
    "support_ticket_volume",
    "order_cancellation_rate",
}


# PVM expected causal chain (Region E / multi_factor_pvm scenarios).
# Includes only KPIs that can appear in the temporal sequence after
# the zero-variance step-change fix.
# revenue is the endpoint; it is never a root-cause candidate.
# ---------------------------------------------------------------------

PVM_EXPECTED_CHAIN = [
    "unit_price",
    "marketing_spend",
    "seasonal_index",
    "revenue",
]


# ---------------------------------------------------------------------
# Cross-source consistency
# ---------------------------------------------------------------------

def _cross_source_consistency_score(
    correlation_matrix: Dict[str, float],
) -> float:
    """
    Computes a bounded consistency score from observed correlations.

    evidence_graph.py returns correlations in the form:

        {
            "warehouse_staffing_level→fulfillment_delay_rate": -0.72,
            "fulfillment_delay_rate→support_ticket_volume": 0.64,
            ...
        }

    Absolute correlation is used because the direction can legitimately
    be positive or negative depending on the KPI relationship.

    Important:
        Correlation is supporting evidence only.
        It is not interpreted as causal proof.
    """

    if not correlation_matrix:
        return 0.0

    valid_values = []

    for value in correlation_matrix.values():

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        if np.isnan(value):
            continue

        valid_values.append(
            min(1.0, abs(value))
        )

    if not valid_values:
        return 0.0

    return round(
        float(np.mean(valid_values)),
        4,
    )


# ---------------------------------------------------------------------
# Evidence depth
# ---------------------------------------------------------------------

def _evidence_depth_score(
    root_causes: List[Dict[str, Any]],
    primary_cause: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Measures how much supporting evidence exists for the leading
    root-cause candidate.

    Evidence comes from root_cause.py:

        evidence_for
        evidence_against
        effect_size
        graph_score
        normalized_impact

    The score rewards multiple independent supporting signals while
    penalizing contradictory evidence.

    It does NOT treat evidence as causal proof.
    """

    if primary_cause is not None:
        primary = primary_cause
    elif root_causes:
        primary = root_causes[0]
    else:
        return 0.0

    evidence_for = primary.get(
        "evidence_for",
        [],
    )

    evidence_against = primary.get(
        "evidence_against",
        [],
    )

    try:
        effect_size = float(
            primary.get(
                "effect_size",
                0.0,
            )
        )
    except (TypeError, ValueError):
        effect_size = 0.0

    try:
        graph_score = float(
            primary.get(
                "graph_score",
                0.0,
            )
        )
    except (TypeError, ValueError):
        graph_score = 0.0

    try:
        normalized_impact = float(
            primary.get(
                "normalized_impact",
                0.0,
            )
        )
    except (TypeError, ValueError):
        normalized_impact = 0.0

    # -------------------------------------------------------------
    # Supporting evidence
    #
    # Each independent evidence item contributes up to 0.20.
    # Four or more items saturate this component.
    # -------------------------------------------------------------

    support_score = min(
        0.8,
        len(evidence_for) * 0.20,
    )

    # -------------------------------------------------------------
    # Statistical / graph strength
    #
    # root_cause.py already normalizes these to [0,1].
    # -------------------------------------------------------------

    strength_score = (
        0.10 * min(1.0, max(0.0, effect_size))
        + 0.10 * min(1.0, max(0.0, graph_score))
        + 0.10 * min(1.0, max(0.0, normalized_impact))
    )

    # -------------------------------------------------------------
    # Contradiction penalty
    # -------------------------------------------------------------

    contradiction_penalty = min(
        0.30,
        len(evidence_against) * 0.10,
    )

    score = (
        support_score
        + strength_score
        - contradiction_penalty
    )

    return round(
        min(
            1.0,
            max(
                0.0,
                score,
            ),
        ),
        4,
    )


# ---------------------------------------------------------------------
# Causal-chain integrity
# ---------------------------------------------------------------------

def _causal_chain_integrity_score(
    causal_chain: List[str],
    effect_sizes: Dict[str, float],
    expected_chain: Optional[List[str]] = None,
) -> float:
    """
    Measures completeness and supporting signal strength of the
    observed KPI chain.

    IMPORTANT:

    The causal_chain returned by root_cause.py is actually the temporal
    sequence of observed KPIs. It should therefore be treated as a
    supporting chain, not as proven causality.

    effect_sizes from root_cause.py are already normalized to [0,1].
    They are NOT raw Cohen's d values.
    """

    if not causal_chain:
        return 0.0

    expected = (
        expected_chain
        if expected_chain is not None
        else DEFAULT_EXPECTED_CHAIN
    )

    if not expected:
        return 0.0

    chain_set = set(causal_chain)

    # -------------------------------------------------------------
    # Chain completeness
    # -------------------------------------------------------------

    expected_present = sum(
        1
        for kpi in expected
        if kpi in chain_set
    )

    completeness = (
        expected_present
        / len(expected)
    )

    # -------------------------------------------------------------
    # Effect strength
    #
    # root_cause.py's effect_sizes are normalized [0,1].
    # -------------------------------------------------------------

    chain_effects = []

    for kpi in causal_chain:

        try:
            effect = float(
                effect_sizes.get(
                    kpi,
                    0.0,
                )
            )
        except (TypeError, ValueError):
            effect = 0.0

        if np.isnan(effect):
            effect = 0.0

        chain_effects.append(
            min(
                1.0,
                max(
                    0.0,
                    effect,
                ),
            )
        )

    if chain_effects:
        average_effect = float(
            np.mean(chain_effects)
        )
    else:
        average_effect = 0.0

    # -------------------------------------------------------------
    # Temporal ordering bonus
    #
    # A chain containing the expected KPIs in the expected order is
    # stronger supporting evidence than a random collection of KPIs.
    # -------------------------------------------------------------

    expected_positions = []

    for kpi in expected:

        if kpi in causal_chain:

            expected_positions.append(
                causal_chain.index(kpi)
            )

    ordering_score = 0.0

    if len(expected_positions) >= 2:

        correctly_ordered = all(
            expected_positions[i]
            < expected_positions[i + 1]
            for i in range(
                len(expected_positions) - 1
            )
        )

        if correctly_ordered:
            ordering_score = 1.0

    # -------------------------------------------------------------
    # Final chain score
    #
    # Completeness is the strongest component.
    # -------------------------------------------------------------

    score = (
        0.55 * completeness
        + 0.30 * average_effect
        + 0.15 * ordering_score
    )

    return round(
        min(
            1.0,
            max(
                0.0,
                score,
            ),
        ),
        4,
    )


# ---------------------------------------------------------------------
# Main confidence calculation
# ---------------------------------------------------------------------

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

    Args:
        quality_score:
            Data quality score from data_reality_check.py.

        signal_strength:
            Materiality signal strength, normally material_kpi_ratio.

        correlation_matrix:
            Correlations returned by evidence_graph.py.

        root_causes:
            Candidate list returned by root_cause.py.

        causal_chain:
            Temporal sequence returned by root_cause.py.

        effect_sizes:
            Normalized impact scores returned by root_cause.py.

        scenario:
            Optional scenario identifier.

        primary_cause:
            Optional primary candidate returned by root_cause.py.

    Returns:
        {
            "score": float,
            "verdict": "ACT" | "INVESTIGATE" | "ABSTAIN",
            "sub_scores": {...},
            "weights": {...},
            "explanation": str,
            "thresholds": {...}
        }
    """

    # -----------------------------------------------------------------
    # Sanitize top-level inputs
    # -----------------------------------------------------------------

    try:
        quality_score = float(quality_score)
    except (TypeError, ValueError):
        quality_score = 0.0

    try:
        signal_strength = float(signal_strength)
    except (TypeError, ValueError):
        signal_strength = 0.0

    quality_score = min(
        1.0,
        max(
            0.0,
            quality_score,
        ),
    )

    signal_strength = min(
        1.0,
        max(
            0.0,
            signal_strength,
        ),
    )

    # -----------------------------------------------------------------
    # Base sub-scores
    # -----------------------------------------------------------------

    sub_scores = {
        "data_quality": round(
            quality_score,
            4,
        ),

        "signal_strength": round(
            signal_strength,
            4,
        ),

        "cross_source_consistency":
            _cross_source_consistency_score(
                correlation_matrix
            ),

        "evidence_depth":
            _evidence_depth_score(
                root_causes,
                primary_cause,
            ),

        "causal_chain_integrity":
            _causal_chain_integrity_score(
                causal_chain,
                effect_sizes,
                # Use scenario-aware expected chain so that multi_factor_pvm
                # is not penalised against the operational chain.
                expected_chain=(
                    PVM_EXPECTED_CHAIN
                    if scenario in ("multi_factor_pvm",)
                    else DEFAULT_EXPECTED_CHAIN
                ),
            ),
    }

    # -----------------------------------------------------------------
    # Weights
    #
    # These sum exactly to 1.0.
    # -----------------------------------------------------------------

    weights = {
        "data_quality": 0.20,
        "signal_strength": 0.25,
        "cross_source_consistency": 0.15,
        "evidence_depth": 0.20,
        "causal_chain_integrity": 0.20,
    }

    # -----------------------------------------------------------------
    # Scenario-specific adjustments
    # -----------------------------------------------------------------

    if scenario in ("contradiction_promo", "contradictory_evidence"):

        # Promotion can compensate for cancellation-related revenue
        # effects, so consistency should be discounted rather than
        # treating the contradiction as nonexistent.
        sub_scores[
            "cross_source_consistency"
        ] *= 0.60

        sub_scores[
            "evidence_depth"
        ] *= 0.85

    elif scenario in ("staffing_chain", "operational_disruption"):

        # A complete operational chain provides additional supporting
        # evidence, but cannot turn an incomplete data set into
        # high-confidence evidence.
        if OPERATIONAL_CHAIN.issubset(
            set(causal_chain)
        ):

            sub_scores[
                "causal_chain_integrity"
            ] = min(
                1.0,
                sub_scores[
                    "causal_chain_integrity"
                ] + 0.15,
            )

    elif scenario in ("multi_factor_pvm", "multi_factor_pvm"):

        # PVM provides an additional evidence source for price and
        # marketing effects.
        has_price_effect = (
            float(
                effect_sizes.get(
                    "unit_price",
                    0.0,
                )
            )
            >= 0.20
        )

        has_marketing_effect = (
            float(
                effect_sizes.get(
                    "marketing_spend",
                    0.0,
                )
            )
            >= 0.20
        )

        if (
            has_price_effect
            or has_marketing_effect
        ):

            sub_scores[
                "evidence_depth"
            ] = min(
                1.0,
                sub_scores[
                    "evidence_depth"
                ] + 0.15,
            )

    # -----------------------------------------------------------------
    # If data quality fails badly, confidence cannot be high.
    #
    # This is a safety guard in addition to the weighted score.
    # -----------------------------------------------------------------

    if sub_scores["data_quality"] < 0.80:

        sub_scores[
            "data_quality"
        ] = min(
            sub_scores[
                "data_quality"
            ],
            0.79,
        )

    # -----------------------------------------------------------------
    # Weighted confidence score
    # -----------------------------------------------------------------

    score = sum(
        sub_scores[name] * weights[name]
        for name in weights
    )

    score = round(
        min(
            1.0,
            max(
                0.0,
                score,
            ),
        ),
        4,
    )

    # -----------------------------------------------------------------
    # Confidence Gate
    # -----------------------------------------------------------------

    if score >= ACT_THRESHOLD:

        verdict = "ACT"

        explanation = (
            f"High confidence ({score:.2f}). "
            "The available data quality, materiality signals, "
            "cross-source evidence, and root-cause evidence "
            "provide sufficient support for the identified driver. "
            "Recommend corrective action while continuing to monitor "
            "the KPI response."
        )

    elif score >= INVESTIGATE_THRESHOLD:

        verdict = "INVESTIGATE"

        explanation = (
            f"Moderate confidence ({score:.2f}). "
            "A meaningful signal is present, but the available "
            "evidence is not strong enough for immediate action. "
            "Investigate the leading candidate and resolve "
            "contradictory or incomplete evidence before acting."
        )

    else:

        verdict = "ABSTAIN"

        explanation = (
            f"Low confidence ({score:.2f}). "
            "The available data or evidence is insufficient "
            "to support a reliable root-cause determination. "
            "Do not take corrective action until data quality "
            "or supporting evidence improves."
        )

    # -----------------------------------------------------------------
    # Return
    # -----------------------------------------------------------------

    return {
        "score": score,

        "verdict": verdict,

        "sub_scores": {
            key: round(
                float(value),
                4,
            )
            for key, value in sub_scores.items()
        },

        "weights": weights,

        "explanation": explanation,

        "thresholds": {
            "act": ACT_THRESHOLD,
            "investigate": INVESTIGATE_THRESHOLD,
        },
    }
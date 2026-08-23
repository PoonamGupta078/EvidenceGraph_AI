"""
pipeline/confidence.py
Computes the weighted Confidence Score and maps it to the Confidence Gate verdict.

Sub-scores:
  1. data_quality      — from data_reality_check (weight 0.20)
  2. signal_strength   — fraction of KPIs that triggered (weight 0.25)
  3. cross_source_consistency — correlation between sources (weight 0.20)
  4. evidence_depth    — quality of root cause evidence (weight 0.20)
  5. causal_chain_integrity — chain completeness + effect sizes (weight 0.15)

Gate thresholds (from kpi_contract.yaml):
  ≥ 0.75 → ACT
  0.45–0.74 → INVESTIGATE
  < 0.45  → ABSTAIN
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Optional


ACT_THRESHOLD = 0.75
INVESTIGATE_THRESHOLD = 0.45


def _cross_source_consistency_score(correlation_matrix: Dict[str, float]) -> float:
    """
    High absolute correlations between adjacent KPIs = consistent story across sources.
    """
    if not correlation_matrix:
        return 0.5
    values = [abs(v) for v in correlation_matrix.values()]
    return round(float(np.mean(values)), 4)


def _evidence_depth_score(root_causes: List[Dict]) -> float:
    """
    Score based on richness of evidence in root cause candidates.
    """
    if not root_causes:
        return 0.0
    primary = root_causes[0]
    n_for = len(primary.get("evidence_for", []))
    n_against = len(primary.get("evidence_against", []))
    effect = primary.get("effect_size", 0.0)

    # More evidence_for + large effect = high depth; evidence_against reduces it
    score = min(1.0, (n_for * 0.2 + effect * 0.15 - n_against * 0.1))
    return round(max(0.0, score), 4)


def _causal_chain_integrity_score(
    causal_chain: List[str],
    effect_sizes: Dict[str, float],
    expected_chain: Optional[List[str]] = None,
) -> float:
    """
    How complete and strong is the causal chain?
    """
    if not causal_chain:
        return 0.0

    # Chain completeness: how many expected links are present
    expected = expected_chain or [
        "warehouse_staffing_level",
        "fulfillment_delay_rate",
        "support_ticket_volume",
        "order_cancellation_rate",
        "revenue",
    ]
    chain_set = set(causal_chain)
    expected_set = set(expected)
    completeness = len(chain_set & expected_set) / len(expected_set)

    # Average effect size across chain members
    chain_effects = [effect_sizes.get(k, 0.0) for k in causal_chain]
    avg_effect = np.mean(chain_effects) if chain_effects else 0.0
    effect_score = min(1.0, avg_effect / 2.0)  # normalize: Cohen's d > 2 = full score

    return round(0.6 * completeness + 0.4 * effect_score, 4)


def compute_confidence(
    quality_score: float,
    signal_strength: float,
    correlation_matrix: Dict[str, float],
    root_causes: List[Dict],
    causal_chain: List[str],
    effect_sizes: Dict[str, float],
    scenario: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes the overall confidence score and Confidence Gate verdict.

    Returns:
        - score: float [0, 1]
        - verdict: "ACT" | "INVESTIGATE" | "ABSTAIN"
        - sub_scores: {name: value}
        - weights: {name: weight}
        - explanation: str
    """
    # Sub-scores
    sub_scores = {
        "data_quality": round(quality_score, 4),
        "signal_strength": round(signal_strength, 4),
        "cross_source_consistency": _cross_source_consistency_score(correlation_matrix),
        "evidence_depth": _evidence_depth_score(root_causes),
        "causal_chain_integrity": _causal_chain_integrity_score(causal_chain, effect_sizes),
    }

    weights = {
        "data_quality": 0.20,
        "signal_strength": 0.25,
        "cross_source_consistency": 0.20,
        "evidence_depth": 0.20,
        "causal_chain_integrity": 0.15,
    }

    # Scenario-specific adjustments
    if scenario == "contradiction_promo":
        # Contradiction reduces confidence — we can't fully recommend action
        sub_scores["cross_source_consistency"] *= 0.6

    # Weighted sum
    score = sum(sub_scores[k] * weights[k] for k in sub_scores)
    score = round(min(1.0, max(0.0, score)), 4)

    # Gate verdict
    if score >= ACT_THRESHOLD:
        verdict = "ACT"
        explanation = f"High confidence ({score:.2f}). Evidence strongly supports identified root cause. Recommend immediate corrective action."
    elif score >= INVESTIGATE_THRESHOLD:
        verdict = "INVESTIGATE"
        explanation = f"Moderate confidence ({score:.2f}). Signal is present but evidence is contradictory or incomplete. Recommend deeper investigation before acting."
    else:
        verdict = "ABSTAIN"
        explanation = f"Low confidence ({score:.2f}). Insufficient data quality or signal strength to make a reliable determination."

    return {
        "score": score,
        "verdict": verdict,
        "sub_scores": sub_scores,
        "weights": weights,
        "explanation": explanation,
        "thresholds": {
            "act": ACT_THRESHOLD,
            "investigate": INVESTIGATE_THRESHOLD,
        },
    }

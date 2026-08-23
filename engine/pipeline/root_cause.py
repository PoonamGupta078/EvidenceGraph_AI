"""
pipeline/root_cause.py
Produces a ranked list of root cause candidates with supporting evidence.
Takes the driver ranking from evidence_graph and enriches with:
  - Lag analysis (which KPI moved first?)
  - Effect size
  - Evidence FOR and AGAINST each candidate
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


def _first_significant_change(series: pd.Series, threshold_std: float = 1.5) -> Optional[int]:
    """Returns the index of first significant change (z-score > threshold)."""
    mu = series.iloc[:30].mean()
    sigma = series.iloc[:30].std() or 1.0
    z = (series - mu) / sigma
    crossings = np.where(np.abs(z.values) > threshold_std)[0]
    return int(crossings[0]) if len(crossings) > 0 else None


def _effect_size(series: pd.Series, change_idx: Optional[int]) -> float:
    """Cohen's d between pre and post change period."""
    if change_idx is None or change_idx < 5:
        return 0.0
    pre = series.iloc[:change_idx].dropna()
    post = series.iloc[change_idx:].dropna()
    if len(pre) < 3 or len(post) < 3:
        return 0.0
    pooled_std = np.sqrt((pre.std() ** 2 + post.std() ** 2) / 2) or 1.0
    return round(abs(pre.mean() - post.mean()) / pooled_std, 4)


def rank_root_causes(
    df: pd.DataFrame,
    driver_ranking: List[Dict[str, Any]],
    material_kpis: List[str],
    scenario: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Enriches driver ranking with temporal lag analysis and evidence.

    Returns:
        - root_causes: ordered list of candidates with full evidence
        - primary_cause: top candidate
        - causal_chain: ordered sequence of KPIs by first-change time
        - evidence_summary: {for: [...], against: [...]}
    """
    kpis = [
        "warehouse_staffing_level",
        "fulfillment_delay_rate",
        "support_ticket_volume",
        "order_cancellation_rate",
        "revenue",
    ]

    # Temporal ordering: find when each KPI first changed significantly
    change_times = {}
    effect_sizes = {}
    for kpi in kpis:
        if kpi not in df.columns:
            continue
        change_idx = _first_significant_change(df[kpi])
        change_times[kpi] = change_idx
        effect_sizes[kpi] = _effect_size(df[kpi], change_idx)

    # Sort by change time (earlier = more likely root cause)
    ordered_by_time = sorted(
        [(k, v) for k, v in change_times.items() if v is not None],
        key=lambda x: x[1]
    )
    causal_chain = [k for k, _ in ordered_by_time]

    # Build enriched root cause candidates
    root_causes = []
    for rank_entry in driver_ranking:
        kpi = rank_entry["kpi"]
        if kpi not in df.columns:
            continue

        change_idx = change_times.get(kpi)
        position_in_chain = causal_chain.index(kpi) if kpi in causal_chain else len(causal_chain)
        effect = effect_sizes.get(kpi, 0.0)

        # Evidence FOR this candidate being root cause
        evidence_for = []
        if kpi == causal_chain[0] if causal_chain else False:
            evidence_for.append(f"Changed FIRST in temporal sequence (day {change_idx})")
        if effect > 1.5:
            evidence_for.append(f"Large effect size (Cohen's d = {effect:.2f})")
        if kpi in material_kpis:
            evidence_for.append("Flagged as materially significant by STL+CUSUM")
        if rank_entry["score"] > 0.3:
            evidence_for.append(f"High graph centrality score ({rank_entry['score']:.3f})")

        # Evidence AGAINST
        evidence_against = []
        if position_in_chain > 1:
            earlier = causal_chain[0] if causal_chain else "another KPI"
            evidence_against.append(f"'{earlier}' changed {position_in_chain} positions earlier in chain")
        if effect < 0.5:
            evidence_against.append(f"Small effect size (Cohen's d = {effect:.2f})")

        # Scenario-specific adjustments
        if scenario == "contradiction_promo" and kpi == "revenue":
            evidence_against.append("Revenue compensated by active promo — chain may be masked")

        root_causes.append({
            "kpi": kpi,
            "label": kpi.replace("_", " ").title(),
            "graph_score": rank_entry["score"],
            "is_material": rank_entry["is_material"],
            "change_day": change_idx,
            "position_in_causal_chain": position_in_chain + 1,
            "effect_size": effect,
            "evidence_for": evidence_for,
            "evidence_against": evidence_against,
            "confidence": round(min(1.0, rank_entry["score"] + 0.1 * len(evidence_for) - 0.05 * len(evidence_against)), 4),
        })

    # Sort by composite score (graph score + temporal position + effect size)
    for rc in root_causes:
        chain_penalty = rc["position_in_causal_chain"] * 0.05
        rc["composite_score"] = round(rc["graph_score"] + rc["effect_size"] * 0.1 - chain_penalty, 4)

    root_causes.sort(key=lambda x: x["composite_score"], reverse=True)

    primary = root_causes[0] if root_causes else None

    # Global evidence summary
    evidence_for = []
    evidence_against = []
    if primary:
        evidence_for = primary["evidence_for"]
        evidence_against = primary["evidence_against"]

    return {
        "root_causes": root_causes,
        "primary_cause": primary,
        "causal_chain": causal_chain,
        "evidence_summary": {
            "for": evidence_for,
            "against": evidence_against,
        },
        "change_times": {k: v for k, v in change_times.items() if v is not None},
        "effect_sizes": effect_sizes,
    }

"""
pipeline/root_cause.py

Ranks root cause candidates by combining temporal sequencing,
effect sizing (Cohen's d), graph scores, and PVM contributions.

Revenue is never considered its own root cause.
PVM volume (quantity) is treated as a balancing item, not a causal driver.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


DEFAULT_BASELINE_WINDOW = 30
PERSISTENCE_POINTS = 2
SIGNIFICANCE_THRESHOLD_STD = 1.5


def _first_significant_change(
    series: pd.Series,
    threshold_std: float = SIGNIFICANCE_THRESHOLD_STD,
) -> Optional[int]:
    """
    Returns the index of the first significant deviation from baseline.

    For zero-variance baselines (e.g. price step-changes), uses direct
    value comparison instead of z-scores. For noisy series, requires
    two consecutive significant points to reduce noise false-positives.
    """

    clean = series.dropna()

    if len(clean) < 5:
        return None

    midpoint = min(DEFAULT_BASELINE_WINDOW, len(clean) // 2)
    baseline = clean.iloc[:midpoint]
    mu = baseline.mean()
    sigma = baseline.std()

    if pd.isna(mu):
        return None

    # Zero-variance baseline: detect deterministic step changes.
    if pd.isna(sigma) or sigma == 0:
        rest = clean.iloc[midpoint:]

        if len(rest) == 0:
            return None

        if mu != 0:
            deviation = ~np.isclose(
                rest.to_numpy(dtype=float),
                float(mu),
                rtol=1e-9,
                atol=1e-9,
            )
        else:
            deviation = np.abs(rest.to_numpy(dtype=float)) > 1e-9

        if deviation.any():
            return int(midpoint + np.flatnonzero(deviation)[0])

        return None

    z = (series - mu) / sigma
    significant = z.abs() >= threshold_std
    significant_values = significant.fillna(False).to_numpy()

    for idx in range(len(significant_values) - 1):
        if significant_values[idx] and significant_values[idx + 1]:
            return int(idx)

    if significant_values.any():
        return int(np.flatnonzero(significant_values)[0])

    return None


def _effect_size(
    series: pd.Series,
    change_idx: Optional[int],
) -> float:
    """Cohen's d between pre- and post-change periods."""

    if change_idx is None or change_idx < 5:
        return 0.0

    pre = series.iloc[:change_idx].dropna()
    post = series.iloc[change_idx:].dropna()

    if len(pre) < 3 or len(post) < 3:
        return 0.0

    pooled_std = np.sqrt((pre.std() ** 2 + post.std() ** 2) / 2)

    if pooled_std == 0 or np.isnan(pooled_std):
        return 0.0

    return round(float(abs(pre.mean() - post.mean()) / pooled_std), 4)


def rank_root_causes(
    df: pd.DataFrame,
    driver_ranking: List[Dict[str, Any]],
    material_kpis: List[str],
    scenario: Optional[str] = None,
    target_kpi: str = "revenue",
    pvm_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Enriches graph driver ranking with temporal evidence and PVM contribution.

    PVM components map to observed KPIs:
        price -> unit_price, marketing -> marketing_spend,
        seasonal -> seasonal_index, volume -> quantity

    The PVM volume component is never the primary driver — it is a
    balancing/residual item that confirms accounting closure.
    """

    observed_kpis = [
        "warehouse_staffing_level",
        "fulfillment_delay_rate",
        "support_ticket_volume",
        "order_cancellation_rate",
        "unit_price",
        "marketing_spend",
        "seasonal_index",
        "quantity",
    ]

    pvm_components: Dict[str, float] = {}
    if pvm_result and pvm_result.get("status") == "OK":
        pvm_components = {
            k: float(v)
            for k, v in pvm_result.get("components", {}).items()
            if v is not None
        }

    pvm_to_observed = {
        "price": "unit_price",
        "marketing": "marketing_spend",
        "seasonal": "seasonal_index",
        "volume": "quantity",
    }

    change_times: Dict[str, Optional[int]] = {}
    impact_scores: Dict[str, float] = {}

    for kpi in observed_kpis:
        if kpi not in df.columns:
            continue

        change_idx = _first_significant_change(df[kpi])
        change_times[kpi] = change_idx
        cohen_d = _effect_size(df[kpi], change_idx)
        impact_scores[kpi] = round(min(1.0, cohen_d / 3.0), 4)

    ordered_by_time = sorted(
        [(kpi, idx) for kpi, idx in change_times.items() if idx is not None],
        key=lambda x: x[1],
    )
    temporal_sequence = [kpi for kpi, _ in ordered_by_time]

    total_pvm_abs = max(1.0, float(sum(abs(v) for v in pvm_components.values())))

    pvm_impact_scores: Dict[str, float] = {}
    for component, observed_kpi in pvm_to_observed.items():
        if component not in pvm_components:
            continue
        pvm_impact_scores[observed_kpi] = round(
            abs(pvm_components[component]) / total_pvm_abs, 4
        )

    root_causes = []

    for rank_entry in driver_ranking:
        kpi = rank_entry.get("kpi")
        if not kpi:
            continue
        if kpi == target_kpi:
            continue
        if kpi not in observed_kpis:
            continue

        change_idx = change_times.get(kpi)

        position_in_sequence = (
            temporal_sequence.index(kpi) + 1
            if kpi in temporal_sequence
            else len(temporal_sequence) + 1
        )

        statistical_impact = impact_scores.get(kpi, 0.0)
        pvm_impact = pvm_impact_scores.get(kpi, 0.0)

        normalized_impact = (
            round(max(statistical_impact, pvm_impact), 4)
            if kpi in pvm_impact_scores
            else statistical_impact
        )

        evidence_for: List[str] = []

        if temporal_sequence and kpi == temporal_sequence[0]:
            evidence_for.append(
                f"Changed FIRST in temporal sequence (day {change_idx})"
            )

        if statistical_impact > 0.4:
            evidence_for.append(
                f"Large statistical impact (normalized score = {statistical_impact:.2f})"
            )

        if kpi in material_kpis:
            evidence_for.append("Flagged as materially significant by statistical detectors")

        if kpi in pvm_impact_scores:
            evidence_for.append("Quantified by PVM decomposition")
            component = next(
                (name for name, obs in pvm_to_observed.items() if obs == kpi), None
            )
            if component:
                usd_value = pvm_components.get(component, 0.0)
                evidence_for.append(f"PVM {component} contribution = ${usd_value:,.2f}")

        graph_score = float(rank_entry.get("score", 0.0))
        if graph_score > 0.2:
            evidence_for.append(f"High graph centrality score ({graph_score:.3f})")

        evidence_against: List[str] = []

        if position_in_sequence > 1:
            earlier = temporal_sequence[0] if temporal_sequence else "another KPI"
            evidence_against.append(f"'{earlier}' changed earlier in the temporal sequence")

        if normalized_impact < 0.15:
            evidence_against.append(
                f"Small impact magnitude (normalized score = {normalized_impact:.2f})"
            )

        if (
            scenario in ("contradiction_promo", "contradictory_evidence")
            and kpi == "order_cancellation_rate"
        ):
            evidence_against.append(
                "Revenue impact may be partially masked by active promotion"
            )

        is_balancing_item = False
        if kpi == "quantity" and "volume" in pvm_components:
            is_balancing_item = True
            evidence_against.append(
                "PVM volume is a residual/balancing component; it confirms "
                "accounting closure, not direct causal proof"
            )

        evidence_score = round(
            min(1.0, graph_score + 0.10 * len(evidence_for) - 0.05 * len(evidence_against)),
            4,
        )

        entry = {
            "kpi": kpi,
            "label": kpi.replace("_", " ").title(),
            "graph_score": round(graph_score, 4),
            "is_material": kpi in material_kpis,
            "change_day": change_idx,
            "position_in_temporal_sequence": position_in_sequence,
            "normalized_impact": normalized_impact,
            "statistical_impact": statistical_impact,
            "pvm_impact": pvm_impact,
            "effect_size": statistical_impact,  # compatibility alias
            "evidence_for": evidence_for,
            "evidence_against": evidence_against,
            "evidence_score": evidence_score,
            "is_balancing_item": is_balancing_item,
        }

        for component, observed_kpi in pvm_to_observed.items():
            if observed_kpi == kpi and component in pvm_components:
                usd_value = pvm_components[component]
                entry["pvm_component"] = component
                entry["contribution_magnitude_usd"] = round(usd_value, 2)
                entry["contribution_score"] = round(abs(usd_value) / total_pvm_abs, 4)
                break

        root_causes.append(entry)

    for rc in root_causes:
        sequence_penalty = rc["position_in_temporal_sequence"] * 0.05
        rc["composite_score"] = round(
            rc["graph_score"] + rc["normalized_impact"] * 0.2 - sequence_penalty, 4
        )

    root_causes.sort(key=lambda x: x["composite_score"], reverse=True)

    # Quantity can be a PVM residual — exclude it from primary cause selection.
    causal_candidates = [rc for rc in root_causes if not rc["is_balancing_item"]]
    primary = causal_candidates[0] if causal_candidates else None

    evidence_for = primary["evidence_for"] if primary else []
    evidence_against = primary["evidence_against"] if primary else []

    return {
        "root_causes": root_causes,
        "primary_cause": primary,
        "temporal_sequence": temporal_sequence,
        "causal_chain": temporal_sequence,  # backward compatibility
        "evidence_summary": {"for": evidence_for, "against": evidence_against},
        "change_times": {k: v for k, v in change_times.items() if v is not None},
        "impact_scores": impact_scores,
        "effect_sizes": impact_scores,  # backward compatibility
        "pvm_impact_scores": pvm_impact_scores,
    }
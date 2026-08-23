"""
pipeline/rbac.py
Role-Based Access Control for persona-filtered API responses.

Personas:
  - gm (General Manager): all KPIs, all regions, confidence scores, action recommendations
  - ops_lead (Operations Lead): operational KPIs only (logistics, fulfillment), no financial margins
  - analyst: full data access, raw scores, graph topology (stretch persona)

RBAC is applied at the response level — the pipeline runs in full,
then results are filtered before being returned to the client.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional


PERSONA_DEFINITIONS = {
    "gm": {
        "label": "General Manager",
        "description": "Full access to all KPIs, regions, confidence scores, and action recommendations.",
        "color": "#7C3AED",
        "icon": "crown",
        "allowed_kpis": [
            "revenue",
            "order_cancellation_rate",
            "fulfillment_delay_rate",
            "support_ticket_volume",
            "warehouse_staffing_level",
        ],
        "allowed_fields": [
            "confidence_score",
            "verdict",
            "action",
            "root_causes",
            "evidence_summary",
            "causal_chain",
            "challenge_summary",
            "pvm_decomposition",
        ],
        "can_see_raw_scores": True,
        "can_see_graph_topology": True,
        "can_see_financial_impact": True,
    },
    "ops_lead": {
        "label": "Operations Lead",
        "description": "Operational KPIs only. No financial margins or revenue figures.",
        "color": "#059669",
        "icon": "settings",
        "allowed_kpis": [
            "fulfillment_delay_rate",
            "support_ticket_volume",
            "warehouse_staffing_level",
            "order_cancellation_rate",
        ],
        "allowed_fields": [
            "verdict",
            "action",
            "root_causes",
            "evidence_summary",
            "causal_chain",
        ],
        "can_see_raw_scores": False,
        "can_see_graph_topology": True,
        "can_see_financial_impact": False,
    },
    "analyst": {
        "label": "Data Analyst",
        "description": "Full data access including raw sub-scores and graph topology.",
        "color": "#0284C7",
        "icon": "chart",
        "allowed_kpis": [
            "revenue",
            "order_cancellation_rate",
            "fulfillment_delay_rate",
            "support_ticket_volume",
            "warehouse_staffing_level",
        ],
        "allowed_fields": [
            "confidence_score",
            "verdict",
            "action",
            "root_causes",
            "evidence_summary",
            "causal_chain",
            "challenge_summary",
            "pvm_decomposition",
            "raw_sub_scores",
            "graph_topology",
            "correlation_matrix",
        ],
        "can_see_raw_scores": True,
        "can_see_graph_topology": True,
        "can_see_financial_impact": True,
    },
}


def get_personas() -> List[Dict[str, Any]]:
    """Returns all persona definitions (for the /personas endpoint)."""
    return [
        {"id": pid, **{k: v for k, v in pdef.items()}}
        for pid, pdef in PERSONA_DEFINITIONS.items()
    ]


def filter_for_persona(
    investigation_result: Dict[str, Any],
    persona_id: str,
) -> Dict[str, Any]:
    """
    Filters an investigation result to show only what this persona is allowed to see.

    Args:
        investigation_result: full investigation result dict
        persona_id: "gm" | "ops_lead" | "analyst"

    Returns:
        Filtered result with a persona_context field added.
    """
    persona = PERSONA_DEFINITIONS.get(persona_id)
    if not persona:
        # Default to ops_lead (most restrictive) for unknown personas
        persona = PERSONA_DEFINITIONS["ops_lead"]
        persona_id = "ops_lead"

    filtered = {}

    # Always include these base fields
    for field in ["region_id", "scenario", "investigation_id", "timestamp"]:
        if field in investigation_result:
            filtered[field] = investigation_result[field]

    # KPI health strip — filter to allowed KPIs only
    if "kpi_health" in investigation_result:
        allowed = persona["allowed_kpis"]
        filtered["kpi_health"] = {
            k: v for k, v in investigation_result["kpi_health"].items()
            if k in allowed
        }

    # Materiality results — filter to allowed KPIs
    if "materiality" in investigation_result:
        mat = investigation_result["materiality"]
        allowed = persona["allowed_kpis"]
        filtered["materiality"] = {
            "material_kpis": [k for k in mat.get("material_kpis", []) if k in allowed],
            "any_material": mat.get("any_material"),
            "signal_strength": mat.get("signal_strength"),
            "kpi_results": {k: v for k, v in mat.get("kpi_results", {}).items() if k in allowed},
        }

    # Confidence score — only for personas with access
    if "confidence" in investigation_result:
        conf = investigation_result["confidence"]
        filtered["confidence"] = {
            "verdict": conf.get("verdict"),
            "explanation": conf.get("explanation"),
        }
        if persona["can_see_raw_scores"]:
            filtered["confidence"]["score"] = conf.get("score")
            filtered["confidence"]["sub_scores"] = conf.get("sub_scores")
            filtered["confidence"]["weights"] = conf.get("weights")

    # Verdict (always visible)
    filtered["verdict"] = investigation_result.get("verdict")

    # Action recommendation (always visible, but financial impact filtered)
    if "action" in investigation_result:
        action = investigation_result["action"].copy()
        if not persona["can_see_financial_impact"]:
            action.pop("estimated_impact", None)
        filtered["action"] = action

    # Root causes
    if "root_causes" in investigation_result and "root_causes" in persona["allowed_fields"]:
        rcs = investigation_result["root_causes"]
        filtered_rcs = []
        for rc in rcs:
            if rc.get("kpi") in persona["allowed_kpis"]:
                filtered_rc = {k: v for k, v in rc.items()}
                if not persona["can_see_raw_scores"]:
                    filtered_rc.pop("graph_score", None)
                    filtered_rc.pop("composite_score", None)
                filtered_rcs.append(filtered_rc)
        filtered["root_causes"] = filtered_rcs

    # Evidence summary
    if "evidence_summary" in investigation_result and "evidence_summary" in persona["allowed_fields"]:
        filtered["evidence_summary"] = investigation_result["evidence_summary"]

    # Causal chain
    if "causal_chain" in investigation_result and "causal_chain" in persona["allowed_fields"]:
        filtered["causal_chain"] = [
            k for k in investigation_result["causal_chain"]
            if k in persona["allowed_kpis"]
        ]

    # Challenge summary
    if "challenge_result" in investigation_result and "challenge_summary" in persona["allowed_fields"]:
        filtered["challenge_result"] = investigation_result["challenge_result"]

    # Evidence graph (topology)
    if "evidence_graph" in investigation_result and persona["can_see_graph_topology"]:
        graph = investigation_result["evidence_graph"]
        allowed = persona["allowed_kpis"]
        filtered["evidence_graph"] = {
            "nodes": [n for n in graph.get("graph_data", {}).get("nodes", []) if n["id"] in allowed],
            "links": [
                l for l in graph.get("graph_data", {}).get("links", [])
                if l["source"] in allowed and l["target"] in allowed
            ],
            "driver_ranking": [r for r in graph.get("driver_ranking", []) if r["kpi"] in allowed],
        }

    # PVM (only if financial access)
    if "pvm_decomposition" in investigation_result and persona["can_see_financial_impact"]:
        if "pvm_decomposition" in persona["allowed_fields"]:
            filtered["pvm_decomposition"] = investigation_result["pvm_decomposition"]

    # Raw sub-scores (analyst only)
    if persona["can_see_raw_scores"] and "raw_sub_scores" in persona["allowed_fields"]:
        filtered["raw_sub_scores"] = investigation_result.get("raw_sub_scores", {})

    # Correlation matrix (analyst only)
    if "correlation_matrix" in persona["allowed_fields"]:
        filtered["correlation_matrix"] = investigation_result.get("correlation_matrix", {})

    # Telemetry (always visible)
    if "telemetry" in investigation_result:
        filtered["telemetry"] = investigation_result["telemetry"]

    # Persona context (always appended)
    filtered["persona_context"] = {
        "persona_id": persona_id,
        "label": persona["label"],
        "description": persona["description"],
        "color": persona["color"],
        "restricted_kpis_hidden": [
            k for k in ["revenue", "order_cancellation_rate", "fulfillment_delay_rate", "support_ticket_volume", "warehouse_staffing_level"]
            if k not in persona["allowed_kpis"]
        ],
    }

    return filtered

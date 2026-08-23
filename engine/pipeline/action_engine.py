"""
pipeline/action_engine.py
Generates structured action recommendations based on Confidence Gate verdict.

Each action has:
  - action_id: unique identifier
  - type: OPERATIONAL | INVESTIGATIVE | ESCALATION | HOLD
  - title: short summary
  - description: full recommendation
  - owner: persona who should own this action
  - priority: HIGH | MEDIUM | LOW
  - estimated_impact: optional revenue/metric impact estimate
  - preconditions: what must be true before acting
  - risks: what could go wrong
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import uuid


ACTION_TEMPLATES = {
    "warehouse_staffing_level": {
        "ACT": {
            "type": "OPERATIONAL",
            "title": "Emergency Warehouse Staffing Recovery",
            "description": (
                "Warehouse staffing is the confirmed root cause. "
                "Immediately activate contingency staffing plan: "
                "(1) Redeploy staff from lower-volume regions. "
                "(2) Engage temporary staffing agency within 48 hours. "
                "(3) Communicate delay timeline to fulfillment team. "
                "(4) Set daily staffing recovery milestones."
            ),
            "owner": "ops_lead",
            "priority": "HIGH",
            "preconditions": ["Staffing data confirmed from WMS", "Delay correlation validated"],
            "risks": ["Temporary staff may require training time", "Redeployment may create gaps in source regions"],
        },
        "INVESTIGATE": {
            "type": "INVESTIGATIVE",
            "title": "Investigate Staffing Pattern Before Acting",
            "description": (
                "Staffing decline detected but contradictory evidence present. "
                "Before committing to staffing remediation: "
                "(1) Confirm staffing data accuracy with WMS team. "
                "(2) Check if delay increase predates staffing drop. "
                "(3) Review promo/demand calendar for external volume spike."
            ),
            "owner": "ops_lead",
            "priority": "MEDIUM",
            "preconditions": ["Cross-segment comparison available"],
            "risks": ["Delay in acting may worsen delay chain"],
        },
    },
    "fulfillment_delay_rate": {
        "ACT": {
            "type": "OPERATIONAL",
            "title": "Activate Delay Recovery Protocol",
            "description": (
                "Fulfillment delay is the primary driver of cancellations and revenue loss. "
                "(1) Prioritize aged orders in the queue. "
                "(2) Communicate proactive ETAs to customers with delayed orders. "
                "(3) Enable expedited shipping option for at-risk orders. "
                "(4) Review SLA breaches and initiate customer recovery credits."
            ),
            "owner": "ops_lead",
            "priority": "HIGH",
            "preconditions": ["Delay confirmed as structural, not one-time"],
            "risks": ["Expedited shipping cost may offset revenue recovery"],
        },
    },
    "revenue": {
        "ACT": {
            "type": "ESCALATION",
            "title": "Escalate Revenue Impact to GM",
            "description": (
                "Revenue loss is confirmed and causal chain is traced. "
                "Escalate to General Manager with full evidence package. "
                "Recommend immediate operational intervention and 30-day revenue recovery plan."
            ),
            "owner": "gm",
            "priority": "HIGH",
            "preconditions": ["Root cause confirmed", "Impact quantified"],
            "risks": ["Escalation without full evidence may reduce credibility"],
        },
        "INVESTIGATE": {
            "type": "INVESTIGATIVE",
            "title": "Decompose Revenue Movement Before Escalating",
            "description": (
                "Revenue signal is material but causation is ambiguous (possible promo compensation). "
                "Run Price-Volume-Mix decomposition to isolate driver contributions before escalating."
            ),
            "owner": "analyst",
            "priority": "MEDIUM",
            "preconditions": ["PVM decomposition available"],
            "risks": ["Delay in escalation may miss response window"],
        },
    },
}

DEFAULT_ABSTAIN_ACTION = {
    "action_id": None,
    "type": "HOLD",
    "title": "Hold — Insufficient Evidence to Act",
    "description": (
        "Confidence score is below the action threshold. "
        "Do not act until data quality or evidence improves. "
        "Recommended next steps: "
        "(1) Identify and resolve data gaps (see data quality report). "
        "(2) Allow more history to accumulate (minimum 14 days needed). "
        "(3) Re-run investigation in 5 business days."
    ),
    "owner": "analyst",
    "priority": "LOW",
    "preconditions": [],
    "risks": ["Delayed action may allow situation to worsen"],
    "estimated_impact": None,
}

DEFAULT_INVESTIGATE_ACTION = {
    "action_id": None,
    "type": "INVESTIGATIVE",
    "title": "Conduct Structured Investigation",
    "description": (
        "Evidence is present but contradictory. Confidence is in the INVESTIGATE zone. "
        "(1) Cross-reference against at least one comparable region/segment. "
        "(2) Verify data provenance for the flagged sources. "
        "(3) Escalate to GM only after resolving the identified contradictions."
    ),
    "owner": "analyst",
    "priority": "MEDIUM",
    "preconditions": ["Challenge engine findings reviewed"],
    "risks": ["Investigation scope creep may delay resolution"],
    "estimated_impact": None,
}


def build_action(
    verdict: str,
    primary_cause: Optional[Dict[str, Any]],
    confidence_score: float,
    revenue_impact_estimate: Optional[float] = None,
    scenario: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates a structured action recommendation.

    Args:
        verdict: "ACT" | "INVESTIGATE" | "ABSTAIN"
        primary_cause: top root cause candidate dict
        confidence_score: float [0, 1]
        revenue_impact_estimate: estimated revenue impact in USD
        scenario: optional scenario hint

    Returns full action schema.
    """
    if verdict == "ABSTAIN":
        action = DEFAULT_ABSTAIN_ACTION.copy()
        action["action_id"] = f"ACT-{str(uuid.uuid4())[:8].upper()}"
        return action

    if verdict == "INVESTIGATE":
        # Try to get a specific investigative action for the primary cause
        cause_kpi = primary_cause.get("kpi", "") if primary_cause else ""
        template = ACTION_TEMPLATES.get(cause_kpi, {}).get("INVESTIGATE")
        if template:
            action = template.copy()
        else:
            action = DEFAULT_INVESTIGATE_ACTION.copy()
        action["action_id"] = f"ACT-{str(uuid.uuid4())[:8].upper()}"
        action["confidence_score"] = confidence_score
        action["estimated_impact"] = (
            f"${abs(revenue_impact_estimate):,.0f} at risk" if revenue_impact_estimate else None
        )
        return action

    # ACT
    cause_kpi = primary_cause.get("kpi", "") if primary_cause else ""
    template = ACTION_TEMPLATES.get(cause_kpi, {}).get("ACT")

    if not template:
        # Generic ACT action
        template = {
            "type": "OPERATIONAL",
            "title": f"Act on {cause_kpi.replace('_', ' ').title()} Root Cause",
            "description": f"Confidence is high ({confidence_score:.2f}). The identified root cause ({cause_kpi}) requires immediate operational response.",
            "owner": "ops_lead",
            "priority": "HIGH",
            "preconditions": ["Root cause confirmed"],
            "risks": ["Acting on wrong cause may waste resources"],
        }

    action = template.copy()
    action["action_id"] = f"ACT-{str(uuid.uuid4())[:8].upper()}"
    action["confidence_score"] = confidence_score
    action["primary_cause"] = cause_kpi
    action["estimated_impact"] = (
        f"${abs(revenue_impact_estimate):,.0f} recoverable revenue" if revenue_impact_estimate else None
    )

    return action

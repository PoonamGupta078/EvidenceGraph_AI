"""
main.py — EvidenceGraph AI Engine
FastAPI application with all investigation endpoints.

Endpoints:
  GET  /health
  GET  /kpis
  POST /investigations/run
  GET  /investigations/{id}
  POST /sandbox/simulate
  GET  /sandbox/levers
  POST /feedback
  GET  /feedback/stats
  GET  /telemetry/{investigation_id}
  GET  /telemetry/aggregate
  GET  /personas
"""

from __future__ import annotations
import os
import uuid
import time
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Pipeline imports
from pipeline.reconciliation import reconcile_sources
from pipeline.data_reality_check import check_data_reality
from pipeline.materiality import detect_materiality
from pipeline.evidence_graph import build_evidence_graph
from pipeline.root_cause import rank_root_causes
from pipeline.confidence import compute_confidence
from pipeline.challenge_engine import run_challenge
from pipeline.action_engine import build_action
from pipeline.pvm_decomposition import decompose_pvm
from pipeline.calendar_reconciliation import check_calendar_effects
from pipeline.rbac import filter_for_persona, get_personas
from pipeline.intervention_sandbox import simulate_intervention, get_lever_options
from rag.retriever import retrieve_evidence
from llm.narrator import generate_narrative
from feedback.store import store_feedback, get_feedback_stats, get_feedback_for_investigation
from telemetry.tracker import record_telemetry, get_telemetry, get_aggregate_telemetry, TelemetryTimer

# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data" / "generated"

REGION_SCENARIOS = {
    "region_a": "staffing_chain",
    "region_b": "contradiction_promo",
    "region_c": "data_quality_abstain",
    "region_d": "sparse_history",
    "region_e": "multi_factor_pvm",
}

# In-memory investigation cache
_investigations: Dict[str, Dict] = {}

# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm: check data files exist."""
    missing = []
    for region in REGION_SCENARIOS:
        path = DATA_DIR / f"{region}.csv"
        if not path.exists():
            missing.append(str(path))
    if missing:
        print(f"⚠️  Missing data files: {missing}")
        print("   Run: python data/generate_synthetic.py")
    else:
        print(f"✅ All {len(REGION_SCENARIOS)} region data files found.")
    yield


app = FastAPI(
    title="EvidenceGraph AI Engine",
    description="Autonomous anomaly investigation engine with typed evidence graph, confidence gate, and LLM narration.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend origins
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────────────────────

class InvestigationRequest(BaseModel):
    region_id: str
    persona_id: str = "gm"
    include_narrative: bool = True
    include_rag: bool = True

class SandboxRequest(BaseModel):
    region_id: str
    lever: str
    lever_value: float

class FeedbackRequest(BaseModel):
    investigation_id: str
    region_id: str
    persona_id: str = "gm"
    verdict: str
    user_verdict: Optional[str] = None
    driver_selected: Optional[str] = None
    rating: str = "correct"
    comment: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_region_data(region_id: str) -> pd.DataFrame:
    path = DATA_DIR / f"{region_id}.csv"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Data file for '{region_id}' not found. Run: python data/generate_synthetic.py",
        )
    return pd.read_csv(path, parse_dates=["date"])


def _kpi_health_strip(df: pd.DataFrame, materiality_result: Dict) -> Dict[str, Any]:
    """Builds the KPI health strip for the frontend."""
    kpis = ["revenue", "order_cancellation_rate", "fulfillment_delay_rate",
            "support_ticket_volume", "warehouse_staffing_level"]
    strip = {}
    for kpi in kpis:
        if kpi not in df.columns:
            continue
        recent = df[kpi].iloc[-7:].dropna()
        prior = df[kpi].iloc[-14:-7].dropna()
        current_val = float(recent.mean()) if len(recent) > 0 else None
        prior_val = float(prior.mean()) if len(prior) > 0 else None
        pct_change = None
        if current_val is not None and prior_val and prior_val != 0:
            pct_change = round((current_val - prior_val) / abs(prior_val) * 100, 2)

        strip[kpi] = {
            "current_value": round(current_val, 2) if current_val else None,
            "prior_value": round(prior_val, 2) if prior_val else None,
            "pct_change_7d": pct_change,
            "is_material": kpi in materiality_result.get("material_kpis", []),
            "trend": "up" if (pct_change or 0) > 0 else "down" if (pct_change or 0) < 0 else "flat",
        }
    return strip


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "regions_available": list(REGION_SCENARIOS.keys()),
    }


@app.get("/personas")
def personas():
    return {"personas": get_personas()}


@app.get("/kpis")
def kpi_overview():
    """
    Returns KPI health strip for all available regions.
    Used by the frontend's main overview screen.
    """
    results = {}
    for region_id in REGION_SCENARIOS:
        try:
            df = _load_region_data(region_id)
            reconciled = reconcile_sources(df)
            aligned_df = reconciled["aligned_df"]
            materiality = detect_materiality(aligned_df)
            strip = _kpi_health_strip(aligned_df, materiality)
            results[region_id] = {
                "label": region_id.replace("_", " ").title(),
                "scenario": REGION_SCENARIOS[region_id],
                "kpi_health": strip,
                "any_material": materiality.get("any_material"),
                "data_days": reconciled.get("total_days"),
            }
        except Exception as e:
            results[region_id] = {"error": str(e)}
    return {"regions": results}


@app.post("/investigations/run")
def run_investigation(req: InvestigationRequest):
    """
    Runs the full investigation pipeline for a region.
    Returns persona-filtered result with optional LLM narrative.
    """
    timer = TelemetryTimer()
    llm_tokens = {"prompt": 0, "completion": 0}
    llm_model = None
    rag_latency = 0.0

    region_id = req.region_id
    scenario = REGION_SCENARIOS.get(region_id)

    if region_id not in REGION_SCENARIOS:
        raise HTTPException(status_code=400, detail=f"Unknown region: {region_id}. Valid: {list(REGION_SCENARIOS.keys())}")

    with timer.stage("pipeline"):
        # 1. Load data
        df = _load_region_data(region_id)

        # 2. Reconcile sources
        reconciled = reconcile_sources(df)
        aligned_df = reconciled["aligned_df"]

        # 3. Data reality check
        reality = check_data_reality(
            aligned_df,
            reconciled["source_completeness"],
            reconciled["total_days"],
        )

        # 4. Early ABSTAIN on data quality failure
        if not reality["passes"]:
            investigation_id = str(uuid.uuid4())
            result = {
                "investigation_id": investigation_id,
                "region_id": region_id,
                "scenario": scenario,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "verdict": "ABSTAIN",
                "abstain_reason": reality["abstain_reason"],
                "data_quality": reality,
                "confidence": {"score": reality["quality_score"], "verdict": "ABSTAIN", "explanation": f"Data quality check failed: {reality['abstain_reason']}"},
                "action": {
                    "type": "HOLD",
                    "title": "Hold — Data Quality Insufficient",
                    "description": f"Cannot investigate: {reality['abstain_reason']}. Resolve data gaps before re-running.",
                    "priority": "LOW",
                },
                "kpi_health": {},
                "material_kpis": [],
                "root_causes": [],
                "evidence_summary": {"for": [], "against": []},
                "causal_chain": [],
                "challenge_result": {},
                "evidence_graph": {"graph_data": {"nodes": [], "links": []}, "driver_ranking": []},
            }
            _investigations[investigation_id] = result
            pipeline_ms = timer.stages.get("pipeline", 0)
            record_telemetry(investigation_id, region_id, pipeline_ms, llm_used=False)
            return filter_for_persona(result, req.persona_id)

        # 5. Materiality detection
        materiality = detect_materiality(aligned_df)

        # 6. KPI health strip
        kpi_health = _kpi_health_strip(aligned_df, materiality)

        # 7. Evidence graph
        evidence_graph = build_evidence_graph(aligned_df, materiality["material_kpis"], region_id, scenario)

        # 8. Root cause ranking
        root_cause = rank_root_causes(aligned_df, evidence_graph["driver_ranking"], materiality["material_kpis"], scenario)

        # 9. PVM decomposition (for Region E or any revenue-material case)
        pvm = None
        if scenario == "multi_factor_pvm" or "revenue" in materiality["material_kpis"]:
            pvm = decompose_pvm(aligned_df)

        # 10. Calendar check
        anomaly_indices = [
            v.get("cusum_detection_index")
            for v in materiality.get("kpi_results", {}).values()
            if v.get("cusum_detection_index") is not None
        ]
        calendar = check_calendar_effects(aligned_df, materiality["material_kpis"], anomaly_indices)

        # 11. Confidence scoring
        confidence = compute_confidence(
            quality_score=reality["quality_score"],
            signal_strength=materiality["signal_strength"],
            correlation_matrix=evidence_graph["correlation_matrix"],
            root_causes=root_cause["root_causes"],
            causal_chain=root_cause["causal_chain"],
            effect_sizes=root_cause["effect_sizes"],
            scenario=scenario,
        )

        # 12. Challenge engine
        comparison_regions = [
            {"region_id": "region_b", "verdict": "INVESTIGATE", "primary_cause_kpi": "warehouse_staffing_level"}
        ] if region_id == "region_a" else []

        challenge = run_challenge(
            aligned_df,
            region_id,
            {
                "verdict": confidence["verdict"],
                "primary_cause": root_cause["primary_cause"],
                "material_kpis": materiality["material_kpis"],
            },
            comparison_regions=comparison_regions or None,
            scenario=scenario,
        )

        # Apply verdict adjustment from challenge engine
        final_verdict = challenge.get("verdict_adjustment") or confidence["verdict"]

        # 13. Revenue impact estimate
        revenue_impact = None
        if "revenue" in kpi_health:
            rh = kpi_health["revenue"]
            if rh.get("pct_change_7d") and rh.get("current_value"):
                revenue_impact = rh["current_value"] * abs(rh["pct_change_7d"]) / 100 * 30

        # 14. Action recommendation
        action = build_action(
            verdict=final_verdict,
            primary_cause=root_cause["primary_cause"],
            confidence_score=confidence["score"],
            revenue_impact_estimate=revenue_impact,
            scenario=scenario,
        )

    # 15. RAG evidence retrieval
    rag_result = {"results": []}
    if req.include_rag:
        rag_start = time.perf_counter()
        primary_kpi = root_cause["primary_cause"]["kpi"] if root_cause["primary_cause"] else ""
        rag_query = f"{primary_kpi} {scenario} delay cancellation staffing"
        rag_result = retrieve_evidence(rag_query, region_id=region_id, top_k=5)
        rag_latency = round((time.perf_counter() - rag_start) * 1000, 2)

    investigation_id = str(uuid.uuid4())
    pipeline_ms = timer.stages.get("pipeline", 0)

    full_result = {
        "investigation_id": investigation_id,
        "region_id": region_id,
        "scenario": scenario,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": final_verdict,
        "confidence": confidence,
        "action": action,
        "kpi_health": kpi_health,
        "material_kpis": materiality["material_kpis"],
        "materiality": materiality,
        "root_causes": root_cause["root_causes"],
        "primary_cause": root_cause["primary_cause"],
        "causal_chain": root_cause["causal_chain"],
        "evidence_summary": root_cause["evidence_summary"],
        "challenge_result": challenge,
        "evidence_graph": evidence_graph,
        "pvm_decomposition": pvm,
        "calendar_check": calendar,
        "data_quality": reality,
        "rag_evidence": rag_result,
        "correlation_matrix": evidence_graph.get("correlation_matrix", {}),
        "raw_sub_scores": confidence.get("sub_scores", {}),
    }

    # 16. LLM narrative
    llm_latency = 0.0
    if req.include_narrative:
        llm_start = time.perf_counter()
        persona_filtered = filter_for_persona(full_result, req.persona_id)
        narrative_result = generate_narrative(persona_filtered, req.persona_id, investigation_id)
        full_result["narrative"] = narrative_result
        llm_latency = round((time.perf_counter() - llm_start) * 1000, 2)
        if "tokens_used" in narrative_result:
            llm_tokens = narrative_result["tokens_used"]
            llm_model = narrative_result.get("model")

    # 17. Record telemetry
    tel = record_telemetry(
        investigation_id=investigation_id,
        region_id=region_id,
        pipeline_latency_ms=pipeline_ms,
        llm_latency_ms=llm_latency,
        rag_latency_ms=rag_latency,
        tokens_prompt=llm_tokens.get("prompt", 0),
        tokens_completion=llm_tokens.get("completion", 0),
        model=llm_model,
        llm_used=req.include_narrative,
    )
    full_result["telemetry"] = tel

    # Cache
    _investigations[investigation_id] = full_result

    # Return persona-filtered
    return filter_for_persona(full_result, req.persona_id)


@app.get("/investigations/{investigation_id}")
def get_investigation(investigation_id: str, persona_id: str = Query(default="gm")):
    if investigation_id not in _investigations:
        raise HTTPException(status_code=404, detail="Investigation not found. It may have expired (in-memory only).")
    return filter_for_persona(_investigations[investigation_id], persona_id)


@app.post("/sandbox/simulate")
def sandbox_simulate(req: SandboxRequest):
    """Runs an intervention simulation for a region and lever."""
    df = _load_region_data(req.region_id)
    reconciled = reconcile_sources(df)
    result = simulate_intervention(reconciled["aligned_df"], req.lever, req.lever_value)
    return result


@app.get("/sandbox/levers")
def sandbox_levers():
    return {"levers": get_lever_options()}


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    record = store_feedback(
        investigation_id=req.investigation_id,
        region_id=req.region_id,
        persona_id=req.persona_id,
        verdict=req.verdict,
        user_verdict=req.user_verdict,
        driver_selected=req.driver_selected,
        rating=req.rating,
        comment=req.comment,
    )
    return {"success": True, "feedback": record}


@app.get("/feedback/stats")
def feedback_stats():
    return get_feedback_stats()


@app.get("/telemetry/{investigation_id}")
def telemetry_for_investigation(investigation_id: str):
    tel = get_telemetry(investigation_id)
    if not tel:
        raise HTTPException(status_code=404, detail="No telemetry found for this investigation.")
    return tel


@app.get("/telemetry/aggregate/stats")
def telemetry_aggregate():
    return get_aggregate_telemetry()

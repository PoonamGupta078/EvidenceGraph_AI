"""
llm/chatbot.py
Investigation-aware conversational chatbot for EvidenceGraph AI.

The chatbot is a context-aware Q&A layer over pre-computed investigation results.
It does NOT re-run the pipeline or re-determine root causes.
All conclusions come directly from the authoritative pipeline results.

The investigation dict passed in MUST already be RBAC-filtered for the
requesting persona.

LLM Priority:
  1. Google Gemini 2.5 Flash (via google-genai SDK)
  2. Groq Llama3 (legacy fallback)
  3. Deterministic keyword-match fallback
"""

from __future__ import annotations
import os
import time
from typing import Any, Dict, List, Optional

from rag.retriever import retrieve_evidence

# ─── Gemini SDK ───────────────────────────────────────────────────────────────
try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# ─── Groq SDK (legacy fallback) ───────────────────────────────────────────────
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

GEMINI_MODEL = "gemini-2.5-flash"
GROQ_MODEL   = "llama3-8b-8192"

# ─── Persona system styles ────────────────────────────────────────────────────
_PERSONA_TONE = {
    "gm":       "Respond concisely in executive business language. Focus on financial recovery, strategic decisions, and impact.",
    "ops_lead": "Respond in direct operational language. Focus on fulfillment delays, staffing levels, support tickets, and immediate operational playbooks. DO NOT invent or estimate restricted financial/revenue figures.",
    "analyst":  "Respond analytically. Include confidence sub-scores, evidence graph relationships, correlation details, and technical caveats.",
}


# ─── System prompt builder ────────────────────────────────────────────────────

def _build_system_prompt(
    investigation: Dict[str, Any],
    persona_id: str,
    dynamic_rag: Optional[Dict[str, Any]] = None,
) -> str:
    verdict      = investigation.get("verdict", "UNKNOWN")
    region       = investigation.get("region_id", "unknown").replace("_", " ").title()
    scenario     = investigation.get("scenario", "unknown").replace("_", " ")
    conf         = (investigation.get("confidence") or {})
    score        = conf.get("score")
    explanation  = conf.get("explanation", "")

    score_str = f"{score:.2f} / 1.00" if score is not None else "Restricted / Not available"

    # Authoritative Primary Cause
    pc = investigation.get("primary_cause") or {}
    if isinstance(pc, dict):
        primary_kpi   = pc.get("kpi", "").replace("_", " ")
        primary_label = pc.get("label", "Unknown")
        primary_conf  = pc.get("confidence")
        primary_conf_str = f" (confidence: {primary_conf:.2f})" if primary_conf is not None else ""
        primary_cause_str = f"{primary_label} [KPI: {primary_kpi}]{primary_conf_str}"
    elif isinstance(pc, str):
        primary_cause_str = pc.replace("_", " ").title()
    else:
        primary_cause_str = "None identified"

    # Root causes
    rc_lines = []
    for rc in (investigation.get("root_causes") or [])[:4]:
        if isinstance(rc, dict):
            kpi_name = rc.get("kpi", "?")
            rc_lines.append(f"  - {rc.get('label', kpi_name)} (KPI: {kpi_name})")
        elif isinstance(rc, str):
            rc_lines.append(f"  - {rc.replace('_', ' ').title()}")
    rc_block = "\n".join(rc_lines) or "  - None listed"

    # Action recommendation
    action       = investigation.get("action") or {}
    action_title = action.get("title", "No action specified")
    action_desc  = action.get("description", "")
    action_owner = action.get("owner", "")
    action_impact = action.get("estimated_impact", "N/A")

    # Causal chain
    chain     = investigation.get("causal_chain") or investigation.get("temporal_sequence") or []
    chain_str = " → ".join([c.replace("_", " ") for c in chain]) if chain else "Not available"

    # Evidence summary
    ev = investigation.get("evidence_summary") or {}
    ev_for = "; ".join((ev.get("for") or [])[:3]) or "None"
    ev_against = "; ".join((ev.get("against") or [])[:2]) or "None"

    # PVM decomposition (if allowed for persona)
    pvm = investigation.get("pvm_decomposition") or {}
    pvm_block = ""
    if pvm:
        comps = pvm.get("components") or {}
        pvm_parts = [f"{k}: ${v:,.0f}" for k, v in comps.items()]
        pvm_block = f"""
PVM Decomposition (Commercial/Factor Drivers):
  Primary driver: {pvm.get('primary_driver', 'N/A')}
  Total revenue change: ${pvm.get('total_change_usd', 0):,.0f}
  Components: {', '.join(pvm_parts) or 'N/A'}
  Note: Volume/quantity is a balancing factor, NOT the root cause.
"""

    # Data Quality
    dq = investigation.get("data_quality") or {}
    dq_passes = dq.get("passes", True)
    dq_score = dq.get("quality_score", "N/A")
    dq_gates = dq.get("gate_results", {})
    failed_gates = [k for k, v in dq_gates.items() if isinstance(v, dict) and not v.get("passed")]
    dq_block = f"Data quality score: {dq_score}, passes: {dq_passes}"
    if failed_gates:
        dq_block += f" | Failed gates: {', '.join(failed_gates)}"

    # Challenge Engine
    challenge = investigation.get("challenge_result") or {}
    challenge_summary = challenge.get("challenge_summary", "No contradictions detected")
    contradictions = challenge.get("contradictions", [])
    challenge_block = challenge_summary
    if contradictions:
        challenge_block += f" | Details: {'; '.join(contradictions[:2])}"

    # KPI Health
    kpi_health = investigation.get("kpi_health") or {}
    material_kpis = [k.replace("_", " ") for k, v in kpi_health.items() if isinstance(v, dict) and v.get("is_material")]

    # Dynamic RAG snippets
    rag_block = ""
    if dynamic_rag and dynamic_rag.get("results"):
        rag_lines = [f"  - [{r.get('id','?')}] ({r.get('category','?')}, score {r.get('score',0):.2f}): \"{r.get('text','')}\"" for r in dynamic_rag["results"]]
        rag_block = "Retrieved Operational/Support Evidence:\n" + "\n".join(rag_lines)

    persona_tone = _PERSONA_TONE.get(persona_id, _PERSONA_TONE["gm"])

    system = f"""You are EvidenceGraph AI — Accenture's autonomous investigation engine assistant for e-commerce order fulfillment.

You are answering user questions regarding a completed anomaly investigation.

══ AUTHORITATIVE INVESTIGATION CONTEXT ═════════════════════════
Region:               {region}
Scenario:             {scenario}
Verdict:              {verdict}
Confidence Score:     {score_str}
Explanation:          {explanation}

AUTHORITATIVE PRIMARY CAUSE (MUST PREFER OVER ALL OTHER CANDIDATES):
  {primary_cause_str}

Ranked Root Causes:
{rc_block}

Causal Propagation Chain: {chain_str}

Material KPIs: {', '.join(material_kpis) or 'None'}

Evidence FOR:    {ev_for}
Evidence AGAINST: {ev_against}

Recommended Remediation Action:
  Title:       {action_title}
  Description: {action_desc}
  Owner:       {action_owner}
  Impact:      {action_impact}

{pvm_block}
Challenge Engine Findings: {challenge_block}
Data Quality Status:       {dq_block}
{rag_block}
══ END CONTEXT ════════════════════════════════════════════════

Persona Tone Directive: {persona_tone}

CRITICAL RULES FOR RESPONDING:
1. DIRECTLY answer the user's specific question. Do not dump the entire investigation summary unless asked.
2. AUTHORITATIVE PRIMARY CAUSE: Always treat `{primary_cause_str}` as the true root cause. NEVER treat volume/quantity balancing items as the root cause.
3. VERDICT EXPLANATIONS:
   - If verdict is `ABSTAIN`: Explain that data quality checks failed ({failed_gates or 'data freshness lag'}) or historical data was sparse, making automated action unsafe.
   - If verdict is `INVESTIGATE`: Explain the contradiction detected by the Challenge Engine ({challenge_block}).
   - If verdict is `ACT`: Explain the high confidence score ({score_str}) and solid causal chain supporting immediate remediation.
4. ROLE RESTRICTION (RBAC): If the user asks about financial impact or revenue numbers and the context shows "Restricted" or lacks financial figures (e.g. for Operations Lead), explicitly state: "Financial and revenue impact metrics are restricted for the Operations Lead role."
5. NO HALLUCINATION: Never invent numbers, KPI values, or causes outside this context.
6. CONCISE: Keep responses to 2–4 clear, professional sentences unless deep analytical detail is requested.
"""
    return system


# ─── Deterministic Fallback ───────────────────────────────────────────────────

def _deterministic_fallback(message: str, investigation: Dict[str, Any], persona_id: str) -> str:
    msg = message.lower()
    verdict = investigation.get("verdict", "UNKNOWN")
    pc = investigation.get("primary_cause") or {}
    kpi_label = pc.get("label", "the primary driver") if isinstance(pc, dict) else str(pc)
    conf = investigation.get("confidence") or {}
    score = conf.get("score")
    action = investigation.get("action") or {}
    action_title = action.get("title", "No action specified")
    dq = investigation.get("data_quality") or {}
    challenge = investigation.get("challenge_result") or {}

    # RBAC check for ops_lead asking for financial figures
    if persona_id == "ops_lead" and any(w in msg for w in ["revenue", "financial", "usd", "cost", "dollar", "money"]):
        return "Financial and revenue impact metrics are restricted for the Operations Lead persona. Please consult the General Manager view."

    if verdict == "ABSTAIN" and any(w in msg for w in ["abstain", "why", "cause", "reason", "quality"]):
        return f"The system abstained because data quality gates failed or historical data was sparse (Quality Score: {dq.get('quality_score', 'N/A')}). Automated action was paused to protect against decisions on stale data."

    if verdict == "INVESTIGATE" and any(w in msg for w in ["investigate", "why", "contradiction", "challenge"]):
        return f"The verdict is **INVESTIGATE** because the Challenge Engine detected contradictory evidence: {challenge.get('challenge_summary', 'conflicting operational signals')}. Operational validation is required before acting."

    if any(w in msg for w in ["root cause", "cause", "why", "reason", "driver", "primary"]):
        score_str = f" (confidence: {score:.2f})" if score is not None else ""
        return f"The primary root cause identified by the pipeline is **{kpi_label}**{score_str}."

    if any(w in msg for w in ["action", "recommend", "fix", "do", "remediation"]):
        return f"The recommended action is: **{action_title}** (Owner: {action.get('owner', 'Operations')})."

    if any(w in msg for w in ["confidence", "score", "certain"]):
        if score is not None:
            return f"The overall confidence score is **{score:.2f}**. {conf.get('explanation', '')}"
        return "Confidence score details are restricted for this persona."

    return (
        f"Investigation for {investigation.get('region_id', 'this region').replace('_',' ').title()} "
        f"returned verdict **{verdict}** with primary driver **{kpi_label}**. "
        f"Recommended action: {action_title}."
    )


# ─── Public API ───────────────────────────────────────────────────────────────

def chat_with_investigation(
    investigation: Dict[str, Any],
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    persona_id: str = "gm",
) -> Dict[str, Any]:
    """
    Answer a question about a completed investigation with dynamic RAG & Gemini LLM.

    Args:
        investigation: RBAC-filtered investigation dict
        message: current user message
        history: prior turns as [{"role": "user"|"assistant", "content": str}]
        persona_id: current persona ("gm" | "ops_lead" | "analyst")

    Returns:
        {
            "reply": str,
            "answer": str,
            "llm_used": bool,
            "model": str | None,
            "sources": List[str],
            "retrieval_method": str,
            "latency_ms": float,
            "note": str | None
        }
    """
    start_time = time.perf_counter()
    history = history or []
    region_id = investigation.get("region_id", "")

    # 1. Dynamic RAG Evidence Retrieval
    dynamic_rag = None
    sources = []
    retrieval_method = "none"
    try:
        # Formulate query using message + primary cause
        pc = investigation.get("primary_cause")
        pc_kpi = pc.get("kpi", "") if isinstance(pc, dict) else str(pc)
        rag_query = f"{message} {pc_kpi} {investigation.get('scenario','')}".strip()
        dynamic_rag = retrieve_evidence(query=rag_query, region_id=region_id, top_k=3)
        retrieval_method = dynamic_rag.get("retrieval_method", "none")
        sources = [r.get("id") for r in dynamic_rag.get("results", []) if r.get("id")]
    except Exception as e:
        print(f"[CHATBOT] Dynamic RAG error: {e}")

    system_prompt = _build_system_prompt(investigation, persona_id, dynamic_rag)

    # 2. Call Gemini 2.5 Flash
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if HAS_GEMINI and gemini_key:
        try:
            client = genai.Client(api_key=gemini_key)

            history_text = ""
            for turn in history[-6:]:
                role = "User" if turn.get("role") == "user" else "Assistant"
                history_text += f"\n{role}: {turn.get('content','')}"

            full_prompt = (
                f"{system_prompt}\n\n"
                f"Conversation History:{history_text}\n\n"
                f"User Question: {message}\n\nAssistant Response:"
            )

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=512,
                    temperature=0.3,
                ),
            )
            reply = response.text.strip()
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "reply": reply,
                "answer": reply,
                "llm_used": True,
                "model": GEMINI_MODEL,
                "sources": sources,
                "retrieval_method": retrieval_method,
                "latency_ms": latency_ms,
            }
        except Exception as e:
            print(f"[CHATBOT] Gemini failed: {e}. Trying Groq / Fallback...")

    # 3. Call Groq Llama3 Fallback
    if HAS_GROQ and os.getenv("GROQ_API_KEY"):
        try:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            messages = [{"role": "system", "content": system_prompt}]
            for turn in history[-6:]:
                messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
            messages.append({"role": "user", "content": message})
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=400,
                temperature=0.3,
            )
            reply = response.choices[0].message.content.strip()
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "reply": reply,
                "answer": reply,
                "llm_used": True,
                "model": GROQ_MODEL,
                "sources": sources,
                "retrieval_method": retrieval_method,
                "latency_ms": latency_ms,
            }
        except Exception as e:
            print(f"[CHATBOT] Groq failed: {e}. Using deterministic fallback.")

    # 4. Deterministic Fallback
    reply = _deterministic_fallback(message, investigation, persona_id)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    return {
        "reply": reply,
        "answer": reply,
        "llm_used": False,
        "model": None,
        "sources": sources,
        "retrieval_method": retrieval_method,
        "latency_ms": latency_ms,
        "note": "LLM key not configured or API limit hit — deterministic engine response used.",
    }

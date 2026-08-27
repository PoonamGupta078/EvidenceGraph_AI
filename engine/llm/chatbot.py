"""
llm/chatbot.py
Investigation-aware conversational chatbot for EvidenceGraph AI.

The chatbot is ONLY a Q&A layer on top of the investigation result.
It does NOT re-run the pipeline or re-determine the root cause.
All causal conclusions come from the pre-computed investigation dict.

The investigation dict passed in must already be RBAC-filtered for the
requesting persona — this module does NOT perform RBAC itself.

LLM priority:
  1. Google Gemini 2.5 Flash (via google-genai SDK)
  2. Groq Llama3 (legacy fallback)
  3. Deterministic keyword-match fallback
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

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
    "gm":       "Respond concisely in executive business language. Focus on impact and decisions.",
    "ops_lead": "Respond in direct operational language. Focus on what went wrong and what to fix.",
    "analyst":  "Respond analytically. Include confidence levels, evidence references, and caveats.",
}

# ─── System prompt builder ────────────────────────────────────────────────────

def _build_system_prompt(investigation: Dict[str, Any], persona_id: str) -> str:
    verdict      = investigation.get("verdict", "UNKNOWN")
    region       = investigation.get("region_id", "unknown").replace("_", " ").title()
    scenario     = investigation.get("scenario", "unknown").replace("_", " ")
    conf         = (investigation.get("confidence") or {})
    score        = conf.get("score", 0.0)
    explanation  = conf.get("explanation", "")

    pc = investigation.get("primary_cause") or {}
    primary_kpi   = pc.get("kpi", "").replace("_", " ")
    primary_label = pc.get("label", "Unknown")
    primary_conf  = pc.get("confidence", 0.0)

    # root causes (up to 3)
    rc_lines = []
    for rc in (investigation.get("root_causes") or [])[:3]:
        rc_lines.append(f"  - {rc.get('label','?')} (KPI: {rc.get('kpi','?')}, confidence: {rc.get('confidence',0):.2f})")
    rc_block = "\n".join(rc_lines) or "  - None available"

    # action
    action       = investigation.get("action") or {}
    action_title = action.get("title", "No action specified")
    action_desc  = action.get("description", "")
    action_owner = action.get("owner", "")

    # causal chain
    chain     = investigation.get("causal_chain") or []
    chain_str = " → ".join(chain) if chain else "Not available"

    # evidence summary
    ev = investigation.get("evidence_summary") or {}
    ev_for = "; ".join((ev.get("for") or [])[:3]) or "None"
    ev_against = "; ".join((ev.get("against") or [])[:2]) or "None"

    # PVM
    pvm = investigation.get("pvm_decomposition") or {}
    pvm_block = ""
    if pvm:
        comps = pvm.get("components") or {}
        pvm_parts = [f"{k}: ${v:,.0f}" for k, v in comps.items()]
        pvm_block = f"""
PVM Decomposition:
  Primary driver: {pvm.get('primary_driver', 'N/A')}
  Total change: ${pvm.get('total_change_usd', 0):,.0f}
  Components: {', '.join(pvm_parts) or 'N/A'}
"""

    # data quality
    dq = investigation.get("data_quality") or {}
    dq_block = f"Data quality score: {dq.get('quality_score', 'N/A')}, passes: {dq.get('passes', 'N/A')}"

    # challenge engine
    challenge = investigation.get("challenge_result") or {}
    challenge_block = challenge.get("challenge_summary", "No contradictions detected")

    # KPI health (material ones only)
    kpi_health = investigation.get("kpi_health") or {}
    material_kpis = [k for k, v in kpi_health.items() if isinstance(v, dict) and v.get("is_material")]

    # RAG evidence (top 2 snippets)
    rag = investigation.get("rag_evidence") or {}
    rag_results = (rag.get("results") or [])[:2]
    rag_block = ""
    if rag_results:
        rag_lines = [f"  [{r.get('score', 0):.2f}] {r.get('text', '')[:120]}" for r in rag_results]
        rag_block = "Retrieved evidence:\n" + "\n".join(rag_lines)

    persona_tone = _PERSONA_TONE.get(persona_id, _PERSONA_TONE["gm"])

    system = f"""You are an AI assistant embedded in EvidenceGraph AI — Accenture's autonomous anomaly investigation engine for e-commerce order fulfillment.

You are answering questions about a specific completed investigation. You must ONLY discuss what the investigation found. Do NOT invent causal relationships, do NOT override the pipeline's conclusions, do NOT speculate beyond the evidence.

══ INVESTIGATION CONTEXT ══════════════════════════════════════
Region:        {region}
Scenario:      {scenario}
Verdict:       {verdict}
Confidence:    {score:.2f} / 1.00
Explanation:   {explanation}

Primary Root Cause:
  KPI:         {primary_kpi}
  Label:       {primary_label}
  Confidence:  {primary_conf:.2f}

All Ranked Root Causes:
{rc_block}

Causal Chain: {chain_str}

Material KPIs: {', '.join(material_kpis) or 'None'}

Evidence FOR the conclusion: {ev_for}
Evidence AGAINST:            {ev_against}

Recommended Action:
  Title: {action_title}
  Description: {action_desc}
  Owner: {action_owner}

{pvm_block}
{rag_block}
Challenge Engine: {challenge_block}
{dq_block}
══ END CONTEXT ════════════════════════════════════════════════

Persona tone instruction: {persona_tone}

CRITICAL ANSWERING RULES:
1. DIRECTLY answer the exact question the user asked. Do not summarise the whole investigation — answer THAT question.
2. If asked "why" something happened, explain the mechanism using the causal chain and evidence above (e.g., "Marketing spend dropped → order cancellation rate rose → fulfillment delays followed → revenue fell"). Use the causal chain field.
3. If asked "why" a value is zero, explain what the investigation found about that component and what absence of that driver means.
4. If asked about a specific KPI or component, focus your answer entirely on that KPI — what its value was, what the evidence says about it, and what role it played.
5. Do NOT open your answer with "The investigation found..." or "Based on the investigation..." — start directly with the answer.
6. Interpret numbers in plain language — do not recite raw stats mechanically.
7. If the information is genuinely not in the context, say "The investigation does not have data on that" — do not guess.
8. Keep answers to 2-4 sentences. Only go longer if the question genuinely requires it.
"""
    return system


def _deterministic_fallback(message: str, investigation: Dict[str, Any]) -> str:
    """Simple keyword-based fallback when no LLM is configured."""
    msg = message.lower()
    verdict = investigation.get("verdict", "UNKNOWN")
    pc = investigation.get("primary_cause") or {}
    kpi_label = pc.get("label", "an identified issue")
    score = (investigation.get("confidence") or {}).get("score", 0.0)
    action = (investigation.get("action") or {}).get("title", "No action specified")

    if any(w in msg for w in ["root cause", "cause", "why", "reason", "driver"]):
        return f"The investigation identified **{kpi_label}** as the primary root cause driving the revenue anomaly (confidence: {score:.2f})."
    if any(w in msg for w in ["verdict", "decision", "act", "investigate", "abstain"]):
        return f"The investigation verdict is **{verdict}** with a confidence score of {score:.2f}."
    if any(w in msg for w in ["action", "recommend", "fix", "do"]):
        return f"The recommended action is: **{action}**."
    if any(w in msg for w in ["confidence", "score", "certain", "sure"]):
        return f"The confidence score is **{score:.2f}**. {(investigation.get('confidence') or {}).get('explanation', '')}"
    return (
        f"This investigation for {investigation.get('region_id', 'the region').replace('_',' ').title()} "
        f"returned verdict **{verdict}** (confidence {score:.2f}). "
        f"Primary driver: {kpi_label}. Recommended action: {action}."
    )


# ─── Public API ───────────────────────────────────────────────────────────────

def chat_with_investigation(
    investigation: Dict[str, Any],
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    persona_id: str = "gm",
) -> Dict[str, Any]:
    """
    Answer a question about a completed investigation.

    Args:
        investigation: RBAC-filtered investigation dict (already persona-filtered by caller)
        message: current user message
        history: prior turns as [{"role": "user"|"assistant", "content": str}]
        persona_id: current persona (for tone only — RBAC already applied)

    Returns:
        {"reply": str, "llm_used": bool, "model": str | None}
    """
    history = history or []
    system_prompt = _build_system_prompt(investigation, persona_id)

    # ── Gemini 2.5 Flash ─────────────────────────────────────────────────────
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if HAS_GEMINI and gemini_key:
        try:
            client = genai.Client(api_key=gemini_key)

            # Build conversation as flat text (Gemini non-chat API)
            history_text = ""
            for turn in history[-6:]:  # keep last 6 turns for context
                role = "User" if turn["role"] == "user" else "Assistant"
                history_text += f"\n{role}: {turn['content']}"

            full_prompt = (
                f"{system_prompt}\n\n"
                f"Conversation so far:{history_text}\n\n"
                f"User: {message}\n\nAssistant:"
            )

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=512,
                    temperature=0.4,
                ),
            )
            reply = response.text.strip()
            return {"reply": reply, "llm_used": True, "model": GEMINI_MODEL}
        except Exception as e:
            print(f"[CHATBOT] Gemini failed: {e}. Trying fallback…")

    # ── Groq Llama3 ──────────────────────────────────────────────────────────
    if HAS_GROQ and os.getenv("GROQ_API_KEY"):
        try:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            messages = [{"role": "system", "content": system_prompt}]
            for turn in history[-6:]:
                messages.append({"role": turn["role"], "content": turn["content"]})
            messages.append({"role": "user", "content": message})
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=400,
                temperature=0.4,
            )
            reply = response.choices[0].message.content.strip()
            return {"reply": reply, "llm_used": True, "model": GROQ_MODEL}
        except Exception as e:
            print(f"[CHATBOT] Groq failed: {e}. Using deterministic fallback.")

    # ── Deterministic fallback ────────────────────────────────────────────────
    reply = _deterministic_fallback(message, investigation)
    return {
        "reply": reply,
        "llm_used": False,
        "model": None,
        "note": "No LLM key configured — deterministic fallback used.",
    }

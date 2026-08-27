"""
llm/narrator.py
Gemini-powered narrative generation layer.

Generates persona-aware investigation narratives using Gemini 2.5 Flash.
The LLM is ONLY used for narrative framing — the confidence score,
verdict, and all quantitative outputs are computed by the non-LLM pipeline.

LLM boundary is explicit: every response includes llm_used: bool.
Caching: same (investigation_id + persona) → cached.

LLM Priority:
  1. Google Gemini 2.5 Flash (via google-genai SDK)
  2. Groq Llama3 (legacy fallback)
  3. Deterministic template (no LLM)
"""

from __future__ import annotations
import os
import json
import hashlib
from typing import Dict, Any, Optional

# ─── Gemini SDK ──────────────────────────────────────────────────────────────
try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# ─── Groq SDK (legacy fallback) ─────────────────────────────────────────────
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

# Simple in-memory cache
_cache: Dict[str, str] = {}

GEMINI_MODEL = "gemini-2.5-flash"
GROQ_MODEL = "llama3-8b-8192"

PERSONA_STYLES = {
    "gm": {
        "style": "executive",
        "instruction": "Write a concise executive summary (3-4 sentences). Focus on business impact and recommended action. Use business language, not technical jargon.",
    },
    "ops_lead": {
        "style": "operational",
        "instruction": "Write a practical operational brief (3-4 sentences). Focus on what went wrong operationally and what needs to happen. Be direct and action-oriented.",
    },
    "analyst": {
        "style": "analytical",
        "instruction": "Write a detailed analytical narrative (4-6 sentences). Include statistical observations, confidence sub-scores, and caveats about data quality.",
    },
}


def _build_cache_key(investigation_id: str, persona_id: str) -> str:
    return hashlib.md5(f"{investigation_id}:{persona_id}".encode()).hexdigest()


def _build_prompt(result: Dict[str, Any], persona_id: str) -> str:
    persona_style = PERSONA_STYLES.get(persona_id, PERSONA_STYLES["gm"])

    verdict = result.get("verdict", "UNKNOWN")
    confidence = result.get("confidence", {})
    score = confidence.get("score", 0.0)
    # Prefer pre-computed primary_cause (already excludes balancing items).
    # Fall back to root_causes[0] only when primary_cause is absent.
    primary_cause = result.get("primary_cause") or (result.get("root_causes") or [{}])[0]
    cause_label = primary_cause.get("label", "Unknown") if primary_cause else "Unknown"
    region = result.get("region_id", "Unknown Region").replace("_", " ").title()

    evidence_for = result.get("evidence_summary", {}).get("for", [])
    evidence_against = result.get("evidence_summary", {}).get("against", [])

    action = result.get("action", {})
    action_title = action.get("title", "No action specified")

    challenges = result.get("challenge_result", {}).get("challenge_summary", "")

    # PVM context if available
    pvm = result.get("pvm_decomposition")
    pvm_context = ""
    if pvm and pvm.get("status") == "OK":
        pvm_context = f"\n- PVM Decomposition: Primary driver = {pvm.get('primary_driver', 'N/A')}, Total change = ${pvm.get('total_change_usd', 0):,.0f}"

    prompt = f"""You are an AI investigation narrator for the EvidenceGraph AI engine — a business intelligence system used by Accenture for e-commerce order fulfillment analytics.

Investigation results for {region}:
- Verdict: {verdict}
- Confidence Score: {score:.2f}/1.00
- Primary Root Cause: {cause_label}
- Action Recommendation: {action_title}
- Evidence FOR the root cause: {'; '.join(evidence_for[:3]) if evidence_for else 'None'}
- Evidence AGAINST: {'; '.join(evidence_against[:2]) if evidence_against else 'None'}
- Challenge findings: {challenges or 'No contradictions detected'}{pvm_context}

Style instruction: {persona_style['instruction']}

Write the narrative now. Do not use bullet points. Do not repeat the raw numbers verbatim — interpret them. Be professional and authoritative."""

    return prompt


def generate_narrative(
    investigation_result: Dict[str, Any],
    persona_id: str = "gm",
    investigation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates a persona-aware narrative for an investigation result.

    LLM priority:
      1. Gemini 2.5 Flash (GOOGLE_API_KEY or GEMINI_API_KEY)
      2. Groq Llama3 (GROQ_API_KEY)
      3. Deterministic fallback

    Args:
        investigation_result: full (or persona-filtered) investigation result
        persona_id: "gm" | "ops_lead" | "analyst"
        investigation_id: optional id for caching

    Returns:
        - narrative: str
        - llm_used: bool (always explicit)
        - model: str | None
        - cached: bool
        - persona_id: str
    """

    # ─── Check cache first ───────────────────────────────────────────────
    cache_key = _build_cache_key(investigation_id or "anon", persona_id)
    if cache_key in _cache:
        return {
            "narrative": _cache[cache_key],
            "llm_used": True,
            "model": "cached",
            "cached": True,
            "persona_id": persona_id,
        }

    prompt = _build_prompt(investigation_result, persona_id)

    # ─── Try Gemini 2.5 Flash first ─────────────────────────────────────
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if HAS_GEMINI and gemini_key:
        try:
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=400,
                    temperature=0.3,
                ),
            )

            narrative = response.text.strip()
            _cache[cache_key] = narrative

            # Extract token usage if available
            tokens_used = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                tokens_used = {
                    "prompt": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "completion": getattr(response.usage_metadata, "candidates_token_count", 0),
                    "total": getattr(response.usage_metadata, "total_token_count", 0),
                }

            return {
                "narrative": narrative,
                "llm_used": True,
                "model": GEMINI_MODEL,
                "cached": False,
                "persona_id": persona_id,
                "tokens_used": tokens_used,
            }
        except Exception as e:
            # Fall through to Groq or deterministic
            print(f"[WARNING] Gemini narrative failed: {e}. Trying fallback...")

    # ─── Fallback: Groq Llama3 ──────────────────────────────────────────
    if HAS_GROQ and os.getenv("GROQ_API_KEY"):
        try:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
            )

            narrative = response.choices[0].message.content.strip()
            _cache[cache_key] = narrative

            return {
                "narrative": narrative,
                "llm_used": True,
                "model": GROQ_MODEL,
                "cached": False,
                "persona_id": persona_id,
                "tokens_used": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens,
                },
            }
        except Exception as e:
            print(f"[WARNING] Groq narrative failed: {e}. Using deterministic fallback.")

    # ─── Deterministic fallback (no LLM) ────────────────────────────────
    verdict = investigation_result.get("verdict", "UNKNOWN")
    confidence = investigation_result.get("confidence", {})
    score = confidence.get("score", 0.0) if isinstance(confidence, dict) else 0.0
    primary_cause = (investigation_result.get("root_causes") or [{}])
    cause = primary_cause[0].get("label", "an identified KPI") if primary_cause else "an identified KPI"

    fallback = {
        "ACT": f"The investigation has reached high confidence ({score:.2f}). {cause} is the confirmed root cause. Immediate action is recommended per the action plan.",
        "INVESTIGATE": f"The investigation reached moderate confidence ({score:.2f}). Contradictory signals were detected — {cause} is the likely driver but further investigation is needed before committing to action.",
        "ABSTAIN": f"Insufficient data quality or history ({score:.2f} confidence). The system recommends holding action until data gaps are resolved.",
    }
    narrative = fallback.get(verdict, f"Investigation complete. Verdict: {verdict}. Confidence: {score:.2f}.")
    return {
        "narrative": narrative,
        "llm_used": False,
        "model": None,
        "cached": False,
        "persona_id": persona_id,
        "note": "No LLM API key configured — deterministic fallback narrative used. Set GOOGLE_API_KEY for Gemini 2.5 Flash or GROQ_API_KEY for Groq Llama3.",
    }

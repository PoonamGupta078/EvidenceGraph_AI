// lib/api.ts — Typed API client for the EvidenceGraph engine

const ENGINE = process.env.NEXT_PUBLIC_ENGINE_URL || "http://localhost:8000";

export type Verdict = "ACT" | "INVESTIGATE" | "ABSTAIN";
export type PersonaId = "gm" | "ops_lead" | "analyst";

export interface KPIData {
  current_value: number | null;
  prior_value: number | null;
  pct_change_7d: number | null;
  is_material: boolean;
  trend: "up" | "down" | "flat";
}

export interface RegionOverview {
  label: string;
  scenario: string;
  kpi_health: Record<string, KPIData>;
  any_material: boolean;
  data_days: number;
  error?: string;
}

export interface InvestigationResult {
  investigation_id: string;
  region_id: string;
  scenario: string;
  timestamp: string;
  verdict: Verdict;
  confidence?: {
    score?: number;
    verdict: Verdict;
    explanation: string;
    sub_scores?: Record<string, number>;
  };
  action?: {
    action_id?: string;
    type: string;
    title: string;
    description: string;
    owner: string;
    priority: string;
    estimated_impact?: string;
  };
  kpi_health?: Record<string, KPIData>;
  material_kpis?: string[];
  root_causes?: Array<{
    kpi: string;
    label: string;
    is_material: boolean;
    confidence: number;
    evidence_for: string[];
    evidence_against: string[];
    effect_size?: number;
    change_day?: number;
  }>;
  primary_cause?: {
    kpi: string;
    label: string;
    is_material: boolean;
    confidence: number;
    evidence_for: string[];
    evidence_against: string[];
    effect_size?: number;
    change_day?: number;
  };
  causal_chain?: string[];
  evidence_summary?: { for: string[]; against: string[] };
  challenge_result?: {
    challenges: Array<{ type: string; description: string; severity: string }>;
    challenge_count: number;
    has_contradictions: boolean;
    challenge_summary: string;
  };
  evidence_graph?: {
    nodes: Array<{ id: string; label: string; material: boolean; centrality: number }>;
    links: Array<{ source: string; target: string; type: string; weight: number; lag_days: number }>;
    driver_ranking: Array<{ kpi: string; score: number; is_material: boolean }>;
  };
  pvm_decomposition?: {
    components: Record<string, number>;
    total_change_usd: number;
    baseline_revenue: number;
    current_revenue: number;
    waterfall_data: Array<{ label: string; value: number; running_total: number; type: string }>;
    primary_driver: string;
  };
  narrative?: {
    narrative: string;
    llm_used: boolean;
    model?: string;
    cached?: boolean;
  };
  telemetry?: {
    latency: { pipeline_ms: number; llm_ms: number; rag_ms: number; total_ms: number };
    tokens?: { prompt: number; completion: number; total: number };
    estimated_cost_usd: number;
    llm_used: boolean;
  };
  rag_evidence?: {
    results: Array<{ id: string; text: string; region: string; category: string; score: number }>;
    retrieval_method: string;
  };
  persona_context?: {
    persona_id: string;
    label: string;
    color: string;
    restricted_kpis_hidden: string[];
  };
  data_quality?: {
    passes: boolean;
    quality_score: number;
    gate_results: Record<string, { passed: boolean; reason: string }>;
  };
  raw_sub_scores?: Record<string, number>;
  abstain_reason?: string;
  calendar_check?: {
    is_likely_calendar_artifact: boolean;
    calendar_findings: Array<{ type: string; finding: string; impact: string }>;
    recommendation: string;
  };
}

export interface Persona {
  id: string;
  label: string;
  description: string;
  color: string;
  icon: string;
}

export interface SandboxResult {
  lever: string;
  lever_value: number;
  simulated_outcomes: Record<string, { baseline: number; simulated: number; delta: number; delta_pct: number }>;
  revenue_recovery_estimate: { low: number; mid: number; high: number };
  model_confidence: number;
}

export interface Lever {
  id: string;
  label: string;
  min: number;
  max: number;
  default_recovery: number;
  unit: string;
}

// ─── Fetch helpers ─────────────────────────────────────────────────────────

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ─── API Functions ──────────────────────────────────────────────────────────

export const api = {
  health: () => fetchJSON<{ status: string; regions_available: string[] }>(`${ENGINE}/health`),

  kpis: () => fetchJSON<{ regions: Record<string, RegionOverview> }>(`${ENGINE}/kpis`),

  personas: () => fetchJSON<{ personas: Persona[] }>(`${ENGINE}/personas`),

  runInvestigation: (region_id: string, persona_id: PersonaId, include_narrative = true) =>
    fetchJSON<InvestigationResult>(`${ENGINE}/investigations/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ region_id, persona_id, include_narrative, include_rag: true }),
    }),

  getInvestigation: (id: string, persona_id: PersonaId) =>
    fetchJSON<InvestigationResult>(`${ENGINE}/investigations/${id}?persona_id=${persona_id}`),

  simulate: (region_id: string, lever: string, lever_value: number) =>
    fetchJSON<SandboxResult>(`${ENGINE}/sandbox/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ region_id, lever, lever_value }),
    }),

  levers: () => fetchJSON<{ levers: Lever[] }>(`${ENGINE}/sandbox/levers`),

  submitFeedback: (data: {
    investigation_id: string;
    region_id: string;
    persona_id: string;
    verdict: string;
    user_verdict?: string;
    driver_selected?: string;
    rating: string;
    comment?: string;
  }) =>
    fetchJSON(`${ENGINE}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    }),

  feedbackStats: () => fetchJSON(`${ENGINE}/feedback/stats`),

  telemetry: (investigation_id: string) => fetchJSON(`${ENGINE}/telemetry/${investigation_id}`),

  chat: (
    investigation_id: string,
    message: string,
    persona_id: PersonaId,
    history: Array<{ role: "user" | "assistant"; content: string }> = []
  ) =>
    fetchJSON<{ reply: string; llm_used: boolean; model: string | null; note?: string }>(
      `${ENGINE}/investigations/${investigation_id}/chat`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, persona_id, history }),
      }
    ),
};

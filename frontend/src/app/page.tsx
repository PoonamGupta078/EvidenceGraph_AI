"use client";

import React, { useState, useEffect, useCallback } from "react";
import { api, RegionOverview, InvestigationResult, PersonaId, Persona } from "@/lib/api";
import PersonaSwitcher from "@/components/PersonaSwitcher";
import KPIHealthStrip from "@/components/KPIHealthStrip";
import EvidenceGraph from "@/components/EvidenceGraph";
import ActionCard from "@/components/ActionCard";
import InterventionSandbox from "@/components/InterventionSandbox";
import InvestigationChat from "@/components/InvestigationChat";

/* ── Icons ─────────────────────────────────────────────────────────────────── */
const IconBrain = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.46 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"/>
    <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.46 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"/>
  </svg>
);
const IconZap = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
  </svg>
);
const IconShield = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
  </svg>
);
const IconChevronRight = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="9 18 15 12 9 6"/>
  </svg>
);
const IconClock = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
  </svg>
);
const IconRefresh = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/>
    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
  </svg>
);
const IconWarning = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
    <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
  </svg>
);

/* ── Loading Overlay ──────────────────────────────────────────────────────── */
function LoadingOverlay() {
  const [stage, setStage] = useState(0);
  const stages = [
    "Reconciling data sources…",
    "Detecting anomalies (CUSUM + STL)…",
    "Building typed relationship graph…",
    "Running PVM decomposition…",
    "Computing confidence gate…",
    "Generating narrative…",
  ];
  useEffect(() => {
    const t = setInterval(() => setStage(s => (s + 1) % stages.length), 1200);
    return () => clearInterval(t);
  }, []);
  return (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", height: 520, gap: 28,
      background: "var(--bg-card)", borderRadius: "var(--radius-lg)",
      border: "1px solid var(--border)",
      position: "relative", overflow: "hidden",
    }}>
      {/* Animated glow */}
      <div style={{
        position: "absolute", width: 300, height: 300, borderRadius: "50%",
        background: "radial-gradient(circle, rgba(161,0,255,0.08) 0%, transparent 70%)",
        animation: "pulse-ring 2s ease-out infinite",
        pointerEvents: "none",
      }}/>
      <div style={{ position: "relative", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <div className="spinner" style={{ width: 48, height: 48, borderWidth: 3 }}/>
        <div style={{
          position: "absolute", width: 20, height: 20, borderRadius: "50%",
          background: "linear-gradient(135deg, var(--accenture-purple), var(--accenture-purple-dark))",
          boxShadow: "0 0 20px rgba(161,0,255,0.5)",
        }}/>
      </div>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--text-primary)", marginBottom: 8 }}>
          Running Investigation
        </div>
        <div style={{
          fontSize: "0.82rem", color: "var(--text-secondary)",
          fontFamily: "var(--font-mono)", minHeight: 20,
          transition: "all 0.3s ease",
        }}>
          {stages[stage]}
        </div>
      </div>
      {/* Step dots */}
      <div style={{ display: "flex", gap: 6 }}>
        {stages.map((_, i) => (
          <div key={i} style={{
            width: i === stage ? 20 : 6, height: 6, borderRadius: 100,
            background: i === stage ? "var(--accenture-purple)" : "var(--border)",
            transition: "all 0.4s cubic-bezier(0.16,1,0.3,1)",
          }}/>
        ))}
      </div>
    </div>
  );
}

/* ── Confidence Sub-Scores Panel ──────────────────────────────────────────── */
function SubScoresPanel({ scores }: { scores: Record<string, number> }) {
  const labels: Record<string, string> = {
    data_quality: "Data Quality",
    signal_strength: "Signal Strength",
    cross_source_consistency: "Cross-Source",
    evidence_depth: "Evidence Depth",
    causal_chain_integrity: "Causal Chain",
  };
  return (
    <div style={{
      width: 220, background: "var(--bg-elevated)",
      padding: 18, borderRadius: "var(--radius-md)",
      border: "1px solid var(--border)", flexShrink: 0,
    }}>
      <h4 style={{ marginBottom: 14 }}>Confidence Sub-Scores</h4>
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {Object.entries(scores).map(([k, v]) => (
          <div key={k}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
              <span style={{ fontSize: "0.72rem", color: "var(--text-secondary)" }}>
                {labels[k] || k.replace(/_/g, " ")}
              </span>
              <span style={{ fontSize: "0.72rem", fontWeight: 700, fontFamily: "var(--font-mono)", color: v >= 0.7 ? "var(--act)" : v >= 0.4 ? "var(--investigate)" : "var(--abstain)" }}>
                {(v * 100).toFixed(0)}%
              </span>
            </div>
            <div className="subscore-track">
              <div className="subscore-fill" style={{ width: `${(v * 100).toFixed(0)}%` }}/>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── PVM Waterfall ────────────────────────────────────────────────────────── */
function PVMWaterfall({ pvm }: { pvm: NonNullable<InvestigationResult["pvm_decomposition"]> }) {
  const comps = Object.entries(pvm.components || {});
  const maxAbs = Math.max(...comps.map(([, v]) => Math.abs(v)), 1);
  const labels: Record<string, string> = {
    price: "Price Effect", volume: "Volume (Residual)",
    marketing: "Marketing Lift", seasonal: "Seasonality",
  };
  const colors: Record<string, string> = {
    price: "#818cf8", volume: "var(--text-muted)",
    marketing: "var(--accenture-purple-light)", seasonal: "var(--investigate)",
  };

  return (
    <div style={{ padding: "4px 0" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 14, flexWrap: "wrap", gap: 8 }}>
        <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)" }}>
          Baseline: <strong style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}>
            ${(pvm.baseline_revenue / 1000).toFixed(0)}K
          </strong>
        </div>
        <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)" }}>
          Current: <strong style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}>
            ${(pvm.current_revenue / 1000).toFixed(0)}K
          </strong>
        </div>
        <div style={{ fontSize: "0.78rem", color: pvm.total_change_usd < 0 ? "var(--abstain)" : "var(--act)" }}>
          Δ <strong style={{ fontFamily: "var(--font-mono)" }}>
            {pvm.total_change_usd < 0 ? "-" : "+"}${(Math.abs(pvm.total_change_usd) / 1000).toFixed(1)}K
          </strong>
        </div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {comps.map(([key, val]) => {
          const pct = Math.abs(val) / maxAbs;
          const isPositive = val > 0;
          return (
            <div key={key}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <span style={{ fontSize: "0.72rem", color: "var(--text-secondary)" }}>{labels[key] || key}</span>
                <span style={{
                  fontSize: "0.72rem", fontWeight: 700, fontFamily: "var(--font-mono)",
                  color: key === "volume" ? "var(--text-muted)" : isPositive ? "var(--act)" : "var(--abstain)",
                }}>
                  {isPositive ? "+" : ""}${(val / 1000).toFixed(1)}K
                </span>
              </div>
              <div style={{ height: 8, background: "var(--bg-elevated)", borderRadius: 4, overflow: "hidden" }}>
                <div style={{
                  height: "100%", width: `${(pct * 100).toFixed(0)}%`,
                  background: colors[key] || "var(--accenture-purple)",
                  borderRadius: 4, opacity: key === "volume" ? 0.4 : 1,
                  transition: "width 1s cubic-bezier(0.16,1,0.3,1)",
                  boxShadow: key !== "volume" ? `0 0 8px ${colors[key] || "var(--accenture-purple)"}60` : "none",
                }}/>
              </div>
            </div>
          );
        })}
      </div>
      {pvm.primary_driver && (
        <div style={{ marginTop: 12, padding: "8px 12px", background: "rgba(161,0,255,0.06)", borderRadius: 6, border: "1px solid rgba(161,0,255,0.15)" }}>
          <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>Primary driver: </span>
          <span style={{ fontSize: "0.72rem", fontWeight: 700, color: "var(--accenture-purple-light)", fontFamily: "var(--font-mono)" }}>
            {pvm.primary_driver}
          </span>
        </div>
      )}
    </div>
  );
}

/* ── RAG Evidence Panel ───────────────────────────────────────────────────── */
function RAGPanel({ rag }: { rag: NonNullable<InvestigationResult["rag_evidence"]> }) {
  if (!rag.results?.length) return null;
  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <h4>Supporting Evidence</h4>
        <span className="chip chip-purple" style={{ textTransform: "none", letterSpacing: 0 }}>
          RAG · {rag.retrieval_method}
        </span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {rag.results.slice(0, 3).map(r => (
          <div key={r.id} style={{
            fontSize: "0.78rem", color: "var(--text-secondary)",
            padding: "10px 14px", borderRadius: 8,
            background: "var(--bg-elevated)",
            border: "1px solid var(--border)",
            borderLeft: `3px solid ${r.score > 0.5 ? "var(--act)" : "var(--border-active)"}`,
            fontStyle: "italic", lineHeight: 1.5,
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
              <span style={{ fontSize: "0.64rem", fontStyle: "normal", fontWeight: 600, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                {r.id} · {r.category} · {r.region}
              </span>
              <span style={{ fontSize: "0.64rem", fontStyle: "normal", color: r.score > 0.5 ? "var(--act)" : "var(--investigate)", fontFamily: "var(--font-mono)", fontWeight: 600 }}>
                {(r.score * 100).toFixed(0)}% match
              </span>
            </div>
            "{r.text}"
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── Temporal Sequence Panel ──────────────────────────────────────────────── */
function TemporalSequence({ chain }: { chain: string[] }) {
  return (
    <div className="card card-sm">
      <h4 style={{ marginBottom: 14 }}>Temporal Sequence</h4>
      <div className="stepper">
        {chain.map((kpi, i) => (
          <div key={kpi} className="step">
            <div className="step-indicator">
              <div className={`step-dot ${i === 0 ? "active" : "complete"}`}>{i + 1}</div>
              {i < chain.length - 1 && <div className="step-line"/>}
            </div>
            <div className="step-content" style={{ paddingBottom: i === chain.length - 1 ? 0 : 18 }}>
              <div className="step-label">{kpi.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}</div>
              {i === 0 && <div className="step-detail">↳ First anomalous change detected</div>}
              {i === chain.length - 1 && kpi === "revenue" && (
                <div className="step-detail" style={{ color: "var(--abstain)" }}>↳ Revenue impact endpoint</div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── Main Page ────────────────────────────────────────────────────────────── */
export default function Home() {
  const [activePersona, setActivePersona] = useState<PersonaId>("gm");
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [regions, setRegions] = useState<Record<string, RegionOverview>>({});
  const [selectedRegion, setSelectedRegion] = useState<string>("region_a");
  const [investigation, setInvestigation] = useState<InvestigationResult | null>(null);
  const [investigationId, setInvestigationId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [engineOnline, setEngineOnline] = useState<boolean | null>(null);
  const [currentTime, setCurrentTime] = useState<string>("");

  // Live clock ticker
  useEffect(() => {
    setCurrentTime(new Date().toLocaleTimeString());
    const timer = setInterval(() => setCurrentTime(new Date().toLocaleTimeString()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Health check
  useEffect(() => {
    api.health()
      .then(() => setEngineOnline(true))
      .catch(() => setEngineOnline(false));
  }, []);

  // Load personas + regions on mount
  useEffect(() => {
    Promise.all([api.personas(), api.kpis()]).then(([pRes, kRes]) => {
      setPersonas(pRes.personas);
      setRegions(kRes.regions);
    }).catch(console.error);
  }, []);

  // POST new investigation only when the region changes
  const runInvestigation = useCallback(() => {
    if (!selectedRegion) return;
    setLoading(true);
    setInvestigation(null);
    setInvestigationId(null);
    api.runInvestigation(selectedRegion, activePersona, true)
      .then(res => {
        setInvestigation(res);
        setInvestigationId(res.investigation_id ?? null);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [selectedRegion]); // NOTE: activePersona intentionally omitted — re-run only on region change

  // On region change: run fresh investigation
  useEffect(() => { runInvestigation(); }, [runInvestigation]);

  // On persona change: GET the persona-filtered view of the same investigation
  useEffect(() => {
    if (!investigationId) return; // no investigation yet — the POST will include the correct persona
    setLoading(true);
    api.getInvestigation(investigationId, activePersona)
      .then(res => setInvestigation(res))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [activePersona]); // eslint-disable-line react-hooks/exhaustive-deps

  const verdictColor = investigation?.verdict === "ACT"
    ? "var(--act)" : investigation?.verdict === "INVESTIGATE"
    ? "var(--investigate)" : "var(--abstain)";

  return (
    <div className="page" style={{ paddingBottom: 80 }}>

      {/* ── Navigation ── */}
      <nav className="nav" style={{ justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
          {/* Brand */}
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{
              width: 34, height: 34,
              background: "linear-gradient(135deg, var(--accenture-purple), var(--accenture-purple-dark))",
              borderRadius: 8,
              display: "flex", alignItems: "center", justifyContent: "center",
              boxShadow: "0 0 16px rgba(161,0,255,0.3)",
            }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="white">
                <polygon points="6,4 18,12 6,20" fill="white"/>
              </svg>
            </div>
            <div>
              <div style={{ fontSize: "1rem", fontWeight: 800, color: "var(--text-primary)", letterSpacing: "-0.025em" }}>
                EvidenceGraph<span style={{ color: "var(--accenture-purple)", fontWeight: 900 }}> AI</span>
              </div>
              <div style={{ fontSize: "0.6rem", color: "var(--text-muted)", fontWeight: 500, marginTop: -1, letterSpacing: "0.04em", textTransform: "uppercase" }}>
                Accenture Innovation Challenge 2026 · HerForge
              </div>
            </div>
          </div>

          <div style={{ width: 1, height: 26, background: "var(--border)" }}/>

          {/* Engine Status */}
          <div style={{
            display: "flex", alignItems: "center", gap: 6,
            padding: "4px 10px", borderRadius: 100,
            background: engineOnline === false ? "var(--abstain-bg)" : "var(--act-bg)",
            border: `1px solid ${engineOnline === false ? "var(--abstain-border)" : "var(--act-border)"}`,
          }}>
            <div style={{
              width: 6, height: 6, borderRadius: "50%",
              background: engineOnline === false ? "var(--abstain)" : "var(--act)",
              boxShadow: `0 0 6px ${engineOnline === false ? "var(--abstain)" : "var(--act)"}`,
              animation: engineOnline ? "glow-pulse 2s ease infinite" : "none",
            }}/>
            <span style={{ fontSize: "0.68rem", fontWeight: 600, color: engineOnline === false ? "var(--abstain)" : "var(--act)" }}>
              {engineOnline === null ? "Connecting…" : engineOnline ? "Engine Online" : "Engine Offline"}
            </span>
          </div>
        </div>

        {/* Right: Persona Switcher */}
        <PersonaSwitcher personas={personas} activePersona={activePersona} onChange={setActivePersona}/>
      </nav>

      {/* ── Page Header ── */}
      <div style={{
        borderBottom: "1px solid var(--border)",
        padding: "14px 28px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        background: "rgba(245,246,250,0.95)",
        backdropFilter: "blur(8px)",
      }}>
        <div>
          <h1 style={{ fontSize: "1.25rem", fontWeight: 800, marginBottom: 2, display: "flex", alignItems: "center", gap: 8 }}>
            <IconBrain/> Anomaly Investigation Dashboard
          </h1>
          <p style={{ fontSize: "0.75rem", margin: 0, color: "var(--text-muted)" }}>
            E-commerce order fulfillment · Revenue impact engine · Track 3: BusinessIntelligence.ai
          </p>
        </div>
        {investigation && !loading && (
          <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
            <div style={{ display: "flex", gap: 20, fontSize: "0.76rem", color: "var(--text-muted)" }}>
              {[
                { label: "Region", val: investigation.region_id?.replace(/_/g, " ").toUpperCase() },
                { label: "Scenario", val: investigation.scenario?.replace(/_/g, " ") },
                { label: "Live Clock", val: currentTime || "—" },
                { label: "Run Time", val: investigation.timestamp ? new Date(investigation.timestamp).toLocaleTimeString() : "—" },
              ].map(({ label, val }) => (
                <div key={label}>
                  <div style={{ fontWeight: 700, color: "var(--text-primary)", marginBottom: 1, fontSize: "0.75rem" }}>{label}</div>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.72rem", color: label === "Live Clock" ? "var(--accenture-purple)" : "var(--text-secondary)", fontWeight: 600 }}>{val}</div>
                </div>
              ))}
            </div>
            <button className="btn btn-secondary" style={{ padding: "7px 14px", fontSize: "0.75rem" }} onClick={runInvestigation}>
              <IconRefresh/> Re-run
            </button>
          </div>
        )}
      </div>

      {/* ── Main Layout ── */}
      <div className="container" style={{ paddingTop: 24, display: "grid", gridTemplateColumns: "300px 1fr", gap: 24, alignItems: "start" }}>

        {/* ── Left Sidebar ── */}
        <aside style={{ position: "sticky", top: 88 }}>
          <KPIHealthStrip regions={regions} selectedRegion={selectedRegion} onSelect={setSelectedRegion}/>
        </aside>

        {/* ── Main Workspace ── */}
        <section style={{ display: "flex", flexDirection: "column", gap: 20, minWidth: 0 }}>

          {loading ? (
            <LoadingOverlay/>
          ) : investigation ? (
            <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: 20 }}>

              {/* ── Engine Offline Banner ── */}
              {engineOnline === false && (
                <div style={{
                  padding: "12px 16px", borderRadius: "var(--radius-md)",
                  background: "var(--abstain-bg)", border: "1px solid var(--abstain-border)",
                  display: "flex", alignItems: "center", gap: 10,
                  fontSize: "0.82rem", color: "var(--abstain)",
                }}>
                  <IconWarning/>
                  <span>Backend engine is offline. Ensure FastAPI is running on <code style={{ fontFamily: "var(--font-mono)" }}>localhost:8000</code>.</span>
                </div>
              )}

              {/* ── Verdict Hero Card ── */}
              <div className="verdict-hero">
                {/* Animated top border glow based on verdict */}
                <div style={{
                  position: "absolute", top: 0, left: 0, right: 0, height: 2,
                  background: `linear-gradient(90deg, ${verdictColor}, ${verdictColor}80, transparent)`,
                }}/>

                <div style={{ display: "flex", alignItems: "flex-start", gap: 24, flexWrap: "wrap" }}>
                  <div style={{ flex: 1, minWidth: 280 }}>
                    {/* Verdict + Confidence */}
                    <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20, flexWrap: "wrap" }}>
                      <span className={`verdict-badge verdict-${investigation.verdict}`}>
                        {investigation.verdict === "ACT" ? "✓" : investigation.verdict === "INVESTIGATE" ? "⚠" : "✕"}
                        {" "}{investigation.verdict}
                      </span>
                      {investigation.confidence?.score !== undefined && (
                        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                          <div style={{
                            height: 6, width: 120,
                            background: "var(--bg-elevated)",
                            borderRadius: 100, overflow: "hidden",
                            border: "1px solid var(--border-subtle)",
                          }}>
                            <div style={{
                              height: "100%",
                              width: `${(investigation.confidence.score * 100).toFixed(0)}%`,
                              background: `linear-gradient(90deg, ${verdictColor}80, ${verdictColor})`,
                              borderRadius: 100, transition: "width 0.8s ease",
                              boxShadow: `0 0 8px ${verdictColor}60`,
                            }}/>
                          </div>
                          <span style={{ fontSize: "0.82rem", fontWeight: 700, color: verdictColor, fontFamily: "var(--font-mono)" }}>
                            {(investigation.confidence.score * 100).toFixed(0)}%
                          </span>
                        </div>
                      )}
                      {investigation.persona_context && (
                        <span className="chip chip-purple">
                          {investigation.persona_context.label}
                        </span>
                      )}
                    </div>

                    {/* Narrative */}
                    <p style={{ fontSize: "0.95rem", color: "var(--text-primary)", lineHeight: 1.75, fontWeight: 400, margin: 0 }}>
                      {investigation.narrative?.narrative || investigation.confidence?.explanation || "Investigation complete."}
                    </p>

                    {/* Primary cause chip */}
                    {investigation.primary_cause && (
                      <div style={{ marginTop: 14, display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                        <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Primary Driver:</span>
                        <span className="chip chip-amber" style={{ fontSize: "0.72rem" }}>
                          <IconZap/>
                          {(investigation.primary_cause as any).kpi?.replace(/_/g, " ")}
                        </span>
                        {investigation.material_kpis?.map(k => (
                          <span key={k} className="chip" style={{ fontSize: "0.68rem" }}>{k.replace(/_/g, " ")}</span>
                        ))}
                      </div>
                    )}

                    {/* Challenge Engine Alert */}
                    {investigation.challenge_result?.has_contradictions && (
                      <div style={{
                        marginTop: 16, padding: "12px 16px",
                        background: "var(--investigate-bg)",
                        borderLeft: "3px solid var(--investigate)",
                        borderRadius: "0 8px 8px 0",
                        fontSize: "0.82rem", color: "var(--text-primary)",
                        display: "flex", gap: 8, alignItems: "flex-start",
                      }}>
                        <IconWarning/>
                        <div>
                          <span style={{ fontWeight: 700, color: "var(--investigate)" }}>Challenge Engine: </span>
                          {investigation.challenge_result.challenge_summary}
                        </div>
                      </div>
                    )}

                    {/* Calendar check */}
                    {investigation.calendar_check?.is_likely_calendar_artifact && (
                      <div style={{
                        marginTop: 12, padding: "10px 14px",
                        background: "rgba(129,140,248,0.08)",
                        borderLeft: "3px solid #818cf8",
                        borderRadius: "0 8px 8px 0",
                        fontSize: "0.78rem", color: "var(--text-secondary)",
                      }}>
                        <span style={{ fontWeight: 700, color: "#818cf8" }}>Calendar Effect: </span>
                        {investigation.calendar_check.recommendation}
                      </div>
                    )}

                    {/* RAG Evidence */}
                    {investigation.rag_evidence && <RAGPanel rag={investigation.rag_evidence}/>}
                  </div>

                  {/* Sub-scores Panel */}
                  {investigation.raw_sub_scores && Object.keys(investigation.raw_sub_scores).length > 0 && (
                    <SubScoresPanel scores={investigation.raw_sub_scores as Record<string, number>}/>
                  )}
                </div>
              </div>

              {/* ── Two Column: Graph + Action/Chain ── */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>

                {/* Evidence Graph */}
                {investigation.evidence_graph && (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
                      <h3>Evidence Graph</h3>
                      <span className="chip chip-purple">Typed Relationship Graph</span>
                    </div>
                    <EvidenceGraph
                      nodes={investigation.evidence_graph.nodes}
                      links={investigation.evidence_graph.links}
                      driverRanking={investigation.evidence_graph.driver_ranking}
                    />
                  </div>
                )}

                {/* Right column */}
                <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
                  {investigation.action && (
                    <ActionCard action={investigation.action} verdict={investigation.verdict}/>
                  )}

                  {/* Temporal Sequence */}
                  {investigation.causal_chain && investigation.causal_chain.length > 0 && (
                    <TemporalSequence chain={investigation.causal_chain}/>
                  )}
                </div>
              </div>

              {/* ── PVM Decomposition + Data Quality ── */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>

                {/* PVM */}
                {investigation.pvm_decomposition && (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
                      <h3>PVM Decomposition</h3>
                      <span className="chip chip-purple">accounting_closure</span>
                    </div>
                    <PVMWaterfall pvm={investigation.pvm_decomposition}/>
                  </div>
                )}

                {/* Data Quality */}
                {investigation.data_quality && (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
                      <h3>Data Quality</h3>
                      <span className={`chip ${investigation.data_quality.passes ? "chip-green" : "chip-red"}`}>
                        <IconShield/>
                        {investigation.data_quality.passes ? "Passed" : "Failed"}
                      </span>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                      {Object.entries(investigation.data_quality.gate_results || {}).map(([gate, res]) => (
                        <div key={gate} style={{
                          padding: "10px 14px", borderRadius: 8,
                          background: "var(--bg-elevated)",
                          border: `1px solid ${res.passed ? "var(--act-border)" : "var(--abstain-border)"}`,
                          display: "flex", alignItems: "flex-start", gap: 10,
                        }}>
                          <span style={{ fontSize: "0.9rem", marginTop: -2 }}>{res.passed ? "✅" : "❌"}</span>
                          <div>
                            <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--text-primary)", marginBottom: 2 }}>
                              {gate.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}
                            </div>
                            <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)" }}>{res.reason}</div>
                          </div>
                        </div>
                      ))}
                      <div style={{ display: "flex", justifyContent: "space-between", padding: "8px 0 0", borderTop: "1px solid var(--border)" }}>
                        <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Quality Score</span>
                        <span style={{
                          fontSize: "0.72rem", fontWeight: 700, fontFamily: "var(--font-mono)",
                          color: investigation.data_quality.quality_score >= 0.7 ? "var(--act)" : "var(--investigate)",
                        }}>
                          {(investigation.data_quality.quality_score * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* ── Intervention Sandbox — visible for all non-ABSTAIN investigations ── */}
              {investigation.verdict !== "ABSTAIN" && (
                <InterventionSandbox regionId={investigation.region_id} scenario={investigation.scenario} />
              )}


              {/* ── Telemetry Footer ── */}
              {investigation.telemetry && (
                <div style={{
                  display: "flex", justifyContent: "flex-end", alignItems: "center", gap: 20,
                  paddingTop: 12, paddingRight: 4,
                  fontSize: "0.68rem", color: "var(--text-muted)",
                  borderTop: "1px solid var(--border-subtle)",
                }}>
                  <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                    <IconClock/> Pipeline: <strong style={{ color: "var(--text-secondary)", fontFamily: "var(--font-mono)" }}>{investigation.telemetry.latency.pipeline_ms.toFixed(0)}ms</strong>
                  </div>
                  {investigation.telemetry.llm_used && (
                    <div>LLM: <strong style={{ color: "var(--text-secondary)", fontFamily: "var(--font-mono)" }}>{investigation.telemetry.latency.llm_ms.toFixed(0)}ms</strong></div>
                  )}
                  {investigation.telemetry.latency.rag_ms > 0 && (
                    <div>RAG: <strong style={{ color: "var(--text-secondary)", fontFamily: "var(--font-mono)" }}>{investigation.telemetry.latency.rag_ms.toFixed(0)}ms</strong></div>
                  )}
                  <div>Est. cost: <strong style={{ color: "var(--text-secondary)", fontFamily: "var(--font-mono)" }}>${investigation.telemetry.estimated_cost_usd.toFixed(6)}</strong></div>
                  {investigation.investigation_id && (
                    <div style={{ color: "var(--border-glow)", fontFamily: "var(--font-mono)", fontSize: "0.62rem" }}>
                      ID: {investigation.investigation_id.slice(0, 8)}
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : null}
        </section>
      </div>

      {/* ── Investigation Chat — floating panel (investigation-aware) ── */}
      {investigation && investigationId && (
        <InvestigationChat
          investigationId={investigationId}
          personaId={activePersona}
          regionId={investigation.region_id}
          scenario={investigation.scenario}
        />
      )}
    </div>
  );
}

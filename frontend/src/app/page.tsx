"use client";
// app/page.tsx

import React, { useState, useEffect } from "react";
import { api, RegionOverview, InvestigationResult, PersonaId, Persona } from "@/lib/api";
import PersonaSwitcher from "@/components/PersonaSwitcher";
import KPIHealthStrip from "@/components/KPIHealthStrip";
import EvidenceGraph from "@/components/EvidenceGraph";
import ActionCard from "@/components/ActionCard";
import InterventionSandbox from "@/components/InterventionSandbox";

export default function Home() {
  const [activePersona, setActivePersona] = useState<PersonaId>("gm");
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [regions, setRegions] = useState<Record<string, RegionOverview>>({});
  const [selectedRegion, setSelectedRegion] = useState<string>("region_a");
  const [investigation, setInvestigation] = useState<InvestigationResult | null>(null);
  const [loading, setLoading] = useState(false);

  // Initial load
  useEffect(() => {
    Promise.all([api.personas(), api.kpis()]).then(([pRes, kRes]) => {
      setPersonas(pRes.personas);
      setRegions(kRes.regions);
    }).catch(console.error);
  }, []);

  // Run investigation when region or persona changes
  useEffect(() => {
    if (!selectedRegion) return;
    setLoading(true);
    setInvestigation(null);
    api.runInvestigation(selectedRegion, activePersona, true)
      .then((res) => setInvestigation(res))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [selectedRegion, activePersona]);

  return (
    <div className="page" style={{ paddingBottom: 60 }}>
      {/* Top Nav */}
      <nav className="nav justify-between">
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 24, height: 24, borderRadius: 6, background: "linear-gradient(135deg, var(--indigo), var(--violet))" }} />
          <span className="nav-brand">EvidenceGraph AI</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
          <PersonaSwitcher personas={personas} activePersona={activePersona} onChange={setActivePersona} />
        </div>
      </nav>

      <main className="container mt-6 grid" style={{ gridTemplateColumns: "300px 1fr", gap: 32 }}>
        
        {/* Left Column: KPI Health Strip */}
        <aside>
          <KPIHealthStrip
            regions={regions}
            selectedRegion={selectedRegion}
            onSelect={setSelectedRegion}
          />
        </aside>

        {/* Right Column: Investigation Workspace */}
        <section style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          {loading && !investigation ? (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: 400, gap: 16 }}>
              <div className="spinner" />
              <div style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>Running autonomous investigation...</div>
            </div>
          ) : investigation ? (
            <>
              {/* Verdict Header */}
              <div className="card" style={{ display: "flex", alignItems: "flex-start", gap: 24, padding: 32 }}>
                <div style={{ flex: 1 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
                    <span className={`verdict-badge verdict-${investigation.verdict}`}>
                      {investigation.verdict}
                    </span>
                    {investigation.confidence?.score !== undefined && (
                      <span style={{ fontSize: "0.9rem", fontWeight: 700, color: "var(--text-primary)" }}>
                        {(investigation.confidence.score * 100).toFixed(0)}% Confidence
                      </span>
                    )}
                  </div>
                  
                  {/* LLM Narrative */}
                  {investigation.narrative ? (
                    <div style={{ fontSize: "1.05rem", color: "var(--text-primary)", lineHeight: 1.6, fontWeight: 500 }}>
                      {investigation.narrative.narrative}
                    </div>
                  ) : (
                    <div style={{ fontSize: "1.05rem", color: "var(--text-primary)", lineHeight: 1.6, fontWeight: 500 }}>
                      {investigation.confidence?.explanation}
                    </div>
                  )}

                  {investigation.challenge_result?.has_contradictions && (
                    <div style={{ marginTop: 16, padding: "10px 14px", background: "var(--investigate-bg)", borderLeft: "3px solid var(--investigate)", borderRadius: "0 4px 4px 0", fontSize: "0.8rem", color: "var(--text-primary)" }}>
                      <span style={{ fontWeight: 700, color: "var(--investigate)", marginRight: 8 }}>Challenge Engine:</span>
                      {investigation.challenge_result.challenge_summary}
                    </div>
                  )}
                  
                  {/* RAG Evidence Summary */}
                  {investigation.rag_evidence && investigation.rag_evidence.results.length > 0 && (
                    <div style={{ marginTop: 16 }}>
                      <h4 style={{ fontSize: "0.7rem", marginBottom: 8 }}>Supporting Context (RAG)</h4>
                      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                        {investigation.rag_evidence.results.slice(0, 2).map(r => (
                          <div key={r.id} style={{ fontSize: "0.75rem", color: "var(--text-secondary)", background: "var(--bg-elevated)", padding: "6px 10px", borderRadius: 4 }}>
                            "{r.text}" <span style={{ color: "var(--text-muted)", marginLeft: 6 }}>— {r.id}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Sub-scores (Analyst / GM) */}
                {investigation.raw_sub_scores && Object.keys(investigation.raw_sub_scores).length > 0 && (
                  <div style={{ width: 220, background: "var(--bg-elevated)", padding: 16, borderRadius: "var(--radius-md)", border: "1px solid var(--border)" }}>
                    <h4 style={{ fontSize: "0.65rem", marginBottom: 12 }}>Sub-Scores</h4>
                    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                      {Object.entries(investigation.raw_sub_scores).map(([k, v]) => (
                        <div key={k} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                          <span style={{ fontSize: "0.7rem", color: "var(--text-secondary)" }}>{k.replace(/_/g, " ")}</span>
                          <span style={{ fontSize: "0.75rem", fontFamily: "var(--font-mono)", fontWeight: 600, color: "var(--text-primary)" }}>{v.toFixed(2)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Layout for Graph and Actions */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
                
                {/* Evidence Graph */}
                {investigation.evidence_graph && (
                  <div className="card" style={{ display: "flex", flexDirection: "column" }}>
                    <h3 style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: 16 }}>Evidence Graph</h3>
                    <EvidenceGraph 
                      nodes={investigation.evidence_graph.nodes}
                      links={investigation.evidence_graph.links}
                      driverRanking={investigation.evidence_graph.driver_ranking}
                    />
                  </div>
                )}

                {/* Action Engine & Causal Chain */}
                <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
                  {investigation.action && (
                    <ActionCard action={investigation.action} verdict={investigation.verdict} />
                  )}

                  {investigation.causal_chain && investigation.causal_chain.length > 0 && (
                    <div className="card card-sm">
                      <h4 style={{ marginBottom: 12 }}>Causal Chain</h4>
                      <div className="stepper">
                        {investigation.causal_chain.map((kpi, i) => (
                          <div key={kpi} className="step">
                            <div className="step-indicator">
                              <div className={`step-dot ${i === 0 ? "active" : "complete"}`}>
                                {i + 1}
                              </div>
                              {i < investigation.causal_chain!.length - 1 && <div className="step-line" />}
                            </div>
                            <div className="step-content" style={{ paddingBottom: i === investigation.causal_chain!.length - 1 ? 0 : 16 }}>
                              <div className="step-label" style={{ color: i === 0 ? "var(--investigate)" : "var(--text-primary)" }}>
                                {kpi.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}
                              </div>
                              {i === 0 && <div className="step-detail">Identified Root Cause</div>}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

              </div>

              {/* Intervention Sandbox (Only for ACT/INVESTIGATE where a lever makes sense) */}
              {investigation.verdict !== "ABSTAIN" && (
                <InterventionSandbox regionId={investigation.region_id} />
              )}

              {/* Telemetry Footer */}
              {investigation.telemetry && (
                <div style={{ display: "flex", justifyContent: "flex-end", gap: 16, fontSize: "0.7rem", color: "var(--text-muted)", marginTop: 16 }}>
                  <span>Pipeline: {investigation.telemetry.latency.pipeline_ms.toFixed(0)}ms</span>
                  {investigation.telemetry.llm_used && (
                    <>
                      <span>LLM: {investigation.telemetry.latency.llm_ms.toFixed(0)}ms</span>
                      <span>Tokens: {investigation.telemetry.tokens?.total}</span>
                    </>
                  )}
                  <span>Cost: ${(investigation.telemetry.estimated_cost_usd * 1000).toFixed(4)} per 1k</span>
                </div>
              )}
            </>
          ) : null}
        </section>

      </main>
    </div>
  );
}

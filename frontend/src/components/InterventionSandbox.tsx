"use client";
import React, { useState, useEffect } from "react";
import { api, Lever, SandboxResult } from "@/lib/api";

interface Props { regionId: string; }

export default function InterventionSandbox({ regionId }: Props) {
  const [levers, setLevers] = useState<Lever[]>([]);
  const [selectedLever, setSelectedLever] = useState<string>("");
  const [leverValue, setLeverValue] = useState<number>(0);
  const [result, setResult] = useState<SandboxResult | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.levers().then(res => {
      setLevers(res.levers);
      if (res.levers.length > 0) {
        setSelectedLever(res.levers[0].id);
        setLeverValue(res.levers[0].default_recovery);
      }
    }).catch(console.error);
  }, []);

  const activeLever = levers.find(l => l.id === selectedLever);

  useEffect(() => {
    if (!selectedLever) return;
    const t = setTimeout(async () => {
      setLoading(true);
      try {
        const res = await api.simulate(regionId, selectedLever, leverValue);
        setResult(res);
      } catch { /* silent */ } finally { setLoading(false); }
    }, 500);
    return () => clearTimeout(t);
  }, [leverValue, selectedLever, regionId]);

  if (!levers.length) return null;

  return (
    <div style={{
      background: "var(--bg-card)",
      border: "1px solid var(--border)",
      borderRadius: "var(--radius-md)",
      overflow: "hidden",
      boxShadow: "var(--shadow-md)",
    }}>
      {/* Premium Header */}
      <div style={{
        padding: "16px 24px",
        borderBottom: "1px solid var(--border-subtle)",
        background: "rgba(161,0,255,0.04)",
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <div>
          <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "var(--text-primary)" }}>Intervention Sandbox</h3>
          <p style={{ fontSize: "0.75rem", color: "var(--text-secondary)", margin: 0 }}>
            Simulate operational changes and forecast revenue recovery
          </p>
        </div>
        <div style={{
          fontSize: "0.65rem", fontWeight: 800, padding: "4px 10px",
          borderRadius: 100, background: "rgba(161,0,255,0.1)",
          color: "var(--accenture-purple-light)", border: "1px solid rgba(161,0,255,0.25)",
          letterSpacing: "0.05em", textTransform: "uppercase",
        }}>
          Counterfactual Model
        </div>
      </div>

      <div style={{ padding: "20px 24px", display: "grid", gridTemplateColumns: "1fr 2fr", gap: 28 }}>
        {/* Controls */}
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <div>
            <label style={{ fontSize: "0.68rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", display: "block", marginBottom: 8 }}>
              Intervention Lever
            </label>
            <select className="input select" value={selectedLever}
              style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)" }}
              onChange={e => {
                setSelectedLever(e.target.value);
                const l = levers.find(x => x.id === e.target.value);
                if (l) setLeverValue(l.default_recovery);
              }}>
              {levers.map(l => <option key={l.id} value={l.id}>{l.label}</option>)}
            </select>
          </div>

          {activeLever && (
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <label style={{ fontSize: "0.68rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                  Adjustment
                </label>
                <span style={{ fontSize: "0.9rem", fontWeight: 800, color: "var(--accenture-purple-light)", fontFamily: "var(--font-mono)" }}>
                  {leverValue}{activeLever.unit}
                </span>
              </div>
              <div className="slider-wrapper">
                <input type="range" className="slider"
                  min={activeLever.min} max={activeLever.max} value={leverValue}
                  onChange={e => setLeverValue(Number(e.target.value))} />
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6, fontSize: "0.65rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                <span>{activeLever.min}{activeLever.unit}</span>
                <span>{activeLever.max}{activeLever.unit}</span>
              </div>
            </div>
          )}
        </div>

        {/* Results */}
        <div style={{
          background: "var(--bg-surface)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-md)",
          padding: 18,
          minHeight: 140,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
        }}>
          {loading && !result ? (
            <div style={{ display: "flex", gap: 12, alignItems: "center", color: "var(--text-muted)", justifyContent: "center" }}>
              <div className="spinner" /> <span style={{ fontSize: "0.82rem", fontFamily: "var(--font-mono)" }}>Running simulation...</span>
            </div>
          ) : result ? (
            <div>
              {/* Revenue recovery hero */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", paddingBottom: 14, marginBottom: 14, borderBottom: "1px solid var(--border-subtle)" }}>
                <div style={{ fontSize: "0.78rem", fontWeight: 600, color: "var(--text-secondary)" }}>
                  Projected Revenue Recovery
                </div>
                <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                  <span style={{ fontSize: "1.35rem", fontWeight: 800, color: "var(--act)", fontFamily: "var(--font-mono)" }}>
                    +${((result.revenue_recovery_estimate?.mid ?? 0) / 1000).toFixed(0)}K
                  </span>
                  <span style={{ fontSize: "0.68rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                    [{((result.revenue_recovery_estimate?.low ?? 0) / 1000).toFixed(0)}K–{((result.revenue_recovery_estimate?.high ?? 0) / 1000).toFixed(0)}K]
                  </span>
                </div>
              </div>

              {/* KPI deltas */}
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {Object.entries(result.simulated_outcomes)
                  .filter(([kpi]) => kpi !== result.lever && kpi !== "revenue")
                  .map(([kpi, data]) => {
                    const isRateKpi = kpi.includes("rate") || kpi.includes("volume");
                    const improved = isRateKpi ? data.delta_pct < 0 : data.delta_pct > 0;
                    const deltaColor = improved ? "var(--act)" : "var(--abstain)";
                    return (
                      <div key={kpi} style={{
                        display: "flex", alignItems: "center", justifyContent: "space-between",
                        fontSize: "0.75rem", padding: "8px 12px", background: "var(--bg-elevated)",
                        borderRadius: 8, border: "1px solid var(--border-subtle)"
                      }}>
                        <span style={{ color: "var(--text-secondary)", fontWeight: 500 }}>
                          {kpi.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}
                        </span>
                        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                          <span style={{ color: "var(--text-muted)", textDecoration: "line-through", fontFamily: "var(--font-mono)" }}>
                            {data.baseline.toFixed(1)}
                          </span>
                          <span style={{ fontWeight: 700, color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}>
                            {data.simulated.toFixed(1)}
                          </span>
                          <span style={{ fontWeight: 700, color: deltaColor, width: 48, textAlign: "right", fontFamily: "var(--font-mono)" }}>
                            {data.delta_pct > 0 ? "+" : ""}{data.delta_pct.toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    );
                  })}
              </div>
              <div style={{ fontSize: "0.62rem", color: "var(--text-muted)", textAlign: "right", marginTop: 12, fontFamily: "var(--font-mono)" }}>
                Model confidence: {((result.model_confidence ?? 0) * 100).toFixed(0)}%
              </div>
            </div>
          ) : (
            <div style={{ textAlign: "center", color: "var(--text-muted)", fontSize: "0.82rem" }}>
              Adjust the lever to simulate impact
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

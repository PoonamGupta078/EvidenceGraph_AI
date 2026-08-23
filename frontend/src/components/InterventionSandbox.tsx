"use client";
// components/InterventionSandbox.tsx

import React, { useState, useEffect } from "react";
import { api, Lever, SandboxResult } from "@/lib/api";

interface Props {
  regionId: string;
}

export default function InterventionSandbox({ regionId }: Props) {
  const [levers, setLevers] = useState<Lever[]>([]);
  const [selectedLever, setSelectedLever] = useState<string>("");
  const [leverValue, setLeverValue] = useState<number>(0);
  const [result, setResult] = useState<SandboxResult | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.levers().then((res) => {
      setLevers(res.levers);
      if (res.levers.length > 0) {
        setSelectedLever(res.levers[0].id);
        setLeverValue(res.levers[0].default_recovery);
      }
    }).catch(console.error);
  }, []);

  const activeLever = levers.find(l => l.id === selectedLever);

  const runSimulation = async (value: number) => {
    if (!selectedLever) return;
    setLoading(true);
    try {
      const res = await api.simulate(regionId, selectedLever, value);
      setResult(res);
    } catch (e) {
      console.error("Simulation failed", e);
    } finally {
      setLoading(false);
    }
  };

  // Debounced simulation
  useEffect(() => {
    const t = setTimeout(() => runSimulation(leverValue), 500);
    return () => clearTimeout(t);
  }, [leverValue, selectedLever, regionId]);

  if (!levers.length) return null;

  return (
    <div style={{
      background: "var(--bg-elevated)",
      border: "1px solid var(--border)",
      borderRadius: "var(--radius-lg)",
      padding: "20px",
    }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
        <div>
          <h3 style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--text-primary)" }}>Intervention Sandbox</h3>
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 2 }}>
            Simulate downstream effects of operational changes.
          </p>
        </div>
        <span style={{ fontSize: "1.5rem" }}>🎛</span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: 30 }}>
        {/* Controls */}
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <div>
            <label style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", display: "block", marginBottom: 8 }}>
              Select Lever
            </label>
            <select
              className="input select"
              value={selectedLever}
              onChange={(e) => {
                setSelectedLever(e.target.value);
                const l = levers.find(x => x.id === e.target.value);
                if (l) setLeverValue(l.default_recovery);
              }}
            >
              {levers.map(l => (
                <option key={l.id} value={l.id}>{l.label}</option>
              ))}
            </select>
          </div>

          {activeLever && (
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
                <label style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                  Adjust Value
                </label>
                <span style={{ fontSize: "0.85rem", fontWeight: 700, color: "var(--indigo-light)" }}>
                  {leverValue}{activeLever.unit}
                </span>
              </div>
              
              <div className="slider-wrapper">
                <input
                  type="range"
                  className="slider"
                  min={activeLever.min}
                  max={activeLever.max}
                  value={leverValue}
                  onChange={(e) => setLeverValue(Number(e.target.value))}
                />
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8, fontSize: "0.7rem", color: "var(--text-muted)" }}>
                <span>{activeLever.min}{activeLever.unit}</span>
                <span>{activeLever.max}{activeLever.unit}</span>
              </div>
            </div>
          )}
        </div>

        {/* Results */}
        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "16px" }}>
          {loading && !result ? (
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--text-muted)", gap: 12 }}>
              <div className="spinner" /> Simulating...
            </div>
          ) : result ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", paddingBottom: 12, borderBottom: "1px solid var(--border)" }}>
                <span style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--text-secondary)" }}>Est. Revenue Recovery</span>
                <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                  <span style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--act)" }}>
                    +${(result.revenue_recovery_estimate.mid / 1000).toFixed(0)}K
                  </span>
                  <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
                    [{((result.revenue_recovery_estimate.low || 0)/1000).toFixed(0)}K – {((result.revenue_recovery_estimate.high || 0)/1000).toFixed(0)}K]
                  </span>
                </div>
              </div>

              <div>
                <h4 style={{ fontSize: "0.75rem", marginBottom: 12, color: "var(--text-muted)" }}>Downstream Impact (Predicted)</h4>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {Object.entries(result.simulated_outcomes).map(([kpi, data]) => {
                    if (kpi === "revenue" || kpi === result.lever) return null; // skip rev (shown above) and the lever itself
                    const isImprovement = data.delta_pct < 0; // for most KPIs like delay/cancel, down is better
                    const color = isImprovement ? "var(--act)" : "var(--abstain)";
                    return (
                      <div key={kpi} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "0.8rem", background: "var(--bg-primary)", padding: "8px 12px", borderRadius: "var(--radius-sm)" }}>
                        <span style={{ color: "var(--text-primary)" }}>{kpi.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}</span>
                        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                          <span style={{ color: "var(--text-muted)", textDecoration: "line-through" }}>
                            {data.baseline.toFixed(1)}
                          </span>
                          <span style={{ color: "var(--text-primary)", fontWeight: 600 }}>
                            {data.simulated.toFixed(1)}
                          </span>
                          <span style={{ color, fontWeight: 700, width: 45, textAlign: "right", fontSize: "0.75rem" }}>
                            {data.delta_pct > 0 ? "+" : ""}{data.delta_pct.toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", textAlign: "right" }}>
                Model Confidence: {(result.model_confidence * 100).toFixed(0)}%
              </div>
            </div>
          ) : (
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--text-muted)", fontSize: "0.85rem" }}>
              Adjust lever to simulate impact
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

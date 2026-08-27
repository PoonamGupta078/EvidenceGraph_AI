"use client";
import React from "react";

interface Action {
  action_id?: string;
  type: string;
  title: string;
  description: string;
  owner: string;
  priority: string;
  estimated_impact?: string;
  preconditions?: string[];
  risks?: string[];
  confidence_score?: number;
}

interface Props {
  action: Action;
  verdict: string;
}

const PRIORITY_COLORS: Record<string, string> = {
  HIGH: "#ff4d4d",
  MEDIUM: "#ffb400",
  LOW: "#00d98b",
};

const TYPE_ICONS: Record<string, string> = {
  OPERATIONAL: "⚙️",
  INVESTIGATIVE: "🔍",
  ESCALATION: "📊",
  HOLD: "⏸️",
};

export default function ActionCard({ action, verdict }: Props) {
  const priorityColor = PRIORITY_COLORS[action.priority] || "#8892a4";
  const verdictColor = verdict === "ACT" ? "var(--act)"
    : verdict === "INVESTIGATE" ? "var(--investigate)"
    : "var(--abstain)";

  return (
    <div style={{
      borderRadius: "var(--radius-md)",
      border: "1px solid var(--border)",
      background: "var(--bg-card)",
      overflow: "hidden",
      boxShadow: "var(--shadow-md)",
      position: "relative",
    }}>
      {/* Top gradient bar */}
      <div style={{
        height: 3,
        background: `linear-gradient(90deg, ${verdictColor}, ${verdictColor}60, transparent)`,
      }}/>

      {/* Header */}
      <div style={{
        padding: "16px 20px",
        borderBottom: "1px solid var(--border-subtle)",
        background: `${verdictColor}08`,
      }}>
        <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
          <span style={{ fontSize: "1.1rem", marginTop: 1 }}>{TYPE_ICONS[action.type] || "📋"}</span>
          <div style={{ flex: 1 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
              <span style={{ fontSize: "0.88rem", fontWeight: 700, color: "var(--text-primary)" }}>
                {action.title}
              </span>
              <span style={{
                fontSize: "0.6rem", fontWeight: 800, padding: "2px 8px",
                borderRadius: 4,
                background: `${priorityColor}18`,
                color: priorityColor,
                border: `1px solid ${priorityColor}35`,
                textTransform: "uppercase",
                letterSpacing: "0.06em",
              }}>
                {action.priority}
              </span>
            </div>
            {action.action_id && (
              <div style={{ fontSize: "0.62rem", fontFamily: "var(--font-mono)", color: "var(--text-muted)", marginTop: 4 }}>
                {action.action_id}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Body */}
      <div style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 14 }}>
        <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)", lineHeight: 1.7, margin: 0 }}>
          {action.description}
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
          <div style={{
            padding: "10px 12px", background: "var(--bg-elevated)",
            borderRadius: 8, border: "1px solid var(--border-subtle)",
          }}>
            <div style={{ fontSize: "0.6rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--text-muted)", marginBottom: 4 }}>
              Owner
            </div>
            <div style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--text-primary)" }}>
              {action.owner?.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}
            </div>
          </div>
          {action.estimated_impact && (
            <div style={{
              padding: "10px 12px", background: "var(--act-bg)",
              borderRadius: 8, border: "1px solid var(--act-border)",
            }}>
              <div style={{ fontSize: "0.6rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--text-muted)", marginBottom: 4 }}>
                Est. Impact
              </div>
              <div style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--act)", fontFamily: "var(--font-mono)" }}>
                {action.estimated_impact}
              </div>
            </div>
          )}
        </div>

        {action.preconditions && action.preconditions.length > 0 && (
          <div>
            <div style={{ fontSize: "0.6rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--text-muted)", marginBottom: 8 }}>
              Preconditions
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
              {action.preconditions.map((p, i) => (
                <div key={i} style={{
                  display: "flex", gap: 8, fontSize: "0.76rem", color: "var(--text-secondary)",
                  padding: "4px 0",
                }}>
                  <span style={{ color: "var(--act)", fontWeight: 700, flexShrink: 0 }}>✓</span> {p}
                </div>
              ))}
            </div>
          </div>
        )}

        {action.risks && action.risks.length > 0 && (
          <div>
            <div style={{ fontSize: "0.6rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--text-muted)", marginBottom: 8 }}>
              Risks
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
              {action.risks.map((r, i) => (
                <div key={i} style={{
                  display: "flex", gap: 8, fontSize: "0.76rem", color: "var(--text-secondary)",
                  padding: "4px 0",
                }}>
                  <span style={{ color: "var(--investigate)", fontWeight: 700, flexShrink: 0 }}>⚠</span> {r}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

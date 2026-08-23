"use client";
// components/ActionCard.tsx

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
  HIGH: "var(--abstain)",
  MEDIUM: "var(--investigate)",
  LOW: "var(--act)",
};

const TYPE_ICONS: Record<string, string> = {
  OPERATIONAL: "⚙",
  INVESTIGATIVE: "🔍",
  ESCALATION: "📊",
  HOLD: "⏸",
};

export default function ActionCard({ action, verdict }: Props) {
  const priorityColor = PRIORITY_COLORS[action.priority] || "var(--text-muted)";

  return (
    <div style={{
      borderRadius: "var(--radius-lg)",
      border: `1px solid var(--${verdict.toLowerCase()}-border, var(--border))`,
      background: `var(--${verdict.toLowerCase()}-bg, var(--bg-card))`,
      overflow: "hidden",
    }}>
      {/* Header */}
      <div style={{
        padding: "16px 20px",
        borderBottom: "1px solid var(--border)",
        display: "flex",
        alignItems: "flex-start",
        gap: 12,
      }}>
        <span style={{ fontSize: "1.2rem" }}>{TYPE_ICONS[action.type] || "📋"}</span>
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
            <h3 style={{ fontSize: "0.95rem", fontWeight: 700, color: "var(--text-primary)" }}>
              {action.title}
            </h3>
            <span style={{
              fontSize: "0.65rem", fontWeight: 700, padding: "2px 8px",
              borderRadius: "100px",
              background: `${priorityColor}20`,
              color: priorityColor,
              border: `1px solid ${priorityColor}50`,
              textTransform: "uppercase",
              letterSpacing: "0.06em",
            }}>
              {action.priority}
            </span>
            <span style={{
              fontSize: "0.65rem", fontWeight: 600, padding: "2px 8px",
              borderRadius: "100px",
              background: "var(--bg-elevated)",
              color: "var(--text-secondary)",
              border: "1px solid var(--border)",
              textTransform: "uppercase",
              letterSpacing: "0.06em",
            }}>
              {action.type}
            </span>
          </div>
          {action.action_id && (
            <div style={{ fontSize: "0.7rem", fontFamily: "var(--font-mono)", color: "var(--text-muted)", marginTop: 4 }}>
              {action.action_id}
            </div>
          )}
        </div>
      </div>

      {/* Body */}
      <div style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 16 }}>
        <p style={{ fontSize: "0.875rem", color: "var(--text-primary)", lineHeight: 1.7 }}>
          {action.description}
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          {/* Owner */}
          <div>
            <div style={{ fontSize: "0.65rem", textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--text-muted)", marginBottom: 4 }}>Owner</div>
            <div style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--text-primary)" }}>
              {action.owner?.replace("_", " ").replace(/\b\w/g, c => c.toUpperCase())}
            </div>
          </div>

          {/* Impact */}
          {action.estimated_impact && (
            <div>
              <div style={{ fontSize: "0.65rem", textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--text-muted)", marginBottom: 4 }}>Estimated Impact</div>
              <div style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--act)" }}>
                {action.estimated_impact}
              </div>
            </div>
          )}

          {/* Confidence */}
          {action.confidence_score !== undefined && (
            <div>
              <div style={{ fontSize: "0.65rem", textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--text-muted)", marginBottom: 4 }}>Confidence</div>
              <div style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--indigo-light)" }}>
                {(action.confidence_score * 100).toFixed(0)}%
              </div>
            </div>
          )}
        </div>

        {/* Preconditions */}
        {action.preconditions && action.preconditions.length > 0 && (
          <div>
            <div style={{ fontSize: "0.65rem", textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--text-muted)", marginBottom: 8 }}>Preconditions</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {action.preconditions.map((p, i) => (
                <div key={i} style={{
                  display: "flex", alignItems: "flex-start", gap: 8,
                  fontSize: "0.78rem", color: "var(--text-secondary)",
                }}>
                  <span style={{ color: "var(--act)", flexShrink: 0, marginTop: 1 }}>✓</span>
                  {p}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Risks */}
        {action.risks && action.risks.length > 0 && (
          <div>
            <div style={{ fontSize: "0.65rem", textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--text-muted)", marginBottom: 8 }}>Risks</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {action.risks.map((r, i) => (
                <div key={i} style={{
                  display: "flex", alignItems: "flex-start", gap: 8,
                  fontSize: "0.78rem", color: "var(--text-secondary)",
                }}>
                  <span style={{ color: "var(--investigate)", flexShrink: 0, marginTop: 1 }}>⚠</span>
                  {r}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

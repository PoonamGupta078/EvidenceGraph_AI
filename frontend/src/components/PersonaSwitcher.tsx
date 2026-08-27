"use client";
import React from "react";
import { Persona, PersonaId } from "@/lib/api";

interface Props {
  personas: Persona[];
  activePersona: PersonaId;
  onChange: (id: PersonaId) => void;
}

const PERSONA_META: Record<string, { color: string; icon: string; glow: string }> = {
  gm: { color: "#a100ff", icon: "👔", glow: "rgba(161,0,255,0.15)" },
  ops_lead: { color: "#00d98b", icon: "⚙️", glow: "rgba(0,217,139,0.15)" },
  analyst: { color: "#818cf8", icon: "📊", glow: "rgba(129,140,248,0.15)" },
};

export default function PersonaSwitcher({ personas, activePersona, onChange }: Props) {
  if (!personas.length) return null;

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <span style={{
        fontSize: "0.65rem", fontWeight: 600, color: "var(--text-muted)",
        textTransform: "uppercase", letterSpacing: "0.08em", marginRight: 6,
      }}>
        View as
      </span>
      {personas.map((p) => {
        const isActive = p.id === activePersona;
        const meta = PERSONA_META[p.id] || { color: "#a100ff", icon: "👤", glow: "rgba(161,0,255,0.15)" };
        return (
          <button
            key={p.id}
            id={`persona-${p.id}`}
            onClick={() => onChange(p.id as PersonaId)}
            title={p.description}
            style={{
              display: "flex", alignItems: "center", gap: 7,
              padding: "6px 14px",
              borderRadius: 8,
              border: isActive ? `1.5px solid ${meta.color}` : "1px solid var(--border)",
              background: isActive ? `${meta.color}12` : "var(--bg-card)",
              color: isActive ? meta.color : "var(--text-secondary)",
              cursor: "pointer",
              fontSize: "0.78rem",
              fontWeight: isActive ? 700 : 500,
              transition: "all 0.25s cubic-bezier(0.16, 1, 0.3, 1)",
              fontFamily: "var(--font-sans)",
              boxShadow: isActive ? `0 0 0 3px ${meta.glow}, 0 0 12px ${meta.glow}` : "none",
              transform: isActive ? "translateY(-1px)" : "none",
            }}
          >
            <span style={{ fontSize: "0.85rem" }}>{meta.icon}</span>
            {isActive && (
              <div style={{
                width: 6, height: 6, borderRadius: "50%",
                background: meta.color,
                boxShadow: `0 0 6px ${meta.color}`,
                flexShrink: 0,
              }}/>
            )}
            {p.label}
          </button>
        );
      })}
    </div>
  );
}

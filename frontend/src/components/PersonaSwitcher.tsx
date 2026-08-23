"use client";
// components/PersonaSwitcher.tsx

import React from "react";
import { Persona, PersonaId } from "@/lib/api";

interface Props {
  personas: Persona[];
  activePersona: PersonaId;
  onChange: (id: PersonaId) => void;
}

const ICONS: Record<string, React.ReactNode> = {
  crown: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
      <path d="M3 19h18v2H3v-2zm.5-9.5l4 4 4.5-9L16.5 13l4-4 1 10H3l.5-9.5z" />
    </svg>
  ),
  settings: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  ),
  chart: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="18" y1="20" x2="18" y2="10" />
      <line x1="12" y1="20" x2="12" y2="4" />
      <line x1="6" y1="20" x2="6" y2="14" />
    </svg>
  ),
};

export default function PersonaSwitcher({ personas, activePersona, onChange }: Props) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <h4 style={{ fontSize: "0.7rem", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--text-muted)", marginBottom: 4 }}>
        Active Persona
      </h4>
      <div style={{ display: "flex", gap: 8 }}>
        {personas.map((p) => {
          const isActive = p.id === activePersona;
          return (
            <button
              key={p.id}
              id={`persona-${p.id}`}
              onClick={() => onChange(p.id as PersonaId)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                padding: "8px 14px",
                borderRadius: "var(--radius-md)",
                border: isActive ? `1px solid ${p.color}40` : "1px solid var(--border)",
                background: isActive ? `${p.color}18` : "var(--bg-elevated)",
                color: isActive ? p.color : "var(--text-secondary)",
                cursor: "pointer",
                fontSize: "0.8rem",
                fontWeight: isActive ? 600 : 500,
                transition: "all var(--transition)",
                fontFamily: "var(--font-sans)",
              }}
              title={p.description}
            >
              <span style={{ color: isActive ? p.color : "var(--text-muted)" }}>
                {ICONS[p.icon] || null}
              </span>
              {p.label}
              {isActive && (
                <span
                  style={{
                    width: 6, height: 6,
                    borderRadius: "50%",
                    background: p.color,
                    boxShadow: `0 0 6px ${p.color}`,
                  }}
                />
              )}
            </button>
          );
        })}
      </div>
      {personas.find(p => p.id === activePersona) && (
        <p style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 2 }}>
          {personas.find(p => p.id === activePersona)?.description}
        </p>
      )}
    </div>
  );
}

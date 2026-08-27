"use client";
import React from "react";

interface GraphNode { id: string; label: string; material: boolean; centrality: number; }
interface GraphLink { source: string; target: string; type: string; weight: number; lag_days: number; }
interface Props {
  nodes: GraphNode[];
  links: GraphLink[];
  driverRanking: Array<{ kpi: string; score: number; is_material: boolean }>;
}

const EDGE_COLORS: Record<string, string> = {
  BUSINESS_RULE_PRIOR: "#7c3aed",
  CORRELATES_WITH: "#4f46e5",
  CONTRADICTS: "#e11d48",
  COMPENSATES: "#d97706",
  LAGS: "#059669",
  PVM_CONTRIBUTION: "#c026d3",
  INDEPENDENT: "#64748b",
};

const EDGE_LABELS: Record<string, string> = {
  BUSINESS_RULE_PRIOR: "Prior",
  CORRELATES_WITH: "Corr.",
  CONTRADICTS: "Contra.",
  COMPENSATES: "Comp.",
  LAGS: "Lag",
  PVM_CONTRIBUTION: "PVM",
  INDEPENDENT: "Indep.",
};

const CHAIN_ORDER = [
  "warehouse_staffing_level", "fulfillment_delay_rate",
  "support_ticket_volume", "order_cancellation_rate", "revenue",
];

export default function EvidenceGraph({ nodes, links, driverRanking }: Props) {
  if (!nodes || nodes.length === 0) return null;

  const width = 600, height = 300, cy = height / 2;
  const ordered = [
    ...CHAIN_ORDER.filter(id => nodes.find(n => n.id === id)),
    ...nodes.filter(n => !CHAIN_ORDER.includes(n.id)).map(n => n.id),
  ];
  const positions: Record<string, { x: number; y: number }> = {};
  ordered.forEach((id, i) => {
    positions[id] = {
      x: 80 + (i / Math.max(1, ordered.length - 1)) * (width - 160),
      y: cy + (i % 2 === 0 ? -40 : 40),
    };
  });

  return (
    <div style={{ padding: 20, background: "#fff", borderRadius: 12, border: "1px solid #e2e8f0" }}>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ overflow: "visible" }}>
        {links.map((link, i) => {
          const s = positions[link.source as string], t = positions[link.target as string];
          if (!s || !t) return null;
          const mx = (s.x + t.x) / 2, my = (s.y + t.y) / 2 - 40;
          return (
            <g key={i}>
              <path d={`M${s.x},${s.y} Q${mx},${my} ${t.x},${t.y}`}
                stroke={EDGE_COLORS[link.type] || "#94a3b8"} strokeWidth={2}
                fill="none" strokeDasharray={link.type === "LAGS" ? "6 4" : "none"}/>
            </g>
          );
        })}
        {nodes.map(node => {
          const pos = positions[node.id];
          if (!pos) return null;
          return (
            <g key={node.id}>
              <circle cx={pos.x} cy={pos.y} r={32} fill={node.material ? "#fef3c7" : "#f1f5f9"} stroke={node.material ? "#d97706" : "#475569"} strokeWidth={3}/>
              <text x={pos.x} y={pos.y + 5} textAnchor="middle" fontSize={10} fontWeight="bold" fill="#1e293b" style={{ pointerEvents: "none" }}>
                {node.label.split(" ").slice(0, 2).join(" ")}
              </text>
            </g>
          );
        })}
      </svg>

      <div style={{ marginTop: 24 }}>
        <h4 style={{ fontSize: 14, fontWeight: 700, color: "#0f172a", marginBottom: 12 }}>Driver Ranking</h4>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {driverRanking.slice(0, 5).map((d, i) => (
            <div key={d.kpi} style={{ display: "flex", alignItems: "center", gap: 12, padding: "8px 0", borderBottom: "1px solid #f1f5f9" }}>
              <span style={{ fontSize: 12, color: "#64748b", width: 20 }}>#{i + 1}</span>
              <span style={{ fontSize: 13, fontWeight: 500, color: "#1e293b", flex: 1 }}>{d.kpi.replace(/_/g, " ")}</span>
              <div style={{ width: 100, height: 6, background: "#e2e8f0", borderRadius: 3, overflow: "hidden" }}>
                <div style={{ height: "100%", width: `${d.score * 100}%`, background: "#4f46e5" }}/>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

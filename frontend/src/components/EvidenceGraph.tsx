"use client";
// components/EvidenceGraph.tsx
// react-force-graph-2d visualization of the typed evidence graph

import React, { useEffect, useRef, useCallback } from "react";

interface GraphNode {
  id: string;
  label: string;
  material: boolean;
  centrality: number;
}

interface GraphLink {
  source: string;
  target: string;
  type: string;
  weight: number;
  lag_days: number;
}

interface Props {
  nodes: GraphNode[];
  links: GraphLink[];
  driverRanking: Array<{ kpi: string; score: number; is_material: boolean }>;
}

const EDGE_COLORS: Record<string, string> = {
  CAUSES: "#6366f1",
  CORRELATES_WITH: "#0ea5e9",
  CONTRADICTS: "#ef4444",
  COMPENSATES: "#f59e0b",
  LAGS: "#8b5cf6",
  INDEPENDENT: "#10b981",
};

export default function EvidenceGraph({ nodes, links, driverRanking }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<any>(null);

  useEffect(() => {
    if (typeof window === "undefined" || !containerRef.current) return;
    if (!nodes || nodes.length === 0) return;

    // Dynamically import to avoid SSR issues
    import("react-force-graph-2d").then((mod) => {
      const ForceGraph2D = mod.default;
      // We render via a custom canvas approach since we can't use JSX here directly
      // The actual render happens through the returned JSX
    });
  }, [nodes, links]);

  if (!nodes || nodes.length === 0) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 200, color: "var(--text-muted)", fontSize: "0.8rem" }}>
        No graph data available
      </div>
    );
  }

  return (
    <div>
      <GraphCanvas nodes={nodes} links={links} />

      {/* Legend */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginTop: 16 }}>
        {Object.entries(EDGE_COLORS).map(([type, color]) => (
          <div key={type} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 20, height: 2, background: color, borderRadius: 1 }} />
            <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>{type}</span>
          </div>
        ))}
      </div>

      {/* Driver ranking */}
      {driverRanking.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <h4 style={{ marginBottom: 10 }}>Driver Ranking (Centrality + Correlation)</h4>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {driverRanking.slice(0, 5).map((d, i) => (
              <div key={d.kpi} style={{
                display: "flex", alignItems: "center", gap: 12,
                padding: "8px 12px",
                borderRadius: "var(--radius-sm)",
                background: d.is_material ? "var(--investigate-bg)" : "var(--bg-elevated)",
                border: d.is_material ? "1px solid var(--investigate-border)" : "1px solid var(--border)",
              }}>
                <span style={{ fontSize: "0.7rem", fontWeight: 700, color: "var(--text-muted)", width: 20 }}>#{i + 1}</span>
                <span style={{ fontSize: "0.8rem", fontWeight: 600, flex: 1, color: d.is_material ? "var(--investigate)" : "var(--text-primary)" }}>
                  {d.kpi.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}
                </span>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <div style={{
                    width: 60, height: 4, borderRadius: 2,
                    background: "var(--bg-primary)",
                    overflow: "hidden",
                  }}>
                    <div style={{
                      height: "100%",
                      width: `${Math.min(100, d.score * 300)}%`,
                      background: d.is_material ? "var(--investigate)" : "var(--indigo)",
                      borderRadius: 2,
                    }} />
                  </div>
                  <span style={{ fontSize: "0.7rem", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                    {d.score.toFixed(3)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// SVG-based graph canvas (no react-force-graph dependency issues)
function GraphCanvas({ nodes, links }: { nodes: GraphNode[]; links: GraphLink[] }) {
  const width = 560;
  const height = 300;
  const cx = width / 2;
  const cy = height / 2;

  // Layout: arrange nodes in a horizontal chain
  const nodeCount = nodes.length;
  const positions: Record<string, { x: number; y: number }> = {};

  const CHAIN_ORDER = [
    "warehouse_staffing_level",
    "fulfillment_delay_rate",
    "support_ticket_volume",
    "order_cancellation_rate",
    "revenue",
  ];

  const orderedNodes = [
    ...CHAIN_ORDER.filter(id => nodes.find(n => n.id === id)),
    ...nodes.filter(n => !CHAIN_ORDER.includes(n.id)).map(n => n.id),
  ];

  orderedNodes.forEach((id, i) => {
    const x = 80 + (i / Math.max(1, orderedNodes.length - 1)) * (width - 160);
    const y = cy + (i % 2 === 0 ? -30 : 30);
    positions[id] = { x, y };
  });

  return (
    <div style={{
      background: "var(--bg-primary)",
      borderRadius: "var(--radius-md)",
      border: "1px solid var(--border)",
      overflow: "hidden",
    }}>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ display: "block" }}>
        {/* Links */}
        {links.map((link, i) => {
          const srcPos = positions[link.source] || positions[typeof link.source === "object" ? (link.source as any).id : link.source];
          const tgtPos = positions[link.target] || positions[typeof link.target === "object" ? (link.target as any).id : link.target];
          if (!srcPos || !tgtPos) return null;
          const color = EDGE_COLORS[link.type] || "#6366f1";
          const mx = (srcPos.x + tgtPos.x) / 2;
          const my = (srcPos.y + tgtPos.y) / 2 - 20;
          return (
            <g key={i}>
              <path
                d={`M ${srcPos.x} ${srcPos.y} Q ${mx} ${my} ${tgtPos.x} ${tgtPos.y}`}
                stroke={color}
                strokeWidth={Math.max(1, link.weight * 3)}
                fill="none"
                strokeOpacity={0.7}
                markerEnd={`url(#arrow-${link.type})`}
              />
              <text x={mx} y={my - 4} textAnchor="middle" fontSize={9} fill={color} opacity={0.8}>
                {link.type}
                {link.lag_days > 0 ? ` (t-${link.lag_days})` : ""}
              </text>
            </g>
          );
        })}

        {/* Arrow markers */}
        <defs>
          {Object.entries(EDGE_COLORS).map(([type, color]) => (
            <marker key={type} id={`arrow-${type}`} markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M0,0 L0,6 L8,3 z" fill={color} opacity={0.8} />
            </marker>
          ))}
        </defs>

        {/* Nodes */}
        {nodes.map((node) => {
          const pos = positions[node.id];
          if (!pos) return null;
          const isHighCentrality = node.centrality > 0.1;
          return (
            <g key={node.id}>
              <circle
                cx={pos.x}
                cy={pos.y}
                r={node.material ? 22 : 18}
                fill={node.material ? "rgba(245,158,11,0.15)" : "rgba(99,102,241,0.12)"}
                stroke={node.material ? "#f59e0b" : "#6366f1"}
                strokeWidth={node.material ? 2 : 1.5}
              />
              {isHighCentrality && (
                <circle
                  cx={pos.x} cy={pos.y}
                  r={node.material ? 28 : 24}
                  fill="none"
                  stroke={node.material ? "rgba(245,158,11,0.2)" : "rgba(99,102,241,0.15)"}
                  strokeWidth={1}
                  strokeDasharray="3 3"
                />
              )}
              <text x={pos.x} y={pos.y + 3} textAnchor="middle" fontSize={8} fontWeight="600" fill={node.material ? "#f59e0b" : "#818cf8"}>
                {node.label.split(" ").slice(0, 2).join(" ")}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

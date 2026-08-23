"use client";
// components/KPIHealthStrip.tsx

import React from "react";
import { RegionOverview } from "@/lib/api";

interface Props {
  regions: Record<string, RegionOverview>;
  selectedRegion: string;
  onSelect: (id: string) => void;
}

const REGION_LABELS: Record<string, string> = {
  region_a: "Region A — Pacific NW",
  region_b: "Region B — Southwest",
  region_c: "Region C — Northeast",
  region_d: "Region D — Midwest",
  region_e: "Region E — Southeast",
};

const KPI_LABELS: Record<string, string> = {
  revenue: "Revenue",
  order_cancellation_rate: "Cancellation Rate",
  fulfillment_delay_rate: "Delay Rate",
  support_ticket_volume: "Support Tickets",
  warehouse_staffing_level: "Staffing Level",
};

const KPI_FORMATS: Record<string, (v: number) => string> = {
  revenue: (v) => `$${(v / 1000).toFixed(0)}K`,
  order_cancellation_rate: (v) => `${v.toFixed(1)}%`,
  fulfillment_delay_rate: (v) => `${v.toFixed(1)}%`,
  support_ticket_volume: (v) => v.toFixed(0),
  warehouse_staffing_level: (v) => `${v.toFixed(0)}%`,
};

function TrendArrow({ trend, pct }: { trend: string; pct: number | null }) {
  const color = trend === "up" ? "var(--act)" : trend === "down" ? "var(--abstain)" : "var(--text-muted)";
  return (
    <span style={{ color, fontSize: "0.7rem", fontWeight: 600 }}>
      {trend === "up" ? "▲" : trend === "down" ? "▼" : "→"}
      {pct !== null ? ` ${Math.abs(pct).toFixed(1)}%` : ""}
    </span>
  );
}

export default function KPIHealthStrip({ regions, selectedRegion, onSelect }: Props) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <h4>Region Overview</h4>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {Object.entries(regions).map(([regionId, region]) => {
          if (region.error) return null;
          const isSelected = regionId === selectedRegion;
          const hasMaterial = region.any_material;

          return (
            <div
              key={regionId}
              id={`region-card-${regionId}`}
              onClick={() => onSelect(regionId)}
              style={{
                padding: "16px 20px",
                borderRadius: "var(--radius-md)",
                border: isSelected ? "1px solid var(--indigo)" : hasMaterial ? "1px solid var(--investigate-border)" : "1px solid var(--border)",
                background: isSelected ? "rgba(99,102,241,0.08)" : hasMaterial ? "var(--investigate-bg)" : "var(--bg-card)",
                cursor: "pointer",
                transition: "all var(--transition)",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
                <div>
                  <div style={{ fontSize: "0.8rem", fontWeight: 600, color: isSelected ? "var(--indigo-light)" : "var(--text-primary)" }}>
                    {REGION_LABELS[regionId] || regionId}
                  </div>
                  <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: 2 }}>
                    {region.data_days} days · {region.scenario?.replace(/_/g, " ")}
                  </div>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  {hasMaterial && (
                    <span style={{
                      fontSize: "0.65rem", fontWeight: 700, padding: "3px 8px",
                      borderRadius: "100px", background: "var(--investigate-bg)",
                      color: "var(--investigate)", border: "1px solid var(--investigate-border)",
                      textTransform: "uppercase", letterSpacing: "0.06em",
                    }}>
                      ⚠ Signal
                    </span>
                  )}
                  {isSelected && (
                    <span style={{ fontSize: "0.65rem", color: "var(--indigo-light)" }}>Selected</span>
                  )}
                </div>
              </div>

              {/* KPI mini-grid */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 8 }}>
                {Object.entries(region.kpi_health || {}).map(([kpi, data]) => {
                  const fmt = KPI_FORMATS[kpi] || ((v: number) => v.toString());
                  const isMaterial = data.is_material;
                  return (
                    <div key={kpi} style={{
                      padding: "8px",
                      borderRadius: "var(--radius-sm)",
                      background: isMaterial ? "rgba(245,158,11,0.08)" : "var(--bg-primary)",
                      border: isMaterial ? "1px solid var(--investigate-border)" : "1px solid transparent",
                    }}>
                      <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", marginBottom: 4, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                        {KPI_LABELS[kpi] || kpi}
                      </div>
                      <div style={{ fontSize: "0.85rem", fontWeight: 700, color: isMaterial ? "var(--investigate)" : "var(--text-primary)" }}>
                        {data.current_value !== null ? fmt(data.current_value) : "—"}
                      </div>
                      <TrendArrow trend={data.trend} pct={data.pct_change_7d} />
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

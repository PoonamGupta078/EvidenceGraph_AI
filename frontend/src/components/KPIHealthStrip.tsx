"use client";
import React from "react";
import { RegionOverview } from "@/lib/api";

interface Props {
  regions: Record<string, RegionOverview>;
  selectedRegion: string;
  onSelect: (id: string) => void;
}

const REGION_META: Record<string, { label: string; flag: string; scenario: string }> = {
  region_a: { label: "Pacific NW", flag: "🌲", scenario: "Staffing Chain" },
  region_b: { label: "Southwest", flag: "☀️", scenario: "Contradiction Promo" },
  region_c: { label: "Northeast", flag: "🏙️", scenario: "Data Quality" },
  region_d: { label: "Midwest", flag: "🌾", scenario: "Sparse History" },
  region_e: { label: "Southeast", flag: "🌊", scenario: "Multi-Factor PVM" },
};

const KPI_SHORT: Record<string, string> = {
  revenue: "Rev",
  order_cancellation_rate: "Cancel",
  fulfillment_delay_rate: "Delay",
  support_ticket_volume: "Tickets",
  warehouse_staffing_level: "Staff",
  unit_price: "Price",
  marketing_spend: "Promo",
  seasonal_index: "Season",
};

const KPI_FMT: Record<string, (v: number) => string> = {
  revenue: (v) => `$${(v / 1000).toFixed(0)}K`,
  order_cancellation_rate: (v) => `${v.toFixed(1)}%`,
  fulfillment_delay_rate: (v) => `${v.toFixed(1)}%`,
  support_ticket_volume: (v) => `${v.toFixed(0)}`,
  warehouse_staffing_level: (v) => `${v.toFixed(0)}%`,
  unit_price: (v) => `$${v.toFixed(2)}`,
  marketing_spend: (v) => `$${(v / 1000).toFixed(1)}K`,
  seasonal_index: (v) => `${v.toFixed(2)}x`,
};

export default function KPIHealthStrip({ regions, selectedRegion, onSelect }: Props) {
  const entries = Object.entries(regions).filter(([, r]) => !r.error);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <div style={{ padding: "0 4px 10px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <h4>Regions</h4>
        <span style={{ fontSize: "0.62rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
          {entries.length} active
        </span>
      </div>
      {entries.map(([regionId, region]) => {
        const isSelected = regionId === selectedRegion;
        const hasMaterial = region.any_material;
        const meta = REGION_META[regionId] || { label: regionId, flag: "📍", scenario: "" };

        return (
          <div
            key={regionId}
            id={`region-card-${regionId}`}
            onClick={() => onSelect(regionId)}
            style={{
              padding: "14px 16px",
              borderRadius: "var(--radius-md)",
              border: isSelected
                ? "1px solid var(--border-active)"
                : hasMaterial
                ? "1px solid var(--investigate-border)"
                : "1px solid var(--border)",
              background: isSelected
                ? "rgba(161,0,255,0.06)"
                : hasMaterial
                ? "var(--investigate-bg)"
                : "var(--bg-card)",
              cursor: "pointer",
              transition: "all 0.2s cubic-bezier(0.16, 1, 0.3, 1)",
              boxShadow: isSelected
                ? "0 0 0 3px rgba(161,0,255,0.1), var(--shadow-md)"
                : "var(--shadow-sm)",
              transform: isSelected ? "translateX(3px)" : "none",
              marginBottom: 8,
              position: "relative",
              overflow: "hidden",
            }}
          >
            {/* Selected indicator bar */}
            {isSelected && (
              <div style={{
                position: "absolute", left: 0, top: 0, bottom: 0, width: 3,
                background: "linear-gradient(to bottom, var(--accenture-purple), var(--accenture-purple-dark))",
                borderRadius: "3px 0 0 3px",
              }}/>
            )}

            {/* Region header */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: "1rem" }}>{meta.flag}</span>
                <div>
                  <div style={{
                    fontSize: "0.82rem", fontWeight: 700,
                    color: isSelected ? "var(--accenture-purple-light)" : "var(--text-primary)",
                  }}>
                    {regionId.replace("_", " ").replace(/\b\w/g, c => c.toUpperCase())}
                  </div>
                  <div style={{ fontSize: "0.62rem", color: "var(--text-muted)", marginTop: 1 }}>
                    {meta.label} · {meta.scenario}
                  </div>
                </div>
              </div>
              {hasMaterial && (
                <span style={{
                  fontSize: "0.58rem", fontWeight: 800, padding: "2px 7px",
                  borderRadius: 100,
                  background: "var(--investigate-bg)",
                  color: "var(--investigate)",
                  border: "1px solid var(--investigate-border)",
                  textTransform: "uppercase",
                  letterSpacing: "0.06em",
                  boxShadow: "0 0 8px var(--investigate-glow)",
                }}>
                  ⚡ Signal
                </span>
              )}
            </div>

            {/* KPI micro-tiles */}
            <div style={{
              display: "grid",
              gridTemplateColumns: `repeat(${Object.keys(region.kpi_health || {}).length || 5}, 1fr)`,
              gap: 4
            }}>
              {Object.entries(region.kpi_health || {}).map(([kpi, data]) => {
                const fmt = KPI_FMT[kpi] || ((v: number) => `${v}`);
                const isMaterial = data.is_material;
                const isUp = data.trend === "up";
                const isDown = data.trend === "down";
                const isRateKpi = kpi.includes("rate") || kpi.includes("volume");
                const trendGood = isRateKpi ? isDown : isUp;
                const trendBad = isRateKpi ? isUp : isDown;

                return (
                  <div key={kpi} style={{
                    padding: "5px 3px",
                    borderRadius: 6,
                    background: isMaterial
                      ? "rgba(255,180,0,0.08)"
                      : "var(--bg-elevated)",
                    border: isMaterial
                      ? "1px solid rgba(255,180,0,0.2)"
                      : "1px solid var(--border-subtle)",
                    textAlign: "center",
                  }}>
                    <div style={{ fontSize: "0.55rem", color: "var(--text-muted)", marginBottom: 3, letterSpacing: "0.03em" }}>
                      {KPI_SHORT[kpi] || kpi}
                    </div>
                    <div style={{
                      fontSize: "0.7rem", fontWeight: 700,
                      fontFamily: "var(--font-mono)",
                      color: isMaterial ? "var(--investigate)" : "var(--text-primary)",
                    }}>
                      {data.current_value !== null ? fmt(data.current_value) : "—"}
                    </div>
                    {data.pct_change_7d !== null && (
                      <div style={{
                        fontSize: "0.55rem", fontWeight: 600, marginTop: 2,
                        color: trendGood ? "var(--act)" : trendBad ? "var(--abstain)" : "var(--text-muted)",
                      }}>
                        {isUp ? "▲" : isDown ? "▼" : "—"}{Math.abs(data.pct_change_7d ?? 0).toFixed(1)}%
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}

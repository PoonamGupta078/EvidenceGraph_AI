"""
pipeline/evidence_graph.py
Builds a typed Evidence Graph using NetworkX.

Nodes: KPI metrics
Edges: typed causal/correlational relationships

Edge types (from kpi_contract.yaml):
  CAUSES, CORRELATES_WITH, CONTRADICTS, COMPENSATES, LAGS, INDEPENDENT

Ranking: NetworkX betweenness centrality + Pearson correlation strength.
GraphSAGE is import-guarded as a stretch enhancement.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Optional

# GraphSAGE stretch (import-guarded — won't crash if torch not installed)
try:
    import torch
    from torch_geometric.nn import SAGEConv
    HAS_GRAPHSAGE = True
except ImportError:
    HAS_GRAPHSAGE = False


KNOWN_CAUSAL_EDGES = [
    # (source_node, target_node, edge_type, lag_days, weight)
    ("warehouse_staffing_level", "fulfillment_delay_rate", "CAUSES", 2, 0.85),
    ("fulfillment_delay_rate", "support_ticket_volume", "CAUSES", 1, 0.80),
    ("support_ticket_volume", "order_cancellation_rate", "CAUSES", 1, 0.75),
    ("order_cancellation_rate", "revenue", "CAUSES", 1, 0.90),
    ("fulfillment_delay_rate", "order_cancellation_rate", "CAUSES", 2, 0.65),
]


def _pearson_correlation(df: pd.DataFrame, col_a: str, col_b: str, lag: int = 0) -> float:
    """Compute Pearson correlation between two columns with optional lag."""
    if col_a not in df.columns or col_b not in df.columns:
        return 0.0
    a = df[col_a].astype(float)
    b = df[col_b].astype(float)
    if lag > 0:
        a = a.iloc[lag:].reset_index(drop=True)
        b = b.iloc[:-lag].reset_index(drop=True)
    valid = a.notna() & b.notna()
    if valid.sum() < 5:
        return 0.0
    try:
        corr = float(np.corrcoef(a[valid], b[valid])[0, 1])
        return round(corr if not np.isnan(corr) else 0.0, 4)
    except Exception:
        return 0.0


def _betweenness_centrality(G: nx.DiGraph) -> Dict[str, float]:
    """Compute normalized betweenness centrality for all nodes."""
    if len(G.nodes) == 0:
        return {}
    bc = nx.betweenness_centrality(G, normalized=True, weight="weight")
    return {k: round(v, 4) for k, v in bc.items()}


def _degree_centrality(G: nx.DiGraph) -> Dict[str, float]:
    """In-degree + out-degree centrality."""
    if len(G.nodes) == 0:
        return {}
    in_deg = dict(G.in_degree(weight="weight"))
    out_deg = dict(G.out_degree(weight="weight"))
    nodes = set(in_deg) | set(out_deg)
    return {n: round((in_deg.get(n, 0) + out_deg.get(n, 0)), 4) for n in nodes}


def build_evidence_graph(
    df: pd.DataFrame,
    material_kpis: List[str],
    region_id: str,
    scenario: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Builds a typed evidence graph for the given region.

    Args:
        df: Aligned, reconciled DataFrame
        material_kpis: KPIs flagged as material
        region_id: e.g. "region_a"
        scenario: optional scenario hint (e.g. "contradiction_promo")

    Returns dict with:
        - graph_data: {nodes, edges} for frontend visualization
        - driver_ranking: ordered list of root driver candidates
        - centrality_scores: {kpi: score}
        - correlation_matrix: {kpi_pair: correlation}
        - graphsage_available: bool
    """
    G = nx.DiGraph()

    all_kpis = [
        "warehouse_staffing_level",
        "fulfillment_delay_rate",
        "support_ticket_volume",
        "order_cancellation_rate",
        "revenue",
    ]

    # Add nodes
    for kpi in all_kpis:
        if kpi in df.columns:
            G.add_node(kpi, material=(kpi in material_kpis), kpi_id=kpi)

    # Add known causal edges
    correlation_matrix = {}
    active_edges = []

    for src, tgt, edge_type, lag, base_weight in KNOWN_CAUSAL_EDGES:
        if src not in G.nodes or tgt not in G.nodes:
            continue

        # Measure actual correlation to validate the edge
        corr = _pearson_correlation(df, src, tgt, lag=lag)
        correlation_matrix[f"{src}→{tgt}"] = corr

        # Edge weight = base_weight * |correlation| 
        edge_weight = round(base_weight * abs(corr), 4) if corr != 0 else base_weight * 0.3

        # Contradiction detection for Region B scenario
        effective_edge_type = edge_type
        if scenario == "contradiction_promo" and tgt == "revenue" and src == "order_cancellation_rate":
            if "promo_discount_pct" in df.columns:
                promo_active = df["promo_discount_pct"].iloc[-30:].mean() > 5
                if promo_active:
                    effective_edge_type = "COMPENSATES"

        G.add_edge(
            src, tgt,
            edge_type=effective_edge_type,
            lag_days=lag,
            weight=edge_weight,
            correlation=corr,
        )
        active_edges.append({
            "source": src,
            "target": tgt,
            "type": effective_edge_type,
            "lag_days": lag,
            "weight": edge_weight,
            "correlation": corr,
        })

    # Add INDEPENDENT edges for Region E (multi-factor PVM)
    if scenario == "multi_factor_pvm":
        pvm_drivers = ["price_effect_usd", "marketing_effect_usd", "seasonal_effect_usd"]
        for driver in pvm_drivers:
            if driver in df.columns:
                G.add_node(driver, material=True, kpi_id=driver)
                corr = _pearson_correlation(df, driver, "revenue")
                correlation_matrix[f"{driver}→revenue"] = corr
                G.add_edge(
                    driver, "revenue",
                    edge_type="INDEPENDENT",
                    lag_days=0,
                    weight=abs(corr),
                    correlation=corr,
                )
                active_edges.append({
                    "source": driver,
                    "target": "revenue",
                    "type": "INDEPENDENT",
                    "lag_days": 0,
                    "weight": abs(corr),
                    "correlation": corr,
                })

    # Centrality-based driver ranking
    betweenness = _betweenness_centrality(G)
    degree = _degree_centrality(G)

    # Combined ranking score: 0.5 * betweenness + 0.3 * degree + 0.2 * materiality_bonus
    all_nodes = list(G.nodes)
    driver_scores = {}
    for node in all_nodes:
        bc_score = betweenness.get(node, 0.0)
        deg_score = min(degree.get(node, 0.0) / max(degree.values(), default=1), 1.0) if degree else 0.0
        mat_bonus = 0.2 if node in material_kpis else 0.0
        # Correlation contribution: max absolute correlation to revenue
        corr_to_revenue = abs(correlation_matrix.get(f"{node}→revenue", 0.0))
        driver_scores[node] = round(0.4 * bc_score + 0.25 * deg_score + 0.2 * mat_bonus + 0.15 * corr_to_revenue, 4)

    driver_ranking = sorted(driver_scores.items(), key=lambda x: x[1], reverse=True)
    driver_ranking = [{"kpi": k, "score": v, "is_material": k in material_kpis} for k, v in driver_ranking]

    # Serialize graph for frontend (react-force-graph-2d compatible)
    graph_nodes = []
    for node in G.nodes(data=True):
        graph_nodes.append({
            "id": node[0],
            "label": node[0].replace("_", " ").title(),
            "material": node[1].get("material", False),
            "centrality": betweenness.get(node[0], 0.0),
        })

    return {
        "graph_data": {
            "nodes": graph_nodes,
            "links": active_edges,
        },
        "driver_ranking": driver_ranking,
        "centrality_scores": {
            "betweenness": betweenness,
            "degree": degree,
        },
        "correlation_matrix": correlation_matrix,
        "graphsage_available": HAS_GRAPHSAGE,
        "edge_count": len(active_edges),
        "node_count": len(graph_nodes),
    }

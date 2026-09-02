"""
pipeline/evidence_graph.py

Builds a typed Evidence Graph and produces a ranked driver list.

Nodes represent observed KPI metrics and PVM contribution items.
Edges carry domain-prior weights combined with empirical lagged correlations.

Edge types:
    BUSINESS_RULE_PRIOR  — domain-knowledge structural relationship
    COMPENSATES          — detected compensation mechanism (contradiction scenario)
    TRANSFORMS_TO        — observed KPI -> PVM attribution node
    PVM_ATTRIBUTION      — PVM node -> revenue

The resulting driver ranking is a heuristic evidence score, not causal proof.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import networkx as nx

from typing import Dict, Any, List, Optional, Tuple


try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv  # pyrefly: ignore [missing-import]
    HAS_TORCH_SAGE = True
except ImportError:
    HAS_TORCH_SAGE = False


# Domain priors: (source, target, edge_type, lag_days, prior_weight)
# These encode domain knowledge, not learned causal relationships.
KNOWN_BUSINESS_PRIORS = [
    ("warehouse_staffing_level", "fulfillment_delay_rate", "BUSINESS_RULE_PRIOR", 2, 0.85),
    ("fulfillment_delay_rate", "support_ticket_volume", "BUSINESS_RULE_PRIOR", 1, 0.80),
    ("support_ticket_volume", "order_cancellation_rate", "BUSINESS_RULE_PRIOR", 1, 0.75),
    ("order_cancellation_rate", "revenue", "BUSINESS_RULE_PRIOR", 1, 0.90),
    ("fulfillment_delay_rate", "order_cancellation_rate", "BUSINESS_RULE_PRIOR", 2, 0.65),
]


def _pearson_correlation(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    lag: int = 0,
) -> float:
    """
    Pearson correlation between two columns with optional lag.

    For lag > 0, source[t] is aligned with target[t + lag].
    """

    if col_a not in df.columns or col_b not in df.columns:
        return 0.0

    try:
        a = df[col_a].astype(float)
        b = df[col_b].astype(float)
    except (TypeError, ValueError):
        return 0.0

    if lag > 0:
        if len(df) <= lag:
            return 0.0
        a = a.iloc[:-lag].reset_index(drop=True)
        b = b.iloc[lag:].reset_index(drop=True)

    valid = a.notna() & b.notna()
    if valid.sum() < 5:
        return 0.0

    try:
        corr = float(np.corrcoef(a.loc[valid], b.loc[valid])[0, 1])
        return 0.0 if np.isnan(corr) else round(corr, 4)
    except Exception:
        return 0.0


def _betweenness_centrality(G: nx.DiGraph) -> Dict[str, float]:
    """Normalized betweenness centrality using 1/weight as edge distance."""

    if len(G.nodes) == 0:
        return {}

    for _, _, data in G.edges(data=True):
        weight = float(data.get("weight", 0.5))
        data["distance"] = 1.0 / max(weight, 1e-6)

    bc = nx.betweenness_centrality(G, normalized=True, weight="distance")
    return {node: round(float(score), 4) for node, score in bc.items()}


def _degree_centrality(G: nx.DiGraph) -> Dict[str, float]:
    """Weighted in-degree + weighted out-degree per node."""

    if len(G.nodes) == 0:
        return {}

    in_deg = dict(G.in_degree(weight="weight"))
    out_deg = dict(G.out_degree(weight="weight"))
    nodes = set(in_deg) | set(out_deg)

    return {
        node: round(float(in_deg.get(node, 0.0) + out_deg.get(node, 0.0)), 4)
        for node in nodes
    }


def _compute_graph_message_passing(
    G: nx.DiGraph,
    correlation_matrix: Dict[str, float],
) -> Tuple[Dict[str, List[float]], Dict[str, float], str]:
    """
    Compute a graph message-passing representation as a supporting ranking signal.

    Uses PyTorch Geometric GraphSAGE when available (deterministic, untrained,
    seed=42), otherwise falls back to a deterministic NumPy two-layer message pass.
    The score is only a supporting feature — it does not establish causality.
    """

    nodes = list(G.nodes())
    if not nodes:
        return {}, {}, "NONE"

    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n_nodes = len(nodes)

    # Node features: [materiality, out_degree, in_degree, |corr_to_revenue|]
    features = []
    for node in nodes:
        material = 1.0 if G.nodes[node].get("material", False) else 0.0
        out_degree = float(G.out_degree(node, weight="weight"))
        in_degree = float(G.in_degree(node, weight="weight"))
        corr_to_revenue = abs(correlation_matrix.get(f"{node}→revenue", 0.0))
        features.append([material, out_degree, in_degree, corr_to_revenue])

    X = np.asarray(features, dtype=np.float32)

    # Symmetric adjacency with self-loops
    A = np.eye(n_nodes, dtype=np.float32)
    for source, target, data in G.edges(data=True):
        i = node_to_idx[source]
        j = node_to_idx[target]
        weight = float(data.get("weight", 0.5))
        A[i, j] += weight
        A[j, i] += weight * 0.5  # allows information to flow back through directed edges

    degree = np.sum(A, axis=1)
    degree_inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    valid_degree = degree > 0
    degree_inv_sqrt[valid_degree] = degree[valid_degree] ** -0.5
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    embeddings = None
    engine_label = None

    if HAS_TORCH_SAGE:
        try:
            edge_indices = []
            for source, target in G.edges():
                i = node_to_idx[source]
                j = node_to_idx[target]
                edge_indices.append([i, j])
                edge_indices.append([j, i])

            edge_index_tensor = (
                torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                if edge_indices
                else torch.zeros((2, 0), dtype=torch.long)
            )
            x_tensor = torch.tensor(X, dtype=torch.float32)

            class SAGEModel(torch.nn.Module):
                def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
                    super().__init__()
                    self.conv1 = SAGEConv(input_dim, hidden_dim)
                    self.conv2 = SAGEConv(hidden_dim, output_dim)

                def forward(self, x, edge_index):
                    x = F.relu(self.conv1(x, edge_index))
                    return self.conv2(x, edge_index)

            torch.manual_seed(42)
            model = SAGEModel(input_dim=4, hidden_dim=8, output_dim=8)
            model.eval()

            with torch.no_grad():
                embeddings = model(x_tensor, edge_index_tensor).cpu().numpy()

            engine_label = "PyTorch Geometric GraphSAGE (Untrained Message-Passing Signal)"
        except Exception:
            embeddings = None

    if embeddings is None:
        rng = np.random.default_rng(42)
        W1 = rng.normal(size=(4, 8)).astype(np.float32) * 0.5
        W2 = rng.normal(size=(8, 8)).astype(np.float32) * 0.5
        H1 = np.maximum(0.0, A_norm @ X @ W1)
        embeddings = np.maximum(0.0, A_norm @ H1 @ W2)
        engine_label = "Native NumPy Graph Message-Passing Signal"

    raw_scores = np.linalg.norm(embeddings, axis=1)
    max_score = float(np.max(raw_scores))
    if max_score <= 0:
        max_score = 1.0

    embedding_dict = {}
    score_dict = {}
    for idx, node in enumerate(nodes):
        embedding_dict[node] = [round(float(v), 4) for v in embeddings[idx]]
        score_dict[node] = round(float(raw_scores[idx] / max_score), 4)

    return embedding_dict, score_dict, engine_label


def build_evidence_graph(
    df: pd.DataFrame,
    material_kpis: List[str],
    region_id: str,
    scenario: Optional[str] = None,
    pvm_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the typed Evidence Graph and return a ranked driver list.

    Combines domain-prior edges, empirical lagged correlations, PVM contribution
    nodes, graph centrality, path influence, and a message-passing signal into
    a heuristic driver ranking. The result is evidence, not causal proof.
    """

    G = nx.DiGraph()

    all_kpis = [
        "warehouse_staffing_level",
        "fulfillment_delay_rate",
        "support_ticket_volume",
        "order_cancellation_rate",
        "revenue",
        "unit_price",
        "marketing_spend",
        "seasonal_index",
        "quantity",
    ]

    for kpi in all_kpis:
        if kpi in df.columns:
            G.add_node(kpi, material=(kpi in material_kpis), kpi_id=kpi, evidence_type="OBSERVED_METRIC")

    correlation_matrix: Dict[str, float] = {}
    active_edges: List[Dict[str, Any]] = []

    # Business-rule prior edges
    for source, target, edge_type, lag_days, prior_weight in KNOWN_BUSINESS_PRIORS:
        if source not in G.nodes or target not in G.nodes:
            continue

        correlation = _pearson_correlation(df, source, target, lag=lag_days)
        correlation_matrix[f"{source}→{target}"] = correlation

        abs_corr = abs(correlation)
        if abs_corr >= 0.5:
            empirical_support = "STRONG"
        elif abs_corr > 0.1:
            empirical_support = "WEAK"
        else:
            empirical_support = "NONE"

        # Prior remains even with zero observed correlation — it is a domain prior.
        edge_weight = round(prior_weight * abs_corr if correlation != 0 else prior_weight * 0.3, 4)
        effective_edge_type = edge_type

        # Detect active promotion compensating for cancellation revenue loss.
        if (
            scenario in ("contradiction_promo", "contradictory_evidence")
            and source == "order_cancellation_rate"
            and target == "revenue"
        ):
            discount_column = (
                "promo_discount_pct" if "promo_discount_pct" in df.columns
                else "discount" if "discount" in df.columns
                else None
            )
            if discount_column is not None:
                recent_window = min(15, len(df))
                baseline_window = min(30, len(df))

                if recent_window > 0 and baseline_window > 0:
                    promo_active = bool(df[discount_column].iloc[-recent_window:].mean() > 5)
                    cancellation_up = bool(
                        df["order_cancellation_rate"].iloc[-recent_window:].mean()
                        > df["order_cancellation_rate"].iloc[:baseline_window].mean()
                    )
                    revenue_not_down = bool(
                        df["revenue"].iloc[-recent_window:].mean()
                        >= df["revenue"].iloc[:baseline_window].mean()
                    )
                    if promo_active and cancellation_up and revenue_not_down:
                        effective_edge_type = "COMPENSATES"

        G.add_edge(
            source, target,
            edge_type=effective_edge_type,
            lag_days=lag_days,
            prior_weight=prior_weight,
            weight=edge_weight,
            distance=1.0 / max(edge_weight, 1e-6),
            correlation=correlation,
            empirical_support=empirical_support,
        )
        active_edges.append({
            "source": source,
            "target": target,
            "type": effective_edge_type,
            "lag_days": lag_days,
            "prior_weight": prior_weight,
            "weight": edge_weight,
            "correlation": correlation,
            "empirical_support": empirical_support,
        })

    # PVM contribution nodes: observed driver -> pvm_node -> revenue
    if pvm_result and pvm_result.get("status") == "OK":
        components = pvm_result.get("components", {})
        driver_mapping = {
            "price": "unit_price",
            "marketing": "marketing_spend",
            "seasonal": "seasonal_index",
            "volume": "quantity",
        }

        for component_name, value_usd in components.items():
            if component_name == "mix":
                continue
            try:
                value_usd = float(value_usd)
            except (TypeError, ValueError):
                continue
            if abs(value_usd) <= 100:
                continue

            pvm_node = f"pvm_{component_name}"
            G.add_node(pvm_node, material=True, kpi_id=pvm_node, evidence_type="PVM_CONTRIBUTION", contribution_usd=value_usd)

            raw_driver = driver_mapping.get(component_name)
            if raw_driver and raw_driver in G.nodes:
                G.add_edge(raw_driver, pvm_node, edge_type="TRANSFORMS_TO", lag_days=0, weight=0.9, distance=1.0 / 0.9, correlation=1.0)
                active_edges.append({"source": raw_driver, "target": pvm_node, "type": "TRANSFORMS_TO", "lag_days": 0, "weight": 0.9, "correlation": 1.0})

            contribution_direction = -1.0 if value_usd < 0 else 1.0
            G.add_edge(pvm_node, "revenue", edge_type="PVM_ATTRIBUTION", lag_days=0, weight=0.85, distance=1.0 / 0.85, correlation=contribution_direction, contribution_usd=value_usd)
            active_edges.append({"source": pvm_node, "target": "revenue", "type": "PVM_ATTRIBUTION", "lag_days": 0, "weight": 0.85, "correlation": contribution_direction, "contribution_usd": value_usd})

    graph_embeddings, graph_scores, engine_label = _compute_graph_message_passing(G, correlation_matrix)
    betweenness = _betweenness_centrality(G)
    degree = _degree_centrality(G)

    path_influence: Dict[str, float] = {}
    if "revenue" in G.nodes:
        for node in G.nodes():
            if node == "revenue":
                path_influence[node] = 0.0
                continue
            try:
                path_length = nx.shortest_path_length(G, source=node, target="revenue", weight="distance")
                path_influence[node] = round(1.0 / max(1.0, float(path_length)), 4)
            except nx.NetworkXNoPath:
                path_influence[node] = 0.0
    else:
        path_influence = {node: 0.0 for node in G.nodes()}

    max_degree = max(degree.values(), default=1.0)
    driver_scores: Dict[str, float] = {}

    for node in G.nodes():
        if node == "revenue":
            continue

        bc_score = betweenness.get(node, 0.0)
        degree_score = min(degree.get(node, 0.0) / max_degree, 1.0)
        g_score = graph_scores.get(node, 0.0)
        path_score = path_influence.get(node, 0.0)
        material_bonus = 0.2 if (node in material_kpis or node.startswith("pvm_")) else 0.0

        driver_scores[node] = round(
            0.10 * g_score
            + 0.35 * bc_score
            + 0.20 * path_score
            + 0.20 * degree_score
            + 0.15 * material_bonus,
            4,
        )

    driver_ranking = [
        {
            "kpi": kpi,
            "score": score,
            "graph_score": graph_scores.get(kpi, 0.0),
            "path_influence": path_influence.get(kpi, 0.0),
            "is_material": (kpi in material_kpis or kpi.startswith("pvm_")),
        }
        for kpi, score in sorted(driver_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    graph_nodes = [
        {
            "id": node,
            "label": node.replace("pvm_", "PVM ").replace("_", " ").title(),
            "material": data.get("material", False),
            "evidence_type": data.get("evidence_type", "OBSERVED_METRIC"),
            "contribution_usd": data.get("contribution_usd", 0.0),
            "centrality": betweenness.get(node, 0.0),
            "path_influence": path_influence.get(node, 0.0),
            "graph_score": graph_scores.get(node, 0.0),
            "graph_embedding": graph_embeddings.get(node, []),
        }
        for node, data in G.nodes(data=True)
    ]

    return {
        "graph_metadata": {
            "region_id": region_id,
            "scenario": scenario,
            "engine": engine_label,
            "causal_claim": False,
        },
        "graph_data": {
            "nodes": graph_nodes,
            "links": active_edges,
        },
        "driver_ranking": driver_ranking,
        "centrality_scores": {
            "betweenness": betweenness,
            "degree": degree,
            "path_influence": path_influence,
        },
        "graph_embeddings": graph_embeddings,
        "graph_scores": graph_scores,
        "correlation_matrix": correlation_matrix,
        "graphsage_available": HAS_TORCH_SAGE,
        "edge_count": len(active_edges),
        "node_count": len(graph_nodes),
    }
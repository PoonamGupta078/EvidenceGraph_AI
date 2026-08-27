"""
pipeline/evidence_graph.py

Builds a typed Evidence Graph using NetworkX and computes
graph-based supporting signals for root-cause ranking.

Nodes:
    - Observed KPI metrics
    - PVM contribution nodes

Edges:
    - BUSINESS_RULE_PRIOR
    - COMPENSATES
    - TRANSFORMS_TO
    - PVM_ATTRIBUTION

Graph signals:
    - Betweenness centrality
    - Weighted degree
    - Path influence to revenue
    - Untrained graph message-passing signal

IMPORTANT:
    This graph is an evidence/ranking structure.
    It does NOT establish causal proof.

    GraphSAGE, when available, is used only as an
    untrained message-passing feature signal.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import networkx as nx

from typing import Dict, Any, List, Optional, Tuple


# ---------------------------------------------------------------------
# Optional PyTorch Geometric support
# ---------------------------------------------------------------------

try:
    import torch
    import torch.nn.functional as F

    # pyrefly: ignore [missing-import]
    from torch_geometric.nn import SAGEConv

    HAS_TORCH_SAGE = True

except ImportError:
    HAS_TORCH_SAGE = False


# ---------------------------------------------------------------------
# Business-rule priors
#
# These are domain priors, NOT learned causal relationships.
#
# source, target, edge_type, lag_days, prior_weight
# ---------------------------------------------------------------------

KNOWN_BUSINESS_PRIORS = [
    (
        "warehouse_staffing_level",
        "fulfillment_delay_rate",
        "BUSINESS_RULE_PRIOR",
        2,
        0.85,
    ),
    (
        "fulfillment_delay_rate",
        "support_ticket_volume",
        "BUSINESS_RULE_PRIOR",
        1,
        0.80,
    ),
    (
        "support_ticket_volume",
        "order_cancellation_rate",
        "BUSINESS_RULE_PRIOR",
        1,
        0.75,
    ),
    (
        "order_cancellation_rate",
        "revenue",
        "BUSINESS_RULE_PRIOR",
        1,
        0.90,
    ),
    (
        "fulfillment_delay_rate",
        "order_cancellation_rate",
        "BUSINESS_RULE_PRIOR",
        2,
        0.65,
    ),
]


# ---------------------------------------------------------------------
# Pearson correlation
# ---------------------------------------------------------------------

def _pearson_correlation(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    lag: int = 0,
) -> float:
    """
    Compute Pearson correlation between two columns with optional lag.

    For lag > 0:
        source[t] is aligned with target[t + lag].
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
        corr = float(
            np.corrcoef(
                a.loc[valid],
                b.loc[valid],
            )[0, 1]
        )

        if np.isnan(corr):
            return 0.0

        return round(corr, 4)

    except Exception:
        return 0.0


# ---------------------------------------------------------------------
# Centrality
# ---------------------------------------------------------------------

def _betweenness_centrality(
    G: nx.DiGraph,
) -> Dict[str, float]:
    """
    Compute normalized betweenness centrality.

    NetworkX interprets edge weight as distance when calculating
    shortest paths, so relationship strength is converted to:

        distance = 1 / weight
    """

    if len(G.nodes) == 0:
        return {}

    for _, _, data in G.edges(data=True):
        weight = float(data.get("weight", 0.5))
        data["distance"] = 1.0 / max(weight, 1e-6)

    bc = nx.betweenness_centrality(
        G,
        normalized=True,
        weight="distance",
    )

    return {
        node: round(float(score), 4)
        for node, score in bc.items()
    }


def _degree_centrality(
    G: nx.DiGraph,
) -> Dict[str, float]:
    """
    Weighted in-degree + weighted out-degree.
    """

    if len(G.nodes) == 0:
        return {}

    in_degree = dict(
        G.in_degree(weight="weight")
    )

    out_degree = dict(
        G.out_degree(weight="weight")
    )

    nodes = set(in_degree) | set(out_degree)

    return {
        node: round(
            float(
                in_degree.get(node, 0.0)
                + out_degree.get(node, 0.0)
            ),
            4,
        )
        for node in nodes
    }


# ---------------------------------------------------------------------
# Untrained graph message passing
# ---------------------------------------------------------------------

def _compute_gnn_embeddings(
    G: nx.DiGraph,
    correlation_matrix: Dict[str, float],
) -> Tuple[
    Dict[str, List[float]],
    Dict[str, float],
    str,
]:
    """
    Compute an untrained graph message-passing representation.

    This is NOT a trained predictive GNN.

    If PyTorch Geometric is available:
        use deterministic, untrained GraphSAGE.

    Otherwise:
        use deterministic NumPy message passing.

    The resulting score is only a supporting ranking feature.
    """

    nodes = list(G.nodes())

    if not nodes:
        return {}, {}, "NONE"

    node_to_idx = {
        node: idx
        for idx, node in enumerate(nodes)
    }

    n_nodes = len(nodes)

    # -------------------------------------------------------------
    # Initial node features
    #
    # 1. materiality
    # 2. weighted out-degree
    # 3. weighted in-degree
    # 4. absolute correlation with revenue
    # -------------------------------------------------------------

    features = []

    for node in nodes:

        material = (
            1.0
            if G.nodes[node].get("material", False)
            else 0.0
        )

        out_degree = float(
            G.out_degree(
                node,
                weight="weight",
            )
        )

        in_degree = float(
            G.in_degree(
                node,
                weight="weight",
            )
        )

        correlation_to_revenue = abs(
            correlation_matrix.get(
                f"{node}→revenue",
                0.0,
            )
        )

        features.append(
            [
                material,
                out_degree,
                in_degree,
                correlation_to_revenue,
            ]
        )

    X = np.asarray(
        features,
        dtype=np.float32,
    )

    # -------------------------------------------------------------
    # Build symmetric adjacency with self-loops
    # -------------------------------------------------------------

    A = np.eye(
        n_nodes,
        dtype=np.float32,
    )

    for source, target, data in G.edges(data=True):

        i = node_to_idx[source]
        j = node_to_idx[target]

        weight = float(
            data.get("weight", 0.5)
        )

        A[i, j] += weight

        # Reverse signal allows information to propagate
        # through the directed business graph.
        A[j, i] += weight * 0.5

    # -------------------------------------------------------------
    # Normalize adjacency
    # -------------------------------------------------------------

    degree = np.sum(A, axis=1)

    degree_inv_sqrt = np.zeros_like(
        degree,
        dtype=np.float32,
    )

    valid_degree = degree > 0

    degree_inv_sqrt[valid_degree] = (
        degree[valid_degree] ** -0.5
    )

    D_inv_sqrt = np.diag(
        degree_inv_sqrt
    )

    A_norm = (
        D_inv_sqrt
        @ A
        @ D_inv_sqrt
    )

    embeddings = None
    gnn_engine = None

    # -------------------------------------------------------------
    # Optional GraphSAGE
    # -------------------------------------------------------------

    if HAS_TORCH_SAGE:

        try:
            edge_indices = []

            for source, target in G.edges():

                i = node_to_idx[source]
                j = node_to_idx[target]

                edge_indices.append([i, j])
                edge_indices.append([j, i])

            if edge_indices:

                edge_index_tensor = torch.tensor(
                    edge_indices,
                    dtype=torch.long,
                ).t().contiguous()

            else:

                edge_index_tensor = torch.zeros(
                    (2, 0),
                    dtype=torch.long,
                )

            x_tensor = torch.tensor(
                X,
                dtype=torch.float32,
            )

            class SAGEModel(torch.nn.Module):

                def __init__(
                    self,
                    input_dim: int,
                    hidden_dim: int,
                    output_dim: int,
                ):
                    super().__init__()

                    self.conv1 = SAGEConv(
                        input_dim,
                        hidden_dim,
                    )

                    self.conv2 = SAGEConv(
                        hidden_dim,
                        output_dim,
                    )

                def forward(
                    self,
                    x,
                    edge_index,
                ):
                    x = self.conv1(
                        x,
                        edge_index,
                    )

                    x = F.relu(x)

                    x = self.conv2(
                        x,
                        edge_index,
                    )

                    return x

            # Deterministic initialization.
            torch.manual_seed(42)

            model = SAGEModel(
                input_dim=4,
                hidden_dim=8,
                output_dim=8,
            )

            model.eval()

            with torch.no_grad():

                embeddings = (
                    model(
                        x_tensor,
                        edge_index_tensor,
                    )
                    .cpu()
                    .numpy()
                )

            gnn_engine = (
                "PyTorch Geometric GraphSAGE "
                "(Untrained Message-Passing Signal)"
            )

        except Exception:

            embeddings = None

    # -------------------------------------------------------------
    # Deterministic NumPy fallback
    # -------------------------------------------------------------

    if embeddings is None:

        rng = np.random.default_rng(42)

        W1 = (
            rng.normal(
                size=(4, 8)
            ).astype(np.float32)
            * 0.5
        )

        W2 = (
            rng.normal(
                size=(8, 8)
            ).astype(np.float32)
            * 0.5
        )

        H1 = np.maximum(
            0.0,
            A_norm @ X @ W1,
        )

        embeddings = np.maximum(
            0.0,
            A_norm @ H1 @ W2,
        )

        gnn_engine = (
            "Native NumPy Graph "
            "Message-Passing Signal"
        )

    # -------------------------------------------------------------
    # Normalize embedding magnitude into supporting score
    # -------------------------------------------------------------

    raw_scores = np.linalg.norm(
        embeddings,
        axis=1,
    )

    max_score = float(
        np.max(raw_scores)
    )

    if max_score <= 0:
        max_score = 1.0

    embedding_dict = {}
    score_dict = {}

    for idx, node in enumerate(nodes):

        embedding_dict[node] = [
            round(float(value), 4)
            for value in embeddings[idx]
        ]

        score_dict[node] = round(
            float(
                raw_scores[idx]
                / max_score
            ),
            4,
        )

    return (
        embedding_dict,
        score_dict,
        gnn_engine,
    )


# ---------------------------------------------------------------------
# Main Evidence Graph builder
# ---------------------------------------------------------------------

def build_evidence_graph(
    df: pd.DataFrame,
    material_kpis: List[str],
    region_id: str,
    scenario: Optional[str] = None,
    pvm_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the typed Evidence Graph.

    The graph combines:

        1. Observed KPI nodes
        2. Business-rule prior relationships
        3. Empirical lagged correlation support
        4. PVM contribution nodes
        5. Path influence toward revenue
        6. Centrality
        7. Untrained message-passing signal

    The resulting ranking is a heuristic evidence score.
    It is NOT causal proof and NOT a probability.
    """

    G = nx.DiGraph()

    # -------------------------------------------------------------
    # Observed KPI nodes
    # -------------------------------------------------------------

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

            G.add_node(
                kpi,
                material=(kpi in material_kpis),
                kpi_id=kpi,
                evidence_type="OBSERVED_METRIC",
            )

    correlation_matrix: Dict[str, float] = {}
    active_edges: List[Dict[str, Any]] = []

    # -------------------------------------------------------------
    # 1. Business-rule prior edges
    # -------------------------------------------------------------

    for (
        source,
        target,
        edge_type,
        lag_days,
        prior_weight,
    ) in KNOWN_BUSINESS_PRIORS:

        if (
            source not in G.nodes
            or target not in G.nodes
        ):
            continue

        correlation = _pearson_correlation(
            df,
            source,
            target,
            lag=lag_days,
        )

        correlation_matrix[
            f"{source}→{target}"
        ] = correlation

        abs_corr = abs(correlation)

        if abs_corr >= 0.5:
            empirical_support = "STRONG"
        elif abs_corr > 0.1:
            empirical_support = "WEAK"
        else:
            empirical_support = "NONE"

        # Prior remains present even if observed correlation
        # is zero. This is a domain prior, not causal proof.
        if correlation != 0:
            edge_weight = round(
                prior_weight * abs_corr,
                4,
            )
        else:
            edge_weight = round(
                prior_weight * 0.3,
                4,
            )

        effective_edge_type = edge_type

        # ---------------------------------------------------------
        # Contradiction / compensation scenario
        # ---------------------------------------------------------

        if (
            scenario in ("contradiction_promo", "contradictory_evidence")
            and source == "order_cancellation_rate"
            and target == "revenue"
        ):

            discount_column = None

            if "promo_discount_pct" in df.columns:
                discount_column = "promo_discount_pct"

            elif "discount" in df.columns:
                discount_column = "discount"

            if discount_column is not None:

                recent_window = min(
                    15,
                    len(df),
                )

                baseline_window = min(
                    30,
                    len(df),
                )

                if (
                    recent_window > 0
                    and baseline_window > 0
                ):

                    promo_active = bool(
                        df[discount_column]
                        .iloc[-recent_window:]
                        .mean()
                        > 5
                    )

                    cancellation_up = bool(
                        df["order_cancellation_rate"]
                        .iloc[-recent_window:]
                        .mean()
                        >
                        df["order_cancellation_rate"]
                        .iloc[:baseline_window]
                        .mean()
                    )

                    revenue_not_down = bool(
                        df["revenue"]
                        .iloc[-recent_window:]
                        .mean()
                        >=
                        df["revenue"]
                        .iloc[:baseline_window]
                        .mean()
                    )

                    if (
                        promo_active
                        and cancellation_up
                        and revenue_not_down
                    ):
                        effective_edge_type = "COMPENSATES"

        G.add_edge(
            source,
            target,
            edge_type=effective_edge_type,
            lag_days=lag_days,
            prior_weight=prior_weight,
            weight=edge_weight,
            distance=1.0 / max(
                edge_weight,
                1e-6,
            ),
            correlation=correlation,
            empirical_support=empirical_support,
        )

        active_edges.append(
            {
                "source": source,
                "target": target,
                "type": effective_edge_type,
                "lag_days": lag_days,
                "prior_weight": prior_weight,
                "weight": edge_weight,
                "correlation": correlation,
                "empirical_support": empirical_support,
            }
        )

    # -------------------------------------------------------------
    # 2. PVM contribution nodes
    #
    # Observed driver → PVM contribution → revenue
    # -------------------------------------------------------------

    if (
        pvm_result
        and pvm_result.get("status") == "OK"
    ):

        components = pvm_result.get(
            "components",
            {},
        )

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

            # Same materiality cutoff used by root_cause.py.
            if abs(value_usd) <= 100:
                continue

            pvm_node = f"pvm_{component_name}"

            G.add_node(
                pvm_node,
                material=True,
                kpi_id=pvm_node,
                evidence_type="PVM_CONTRIBUTION",
                contribution_usd=value_usd,
            )

            raw_driver = driver_mapping.get(
                component_name
            )

            # Observed driver → PVM attribution node
            if (
                raw_driver
                and raw_driver in G.nodes
            ):

                G.add_edge(
                    raw_driver,
                    pvm_node,
                    edge_type="TRANSFORMS_TO",
                    lag_days=0,
                    weight=0.9,
                    distance=1.0 / 0.9,
                    correlation=1.0,
                )

                active_edges.append(
                    {
                        "source": raw_driver,
                        "target": pvm_node,
                        "type": "TRANSFORMS_TO",
                        "lag_days": 0,
                        "weight": 0.9,
                        "correlation": 1.0,
                    }
                )

            # PVM attribution → revenue
            contribution_direction = (
                -1.0
                if value_usd < 0
                else 1.0
            )

            G.add_edge(
                pvm_node,
                "revenue",
                edge_type="PVM_ATTRIBUTION",
                lag_days=0,
                weight=0.85,
                distance=1.0 / 0.85,
                correlation=contribution_direction,
                contribution_usd=value_usd,
            )

            active_edges.append(
                {
                    "source": pvm_node,
                    "target": "revenue",
                    "type": "PVM_ATTRIBUTION",
                    "lag_days": 0,
                    "weight": 0.85,
                    "correlation": contribution_direction,
                    "contribution_usd": value_usd,
                }
            )

    # -------------------------------------------------------------
    # 3. Graph message-passing supporting signal
    # -------------------------------------------------------------

    (
        gnn_embeddings,
        gnn_scores,
        gnn_engine,
    ) = _compute_gnn_embeddings(
        G,
        correlation_matrix,
    )

    # -------------------------------------------------------------
    # 4. Centrality
    # -------------------------------------------------------------

    betweenness = _betweenness_centrality(G)
    degree = _degree_centrality(G)

    # -------------------------------------------------------------
    # 5. Path influence toward revenue
    # -------------------------------------------------------------

    path_influence: Dict[str, float] = {}

    if "revenue" in G.nodes:

        for node in G.nodes():

            if node == "revenue":
                path_influence[node] = 0.0
                continue

            try:

                path_length = nx.shortest_path_length(
                    G,
                    source=node,
                    target="revenue",
                    weight="distance",
                )

                path_influence[node] = round(
                    1.0 / max(
                        1.0,
                        float(path_length),
                    ),
                    4,
                )

            except nx.NetworkXNoPath:

                path_influence[node] = 0.0

    else:

        path_influence = {
            node: 0.0
            for node in G.nodes()
        }

    # -------------------------------------------------------------
    # 6. Driver ranking
    #
    # Revenue stays in the graph but is explicitly excluded
    # from driver candidates.
    # -------------------------------------------------------------

    max_degree = max(
        degree.values(),
        default=1.0,
    )

    driver_scores: Dict[str, float] = {}

    for node in G.nodes():

        # Revenue is the target, not a candidate driver.
        if node == "revenue":
            continue

        bc_score = betweenness.get(
            node,
            0.0,
        )

        degree_score = min(
            degree.get(
                node,
                0.0,
            ) / max_degree,
            1.0,
        )

        gnn_score = gnn_scores.get(
            node,
            0.0,
        )

        path_score = path_influence.get(
            node,
            0.0,
        )

        material_bonus = (
            0.2
            if (
                node in material_kpis
                or node.startswith("pvm_")
            )
            else 0.0
        )

        # Heuristic evidence-ranking score.
        #
        # This is NOT:
        #   - a probability
        #   - a confidence score
        #   - causal proof
        #
        driver_scores[node] = round(
            0.10 * gnn_score
            + 0.35 * bc_score
            + 0.20 * path_score
            + 0.20 * degree_score
            + 0.15 * material_bonus,
            4,
        )

    driver_ranking = sorted(
        driver_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    driver_ranking = [
        {
            "kpi": kpi,
            "score": score,
            "gnn_score": gnn_scores.get(
                kpi,
                0.0,
            ),
            "path_influence": path_influence.get(
                kpi,
                0.0,
            ),
            "is_material": (
                kpi in material_kpis
                or kpi.startswith("pvm_")
            ),
        }
        for kpi, score in driver_ranking
    ]

    # -------------------------------------------------------------
    # 7. Serialize graph for frontend
    # -------------------------------------------------------------

    graph_nodes = []

    for node, data in G.nodes(
        data=True
    ):

        graph_nodes.append(
            {
                "id": node,

                "label": (
                    node
                    .replace(
                        "pvm_",
                        "PVM ",
                    )
                    .replace(
                        "_",
                        " ",
                    )
                    .title()
                ),

                "material": data.get(
                    "material",
                    False,
                ),

                "evidence_type": data.get(
                    "evidence_type",
                    "OBSERVED_METRIC",
                ),

                "contribution_usd": data.get(
                    "contribution_usd",
                    0.0,
                ),

                "centrality": betweenness.get(
                    node,
                    0.0,
                ),

                "path_influence": path_influence.get(
                    node,
                    0.0,
                ),

                "gnn_score": gnn_scores.get(
                    node,
                    0.0,
                ),

                "gnn_embedding": gnn_embeddings.get(
                    node,
                    [],
                ),
            }
        )

    # -------------------------------------------------------------
    # 8. Return
    # -------------------------------------------------------------

    return {
        "graph_metadata": {
            "region_id": region_id,
            "scenario": scenario,
            "gnn_engine": gnn_engine,
            "gnn_role": "SUPPORTING_FEATURE_SIGNAL",
            "causal_claim": False,
            "causal_proof": False,
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

        "gnn_embeddings": gnn_embeddings,
        "gnn_scores": gnn_scores,
        "correlation_matrix": correlation_matrix,

        "graphsage_available": HAS_TORCH_SAGE,

        "edge_count": len(active_edges),
        "node_count": len(graph_nodes),
    }
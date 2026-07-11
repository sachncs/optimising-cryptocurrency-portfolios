"""Correlation graphs and consensus Louvain clustering.

Implements the network-analysis layer of the consensus-clustered
portfolio framework described in `arXiv:2505.24831v2
<https://arxiv.org/abs/2505.24831v2>`_.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def correlation_distance_matrix(returns_window: pd.DataFrame) -> pd.DataFrame:
    """Compute the pairwise correlation-distance matrix of a returns window.

    The distance between assets ``i`` and ``j`` is
    ``sqrt(2 * (1 - rho_ij))`` where ``rho_ij`` is the Pearson
    correlation of their returns. This is a proper metric on the space
    of standardised return vectors.
    """
    corr = returns_window.corr(method="pearson").clip(-1.0, 1.0)
    dist = np.sqrt(2 * (1 - corr))
    array = np.array(dist, dtype=float, copy=True)
    np.fill_diagonal(array, 0.0)
    return pd.DataFrame(array, index=corr.index, columns=corr.columns)


def build_weighted_graph_from_distance(distance_matrix: pd.DataFrame) -> nx.Graph:
    """Build a weighted similarity graph from a distance matrix.

    Edge weight is the monotone, bounded similarity transform
    ``1 / (1 + d)``.
    """
    g = nx.Graph()
    assets = list(distance_matrix.columns)
    g.add_nodes_from(assets)
    for i, a in enumerate(assets):
        for j in range(i + 1, len(assets)):
            b = assets[j]
            d = float(distance_matrix.loc[a, b])
            sim = 1.0 / (1.0 + d)
            g.add_edge(a, b, weight=sim)
    return g


def louvain_partition(graph: nx.Graph, seed: int | None = None) -> list[set[str]]:
    """Run Louvain community detection on a weighted graph."""
    if graph.number_of_nodes() == 0:
        return []
    if graph.number_of_edges() == 0:
        return [{n} for n in graph.nodes]
    communities = nx.community.louvain_communities(graph, weight="weight", seed=seed)
    return [set(c) for c in communities]


def consensus_similarity_matrix(
    partitions: list[list[set[str]]],
    assets: list[str],
) -> np.ndarray:
    """Aggregate partitions into a co-occurrence similarity matrix.

    The result is the consensus co-membership probability in
    ``[0, 1]``, scaled by ``1/R`` so the diagonal is exactly ``1``.
    """
    n = len(assets)
    idx = {a: i for i, a in enumerate(assets)}
    sim = np.zeros((n, n), dtype=float)
    if not partitions:
        np.fill_diagonal(sim, 1.0)
        return sim
    for part in partitions:
        for community in part:
            members = [idx[m] for m in community if m in idx]
            for i in members:
                sim[i, i] += 1
            for p in range(len(members)):
                for q in range(p + 1, len(members)):
                    i, j = members[p], members[q]
                    sim[i, j] += 1
                    sim[j, i] += 1
    sim /= float(len(partitions))
    np.fill_diagonal(sim, 1.0)
    return sim


def stable_clusters_from_similarity(
    similarity: np.ndarray, assets: list[str], threshold: float = 0.5
) -> list[list[str]]:
    """Threshold the consensus matrix and take connected components."""
    if similarity.shape[0] != len(assets):
        raise ValueError("Similarity matrix and assets length mismatch")
    g = nx.Graph()
    g.add_nodes_from(assets)
    n = len(assets)
    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] >= threshold:
                g.add_edge(assets[i], assets[j], weight=float(similarity[i, j]))
    return [sorted(list(c)) for c in nx.connected_components(g)]


__all__ = [
    "build_weighted_graph_from_distance",
    "consensus_similarity_matrix",
    "correlation_distance_matrix",
    "louvain_partition",
    "stable_clusters_from_similarity",
]
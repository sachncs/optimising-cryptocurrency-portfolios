"""Correlation graphs and consensus Louvain clustering.

This module implements the network-analysis layer of the consensus-clustered
portfolio framework described in `arXiv:2505.24831v2
<https://arxiv.org/abs/2505.24831v2>`_. It builds a weighted asset
similarity graph from a rolling correlation matrix, runs Louvain community
detection on the graph multiple times, and aggregates the resulting
partitions into a co-occurrence consensus similarity matrix that drives
cluster selection.

Pipeline
--------
1. :func:`correlation_distance_matrix` -- converts a returns window to a
   pairwise distance matrix using ``sqrt(2*(1-rho))``, which is a proper
   metric on standardised return vectors (it is proportional to the
   Euclidean distance after row-wise centering and scaling).
2. :func:`build_weighted_graph_from_distance` -- turns the distance matrix
   into a ``networkx.Graph`` with edge weight ``1 / (1 + d)``. The
   ``1/(1+d)`` map is a simple, monotone, bounded similarity transform.
3. :func:`louvain_partition` -- runs Louvain community detection on the
   graph (with a configurable random seed for reproducibility).
4. :func:`consensus_similarity_matrix` -- counts how often every pair of
   assets lands in the same community across ``R`` Louvain runs and
   divides by ``R``. The result is the *consensus* co-membership matrix.
5. :func:`stable_clusters_from_similarity` -- thresholds the consensus
   matrix and takes connected components to produce the final stable
   clusters used for portfolio construction.

The co-occurrence count is the heart of the consensus step. The function
is annotated below with inline comments that explain the accumulator.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def correlation_distance_matrix(returns_window: pd.DataFrame) -> pd.DataFrame:
    """Compute the pairwise correlation-distance matrix of a returns window.

    The distance between assets ``i`` and ``j`` is ``sqrt(2 * (1 - rho_ij))``
    where ``rho_ij`` is the Pearson correlation of their returns. This is a
    proper metric on the space of standardised return vectors (it is
    proportional to Euclidean distance after standardising each row) and is
    well-defined for any correlation in ``[-1, 1]``.

    Args:
        returns_window: A returns ``pd.DataFrame`` with one column per
            asset. Index is preserved but its values are not used.

    Returns:
        A symmetric ``pd.DataFrame`` with zero diagonal and pairwise
        distances in ``[0, 2]``.

    Implementation notes:
        ``corr`` is clipped to ``[-1, 1]`` to defend against floating-point
        drift that could produce ``|rho| > 1`` for perfectly collinear
        pairs. The diagonal is forced to zero so the downstream similarity
        transform is well-behaved on the diagonal. ``np.array(dist, dtype=float, copy=True)``
        forces a writable buffer -- ``np.fill_diagonal`` mutates in place
        and chained arithmetic (``2 * (1 - corr)`` then ``np.sqrt``)
        can produce a read-only NumPy view in recent versions.
    """
    corr = returns_window.corr(method="pearson").clip(-1.0, 1.0)
    dist = np.sqrt(2 * (1 - corr))
    # Materialise a writable copy before mutating the diagonal; chained
    # arithmetic on DataFrame views may return a read-only NumPy array.
    array = np.array(dist, dtype=float, copy=True)
    np.fill_diagonal(array, 0.0)
    return pd.DataFrame(array, index=corr.index, columns=corr.columns)


def build_weighted_graph_from_distance(distance_matrix: pd.DataFrame) -> nx.Graph:
    """Build a weighted similarity graph from a distance matrix.

    Edge weight is the monotone, bounded similarity transform
    ``1 / (1 + d)``. This maps a distance of ``0`` to similarity ``1``
    (identical assets) and a distance of ``+inf`` to similarity ``0``
    (uncorrelated assets). Bounded output keeps Louvain numerically stable.

    Args:
        distance_matrix: Square, symmetric ``pd.DataFrame`` with zero
            diagonal and non-negative entries.

    Returns:
        A ``networkx.Graph`` with one node per asset column and one edge
        per off-diagonal entry. Self-loops are not added.
    """
    g = nx.Graph()
    assets = list(distance_matrix.columns)
    g.add_nodes_from(assets)
    for i, a in enumerate(assets):
        # Iterate over the upper triangle (``j > i``) so every edge is
        # added exactly once.
        for j in range(i + 1, len(assets)):
            b = assets[j]
            d = float(distance_matrix.loc[a, b])
            sim = 1.0 / (1.0 + d)
            g.add_edge(a, b, weight=sim)
    return g


def louvain_partition(graph: nx.Graph, seed: int | None = None) -> list[set[str]]:
    """Run Louvain community detection on a weighted graph.

    Args:
        graph: A ``networkx.Graph`` whose ``weight`` attribute is used as
            the edge weight.
        seed: Optional RNG seed for reproducibility.

    Returns:
        A list of frozenset-like communities, each holding the node
        labels in that community. Empty graphs return an empty list;
        edgeless graphs fall back to one singleton community per node.

    Notes:
        Louvain is a greedy modularity-maximisation heuristic -- the
        ``seed`` only changes the tie-breaking order, not the algorithm.
        Two Louvain runs with the same ``seed`` produce identical
        partitions.
    """
    if graph.number_of_nodes() == 0:
        return []
    if graph.number_of_edges() == 0:
        # No edges to optimise over; Louvain would return a single
        # community containing all nodes, which would defeat the consensus
        # step. Emit singleton communities instead so the consensus matrix
        # is well-defined.
        return [{n} for n in graph.nodes]
    communities = nx.community.louvain_communities(graph, weight="weight", seed=seed)
    return [set(c) for c in communities]


def consensus_similarity_matrix(
    partitions: list[list[set[str]]],
    assets: list[str],
) -> np.ndarray:
    """Aggregate a list of partitions into a co-occurrence similarity matrix.

    The consensus step counts how often each pair of assets lands in the
    same community across ``R`` independent Louvain partitions. The
    resulting matrix is the consensus co-membership probability, scaled
    by ``1/R`` so the diagonal is exactly ``1`` (every asset co-occurs
    with itself in every run).

    Args:
        partitions: List of ``R`` partitions. Each partition is a list of
            communities (each community a set of asset labels).
        assets: Canonical asset ordering for the output matrix.

    Returns:
        Square ``np.ndarray`` of shape ``(len(assets), len(assets))`` with
        ones on the diagonal and off-diagonal entries in ``[0, 1]``.

    Example:
        >>> assets = ["a", "b", "c"]
        >>> partitions = [[{"a", "b"}, {"c"}]]
        >>> consensus_similarity_matrix(partitions, assets)
        array([[1. , 1. , 0. ],
               [1. , 1. , 0. ],
               [0. , 0. , 1. ]])
    """
    n = len(assets)
    idx = {a: i for i, a in enumerate(assets)}
    sim = np.zeros((n, n), dtype=float)
    if not partitions:
        # No partitions to aggregate: default to the identity matrix so
        # downstream consumers (cluster extraction) treat every asset as
        # its own singleton community.
        np.fill_diagonal(sim, 1.0)
        return sim
    for part in partitions:
        for community in part:
            # Map every asset in the community to its matrix index.
            members = [idx[m] for m in community if m in idx]
            # Self-co-occurrence: an asset is always in the same community
            # as itself, so add one to every diagonal entry of the
            # community.
            for i in members:
                sim[i, i] += 1
            # Off-diagonal co-occurrence: every distinct pair inside the
            # community increments the corresponding symmetric entry by
            # one. The triple loop is the most concise expression of the
            # upper-triangle accumulation; with ``len(community) <= 50``
            # in practice, the constant factor is negligible compared to
            # the rest of the pipeline.
            for p in range(len(members)):
                for q in range(p + 1, len(members)):
                    i, j = members[p], members[q]
                    sim[i, j] += 1
                    sim[j, i] += 1
    # Normalise by the number of runs so the matrix represents a *probability*
    # in [0, 1] rather than a raw count.
    sim /= float(len(partitions))
    np.fill_diagonal(sim, 1.0)
    return sim


def stable_clusters_from_similarity(
    similarity: np.ndarray, assets: list[str], threshold: float = 0.5
) -> list[list[str]]:
    """Extract stable clusters by thresholding the consensus matrix.

    The threshold defines the co-membership probability at which an edge
    is drawn between two assets in a similarity graph; the resulting
    connected components are the stable clusters.

    Args:
        similarity: Square ``np.ndarray`` from
            :func:`consensus_similarity_matrix`.
        assets: Asset labels in the same order as ``similarity``'s axes.
        threshold: Co-membership probability cutoff. Defaults to ``0.5``
            (assets must land in the same community in *at least* half of
            the Louvain runs to be considered stable neighbours).

    Returns:
        A list of clusters, where each cluster is a sorted list of asset
        labels. Clusters of size 1 are emitted for assets with no
        above-threshold neighbours.

    Raises:
        ValueError: If the similarity matrix shape does not match
            ``len(assets)``.
    """
    if similarity.shape[0] != len(assets):
        raise ValueError("Similarity matrix and assets length mismatch")
    g = nx.Graph()
    g.add_nodes_from(assets)
    n = len(assets)
    for i in range(n):
        # Iterate over the upper triangle to add each edge once.
        for j in range(i + 1, n):
            if similarity[i, j] >= threshold:
                g.add_edge(assets[i], assets[j], weight=float(similarity[i, j]))
    return [sorted(list(c)) for c in nx.connected_components(g)]

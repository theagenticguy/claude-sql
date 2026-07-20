"""Pure Leiden+CPM + mutual-kNN community-detection math (no I/O).

Lifted verbatim from ``analytics/community_worker.py`` (MIGRATION Phase C /
T-3-1). These functions operate on in-memory numpy matrices and plain-scalar
hyperparameters and return in-memory results — no DuckDB connection, no
parquet, no LanceDB. The orchestration that loads centroids, projects
``Settings`` into a ``CommunityConfig``, and writes the primary/sidecar parquet
lives in ``application.use_cases.community``.

igraph and leidenalg are lazy-imported inside the functions that need them so
importing this module (e.g. via the analytics shim) doesn't drag the graph
import subtree onto the CLI fast path.

Determinism: ``seed`` flows into ``leidenalg.find_partition(seed=...)`` and into
the bisection RNG via ``Optimiser.set_rng_seed``. Same seed + same input ⇒
byte-identical output across runs. Cluster IDs are made stable by relabeling
communities by descending size after detection.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Literal

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    import igraph as ig


#: Community id used for singleton / unclusterable sessions.  -1 is an obvious
#: out-of-band sentinel that stays negative even after ``Int32`` serialization.
NOISE_COMMUNITY_ID: int = -1

#: Resolution preset literal; agents pass ``--resolution {coarse, medium, fine}``
#: and the orchestrator picks γ from the resolution profile accordingly.
ResolutionLevel = Literal["coarse", "medium", "fine"]


def _build_mutual_knn(
    sim: np.ndarray, *, k: int, floor: float
) -> tuple[list[tuple[int, int]], list[float]]:
    """Build a mutual-kNN edge list from a precomputed symmetric similarity matrix.

    Returns parallel ``(edges, weights)`` with each edge as a sorted
    ``(u, v)`` pair (u < v) so igraph never sees a duplicate.  Weight equals
    ``sim[u, v]`` directly: the input matrix is symmetric so
    ``max(w_ij, w_ji)`` is a no-op.

    ``sim`` must have its diagonal zeroed before being passed in; otherwise
    every node would pick itself as a top-k neighbor and the floor filter
    might let those through.
    """
    n = sim.shape[0]
    if n < 2:
        return [], []
    k_eff = min(k, n - 1)

    # argpartition returns top-k indices per row; we don't care about order
    # inside the top-k slice because the mutual filter is symmetric.
    top = np.argpartition(-sim, kth=k_eff - 1, axis=1)[:, :k_eff]

    # Build a boolean N×N "is-in-my-top-k" mask, then AND with its transpose
    # to get the mutual-kNN adjacency.
    in_top = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), k_eff)
    cols = top.reshape(-1)
    in_top[rows, cols] = True
    mutual = in_top & in_top.T

    # Apply edge floor.
    weighted = np.where(mutual & (sim >= floor), sim, 0.0)

    # Take upper triangle to avoid duplicates.
    iu, ju = np.triu_indices(n, k=1)
    keep = weighted[iu, ju] > 0.0
    edges = list(zip(iu[keep].tolist(), ju[keep].tolist(), strict=True))
    weights = weighted[iu, ju][keep].tolist()
    return edges, weights


def _build_igraph(n_nodes: int, edges: list[tuple[int, int]], weights: list[float]) -> ig.Graph:
    """Construct an undirected weighted ``igraph.Graph``.

    The edge attribute MUST be named ``"weight"`` -- ``leidenalg`` looks it up
    by string when ``find_partition(weights="weight", ...)`` is called.
    """
    import igraph as ig

    g = ig.Graph(n=n_nodes, edges=edges, directed=False)
    g.es["weight"] = weights
    return g


def _compute_resolution_profile(
    g: ig.Graph,
    *,
    range_lo: float,
    range_hi: float,
    seed: int,
) -> list[tuple[float, int, float, int]]:
    """Run bisection over γ via ``Optimiser.resolution_profile``.

    Returns rows ``(gamma, n_communities, quality, plateau_length)`` ordered
    by γ ascending, where ``plateau_length`` is the gap between consecutive
    γ change-points (a proxy for partition stability).

    Determinism: ``Optimiser.set_rng_seed(seed)`` keys the bisection RNG.
    """
    import leidenalg as la

    optimiser = la.Optimiser()
    optimiser.set_rng_seed(seed)
    profile = optimiser.resolution_profile(
        g,
        la.CPMVertexPartition,
        weights="weight",
        resolution_range=(range_lo, range_hi),
    )
    if not profile:
        return []

    # γ values are the change-points returned by bisection. Plateau length
    # for partition i is γ[i+1] - γ[i] (the last entry's plateau extends to
    # range_hi).
    rows: list[tuple[float, int, float, int]] = []
    for i, partition in enumerate(profile):
        gamma = float(getattr(partition, "resolution_parameter"))  # noqa: B009
        membership = list(getattr(partition, "membership"))  # noqa: B009
        n_comm = len(set(membership))
        quality = float(getattr(partition, "quality")())  # noqa: B009
        if i + 1 < len(profile):
            next_gamma = float(
                getattr(profile[i + 1], "resolution_parameter")  # noqa: B009
            )
        else:
            next_gamma = range_hi
        plateau = max(0, round((next_gamma - gamma) * 10000))
        rows.append((gamma, n_comm, quality, plateau))
    return rows


def _pick_zoom(profile: list[tuple[float, int, float, int]], level: ResolutionLevel) -> float:
    """Pick a γ from the precomputed resolution profile.

    - ``medium`` -> midpoint of the longest plateau.
    - ``coarse`` -> γ at the lowest n_communities plateau (with n ≥ 2).
    - ``fine``   -> γ at the highest n_communities plateau.

    Falls back to the median γ in the profile when the requested level can't
    be served (e.g., ``coarse`` when every partition has only one community).
    """
    if not profile:
        raise RuntimeError("empty resolution profile - cannot pick γ")

    if level == "medium":
        # Longest plateau wins; ties broken by smallest γ for stability.
        idx = max(range(len(profile)), key=lambda i: (profile[i][3], -profile[i][0]))
        return profile[idx][0]

    if level == "coarse":
        # Lowest n_communities partition with at least 2 communities.
        eligible = [(i, row) for i, row in enumerate(profile) if row[1] >= 2]
        if not eligible:
            return profile[len(profile) // 2][0]
        idx = min(eligible, key=lambda pair: (pair[1][1], pair[1][0]))[0]
        return profile[idx][0]

    # fine
    idx = max(range(len(profile)), key=lambda i: (profile[i][1], -profile[i][0]))
    return profile[idx][0]


def _run_leiden_cpm(g: ig.Graph, *, gamma: float, seed: int, n_iterations: int) -> object:
    """Run Leiden+CPM with deterministic seeding.

    Returns a ``leidenalg.CPMVertexPartition`` (which subclasses
    ``igraph.VertexClustering`` and carries ``.quality()``,
    ``.membership``, ``.resolution_parameter``).  Typed as ``object`` because
    the leidenalg type stubs do not expose ``quality()`` on the partition
    class hierarchy under ty/Pyright; callers cast or ``getattr``.
    """
    import leidenalg as la

    return la.find_partition(
        g,
        la.CPMVertexPartition,
        weights="weight",
        resolution_parameter=gamma,
        seed=seed,
        n_iterations=n_iterations,
    )


def _warn_disconnected(g: ig.Graph, labels: list[int]) -> None:
    """Log a warning for any community whose induced subgraph is disconnected.

    Park et al. (2024) reported up to 16% of Leiden communities disconnected
    on biomedical citation graphs.  On symmetric mutual-kNN over normalized
    cosine centroids the rate is materially lower, so we only warn -- no
    splitting.  If this fires on the live corpus regularly, bring in the
    Park et al. Connectivity Modifier as a focused follow-up.
    """
    by_cid: dict[int, list[int]] = defaultdict(list)
    for node_idx, cid in enumerate(labels):
        if cid >= 0:
            by_cid[cid].append(node_idx)
    for cid, nodes in by_cid.items():
        if len(nodes) < 3:
            continue
        sub = g.induced_subgraph(nodes)
        if not sub.is_connected(mode="weak"):
            comp_sizes = sorted((len(c) for c in sub.connected_components()), reverse=True)
            logger.warning(
                "Community {} has {} weakly-connected components (sizes={}); "
                "consider rerunning with a different gamma or filing an issue if "
                "this fires regularly.",
                cid,
                len(comp_sizes),
                comp_sizes,
            )


def _compute_medoid_and_coherence(
    sim: np.ndarray, labels: list[int]
) -> tuple[set[int], dict[int, float]]:
    """Per community: medoid node index + mean intra-community cosine.

    The medoid is the node with the highest mean cosine similarity to the
    other community members.  Communities of size 1 have themselves as
    medoid and coherence 1.0.  Returns the set of node indices that are
    medoids and a ``{community_id: coherence}`` map (excluding noise).
    """
    by_cid: dict[int, list[int]] = defaultdict(list)
    for node_idx, cid in enumerate(labels):
        if cid >= 0:
            by_cid[cid].append(node_idx)

    medoid_indices: set[int] = set()
    coherence: dict[int, float] = {}
    for cid, nodes in by_cid.items():
        if len(nodes) == 1:
            medoid_indices.add(nodes[0])
            coherence[cid] = 1.0
            continue
        sub = sim[np.ix_(nodes, nodes)]
        # Mean cosine to other members; subtract the diagonal contribution.
        n = len(nodes)
        per_row_sum = sub.sum(axis=1) - np.diag(sub)
        mean_to_others = per_row_sum / (n - 1)
        medoid_local = int(np.argmax(mean_to_others))
        medoid_indices.add(nodes[medoid_local])
        # Community-level coherence is the mean over the off-diagonal upper
        # triangle (==  mean pairwise cosine within the community).
        iu = np.triu_indices(n, k=1)
        coherence[cid] = float(sub[iu].mean()) if iu[0].size else 1.0
    return medoid_indices, coherence


def _relabel_and_collapse(
    raw_labels: list[int],
    sids: list[str],
    *,
    min_size: int,
    medoid_indices: set[int],
    coherence: dict[int, float],
    gamma_used: float,
) -> tuple[list[tuple[str, int, int, bool, float, float]], int, int]:
    """Stable relabel by descending size; collapse small communities to noise.

    Returns ``(rows, n_real_communities, n_noise)``.  Each row is
    ``(session_id, community_id, size, is_medoid, coherence, gamma_used)``.
    Communities below ``min_size`` get ``community_id = NOISE_COMMUNITY_ID``,
    ``is_medoid = False``, ``coherence = 0.0``.
    """
    raw_to_nodes: dict[int, list[int]] = defaultdict(list)
    for node_idx, cid in enumerate(raw_labels):
        raw_to_nodes[cid].append(node_idx)

    # Sort communities by descending size, then by smallest node index for
    # tiebreaks so the relabel is deterministic.
    sorted_raw = sorted(
        raw_to_nodes.items(), key=lambda kv: (-len(kv[1]), min(kv[1])) if kv[1] else (0, 0)
    )

    rows: list[tuple[str, int, int, bool, float, float]] = []
    new_id = 0
    n_real = 0
    n_noise = 0
    for raw_cid, nodes in sorted_raw:
        size = len(nodes)
        if size < min_size:
            for node_idx in nodes:
                rows.append((sids[node_idx], NOISE_COMMUNITY_ID, 0, False, 0.0, gamma_used))
                n_noise += 1
            continue
        n_real += 1
        comm_coherence = coherence.get(raw_cid, 0.0)
        rows.extend(
            (
                sids[node_idx],
                new_id,
                size,
                node_idx in medoid_indices,
                comm_coherence,
                gamma_used,
            )
            for node_idx in nodes
        )
        new_id += 1
    return rows, n_real, n_noise


__all__ = [
    "NOISE_COMMUNITY_ID",
    "ResolutionLevel",
    "_build_igraph",
    "_build_mutual_knn",
    "_compute_medoid_and_coherence",
    "_compute_resolution_profile",
    "_pick_zoom",
    "_relabel_and_collapse",
    "_run_leiden_cpm",
    "_warn_disconnected",
]

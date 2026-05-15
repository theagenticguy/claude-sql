"""Session-level community detection via mutual-kNN cosine graph + Leiden+CPM.

Pipeline
--------
1. Compute session centroid embeddings: group message embeddings by
   ``messages.session_id`` and average (mean over rows, then L2-normalize).
2. Build a mutual-kNN cosine graph (k=15 by default) over session centroids,
   symmetrize with ``max(w_ij, w_ji)``, drop edges below ``leiden_edge_floor``.
3. Pick γ for CPM: when the user does not pass an explicit γ, run
   ``leidenalg.Optimiser.resolution_profile`` over
   ``[leiden_resolution_range_lo, leiden_resolution_range_hi]`` and pick the
   midpoint of the longest plateau in n_communities-vs-γ.  When the user does
   pass γ, skip the profile entirely.
4. Run ``leidenalg.find_partition(g, CPMVertexPartition, weights="weight",
   resolution_parameter=γ, seed=settings.seed, n_iterations=-1)``.
5. Warn (do not split) on communities whose induced subgraph has multiple
   weakly-connected components.  Park et al. 2024's 16% disconnection rate is
   on biomedical citation graphs; symmetric mutual-kNN over normalized cosine
   centroids is a different regime.  If this warning fires on the live corpus
   we'll add a splitter; until then, the warn-only path is delete-in-30s
   reversible.
6. Compute per-community medoid (max mean intra-community cosine) and
   coherence (mean intra-community cosine).
7. Stable relabel by descending size; collapse communities below
   ``leiden_min_community_size`` to ``NOISE_COMMUNITY_ID = -1``.
8. Write primary parquet ``session_communities.parquet`` with columns
   ``(session_id, community_id, size, is_medoid, coherence, gamma_used)`` and
   conditionally write the resolution-profile sidecar
   ``community_profile.parquet`` when auto-γ ran.

Why CPM and not modularity
--------------------------
For cosine-similarity edge graphs the CPM γ has a closed-form interpretation
(Traag-Van Dooren-Nesterov 2011): communities have internal density ≥ γ and
external density ≤ γ, both expressed in the same units as the edge weights
(cosine).  This gives an LLM agent driving the CLI a directly interpretable
"tighter or looser clusters" knob.  Modularity's resolution parameter has no
such semantics on weighted graphs.

Public API
----------
run_communities(con, settings, *, force=False, gamma=None, resolution="medium")
    -> dict[str, int | float | str]
neighbors_of(con, settings, session_id, *, top_k=15) -> pl.DataFrame

Determinism
-----------
``settings.seed`` flows into ``leidenalg.find_partition(seed=...)`` and into
the bisection RNG via ``Optimiser.set_rng_seed``.  Same seed + same input ⇒
byte-identical parquet output across runs.  Cluster IDs are made stable by
relabeling communities by descending size after detection.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import duckdb
import numpy as np
import polars as pl
from loguru import logger

from claude_sql.core.config import Settings

if TYPE_CHECKING:
    import duckdb
    import igraph as ig


#: Community id used for singleton / unclusterable sessions.  -1 is an obvious
#: out-of-band sentinel that stays negative even after ``Int32`` serialization.
NOISE_COMMUNITY_ID: int = -1

#: Resolution preset literal; agents pass ``--resolution {coarse, medium, fine}``
#: and the orchestrator picks γ from the resolution profile accordingly.
ResolutionLevel = Literal["coarse", "medium", "fine"]


def _load_session_centroids(
    con: duckdb.DuckDBPyConnection, embeddings_parquet_path: Path
) -> tuple[list[str], np.ndarray]:
    """Return ``(session_ids, centroids)`` where centroids is ``(N_sessions, dim)`` float32.

    Joins the ``message_embeddings`` view (LanceDB-backed via ``register_vss``)
    to the v1 ``messages`` view on uuid, then aggregates inside DuckDB
    (unnest with position → ``avg`` per (session, dim_index) → ordered
    ``list``). The L2-normalize step stays in numpy where
    ``np.linalg.norm`` is faster on a contiguous (N, dim) matrix.

    ``embeddings_parquet_path`` is accepted for back-compat with callers that
    still pass it but is no longer consulted — the connection's
    ``message_embeddings`` view is the source of truth.
    """
    del embeddings_parquet_path  # legacy kwarg — view is the source of truth now
    logger.info("Loading message embeddings and joining to sessions...")
    sql = """
        WITH joined AS (
            SELECT CAST(m.session_id AS VARCHAR) AS session_id,
                   e.embedding::FLOAT[] AS emb
              FROM message_embeddings e
              JOIN messages m
                ON CAST(m.uuid AS VARCHAR) = e.uuid
        ),
        unrolled AS (
            SELECT session_id,
                   generate_subscripts(emb, 1) AS pos,
                   unnest(emb) AS v
              FROM joined
        ),
        agg AS (
            SELECT session_id, pos, avg(v) AS m
              FROM unrolled
             GROUP BY 1, 2
        )
        SELECT session_id, list(m ORDER BY pos) AS centroid
          FROM agg
         GROUP BY 1
         ORDER BY 1
    """
    try:
        df = con.execute(sql).pl()
    except duckdb.CatalogException as exc:
        # ``message_embeddings`` view isn't bound on this connection — happens
        # when the caller skipped ``register_vss`` and there's no Lance
        # dataset. Surface the same shape the legacy parquet path raised.
        raise RuntimeError(
            "No embeddings parquet (or Lance dataset) reachable — register_vss "
            "must be called first or `claude-sql embed` must produce data."
        ) from exc
    if len(df) == 0:
        raise RuntimeError(
            "No rows returned joining embeddings to messages - check that the "
            "Lance embeddings exist and the messages view is registered."
        )

    sids = df["session_id"].to_list()
    centroids = np.stack([np.asarray(c, dtype=np.float32) for c in df["centroid"].to_list()])
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.where(norms == 0, 1.0, norms)
    logger.info("Computed {} session centroids (dim={})", len(sids), centroids.shape[1])
    return sids, centroids


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


def neighbors_of(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    session_id: str,
    *,
    top_k: int = 15,
) -> pl.DataFrame:
    """Return the top-k cosine neighbors of ``session_id`` in centroid space.

    Bypasses Leiden entirely: loads centroids → cosine to target → top-k.
    Joins to ``session_communities.parquet`` if it exists so each neighbor
    carries its ``community_id`` and ``is_medoid`` flag; otherwise the
    output has just ``(neighbor_session_id, weight)`` and a warning fires.
    """
    sids, centroids = _load_session_centroids(con, settings.embeddings_parquet_path)
    if session_id not in sids:
        raise ValueError(f"session_id {session_id!r} not found in embeddings corpus")
    target_idx = sids.index(session_id)
    target = centroids[target_idx]
    sim = centroids @ target
    sim[target_idx] = -1.0  # exclude self
    k_eff = min(top_k, len(sids) - 1)
    if k_eff <= 0:
        return pl.DataFrame(schema={"neighbor_session_id": pl.Utf8, "weight": pl.Float32})
    top_idx = np.argpartition(-sim, kth=k_eff - 1)[:k_eff]
    top_idx = top_idx[np.argsort(-sim[top_idx])]
    out = pl.DataFrame(
        {
            "neighbor_session_id": [sids[i] for i in top_idx],
            "weight": sim[top_idx].astype(np.float32).tolist(),
        }
    )

    comm_path = settings.communities_parquet_path
    if comm_path.exists() and comm_path.stat().st_size > 16:
        comm = pl.read_parquet(comm_path).select(["session_id", "community_id", "is_medoid"])
        out = out.join(
            comm,
            left_on="neighbor_session_id",
            right_on="session_id",
            how="left",
        )
    else:
        logger.warning(
            "communities parquet missing at {}; --neighbors-of will not "
            "carry community_id / is_medoid. Run `claude-sql community` first.",
            comm_path,
        )
    return out


def run_communities(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    force: bool = False,
    gamma: float | None = None,
    resolution: ResolutionLevel = "medium",
) -> dict[str, int | float | str]:
    """Run Leiden+CPM on session centroids; write primary parquet (+ optional sidecar).

    Parameters
    ----------
    con
        An open DuckDB connection with the v1 ``messages`` view registered.
    settings
        Configured Settings.  Honors every ``leiden_*`` field plus ``seed``.
    force
        If False and output parquet exists, skip recomputation.
    gamma
        Explicit CPM resolution parameter γ.  When ``None`` (default), the
        worker runs ``Optimiser.resolution_profile`` and picks γ via the
        ``resolution`` preset.  When set, the profile is skipped and the
        sidecar parquet is NOT written.
    resolution
        ``"coarse" | "medium" | "fine"`` -- which plateau to pick from the
        resolution profile.  Ignored when ``gamma`` is explicit.

    Returns
    -------
    dict
        ``{"sessions", "communities", "noise", "gamma_used", "quality",
        "algorithm"}``.  ``communities`` counts only real (>= min-size)
        communities; singletons are aggregated into ``noise``.
    """
    out_path = settings.communities_parquet_path
    profile_path = settings.community_profile_parquet_path

    if out_path.exists() and out_path.stat().st_size > 16 and not force:
        logger.info("Communities parquet exists at {}; pass force=True to rebuild", out_path)
        df = pl.read_parquet(out_path)
        real = df.filter(pl.col("community_id") != NOISE_COMMUNITY_ID)
        noise_n = df.height - real.height
        n_comm = int(real["community_id"].n_unique()) if real.height else 0
        cached_gamma = (
            float(df["gamma_used"][0]) if "gamma_used" in df.columns and df.height else 0.0
        )
        return {
            "sessions": int(df.height),
            "communities": n_comm,
            "noise": noise_n,
            "gamma_used": cached_gamma,
            "quality": float("nan"),
            "algorithm": "leiden_cpm",
        }

    sids, centroids = _load_session_centroids(con, settings.embeddings_parquet_path)

    t0 = time.monotonic()
    logger.info("Computing pairwise cosine similarity over {} sessions...", len(sids))
    sim = centroids @ centroids.T
    np.fill_diagonal(sim, 0.0)

    edges, weights = _build_mutual_knn(
        sim, k=settings.leiden_knn_k, floor=settings.leiden_edge_floor
    )
    logger.info(
        "Mutual-kNN graph: {} nodes, {} edges (k={}, floor={:.2f}) in {:.1f}s",
        len(sids),
        len(edges),
        settings.leiden_knn_k,
        settings.leiden_edge_floor,
        time.monotonic() - t0,
    )

    g = _build_igraph(len(sids), edges, weights)

    profile_rows: list[tuple[float, int, float, int]] | None = None
    if gamma is None:
        # Honor settings.leiden_resolution if it's set; else run the profile.
        if settings.leiden_resolution is not None:
            gamma_used = float(settings.leiden_resolution)
        else:
            t1 = time.monotonic()
            profile_rows = _compute_resolution_profile(
                g,
                range_lo=settings.leiden_resolution_range_lo,
                range_hi=settings.leiden_resolution_range_hi,
                seed=settings.seed,
            )
            logger.info(
                "Resolution profile: {} γ change-points in {:.1f}s",
                len(profile_rows),
                time.monotonic() - t1,
            )
            if not profile_rows:
                # Empty profile is rare (graph with no edges) -- fall back to
                # the midpoint of the configured range so we still produce
                # output rather than crashing the analyze chain.
                gamma_used = (
                    settings.leiden_resolution_range_lo + settings.leiden_resolution_range_hi
                ) / 2.0
            else:
                gamma_used = _pick_zoom(profile_rows, resolution)
    else:
        gamma_used = float(gamma)

    t2 = time.monotonic()
    partition = _run_leiden_cpm(
        g,
        gamma=gamma_used,
        seed=settings.seed,
        n_iterations=settings.leiden_n_iterations,
    )
    # ``CPMVertexPartition`` carries ``.quality()`` and ``.membership`` but the
    # leidenalg stubs don't expose them; ``getattr`` keeps ty + Pyright happy
    # without changing runtime behavior.
    quality = float(getattr(partition, "quality")())  # noqa: B009
    membership = list(getattr(partition, "membership"))  # noqa: B009
    n_communities = len({m for m in membership if m >= 0})
    logger.info(
        "Leiden+CPM γ={:.4f}: {} raw communities (quality={:.4f}) in {:.1f}s",
        gamma_used,
        n_communities,
        quality,
        time.monotonic() - t2,
    )

    labels = membership
    _warn_disconnected(g, labels)

    medoid_indices, coherence = _compute_medoid_and_coherence(sim, labels)

    rows, n_real, n_noise = _relabel_and_collapse(
        labels,
        sids,
        min_size=settings.leiden_min_community_size,
        medoid_indices=medoid_indices,
        coherence=coherence,
        gamma_used=gamma_used,
    )

    df = pl.DataFrame(
        rows,
        schema={
            "session_id": pl.Utf8,
            "community_id": pl.Int32,
            "size": pl.Int32,
            "is_medoid": pl.Boolean,
            "coherence": pl.Float32,
            "gamma_used": pl.Float32,
        },
        orient="row",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    logger.info(
        "Leiden+CPM: {} kept >= {} sessions, {} singletons -> noise. Wrote {} rows to {}",
        n_real,
        settings.leiden_min_community_size,
        n_noise,
        len(df),
        out_path,
    )

    if profile_rows is not None:
        prof_df = pl.DataFrame(
            profile_rows,
            schema={
                "gamma": pl.Float64,
                "n_communities": pl.Int32,
                "quality": pl.Float64,
                "plateau_length": pl.Int32,
            },
            orient="row",
        )
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        prof_df.write_parquet(profile_path)
        logger.info("Wrote {} γ-points to {}", prof_df.height, profile_path)

    size_dist = (
        df.filter(pl.col("community_id") != NOISE_COMMUNITY_ID)
        .group_by("community_id")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(5)
    )
    logger.info("Top 5 community sizes: {}", size_dist.to_dicts())

    return {
        "sessions": int(df.height),
        "communities": n_real,
        "noise": n_noise,
        "gamma_used": float(gamma_used),
        "quality": quality,
        "algorithm": "leiden_cpm",
    }

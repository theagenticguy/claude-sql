"""Session-level community detection via cosine graph + Louvain.

Pipeline
--------
1. Compute session centroid embeddings: group message embeddings by
   ``messages.session_id`` and average (mean over rows, then L2-normalize).
2. Pick an adaptive cosine threshold that yields a connected but not
   dense graph (target average degree in ``[target_avg_degree_low,
   target_avg_degree_high]``), falling back to ``louvain_edge_threshold``
   when the adaptive fit fails.
3. Threshold the pairwise similarity into a sparse graph.
4. Run ``networkx.community.louvain_communities`` with seed=42.
5. Collapse singleton communities into a single ``community_id=-1`` "noise"
   bucket so the output parquet stays scannable.
6. Write ``session_communities.parquet`` with
   ``(session_id, community_id, size)``.

Why adaptive thresholding
-------------------------
The previous design used a fixed ``louvain_edge_threshold=0.75`` absolute
cosine cut.  Against ``int8``-quantized embeddings the pairwise cosine
distribution is both narrower and higher-mean than against unquantized
``float32``, so 0.75 excluded the majority of "related" session pairs and
Louvain produced thousands of singleton communities (on the real corpus:
1,512 communities for 6,329 sessions, with only 21 ≥ 20 sessions).

The adaptive path picks the percentile-based cut that puts average degree
in the 8–15 range (Louvain's typical sweet spot for this kind of graph),
so the answer stabilizes whether the underlying embeddings are int8,
float16, or float32.

Public API
----------
run_communities(con, settings, *, force=False) -> dict[str, int]
    Expects ``con`` to have v1 views registered (``messages`` + access to
    the embeddings parquet via ``message_embeddings`` table or direct parquet
    read).  Returns ``{sessions, communities, noise, threshold}``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from loguru import logger

from claude_sql.config import Settings

if TYPE_CHECKING:
    import duckdb


#: Community id used for singleton / unclusterable sessions.  -1 is an obvious
#: out-of-band sentinel that stays negative even after ``Int32`` serialization.
NOISE_COMMUNITY_ID: int = -1


def _load_session_centroids(
    con: duckdb.DuckDBPyConnection, embeddings_parquet_path: Path
) -> tuple[list[str], np.ndarray]:
    """Return ``(session_ids, centroids)`` where centroids is ``(N_sessions, dim)`` float32.

    Joins the embeddings parquet to the v1 ``messages`` view on uuid, groups
    by session_id, and averages the embedding.  Normalizes each centroid to
    unit L2 length so dot products equal cosine similarity.
    """
    logger.info("Loading message embeddings and joining to sessions...")
    sql = """
        SELECT CAST(m.session_id AS VARCHAR) AS session_id,
               e.embedding
          FROM read_parquet(?) e
          JOIN messages m
            ON CAST(m.uuid AS VARCHAR) = e.uuid
    """
    df = con.execute(sql, [str(embeddings_parquet_path)]).pl()
    logger.info("Joined {} rows (messages x embeddings)", len(df))

    if len(df) == 0:
        raise RuntimeError(
            "No rows returned joining embeddings to messages - check that the "
            "embeddings parquet exists and the messages view is registered."
        )

    emb = df["embedding"].to_numpy()
    if emb.ndim == 1:
        emb = np.vstack(list(emb))
    emb = np.ascontiguousarray(emb, dtype=np.float32)

    session_ids_col = df["session_id"].to_numpy()
    unique = sorted(set(session_ids_col.tolist()))
    sid_to_idx = {sid: i for i, sid in enumerate(unique)}
    sums = np.zeros((len(unique), emb.shape[1]), dtype=np.float32)
    counts = np.zeros(len(unique), dtype=np.int32)
    for i, sid in enumerate(session_ids_col):
        idx = sid_to_idx[sid]
        sums[idx] += emb[i]
        counts[idx] += 1
    centroids = sums / counts[:, None]
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.where(norms == 0, 1.0, norms)
    logger.info("Computed {} session centroids (dim={})", len(unique), centroids.shape[1])
    return unique, centroids


def _pick_adaptive_threshold(
    sim: np.ndarray,
    *,
    floor: float,
    target_avg_degree_low: float,
    target_avg_degree_high: float,
) -> float:
    """Pick a cosine threshold that puts the average graph degree in the target band.

    Binary-searches the upper-triangular similarity quantiles.  Falls back to
    ``floor`` when the graph is too sparse to hit the band even at q=0.
    """
    n = sim.shape[0]
    if n < 2:
        return floor
    # Upper triangle only -- pairwise matrix is symmetric; diag already zeroed.
    iu = np.triu_indices(n, k=1)
    upper = sim[iu]
    if upper.size == 0:
        return floor

    target_edges_low = n * target_avg_degree_low / 2.0
    target_edges_high = n * target_avg_degree_high / 2.0

    # Quantile search: try a handful of candidate thresholds and pick the
    # lowest one that stays under target_edges_high, then clamp up to floor.
    # Using np.quantile ensures the candidate values are monotonically
    # decreasing -> edge count monotonically increasing -- no binary search
    # machinery needed.
    quantiles = np.linspace(0.999, 0.80, 20)
    candidates = np.quantile(upper, quantiles)
    for thr in candidates:
        n_edges = int((upper >= thr).sum())
        if target_edges_low <= n_edges <= target_edges_high:
            return max(float(thr), floor)
    # No quantile hit the target band: pick the tightest threshold that stays
    # at or under the high target to avoid a hairball.
    for thr in candidates:
        n_edges = int((upper >= thr).sum())
        if n_edges <= target_edges_high:
            return max(float(thr), floor)
    return floor


def run_communities(
    con: duckdb.DuckDBPyConnection, settings: Settings, *, force: bool = False
) -> dict[str, int | float]:
    """Run Louvain on session centroids, write parquet.

    Parameters
    ----------
    con
        An open DuckDB connection with the v1 ``messages`` view registered.
    settings
        Configured Settings.  Honors ``louvain_edge_threshold`` (the floor
        cosine value), ``louvain_resolution``, ``louvain_target_avg_degree_low``,
        ``louvain_target_avg_degree_high``, and ``louvain_min_community_size``.
    force
        If False and output parquet exists, skip recomputation.

    Returns
    -------
    dict
        ``{sessions, communities, noise, threshold}`` — ``communities`` counts
        only real (>= min-size) communities; singletons are aggregated into
        ``noise``.  ``threshold`` is the adaptive cut actually used.
    """
    out_path = settings.communities_parquet_path

    if out_path.exists() and out_path.stat().st_size > 16 and not force:
        logger.info("Communities parquet exists at {}; pass force=True to rebuild", out_path)
        df = pl.read_parquet(out_path)
        real = df.filter(pl.col("community_id") != NOISE_COMMUNITY_ID)
        noise_n = df.height - real.height
        n_comm = int(real["community_id"].n_unique()) if real.height else 0
        return {
            "sessions": int(df.height),
            "communities": n_comm,
            "noise": noise_n,
            "threshold": float("nan"),
        }

    import networkx as nx
    from scipy.sparse import csr_matrix

    sids, centroids = _load_session_centroids(con, settings.embeddings_parquet_path)

    t0 = time.monotonic()
    logger.info("Computing pairwise cosine similarity over {} sessions...", len(sids))
    sim = centroids @ centroids.T
    np.fill_diagonal(sim, 0.0)

    floor = settings.louvain_edge_threshold
    threshold = _pick_adaptive_threshold(
        sim,
        floor=floor,
        target_avg_degree_low=settings.louvain_target_avg_degree_low,
        target_avg_degree_high=settings.louvain_target_avg_degree_high,
    )

    # Apply the chosen threshold.  Operate on a copy so we keep the raw sim
    # matrix intact for logging if we need to probe.
    masked = np.where(sim >= threshold, sim, 0.0).astype(np.float32, copy=False)
    sp = csr_matrix(masked)
    logger.info(
        "Graph: {} nodes, {} edges (threshold={:.3f}, avg_degree={:.1f}) in {:.1f}s",
        sp.shape[0],
        sp.nnz // 2,
        threshold,
        (sp.nnz / sp.shape[0]) if sp.shape[0] else 0.0,
        time.monotonic() - t0,
    )

    graph = nx.from_scipy_sparse_array(sp)
    t1 = time.monotonic()
    communities = nx.community.louvain_communities(
        graph,
        weight="weight",
        resolution=settings.louvain_resolution,
        seed=settings.seed,
    )

    min_size = settings.louvain_min_community_size
    rows: list[tuple[str, int, int]] = []
    real_communities = 0
    noise_count = 0
    # networkx.community.louvain_communities returns list[set[int]] at
    # runtime but the default stubs only expose it as Sized.  Cast through
    # typing.cast so the type checker agrees with reality.
    from typing import cast

    typed_communities = cast("list[set[int]]", communities)
    sorted_communities = cast(
        "list[set[int]]",
        sorted(typed_communities, key=len, reverse=True),
    )
    for cid, comm in enumerate(sorted_communities):
        size = len(comm)
        if size < min_size:
            for node in comm:
                rows.append((sids[node], NOISE_COMMUNITY_ID, 0))
                noise_count += 1
            continue
        real_communities += 1
        for node in comm:
            rows.append((sids[node], cid, size))

    logger.info(
        "Louvain: {} raw communities ({} kept >= {} sessions, {} singletons -> noise) in {:.1f}s",
        len(communities),
        real_communities,
        min_size,
        noise_count,
        time.monotonic() - t1,
    )

    df = pl.DataFrame(
        rows,
        schema={
            "session_id": pl.Utf8,
            "community_id": pl.Int32,
            "size": pl.Int32,
        },
        orient="row",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    logger.info("Wrote {} rows to {}", len(df), out_path)

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
        "communities": real_communities,
        "noise": noise_count,
        "threshold": float(threshold),
    }

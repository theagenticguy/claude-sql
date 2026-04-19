"""Session-level community detection via cosine graph + Louvain.

Pipeline
--------
1. Compute session centroid embeddings: group message embeddings by
   ``messages.session_id`` and average (mean over rows, then L2-normalize).
2. Build a sparse pairwise cosine-similarity matrix.
3. Threshold at ``louvain_edge_threshold`` (default 0.75) to form a sparse graph.
4. Run ``networkx.community.louvain_communities`` with seed=42.
5. Write ``session_communities.parquet`` with ``(session_id, community_id, size)``.

Public API
----------
run_communities(con, settings, *, force=False) -> dict[str, int]
    Expects ``con`` to have v1 views registered (``messages`` + access to
    the embeddings parquet via ``message_embeddings`` table or direct parquet
    read).  Returns stats dict.
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


def _load_session_centroids(
    con: duckdb.DuckDBPyConnection, embeddings_parquet_path: Path
) -> tuple[list[str], np.ndarray]:
    """Return ``(session_ids, centroids)`` where centroids is ``(N_sessions, dim)`` float32.

    Joins the embeddings parquet to the v1 ``messages`` view on uuid, groups
    by session_id, and averages the embedding.  Normalizes each centroid to
    unit L2 length so dot products equal cosine similarity.
    """
    logger.info("Loading message embeddings and joining to sessions...")
    # Pull (session_id, embedding) pairs via DuckDB.  The CAST to VARCHAR is
    # critical: messages.session_id is UUID-typed while the embeddings parquet
    # uuid column is VARCHAR.
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

    # polars Array(Float32, dim) -> numpy 2D.  .to_numpy() on an Array column
    # yields a 2D ndarray; on a List column it yields a 1D object ndarray of
    # sub-arrays.  Handle both.
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
    # L2-normalize so pairwise dot products equal cosine similarity.
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.where(norms == 0, 1.0, norms)
    logger.info("Computed {} session centroids (dim={})", len(unique), centroids.shape[1])
    return unique, centroids


def run_communities(
    con: duckdb.DuckDBPyConnection, settings: Settings, *, force: bool = False
) -> dict[str, int]:
    """Run Louvain on session centroids, write parquet.

    Parameters
    ----------
    con
        An open DuckDB connection with v1 ``messages`` view registered.
    settings
        Configured Settings.
    force
        If False and output parquet exists, skip recomputation.
    """
    out_path = settings.communities_parquet_path

    if out_path.exists() and out_path.stat().st_size > 16 and not force:
        logger.info("Communities parquet exists at {}; pass force=True to rebuild", out_path)
        df = pl.read_parquet(out_path)
        n_comm = int(df["community_id"].n_unique())
        return {"sessions": len(df), "communities": n_comm}

    # Heavy imports here - scipy and networkx are multi-hundred-ms each.
    import networkx as nx
    from scipy.sparse import csr_matrix

    sids, centroids = _load_session_centroids(con, settings.embeddings_parquet_path)

    t0 = time.monotonic()
    # Pairwise cosine via dot product (centroids already unit-normed).  For
    # modest N (<20k) the dense matmul is fastest.
    logger.info("Computing pairwise cosine similarity over {} sessions...", len(sids))
    sim = centroids @ centroids.T  # (N, N) float32
    np.fill_diagonal(sim, 0.0)  # drop self-loops
    thr = settings.louvain_edge_threshold
    # Mask below threshold.
    sim[sim < thr] = 0.0
    sp = csr_matrix(sim)
    logger.info(
        "Graph: {} nodes, {} edges (threshold={:.2f}) in {:.1f}s",
        sp.shape[0],
        sp.nnz // 2,
        thr,
        time.monotonic() - t0,
    )

    if sp.nnz == 0:
        logger.warning("Zero edges at threshold {:.2f} - lowering to 0.5", thr)
        sim = centroids @ centroids.T
        np.fill_diagonal(sim, 0.0)
        sim[sim < 0.5] = 0.0
        sp = csr_matrix(sim)

    graph = nx.from_scipy_sparse_array(sp)
    t1 = time.monotonic()
    communities = nx.community.louvain_communities(
        graph,
        weight="weight",
        resolution=settings.louvain_resolution,
        seed=settings.seed,
    )
    logger.info(
        "Louvain: {} communities in {:.1f}s",
        len(communities),
        time.monotonic() - t1,
    )

    # Flatten to rows.
    rows: list[tuple[str, int, int]] = []
    for cid, comm in enumerate(communities):
        size = len(comm)
        for node in comm:
            rows.append((sids[node], cid, size))

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

    size_dist = df.group_by("community_id").agg(pl.len().alias("n")).sort("n", descending=True)
    top = size_dist.head(5)
    logger.info("Top 5 community sizes: {}", top.to_dicts())

    return {"sessions": len(df), "communities": len(communities)}

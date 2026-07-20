"""Session-level community detection via mutual-kNN cosine graph + Leiden+CPM.

Orchestration for the ``community`` command (MIGRATION Phase C / T-3-1). This
module owns the I/O: load session centroids off the ``message_embeddings`` view,
project ``Settings`` into a ``CommunityConfig``, and write the primary
``session_communities.parquet`` plus the optional resolution-profile sidecar.
The pure graph-build + resolution-profile + partition + medoid/coherence +
relabel math lives in :mod:`claude_sql.domain.structure.community`.

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
   weakly-connected components.
6. Compute per-community medoid (max mean intra-community cosine) and
   coherence (mean intra-community cosine).
7. Stable relabel by descending size; collapse communities below
   ``leiden_min_community_size`` to ``NOISE_COMMUNITY_ID = -1``.
8. Write primary parquet ``session_communities.parquet`` with columns
   ``(session_id, community_id, size, is_medoid, coherence, gamma_used)`` and
   conditionally write the resolution-profile sidecar
   ``community_profile.parquet`` when auto-γ ran.

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
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from loguru import logger

from claude_sql.domain.structure.community import (
    NOISE_COMMUNITY_ID,
    ResolutionLevel,
    _build_igraph,
    _build_mutual_knn,
    _compute_medoid_and_coherence,
    _compute_resolution_profile,
    _pick_zoom,
    _relabel_and_collapse,
    _run_leiden_cpm,
    _warn_disconnected,
)
from claude_sql.infrastructure.settings import Settings


def _load_session_centroids(
    con: duckdb.DuckDBPyConnection, embeddings_parquet_path: Path
) -> tuple[list[str], np.ndarray]:
    """Return ``(session_ids, centroids)`` where centroids is ``(N_sessions, dim)`` float32.

    Joins the ``message_embeddings`` view (LanceDB-backed via ``register_vss``)
    to the v1 ``messages`` view on uuid, pulls one ``(session_id, embedding)``
    row per message ordered by session, then computes per-session means in
    numpy with a single ``np.add.reduceat`` segmented sum (sessions are
    contiguous after the ``ORDER BY``) followed by an L2-normalize. This keeps
    the intermediate at ``N_messages`` rows rather than the ``N_messages ×
    dim`` explosion the prior ``unnest``-per-dimension aggregation produced.

    ``embeddings_parquet_path`` is accepted for back-compat with callers that
    still pass it but is no longer consulted — the connection's
    ``message_embeddings`` view is the source of truth.
    """
    del embeddings_parquet_path  # legacy kwarg — view is the source of truth now
    logger.info("Loading message embeddings and joining to sessions...")
    # Pull one row per message (session_id, embedding) ordered by session, then
    # compute per-session means in numpy via a single segmented reduction.
    #
    # The prior implementation unnested every embedding into ``dim`` rows
    # (``generate_subscripts`` + ``unnest``) and grouped on (session, pos) —
    # that explodes the working set to N_messages × dim rows before the
    # average. Carrying the FLOAT[dim] vector through the join and reducing it
    # in numpy keeps the intermediate at N_messages rows and is 1.4–1.8×
    # faster on a 24k–96k-message corpus (measured), with the win widening as
    # the corpus grows. ``ORDER BY session_id`` makes the sessions contiguous
    # so ``np.add.reduceat`` can segment-sum without a Python per-session loop.
    sql = """
        SELECT CAST(m.session_id AS VARCHAR) AS session_id,
               e.embedding AS emb
          FROM message_embeddings e
          JOIN messages m
            ON CAST(m.uuid AS VARCHAR) = e.uuid
         ORDER BY session_id
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

    # ``emb`` is a DuckDB ``FLOAT[dim]`` column; polars surfaces it as a
    # fixed-size ``Array(Float32, dim)`` dtype (occasionally a variable
    # ``List`` if the cast was lost upstream). ``Series.to_numpy()`` extracts
    # the buffer directly into a contiguous ``(N_messages, dim)`` matrix; for
    # the fixed-``Array`` case it's a near-zero-copy view. The prior
    # ``np.asarray(series.to_list(), ...)`` boxed every one of the
    # N_messages × dim float32 values into a Python ``float`` object first —
    # measured at 6–10× slower and ~43× higher peak RSS on a 6k–96k-message
    # corpus (e.g. 7.4 s / 3.5 GB → 1.2 s / 83 MB at 96k). This is the
    # read-side analog of the SQL-side ``unnest`` explosion removed in #65.
    emb_arr = df["emb"].to_numpy()
    if emb_arr.ndim == 1:
        # Variable ``List`` dtype (or object array of rows) — stack to 2-D.
        emb_arr = np.stack(list(emb_arr))
    emb_np = np.ascontiguousarray(emb_arr, dtype=np.float32)
    sessions = df["session_id"].to_numpy()
    # ``return_index`` gives each group's first row offset on the sorted array;
    # ``np.unique`` returns the labels already sorted, matching the prior
    # ``ORDER BY 1`` contract. ``reduceat`` sums each [start_i, start_{i+1})
    # segment in one pass.
    sids_arr, starts, counts = np.unique(sessions, return_index=True, return_counts=True)
    summed = np.add.reduceat(emb_np, starts, axis=0)
    centroids = (summed / counts[:, None]).astype(np.float32)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.where(norms == 0, 1.0, norms)
    sids = sids_arr.tolist()
    logger.info("Computed {} session centroids (dim={})", len(sids), centroids.shape[1])
    return sids, centroids


def count_candidate_sessions(con: duckdb.DuckDBPyConnection) -> int:
    """Return the count of distinct sessions that have at least one embedding.

    The pure-SQL dry-run counter behind ``community --dry-run``: how many
    sessions Leiden+CPM would consider (one node per session with embeddings).
    The connection has ``message_embeddings`` registered as a view over the
    LanceDB dataset (ATTACH'd in ``register_vss``); this queries through the
    view rather than re-opening parquet shards directly.
    """
    row = con.execute(
        """
        SELECT COUNT(DISTINCT m.session_id) AS candidate_sessions
          FROM message_embeddings e
          JOIN messages m
            ON CAST(m.uuid AS VARCHAR) = e.uuid
        """,
    ).fetchone()
    return int(row[0]) if row else 0


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

    # Project the god-Settings down to the Leiden+CPM slice once (T-2-4). The
    # graph-build + profile + partition below read only ``cfg`` — no Bedrock
    # model ID or transcript glob leaks into the pure community math. The
    # private helpers already take plain scalars (their signatures are
    # test-pinned), so ``cfg`` is unpacked at each call site.
    cfg = settings.community_config()

    sids, centroids = _load_session_centroids(con, settings.embeddings_parquet_path)

    t0 = time.monotonic()
    logger.info("Computing pairwise cosine similarity over {} sessions...", len(sids))
    sim = centroids @ centroids.T
    np.fill_diagonal(sim, 0.0)

    edges, weights = _build_mutual_knn(sim, k=cfg.leiden_knn_k, floor=cfg.leiden_edge_floor)
    logger.info(
        "Mutual-kNN graph: {} nodes, {} edges (k={}, floor={:.2f}) in {:.1f}s",
        len(sids),
        len(edges),
        cfg.leiden_knn_k,
        cfg.leiden_edge_floor,
        time.monotonic() - t0,
    )

    g = _build_igraph(len(sids), edges, weights)

    profile_rows: list[tuple[float, int, float, int]] | None = None
    if gamma is None:
        # Honor cfg.leiden_resolution if it's set; else run the profile.
        if cfg.leiden_resolution is not None:
            gamma_used = float(cfg.leiden_resolution)
        else:
            t1 = time.monotonic()
            profile_rows = _compute_resolution_profile(
                g,
                range_lo=cfg.leiden_resolution_range_lo,
                range_hi=cfg.leiden_resolution_range_hi,
                seed=cfg.seed,
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
                gamma_used = (cfg.leiden_resolution_range_lo + cfg.leiden_resolution_range_hi) / 2.0
            else:
                gamma_used = _pick_zoom(profile_rows, resolution)
    else:
        gamma_used = float(gamma)

    t2 = time.monotonic()
    partition = _run_leiden_cpm(
        g,
        gamma=gamma_used,
        seed=cfg.seed,
        n_iterations=cfg.leiden_n_iterations,
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
        min_size=cfg.leiden_min_community_size,
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
        cfg.leiden_min_community_size,
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


__all__ = [
    "_load_session_centroids",
    "count_candidate_sessions",
    "neighbors_of",
    "run_communities",
]

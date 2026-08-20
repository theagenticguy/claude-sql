"""Cluster message embeddings via UMAP + HDBSCAN.

Orchestration for the ``cluster`` command (MIGRATION Phase C / T-3-1). This
module owns the I/O: read the LanceDB embeddings matrix, the mtime-sidecar
cache check, and the ``clusters.parquet`` write. The pure UMAP+HDBSCAN fit
lives in :func:`claude_sql.domain.structure.cluster.cluster_embeddings`.

Pipeline
--------
1. Read the LanceDB embeddings into a numpy float32 matrix.
2. UMAP reduce to 50d (for HDBSCAN) and 2d (for viz), both with ``seed``.
3. HDBSCAN on the 50d projection → cluster_id per row, -1 for noise.
4. Write ``clusters.parquet`` with ``(uuid, cluster_id, x, y, is_noise)``.

Public API
----------
run_clustering(settings, *, force=False) -> dict[str, int]
    Read embeddings, compute clusters, write output parquet.  Returns stats.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from loguru import logger

from claude_sql.domain.structure.cluster import cluster_embeddings
from claude_sql.infrastructure.freshness import (
    is_output_fresh,
    newest_mtime_ns,
    sidecar_for,
    stamp_output,
)
from claude_sql.infrastructure.settings import Settings

if TYPE_CHECKING:
    from claude_sql.application.ports import VectorStorePort


def _load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    """Read the LanceDB embeddings table → (uuid_list, embedding_matrix[float32]).

    Matrix shape (N, dim). Reads via the LanceDB Python API directly (not
    through the DuckDB ``message_embeddings`` view) so this worker can run
    independently of view registration on the calling connection.
    """
    # Deferred so importing this module via the CLI for a non-cluster command
    # doesn't pull in the ~2.6s lancedb import subtree.
    from claude_sql.infrastructure import lance_store

    db = lance_store.connect_db(path)
    if not lance_store._has_table(db, lance_store.TABLE_NAME):
        return [], np.zeros((0, 0), dtype=np.float32)
    tbl = db.open_table(lance_store.TABLE_NAME)
    arrow = tbl.to_arrow().select(["uuid", "embedding"])
    raw = pl.from_arrow(arrow)
    df = raw if isinstance(raw, pl.DataFrame) else raw.to_frame()
    uuids = df["uuid"].to_list()
    emb = df["embedding"].to_numpy()
    if emb.ndim == 1:
        emb = np.stack(list(emb))
    return uuids, np.ascontiguousarray(emb, dtype=np.float32)


def _stats_from_parquet(df: pl.DataFrame) -> dict[str, int]:
    """Derive the ``run_clustering`` stats dict from an existing clusters parquet.

    The ONE definition of those three numbers for every cache-hit path, because
    they are not interchangeable quantities and the compute path establishes what
    each one means:

    * ``total``  — rows (one per embedded message).
    * ``clusters`` — DISTINCT non-noise ``cluster_id`` values. The compute path
      reports ``labels.max() + 1``, a cluster count; reading it back as
      ``(cluster_id >= 0).sum()`` yields the count of non-noise *rows* instead,
      which is the same key carrying a different quantity. Observed on the live
      corpus as ``analyze/cluster: 139054 messages, 80057 clusters`` against a
      parquet holding 1,080 real clusters.
    * ``noise`` — rows labelled ``-1``. A row count on both paths, so it agrees.
    """
    real = df.filter(pl.col("cluster_id") >= 0)
    return {
        "total": len(df),
        "clusters": int(real["cluster_id"].n_unique()),
        "noise": int((df["cluster_id"] < 0).sum()),
    }


def run_clustering(
    settings: Settings, *, force: bool = False, store: VectorStorePort | None = None
) -> dict[str, int]:
    """Run UMAP + HDBSCAN on the embeddings parquet.

    Parameters
    ----------
    settings
        Configured Settings; reads ``embeddings_parquet_path`` and writes
        ``clusters_parquet_path``.
    force
        If False and clusters_parquet_path exists, return its stats without
        recomputing.  If True, always rerun.
    store
        Optional :class:`VectorStorePort` used for the "embeddings present"
        row-count guard. Defaults to the module-backed
        :class:`~claude_sql.infrastructure.adapters.LanceVectorStore` over
        ``settings.lance_uri`` (deferred build keeps the lancedb import off the
        CLI's non-cluster module-load path).

    Returns
    -------
    dict
        ``{"total": N, "clusters": K, "noise": M}`` where K excludes the
        noise cluster (label -1).
    """
    out_path = settings.clusters_parquet_path
    in_path = settings.lance_uri

    if store is None:
        from claude_sql.infrastructure.adapters import LanceVectorStore

        store = LanceVectorStore(in_path)
    if store.count_rows() == 0:
        raise FileNotFoundError(
            f"LanceDB embeddings missing at {in_path}. Run `claude-sql embed` first."
        )

    # Mtime-sidecar fast path: if the Lance dataset hasn't moved since the last
    # successful clustering, skip the UMAP+HDBSCAN refit (measured 1,247 s on
    # 128,453 embeddings with the viz projection on, 440 s without). Shares one
    # implementation with ``terms`` and ``community`` via ``infrastructure.freshness``
    # so the three stages cannot drift into different notions of "stale".
    sidecar = sidecar_for(out_path, input_name="embeddings")
    in_mtime_ns = newest_mtime_ns(in_path)
    if not force and is_output_fresh(out_path, sidecar=sidecar, input_mtime_ns=in_mtime_ns):
        logger.info("Embeddings unchanged since last cluster run; reusing {}.", out_path)
        return _stats_from_parquet(pl.read_parquet(out_path))

    # Legacy short-circuit: a clusters parquet exists but no sidecar (older
    # install before the mtime-skip landed). Trust the parquet and stamp
    # a sidecar so the next call hits the fast path. Forces a rebuild only
    # when ``force=True`` is set explicitly.
    if not force and out_path.exists() and out_path.stat().st_size > 16 and not sidecar.exists():
        logger.info(
            "Clusters parquet at {} predates sidecar; reusing and stamping mtime.",
            out_path,
        )
        stamp_output(sidecar, in_mtime_ns)
        return _stats_from_parquet(pl.read_parquet(out_path))

    # Project the god-Settings down to the pure-math slice (T-2-4). The
    # UMAP/HDBSCAN fit reads only ``cfg`` — it never sees a Bedrock model ID or
    # a transcript glob. The fit itself lives in the domain hexagon.
    cfg = settings.clustering_config()

    t0 = time.monotonic()
    uuids, matrix = _load_embeddings(in_path)
    logger.info("Loaded {} embeddings, shape={}, dtype={}", len(uuids), matrix.shape, matrix.dtype)

    labels, coords = cluster_embeddings(matrix, cfg)
    k = int(labels.max()) + 1 if labels.max() >= 0 else 0
    noise = int((labels < 0).sum())

    # Hand polars the numpy arrays directly — it ingests contiguous arrays
    # near-zero-copy. Round-tripping through ``.tolist()`` materialized N
    # boxed Python ints/floats/bools per column just to have polars re-parse
    # them back into the typed columns the schema already pins (mirrors the
    # read-side boxing fix in #68, now on the write side). ``coords`` columns
    # are sliced views, so copy to contiguous float32 before handing them over.
    #
    # ``x`` / ``y`` stay in the schema when the viz projection is off, holding
    # NULLs. The ``message_clusters`` view and its ``VIEW_SCHEMA`` entry name
    # those columns, so dropping them would break every bind; NULL says "not
    # computed" where 0.0 would assert a position at the origin.
    viz_x = np.ascontiguousarray(coords[:, 0], dtype=np.float32) if coords is not None else None
    viz_y = np.ascontiguousarray(coords[:, 1], dtype=np.float32) if coords is not None else None
    df = pl.DataFrame(
        {
            "uuid": uuids,
            "cluster_id": labels.astype(np.int32),
            "x": viz_x if viz_x is not None else [None] * len(uuids),
            "y": viz_y if viz_y is not None else [None] * len(uuids),
            "is_noise": labels < 0,
        },
        schema={
            "uuid": pl.Utf8,
            "cluster_id": pl.Int32,
            "x": pl.Float32,
            "y": pl.Float32,
            "is_noise": pl.Boolean,
        },
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    stamp_output(sidecar, in_mtime_ns)
    logger.info(
        "Wrote {} rows to {} (total elapsed: {:.1f}s)",
        len(df),
        out_path,
        time.monotonic() - t0,
    )

    return {"total": len(uuids), "clusters": k, "noise": noise}


__all__ = ["_load_embeddings", "_stats_from_parquet", "run_clustering"]

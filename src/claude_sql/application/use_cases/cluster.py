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

    # Mtime-sidecar fast path: if the Lance dataset hasn't moved since the
    # last successful clustering, skip the ~40 s UMAP+HDBSCAN refit. The
    # sidecar lives next to the output parquet and stores the dataset
    # directory's nanosecond mtime; hand-edits to clusters.parquet alone
    # won't bust the cache (use ``force=True`` for that).
    sidecar = out_path.with_suffix(out_path.suffix + ".embeddings_mtime")
    # Walk the Lance directory tree once and take the max mtime — Lance
    # writes new fragments to disk, so the deepest mtime tracks the data.
    import contextlib

    candidate_mtimes = [in_path.stat().st_mtime_ns]
    for child in in_path.rglob("*"):
        with contextlib.suppress(OSError):
            candidate_mtimes.append(child.stat().st_mtime_ns)
    in_mtime_ns = max(candidate_mtimes)
    if (
        not force
        and out_path.exists()
        and out_path.stat().st_size > 16
        and sidecar.exists()
        and sidecar.read_text().strip() == str(in_mtime_ns)
    ):
        logger.info("Embeddings unchanged since last cluster run; reusing {}.", out_path)
        df = pl.read_parquet(out_path)
        return {
            "total": len(df),
            "clusters": int((df["cluster_id"] >= 0).sum()),
            "noise": int((df["cluster_id"] < 0).sum()),
        }

    # Legacy short-circuit: a clusters parquet exists but no sidecar (older
    # install before the mtime-skip landed). Trust the parquet and stamp
    # a sidecar so the next call hits the fast path. Forces a rebuild only
    # when ``force=True`` is set explicitly.
    if not force and out_path.exists() and out_path.stat().st_size > 16 and not sidecar.exists():
        logger.info(
            "Clusters parquet at {} predates sidecar; reusing and stamping mtime.",
            out_path,
        )
        sidecar.write_text(str(in_mtime_ns))
        df = pl.read_parquet(out_path)
        return {
            "total": len(df),
            "clusters": int((df["cluster_id"] >= 0).sum()),
            "noise": int((df["cluster_id"] < 0).sum()),
        }

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
    df = pl.DataFrame(
        {
            "uuid": uuids,
            "cluster_id": labels.astype(np.int32),
            "x": np.ascontiguousarray(coords[:, 0], dtype=np.float32),
            "y": np.ascontiguousarray(coords[:, 1], dtype=np.float32),
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
    sidecar.write_text(str(in_mtime_ns))
    logger.info(
        "Wrote {} rows to {} (total elapsed: {:.1f}s)",
        len(df),
        out_path,
        time.monotonic() - t0,
    )

    return {"total": len(uuids), "clusters": k, "noise": noise}


__all__ = ["_load_embeddings", "run_clustering"]

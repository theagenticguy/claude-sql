"""Cluster message embeddings via UMAP + HDBSCAN.

Pipeline
--------
1. Read the embeddings parquet into a numpy float32 matrix.
2. UMAP reduce to 50d (for HDBSCAN) and 2d (for viz), both with ``seed=42``.
3. HDBSCAN on the 50d projection → cluster_id per row, -1 for noise.
4. Write ``clusters.parquet`` with ``(uuid, cluster_id, x, y, is_noise)``.

Public API
----------
run_clustering(settings, *, force=False) -> dict[str, int]
    Read parquet, compute clusters, write output parquet.  Returns stats dict.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from claude_sql.config import Settings


def _load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    """Read parquet → (uuid_list, embedding_matrix[float32]).  Matrix shape (N, dim)."""
    df = pl.read_parquet(path)
    uuids = df["uuid"].to_list()
    # polars Array[Float32, dim] → 2D numpy (N, dim).  to_numpy() on an Array column
    # returns 2D when the element type is fixed-size Array; fall back to vstack if not.
    emb = df["embedding"].to_numpy()
    if emb.ndim == 1:
        # Object array of 1D arrays (ragged container); stack into a 2D matrix.
        emb = np.stack(list(emb))
    return uuids, np.ascontiguousarray(emb, dtype=np.float32)


def run_clustering(settings: Settings, *, force: bool = False) -> dict[str, int]:
    """Run UMAP + HDBSCAN on the embeddings parquet.

    Parameters
    ----------
    settings
        Configured Settings; reads ``embeddings_parquet_path`` and writes
        ``clusters_parquet_path``.
    force
        If False and clusters_parquet_path exists, return its stats without
        recomputing.  If True, always rerun.

    Returns
    -------
    dict
        ``{"total": N, "clusters": K, "noise": M}`` where K excludes the
        noise cluster (label -1).
    """
    out_path = settings.clusters_parquet_path
    in_path = settings.embeddings_parquet_path

    if not in_path.exists() or in_path.stat().st_size < 16:
        raise FileNotFoundError(
            f"Embeddings parquet missing at {in_path}.  Run `claude-sql embed` first."
        )

    if out_path.exists() and out_path.stat().st_size > 16 and not force:
        logger.info("Clusters parquet already exists at {}.  Pass force=True to rebuild.", out_path)
        df = pl.read_parquet(out_path)
        return {
            "total": len(df),
            "clusters": int((df["cluster_id"] >= 0).sum()),
            "noise": int((df["cluster_id"] < 0).sum()),
        }

    # Heavy imports inside the function so module import stays cheap.
    import hdbscan
    import umap

    t0 = time.monotonic()
    uuids, X = _load_embeddings(in_path)  # noqa: N806 — X follows sklearn/ML matrix convention
    logger.info("Loaded {} embeddings, shape={}, dtype={}", len(uuids), X.shape, X.dtype)

    logger.info(
        "UMAP → {}d (clustering): n_neighbors={}, min_dist={}, metric={}",
        settings.umap_n_components_50,
        settings.umap_n_neighbors,
        settings.umap_min_dist_cluster,
        settings.umap_metric,
    )
    reducer_cluster = umap.UMAP(
        n_components=settings.umap_n_components_50,
        n_neighbors=settings.umap_n_neighbors,
        min_dist=settings.umap_min_dist_cluster,
        metric=settings.umap_metric,
        random_state=settings.seed,
    )
    X50 = reducer_cluster.fit_transform(X)  # noqa: N806 — 50d projection matrix
    logger.info("UMAP 50d done in {:.1f}s", time.monotonic() - t0)

    t1 = time.monotonic()
    logger.info(
        "UMAP → {}d (viz): n_neighbors={}, min_dist={}",
        settings.umap_n_components_2,
        settings.umap_n_neighbors,
        settings.umap_min_dist_viz,
    )
    reducer_viz = umap.UMAP(
        n_components=settings.umap_n_components_2,
        n_neighbors=settings.umap_n_neighbors,
        min_dist=settings.umap_min_dist_viz,
        metric=settings.umap_metric,
        random_state=settings.seed,
    )
    X2 = reducer_viz.fit_transform(X)  # noqa: N806 — 2d viz projection matrix
    logger.info("UMAP 2d done in {:.1f}s", time.monotonic() - t1)

    t2 = time.monotonic()
    logger.info(
        "HDBSCAN: min_cluster_size={}, min_samples={}",
        settings.hdbscan_min_cluster_size,
        settings.hdbscan_min_samples,
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=settings.hdbscan_min_cluster_size,
        min_samples=settings.hdbscan_min_samples,
        metric="euclidean",  # post-UMAP space is euclidean-friendly
        core_dist_n_jobs=-1,
        prediction_data=False,
    )
    labels = clusterer.fit_predict(X50)
    k = int(labels.max()) + 1 if labels.max() >= 0 else 0
    noise = int((labels < 0).sum())
    logger.info(
        "HDBSCAN done in {:.1f}s: {} clusters, {} noise ({:.1%})",
        time.monotonic() - t2,
        k,
        noise,
        noise / len(labels) if len(labels) else 0,
    )

    df = pl.DataFrame(
        {
            "uuid": uuids,
            "cluster_id": labels.astype(np.int32).tolist(),
            "x": X2[:, 0].astype(np.float32).tolist(),
            "y": X2[:, 1].astype(np.float32).tolist(),
            "is_noise": (labels < 0).tolist(),
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
    logger.info(
        "Wrote {} rows to {} (total elapsed: {:.1f}s)",
        len(df),
        out_path,
        time.monotonic() - t0,
    )

    return {"total": len(uuids), "clusters": k, "noise": noise}

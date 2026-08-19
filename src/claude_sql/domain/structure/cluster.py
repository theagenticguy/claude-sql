"""Pure UMAP + HDBSCAN clustering math (no I/O).

Carved from ``analytics/cluster_worker.run_clustering`` (MIGRATION Phase C /
T-3-1). ``cluster_embeddings`` takes an in-memory ``(N, dim)`` float32 matrix
and a frozen ``ClusteringConfig`` and returns the per-row cluster labels plus a
2-D viz projection. It performs no I/O — the LanceDB read, the mtime-sidecar
cache check, and the parquet write stay in
``application.use_cases.cluster.run_clustering``.

umap and hdbscan are lazy-imported inside the function so importing this module
(e.g. via the analytics shim) doesn't drag their import subtrees onto the CLI
fast path.

Determinism: ``cfg.seed`` threads into both UMAP ``random_state`` calls so
same-seed reruns produce byte-identical cluster IDs and coordinates.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from claude_sql.domain.config import ClusteringConfig


def cluster_embeddings(
    matrix: np.ndarray, cfg: ClusteringConfig
) -> tuple[np.ndarray, np.ndarray | None]:
    """Reduce with UMAP (50d, optionally + 2d) and label with HDBSCAN.

    The 50-d reduction is load-bearing: HDBSCAN's density estimates degrade at
    the embedding provider's native width (1024-d for Cohere Embed v4), and the
    post-UMAP space is what makes ``metric="euclidean"`` below sound.

    The 2-d reduction is a viz projection and nothing in the codebase reads it —
    it is written to ``clusters.parquet`` as ``x`` / ``y`` and available to
    ad-hoc SQL only. It is also the single most expensive step: measured on
    128,453 embeddings, the 50-d fit took 414 s and the 2-d fit took 807 s, so
    it is 66% of the stage's wall clock. ``cfg.compute_viz_coords`` gates it, and
    the default is off.

    Parameters
    ----------
    matrix
        Contiguous ``(N, dim)`` float32 embedding matrix.
    cfg
        UMAP + HDBSCAN hyperparameters (+ ``seed``, ``compute_viz_coords``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        ``(labels, coords)`` where ``labels`` is an ``(N,)`` int array of
        cluster ids (``-1`` for noise). ``coords`` is the ``(N, 2)`` float UMAP
        viz projection, or ``None`` when ``cfg.compute_viz_coords`` is false —
        ``None``, not zeros, so the caller writes SQL NULLs and a reader can
        tell "not computed" from "projected to the origin".
    """
    # Heavy imports inside the function so module import stays cheap.
    import hdbscan
    import umap

    t0 = time.monotonic()
    logger.info(
        "UMAP → {}d (clustering): n_neighbors={}, min_dist={}, metric={}",
        cfg.umap_n_components_50,
        cfg.umap_n_neighbors,
        cfg.umap_min_dist_cluster,
        cfg.umap_metric,
    )
    reducer_cluster = umap.UMAP(
        n_components=cfg.umap_n_components_50,
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist_cluster,
        metric=cfg.umap_metric,
        random_state=cfg.seed,
    )
    x50 = reducer_cluster.fit_transform(matrix)  # 50d projection matrix
    logger.info("UMAP 50d done in {:.1f}s", time.monotonic() - t0)

    coords: np.ndarray | None = None
    if cfg.compute_viz_coords:
        t1 = time.monotonic()
        logger.info(
            "UMAP → {}d (viz): n_neighbors={}, min_dist={}",
            cfg.umap_n_components_2,
            cfg.umap_n_neighbors,
            cfg.umap_min_dist_viz,
        )
        reducer_viz = umap.UMAP(
            n_components=cfg.umap_n_components_2,
            n_neighbors=cfg.umap_n_neighbors,
            min_dist=cfg.umap_min_dist_viz,
            metric=cfg.umap_metric,
            random_state=cfg.seed,
        )
        coords = reducer_viz.fit_transform(matrix)  # 2d viz projection matrix
        logger.info("UMAP 2d done in {:.1f}s", time.monotonic() - t1)
    else:
        logger.info("UMAP 2d viz projection skipped (no consumer; enable with --viz)")

    t2 = time.monotonic()
    logger.info(
        "HDBSCAN: min_cluster_size={}, min_samples={}",
        cfg.hdbscan_min_cluster_size,
        cfg.hdbscan_min_samples,
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
        metric="euclidean",  # post-UMAP space is euclidean-friendly
        core_dist_n_jobs=-1,
        prediction_data=False,
    )
    labels = clusterer.fit_predict(x50)
    k = int(labels.max()) + 1 if labels.max() >= 0 else 0
    noise = int((labels < 0).sum())
    logger.info(
        "HDBSCAN done in {:.1f}s: {} clusters, {} noise ({:.1%})",
        time.monotonic() - t2,
        k,
        noise,
        noise / len(labels) if len(labels) else 0,
    )
    return labels, coords


__all__ = ["cluster_embeddings"]

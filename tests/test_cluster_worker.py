"""Synthetic-fixture smoke test for UMAP+HDBSCAN clustering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from claude_sql.cluster_worker import run_clustering
from claude_sql.config import Settings


def _make_synthetic_embeddings(
    path: Path,
    *,
    n_clusters: int = 5,
    per_cluster: int = 50,
    dim: int = 64,
    seed: int = 0,
) -> None:
    """Write a synthetic embeddings parquet with obvious cluster structure."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim)).astype(np.float32)
    # Normalize centers and spread points around each
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    vectors: list[np.ndarray] = []
    ids: list[str] = []
    for i in range(n_clusters):
        noise = rng.normal(scale=0.05, size=(per_cluster, dim)).astype(np.float32)
        pts = centers[i] + noise
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        vectors.append(pts)
        ids.extend(f"cluster-{i:02d}-pt-{j:03d}" for j in range(per_cluster))
    e = np.vstack(vectors)
    df = pl.DataFrame(
        {
            "uuid": ids,
            "model": ["test"] * len(ids),
            "dim": [dim] * len(ids),
            "embedding": e,
        },
        schema={
            "uuid": pl.Utf8,
            "model": pl.Utf8,
            "dim": pl.UInt16,
            "embedding": pl.Array(pl.Float32, dim),
        },
    )
    df.write_parquet(path)


@pytest.fixture
def synthetic_settings(tmp_path: Path) -> Settings:
    emb = tmp_path / "embeddings.parquet"
    _make_synthetic_embeddings(emb, n_clusters=5, per_cluster=50, dim=64, seed=42)
    return Settings(
        embeddings_parquet_path=emb,
        clusters_parquet_path=tmp_path / "clusters.parquet",
        # Shrink UMAP for speed -- n_neighbors must be < n_samples
        umap_n_components_50=10,
        umap_n_components_2=2,
        umap_n_neighbors=15,
        hdbscan_min_cluster_size=10,
        hdbscan_min_samples=3,
        output_dimension=1024,  # not used by clustering; keep default-shaped
    )


def test_clustering_smoke(synthetic_settings: Settings) -> None:
    stats = run_clustering(synthetic_settings, force=True)
    assert stats["total"] == 250
    # Synthetic fixture was five well-separated clusters; HDBSCAN should find at least 2
    assert stats["clusters"] >= 2
    # Output parquet exists with the right columns
    df = pl.read_parquet(synthetic_settings.clusters_parquet_path)
    assert set(df.columns) == {"uuid", "cluster_id", "x", "y", "is_noise"}
    assert len(df) == 250


def test_clustering_skips_when_parquet_exists(synthetic_settings: Settings) -> None:
    run_clustering(synthetic_settings, force=True)
    # Second call with force=False should skip recomputation
    stats = run_clustering(synthetic_settings, force=False)
    assert stats["total"] == 250


def test_clustering_errors_when_input_missing(tmp_path: Path) -> None:
    s = Settings(
        embeddings_parquet_path=tmp_path / "nope.parquet",
        clusters_parquet_path=tmp_path / "clusters.parquet",
    )
    with pytest.raises(FileNotFoundError):
        run_clustering(s, force=True)

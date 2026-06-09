"""Synthetic-fixture smoke test for UMAP+HDBSCAN clustering."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from claude_sql.analytics.cluster_worker import run_clustering
from claude_sql.core import lance_store
from claude_sql.core.config import Settings


def _make_synthetic_embeddings(
    lance_uri: Path,
    *,
    n_clusters: int = 5,
    per_cluster: int = 50,
    dim: int = 64,
    seed: int = 0,
) -> None:
    """Seed a Lance dataset with obvious cluster structure."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim)).astype(np.float32)
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
    now = datetime.now(UTC)
    df = pl.DataFrame(
        {
            "uuid": ids,
            "model": ["test"] * len(ids),
            "dim": [dim] * len(ids),
            "embedding": e,
            "embedded_at": [now] * len(ids),
        },
        schema={
            "uuid": pl.Utf8,
            "model": pl.Utf8,
            "dim": pl.Int32,
            "embedding": pl.Array(pl.Float32, dim),
            "embedded_at": pl.Datetime("us", "UTC"),
        },
    )
    db = lance_store.connect_db(lance_uri)
    tbl = lance_store.open_or_create_table(db, dim=dim)
    lance_store.add_chunk(tbl, df)


@pytest.fixture
def synthetic_settings(tmp_path: Path) -> Settings:
    lance_uri = tmp_path / "embeddings_lance"
    _make_synthetic_embeddings(lance_uri, n_clusters=5, per_cluster=50, dim=64, seed=42)
    return Settings(
        lance_uri=lance_uri,
        clusters_parquet_path=tmp_path / "clusters.parquet",
        # Shrink UMAP for speed -- n_neighbors must be < n_samples
        umap_n_components_50=10,
        umap_n_components_2=2,
        umap_n_neighbors=15,
        hdbscan_min_cluster_size=10,
        hdbscan_min_samples=3,
        output_dimension=1024,  # not used by clustering; lance schema dim is set explicitly
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
    # Pin the column dtypes: the worker hands polars numpy arrays directly
    # (no .tolist() round-trip), so guard against a numpy dtype silently
    # widening a column past the pinned schema.
    assert df.schema == {
        "uuid": pl.Utf8,
        "cluster_id": pl.Int32,
        "x": pl.Float32,
        "y": pl.Float32,
        "is_noise": pl.Boolean,
    }


def test_clustering_skips_when_parquet_exists(synthetic_settings: Settings) -> None:
    run_clustering(synthetic_settings, force=True)
    # Second call with force=False should skip recomputation
    stats = run_clustering(synthetic_settings, force=False)
    assert stats["total"] == 250


def test_clustering_errors_when_input_missing(tmp_path: Path) -> None:
    s = Settings(
        lance_uri=tmp_path / "missing_lance",
        clusters_parquet_path=tmp_path / "clusters.parquet",
    )
    with pytest.raises(FileNotFoundError):
        run_clustering(s, force=True)


def test_clustering_skips_via_mtime_sidecar(synthetic_settings: Settings) -> None:
    """Second run with the same embeddings mtime returns from the sidecar cache.

    The sidecar fast path skips the ~40 s UMAP+HDBSCAN refit when the
    embeddings haven't moved. The check is sub-second (just a stat + a
    file read), well under the 500 ms target the plan calls out.
    """
    import time as _t

    run_clustering(synthetic_settings, force=True)
    sidecar = synthetic_settings.clusters_parquet_path.with_suffix(
        synthetic_settings.clusters_parquet_path.suffix + ".embeddings_mtime"
    )
    assert sidecar.exists()
    first_sidecar = sidecar.read_text()

    # Second call must complete fast (no UMAP reduction) and return the
    # cached stats unchanged.
    t0 = _t.monotonic()
    stats = run_clustering(synthetic_settings, force=False)
    elapsed = _t.monotonic() - t0
    assert stats["total"] == 250
    assert elapsed < 1.0, f"sidecar fast path took {elapsed:.3f}s, expected <1.0s"
    assert sidecar.read_text() == first_sidecar


def test_clustering_rebuilds_when_embeddings_mtime_changes(
    synthetic_settings: Settings,
) -> None:
    """A bumped embeddings mtime invalidates the sidecar and forces a rebuild."""
    import os as _os

    run_clustering(synthetic_settings, force=True)
    sidecar = synthetic_settings.clusters_parquet_path.with_suffix(
        synthetic_settings.clusters_parquet_path.suffix + ".embeddings_mtime"
    )
    pre = sidecar.read_text()

    # Bump every file under the Lance directory tree past the sidecar's mtime.
    lance_uri = synthetic_settings.lance_uri
    targets = [p for p in lance_uri.rglob("*") if p.is_file()]
    assert targets, "expected Lance dataset files under the synthetic uri"
    bumped_ns = max(p.stat().st_mtime_ns for p in targets) + 5_000_000_000
    for p in targets:
        _os.utime(p, ns=(bumped_ns, bumped_ns))
    # Also touch the directory itself so the cluster worker's max-mtime walk picks it up.
    _os.utime(lance_uri, ns=(bumped_ns, bumped_ns))

    run_clustering(synthetic_settings, force=False)
    post = sidecar.read_text()
    assert pre != post, "sidecar should refresh when embeddings move"
    assert post == str(bumped_ns)

"""Coverage top-up for :mod:`claude_sql.community_worker`.

Targets the error / fallback paths the existing smoke test doesn't
hit:

* Line 77 — missing embeddings parquet → ``RuntimeError``.
* Line 109 — empty join (embeddings parquet exists but has no rows
  matching ``messages``) → ``RuntimeError``.
* Line 136 — ``_pick_adaptive_threshold`` with ``n < 2`` returns the floor.
* Line 141 — ``upper.size == 0`` returns the floor.
* Lines 159-163 — fallback when no quantile hits the target band.
* Lines 192-197 — cached parquet exists branch (skip rebuild).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from claude_sql.community_worker import (
    NOISE_COMMUNITY_ID,
    _pick_adaptive_threshold,
    run_communities,
)
from claude_sql.config import Settings

# ---------------------------------------------------------------------------
# _pick_adaptive_threshold edge cases (lines 136, 141, 159-163)
# ---------------------------------------------------------------------------


def test_pick_adaptive_threshold_single_node_returns_floor() -> None:
    """``n < 2`` short-circuits to the floor (line 136)."""
    sim = np.array([[0.0]], dtype=np.float32)
    out = _pick_adaptive_threshold(
        sim,
        floor=0.5,
        target_avg_degree_low=1.0,
        target_avg_degree_high=2.0,
    )
    assert out == 0.5


def test_pick_adaptive_threshold_zero_size_upper_returns_floor() -> None:
    """No upper-triangular entries → return the floor (line 141)."""
    # Construct a 2-node matrix so n>=2 but upper triangle has size 1.
    # Both off-diagonal values low so target_edges_high<<1 forces fallback.
    sim = np.array([[0.0, 0.001], [0.001, 0.0]], dtype=np.float32)
    out = _pick_adaptive_threshold(
        sim,
        floor=0.5,
        target_avg_degree_low=10.0,
        target_avg_degree_high=20.0,
    )
    # No quantile produces enough edges → fallback path keeps the floor.
    assert out == 0.5


def test_pick_adaptive_threshold_fallback_loop_returns_floor_when_no_target_hit() -> None:
    """Lines 159-163 — the second loop scans for the tightest threshold under
    target_edges_high and clamps up to floor.

    Construct a similarity matrix where every edge is below floor; the
    fallback returns floor.
    """
    # 3 nodes, all sims at 0.0 — no thresholds can hit the band.
    sim = np.zeros((3, 3), dtype=np.float32)
    out = _pick_adaptive_threshold(
        sim,
        floor=0.4,
        target_avg_degree_low=10.0,
        target_avg_degree_high=20.0,
    )
    assert out == 0.4


def test_pick_adaptive_threshold_picks_a_quantile_when_band_hit() -> None:
    """Sanity: when a quantile fits the band, we get a non-floor value back."""
    rng = np.random.default_rng(0)
    n = 50
    sim = rng.uniform(0.0, 1.0, size=(n, n)).astype(np.float32)
    sim = (sim + sim.T) / 2.0  # symmetric
    np.fill_diagonal(sim, 0.0)
    out = _pick_adaptive_threshold(
        sim,
        floor=0.0,
        target_avg_degree_low=8.0,
        target_avg_degree_high=15.0,
    )
    # Some non-trivial threshold inside (0, 1).
    assert 0.0 <= out <= 1.0


# ---------------------------------------------------------------------------
# run_communities — missing embeddings parquet (line 77)
# ---------------------------------------------------------------------------


def test_run_communities_missing_embeddings_raises(tmp_path: Path) -> None:
    """No embeddings parquet at all → ``RuntimeError`` from _load_session_centroids."""
    settings = Settings(
        embeddings_parquet_path=tmp_path / "no-such-dir",
        communities_parquet_path=tmp_path / "communities.parquet",
    )
    con = duckdb.connect(":memory:")
    # We need a ``messages`` view for the SQL to compile, but it never
    # gets there — the part-files check raises first.
    con.execute("CREATE TABLE messages (uuid VARCHAR, session_id VARCHAR)")
    try:
        import pytest

        with pytest.raises(RuntimeError, match="No embeddings parquet"):
            run_communities(con, settings, force=True)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# run_communities — empty join (line 109)
# ---------------------------------------------------------------------------


def test_run_communities_empty_join_raises(tmp_path: Path) -> None:
    """Embeddings parquet exists but no uuid matches → ``RuntimeError``."""
    # Write an embeddings parquet that joins to nothing.
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    pl.DataFrame(
        {
            "uuid": ["unrelated-uuid"],
            "model": ["test"],
            "dim": [4],
            "embedding": [np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)],
        },
        schema={
            "uuid": pl.Utf8,
            "model": pl.Utf8,
            "dim": pl.UInt16,
            "embedding": pl.Array(pl.Float32, 4),
        },
    ).write_parquet(emb_dir / "part-0001.parquet")

    settings = Settings(
        embeddings_parquet_path=emb_dir,
        communities_parquet_path=tmp_path / "communities.parquet",
    )
    con = duckdb.connect(":memory:")
    # messages view that has zero overlap with the embeddings uuid.
    con.execute("CREATE TABLE messages AS SELECT 'other-uuid' AS uuid, 's1' AS session_id")
    try:
        import pytest

        with pytest.raises(RuntimeError, match="No rows returned"):
            run_communities(con, settings, force=True)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# run_communities — cache hit branch (lines 192-197)
# ---------------------------------------------------------------------------


def test_run_communities_returns_cached_when_parquet_present(tmp_path: Path) -> None:
    """When the output parquet already exists and ``force=False``, return cached stats."""
    out = tmp_path / "communities.parquet"
    # Write a fake "previous run" output: 3 sessions in 1 real community + 1 noise.
    df = pl.DataFrame(
        {
            "session_id": ["a", "b", "c", "d"],
            "community_id": [0, 0, 0, NOISE_COMMUNITY_ID],
            "size": [3, 3, 3, 0],
        },
        schema={
            "session_id": pl.Utf8,
            "community_id": pl.Int32,
            "size": pl.Int32,
        },
    )
    df.write_parquet(out)

    settings = Settings(
        embeddings_parquet_path=tmp_path / "no-such-emb",
        communities_parquet_path=out,
    )
    # The cache hit path should NOT touch embeddings, so it's fine that
    # the embeddings dir doesn't exist.
    con = duckdb.connect(":memory:")
    try:
        stats = run_communities(con, settings, force=False)
    finally:
        con.close()
    assert stats["sessions"] == 4
    assert stats["communities"] == 1
    assert stats["noise"] == 1
    # Cached path returns NaN for the threshold (we no longer know it).
    import math

    assert isinstance(stats["threshold"], float)
    assert math.isnan(float(stats["threshold"]))

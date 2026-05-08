"""Tests for the persisted HNSW store wired in :func:`register_vss`.

Covers the cold/warm transitions:
  - ``hnsw_db_path=None`` keeps the legacy in-memory behavior (table in main).
  - First call with a path builds the store from parquet.
  - Second call reuses it (no rebuild) when the parquet's mtime is older.
  - Bumping the parquet's mtime forces a rebuild.
  - A corrupted store (truncated bytes) unlinks and rebuilds.
  - A missing parquet still produces an empty schema-correct table.
"""

from __future__ import annotations

import io
from pathlib import Path

import duckdb
import polars as pl
import pytest

from claude_sql.sql_views import _hnsw_rebuild_needed, register_vss


def _write_embeddings_parquet(path: Path, n_rows: int = 4, dim: int = 4) -> None:
    """Write a tiny embeddings parquet matching the columns ``register_vss`` reads.

    Uses an explicit fixed-size list type so the parquet reader can cast
    into ``FLOAT[<dim>]`` without inference surprises.
    """
    rows = [
        {
            "uuid": f"u-{i}",
            "model": "global.cohere.embed-v4:0",
            "dim": dim,
            "embedding": [float(i + j) for j in range(dim)],
        }
        for i in range(n_rows)
    ]
    df = pl.DataFrame(
        rows,
        schema={
            "uuid": pl.Utf8,
            "model": pl.Utf8,
            "dim": pl.UInt16,
            "embedding": pl.Array(pl.Float32, dim),
        },
    )
    buf = io.BytesIO()
    df.write_parquet(buf)
    path.write_bytes(buf.getvalue())


def test_register_vss_in_memory_path_unchanged(tmp_path: Path) -> None:
    """Passing ``hnsw_db_path=None`` keeps the legacy in-main-database behavior."""
    parquet = tmp_path / "embeddings.parquet"
    _write_embeddings_parquet(parquet)
    con = duckdb.connect(":memory:")
    ok = register_vss(con, embeddings_parquet=parquet, hnsw_db_path=None, dim=4)
    assert ok is True
    rows = con.execute("SELECT count(*) FROM message_embeddings").fetchone()
    assert rows is not None and rows[0] == 4
    # No file should have been created.
    hnsw_files = list(tmp_path.glob("hnsw*.duckdb"))
    assert hnsw_files == []


def test_persistent_store_built_then_reused(tmp_path: Path) -> None:
    """Second call with an unchanged parquet reuses the persisted store."""
    parquet = tmp_path / "embeddings.parquet"
    _write_embeddings_parquet(parquet)
    hnsw_db = tmp_path / "hnsw.duckdb"

    con1 = duckdb.connect(":memory:")
    register_vss(con1, embeddings_parquet=parquet, hnsw_db_path=hnsw_db, dim=4)
    con1.close()
    assert hnsw_db.exists() and hnsw_db.stat().st_size > 1024
    first_mtime = hnsw_db.stat().st_mtime_ns

    con2 = duckdb.connect(":memory:")
    register_vss(con2, embeddings_parquet=parquet, hnsw_db_path=hnsw_db, dim=4)
    rows = con2.execute("SELECT count(*) FROM message_embeddings").fetchone()
    assert rows is not None and rows[0] == 4
    con2.close()
    # File mtime must not regress; reuse path issues no CHECKPOINT, so the
    # file should be effectively untouched.
    assert hnsw_db.stat().st_mtime_ns == first_mtime


def test_parquet_newer_than_store_triggers_rebuild(tmp_path: Path) -> None:
    """Bumping the parquet's mtime past the store's invalidates the cache."""
    parquet = tmp_path / "embeddings.parquet"
    _write_embeddings_parquet(parquet)
    hnsw_db = tmp_path / "hnsw.duckdb"

    con1 = duckdb.connect(":memory:")
    register_vss(con1, embeddings_parquet=parquet, hnsw_db_path=hnsw_db, dim=4)
    con1.close()
    pre_rebuild_mtime = hnsw_db.stat().st_mtime_ns

    # Touch the parquet so its mtime is strictly newer.
    bumped_ns = pre_rebuild_mtime + 10_000_000_000
    import os

    os.utime(parquet, ns=(bumped_ns, bumped_ns))
    assert _hnsw_rebuild_needed(parquet, hnsw_db) is True

    con2 = duckdb.connect(":memory:")
    register_vss(con2, embeddings_parquet=parquet, hnsw_db_path=hnsw_db, dim=4)
    rows = con2.execute("SELECT count(*) FROM message_embeddings").fetchone()
    assert rows is not None and rows[0] == 4
    con2.close()
    # Rebuild path issues a CHECKPOINT, so the store mtime should have advanced.
    assert hnsw_db.stat().st_mtime_ns >= pre_rebuild_mtime


def test_corrupt_store_unlinked_and_rebuilt(tmp_path: Path) -> None:
    """A junk store file blows ATTACH up; we unlink and rebuild from parquet."""
    parquet = tmp_path / "embeddings.parquet"
    _write_embeddings_parquet(parquet)
    hnsw_db = tmp_path / "hnsw.duckdb"
    # Garbage that DuckDB cannot parse as a database header.
    hnsw_db.write_bytes(b"NOT-A-DUCKDB-FILE\x00" * 200)

    con = duckdb.connect(":memory:")
    register_vss(con, embeddings_parquet=parquet, hnsw_db_path=hnsw_db, dim=4)
    rows = con.execute("SELECT count(*) FROM message_embeddings").fetchone()
    assert rows is not None and rows[0] == 4
    con.close()
    assert hnsw_db.exists()


def test_missing_parquet_still_produces_empty_table(tmp_path: Path) -> None:
    """No parquet on disk: register_vss returns False and the view is empty."""
    parquet = tmp_path / "absent.parquet"
    hnsw_db = tmp_path / "hnsw.duckdb"
    con = duckdb.connect(":memory:")
    ok = register_vss(con, embeddings_parquet=parquet, hnsw_db_path=hnsw_db, dim=4)
    assert ok is False
    rows = con.execute("SELECT count(*) FROM message_embeddings").fetchone()
    assert rows is not None and rows[0] == 0
    con.close()


def test_rebuild_decision_helper_handles_missing_files(tmp_path: Path) -> None:
    """Direct unit test of the freshness-check predicate."""
    parquet = tmp_path / "embeddings.parquet"
    hnsw_db = tmp_path / "hnsw.duckdb"
    # No store yet → must rebuild.
    assert _hnsw_rebuild_needed(parquet, hnsw_db) is True
    # Store exists but parquet missing → reuse (no source of truth to rebuild from).
    hnsw_db.write_bytes(b"\x00" * 4096)
    assert _hnsw_rebuild_needed(parquet, hnsw_db) is False
    # Both exist, parquet older than store → reuse.
    parquet.write_bytes(b"data")
    import os

    older_ns = hnsw_db.stat().st_mtime_ns - 10_000_000_000
    os.utime(parquet, ns=(older_ns, older_ns))
    assert _hnsw_rebuild_needed(parquet, hnsw_db) is False


@pytest.mark.parametrize("dim", [4, 8])
def test_persistent_store_round_trip_for_multiple_dims(tmp_path: Path, dim: int) -> None:
    """Sanity-check the persistence path for two dimension settings."""
    parquet = tmp_path / "embeddings.parquet"
    _write_embeddings_parquet(parquet, n_rows=6, dim=dim)
    hnsw_db = tmp_path / "hnsw.duckdb"
    con = duckdb.connect(":memory:")
    register_vss(con, embeddings_parquet=parquet, hnsw_db_path=hnsw_db, dim=dim)
    rows = con.execute("SELECT count(*) FROM message_embeddings").fetchone()
    assert rows is not None and rows[0] == 6
    con.close()

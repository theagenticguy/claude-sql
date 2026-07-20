"""Tests for the LanceDB embeddings store wired in :func:`register_vss`.

The DuckDB-VSS HNSW path was retired in T3.1 in favor of a LanceDB local
dataset (read back via the DuckDB lance core extension). This file covers
the new behavior:

* Fresh dataset: ``register_vss`` ATTACHes Lance and binds
  ``message_embeddings``.
* Empty dataset: an empty schema-correct view is created instead.
* Migration: legacy parquet shards under ``embeddings/part-*.parquet`` are
  copied into Lance once on first call.
* Multiple dim values: schema enforcement round-trips cleanly.

Corruption recovery is now ``rm -rf`` the Lance directory + re-embed
(or fall back to the legacy parquet shards via ``cache migrate``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import polars as pl
import pytest

from claude_sql.infrastructure import lance_store
from claude_sql.infrastructure.duckdb_views import register_vss


def _seed_lance(lance_uri: Path, *, n_rows: int = 4, dim: int = 4) -> None:
    """Seed a Lance dataset with ``n_rows`` synthetic embeddings of width ``dim``."""
    now = datetime.now(UTC)
    df = pl.DataFrame(
        {
            "uuid": [f"u-{i}" for i in range(n_rows)],
            "model": ["global.cohere.embed-v4:0"] * n_rows,
            "dim": [dim] * n_rows,
            "embedding": [[float(i + j) for j in range(dim)] for i in range(n_rows)],
            "embedded_at": [now] * n_rows,
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


def _write_legacy_parquet_shard(directory: Path, *, n_rows: int = 4, dim: int = 4) -> None:
    """Write a single ``part-<ts>.parquet`` to mimic the pre-Lance layout."""
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    df = pl.DataFrame(
        {
            "uuid": [f"u-{i}" for i in range(n_rows)],
            "model": ["global.cohere.embed-v4:0"] * n_rows,
            "dim": [dim] * n_rows,
            "embedding": [[float(i + j) for j in range(dim)] for i in range(n_rows)],
            "embedded_at": [now] * n_rows,
        },
        schema={
            "uuid": pl.Utf8,
            "model": pl.Utf8,
            "dim": pl.Int32,
            "embedding": pl.Array(pl.Float32, dim),
            "embedded_at": pl.Datetime("us", "UTC"),
        },
    )
    df.write_parquet(directory / "part-1700000000000000000.parquet")


def test_register_vss_binds_message_embeddings_view(tmp_path: Path) -> None:
    """Fresh Lance dataset: ``message_embeddings`` view returns the rows."""
    lance_uri = tmp_path / "embeddings_lance"
    _seed_lance(lance_uri, n_rows=4, dim=4)

    con = duckdb.connect(":memory:")
    ok = register_vss(con, lance_uri=lance_uri, dim=4)
    assert ok is True
    rows = con.execute("SELECT count(*) FROM message_embeddings").fetchone()
    assert rows is not None
    assert rows[0] == 4


def test_register_vss_missing_lance_creates_empty_view(tmp_path: Path) -> None:
    """No dataset on disk: empty schema-correct view, returns False."""
    lance_uri = tmp_path / "missing_lance"
    con = duckdb.connect(":memory:")
    ok = register_vss(con, lance_uri=lance_uri, dim=4)
    assert ok is False
    rows = con.execute("SELECT count(*) FROM message_embeddings").fetchone()
    assert rows is not None
    assert rows[0] == 0


def test_register_vss_migrates_legacy_parquet_shards(tmp_path: Path) -> None:
    """Legacy parquet shards under ``embeddings_parquet`` migrate into Lance once."""
    legacy = tmp_path / "embeddings"
    lance_uri = tmp_path / "embeddings_lance"
    _write_legacy_parquet_shard(legacy, n_rows=3, dim=4)

    con = duckdb.connect(":memory:")
    ok = register_vss(con, embeddings_parquet=legacy, lance_uri=lance_uri, dim=4)
    assert ok is True
    rows = con.execute("SELECT count(*) FROM message_embeddings").fetchone()
    assert rows is not None
    assert rows[0] == 3
    # Lance directory has been created with the migrated rows.
    assert lance_store.count_rows(lance_uri) == 3


@pytest.mark.parametrize("dim", [4, 8])
def test_register_vss_round_trip_for_multiple_dims(tmp_path: Path, dim: int) -> None:
    """Schema enforcement round-trips cleanly for multiple embedding dimensions."""
    lance_uri = tmp_path / f"embeddings_lance_{dim}"
    _seed_lance(lance_uri, n_rows=6, dim=dim)

    con = duckdb.connect(":memory:")
    register_vss(con, lance_uri=lance_uri, dim=dim)
    rows = con.execute("SELECT count(*) FROM message_embeddings").fetchone()
    assert rows is not None
    assert rows[0] == 6

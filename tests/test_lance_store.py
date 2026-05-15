"""Tests pinning the LanceDB integration to its actual API shape.

Two gotchas surfaced during the parquet→Lance migration:

1. ``lancedb.DBConnection.list_tables()`` returns a pydantic
   ``ListTablesResponse(tables=[...], page_token=...)`` — not a flat
   list, not a paginated tuple sequence. ``.table_names()`` is
   deprecated. ``_has_table`` MUST use ``list_tables().tables``.

2. DuckDB's ``ATTACH '<dir>' (TYPE LANCE)`` succeeds against any path —
   even one with no Lance dataset on disk. The catalog error fires
   later, at SELECT/CREATE-VIEW time, with
   ``Catalog Error: Table … embeddings does not exist``. The right
   empty-dataset gate is "is the table actually in the namespace?",
   not "does the directory have any files in it?".

Both behaviors were verified live against lancedb 0.30.2 and DuckDB
1.5.2 before this file was written.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import polars as pl

from claude_sql.core import lance_store
from claude_sql.core.sql_views import register_vss


def _seed_one_row(lance_uri: Path, uuid: str, *, dim: int = 4) -> None:
    """Append a single Lance row at ``lance_uri`` so the table exists."""
    db = lance_store.connect_db(lance_uri)
    tbl = lance_store.open_or_create_table(db, dim=dim)
    df = pl.DataFrame(
        {
            "uuid": [uuid],
            "model": ["test"],
            "dim": [dim],
            "embedding": [[0.1] * dim],
            "embedded_at": [datetime.now(UTC)],
        },
        schema={
            "uuid": pl.Utf8,
            "model": pl.Utf8,
            "dim": pl.Int32,
            "embedding": pl.Array(pl.Float32, dim),
            "embedded_at": pl.Datetime("us", "UTC"),
        },
    )
    lance_store.add_chunk(tbl, df)


def test_has_table_returns_true_when_present(tmp_path: Path) -> None:
    """After ``open_or_create_table`` runs, ``_has_table`` finds it."""
    lance_uri = tmp_path / "lance"
    db = lance_store.connect_db(lance_uri)
    lance_store.open_or_create_table(db, dim=4)
    assert lance_store._has_table(db, lance_store.TABLE_NAME) is True


def test_has_table_returns_false_when_dataset_empty(tmp_path: Path) -> None:
    """A ``connect_db`` call against a fresh dir leaves the namespace empty."""
    lance_uri = tmp_path / "lance"
    db = lance_store.connect_db(lance_uri)
    # No create_table — the directory exists but the namespace has no tables.
    assert lance_store._has_table(db, lance_store.TABLE_NAME) is False


def test_register_vss_empty_namespace_creates_fallback_table(tmp_path: Path) -> None:
    """Empty Lance dir: register_vss creates an empty ``message_embeddings`` and returns False.

    This is the regression test for the bug we just fixed: the previous gate
    (``not lance_uri.exists() or not any(lance_uri.iterdir())``) classified a
    directory with metadata but no embeddings table as "non-empty" and tried
    to ATTACH + bind a view, which then exploded at view-bind time. The new
    gate probes the namespace via ``_has_table`` instead.
    """
    lance_uri = tmp_path / "lance_empty_namespace"
    # Touch the directory + a marker file so the old filesystem heuristic would
    # classify it as "non-empty" — proving the new gate is namespace-based.
    lance_uri.mkdir(parents=True, exist_ok=True)
    (lance_uri / "stray.txt").write_text("not a lance manifest")

    con = duckdb.connect(":memory:")
    ok = register_vss(con, lance_uri=lance_uri, dim=4)
    assert ok is False

    rows = con.execute("SELECT count(*) FROM message_embeddings").fetchone()
    assert rows is not None
    assert rows[0] == 0

    # The semantic_search macro depends on message_embeddings binding cleanly;
    # bind it directly here (the rest of register_macros wants v1 views, which
    # this lance-isolated test deliberately doesn't register).
    con.execute(
        """
        CREATE OR REPLACE MACRO semantic_search(query_vec, k) AS TABLE (
            SELECT me.uuid,
                   array_cosine_similarity(me.embedding, query_vec) AS sim,
                   array_distance(me.embedding, query_vec)          AS distance
            FROM message_embeddings me
            ORDER BY array_distance(me.embedding, query_vec)
            LIMIT k
        );
        """
    )
    rows = con.execute(
        "SELECT * FROM semantic_search([0.1, 0.1, 0.1, 0.1]::FLOAT[4], 5)"
    ).fetchall()
    assert rows == []  # empty fallback returns no results, doesn't error


def test_register_vss_attaches_real_dataset(tmp_path: Path) -> None:
    """A seeded Lance dataset: ATTACH + view bind + ``semantic_search`` macro all work end-to-end."""
    lance_uri = tmp_path / "lance_seeded"
    _seed_one_row(lance_uri, "abc-123", dim=4)
    _seed_one_row(lance_uri, "def-456", dim=4)
    _seed_one_row(lance_uri, "ghi-789", dim=4)

    con = duckdb.connect(":memory:")
    ok = register_vss(con, lance_uri=lance_uri, dim=4)
    assert ok is True

    count = con.execute("SELECT count(*) FROM message_embeddings").fetchone()
    assert count is not None
    assert count[0] == 3

    # Bind semantic_search directly (full register_macros wants v1 views).
    con.execute(
        """
        CREATE OR REPLACE MACRO semantic_search(query_vec, k) AS TABLE (
            SELECT me.uuid,
                   array_cosine_similarity(me.embedding, query_vec) AS sim,
                   array_distance(me.embedding, query_vec)          AS distance
            FROM message_embeddings me
            ORDER BY array_distance(me.embedding, query_vec)
            LIMIT k
        );
        """
    )
    rows = con.execute(
        "SELECT * FROM semantic_search([0.1, 0.1, 0.1, 0.1]::FLOAT[4], 2)"
    ).fetchall()
    assert len(rows) == 2

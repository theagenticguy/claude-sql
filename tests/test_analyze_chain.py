"""Regression-pin tests for the ``analyze`` chain stale-connection bug.

RFC §9.6 documents a bug where the shared DuckDB connection used by
``analyze`` binds ``message_embeddings`` against the (possibly empty)
LanceDB namespace at connection-open time. After the ``embed`` stage
populates Lance, the previously-bound view still points at the empty
namespace -- ``community`` (and any later stage that reads vectors)
silently sees zero rows.

The fix in :func:`claude_sql.cli.analyze` re-calls
:func:`claude_sql.sql_views.register_vss` after embed and again before
community, with a defensive :func:`claude_sql.cli._refresh_analytics_views`
after each LLM stage.

This module pins the underlying invariant so a future refactor can't
quietly drop the rebind:

* The LanceDB namespace can grow rows BETWEEN two
  ``register_vss`` calls on the SAME DuckDB connection, and only the
  second call surfaces them through the ``message_embeddings`` view.

It does NOT exercise the full ``analyze`` chain end-to-end -- that
would either hit Bedrock (forbidden by CLAUDE.md) or require mocking
seven workers. The unit-level regression test is enough; the call-site
discipline lives in :func:`claude_sql.cli._rebind_vss` and is reviewed
through the explicit ``stage=`` kwarg at the two call sites.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import polars as pl

from claude_sql import lance_store
from claude_sql.cli import _rebind_vss
from claude_sql.config import Settings
from claude_sql.sql_views import register_vss


def _seed_lance_row(lance_uri: Path, uuid: str, *, dim: int = 4) -> None:
    """Append one row to the Lance embeddings table at ``lance_uri``."""
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


def _make_settings(tmp_path: Path, *, dim: int = 256) -> Settings:
    """Tiny on-disk Settings for the analyze rebind tests."""
    cache = tmp_path / "claude"
    cache.mkdir(parents=True, exist_ok=True)
    return Settings(
        embeddings_parquet_path=cache / "embeddings",
        lance_uri=cache / "embeddings_lance",
        classifications_parquet_path=cache / "classifications",
        trajectory_parquet_path=cache / "trajectory",
        conflicts_parquet_path=cache / "conflicts",
        clusters_parquet_path=cache / "clusters.parquet",
        cluster_terms_parquet_path=cache / "cluster_terms.parquet",
        communities_parquet_path=cache / "communities.parquet",
        user_friction_parquet_path=cache / "user_friction",
        skills_catalog_parquet_path=cache / "skills_catalog.parquet",
        checkpoint_db_path=cache / "state.db",
        duckdb_temp_dir=cache / "duckdb_tmp",
        user_skills_dir=cache / "skills",
        plugins_cache_dir=cache / "plugins_cache",
        output_dimension=dim,
        embed_concurrency=2,
        llm_concurrency=2,
        batch_size=4,
    )


def test_register_vss_re_bind_after_lance_write(tmp_path: Path) -> None:
    """Pin the analyze-chain re-bind invariant: writes to Lance after the
    initial ``register_vss`` are invisible until ``_rebind_vss`` runs.

    1. Empty Lance namespace -> register_vss creates an empty fallback
       table; ``message_embeddings`` reports 0 rows.
    2. Append vectors directly to the Lance table from a side channel
       (mimicking what ``embed_worker`` does inside ``analyze``).
    3. Without a re-bind, ``message_embeddings`` still reports 0 rows
       on the SAME connection -- this is the bug.
    4. Call :func:`claude_sql.cli._rebind_vss` -- now the row count
       matches the Lance table.

    Step 4 calls the helper rather than ``register_vss`` directly because
    the helper carries the load-bearing DROP+DETACH needed to switch the
    fallback ``message_embeddings`` table over to the populated-Lance
    view shape (see :func:`_rebind_vss` docstring).
    """
    settings = _make_settings(tmp_path, dim=256)
    dim = settings.output_dimension
    lance_uri = settings.lance_uri

    con = duckdb.connect(":memory:")

    # Step 1: empty namespace -> fallback table, 0 rows.
    ok = register_vss(con, lance_uri=lance_uri, dim=dim)
    assert ok is False
    initial = con.execute("SELECT count(*) FROM message_embeddings").fetchone()
    assert initial is not None
    assert initial[0] == 0

    # Step 2: append three vectors to Lance through a separate code path,
    # the way embed_worker does inside ``analyze``.
    _seed_lance_row(lance_uri, "uuid-1", dim=dim)
    _seed_lance_row(lance_uri, "uuid-2", dim=dim)
    _seed_lance_row(lance_uri, "uuid-3", dim=dim)
    assert lance_store.count_rows(lance_uri) == 3

    # Step 3: WITHOUT a re-bind, the view still reports 0 rows. This is
    # the stale-connection symptom RFC §9.6 names.
    stale = con.execute("SELECT count(*) FROM message_embeddings").fetchone()
    assert stale is not None
    assert stale[0] == 0, (
        "Expected the pre-rebind view to be stale (0 rows). "
        "If this assertion fails, DuckDB+Lance has started auto-refreshing "
        "the ATTACH'd namespace and the analyze re-bind discipline can be "
        "relaxed -- audit cli.analyze before deleting anything."
    )

    # Step 4: re-bind via the helper. Now the view sees the rows.
    _rebind_vss(con, settings, stage="test-after-embed")
    refreshed = con.execute("SELECT count(*) FROM message_embeddings").fetchone()
    assert refreshed is not None
    assert refreshed[0] == 3
    assert refreshed[0] > stale[0]


def test_rebind_vss_helper_uses_settings(tmp_path: Path) -> None:
    """Pin :func:`claude_sql.cli._rebind_vss` against its Settings contract.

    The helper is the call site that ``analyze`` uses after embed and
    between cluster_terms and community. It must read ``lance_uri``,
    ``embeddings_parquet_path``, ``output_dimension``, and ``hnsw_metric``
    from the passed Settings -- if any of those wires get cut the
    analyze chain regresses to the stale-view bug silently.
    """
    settings = _make_settings(tmp_path, dim=256)

    con = duckdb.connect(":memory:")
    # Initial bind: empty namespace -> fallback empty table.
    _rebind_vss(con, settings, stage="test-init")
    initial = con.execute("SELECT count(*) FROM message_embeddings").fetchone()
    assert initial is not None
    assert initial[0] == 0

    # Seed Lance through the side channel -- helper has to surface it on rebind.
    _seed_lance_row(settings.lance_uri, "rebind-uuid-1", dim=settings.output_dimension)
    _seed_lance_row(settings.lance_uri, "rebind-uuid-2", dim=settings.output_dimension)

    _rebind_vss(con, settings, stage="test-after-embed")
    after = con.execute("SELECT count(*) FROM message_embeddings").fetchone()
    assert after is not None
    assert after[0] == 2


def test_rebind_vss_idempotent_when_already_view(tmp_path: Path) -> None:
    """Re-binding when the prior shape is already a VIEW must not crash.

    After embed populates Lance, the first rebind switches table->view.
    A second rebind (e.g. between cluster_terms and community in the
    analyze chain) must not blow up on the alread-view shape: the
    ``DROP VIEW IF EXISTS + CREATE OR REPLACE VIEW`` round trip is the
    idempotency guarantee.
    """
    settings = _make_settings(tmp_path, dim=256)
    _seed_lance_row(settings.lance_uri, "always-here-1", dim=settings.output_dimension)

    con = duckdb.connect(":memory:")
    _rebind_vss(con, settings, stage="first")
    one = con.execute("SELECT count(*) FROM message_embeddings").fetchone()
    assert one is not None
    assert one[0] == 1

    # Second rebind on a view -- must not crash, must still see the row.
    _rebind_vss(con, settings, stage="second")
    two = con.execute("SELECT count(*) FROM message_embeddings").fetchone()
    assert two is not None
    assert two[0] == 1

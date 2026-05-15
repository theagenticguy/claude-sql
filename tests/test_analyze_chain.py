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
import pytest

from claude_sql.app.cli import _rebind_vss
from claude_sql.core import lance_store
from claude_sql.core.config import Settings
from claude_sql.core.sql_views import register_vss


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


def test_analyze_chain_stage_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the analyze chain stage ordering: ingest BEFORE embed, embed
    BEFORE cluster, cluster BEFORE community, plus _rebind_vss called AFTER
    embed AND between terms/community.

    v1.0 added a new ``ingest`` stage at position 1 (RFC §10 — windowed
    pipelines). The contract is:

    * ``analyze/ingest`` runs first under the shared connection.
    * ``analyze/embed`` runs second; it MUST be able to read
      ``ingest_stamps.canonical_uuid`` to skip near-duplicates, so any
      future refactor that moves embed before ingest silently regresses
      embed-cost recovery.
    * ``_rebind_vss`` fires after embed (RFC §9.6) AND between
      cluster_terms and community.

    This test patches every worker entrypoint with a recorder and asserts
    the captured call sequence. It does NOT exercise real Bedrock or
    DuckDB I/O -- the workers are stub callables. The shared connection
    is a real ``:memory:`` DuckDB so ``_open_connection_full`` and
    ``_rebind_vss`` run unstubbed against a tiny on-disk Lance dataset.
    """
    import asyncio

    from claude_sql.app import cli as cli_mod

    settings = _make_settings(tmp_path, dim=256)
    # Seed Lance with one row so register_vss does the table->view switch
    # cleanly; the test only cares about call order, not content.
    _seed_lance_row(settings.lance_uri, "stage-order-uuid", dim=settings.output_dimension)

    calls: list[str] = []

    # Stub every worker.  Each one records its name and returns a
    # plausible value for the logger format strings the analyze body uses.
    def _fake_skills_sync(*_a: object, **_kw: object) -> dict[str, int]:
        calls.append("skills_sync")
        return {"rows": 0, "skills": 0, "commands": 0, "builtins": 0}

    def _fake_ingest_count(*_a: object, **_kw: object) -> int:
        calls.append("ingest_count")
        return 0

    def _fake_ingest_stamp(*_a: object, **_kw: object) -> int:
        calls.append("ingest_stamp")
        return 0

    def _fake_ingest_resolve(*_a: object, **_kw: object) -> int:
        calls.append("ingest_resolve")
        return 0

    async def _fake_embed(*_a: object, **_kw: object) -> dict[str, object]:
        calls.append("embed")
        return {"pipeline": "embed", "candidates": 0}

    def _fake_cluster(*_a: object, **_kw: object) -> dict[str, int]:
        calls.append("cluster")
        return {"total": 0, "clusters": 0, "noise": 0}

    def _fake_terms(*_a: object, **_kw: object) -> dict[str, int]:
        calls.append("terms")
        return {"clusters": 0, "terms": 0}

    def _fake_community(*_a: object, **_kw: object) -> dict[str, int]:
        calls.append("community")
        return {"sessions": 0, "communities": 0}

    def _fake_classify(*_a: object, **_kw: object) -> dict[str, object]:
        calls.append("classify")
        return {"pipeline": "classify"}

    def _fake_trajectory(*_a: object, **_kw: object) -> dict[str, object]:
        calls.append("trajectory")
        return {"pipeline": "trajectory"}

    def _fake_conflicts(*_a: object, **_kw: object) -> dict[str, object]:
        calls.append("conflicts")
        return {"pipeline": "conflicts"}

    def _fake_friction(*_a: object, **_kw: object) -> dict[str, object]:
        calls.append("friction")
        return {"pipeline": "friction"}

    def _fake_rebind(*_a: object, stage: str = "", **_kw: object) -> None:
        calls.append(f"rebind:{stage}")

    def _fake_resolve_settings(*_a: object, **_kw: object) -> Settings:
        return settings

    def _fake_open_connection(*_a: object, **_kw: object) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(":memory:")

    # Monkeypatch every worker entrypoint at the cli module's import site
    # (where the analyze function reads them).
    monkeypatch.setattr(cli_mod._skills_catalog, "sync", _fake_skills_sync)
    monkeypatch.setattr(cli_mod, "_ingest_count_pending", _fake_ingest_count)
    monkeypatch.setattr(cli_mod, "_ingest_stamp_messages", _fake_ingest_stamp)
    monkeypatch.setattr(cli_mod, "_ingest_resolve_canonicals", _fake_ingest_resolve)
    monkeypatch.setattr(cli_mod, "run_backfill", _fake_embed)
    monkeypatch.setattr(cli_mod, "run_clustering", _fake_cluster)
    monkeypatch.setattr(cli_mod, "run_terms", _fake_terms)
    monkeypatch.setattr(cli_mod, "run_communities", _fake_community)
    monkeypatch.setattr(cli_mod, "classify_sessions", _fake_classify)
    monkeypatch.setattr(cli_mod, "trajectory_messages", _fake_trajectory)
    monkeypatch.setattr(cli_mod, "detect_conflicts", _fake_conflicts)
    monkeypatch.setattr(cli_mod, "detect_user_friction", _fake_friction)
    monkeypatch.setattr(cli_mod, "_rebind_vss", _fake_rebind)
    monkeypatch.setattr(cli_mod, "_resolve_settings", _fake_resolve_settings)
    monkeypatch.setattr(cli_mod, "_open_connection_full", _fake_open_connection)
    monkeypatch.setattr(cli_mod, "_refresh_analytics_views", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli_mod, "_configure", lambda *_a, **_kw: None)

    # Wire asyncio.run to drive the embed coroutine without a real loop.
    monkeypatch.setattr(
        asyncio,
        "run",
        lambda coro: asyncio.new_event_loop().run_until_complete(coro),
    )

    # Run the chain WITH --no-dry-run so the ingest stamp/resolve path
    # exercises (dry-run only calls _ingest_count_pending).
    cli_mod.analyze(
        since_days=1,
        limit=None,
        dry_run=False,
        no_thinking=False,
        skip_ingest=False,
        skip_embed=False,
        skip_classify=False,
        skip_trajectory=False,
        skip_conflicts=False,
        skip_friction=False,
        skip_cluster=False,
        skip_community=False,
        skip_skills_sync=False,
        force_cluster=False,
        force_community=False,
        common=None,
    )

    # The chain ran end-to-end; capture the order.
    assert "skills_sync" in calls
    assert "ingest_stamp" in calls
    assert "embed" in calls
    assert "cluster" in calls
    assert "terms" in calls
    assert "community" in calls
    assert "classify" in calls
    assert "trajectory" in calls
    assert "conflicts" in calls
    assert "friction" in calls

    # Hard ordering invariants.
    def idx(name: str) -> int:
        return calls.index(name)

    # ingest BEFORE embed -- the v1.0 contract that lets embed read
    # canonical_uuid and skip near-duplicates.
    assert idx("ingest_stamp") < idx("embed"), (
        f"ingest must run before embed in the analyze chain; got order: {calls}"
    )
    assert idx("ingest_resolve") < idx("embed"), (
        f"ingest_resolve must run before embed; got order: {calls}"
    )

    # embed BEFORE cluster (cluster reads embeddings).
    assert idx("embed") < idx("cluster")
    # cluster BEFORE terms (terms reads cluster IDs).
    assert idx("cluster") < idx("terms")
    # terms BEFORE community (community reads cluster terms via the macro).
    assert idx("terms") < idx("community")
    # community BEFORE the LLM stages.
    assert idx("community") < idx("classify")
    # LLM order: classify -> trajectory -> conflicts -> friction.
    assert idx("classify") < idx("trajectory")
    assert idx("trajectory") < idx("conflicts")
    assert idx("conflicts") < idx("friction")

    # _rebind_vss must fire AFTER embed and AGAIN between terms and community.
    rebind_after_embed = [c for c in calls if c == "rebind:embed"]
    rebind_between = [c for c in calls if c == "rebind:cluster_terms"]
    assert rebind_after_embed, "expected _rebind_vss(stage='embed') after embed"
    assert rebind_between, "expected _rebind_vss(stage='cluster_terms') between terms and community"
    # Position checks: the embed-rebind sits AFTER embed and BEFORE cluster;
    # the cluster_terms-rebind sits AFTER terms and BEFORE community.
    assert calls.index("rebind:embed") > idx("embed")
    assert calls.index("rebind:embed") < idx("cluster")
    assert calls.index("rebind:cluster_terms") > idx("terms")
    assert calls.index("rebind:cluster_terms") < idx("community")

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

from claude_sql.infrastructure import lance_store
from claude_sql.infrastructure.duckdb_views import register_vss
from claude_sql.infrastructure.settings import Settings
from claude_sql.interfaces.cli.app import _rebind_vss


def _seed_lance_row(lance_uri: Path, uuid: str, *, dim: int = 4) -> None:
    """Append one row to the Lance embeddings table at ``lance_uri``.

    Stamps the default (cohere-bedrock) provider's ``model_id`` so the
    fail-loud provider guard on ``_rebind_vss`` / ``register_vss`` passes: these
    tests exercise the stale-connection rebind, not a provider switch.
    """
    db = lance_store.connect_db(lance_uri)
    tbl = lance_store.open_or_create_table(db, dim=dim)
    df = pl.DataFrame(
        {
            "uuid": [uuid],
            "model": ["global.cohere.embed-v4:0"],
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

    from claude_sql.interfaces.cli import app as cli_mod

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

    # Monkeypatch every worker entrypoint at its SOURCE module. ``analyze``
    # defers these imports into its body (``from claude_sql.application.use_cases.X
    # import name``), so the import statement re-reads ``X.name`` each call — patching
    # the source attribute is picked up, whereas patching ``cli_mod.name`` would
    # not (the name is no longer a cli module attribute). See the module-top
    # NOTE in cli.py and test_pr3_perf.py::test_cli_import_is_lean.
    monkeypatch.setattr("claude_sql.application.use_cases.skills.sync", _fake_skills_sync)
    monkeypatch.setattr("claude_sql.application.use_cases.ingest.count_pending", _fake_ingest_count)
    monkeypatch.setattr(
        "claude_sql.application.use_cases.ingest.stamp_messages", _fake_ingest_stamp
    )
    monkeypatch.setattr(
        "claude_sql.application.use_cases.ingest.resolve_canonicals", _fake_ingest_resolve
    )
    monkeypatch.setattr("claude_sql.application.use_cases.embed.run_backfill", _fake_embed)
    monkeypatch.setattr("claude_sql.application.use_cases.cluster.run_clustering", _fake_cluster)
    monkeypatch.setattr("claude_sql.application.use_cases.terms.run_terms", _fake_terms)
    monkeypatch.setattr(
        "claude_sql.application.use_cases.community.run_communities", _fake_community
    )
    monkeypatch.setattr(
        "claude_sql.application.use_cases.classify.classify_sessions", _fake_classify
    )
    monkeypatch.setattr(
        "claude_sql.application.use_cases.trajectory.trajectory_messages", _fake_trajectory
    )
    monkeypatch.setattr(
        "claude_sql.application.use_cases.conflicts.detect_conflicts", _fake_conflicts
    )
    monkeypatch.setattr(
        "claude_sql.application.use_cases.friction.detect_user_friction", _fake_friction
    )
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


def _install_analyze_recorders(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, calls: list[str]
) -> None:
    """Patch every ``analyze`` seam with an ordered recorder.

    Shared by the double-refresh and vss-first-ordering tests below. Records
    ingest stamp/resolve, embed, the cluster/term/community/LLM stages, plus
    the two lifecycle seams ``_refresh_analytics_views`` (as ``"refresh"``)
    and ``_rebind_vss`` (as ``"rebind:<stage>"``) into ``calls`` in the exact
    order the ``analyze`` body invokes them. Everything is a stub; no Bedrock,
    no real DuckDB registration (the shared connection is a bare ``:memory:``).
    """
    import asyncio

    from claude_sql.interfaces.cli import app as cli_mod

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

    def _fake_refresh(*_a: object, **_kw: object) -> None:
        calls.append("refresh")

    def _fake_rebind(*_a: object, stage: str = "", **_kw: object) -> None:
        calls.append(f"rebind:{stage}")

    def _fake_resolve_settings(*_a: object, **_kw: object) -> Settings:
        return settings

    def _fake_open_connection(*_a: object, **_kw: object) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(":memory:")

    monkeypatch.setattr("claude_sql.application.use_cases.skills.sync", _fake_skills_sync)
    monkeypatch.setattr("claude_sql.application.use_cases.ingest.count_pending", _fake_ingest_count)
    monkeypatch.setattr(
        "claude_sql.application.use_cases.ingest.stamp_messages", _fake_ingest_stamp
    )
    monkeypatch.setattr(
        "claude_sql.application.use_cases.ingest.resolve_canonicals", _fake_ingest_resolve
    )
    monkeypatch.setattr("claude_sql.application.use_cases.embed.run_backfill", _fake_embed)
    monkeypatch.setattr("claude_sql.application.use_cases.cluster.run_clustering", _fake_cluster)
    monkeypatch.setattr("claude_sql.application.use_cases.terms.run_terms", _fake_terms)
    monkeypatch.setattr(
        "claude_sql.application.use_cases.community.run_communities", _fake_community
    )
    monkeypatch.setattr(
        "claude_sql.application.use_cases.classify.classify_sessions", _fake_classify
    )
    monkeypatch.setattr(
        "claude_sql.application.use_cases.trajectory.trajectory_messages", _fake_trajectory
    )
    monkeypatch.setattr(
        "claude_sql.application.use_cases.conflicts.detect_conflicts", _fake_conflicts
    )
    monkeypatch.setattr(
        "claude_sql.application.use_cases.friction.detect_user_friction", _fake_friction
    )
    monkeypatch.setattr(cli_mod, "_rebind_vss", _fake_rebind)
    monkeypatch.setattr(cli_mod, "_refresh_analytics_views", _fake_refresh)
    monkeypatch.setattr(cli_mod, "_resolve_settings", _fake_resolve_settings)
    monkeypatch.setattr(cli_mod, "_open_connection_full", _fake_open_connection)
    monkeypatch.setattr(cli_mod, "_configure", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        asyncio,
        "run",
        lambda coro: asyncio.new_event_loop().run_until_complete(coro),
    )


def test_analyze_double_refresh_around_ingest_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin the double-refresh bracketing ``ingest`` resolve (cli.py:2472+2475).

    Under ``--no-dry-run`` the ingest stage stamps SimHash shards, then runs
    the canonical-resolution pass. Both writes land as fresh parquet shards
    that the shared connection's frozen ``read_parquet([...])`` view lists
    do NOT yet see. The body must therefore:

    * ``_refresh_analytics_views`` AFTER ``stamp_messages`` and BEFORE
      ``resolve_canonicals`` -- so resolve reads the freshly-stamped rows
      through the re-bound ``ingest_stamps`` view (a stamp shard is invisible
      without the re-register).
    * ``_refresh_analytics_views`` AGAIN after ``resolve_canonicals`` -- so
      the downstream ``embed`` stage reads the resolved ``canonical_uuid``.

    Drop either refresh and ingest-driven embed-cost recovery silently
    regresses. This asserts the exact ``stamp -> refresh -> resolve ->
    refresh`` sub-sequence.
    """
    from claude_sql.interfaces.cli import app as cli_mod

    settings = _make_settings(tmp_path, dim=256)
    _seed_lance_row(settings.lance_uri, "double-refresh-uuid", dim=settings.output_dimension)

    calls: list[str] = []
    _install_analyze_recorders(monkeypatch, settings, calls)

    # Only the ingest stage matters here; skip the rest to keep the recorded
    # stream tight. ``--no-dry-run`` is what routes through stamp+resolve.
    cli_mod.analyze(
        since_days=1,
        limit=None,
        dry_run=False,
        no_thinking=False,
        skip_ingest=False,
        skip_embed=True,
        skip_classify=True,
        skip_trajectory=True,
        skip_conflicts=True,
        skip_friction=True,
        skip_cluster=True,
        skip_community=True,
        skip_skills_sync=True,
        force_cluster=False,
        force_community=False,
        common=None,
    )

    assert "ingest_stamp" in calls, f"ingest stamp did not run: {calls}"
    assert "ingest_resolve" in calls, f"ingest resolve did not run: {calls}"

    # The exact bracket: stamp, refresh, resolve, refresh. Slice the stream to
    # the ingest window and assert the ordered shape.
    stamp_i = calls.index("ingest_stamp")
    resolve_i = calls.index("ingest_resolve")
    assert resolve_i > stamp_i, f"resolve must follow stamp: {calls}"

    # A refresh must sit strictly between stamp and resolve (the stamped shard
    # is invisible to resolve otherwise).
    between = [c for c in calls[stamp_i + 1 : resolve_i] if c == "refresh"]
    assert between, (
        "expected _refresh_analytics_views between ingest stamp and resolve "
        f"(cli.py:2472 semantics -- stamped shard invisible without re-register); got: {calls}"
    )

    # A refresh must also sit after resolve (before embed would read canonical_uuid).
    after = [c for c in calls[resolve_i + 1 :] if c == "refresh"]
    assert after, (
        "expected _refresh_analytics_views after ingest resolve "
        f"(cli.py:2475 semantics -- resolved canonical_uuid must be visible downstream); got: {calls}"
    )


def test_analyze_rebind_before_refresh_after_embed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin the VSS-first-then-analytics ordering after embed (cli.py:2494->2495).

    After ``embed`` populates the Lance namespace the body must re-bind VSS
    FIRST (``_rebind_vss(stage="embed")``) so ``message_embeddings`` surfaces
    the fresh vectors, THEN refresh analytics so the parquet-existence gates
    re-evaluate against the freshly written shards (RFC §9.6). If the two ever
    swap, the analytics refresh re-binds views that still read the pre-embed
    Lance state and the downstream cluster/community stages see an empty
    ``message_embeddings``.

    Asserts the ordered pair: ``embed -> rebind:embed -> refresh`` with no
    intervening ``refresh`` between embed and its rebind.
    """
    from claude_sql.interfaces.cli import app as cli_mod

    settings = _make_settings(tmp_path, dim=256)
    _seed_lance_row(settings.lance_uri, "vss-first-uuid", dim=settings.output_dimension)

    calls: list[str] = []
    _install_analyze_recorders(monkeypatch, settings, calls)

    # Run ingest + embed only; the rest is noise for this ordering claim.
    cli_mod.analyze(
        since_days=1,
        limit=None,
        dry_run=False,
        no_thinking=False,
        skip_ingest=True,
        skip_embed=False,
        skip_classify=True,
        skip_trajectory=True,
        skip_conflicts=True,
        skip_friction=True,
        skip_cluster=True,
        skip_community=True,
        skip_skills_sync=True,
        force_cluster=False,
        force_community=False,
        common=None,
    )

    assert "embed" in calls, f"embed did not run: {calls}"
    assert "rebind:embed" in calls, f"expected rebind after embed: {calls}"

    embed_i = calls.index("embed")
    rebind_i = calls.index("rebind:embed")
    # The embed rebind fires after embed.
    assert rebind_i > embed_i, f"rebind:embed must follow embed: {calls}"
    # The first refresh AFTER embed must come AFTER the rebind -- VSS first.
    refresh_after_embed = [i for i, c in enumerate(calls) if c == "refresh" and i > embed_i]
    assert refresh_after_embed, f"expected a refresh after embed: {calls}"
    assert refresh_after_embed[0] > rebind_i, (
        "VSS re-bind must precede the analytics refresh after embed "
        f"(cli.py:2494 before :2495 -- RFC §9.6); got: {calls}"
    )
    # No refresh may sneak between embed and its rebind.
    assert not [c for c in calls[embed_i + 1 : rebind_i] if c == "refresh"], (
        f"no analytics refresh may run between embed and its VSS rebind; got: {calls}"
    )


def test_refresh_analytics_views_surfaces_new_shard(tmp_path: Path) -> None:
    """Stale-read regression: a parquet shard written mid-run is invisible to
    the shared connection until ``_refresh_analytics_views`` re-binds.

    This is the parquet-side analog of the Lance stale-view bug the module's
    other tests pin. The analytics views bind ``read_parquet([<part files>])``
    with the path list captured and FROZEN at registration time. A new shard
    dropped into the cache directory afterwards is NOT in the frozen list, so
    a query through the un-refreshed connection does not see it. The
    ``analyze`` body's ``_refresh_analytics_views`` calls after every
    shard-producing stage are what re-freeze the list against the current
    directory contents.

    1. Write shard #1 into the classifications cache, register analytics ->
       the view reports 1 row.
    2. Write shard #2 directly to the directory (mimicking a mid-run worker
       write).
    3. WITHOUT a refresh, the view still reports 1 row -- the frozen path list
       does not include shard #2.
    4. Call ``_refresh_analytics_views`` -- now the view reports 2 rows.
    """
    from datetime import UTC, datetime

    from claude_sql.infrastructure.parquet_cache import write_part
    from claude_sql.interfaces.cli.app import _refresh_analytics_views

    settings = _make_settings(tmp_path, dim=256)
    cls_dir = settings.classifications_parquet_path
    cls_dir.mkdir(parents=True, exist_ok=True)

    def _row(session_id: str) -> pl.DataFrame:
        # Minimal shape; register_analytics falls back to SELECT * when the
        # curated projection's aliased columns are absent, so count(*) is
        # schema-agnostic here.
        return pl.DataFrame(
            {
                "session_id": [session_id],
                "autonomy_tier": ["assisted"],
                "success": [True],
                "work_category": ["build"],
                "goal": ["ship"],
                "confidence": [0.9],
                "classified_at": [datetime.now(UTC)],
            }
        )

    # Step 1: shard #1 + register -> 1 row.
    write_part(cls_dir, _row("sess-1"))
    con = duckdb.connect(":memory:")
    _refresh_analytics_views(con, settings)
    one = con.execute("SELECT count(*) FROM session_classifications").fetchone()
    assert one is not None
    assert one[0] == 1

    # Step 2: shard #2 lands directly on disk (a mid-run worker write).
    write_part(cls_dir, _row("sess-2"))

    # Step 3: WITHOUT a refresh the frozen path list still points at shard #1 only.
    stale = con.execute("SELECT count(*) FROM session_classifications").fetchone()
    assert stale is not None
    assert stale[0] == 1, (
        "Expected the pre-refresh view to be stale (1 row). If this fails, "
        "DuckDB has started re-globbing the parquet directory on every scan "
        "and the analyze refresh discipline can be relaxed -- audit "
        "cli.analyze before deleting anything."
    )

    # Step 4: refresh re-freezes the path list against the current directory.
    _refresh_analytics_views(con, settings)
    refreshed = con.execute("SELECT count(*) FROM session_classifications").fetchone()
    assert refreshed is not None
    assert refreshed[0] == 2
    assert refreshed[0] > stale[0]
    con.close()

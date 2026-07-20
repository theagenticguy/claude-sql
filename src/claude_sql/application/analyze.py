"""The ``analyze`` orchestration, lifted out of the CLI.

This is the composite analytics pipeline (``embed → structure → LLM
analytics``) that ``claude-sql analyze`` drives. It was lifted verbatim from
``app.cli.analyze`` during the Wave-4 rebind cut so the highest-risk stretch of
the codebase -- the RFC §9.6 rebind lifecycle -- lives in one testable
application-layer entrypoint instead of inline in a CLI command body.

The EXACT stage order, the two-axis rebind (VSS re-bind then analytics
refresh), the double-refresh bracketing the ingest resolve, the dry-run
semantics, and the ``try/finally`` connection close are preserved byte-for-byte
from the CLI. The regression pins in ``tests/test_analyze_chain.py`` guard the
ordering.

Injection shape
---------------
:func:`run_analyze` takes the connection factory and the two lifecycle seams
(``refresh_fn`` / ``rebind_fn``) as injectable callables defaulting to the
:mod:`claude_sql.infrastructure.duckdb_connection` functions. The CLI passes
its OWN module attributes (``cli._open_connection_full`` /
``cli._refresh_analytics_views`` / ``cli._rebind_vss``) so the existing
module-object monkeypatches in the test suite (``monkeypatch.setattr(cli_mod,
"_rebind_vss", ...)``) keep biting through the injected references. The heavy
analytics worker imports are DEFERRED into the body -- mirroring the old CLI
command -- so importing this module never drags boto3/numpy onto the load path,
and so ``test_analyze_chain``'s source-level worker patches
(``claude_sql.application.use_cases.X.name``) are re-read on each call.

:class:`DuckDbReader` is the concrete :class:`~claude_sql.application.ports.ReaderPort`
implementation: it holds the shared connection + settings and exposes
``connection`` / ``query`` / ``refresh_analytics_views`` / ``rebind_vss`` with
the injected lifecycle functions bound in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from claude_sql.infrastructure.duckdb_connection import (
    open_connection_full as _default_open_connection,
    rebind_vss as _default_rebind,
    refresh_analytics_views as _default_refresh,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import duckdb
    import polars as pl

    from claude_sql.infrastructure.settings import Settings


class DuckDbReader:
    """Concrete :class:`~claude_sql.application.ports.ReaderPort`.

    Holds the shared, stateful DuckDB connection and the ``Settings`` that
    drove its registration, and exposes the register -> write -> rebind
    lifecycle explicitly (see the ``ReaderPort`` docstring / RFC §9.6). The two
    lifecycle seams are injectable so the CLI can thread its own module
    attributes through for monkeypatch compatibility; they default to the
    infrastructure implementations.
    """

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        settings: Settings,
        *,
        refresh_fn: Callable[[duckdb.DuckDBPyConnection, Settings], None] | None = None,
        rebind_fn: Callable[..., None] | None = None,
    ) -> None:
        self._con = con
        self._settings = settings
        self._refresh_fn = refresh_fn if refresh_fn is not None else _default_refresh
        self._rebind_fn = rebind_fn if rebind_fn is not None else _default_rebind

    def connection(self) -> duckdb.DuckDBPyConnection:
        """Return the underlying (shared, stateful) DuckDB connection handle."""
        return self._con

    def query(self, sql: str, params: list[Any] | None = None) -> pl.DataFrame:
        """Execute ``sql`` and return the result as a polars DataFrame."""
        if params is None:
            return self._con.execute(sql).pl()
        return self._con.execute(sql, params).pl()

    def refresh_analytics_views(self) -> None:
        """Re-bind the analytics views so mid-run parquet writes become visible."""
        self._refresh_fn(self._con, self._settings)

    def rebind_vss(self, stage: str) -> None:
        """Re-bind ``message_embeddings`` against the (mutated) Lance namespace."""
        self._rebind_fn(self._con, self._settings, stage=stage)


def run_analyze(
    settings: Settings,
    *,
    since_days: int | None = 30,
    limit: int | None = None,
    dry_run: bool = True,
    no_thinking: bool = False,
    skip_ingest: bool = False,
    skip_embed: bool = False,
    skip_classify: bool = False,
    skip_trajectory: bool = False,
    skip_conflicts: bool = False,
    skip_friction: bool = False,
    skip_cluster: bool = False,
    skip_community: bool = False,
    skip_skills_sync: bool = False,
    force_cluster: bool = False,
    force_community: bool = False,
    open_connection: Callable[..., duckdb.DuckDBPyConnection] | None = None,
    refresh_fn: Callable[[duckdb.DuckDBPyConnection, Settings], None] | None = None,
    rebind_fn: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Run the full analytics pipeline end-to-end: embed → structure → LLM analytics.

    Stages (in order)
    -----------------
    0. skills sync  (filesystem walk; zero-cost; produces skills_catalog.parquet)
    1. ingest       (tiktoken + blake2b SimHash; zero-cost; honors ``dry_run``)
    2. embed        (Bedrock Cohere Embed v4; honors ``dry_run``)
    3. cluster      (UMAP+HDBSCAN; zero-cost; ``force_cluster`` to rebuild)
    4. terms        (c-TF-IDF labels for clusters; zero-cost)
    5. community    (Leiden+CPM; zero-cost; ``force_community`` to rebuild)
    6. classify     (Sonnet 4.6; honors ``dry_run``)
    7. trajectory   (Sonnet 4.6; honors ``dry_run``)
    8. conflicts    (Sonnet 4.6; honors ``dry_run``)
    9. friction     (Sonnet 4.6; honors ``dry_run``)

    ``settings`` must already carry any provider overrides (the CLI applies
    ``--embedding-provider`` / ``--llm-analytics-provider`` before calling this).
    ``open_connection`` / ``refresh_fn`` / ``rebind_fn`` default to the
    infrastructure implementations; the CLI passes its own module attributes so
    module-object monkeypatches keep biting.

    Returns a per-stage summary dict.
    """
    import asyncio

    from claude_sql.application.use_cases import skills as _skills_catalog
    from claude_sql.application.use_cases.classify import classify_sessions
    from claude_sql.application.use_cases.cluster import run_clustering
    from claude_sql.application.use_cases.community import run_communities
    from claude_sql.application.use_cases.conflicts import detect_conflicts
    from claude_sql.application.use_cases.embed import run_backfill
    from claude_sql.application.use_cases.friction import detect_user_friction
    from claude_sql.application.use_cases.ingest import (
        count_pending as _ingest_count_pending,
        resolve_canonicals as _ingest_resolve_canonicals,
        stamp_messages as _ingest_stamp_messages,
    )
    from claude_sql.application.use_cases.terms import run_terms
    from claude_sql.application.use_cases.trajectory import trajectory_messages

    _open = open_connection if open_connection is not None else _default_open_connection

    summary: dict[str, Any] = {"dry_run": dry_run}

    # 0. Skills catalog sync (filesystem walk, zero cost).  Runs even in
    # dry-run because it does not hit Bedrock; opt out via skip_skills_sync
    # if you want to keep the parquet frozen.
    if not skip_skills_sync:
        stats = _skills_catalog.sync(settings)
        logger.info(
            "analyze/skills: wrote {} rows to {} ({} skills, {} commands, {} builtins)",
            stats["rows"],
            settings.skills_catalog_parquet_path,
            stats["skills"],
            stats["commands"],
            stats["builtins"],
        )
        summary["skills"] = stats

    # Open ONE shared DuckDB connection for every stage that needs the catalog.
    # ``run_clustering`` reads parquet directly and is the only stage that
    # doesn't take ``con``; everything else threads this single registered
    # connection through. Stages that produce new parquet shards mid-run
    # (embed -> embeddings/part-*.parquet; cluster -> clusters.parquet) call
    # :meth:`DuckDbReader.refresh_analytics_views` afterwards so downstream
    # stages see the freshly written data.
    con = _open(settings)
    reader = DuckDbReader(con, settings, refresh_fn=refresh_fn, rebind_fn=rebind_fn)
    try:
        # 1. Ingest stamps (zero-cost: tiktoken + blake2b SimHash).  Always
        # safe to run -- pure CPU, no Bedrock.  Honors dry_run to support
        # an agent's "preview before commit" pattern even though the cost is
        # zero.  Resolution pass is wired so embed (next) can read the
        # ``canonical_uuid`` column to skip near-dup messages.
        if not skip_ingest:
            if dry_run:
                n = _ingest_count_pending(con, settings, since_days=since_days, limit=limit)
                logger.info("analyze/ingest: {} candidate rows (dry_run=True)", n)
            else:
                stamped = _ingest_stamp_messages(con, settings, since_days=since_days, limit=limit)
                reader.refresh_analytics_views()
                resolved = _ingest_resolve_canonicals(con, settings)
                logger.info("analyze/ingest: stamped={} resolved={}", stamped, resolved)
                reader.refresh_analytics_views()

        # 2. Embed (reuses embed_worker).  Silently skipped if the parquet is up to date.
        if not skip_embed:
            n = asyncio.run(
                run_backfill(
                    con=con,
                    settings=settings,
                    since_days=since_days,
                    limit=limit,
                    dry_run=dry_run,
                )
            )
            logger.info("analyze/embed: {} new embeddings (dry_run={})", n, dry_run)
            # New embeddings landed in the Lance namespace; the
            # ``message_embeddings`` view was bound at connection-open against
            # the (possibly empty) prior state and is now stale. Re-bind VSS
            # FIRST so subsequent stages see the fresh vectors, then refresh
            # analytics so the parquet-existence gates re-evaluate (RFC §9.6).
            reader.rebind_vss("embed")
            reader.refresh_analytics_views()

        # 2. Cluster (reads embeddings parquet, writes clusters.parquet).  Non-LLM.
        if not skip_cluster:
            stats = run_clustering(settings, force=force_cluster)
            logger.info(
                "analyze/cluster: {} messages, {} clusters, {} noise",
                stats["total"],
                stats["clusters"],
                stats["noise"],
            )
            # clusters.parquet just got rewritten -- re-bind so terms/community
            # /classify/trajectory read the fresh cluster IDs.
            reader.refresh_analytics_views()
            tstats = run_terms(con, settings, force=force_cluster)
            logger.info(
                "analyze/terms: {} clusters, {} term-rows",
                tstats["clusters"],
                tstats["terms"],
            )

        # 3. Community detection (non-LLM, runs in parallel conceptually with cluster).
        # RFC §9.6: this is THE explicitly-named bug location -- community
        # reads message_embeddings (via the session-centroid build) and would
        # see an empty view if embed populated Lance after connection-open
        # without an intervening rebind. Re-bind VSS + refresh analytics so
        # both the Lance dataset and the cluster_terms parquet land cleanly.
        if not skip_community:
            reader.rebind_vss("cluster_terms")
            reader.refresh_analytics_views()
            cstats = run_communities(con, settings, force=force_community)
            logger.info(
                "analyze/community: {} sessions, {} communities",
                cstats["sessions"],
                cstats["communities"],
            )

        # 4. Session classification (LLM).
        if not skip_classify:
            n = classify_sessions(
                con,
                settings,
                since_days=since_days,
                limit=limit,
                dry_run=dry_run,
                no_thinking=no_thinking,
            )
            logger.info("analyze/classify: {} sessions (dry_run={})", n, dry_run)
            # Refresh so any subsequent dashboard query in the same session
            # sees the fresh classifications parquet shard.
            reader.refresh_analytics_views()

        # 5. Trajectory (LLM).
        if not skip_trajectory:
            n = trajectory_messages(
                con,
                settings,
                since_days=since_days,
                limit=limit,
                dry_run=dry_run,
                no_thinking=no_thinking,
            )
            logger.info("analyze/trajectory: {} messages (dry_run={})", n, dry_run)
            reader.refresh_analytics_views()

        # 6. Conflicts (LLM, requires full session context).
        if not skip_conflicts:
            n = detect_conflicts(
                con,
                settings,
                since_days=since_days,
                limit=limit,
                dry_run=dry_run,
                no_thinking=no_thinking,
            )
            logger.info("analyze/conflicts: {} sessions (dry_run={})", n, dry_run)
            reader.refresh_analytics_views()

        # 7. Friction (LLM, short-message scope).
        if not skip_friction:
            n = detect_user_friction(
                con,
                settings,
                since_days=since_days,
                limit=limit,
                dry_run=dry_run,
                no_thinking=no_thinking,
            )
            logger.info("analyze/friction: {} rows (dry_run={})", n, dry_run)
            reader.refresh_analytics_views()
    finally:
        con.close()

    logger.info("analyze: done")
    return summary

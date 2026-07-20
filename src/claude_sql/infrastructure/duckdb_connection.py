"""DuckDB connection lifecycle: open, tune, register, and re-bind.

This module owns the stateful DuckDB connection seam for the v2 hexagonal
reshape. It was lifted verbatim out of ``app.cli`` during the Wave-4 rebind
cut so that the ``ReaderPort`` implementation (``DuckDbReader`` in
``application.analyze``) and the CLI command bodies share ONE definition of
the register -> write -> rebind lifecycle.

The load-bearing subtlety this module exists to protect is RFC §9.6: the
analytics views bind ``read_parquet([<frozen path list>])`` and VSS binds
``message_embeddings`` against the Lance namespace state AT REGISTRATION TIME.
A stage that writes new parquet shards (``embed``, ``cluster``, ``ingest``) or
populates LanceDB mid-run does NOT see its own writes until the views are
re-bound. :func:`refresh_analytics_views` re-freezes the parquet path list;
:func:`rebind_vss` re-binds ``message_embeddings`` against the mutated Lance
namespace. Folding either into a naive ``query(sql)`` silently reintroduces
the ``analyze`` stale-connection bug (``community`` reads 0 embedding rows).

Every public function keeps an underscore-prefixed alias so the historic
``app.cli`` bindings and the test monkeypatches (``test_config`` imports
``_resolve_memory_limit``; ``test_cli`` uses ``_open_connection_full``;
``test_cli_coverage`` uses ``_maybe_migrate_legacy_caches``) resolve
unchanged. ``duckdb`` is imported eagerly here, mirroring ``app.cli`` (which
already imports it at module top) — this module is only reached from paths
that were going to touch DuckDB anyway, so it does not widen the CLI
cold-start import surface pinned by ``test_cli_import_is_lean``.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
from typing import TYPE_CHECKING

import duckdb
from loguru import logger

from claude_sql.infrastructure.duckdb_views import (
    MACRO_NAMES,
    VIEW_NAMES,
    register_all,
    register_analytics,
    register_vss,
)
from claude_sql.infrastructure.home import claude_sql_home, recognized_legacy_caches

if TYPE_CHECKING:
    from claude_sql.infrastructure.settings import Settings


_PERCENT_LIMIT_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%\s*$")


def resolve_memory_limit(limit: str) -> str:
    """Translate ``"<n>%"`` into an absolute size DuckDB accepts.

    DuckDB's ``memory_limit`` parser only knows ``KB / MB / GB / TB`` and the
    binary variants. Percentage strings are rejected, so we resolve them
    against the host's reported total memory before the PRAGMA fires. Any
    other form passes through unchanged so the env var can still pin an
    absolute size like ``"4GB"`` directly.
    """
    match = _PERCENT_LIMIT_RE.match(limit)
    if match is None:
        return limit.strip()
    fraction = float(match.group(1)) / 100.0
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        # Non-POSIX or restricted host — fall back to a conservative 4 GiB.
        total_bytes = 4 * 1024**3
    else:
        total_bytes = page_size * phys_pages
    target_mib = max(1, int((total_bytes * fraction) // (1024 * 1024)))
    return f"{target_mib}MiB"


_MIGRATION_MARKER = ".migration_complete"


def maybe_migrate_legacy_caches() -> None:
    """One-time move of recognized caches from ``~/.claude/`` → ``CLAUDE_SQL_HOME``.

    Runs at most once per ``CLAUDE_SQL_HOME`` directory: a sentinel file
    (``.migration_complete``) in the home short-circuits subsequent calls.
    Every move is wrapped in ``shutil.move`` and the whole loop is guarded
    against ``OSError`` so a hostile filesystem (read-only mount, EACCES
    on a single subtree) can't crash startup — we log a warning and the
    user can rerun ``claude-sql`` later or migrate manually.

    Idempotent. Safe to call from any subcommand entry point. The first
    successful run stamps the marker; later runs are no-ops.
    """
    home = claude_sql_home()
    marker = home / _MIGRATION_MARKER
    if marker.exists():
        return
    legacy = recognized_legacy_caches()
    # When the new home already holds caches (e.g. a fresh install on a
    # box that never had the old layout) we still want to stamp the
    # marker so we don't keep probing legacy on every invocation.
    if not legacy:
        try:
            marker.touch()
        except OSError as exc:
            logger.warning("could not stamp migration marker at {}: {}", marker, exc)
        return
    try:
        names = ", ".join(sorted(legacy))
        logger.info("migrating claude-sql caches: ~/.claude/{{{}}} -> {}/", names, home)
        for name, src in legacy.items():
            dst = home / name
            if dst.exists():
                # Already populated under the new home — leave the legacy
                # copy in place rather than overwriting; the user can
                # reconcile manually if they care.
                logger.warning("skipping migrate {}: destination {} already exists", src, dst)
                continue
            shutil.move(str(src), str(dst))
        marker.touch()
    except OSError as exc:
        # Defensive: any IO error here aborts the migration but does NOT
        # crash startup. The next invocation will retry from a clean
        # state (marker not stamped → loop runs again).
        logger.warning("legacy cache migration skipped: {}", exc)


def apply_duckdb_pragmas(con: duckdb.DuckDBPyConnection, settings: Settings) -> None:
    """Set the tuning PRAGMAs both connection helpers share.

    Centralized so :func:`open_connection_full` and
    :func:`open_connection_introspect` stay in sync. Threads, memory_limit,
    and temp_directory all come from ``settings``; the spill directory is
    materialized on disk before DuckDB sees the path because DuckDB will
    happily fail later when it tries to write a spill file.
    """
    settings.duckdb_temp_dir.mkdir(parents=True, exist_ok=True)
    memory_limit = resolve_memory_limit(settings.duckdb_memory_limit)
    con.execute(f"SET threads = {int(settings.duckdb_threads)}")
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET temp_directory = '{settings.duckdb_temp_dir}'")
    con.execute("SET enable_object_cache = true")
    con.execute("SET preserve_insertion_order = false")


def open_connection_full(settings: Settings, *, sql: str = "") -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with every claude-sql object wired.

    Tuning PRAGMAs are set before view registration so the registration
    queries themselves benefit from the higher thread count and the spill
    directory pointed at real disk (Amazon devboxes ship ``/tmp`` as a
    4 GB tmpfs that thrashes once a clustering run starts spilling).

    When ``sql`` is provided we substring-scan it via :func:`sql_uses_vss`
    and skip ``register_vss`` (plus the ``semantic_search`` macro) when no
    vector tokens appear -- the common ``query``/``explain`` path then
    avoids the ``INSTALL vss; LOAD vss; ATTACH hnsw.duckdb`` round-trips
    that otherwise dominate a cold start. The default ``sql=""`` keeps the
    legacy behavior (register everything) for callers that don't know what
    SQL is coming -- ``shell``, ``analyze`` stages, and the ``search``
    command which builds its own VSS-bearing SQL after the connection is
    already open.
    """
    maybe_migrate_legacy_caches()
    con = duckdb.connect(":memory:")
    apply_duckdb_pragmas(con, settings)
    skip_vss = bool(sql) and not sql_uses_vss(sql)
    register_all(con, settings=settings, skip_vss=skip_vss)
    return con


def open_connection_introspect(settings: Settings) -> duckdb.DuckDBPyConnection:
    """Bare DuckDB connection — PRAGMAs only, no view/macro registration.

    For commands that don't need the catalog (``schema`` reads the static
    :data:`VIEW_SCHEMA` dict; trivial scalar queries like ``SELECT 1`` or
    ``SELECT current_timestamp`` don't reference any view). Returning a
    bare connection avoids the ~25 s :func:`register_all` chain entirely.
    """
    maybe_migrate_legacy_caches()
    con = duckdb.connect(":memory:")
    apply_duckdb_pragmas(con, settings)
    return con


def refresh_analytics_views(con: duckdb.DuckDBPyConnection, settings: Settings) -> None:
    """Re-register analytics views after a stage produced new parquet shards.

    The analytics views (``message_clusters``, ``message_embeddings``, etc.)
    are bound at :func:`register_all` time as ``read_parquet([list of
    paths])`` -- the path list is captured then and frozen. New shards
    written mid-run by ``embed`` (new ``embeddings/part-*.parquet``) or
    ``cluster`` (a refreshed ``clusters.parquet``) are invisible until the
    views are re-bound, which is what this helper does.

    Pure DDL: re-runs :func:`register_analytics` against the same connection.
    No ``v_raw_events`` rebuild, no VSS rebuild -- cheap relative to a fresh
    :func:`open_connection_full`, which is the whole point of the T1.3
    shared-connection refactor.
    """
    register_analytics(con, settings=settings)


def rebind_vss(con: duckdb.DuckDBPyConnection, settings: Settings, *, stage: str) -> None:
    """Re-bind ``message_embeddings`` against the (possibly mutated) Lance namespace.

    The ``analyze`` chain shares one DuckDB connection across stages (T1.3).
    ``register_vss`` runs once at connection-open time and binds
    ``message_embeddings`` against whatever Lance namespace state exists then.
    After ``embed`` populates LanceDB the previously-bound view still points at
    the empty namespace -- subsequent stages (``community`` is the load-bearing
    one) read 0 rows. RFC §9.6 names this as the analyze stale-connection bug.

    Calling :func:`register_vss` again issues a ``CREATE OR REPLACE VIEW``
    against the live Lance dataset, which is the cheap fix.

    Drops any prior ``message_embeddings`` object first so the second bind is
    free to switch type. The first :func:`register_vss` call against an empty
    Lance namespace creates a fallback **table**; the second call (after embed
    populates Lance) wants to bind a **view** -- DuckDB rejects
    ``CREATE OR REPLACE VIEW`` over an existing table of the same name, so we
    drop both shapes proactively. ``DETACH lance_store`` clears any prior
    ``ATTACH`` so :func:`register_vss` can reissue it without an alias clash.
    """
    # Drop whichever shape (TABLE or VIEW) the prior register_vss created.
    # DuckDB's ``DROP VIEW IF EXISTS`` still errors if the existing object is a
    # Table (and vice versa) -- ``IF EXISTS`` only suppresses the not-found
    # error, not the type-mismatch error. Try VIEW first; on type mismatch fall
    # through to dropping the TABLE shape.
    try:
        con.execute("DROP VIEW IF EXISTS message_embeddings;")
    except duckdb.Error:
        # Existing object is a TABLE, not a VIEW -- drop it as TABLE.
        con.execute("DROP TABLE IF EXISTS message_embeddings;")
    # No prior ATTACH on this connection (first rebind, or skip_vss path) -->
    # DETACH against a missing alias raises; nothing to clean up.
    with contextlib.suppress(duckdb.Error):
        con.execute("DETACH lance_store;")
    expected_model, expected_dim = settings.expected_embedding_identity()
    register_vss(
        con,
        embeddings_parquet=settings.embeddings_parquet_path,
        lance_uri=settings.lance_uri,
        dim=int(settings.output_dimension),
        metric=settings.hnsw_metric,
        expected_model=expected_model,
        expected_dim=expected_dim,
    )
    logger.debug("register_vss re-bind after {}", stage)


# Tokens that imply a SQL statement needs the VSS extension and the
# ``message_embeddings`` table. Substring-matched against the lower-cased SQL.
# False positives (a literal like 'no message_embeddings here') just register
# VSS unnecessarily -- no correctness regression. False negatives can't happen
# without the user genuinely referencing one of these names in the SQL text.
_VSS_TOKENS: tuple[str, ...] = (
    "message_embeddings",
    "semantic_search",
    "array_cosine_",
    "array_distance",
)


def sql_uses_vss(sql: str) -> bool:
    """Cheap pre-flight: does ``sql`` need VSS / ``message_embeddings``?

    Used by :func:`open_connection_full` to gate ``register_vss`` and the
    ``semantic_search`` macro. Triggers on direct table/macro references
    (``message_embeddings``, ``semantic_search``) and on the two array
    distance functions the ``search`` command's hand-written SQL uses
    (``array_cosine_*``, ``array_distance``).
    """
    lowered = sql.lower()
    return any(tok in lowered for tok in _VSS_TOKENS)


def sql_uses_catalog(sql: str) -> bool:
    """Cheap pre-flight: does ``sql`` reference any registered view/macro?

    Substring-matches against ``VIEW_NAMES + MACRO_NAMES`` (case-insensitive)
    and against :data:`_VSS_TOKENS` so a query like
    ``SELECT count(*) FROM message_embeddings`` -- whose table is NOT in
    :data:`VIEW_NAMES` because it lives inside the attached ``hnsw_store`` --
    still routes to the full connection that registers VSS for it.
    False positives (a string literal containing ``'sessions'``) just trigger
    the slow path -- no correctness regression. False negatives can't happen
    if the user genuinely references a view: the name has to appear in the
    SQL text.
    """
    lowered = sql.lower()
    if any(name.lower() in lowered for name in (*VIEW_NAMES, *MACRO_NAMES)):
        return True
    return any(tok in lowered for tok in _VSS_TOKENS)


# ---------------------------------------------------------------------------
# Underscore aliases — historic ``app.cli`` names + test monkeypatch targets.
# Kept so ``test_config``/``test_cli``/``test_cli_coverage``/``test_pr3_perf``
# and the ``app.cli`` module bindings resolve unchanged after the move.
# ---------------------------------------------------------------------------
_resolve_memory_limit = resolve_memory_limit
_maybe_migrate_legacy_caches = maybe_migrate_legacy_caches
_apply_duckdb_pragmas = apply_duckdb_pragmas
_open_connection_full = open_connection_full
_open_connection_introspect = open_connection_introspect
_refresh_analytics_views = refresh_analytics_views
_rebind_vss = rebind_vss
_sql_uses_vss = sql_uses_vss
_sql_uses_catalog = sql_uses_catalog


__all__ = [
    "_MIGRATION_MARKER",
    "_PERCENT_LIMIT_RE",
    "_VSS_TOKENS",
    "_apply_duckdb_pragmas",
    "_maybe_migrate_legacy_caches",
    "_open_connection_full",
    "_open_connection_introspect",
    "_rebind_vss",
    "_refresh_analytics_views",
    "_resolve_memory_limit",
    "_sql_uses_catalog",
    "_sql_uses_vss",
    "apply_duckdb_pragmas",
    "maybe_migrate_legacy_caches",
    "open_connection_full",
    "open_connection_introspect",
    "rebind_vss",
    "refresh_analytics_views",
    "resolve_memory_limit",
    "sql_uses_catalog",
    "sql_uses_vss",
]

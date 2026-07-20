"""Cyclopts CLI entry point for ``claude-sql``.

Wires the ``claude-sql`` console script to its subcommands.  Shared
flags -- ``--verbose`` / ``--quiet``, ``--glob``, ``--subagent-glob``,
``--format`` -- live on a flattened :class:`Common` dataclass so callers write
``claude-sql query ... --format json`` instead of ``--common.format json``.

Agent-friendly defaults
-----------------------
* ``--format auto`` emits a human table on a TTY and machine-readable JSON
  when stdout is a pipe, so agents do not have to set a flag.
* DuckDB errors are classified into parse / catalog / runtime and mapped to
  stable exit codes (64 / 65 / 70) with a JSON error payload on non-TTY.
* ``--quiet`` is honored by every subcommand; view registration goes to DEBUG
  so the default stderr stays quiet for routine reads.

``asyncio`` and subprocess imports are performed lazily inside the relevant
commands so that the fast path (``schema``, ``query``, ``explain``) does not
drag extra modules into startup.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, override

import duckdb
import polars as pl
from cyclopts import App, Parameter
from loguru import logger
from pydantic import ValidationError

from claude_sql.domain.errors import (
    EXIT_CODES,
    ClassifiedError,
    InputValidationError,
)

# The DuckDB connection lifecycle moved to
# ``infrastructure.duckdb_connection`` in the Wave-4 rebind cut. These names are
# re-bound here under their historic underscore spellings so the command bodies
# below and the test monkeypatches (``test_analyze_chain`` patches
# ``cli._rebind_vss`` / ``_refresh_analytics_views`` / ``_open_connection_full``
# as module attributes; ``test_config`` / ``test_cli`` import
# ``cli._resolve_memory_limit``; ``test_cli_coverage`` calls
# ``cli._maybe_migrate_legacy_caches``) keep resolving. ``_resolve_memory_limit``
# and ``_maybe_migrate_legacy_caches`` are referenced only by those tests, so
# they ride in ``__all__`` as intentional re-exports.
from claude_sql.infrastructure.duckdb_connection import (
    _maybe_migrate_legacy_caches as _maybe_migrate_legacy_caches,  # noqa: PLC0414 — test re-export
    _open_connection_full,
    _open_connection_introspect,
    _rebind_vss,
    _refresh_analytics_views,
    _resolve_memory_limit as _resolve_memory_limit,  # noqa: PLC0414 — test re-export
    _sql_uses_catalog,
)
from claude_sql.infrastructure.duckdb_s3 import configure_s3, settings_need_s3
from claude_sql.infrastructure.duckdb_views import (
    MACRO_SIGNATURES,
    VIEW_SCHEMA,
    _parquet_is_populated,
    register_all,
    register_raw,
    register_views,
)
from claude_sql.infrastructure.home import claude_sql_home, recognized_legacy_caches
from claude_sql.infrastructure.logging_setup import configure_logging
from claude_sql.infrastructure.parquet_cache import (
    count_rows,
    is_sharded_dir,
    iter_part_files,
)
from claude_sql.infrastructure.settings import Settings
from claude_sql.infrastructure.sqlite_state import checkpointer

# NOTE — heavy analytics worker imports are DEFERRED into the command bodies
# that use them (see the ``import`` statements inside ``embed``, ``search``,
# ``classify``, ``cluster``, ``community``, ``analyze``, …). Each worker drags a
# ~0.5–1.4 s import subtree (boto3 via ``llm_shared``, ``schemas``, numpy/polars
# re-exports); the fast read-only path (``schema``/``query``/``explain``/
# ``peek``/``list-cache``/``shell``/``--version``/``--help``) touches none of
# them. Importing them at module top paid ~0.94 s on EVERY invocation. This
# extends the #77 lance_store deferral to the worker modules themselves; pinned
# by ``test_cli_import_is_lean`` (fresh-interpreter sys.modules assertion).
from claude_sql.interfaces.cli.install_source import format_version
from claude_sql.interfaces.cli.output import (
    OutputFormat,
    emit_dataframe,
    emit_error,
    emit_json,
    resolve_format,
    run_or_die,
    validate_glob,
)

_APP_HELP = """\
Zero-copy SQL + Cohere Embed v4 semantic search + Sonnet 4.6 analytics over
~/.claude/ JSONL transcripts (and their subagent sidecars).

Surfaces at a glance
--------------------
  schema / list-cache / explain   introspection (read-only, zero cost)
  query / shell                   run SQL against 18 views + 14 macros
  embed / search                  Cohere Embed v4 + HNSW cosine search
  classify / trajectory /         Sonnet 4.6 analytics -- each defaults to
  conflicts / friction            --dry-run; pass --no-dry-run to spend
  cluster / terms / community     UMAP+HDBSCAN, c-TF-IDF, Leiden+CPM
  analyze                         composite pipeline over every stage above

!! FLAG PLACEMENT — flags attach to a SUBCOMMAND, not the binary !!
-------------------------------------------------------------------
Every flag (--format, --quiet, --verbose, --glob, --subagent-glob, and
every per-command flag) goes AFTER the subcommand name. This applies
to global-feeling flags too: `--quiet` must come after the subcommand.

OK:
    claude-sql query --format json "SELECT 1"
    claude-sql schema --quiet --format json
    claude-sql classify --no-dry-run --limit 5
FAIL (cyclopts: "Unused Tokens: ['schema']"):
    claude-sql --quiet schema --format json
FAIL (flag swallowed as subcommand arg):
    claude-sql --format json query "SELECT 1"

Output & exit codes
-------------------
* --format {auto,table,json,ndjson,csv} on every subcommand. auto = table on
  TTY / json on pipe, so `claude-sql <cmd> | jq` works without a flag.
* 0   success
* 2   missing embeddings parquet (run: claude-sql embed --since-days N --no-dry-run)
* 64  invalid input -- malformed --glob, unparseable SQL, bad flag
* 65  catalog error -- unknown view/column; run `claude-sql schema` for the catalog
* 70  runtime error -- everything else DuckDB raises (check --format json stderr)
* 127 system `duckdb` binary not on PATH (only affects `shell`)

Cost guard
----------
Every command that calls Bedrock (embed, classify, trajectory, conflicts,
friction, analyze) defaults to --dry-run. Dry-run emits a plan JSON to stdout
with candidate counts, estimated tokens, and dollar estimate -- agents can
parse that to decide whether to proceed. Real spend requires --no-dry-run.

Glob scoping (cheaper workers)
------------------------------
Narrow to one project with --glob to cut worker budget:
    --glob "/home/you/.claude/projects/-efs-you-workplace-myproject/*.jsonl"
At most one '**' segment is allowed per pattern (DuckDB limitation) -- the
CLI rejects multi-star globs with a clear hint before DuckDB sees them.
"""


# Defined locally (not imported from ``community_worker``) so the ``community``
# command signature does not drag the igraph/leidenalg import subtree onto the
# module-load path. This IS the public ``--resolution`` CLI contract; it must
# stay byte-identical to ``community_worker.ResolutionLevel`` — pinned by
# ``test_resolution_level_matches_worker`` in test_pr3_perf.py.
ResolutionLevel = Literal["coarse", "medium", "fine"]


app = App(
    name="claude-sql",
    version=format_version,
    help=_APP_HELP,
)


@Parameter(name="*")
@dataclass
class Common:
    """Shared CLI flags flattened onto every subcommand.

    ``verbose`` and its paired ``--quiet`` negation both map to this single
    bool (cyclopts uses the ``negative=`` argument to wire the "opposite"
    flag onto the same field).  ``quiet`` is the one extra concept the
    dataclass needs to carry: it cannot piggyback on ``verbose`` because
    the two states are not symmetric (verbose forces DEBUG, quiet forces
    ERROR, and the default is INFO).
    """

    verbose: bool = False
    quiet: bool = False
    glob: str | None = None
    subagent_glob: str | None = None
    format: Annotated[OutputFormat, Parameter(name="--format")] = OutputFormat.AUTO


def _configure(common: Common | None) -> None:
    """Install logging based on the shared flags."""
    configure_logging(
        verbose=common.verbose if common else False,
        quiet=common.quiet if common else False,
    )


def _fmt(common: Common | None) -> OutputFormat:
    """Resolve the effective output format for a subcommand."""
    return common.format if common else OutputFormat.AUTO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_settings(common: Common | None) -> Settings:
    """Build :class:`Settings` from env then apply CLI overrides.

    Validates ``--glob`` / ``--subagent-glob`` up front so DuckDB never sees
    a pattern it cannot consume (e.g. ``**/.../**``).  On failure emits a
    classified error and exits with code 64 so every subcommand gets the
    same treatment without wrapping each call site.
    """
    settings = Settings()
    if common is None:
        return settings
    try:
        validate_glob(common.glob, flag="--glob")
        validate_glob(common.subagent_glob, flag="--subagent-glob")
    except InputValidationError as exc:
        err = ClassifiedError(
            kind="invalid_input",
            exit_code=EXIT_CODES["invalid_input"],
            message=str(exc),
            hint=exc.hint,
        )
        emit_error(err, _fmt(common))
        sys.exit(err.exit_code)
    updates: dict[str, str] = {}
    if common.glob is not None:
        updates["default_glob"] = common.glob
    if common.subagent_glob is not None:
        updates["subagent_glob"] = common.subagent_glob
    if not updates:
        return settings
    return settings.model_copy(update=updates)


def _apply_embedding_provider(settings: Settings, provider: str | None) -> Settings:
    """Return ``settings`` with ``embedding_provider`` overridden when given.

    The ``--embedding-provider`` CLI flag on ``search`` / ``embed`` / ``analyze``
    threads through here. ``None`` leaves the env/default provider in place;
    pydantic validates the Literal domain on ``model_copy`` and surfaces a
    classified error (exit 64) on an unknown value.
    """
    if provider is None:
        return settings
    try:
        # model_copy skips validation, so re-validate the dumped dict to reject
        # an unknown provider on the Literal domain.
        return Settings.model_validate({**settings.model_dump(), "embedding_provider": provider})
    except ValidationError as exc:
        err = ClassifiedError(
            kind="invalid_input",
            exit_code=EXIT_CODES["invalid_input"],
            message=f"invalid --embedding-provider {provider!r}: {exc}",
            hint="expected one of: cohere-bedrock, ollama, onnx-bge",
        )
        emit_error(err, OutputFormat.AUTO)
        sys.exit(err.exit_code)


def _apply_llm_analytics_provider(settings: Settings, provider: str | None) -> Settings:
    """Return ``settings`` with ``llm_analytics_provider`` overridden when given.

    The opt-in ``--llm-analytics-provider`` flag on ``trajectory`` / ``analyze``
    threads through here. ``None`` leaves the env/default provider
    (``sonnet-bedrock``) in place; pydantic validates the Literal domain and
    surfaces a classified error (exit 64) on an unknown value. Mirrors
    :func:`_apply_embedding_provider`, the Wave B embedding-provider seam.
    """
    if provider is None:
        return settings
    try:
        return Settings.model_validate(
            {**settings.model_dump(), "llm_analytics_provider": provider}
        )
    except ValidationError as exc:
        err = ClassifiedError(
            kind="invalid_input",
            exit_code=EXIT_CODES["invalid_input"],
            message=f"invalid --llm-analytics-provider {provider!r}: {exc}",
            hint="expected one of: sonnet-bedrock, strands-luna",
        )
        emit_error(err, OutputFormat.AUTO)
        sys.exit(err.exit_code)


def _emit_worker_result(result: int | dict[str, Any], common: Common | None, pipeline: str) -> None:
    """Normalize worker results for stdout.

    Workers return either an ``int`` (rows processed) or a plan ``dict`` when
    ``--dry-run`` is set. Agents parse stdout JSON, so we always emit something
    machine-readable: the plan dict under dry-run, or a compact summary dict
    when real work runs.
    """
    fmt = _fmt(common)
    if isinstance(result, dict):
        emit_json(result, fmt)
    else:
        emit_json({"pipeline": pipeline, "rows_processed": int(result), "dry_run": False}, fmt)


# EXPLAIN plan markers that indicate pushdown or noteworthy physical ops.
_EXPLAIN_MARKERS: tuple[str, ...] = (
    "READ_JSON",
    "Filters:",
    "Projection",
    "Filter",
    "HASH_JOIN",
    "HNSW_INDEX_SCAN",
    "HASH_GROUP_BY",
)


def _describe_checkpoint_entry(path: Path) -> dict[str, object]:
    """Report the persistent SQLite checkpoint file alongside the parquet caches.

    Keeps the same ``{name, path, exists[, bytes, mtime, rows]}`` shape as
    :func:`_describe_cache_entry` so ``list-cache`` stays homogeneous. Row
    count is queried via :func:`checkpointer.count_rows`, which transparently
    migrates from the legacy DuckDB file at first read.
    """
    exists = path.exists() and path.is_file()
    legacy_path = path.parent / "claude_sql.duckdb"
    legacy_exists = legacy_path.exists() and legacy_path.is_file()

    # When the SQLite file doesn't exist but the legacy DuckDB does, calling
    # count_rows triggers a one-time migration (creating the SQLite file as a
    # side effect). Run it eagerly so list-cache reflects the post-migration
    # state in the same call.
    if not exists and legacy_exists:
        # Eagerly trigger the one-time migration so list-cache reflects the
        # post-migration state. A corrupt or unreadable legacy file silently
        # skips — the surrounding entry still reports exists/bytes correctly.
        with contextlib.suppress(duckdb.Error, sqlite3.DatabaseError):
            checkpointer.count_rows(path)
        exists = path.exists() and path.is_file()

    entry: dict[str, object] = {"name": "session_checkpoint", "path": str(path), "exists": exists}
    if not exists:
        return entry
    stat = path.stat()
    entry["bytes"] = stat.st_size
    entry["mtime"] = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
    try:
        entry["rows"] = checkpointer.count_rows(path)
    except (duckdb.Error, sqlite3.DatabaseError):
        # Corrupt or unreadable file — list-cache should still report exists/bytes.
        entry["rows"] = None
    return entry


def _describe_lance_entry(path: Path) -> dict[str, object]:
    """Report the LanceDB embeddings store alongside the parquet caches.

    Same ``{name, path, exists[, bytes, mtime, rows]}`` shape as
    :func:`_describe_cache_entry`. Row count comes from
    :func:`lance_store.count_rows` (zero when the table is missing); ``bytes``
    sums every file under the dataset directory and ``mtime`` is the deepest
    nanosecond mtime across the tree (Lance writes new fragments, so the
    deepest mtime tracks the data).
    """
    from claude_sql.infrastructure import lance_store

    exists = path.exists() and path.is_dir()
    entry: dict[str, object] = {"name": "embeddings_lance", "path": str(path), "exists": exists}
    if not exists:
        return entry
    total_bytes = 0
    latest_mtime = path.stat().st_mtime
    for child in path.rglob("*"):
        if child.is_file():
            with contextlib.suppress(OSError):
                st = child.stat()
                total_bytes += st.st_size
                latest_mtime = max(latest_mtime, st.st_mtime)
    entry["bytes"] = total_bytes
    entry["mtime"] = datetime.fromtimestamp(latest_mtime, tz=UTC).isoformat()
    try:
        entry["rows"] = lance_store.count_rows(path)
    except (OSError, ValueError, RuntimeError):
        entry["rows"] = None
    return entry


def _describe_cache_entry(name: str, path: Path) -> dict[str, object]:
    """Collect filesystem metadata about one parquet cache entry.

    Handles both legacy single-file caches and the sharded directory layout
    (``<dir>/part-*.parquet``).  For a sharded directory, ``bytes`` is the
    sum across parts, ``mtime`` is the latest part's modification time,
    and ``rows`` is the union row count.

    Row counts are read via :func:`count_rows` (footer-only ``scan_parquet``)
    so the call is cheap even on very large caches.  A zero-byte / corrupt
    part surfaces ``rows=None`` rather than aborting the whole listing.
    """
    parts = iter_part_files(path)
    exists = bool(parts) or path.exists()
    entry: dict[str, object] = {"name": name, "path": str(path), "exists": exists}
    if not exists:
        return entry
    if not parts:
        # Path exists (e.g. an empty directory) but has no part files; surface
        # the bare directory mtime so users can still see "we made the dir".
        stat = path.stat()
        entry["bytes"] = 0
        entry["mtime"] = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
        entry["rows"] = 0
        return entry
    total_bytes = 0
    latest_mtime = 0.0
    healthy = True
    for p in parts:
        st = p.stat()
        total_bytes += st.st_size
        latest_mtime = max(latest_mtime, st.st_mtime)
        if st.st_size <= 16:
            healthy = False
    entry["bytes"] = total_bytes
    entry["mtime"] = datetime.fromtimestamp(latest_mtime, tz=UTC).isoformat()
    if not healthy:
        entry["rows"] = None
        return entry
    try:
        entry["rows"] = count_rows(path)
    except (OSError, ValueError):
        # ``count_rows`` is a polars scan; an unreadable footer surfaces here.
        entry["rows"] = None
    return entry


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@app.command
def shell(*, common: Common | None = None) -> None:
    """Launch the interactive duckdb REPL with every view, macro, and the HNSW index pre-registered.

    When to use
    -----------
    Interactive exploration -- iterating on SQL joins, inspecting macros,
    feeling out the catalog. Agents should prefer ``query`` (single-shot)
    or ``shell`` via a subprocess only when they truly need a session.

    What it does
    ------------
    1. Creates a temporary on-disk DuckDB file.
    2. Runs ``register_all`` to materialize 18 views + 14 macros + VSS.
    3. Execs the system ``duckdb`` binary against the file.

    Exit codes
    ----------
    * 127  ``duckdb`` binary not on PATH (install it with `uv tool install
      duckdb` or your OS package manager).

    Notes
    -----
    The temp DB path is printed on startup so you can reopen it later or
    delete it. The path is NOT cleaned up automatically on exit -- that's
    intentional so long-running sessions can be resumed.
    """
    _configure(common)
    settings = _resolve_settings(common)

    # mkstemp returns a tuple of (fd, path); we want the path only — duckdb
    # opens its own handle.  Closing the fd immediately keeps the file but
    # releases the descriptor (mkstemp is preferred over NamedTemporaryFile
    # here because we never write to the handle; we just need a unique
    # path that already exists on disk so duckdb can open it).
    fd, db_path = tempfile.mkstemp(suffix=".duckdb")
    os.close(fd)

    con = duckdb.connect(db_path)
    try:
        register_all(con, settings=settings)
    finally:
        con.close()

    logger.info("Opening DuckDB REPL with pre-registered views + macros + HNSW index")
    logger.info("(Exit with .quit; DB persists at {})", db_path)
    try:
        subprocess.run(["duckdb", db_path], check=False)
    except FileNotFoundError:
        logger.error(
            "`duckdb` binary not found on PATH. Install it or run queries via "
            "`claude-sql query '<sql>'`. DB persists at {}",
            db_path,
        )
        sys.exit(EXIT_CODES["duckdb_missing"])


def _profile_path_for(label: str) -> Path:
    """Build the destination path used by ``--profile-json``.

    Splits filename composition out of the writer so callers can configure
    DuckDB's ``profiling_output`` PRAGMA before the profiled query runs
    (DuckDB writes the JSON itself; we just read it back to confirm the
    file landed and surface its location to the user).
    """
    profiling_dir = claude_sql_home() / "profiling"
    profiling_dir.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-") or "profile"
    return profiling_dir / f"{safe_label}-{int(time.time() * 1000)}.json"


def _capture_profile(con: duckdb.DuckDBPyConnection, label: str) -> Path:
    """Run a profiled query and return where DuckDB persisted the JSON output.

    Caller must have set ``enable_profiling = 'json'`` and pointed
    ``profiling_output`` at a file *before* executing the query of
    interest. We synthesize the output path here, set the PRAGMAs, and
    return the path the next query will populate. The caller is
    responsible for executing exactly one statement after this returns.
    """
    out_path = _profile_path_for(label)
    # Escape single-quotes in the path for the SQL literal; tmp paths can
    # contain unusual characters under pytest.
    safe_path = str(out_path).replace("'", "''")
    con.execute("SET enable_profiling = 'json'")
    con.execute(f"SET profiling_output = '{safe_path}'")
    return out_path


@app.command
def query(
    sql: str,
    /,
    *,
    profile_json: bool = False,
    common: Common | None = None,
) -> None:
    """Run one SQL query against the claude-sql catalog and emit results.

    When to use
    -----------
    Read-only exploration and aggregation against the 18 pre-registered
    views. The catalog is free (no Bedrock, no LLM, no cost), so run queries
    liberally -- they're the cheapest way to introspect sessions / messages
    / tool calls / analytics.

    Positional args
    ---------------
    SQL
        A single SQL statement. Multi-statement scripts are rejected by
        DuckDB's single-exec path -- use ``shell`` for those.

    Key flags
    ---------
    --glob PATTERN
        Narrow the universe of JSONLs scanned. Must have at most one '**'
        segment. Example:
            --glob "/home/you/.claude/projects/-efs-you-myproject/*.jsonl"
    --subagent-glob PATTERN
        Same, for subagent sidecar files.
    --format {auto,table,json,ndjson,csv}
        auto emits table on TTY, json on pipe.

    Output
    ------
    TTY default: Polars-rendered table.
    Non-TTY: JSON array of row dicts (ideal for `jq` / agent parsing).

    Exit codes
    ----------
    * 64  parse_error   malformed SQL (see error.hint for the fix)
    * 65  catalog_error unknown view/macro/column (try ``schema``)
    * 70  runtime_error everything else DuckDB raises

    Catalog discovery
    -----------------
    Run ``claude-sql schema --format json`` for the full view + macro list,
    or ``claude-sql list-cache`` to see which analytics parquets exist.

    Examples
    --------
    Session counts:
        claude-sql query "SELECT COUNT(*) FROM sessions"
    Top assistants by token spend:
        claude-sql query --format json "
          SELECT model, SUM(input_tokens + output_tokens) AS toks
          FROM messages GROUP BY 1 ORDER BY 2 DESC LIMIT 5"
    """
    _configure(common)
    settings = _resolve_settings(common)
    fmt = _fmt(common)
    con = (
        _open_connection_full(settings, sql=sql)
        if _sql_uses_catalog(sql)
        else _open_connection_introspect(settings)
    )
    try:
        profile_path: Path | None = None
        if profile_json:
            profile_path = _capture_profile(con, label="query")
        df = run_or_die(lambda: con.execute(sql).pl(), fmt=fmt)
        emit_dataframe(df, fmt)
        if profile_path is not None:
            logger.info("Wrote profile JSON: {}", profile_path)
    finally:
        con.close()


@app.command
def explain(
    sql: str,
    /,
    *,
    analyze: bool = False,
    profile_json: bool = False,
    common: Common | None = None,
) -> None:
    """Show the DuckDB query plan and highlight pushdown / noteworthy operators.

    When to use
    -----------
    Before running a ``query`` that might scan a lot of JSONLs -- confirm
    filter pushdown, spot accidental full scans, verify HNSW_INDEX_SCAN
    kicks in for vector searches.

    Flags
    -----
    --analyze
        Run ``EXPLAIN ANALYZE`` (executes the query and reports real
        timings). Off by default so probing slow queries is free.
    --format {auto,table,json,...}
        TTY table highlights READ_JSON / Filter / HASH_JOIN / HASH_GROUP_BY
        / HNSW_INDEX_SCAN in green. JSON emits ``{"plan": "<text>"}``.

    Exit codes
    ----------
    Same as ``query``: 64 parse / 65 catalog / 70 runtime.
    """
    _configure(common)
    settings = _resolve_settings(common)
    fmt = resolve_format(_fmt(common))
    con = (
        _open_connection_full(settings, sql=sql)
        if _sql_uses_catalog(sql)
        else _open_connection_introspect(settings)
    )
    try:
        profile_path: Path | None = None
        if profile_json:
            profile_path = _capture_profile(con, label="explain")
        prefix = "EXPLAIN ANALYZE " if analyze else "EXPLAIN "
        rows = run_or_die(lambda: con.execute(prefix + sql).fetchall(), fmt=fmt)
        # EXPLAIN rows are (type, plan_text) tuples; the plan sits in the last
        # column regardless of row shape.
        text = "\n".join(str(r[-1]) for r in rows)
        if fmt is OutputFormat.TABLE:
            for line in text.splitlines():
                if any(m in line for m in _EXPLAIN_MARKERS):
                    print(f"\033[92m{line}\033[0m")
                else:
                    print(line)
        else:
            emit_json({"plan": text}, fmt)
        if profile_path is not None:
            logger.info("Wrote profile JSON: {}", profile_path)
    finally:
        con.close()


def _compute_cached_map(settings: Settings) -> dict[str, bool]:
    """Map analytics view + analytics macro names to parquet-existence.

    A name appears in this map when its data lives in a parquet that may
    or may not exist on disk (the v2 analytics surface). v1 views always
    have data (the transcript globs are always present), so they don't
    appear here. Use ``list-cache`` for byte counts / mtimes / row counts.
    """
    # Analytics view → backing parquet path on Settings.
    view_paths: dict[str, Path] = {
        "session_classifications": settings.classifications_parquet_path,
        "session_goals": settings.classifications_parquet_path,
        "message_trajectory": settings.trajectory_parquet_path,
        "session_conflicts": settings.conflicts_parquet_path,
        "message_clusters": settings.clusters_parquet_path,
        "cluster_terms": settings.cluster_terms_parquet_path,
        "session_communities": settings.communities_parquet_path,
        "community_profile": settings.community_profile_parquet_path,
        "user_friction": settings.user_friction_parquet_path,
        "skills_catalog": settings.skills_catalog_parquet_path,
        "skill_usage": settings.skills_catalog_parquet_path,
    }
    cached: dict[str, bool] = {
        name: _parquet_is_populated(path) for name, path in view_paths.items()
    }
    # Analytics macros depend on the same parquets as their backing views;
    # surface them too so an agent asking "is friction_rate ready?" gets a
    # direct yes/no without re-deriving the dependency.
    macro_paths: dict[str, tuple[Path, ...]] = {
        "autonomy_trend": (settings.classifications_parquet_path,),
        "work_mix": (settings.classifications_parquet_path,),
        "success_rate_by_work": (settings.classifications_parquet_path,),
        "cluster_top_terms": (settings.cluster_terms_parquet_path,),
        "community_top_topics": (
            settings.cluster_terms_parquet_path,
            settings.communities_parquet_path,
            settings.clusters_parquet_path,
        ),
        "sentiment_arc": (settings.trajectory_parquet_path,),
        "friction_counts": (settings.user_friction_parquet_path,),
        "friction_rate": (settings.user_friction_parquet_path,),
        "friction_examples": (settings.user_friction_parquet_path,),
        "unused_skills": (settings.skills_catalog_parquet_path,),
    }
    for macro_name, paths in macro_paths.items():
        cached[macro_name] = all(_parquet_is_populated(p) for p in paths)
    return cached


@app.command
def schema(*, common: Common | None = None) -> None:
    """List every registered view (with columns) and every macro signature.

    When to use
    -----------
    First thing an agent should call after ``--help``: it's the canonical
    catalog. Use it to discover column names before composing ``query``
    calls -- e.g., ``session_classifications`` uses both ``autonomy_tier``
    (canonical) and ``autonomy`` (alias), and the schema lists both.

    Implementation
    --------------
    Reads the static :data:`VIEW_SCHEMA` and :data:`MACRO_SIGNATURES`
    dicts -- no DuckDB connection, no JSON schema inference, no view
    registration. Sub-50ms even on large corpora. The ``cached`` map is
    keyed by analytics view + analytics macro names so an agent can tell
    which parquet-backed entries are populated; use ``list-cache`` for
    byte counts and mtimes.

    Output shape (non-TTY / JSON)
    -----------------------------
    ::

        {
          "views": {
            "sessions": [{"column": "session_id", "type": "VARCHAR"}, ...],
            "messages": [...],
            ...
          },
          "macros": [{"name": "ago", "params": ["interval_text"]}, ...],
          "cached": {"user_friction": false, "friction_rate": false, ...}
        }

    Only v1 (transcript-derived) views appear under ``views`` -- v2
    analytics views are parquet-backed; their schema source-of-truth is
    the parquet metadata. Use ``cached`` to see which v2 views can be
    queried right now.
    """
    _configure(common)
    settings = _resolve_settings(common)
    fmt = resolve_format(_fmt(common))
    cached = _compute_cached_map(settings)
    payload = {
        "views": {
            name: [{"column": c, "type": t} for c, t in cols] for name, cols in VIEW_SCHEMA.items()
        },
        "macros": [{"name": n, "params": list(p)} for n, p in MACRO_SIGNATURES.items()],
        "cached": cached,
    }
    if fmt is OutputFormat.TABLE:
        for name, cols in VIEW_SCHEMA.items():
            print(f"\n\033[1m{name}\033[0m ({len(cols)} cols)")
            for col, col_type in cols:
                print(f"  {col:<28} {col_type}")
        print(f"\n\033[1mMacros\033[0m ({len(MACRO_SIGNATURES)})")
        for n, params in MACRO_SIGNATURES.items():
            tag = "" if cached.get(n, True) else "  [cache empty]"
            print(f"  {n}({', '.join(params)}){tag}")
        return
    emit_json(payload, fmt)


@app.command(name="list-cache")
def list_cache(*, common: Common | None = None) -> None:
    """Report each parquet cache's presence, size, freshness, and row count.

    When to use
    -----------
    Before running ``search`` (requires ``embeddings``) or composing
    analytics queries (require ``session_classifications`` /
    ``message_trajectory`` / ``session_conflicts`` / ``message_clusters``
    / ``cluster_terms`` / ``session_communities`` / ``user_friction``).

    What it reports
    ---------------
    One entry per cache (plus the persistent checkpointer DB):
    ``{name, path, exists, bytes, mtime, rows}``.  When ``exists`` is
    false, ``bytes`` / ``mtime`` / ``rows`` are omitted.

    How to populate each cache
    --------------------------
    * embeddings              → ``claude-sql embed --no-dry-run``
    * session_classifications → ``claude-sql classify --no-dry-run``
    * message_trajectory      → ``claude-sql trajectory --no-dry-run``
    * session_conflicts       → ``claude-sql conflicts --no-dry-run``
    * message_clusters        → ``claude-sql cluster``
    * cluster_terms           → ``claude-sql terms``
    * session_communities     → ``claude-sql community``
    * user_friction           → ``claude-sql friction --no-dry-run``
    * skills_catalog          → ``claude-sql skills sync``
    """
    _configure(common)
    settings = _resolve_settings(common)
    fmt = resolve_format(_fmt(common))
    entries: list[dict[str, object]] = [
        _describe_lance_entry(settings.lance_uri),
        _describe_cache_entry("embeddings_legacy", settings.embeddings_parquet_path),
        _describe_cache_entry("session_classifications", settings.classifications_parquet_path),
        _describe_cache_entry("message_trajectory", settings.trajectory_parquet_path),
        _describe_cache_entry("session_conflicts", settings.conflicts_parquet_path),
        _describe_cache_entry("message_clusters", settings.clusters_parquet_path),
        _describe_cache_entry("cluster_terms", settings.cluster_terms_parquet_path),
        _describe_cache_entry("session_communities", settings.communities_parquet_path),
        _describe_cache_entry("community_profile", settings.community_profile_parquet_path),
        _describe_cache_entry("user_friction", settings.user_friction_parquet_path),
        _describe_cache_entry("skills_catalog", settings.skills_catalog_parquet_path),
        _describe_cache_entry("ingest_stamps", settings.ingest_stamps_parquet_path),
        _describe_checkpoint_entry(settings.checkpoint_db_path),
    ]
    # One-time deprecation breadcrumb: surface any caches that still live
    # under the old ``~/.claude/`` root after the auto-migration ran. This
    # is rare (auto-migration would have moved them) but possible if the
    # marker file got stamped before a manual restore — flag it so the
    # user can ``claude-sql cache migrate`` or move them by hand.
    for name, legacy_path in recognized_legacy_caches().items():
        entries.append(
            {
                "name": f"legacy:{name}",
                "path": str(legacy_path),
                "exists": True,
                "bytes": None,
                "mtime": None,
                "rows": None,
            }
        )

    if fmt is OutputFormat.TABLE:
        df = pl.DataFrame(entries)
        emit_dataframe(df, OutputFormat.TABLE)
        return
    # JSON / NDJSON / CSV -- emit the list directly so downstream tooling
    # doesn't have to unwrap a wrapper object.
    if fmt is OutputFormat.NDJSON:
        for entry in entries:
            sys.stdout.write(json.dumps(entry, default=str))
            sys.stdout.write("\n")
        return
    if fmt is OutputFormat.CSV:
        emit_dataframe(pl.DataFrame(entries), OutputFormat.CSV)
        return
    emit_json(entries, fmt)


_PEEK_SAMPLE_CHARS = 240
_PEEK_TOP_TOOLS = 10


@app.command
def peek(session_id: str, /, *, common: Common | None = None) -> None:
    """One-shot summary of a session: lines, role mix, top tools, samples.

    When to use
    -----------
    Quick session introspection -- replaces the recurring "open JSONL,
    count lines, peek at first/last messages" inline-Python pattern.
    Reads the catalog only; no Bedrock, no parquet caches.

    Output (JSON)
    -------------
    ``{session_id, source_file, total_lines, first_ts, last_ts,
    roles{role: count}, top_tools[{name, count}],
    samples{first_user, last_user, first_assistant_text}}``

    Each ``samples`` slot is ``null`` when no qualifying message exists
    (e.g. session has only short text, where ``messages_text`` drops
    rows under 32 characters).

    Exit codes
    ----------
    * 64  invalid_input  malformed --glob
    * 65  not_found      session_id absent from the corpus
    * 70  runtime_error  any other DuckDB failure
    """
    from claude_sql.application.use_cases.peek import peek_session

    _configure(common)
    settings = _resolve_settings(common)
    fmt = resolve_format(_fmt(common))
    con = _open_connection_full(settings, sql="FROM messages")
    try:
        payload = run_or_die(
            lambda: peek_session(
                con,
                session_id,
                sample_chars=_PEEK_SAMPLE_CHARS,
                top_tools=_PEEK_TOP_TOOLS,
            ),
            fmt=fmt,
        )
        if payload is None:
            err = ClassifiedError(
                kind="not_found",
                exit_code=EXIT_CODES["catalog_error"],
                message=f"session_id {session_id!r} not found in corpus",
                hint=(
                    'run `claude-sql query "SELECT session_id FROM sessions LIMIT 5"` for valid ids'
                ),
            )
            emit_error(err, fmt)
            sys.exit(err.exit_code)
    finally:
        con.close()

    if fmt is OutputFormat.TABLE:
        _peek_render_table(payload)
        return
    emit_json(payload, fmt)


def _peek_render_table(p: dict[str, Any]) -> None:
    print(
        f"session: {p['session_id']}   ({p['total_lines']} lines, {p['first_ts']} → {p['last_ts']})"
    )
    print(f"source : {p['source_file']}")
    print()
    print("roles:")
    for role, n in p["roles"].items():
        print(f"  {role:<12} {n}")
    print()
    print("top tools:")
    if not p["top_tools"]:
        print("  (none)")
    for entry in p["top_tools"]:
        print(f"  {entry['name']:<24} {entry['count']}")
    print()
    print("samples:")
    label = {
        "first_user": "first user",
        "last_user": "last  user",
        "first_assistant_text": "first asst",
    }
    for slot, sample in p["samples"].items():
        if sample is None:
            print(f"  [{label[slot]}] (none)")
            continue
        print(f"  [{label[slot]}, {sample['ts']}] {sample['text']}")


# ---------------------------------------------------------------------------
# ``cache`` sub-app — compact / migrate the sharded worker-output parquets.
# ---------------------------------------------------------------------------
#
# Workers (embed, classify, trajectory, conflicts, friction) write each
# chunk as a fresh ``part-<ts_ns>.parquet`` under their cache directory.
# Over time many small parts accumulate; ``cache compact`` consolidates
# them into a single ``part-compacted-<ts>.parquet`` and removes the
# originals.  ``cache migrate`` walks legacy single-file caches that
# pre-date this layout and moves each one into a sibling directory with
# its existing mtime preserved so the HNSW persistence and cluster-mtime
# sidecar logic stay valid.
#
# Both commands honour ``--dry-run`` (default ``True``) the same way every
# Bedrock-bearing command does in this codebase: nothing happens until you
# pass ``--no-dry-run``.

cache_app = App(
    name="cache",
    help=(
        "Manage the sharded worker-output parquet caches.\n\n"
        "  cache compact  consolidates many ``part-*.parquet`` shards into one.\n"
        "  cache migrate  moves a legacy single-file cache into the new dir layout.\n\n"
        "Both commands default to --dry-run; pass --no-dry-run to act."
    ),
)
app.command(cache_app)


def _resolve_cache_paths(settings: Settings) -> dict[str, Path]:
    """Return ``{cache_name: path}`` for every worker-append cache.

    These are the five caches with sharded-write semantics: writers append
    by dropping fresh parts, so they accumulate and benefit from ``compact``.
    The four single-write caches (``clusters``, ``cluster_terms``,
    ``communities``, ``skills_catalog``) and the checkpoint DB don't fit
    the same pattern and are intentionally excluded.
    """
    return {
        "embeddings": settings.embeddings_parquet_path,
        "session_classifications": settings.classifications_parquet_path,
        "message_trajectory": settings.trajectory_parquet_path,
        "session_conflicts": settings.conflicts_parquet_path,
        "user_friction": settings.user_friction_parquet_path,
    }


@cache_app.command(name="compact")
def cache_compact(
    *,
    name: str | None = None,
    dry_run: bool = True,
    common: Common | None = None,
) -> None:
    """Consolidate ``part-*.parquet`` shards into a single compacted part file.

    Walks each sharded cache directory, reads every part, writes a fresh
    ``part-compacted-<ts_ns>.parquet`` containing the union, and only after
    that succeeds removes the originals.  Legacy single-file caches and
    caches with zero or one parts are left alone — there is nothing to
    consolidate.

    Flags
    -----
    --name <cache>    Restrict to one of: embeddings, session_classifications,
                      message_trajectory, session_conflicts, user_friction.
                      Default is "all five".
    --dry-run         Default True. Pass ``--no-dry-run`` to actually rewrite.
    """
    _configure(common)
    settings = _resolve_settings(common)
    fmt = resolve_format(_fmt(common))

    targets = _resolve_cache_paths(settings)
    if name is not None:
        if name not in targets:
            err = ClassifiedError(
                kind="invalid_input",
                exit_code=EXIT_CODES["invalid_input"],
                message=f"Unknown cache name: {name!r}",
                hint=f"Pick one of: {', '.join(sorted(targets))}",
            )
            emit_error(err, _fmt(common))
            sys.exit(err.exit_code)
        targets = {name: targets[name]}

    summaries: list[dict[str, object]] = []
    for cache_name, path in targets.items():
        parts = iter_part_files(path)
        if len(parts) <= 1 or not is_sharded_dir(path):
            summaries.append(
                {
                    "name": cache_name,
                    "path": str(path),
                    "parts": len(parts),
                    "action": "skip",
                    "reason": "no_compaction_needed",
                }
            )
            continue
        if dry_run:
            total_bytes = sum(p.stat().st_size for p in parts)
            summaries.append(
                {
                    "name": cache_name,
                    "path": str(path),
                    "parts": len(parts),
                    "bytes": total_bytes,
                    "action": "would_compact",
                }
            )
            continue
        # Read the union via polars, write a fresh compacted shard, delete
        # the originals only after the write succeeds.  Any IO error here
        # leaves the directory intact so a retry does not lose data.
        df = pl.read_parquet([str(p) for p in parts])
        compacted = path / f"part-compacted-{time.time_ns()}.parquet"
        df.write_parquet(compacted)
        for p in parts:
            p.unlink()
        summaries.append(
            {
                "name": cache_name,
                "path": str(path),
                "parts": len(parts),
                "rows": int(df.height),
                "compacted_to": str(compacted),
                "action": "compacted",
            }
        )

    if fmt is OutputFormat.TABLE:
        emit_dataframe(pl.DataFrame(summaries), OutputFormat.TABLE)
        return
    if fmt is OutputFormat.NDJSON:
        for s in summaries:
            sys.stdout.write(json.dumps(s, default=str))
            sys.stdout.write("\n")
        return
    if fmt is OutputFormat.CSV:
        emit_dataframe(pl.DataFrame(summaries), OutputFormat.CSV)
        return
    emit_json(summaries, fmt)


@cache_app.command(name="migrate")
def cache_migrate(
    *,
    dry_run: bool = True,
    common: Common | None = None,
) -> None:
    """Move legacy single-file caches into the sharded directory layout.

    For each of the five worker-append caches, looks for the historical
    ``~/.claude/<name>.parquet`` file alongside the new
    ``~/.claude/<name>/`` directory. When a single-file cache exists, the
    file is moved (not copied) into the directory as
    ``part-<original_mtime_ns>.parquet`` so subsequent runs treat it as
    just another shard. The original mtime is preserved on the new file so
    HNSW-persistence freshness checks behave identically.

    Flags
    -----
    --dry-run    Default True. Pass ``--no-dry-run`` to actually move files.
    """
    _configure(common)
    settings = _resolve_settings(common)
    fmt = resolve_format(_fmt(common))

    targets = _resolve_cache_paths(settings)
    summaries: list[dict[str, object]] = []
    for cache_name, dir_path in targets.items():
        # Legacy single-file path is the same parent directory + the cache
        # name + ".parquet" — that's what ``_default_*_parquet`` returned
        # before this PR.
        legacy = dir_path.with_suffix(".parquet")
        # Some users may have customised the cache path explicitly; we only
        # touch the canonical sibling, never an arbitrary user file.
        if not legacy.is_file():
            summaries.append(
                {
                    "name": cache_name,
                    "from": str(legacy),
                    "to": str(dir_path),
                    "action": "skip",
                    "reason": "no_legacy_file",
                }
            )
            continue
        original_ns = legacy.stat().st_mtime_ns
        target = dir_path / f"part-{original_ns}.parquet"
        if dry_run:
            summaries.append(
                {
                    "name": cache_name,
                    "from": str(legacy),
                    "to": str(target),
                    "bytes": legacy.stat().st_size,
                    "action": "would_move",
                }
            )
            continue
        dir_path.mkdir(parents=True, exist_ok=True)
        # ``rename`` preserves contents and mtime when both paths live on
        # the same filesystem — for the canonical ``~/.claude/`` layout
        # they always do. ``os.utime`` is a defensive belt+suspenders.
        legacy.rename(target)
        os.utime(target, ns=(original_ns, original_ns))
        summaries.append(
            {
                "name": cache_name,
                "from": str(legacy),
                "to": str(target),
                "action": "migrated",
            }
        )

    if fmt is OutputFormat.TABLE:
        emit_dataframe(pl.DataFrame(summaries), OutputFormat.TABLE)
        return
    if fmt is OutputFormat.NDJSON:
        for s in summaries:
            sys.stdout.write(json.dumps(s, default=str))
            sys.stdout.write("\n")
        return
    if fmt is OutputFormat.CSV:
        emit_dataframe(pl.DataFrame(summaries), OutputFormat.CSV)
        return
    emit_json(summaries, fmt)


# ---------------------------------------------------------------------------
# ``skills`` sub-app — catalog of locally-available Skills and slash commands.
# ---------------------------------------------------------------------------

skills_app = App(
    name="skills",
    help=(
        "Seed and inspect the local Skills catalog.\n\n"
        "The catalog binds skill_id (e.g. 'erpaval', 'personal-plugins:erpaval') "
        "to its human description, source plugin, and version so skill_usage can "
        "enrich raw invocations. Seeded from ~/.claude/skills/ and "
        "~/.claude/plugins/cache/**; no Bedrock cost."
    ),
)
app.command(skills_app)


@skills_app.command(name="sync")
def skills_sync(
    *,
    dry_run: bool = False,
    common: Common | None = None,
) -> None:
    """Walk ``~/.claude/skills`` and ``~/.claude/plugins/cache`` → skills_catalog.parquet.

    Sources
    -------
    * ``~/.claude/skills/<name>/SKILL.md``                        → ``user-skill``
    * ``<plugins_cache>/<owner>/<plugin>/<v>/skills/<n>/SKILL.md``
      → ``plugin-skill`` (bare + ``<plugin>:<n>``)
    * ``<plugins_cache>/<owner>/<plugin>/<v>/commands/<n>.md``
      → ``plugin-command`` (bare + ``<plugin>:<n>``)
    * Built-in slash commands (``/clear``, ``/compact``, …)       → ``builtin``

    Cost: zero (pure filesystem walk).  Run whenever you install or
    upgrade a plugin; ``claude-sql analyze`` runs it automatically.

    Flags
    -----
    --dry-run  Count rows without writing the parquet.  Useful for
               previewing how many skills will be catalogued.
    """
    from claude_sql.application.use_cases import skills as _skills_catalog

    _configure(common)
    settings = _resolve_settings(common)
    stats = _skills_catalog.sync(settings, dry_run=dry_run)
    target = "would write" if dry_run else "wrote"
    logger.info(
        "skills sync: {} {} rows to {} ({} skills, {} commands, {} builtins)",
        target,
        stats["rows"],
        settings.skills_catalog_parquet_path,
        stats["skills"],
        stats["commands"],
        stats["builtins"],
    )


@skills_app.command(name="ls")
def skills_ls(
    *,
    kind: str | None = None,
    plugin: str | None = None,
    common: Common | None = None,
) -> None:
    """List entries from the skills catalog parquet.

    Run ``claude-sql skills sync`` first.  Emits the catalog in the
    shared ``--format`` shape (table on TTY, JSON on pipe).

    Flags
    -----
    --kind <value>    Filter by ``source_kind`` (``user-skill``,
                      ``plugin-skill``, ``plugin-command``, ``builtin``).
    --plugin <value>  Filter by plugin name (exact match).
    """
    _configure(common)
    settings = _resolve_settings(common)
    fmt = resolve_format(_fmt(common))
    path = settings.skills_catalog_parquet_path
    if not path.exists():
        logger.error(
            "skills catalog parquet missing at {}. Run `claude-sql skills sync` first.",
            path,
        )
        sys.exit(EXIT_CODES["no_embeddings"])
    df = pl.read_parquet(path)
    if kind is not None:
        df = df.filter(pl.col("source_kind") == kind)
    if plugin is not None:
        df = df.filter(pl.col("plugin") == plugin)
    df = df.sort(["source_kind", "plugin", "name"], nulls_last=True)
    if fmt is OutputFormat.TABLE:
        emit_dataframe(df, OutputFormat.TABLE)
        return
    if fmt is OutputFormat.CSV:
        emit_dataframe(df, OutputFormat.CSV)
        return
    if fmt is OutputFormat.NDJSON:
        for row in df.iter_rows(named=True):
            sys.stdout.write(json.dumps(row, default=str))
            sys.stdout.write("\n")
        return
    emit_json(df.to_dicts(), fmt)


@app.command
def embed(
    *,
    since_days: int | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    embedding_provider: str | None = None,
    common: Common | None = None,
) -> None:
    """Embed new messages with the active embedding provider and append to LanceDB.

    Cost
    ----
    Calls Bedrock (``global.cohere.embed-v4:0``) on every unembedded
    message. ``--dry-run`` is OFF by default here (unlike LLM workers);
    pass it if you only want to see the plan.

    Flags
    -----
    --since-days N     Only consider messages newer than N days.
    --limit N          Cap the number of messages embedded this run.
    --dry-run          Preview only; emit plan JSON, no embedding calls.
    --embedding-provider {cohere-bedrock,ollama,onnx-bge}
                       Override the embedding backend for this run. Switching
                       providers requires an empty store (rm -rf the Lance dir);
                       the fail-loud guard refuses to mix vector spaces.
    --glob PATTERN     Narrow the universe (see top-level --help).

    Dry-run output (stdout JSON)
    ----------------------------
    ::

        {
            "pipeline": "embed",
            "candidates": N,
            "batches": B,
            "batch_size": 96,
            "concurrency": 2,
            "model": "...",
            "since_days": null,
            "limit": null,
            "dry_run": true,
        }

    Real-run output
    ---------------
    ``{"pipeline": "embed", "rows_processed": N, "dry_run": false}``

    Exit codes: 0 success, 70 runtime (Bedrock / DuckDB failure).
    """
    import asyncio

    from claude_sql.application.use_cases.embed import run_backfill

    _configure(common)
    settings = _apply_embedding_provider(_resolve_settings(common), embedding_provider)
    con = duckdb.connect(":memory:")
    try:
        if settings_need_s3(settings):
            configure_s3(con, settings)
        register_raw(
            con,
            glob=settings.default_glob,
            subagent_glob=settings.subagent_glob,
            subagent_meta_glob=settings.subagent_meta_glob,
        )
        register_views(con)
        result = asyncio.run(
            run_backfill(
                con=con,
                settings=settings,
                since_days=since_days,
                limit=limit,
                dry_run=dry_run,
            )
        )
        logger.info("Embedded {} messages (dry_run={})", result, dry_run)
        _emit_worker_result(result, common, pipeline="embed")
    finally:
        con.close()


@app.command
def search(
    query_text: str,
    /,
    *,
    k: int = 10,
    embedding_provider: str | None = None,
    common: Common | None = None,
) -> None:
    """Semantic top-k nearest-neighbor search over message embeddings via HNSW.

    Pipeline
    --------
    1. Embed ``query_text`` with Cohere Embed v4 ``search_query`` mode.
    2. DuckDB VSS HNSW cosine lookup against the existing embeddings parquet.
    3. Join back to ``messages_text`` for a 200-char snippet.

    Prereq
    ------
    The embeddings parquet must exist. If it's empty or missing, the
    command exits with code 2 and a hint. Run
    ``claude-sql embed --since-days 7 --no-dry-run`` to populate.

    Positional args
    ---------------
    QUERY_TEXT    A single natural-language query string.

    Flags
    -----
    --k N              Top-k (default 10).
    --embedding-provider {cohere-bedrock,ollama,onnx-bge}
                       Override the embedding backend used to embed the query.
                       Must match the provider that wrote the store, else the
                       fail-loud guard raises rather than return garbage scores.
    --glob PATTERN     Narrow the messages_text view before the HNSW join.
    --format ...       See top-level --help.

    Output columns
    --------------
    uuid, session_id, role, sim (cosine similarity ∈ [-1, 1]), snippet.
    Sorted by cosine distance ascending -- highest sim first.

    When to prefer ``query`` instead
    --------------------------------
    Semantic search is good at recall but bad at tie-breaking when the
    topic is over-represented in the corpus. If you are pinpointing a
    single known session (not a theme) and the subject is frequent --
    "the claude-sql session where I ran over 30 days", "the session
    where the test suite failed" -- a literal ILIKE on a distinctive
    token finds it in one hop:

        claude-sql query "SELECT DISTINCT session_id FROM messages_text
          WHERE text_content ILIKE '%--since-days 30%'"

    Good distinctive tokens: exact CLI flags, dollar amounts from a
    dry-run cost table, precise error strings, the exact command the
    user ran. If the first search returns >3 plausible sessions at
    similar ``sim``, stop rephrasing and switch modality.

    Exit codes: 0 success, 2 no_embeddings, 70 runtime.
    """
    from claude_sql.application.use_cases import embed as _embed_use_case
    from claude_sql.infrastructure.session_search import DuckDbSessionSearch

    _configure(common)
    settings = _apply_embedding_provider(_resolve_settings(common), embedding_provider)
    fmt = _fmt(common)

    # Route the hand-rolled cosine-kNN SQL through the DuckDbSessionSearch
    # adapter (SessionSearchPort). The one CLI-specific twist is the query
    # embedding: it must flow through the ``embed_query`` USE-CASE (not
    # build_embedder directly) so the deferred-import monkeypatch seam in
    # test_search stays effective. Subclassing to override ``embed_query``
    # keeps ``_embedder=None`` so the store fail-loud guard's expected identity
    # remains exactly ``settings.expected_embedding_identity()`` (dim=None for
    # the probe-only local providers) — identical to the pre-adapter behavior.
    class _CliSearch(DuckDbSessionSearch):
        @override
        def embed_query(self, text: str) -> list[float]:
            return _embed_use_case.embed_query(text, settings=settings)

    searcher = _CliSearch(settings)
    try:
        hits = run_or_die(lambda: searcher.search(query_text, k=k), fmt=fmt)
        if not hits:
            # Empty embeddings store (adapter returns [] rather than exiting).
            logger.error("No embeddings yet. Run: claude-sql embed --since-days 7")
            sys.exit(EXIT_CODES["no_embeddings"])

        df = pl.DataFrame(
            {
                "uuid": [h.uuid for h in hits],
                "session_id": [h.session_id for h in hits],
                "role": [h.role for h in hits],
                "sim": [h.cosine_sim for h in hits],
                "snippet": [h.snippet for h in hits],
            }
        )
        emit_dataframe(df, fmt, table_rows=k, table_str_len=200)
    finally:
        searcher.close()


@app.command
def classify(
    *,
    since_days: int | None = None,
    limit: int | None = None,
    dry_run: bool = True,
    no_thinking: bool = False,
    common: Common | None = None,
) -> None:
    """Classify sessions with Sonnet 4.6: autonomy tier, work category, success, goal.

    Output columns (``session_classifications`` view)
    -------------------------------------------------
    session_id, autonomy_tier ∈ {autonomous,assisted,manual},
    work_category (sde/admin/strategy_business/thought_leadership/other),
    success ∈ {success,partial,failure,unknown}, goal (string),
    confidence ∈ [0,1], classified_at.
    Alias columns added by the view layer: ``autonomy``,
    ``success_outcome``, ``category`` (same values as above).

    Cost (defaults to --dry-run)
    ----------------------------
    Back-of-envelope ~8K input + ~300 output tokens per session. With
    Sonnet 4.6 pricing, 1,000 sessions ≈ $25-30. Always start with
    ``--dry-run`` (default) to see the plan JSON, then confirm with
    ``--no-dry-run``.

    Flags
    -----
    --since-days N   Only classify sessions newer than N days.
    --limit N        Cap at N sessions this run.
    --dry-run        (DEFAULT) emit plan JSON, no Bedrock calls.
    --no-dry-run     Spend real money.
    --no-thinking    Disable Sonnet adaptive thinking (cheaper, less precise).
    --glob PATTERN   Narrow the corpus (recommended for first runs).

    Dry-run stdout JSON
    -------------------
    ``{"pipeline":"classify","candidates":N,"llm_calls":N,
       "avg_input_tokens":8000,"avg_output_tokens":300,
       "estimated_cost_usd":X,"model":"...","thinking":"adaptive",
       "since_days":null,"limit":null,"dry_run":true}``

    Checkpointing
    -------------
    Session-level checkpoint in ``~/.claude/claude_sql.duckdb`` means
    reruns on unchanged sessions are free -- only sessions whose JSONL
    mtime changed are re-processed.
    """
    from claude_sql.application.use_cases.classify import classify_sessions

    _configure(common)
    settings = _resolve_settings(common)
    con = _open_connection_full(settings)
    try:
        result = classify_sessions(
            con,
            settings,
            since_days=since_days,
            limit=limit,
            dry_run=dry_run,
            no_thinking=no_thinking,
        )
        logger.info("classify: {} sessions processed (dry_run={})", result, dry_run)
        _emit_worker_result(result, common, pipeline="classify")
    finally:
        con.close()


@app.command
def trajectory(
    *,
    since_days: int | None = None,
    limit: int | None = None,
    dry_run: bool = True,
    no_thinking: bool = False,
    llm_analytics_provider: str | None = None,
    common: Common | None = None,
) -> None:
    """Per-session windowed sentiment + topic-transition classification (structured output).

    Output columns (``message_trajectory`` view)
    --------------------------------------------
    One row per text turn, keyed on (prev_uuid, curr_uuid):
    prev_sentiment / curr_sentiment ∈ {positive,neutral,negative},
    delta ∈ {-2..2}, is_transition (boolean), transition_kind
    (6-value enum), confidence ∈ [0,1], classified_at.
    Alias columns: ``sentiment`` (= curr_sentiment), ``transition``
    (= is_transition).

    Pipeline
    --------
    Adjacent text-turn pairs are batched ≤16 windows per structured-output
    call. The host verifies completeness by uuid-pair echo and stamps
    neutral placeholders on persistent misses.

    Provider (Wave D, opt-in)
    -------------------------
    --llm-analytics-provider {sonnet-bedrock,strands-luna}
        Which structured-output backend to run. Default ``sonnet-bedrock``
        (unchanged behavior). ``strands-luna`` routes through GPT-5.6-Luna
        on the Bedrock Mantle Responses API (needs the [llm-analytics]
        extra: ``uv add 'claude-sql[llm-analytics]'``).

    Cost: defaults to ``--dry-run``.

    Flags / exit codes otherwise identical to ``classify``.  See its help
    for the dry-run JSON schema.
    """
    from claude_sql.application.use_cases.trajectory import trajectory_messages

    _configure(common)
    settings = _apply_llm_analytics_provider(_resolve_settings(common), llm_analytics_provider)
    con = _open_connection_full(settings)
    try:
        result = trajectory_messages(
            con,
            settings,
            since_days=since_days,
            limit=limit,
            dry_run=dry_run,
            no_thinking=no_thinking,
        )
        logger.info("trajectory: {} messages processed (dry_run={})", result, dry_run)
        _emit_worker_result(result, common, pipeline="trajectory")
    finally:
        con.close()


@app.command
def conflicts(
    *,
    since_days: int | None = None,
    limit: int | None = None,
    dry_run: bool = True,
    no_thinking: bool = False,
    common: Common | None = None,
) -> None:
    """Per-session stance-conflict detection via Sonnet 4.6.

    What it finds
    -------------
    Places where the user and the agent disagreed on approach or scope,
    or where the agent contradicted itself. Each conflict gets two stance
    snippets (``stance_a`` / ``stance_b``), a resolution label
    ∈ {resolved, unresolved, abandoned, null}, and a detected_at timestamp.

    Output columns (``session_conflicts`` view)
    -------------------------------------------
    session_id, conflict_idx, stance_a, stance_b, resolution,
    detected_at, empty. Alias: ``conflict_resolution`` = resolution.

    Cost: defaults to ``--dry-run``. ~6K input / 400 output tokens / session.
    Flags / exit codes identical to ``classify``.
    """
    from claude_sql.application.use_cases.conflicts import detect_conflicts

    _configure(common)
    settings = _resolve_settings(common)
    con = _open_connection_full(settings)
    try:
        result = detect_conflicts(
            con,
            settings,
            since_days=since_days,
            limit=limit,
            dry_run=dry_run,
            no_thinking=no_thinking,
        )
        logger.info("conflicts: {} sessions processed (dry_run={})", result, dry_run)
        _emit_worker_result(result, common, pipeline="conflicts")
    finally:
        con.close()


@app.command
def friction(
    *,
    since_days: int | None = None,
    limit: int | None = None,
    dry_run: bool = True,
    no_thinking: bool = False,
    common: Common | None = None,
) -> None:
    """Classify short user messages (≤300 chars) for friction signals.

    Labels
    ------
    status_ping / unmet_expectation / confusion / interruption /
    correction / frustration / none.

    Pipeline
    --------
    1. Pull user-role messages ≤ ``CLAUDE_SQL_FRICTION_MAX_CHARS`` (300).
    2. Regex fast-path catches ``status_ping`` / ``interruption`` /
       ``correction`` at 0.9 confidence.
    3. Everything else → Sonnet 4.6 with the USER_FRICTION_SCHEMA.

    Output columns (``user_friction`` view)
    ---------------------------------------
    uuid, session_id, ts, label, source ∈ {regex, llm, refused},
    confidence, rationale, text (the original user message).

    Cost: defaults to ``--dry-run``. Short prompts (~200 in / 60 out),
    so even 10K candidates cost ≈ $3-4.
    Flags / exit codes identical to ``classify``.
    """
    from claude_sql.application.use_cases.friction import detect_user_friction

    _configure(common)
    settings = _resolve_settings(common)
    con = _open_connection_full(settings)
    try:
        result = detect_user_friction(
            con,
            settings,
            since_days=since_days,
            limit=limit,
            dry_run=dry_run,
            no_thinking=no_thinking,
        )
        logger.info("friction: {} rows written (dry_run={})", result, dry_run)
        _emit_worker_result(result, common, pipeline="friction")
    finally:
        con.close()


@app.command
def ingest(
    *,
    since_days: int | None = None,
    limit: int | None = None,
    dry_run: bool = True,
    common: Common | None = None,
) -> None:
    """Stamp every message with ``approx_tokens`` / ``simhash64`` / canonical_uuid.

    Pipeline
    --------
    1. Pull messages_text rows whose ``uuid`` is not in ``ingest_stamps``.
    2. Compute ``approx_tokens`` (cl100k × 0.78 Anthropic ratio) and
       ``simhash64`` (blake2b over word 3-grams) for each.
    3. Write to ``~/.claude/ingest_stamps/part-<ts>.parquet`` shards.
    4. Run ``canonical_uuid_resolve`` (DuckDB self-join with bit_count(xor)
       ≤ 3 over a top-16-bit bucket) and rewrite the parquet with the
       resolved ``canonical_uuid`` column populated.

    Cost: zero (CPU-only, no Bedrock).  Defaults to ``--dry-run`` so an
    agent can preview the pending count before writing anything.

    Flags
    -----
    --since-days N      Only stamp messages newer than N days.
    --limit N           Cap stamped rows this run.
    --dry-run           (DEFAULT) emit plan JSON, no parquet writes.
    --no-dry-run        Stamp + resolve.

    Dry-run output (stdout JSON)
    ----------------------------
    ``{"pipeline":"ingest","candidates":N,"since_days":...,"limit":...,
       "dry_run":true}``

    Real-run output: ``{"pipeline":"ingest","rows_processed":N,
    "dry_run":false}``.
    """
    from claude_sql.application.use_cases.ingest import (
        count_pending as _ingest_count_pending,
        resolve_canonicals as _ingest_resolve_canonicals,
        stamp_messages as _ingest_stamp_messages,
    )

    _configure(common)
    settings = _resolve_settings(common)
    con = _open_connection_full(settings)
    try:
        if dry_run:
            n = _ingest_count_pending(con, settings, since_days=since_days, limit=limit)
            logger.info("ingest --dry-run: {} pending rows", n)
            _emit_worker_result(
                {
                    "pipeline": "ingest",
                    "candidates": n,
                    "since_days": since_days,
                    "limit": limit,
                    "dry_run": True,
                },
                common,
                pipeline="ingest",
            )
            return
        stamped = _ingest_stamp_messages(con, settings, since_days=since_days, limit=limit)
        # Re-bind the analytics views so resolve_canonicals (and any later
        # query in the same process) sees the freshly-written shards.
        _refresh_analytics_views(con, settings)
        resolved = _ingest_resolve_canonicals(con, settings)
        logger.info("ingest: stamped={} resolved={}", stamped, resolved)
        _emit_worker_result(stamped, common, pipeline="ingest")
    finally:
        con.close()


@app.command
def cluster(*, force: bool = False, common: Common | None = None) -> None:
    """Cluster message embeddings with UMAP (8D) + HDBSCAN. Writes clusters.parquet.

    Prereq
    ------
    The embeddings parquet must exist. Run ``embed --no-dry-run`` first.

    Output columns (``message_clusters`` view)
    ------------------------------------------
    uuid, cluster_id (int; -1 = noise), probability (HDBSCAN soft label).

    Cost: zero (CPU-only, no Bedrock). Seeded by ``CLAUDE_SQL_SEED=42`` so
    cluster IDs are stable across reruns unless the embedding set changes.

    Flags
    -----
    --force   Re-cluster even if clusters.parquet already exists.
    """
    from claude_sql.application.use_cases.cluster import run_clustering

    _configure(common)
    settings = _resolve_settings(common)
    stats = run_clustering(settings, force=force)
    logger.info(
        "cluster: {} messages, {} clusters, {} noise ({:.1%})",
        stats["total"],
        stats["clusters"],
        stats["noise"],
        stats["noise"] / stats["total"] if stats["total"] else 0,
    )


@app.command
def terms(*, force: bool = False, common: Common | None = None) -> None:
    """Compute c-TF-IDF per-cluster term labels; writes cluster_terms.parquet.

    Prereq: ``cluster`` (i.e., clusters.parquet must exist).

    Output columns (``cluster_terms`` view)
    ---------------------------------------
    cluster_id (int), term (unigram or bigram), weight (float),
    rank (int, 1 = strongest term in that cluster).

    Math: per-class TF → IDF → L1 normalize, ngram (1,2), min_df=2.
    Cost: zero (sklearn CountVectorizer). See CLAUDE.md for design rationale.

    Flags
    -----
    --force   Recompute even if cluster_terms.parquet already exists.
    """
    from claude_sql.application.use_cases.terms import run_terms

    _configure(common)
    settings = _resolve_settings(common)
    con = _open_connection_full(settings)
    try:
        tstats = run_terms(con, settings, force=force)
        logger.info(
            "terms: {} clusters, {} term-rows",
            tstats["clusters"],
            tstats["terms"],
        )
    finally:
        con.close()


@app.command
def community(
    *,
    force: bool = False,
    gamma: float | None = None,
    resolution: ResolutionLevel = "medium",
    neighbors_of_session: Annotated[str | None, Parameter(name=["--neighbors-of"])] = None,
    top_k: int = 15,
    dry_run: bool = False,
    common: Common | None = None,
) -> None:
    """Session-level Leiden+CPM community detection over a mutual-kNN cosine graph.

    Prereq: ``embed`` (needs the embeddings parquet).

    Output columns (``session_communities`` view)
    ---------------------------------------------
    session_id, community_id (int; -1 = noise / sub-min-size),
    size (int), is_medoid (bool — best representative session of its
    community), coherence (float — mean intra-community cosine),
    gamma_used (float — the CPM γ used at run time).

    Sidecar (``community_profile`` view, written when auto-γ runs)
    --------------------------------------------------------------
    gamma, n_communities, quality, plateau_length — one row per γ tested
    by ``leidenalg.Optimiser.resolution_profile``.  Lets the agent ask
    "what γ would yield 50 communities?" without rerunning Leiden.

    Method: build a session-centroid mutual-kNN graph (k=15, edge floor 0.3
    by default), then ``leidenalg.find_partition`` with
    ``CPMVertexPartition``.  CPM γ is auto-picked from the resolution
    profile via the longest-plateau heuristic (Traag et al.); the
    ``--resolution {coarse, medium, fine}`` flag picks alternate plateaus
    of the same profile (no extra Leiden runs).

    Cost: zero (CPU only).  Seeded by ``CLAUDE_SQL_SEED=42``.  For top
    terms per community, run
    ``claude-sql query "SELECT * FROM community_top_topics(<cid>, 10)"``.

    Flags
    -----
    --force                 Re-detect even if session_communities.parquet exists.
    --gamma FLOAT           Explicit CPM γ; skips the resolution profile + sidecar.
                            Mutually exclusive with --resolution / --force / --neighbors-of.
    --resolution {coarse,medium,fine}
                            Pick a γ plateau without specifying a value.
                            Default 'medium' = longest plateau.  Ignored if --gamma set.
    --neighbors-of SID      Early-return path: skip Leiden, return top-k cosine
                            neighbors of SID.  Reads centroids on the fly +
                            joins session_communities.parquet if it exists.
    --top-k N               Used with --neighbors-of (default 15).
    --dry-run               Plan-only: count candidate sessions via SQL, do not
                            run Leiden.  Honors agent JSON output for free.

    Exit codes
    ----------
    0   success
    64  invalid input (e.g., --neighbors-of combined with partition flags)
    """
    from claude_sql.application.use_cases.community import (
        count_candidate_sessions,
        neighbors_of,
        run_communities,
    )

    _configure(common)
    settings = _resolve_settings(common)
    fmt = _fmt(common)

    if neighbors_of_session is not None and (gamma is not None or force or dry_run):
        err = ClassifiedError(
            kind="invalid_input",
            exit_code=EXIT_CODES["invalid_input"],
            message=(
                "--neighbors-of is mutually exclusive with --gamma, --force, "
                "and --dry-run; pass only --neighbors-of and --top-k."
            ),
            hint="Run `claude-sql community --neighbors-of <sid> --top-k 15` alone.",
        )
        emit_error(err, fmt)
        sys.exit(err.exit_code)

    con = _open_connection_full(settings)
    try:
        if neighbors_of_session is not None:
            df = neighbors_of(con, settings, neighbors_of_session, top_k=top_k)
            emit_dataframe(df, fmt, table_rows=top_k)
            return

        if dry_run:
            n = count_candidate_sessions(con)
            plan: dict[str, object] = {
                "pipeline": "community",
                "candidate_sessions": n,
                "knn_k": settings.leiden_knn_k,
                "edge_floor": settings.leiden_edge_floor,
                "min_community_size": settings.leiden_min_community_size,
                "gamma": gamma if gamma is not None else "auto",
                "resolution": resolution,
                "would_write": [
                    str(settings.communities_parquet_path),
                ]
                + ([] if gamma is not None else [str(settings.community_profile_parquet_path)]),
                "dry_run": True,
            }
            emit_json(plan, fmt)
            return

        stats = run_communities(
            con,
            settings,
            force=force,
            gamma=gamma,
            resolution=resolution,
        )
        import math

        quality_val = stats["quality"]
        quality_log = (
            quality_val if isinstance(quality_val, float) and not math.isnan(quality_val) else 0.0
        )
        logger.info(
            "community: {} sessions, {} communities (γ={:.4f}, quality={:.4f})",
            stats["sessions"],
            stats["communities"],
            stats["gamma_used"],
            quality_log,
        )
        _emit_worker_result(stats, common, pipeline="community")
    finally:
        con.close()


@app.command
def analyze(
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
    embedding_provider: str | None = None,
    llm_analytics_provider: str | None = None,
    common: Common | None = None,
) -> None:
    """Run the full analytics pipeline end-to-end: embed → structure → LLM analytics.

    Stages (in order)
    -----------------
    0. skills sync  (filesystem walk; zero-cost; produces skills_catalog.parquet)
    1. ingest       (tiktoken + blake2b SimHash; zero-cost; honors --dry-run)
    2. embed        (Bedrock Cohere Embed v4; honors --dry-run)
    3. cluster      (UMAP+HDBSCAN; zero-cost; --force_cluster to rebuild)
    4. terms        (c-TF-IDF labels for clusters; zero-cost)
    5. community    (Leiden+CPM; zero-cost; --force-community to rebuild)
    6. classify     (Sonnet 4.6; honors --dry-run)
    7. trajectory   (Sonnet 4.6; honors --dry-run)
    8. conflicts    (Sonnet 4.6; honors --dry-run)
    9. friction     (Sonnet 4.6; honors --dry-run)

    Cost
    ----
    Every LLM-touching stage defaults to ``--dry-run`` -- stdout logs the
    plan per stage. Pass ``--no-dry-run`` to execute for real.

    Flags
    -----
    --since-days N         Scope all stages to the last N days (default 30).
    --limit N              Cap each LLM stage at N items.
    --dry-run / --no-dry-run  (default --dry-run)
    --no-thinking          Disable Sonnet adaptive thinking across all stages.
    --skip-<stage>         Drop a stage:
                           ingest, embed, cluster, community, classify,
                           trajectory, conflicts, friction. Terms is bound
                           to cluster.
    --force-cluster        Rebuild clusters.parquet (+ terms) even if present.
    --force-community      Rebuild session_communities.parquet even if present.
    --embedding-provider {cohere-bedrock,ollama,onnx-bge}
                           Override the embedding backend for the embed stage.
    --llm-analytics-provider {sonnet-bedrock,strands-luna}
                           Structured-output backend for the trajectory stage.
                           Default sonnet-bedrock (unchanged); strands-luna is
                           the opt-in GPT-5.6-Luna path ([llm-analytics] extra).
    --glob / --subagent-glob  Narrow the corpus (applies to every stage).

    Typical recipes
    ---------------
    Preview spend over the last week::

        claude-sql analyze --since-days 7

    Run the non-LLM stages only (cluster + terms + community)::

        claude-sql analyze --skip-embed --skip-classify \
            --skip-trajectory --skip-conflicts --skip-friction \
            --force-cluster --force-community
    """
    from claude_sql.application.analyze import run_analyze

    _configure(common)
    settings = _apply_embedding_provider(_resolve_settings(common), embedding_provider)
    settings = _apply_llm_analytics_provider(settings, llm_analytics_provider)

    # Delegate the stage chain to the application layer. The connection factory
    # and the two rebind-lifecycle seams are threaded through as THIS module's
    # attributes (``_open_connection_full`` / ``_refresh_analytics_views`` /
    # ``_rebind_vss``) so the module-object monkeypatches in the test suite keep
    # biting through the injected references (see test_analyze_chain).
    run_analyze(
        settings,
        since_days=since_days,
        limit=limit,
        dry_run=dry_run,
        no_thinking=no_thinking,
        skip_ingest=skip_ingest,
        skip_embed=skip_embed,
        skip_classify=skip_classify,
        skip_trajectory=skip_trajectory,
        skip_conflicts=skip_conflicts,
        skip_friction=skip_friction,
        skip_cluster=skip_cluster,
        skip_community=skip_community,
        skip_skills_sync=skip_skills_sync,
        force_cluster=force_cluster,
        force_community=force_community,
        open_connection=_open_connection_full,
        refresh_fn=_refresh_analytics_views,
        rebind_fn=_rebind_vss,
    )


@app.default
def _default(*, common: Common | None = None) -> None:
    """Print a hint when ``claude-sql`` is invoked without a subcommand."""
    del common
    print("claude-sql - pass a subcommand or --help")
    print("  schema | query | explain | shell | list-cache | peek")
    print("  ingest | embed | search")
    print("  classify | trajectory | conflicts | friction | cluster | terms | community | analyze")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point wired into ``[project.scripts]`` in ``pyproject.toml``."""
    app()


if __name__ == "__main__":
    main()

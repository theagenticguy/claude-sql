"""Cyclopts CLI entry point for ``claude-sql``.

Wires the ``claude-sql`` console script to its thirteen subcommands.  Shared
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

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import duckdb
import polars as pl
from cyclopts import App, Parameter
from loguru import logger

from claude_sql import (
    binding as _binding,
    blind_handover as _blind_handover,
    checkpointer,
    freeze as _freeze,
    judge_worker as _judge_worker,
    judges as _judge_catalog,
    kappa_worker as _kappa_worker,
    skills_catalog as _skills_catalog,
    ungrounded_worker as _ungrounded_worker,
)
from claude_sql.cluster_worker import run_clustering
from claude_sql.community_worker import (
    ResolutionLevel,
    neighbors_of,
    run_communities,
)
from claude_sql.config import Settings
from claude_sql.embed_worker import embed_query, run_backfill
from claude_sql.friction_worker import detect_user_friction
from claude_sql.install_source import format_version
from claude_sql.llm_worker import classify_sessions, detect_conflicts, trajectory_messages
from claude_sql.logging_setup import configure_logging
from claude_sql.output import (
    EXIT_CODES,
    ClassifiedError,
    InputValidationError,
    OutputFormat,
    emit_dataframe,
    emit_error,
    emit_json,
    resolve_format,
    run_or_die,
    validate_glob,
)
from claude_sql.parquet_shards import (
    count_rows,
    is_sharded_dir,
    iter_part_files,
)
from claude_sql.review_sheet_render import render_markdown, render_refusal_markdown
from claude_sql.review_sheet_worker import generate_review_sheet
from claude_sql.sql_views import (
    MACRO_NAMES,
    MACRO_SIGNATURES,
    VIEW_NAMES,
    VIEW_SCHEMA,
    _parquet_is_populated,
    register_all,
    register_raw,
    register_views,
)
from claude_sql.terms_worker import run_terms

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
    --glob "/home/you/.claude/projects/-efs-you-workplace-bonk/*.jsonl"
At most one '**' segment is allowed per pattern (DuckDB limitation) -- the
CLI rejects multi-star globs with a clear hint before DuckDB sees them.
"""


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


_PERCENT_LIMIT_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%\s*$")


def _resolve_memory_limit(limit: str) -> str:
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


def _apply_duckdb_pragmas(con: duckdb.DuckDBPyConnection, settings: Settings) -> None:
    """Set the tuning PRAGMAs both connection helpers share.

    Centralized so :func:`_open_connection_full` and
    :func:`_open_connection_introspect` stay in sync. Threads, memory_limit,
    and temp_directory all come from ``settings``; the spill directory is
    materialized on disk before DuckDB sees the path because DuckDB will
    happily fail later when it tries to write a spill file.
    """
    settings.duckdb_temp_dir.mkdir(parents=True, exist_ok=True)
    memory_limit = _resolve_memory_limit(settings.duckdb_memory_limit)
    con.execute(f"SET threads = {int(settings.duckdb_threads)}")
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET temp_directory = '{settings.duckdb_temp_dir}'")
    con.execute("SET enable_object_cache = true")
    con.execute("SET preserve_insertion_order = false")


def _open_connection_full(settings: Settings) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with every claude-sql object wired.

    Tuning PRAGMAs are set before view registration so the registration
    queries themselves benefit from the higher thread count and the spill
    directory pointed at real disk (Amazon devboxes ship ``/tmp`` as a
    4 GB tmpfs that thrashes once a clustering run starts spilling).
    """
    con = duckdb.connect(":memory:")
    _apply_duckdb_pragmas(con, settings)
    register_all(con, settings=settings)
    return con


def _open_connection_introspect(settings: Settings) -> duckdb.DuckDBPyConnection:
    """Bare DuckDB connection — PRAGMAs only, no view/macro registration.

    For commands that don't need the catalog (``schema`` reads the static
    :data:`VIEW_SCHEMA` dict; trivial scalar queries like ``SELECT 1`` or
    ``SELECT current_timestamp`` don't reference any view). Returning a
    bare connection avoids the ~25 s :func:`register_all` chain entirely.
    """
    con = duckdb.connect(":memory:")
    _apply_duckdb_pragmas(con, settings)
    return con


def _sql_uses_catalog(sql: str) -> bool:
    """Cheap pre-flight: does ``sql`` reference any registered view/macro?

    Substring-matches against ``VIEW_NAMES + MACRO_NAMES`` (case-insensitive).
    False positives (a string literal containing ``'sessions'``) just trigger
    the slow path — no correctness regression. False negatives can't happen
    if the user genuinely references a view: the name has to appear in the
    SQL text.
    """
    lowered = sql.lower()
    return any(name.lower() in lowered for name in (*VIEW_NAMES, *MACRO_NAMES))


# Maintained for backwards compatibility with tests that still call
# ``cli._open_connection``. New code should pick the explicit variant.
_open_connection = _open_connection_full


def _emit_worker_result(result: int | dict, common: Common | None, pipeline: str) -> None:
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
    """Report the persistent DuckDB checkpoint file alongside the parquet caches.

    Keeps the same ``{name, path, exists[, bytes, mtime, rows]}`` shape as
    :func:`_describe_cache_entry` so ``list-cache`` stays homogeneous.  Row
    count is queried via :func:`checkpointer.count_rows`.
    """
    exists = path.exists() and path.is_file()
    entry: dict[str, object] = {"name": "session_checkpoint", "path": str(path), "exists": exists}
    if not exists:
        return entry
    stat = path.stat()
    entry["bytes"] = stat.st_size
    entry["mtime"] = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
    try:
        entry["rows"] = checkpointer.count_rows(path)
    except duckdb.Error:
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
    profiling_dir = Path(os.path.expanduser("~/.claude/profiling/"))
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
            --glob "/home/you/.claude/projects/-efs-you-bonk/*.jsonl"
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
        _open_connection_full(settings)
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
        _open_connection_full(settings)
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
    entries = [
        _describe_cache_entry("embeddings", settings.embeddings_parquet_path),
        _describe_cache_entry("session_classifications", settings.classifications_parquet_path),
        _describe_cache_entry("message_trajectory", settings.trajectory_parquet_path),
        _describe_cache_entry("session_conflicts", settings.conflicts_parquet_path),
        _describe_cache_entry("message_clusters", settings.clusters_parquet_path),
        _describe_cache_entry("cluster_terms", settings.cluster_terms_parquet_path),
        _describe_cache_entry("session_communities", settings.communities_parquet_path),
        _describe_cache_entry("community_profile", settings.community_profile_parquet_path),
        _describe_cache_entry("user_friction", settings.user_friction_parquet_path),
        _describe_cache_entry("skills_catalog", settings.skills_catalog_parquet_path),
        _describe_checkpoint_entry(settings.checkpoint_db_path),
    ]

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
    common: Common | None = None,
) -> None:
    """Embed new messages with Cohere Embed v4 and append to the embeddings parquet.

    Cost
    ----
    Calls Bedrock (``global.cohere.embed-v4:0``) on every unembedded
    message. ``--dry-run`` is OFF by default here (unlike LLM workers);
    pass it if you only want to see the plan.

    Flags
    -----
    --since-days N     Only consider messages newer than N days.
    --limit N          Cap the number of messages embedded this run.
    --dry-run          Preview only; emit plan JSON, no Bedrock calls.
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

    _configure(common)
    settings = _resolve_settings(common)
    con = duckdb.connect(":memory:")
    try:
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
    _configure(common)
    settings = _resolve_settings(common)
    fmt = _fmt(common)
    con = _open_connection_full(settings)
    try:
        row = con.execute("SELECT count(*) FROM message_embeddings").fetchone()
        count = int(row[0]) if row else 0
        if count == 0:
            logger.error("No embeddings yet. Run: claude-sql embed --since-days 7")
            sys.exit(EXIT_CODES["no_embeddings"])

        qv = embed_query(query_text, settings=settings)
        dim = int(settings.output_dimension)
        # Rank by cosine similarity descending.  The HNSW index was built with
        # metric='cosine', so ORDER BY array_cosine_distance (== 1 - sim) ASC
        # is what triggers the index lookup.  Using array_distance here (L2)
        # would silently bypass the index AND give wrong ranks because the
        # raw int8-cast-to-float document vectors have magnitudes in the
        # thousands while the query vector is unit-normalized.
        df = run_or_die(
            lambda: con.execute(
                f"""
                WITH qv AS (SELECT CAST(? AS FLOAT[{dim}]) AS v)
                SELECT CAST(mt.uuid AS VARCHAR)  AS uuid,
                       CAST(mt.session_id AS VARCHAR) AS session_id,
                       mt.role,
                       array_cosine_similarity(me.embedding, (SELECT v FROM qv)) AS sim,
                       substr(mt.text_content, 1, 200) AS snippet
                FROM message_embeddings me
                JOIN messages_text mt ON CAST(mt.uuid AS VARCHAR) = me.uuid
                ORDER BY array_cosine_distance(me.embedding, (SELECT v FROM qv)) ASC
                LIMIT ?
                """,
                [qv, k],
            ).pl(),
            fmt=fmt,
        )
        emit_dataframe(df, fmt, table_rows=k, table_str_len=200)
    finally:
        con.close()


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
    common: Common | None = None,
) -> None:
    """Per-message sentiment + topic-transition classification (regex prefilter → Sonnet 4.6).

    Output columns (``message_trajectory`` view)
    --------------------------------------------
    uuid, sentiment_delta ∈ {positive,neutral,negative},
    is_transition (boolean -- does this message mark a topic shift?),
    confidence ∈ [0,1], classified_at.
    Alias columns: ``sentiment`` (same as sentiment_delta),
    ``transition`` (same as is_transition).

    Pipeline
    --------
    1. Regex prefilter catches ~50% of obvious transitions for free.
    2. Sonnet 4.6 classifies the remainder with structured output.

    Cost: defaults to ``--dry-run``. ~500 input / 50 output tokens per LLM
    call.

    Flags / exit codes identical to ``classify``.  See its help for the
    dry-run JSON schema.
    """
    _configure(common)
    settings = _resolve_settings(common)
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
            row = con.execute(
                """
                SELECT COUNT(DISTINCT m.session_id) AS candidate_sessions
                  FROM read_parquet(?) e
                  JOIN messages m
                    ON CAST(m.uuid AS VARCHAR) = e.uuid
                """,
                [str(settings.embeddings_parquet_path)],
            ).fetchone()
            n = int(row[0]) if row else 0
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
    common: Common | None = None,
) -> None:
    """Run the full analytics pipeline end-to-end: embed → structure → LLM analytics.

    Stages (in order)
    -----------------
    0. skills sync  (filesystem walk; zero-cost; produces skills_catalog.parquet)
    1. embed        (Bedrock Cohere Embed v4; honors --dry-run)
    2. cluster      (UMAP+HDBSCAN; zero-cost; --force_cluster to rebuild)
    3. terms        (c-TF-IDF labels for clusters; zero-cost)
    4. community    (Leiden+CPM; zero-cost; --force-community to rebuild)
    5. classify     (Sonnet 4.6; honors --dry-run)
    6. trajectory   (Sonnet 4.6; honors --dry-run)
    7. conflicts    (Sonnet 4.6; honors --dry-run)
    8. friction     (Sonnet 4.6; honors --dry-run)

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
                           embed, cluster, community, classify, trajectory,
                           conflicts, friction. Terms is bound to cluster.
    --force-cluster        Rebuild clusters.parquet (+ terms) even if present.
    --force-community      Rebuild session_communities.parquet even if present.
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
    import asyncio

    _configure(common)
    settings = _resolve_settings(common)

    # 0. Skills catalog sync (filesystem walk, zero cost).  Runs even in
    # --dry-run because it does not hit Bedrock; opt out via
    # --skip-skills-sync if you want to keep the parquet frozen.
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

    # 1. Embed (reuses embed_worker).  Silently skipped if the parquet is up to date.
    if not skip_embed:
        con = _open_connection_full(settings)
        try:
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
        finally:
            con.close()

    # 2. Cluster (reads embeddings parquet, writes clusters.parquet).  Non-LLM.
    if not skip_cluster:
        stats = run_clustering(settings, force=force_cluster)
        logger.info(
            "analyze/cluster: {} messages, {} clusters, {} noise",
            stats["total"],
            stats["clusters"],
            stats["noise"],
        )
        con = _open_connection_full(settings)
        try:
            tstats = run_terms(con, settings, force=force_cluster)
            logger.info(
                "analyze/terms: {} clusters, {} term-rows",
                tstats["clusters"],
                tstats["terms"],
            )
        finally:
            con.close()

    # 3. Community detection (non-LLM, runs in parallel conceptually with cluster).
    if not skip_community:
        con = _open_connection_full(settings)
        try:
            cstats = run_communities(con, settings, force=force_community)
            logger.info(
                "analyze/community: {} sessions, {} communities",
                cstats["sessions"],
                cstats["communities"],
            )
        finally:
            con.close()

    # 4. Session classification (LLM).
    if not skip_classify:
        con = _open_connection_full(settings)
        try:
            n = classify_sessions(
                con,
                settings,
                since_days=since_days,
                limit=limit,
                dry_run=dry_run,
                no_thinking=no_thinking,
            )
            logger.info("analyze/classify: {} sessions (dry_run={})", n, dry_run)
        finally:
            con.close()

    # 5. Trajectory (LLM).
    if not skip_trajectory:
        con = _open_connection_full(settings)
        try:
            n = trajectory_messages(
                con,
                settings,
                since_days=since_days,
                limit=limit,
                dry_run=dry_run,
                no_thinking=no_thinking,
            )
            logger.info("analyze/trajectory: {} messages (dry_run={})", n, dry_run)
        finally:
            con.close()

    # 6. Conflicts (LLM, requires full session context).
    if not skip_conflicts:
        con = _open_connection_full(settings)
        try:
            n = detect_conflicts(
                con,
                settings,
                since_days=since_days,
                limit=limit,
                dry_run=dry_run,
                no_thinking=no_thinking,
            )
            logger.info("analyze/conflicts: {} sessions (dry_run={})", n, dry_run)
        finally:
            con.close()

    # 7. Friction (LLM, short-message scope).
    if not skip_friction:
        con = _open_connection_full(settings)
        try:
            n = detect_user_friction(
                con,
                settings,
                since_days=since_days,
                limit=limit,
                dry_run=dry_run,
                no_thinking=no_thinking,
            )
            logger.info("analyze/friction: {} rows (dry_run={})", n, dry_run)
        finally:
            con.close()

    logger.info("analyze: done")


@app.command(name="judges")
def judges_cmd(*, common: Common | None = None) -> None:
    """List the cross-provider Bedrock judge catalog (shortname, model ID, family, notes)."""
    _configure(common)
    fmt = _fmt(common)
    rows = [
        {
            "shortname": j.shortname,
            "model_id": j.model_id,
            "provider": j.provider,
            "family": j.family,
            "role": j.role,
            "notes": j.notes,
        }
        for j in _judge_catalog.catalog()
    ]
    df = pl.DataFrame(rows)
    emit_dataframe(df, fmt=fmt)


@app.command(name="freeze")
def freeze_cmd(
    rubric: Path,
    /,
    *,
    panel: str,
    embed_model: str = "global.cohere.embed-v4:0",
    seed: int = 42,
    min_turns: int = 10,
    max_turns: int = 40,
    common: Common | None = None,
) -> None:
    """Pre-register a study: write an immutable manifest under ~/.claude/studies/<sha>/.

    ``panel`` is a comma-separated list of judge shortnames (see ``claude-sql
    judges``).  The returned SHA is what every downstream worker consumes.
    """
    _configure(common)
    fmt = _fmt(common)
    panel_list = [s.strip() for s in panel.split(",") if s.strip()]
    if not panel_list:
        raise InputValidationError("--panel must have at least one shortname")
    scope = _freeze.SessionScope(min_turns=min_turns, max_turns=max_turns)
    study = _freeze.freeze(
        rubric_path=rubric,
        panel_shortnames=tuple(panel_list),
        embed_model_id=embed_model,
        session_scope=scope,
        seed=seed,
    )
    emit_json(
        {
            "manifest_sha": study.manifest_sha,
            "rubric_path": study.rubric_path,
            "panel_shortnames": list(study.panel_shortnames),
            "commit_sha": study.commit_sha,
            "created_at_utc": study.created_at_utc,
        },
        fmt=fmt,
    )


@app.command(name="replay")
def replay_cmd(manifest_sha: str, /, *, common: Common | None = None) -> None:
    """Load and echo a frozen study manifest by SHA."""
    _configure(common)
    fmt = _fmt(common)
    study = _freeze.replay(manifest_sha)
    emit_json(study.to_dict(), fmt=fmt)


@app.command(name="blind-handover")
def blind_handover_cmd(
    input_path: Path,
    /,
    output_path: Path,
    *,
    common: Common | None = None,
) -> None:
    """Strip identity markers from a parquet of sessions for grader-safe handover.

    Input parquet must have (session_id, text) columns.  Writes the same
    parquet with text stripped and an ``original_hash`` column added.
    """
    _configure(common)
    df = pl.read_parquet(input_path)
    required = {"session_id", "text"}
    missing = required - set(df.columns)
    if missing:
        raise InputValidationError(f"input parquet missing columns: {sorted(missing)}")
    stripped = [_blind_handover.strip_text(t) for t in df["text"].to_list()]
    out = df.with_columns(
        pl.Series("text", [r.text for r in stripped]),
        pl.Series(
            "original_hash",
            [_blind_handover.original_hash(s) for s in df["session_id"].to_list()],
        ),
    )
    out.write_parquet(output_path)
    logger.info("blind-handover: wrote {} stripped rows to {}", out.height, output_path)


@app.command(name="judge")
def judge_cmd(
    manifest_sha: str,
    /,
    *,
    sessions_parquet: Path,
    output_parquet: Path,
    dry_run: bool = True,
    concurrency: int = 4,
    region: str = "us-east-1",
    common: Common | None = None,
) -> None:
    """Dispatch a frozen study's judge panel over a sessions parquet.

    ``sessions_parquet`` must have (session_id, text) columns.  Defaults to
    ``--dry-run`` per the project cost-guard convention.
    """
    _configure(common)
    fmt = _fmt(common)
    study = _freeze.replay(manifest_sha)
    df = pl.read_parquet(sessions_parquet)
    required = {"session_id", "text"}
    missing = required - set(df.columns)
    if missing:
        raise InputValidationError(f"sessions parquet missing columns: {sorted(missing)}")
    sessions = list(zip(df["session_id"].to_list(), df["text"].to_list(), strict=True))
    result = _judge_worker.run(
        sessions=sessions,
        panel_shortnames=list(study.panel_shortnames),
        rubric_yaml_path=Path(study.rubric_path),
        freeze_sha=study.manifest_sha,
        out_parquet=output_parquet,
        dry_run=dry_run,
        concurrency=concurrency,
        region=region,
    )
    if isinstance(result, _judge_worker.GradePlan):
        emit_json(
            {
                "dry_run": True,
                "n_sessions": result.n_sessions,
                "n_judges": result.n_judges,
                "n_axes": result.n_axes,
                "n_calls": result.n_calls,
                "est_input_tokens": result.est_input_tokens,
                "est_output_tokens": result.est_output_tokens,
                "est_usd": result.est_usd,
            },
            fmt=fmt,
        )
    else:
        emit_json({"dry_run": False, "n_scores": len(result), "out": str(output_parquet)}, fmt=fmt)


@app.command(name="ungrounded-claim")
def ungrounded_cmd(
    manifest_sha: str,
    /,
    *,
    turns_parquet: Path,
    output_parquet: Path,
    common: Common | None = None,
) -> None:
    """Run the ungrounded-claim detector over a turns parquet.

    ``turns_parquet`` needs (session_id, turn_idx, assistant_text,
    tool_output_text) columns.  Writes per-claim grounded flags.
    """
    _configure(common)
    fmt = _fmt(common)
    study = _freeze.replay(manifest_sha)
    df = pl.read_parquet(turns_parquet)
    required = {"session_id", "turn_idx", "assistant_text", "tool_output_text"}
    missing = required - set(df.columns)
    if missing:
        raise InputValidationError(f"turns parquet missing columns: {sorted(missing)}")
    turns = [
        _ungrounded_worker.Turn(
            session_id=row["session_id"],
            turn_idx=int(row["turn_idx"]),
            assistant_text=row["assistant_text"],
            tool_output_text=row["tool_output_text"],
        )
        for row in df.iter_rows(named=True)
    ]
    out = _ungrounded_worker.detect(turns, freeze_sha=study.manifest_sha)
    _ungrounded_worker.to_parquet(out, output_parquet)
    summary = _ungrounded_worker.summarize(out)
    emit_dataframe(summary, fmt=fmt)


@app.command(name="kappa")
def kappa_cmd(
    scores_parquet: Path,
    /,
    *,
    bootstrap: int = 1000,
    floor: float = 0.6,
    delta_gate: Path | None = None,
    common: Common | None = None,
) -> None:
    """Compute Cohen's + Fleiss' kappa with bootstrapped 95% CI.

    Exits non-zero (66) if any axis has Fleiss kappa below ``--floor`` OR
    if ``--delta-gate <prior.parquet>`` is set and the delta-kappa CI
    excludes zero on any axis (pre-registered stopping rule).
    """
    _configure(common)
    fmt = _fmt(common)
    df = _kappa_worker.load_scores(scores_parquet)
    pairs = _kappa_worker.compute_pairwise(df, n_bootstrap=bootstrap)
    fleiss = _kappa_worker.compute_fleiss(df, n_bootstrap=bootstrap)
    report = {
        "pairs": [
            {
                "axis": p.axis,
                "judge_a": p.judge_a,
                "judge_b": p.judge_b,
                "n_items": p.n_items,
                "kappa": round(p.kappa, 4),
                "ci_low": round(p.ci_low, 4),
                "ci_high": round(p.ci_high, 4),
            }
            for p in pairs
        ],
        "fleiss": [
            {
                "axis": f.axis,
                "n_judges": f.n_judges,
                "n_items": f.n_items,
                "kappa": round(f.kappa, 4),
                "ci_low": round(f.ci_low, 4),
                "ci_high": round(f.ci_high, 4),
                "below_floor": f.kappa < floor,
            }
            for f in fleiss
        ],
        "floor": floor,
    }
    any_gate_tripped = any(row["below_floor"] for row in report["fleiss"])
    if delta_gate is not None:
        prior_df = _kappa_worker.load_scores(delta_gate)
        prior_fleiss = {
            f.axis: f for f in _kappa_worker.compute_fleiss(prior_df, n_bootstrap=bootstrap)
        }
        delta_rows = []
        for cur in fleiss:
            prior = prior_fleiss.get(cur.axis)
            if prior is None:
                continue
            tripped = _kappa_worker.delta_gate_excludes_zero(cur, prior, n_bootstrap=bootstrap)
            delta_rows.append(
                {
                    "axis": cur.axis,
                    "delta_excludes_zero": tripped,
                    "current_kappa": cur.kappa,
                    "prior_kappa": prior.kappa,
                }
            )
            any_gate_tripped = any_gate_tripped or tripped
        report["delta_gate"] = delta_rows
    emit_json(report, fmt=fmt)
    if any_gate_tripped:
        sys.exit(66)


@app.command(name="bind")
def bind_cmd(
    *,
    repo: Path | None = None,
    commit_msg: Path | None = None,
    dry_run: bool = False,
    common: Common | None = None,
) -> None:
    """Attach the transcript-PR binding (trailers + git-notes JSON) to a commit.

    Pre-commit-hook entry point per RFC 0001 (see
    ``docs/rfc/0001-transcript-pr-binding.md``).  Wires into a
    ``prepare-commit-msg`` lefthook job so the trailer lands in the
    user's editor before they confirm the message.

    Discovery order for the commit-message file:

    1. ``--commit-msg PATH`` flag if set.
    2. ``GIT_PARAMS`` / ``$1`` from the hook -- we re-read it from
       the ``CLAUDE_SQL_BIND_COMMIT_MSG`` env var, which is the
       lefthook-friendly way to pass the hook's ``{0}`` arg through.
    3. ``<repo>/.git/COMMIT_EDITMSG`` as a last-ditch fallback.

    Resolves the active transcript via
    :func:`claude_sql.binding.find_active_transcript` (latest mtime
    under ``~/.claude/projects/<projectified-cwd>/*.jsonl``); when no
    transcript is found the command exits 0 cleanly without touching
    the message — bind is best-effort by design.

    With ``--dry-run`` (default ``False``), prints the planned
    binding as JSON and writes nothing.  Off ``--dry-run``, writes
    the three trailers in place and a JSON note under
    ``refs/notes/transcripts``.
    """
    _configure(common)
    fmt = _fmt(common)
    repo_path = repo.resolve() if repo is not None else _binding._resolve_repo(None)
    cwd = Path.cwd()
    transcript = _binding.find_active_transcript(cwd)
    if transcript is None:
        emit_json(
            {
                "bound": False,
                "reason": "no-active-transcript",
                "cwd": str(cwd),
                "projects_dir": f"~/.claude/projects/{_binding.projectify(cwd)}",
            },
            fmt=fmt,
        )
        return
    binding = _binding.build_binding(transcript_path=transcript)

    msg_path: Path | None = commit_msg
    if msg_path is None:
        env_path = os.environ.get("CLAUDE_SQL_BIND_COMMIT_MSG")
        if env_path:
            msg_path = Path(env_path)
    if msg_path is None:
        candidate = repo_path / ".git" / "COMMIT_EDITMSG"
        if candidate.exists():
            msg_path = candidate

    if dry_run:
        emit_json(
            {
                "bound": False,
                "dry_run": True,
                "transcript_path": str(transcript),
                "binding": binding.to_dict(),
                "note_payload": binding.to_note_payload(),
                "commit_msg_path": str(msg_path) if msg_path else None,
                "repo": str(repo_path),
            },
            fmt=fmt,
        )
        return

    if msg_path is None:
        err = ClassifiedError(
            kind="invalid_input",
            exit_code=EXIT_CODES["invalid_input"],
            message="no commit-message file found; pass --commit-msg or run from a prepare-commit-msg hook",
            hint="set --commit-msg PATH or CLAUDE_SQL_BIND_COMMIT_MSG=$1 in your hook",
        )
        emit_error(err, fmt)
        sys.exit(err.exit_code)

    try:
        _binding.write_trailer(msg_path, binding)
    except _binding.GitInvocationError as exc:
        err = ClassifiedError(
            kind="runtime_error",
            exit_code=EXIT_CODES["runtime_error"],
            message=f"git interpret-trailers failed: {exc.stderr.strip()}",
            hint=None,
        )
        emit_error(err, fmt)
        sys.exit(err.exit_code)

    # Note write is best-effort: we have a HEAD commit only when bind
    # runs *after* the commit (e.g., post-commit hook).  In a
    # prepare-commit-msg flow the commit doesn't exist yet, so we skip
    # the note here and the integration relies on a separate
    # post-commit step.  When the caller is invoking us with --commit
    # already created (e.g., backfill), they pass --no-dry-run with a
    # repo containing HEAD.
    head_cp = _binding._run_git(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
    )
    if head_cp.returncode == 0:
        commit_sha = head_cp.stdout.strip()
        try:
            _binding.write_note(repo_path, commit_sha, binding)
        except _binding.GitInvocationError as exc:
            logger.warning("git notes write failed (non-fatal): {}", exc.stderr.strip())
            commit_sha = ""
    else:
        commit_sha = ""

    emit_json(
        {
            "bound": True,
            "dry_run": False,
            "transcript_path": str(transcript),
            "binding": binding.to_dict(),
            "commit_msg_path": str(msg_path),
            "repo": str(repo_path),
            "commit_sha": commit_sha,
        },
        fmt=fmt,
    )


@app.command(name="resolve")
def resolve_cmd(
    commit_sha: str,
    /,
    *,
    repo: Path | None = None,
    all_sources: bool = False,
    common: Common | None = None,
) -> None:
    """Resolve a commit's bound transcript per RFC 0001 §Resolution precedence.

    Reads the ``Claude-Transcript-*`` trailers first; falls back to
    the JSON note under ``refs/notes/transcripts``; raises a loud
    error (exit 70) when both surfaces disagree on the digest.
    Returns the parsed binding as JSON.

    Flags
    -----
    --repo PATH
        Repository root.  Defaults to ``git rev-parse --show-toplevel``
        from the current cwd.
    --all-sources
        Return ``{"trailer": ..., "note": ...}`` instead of merging.
        Diagnostic flow for investigating mismatches; never raises on
        disagreement.

    Exit codes
    ----------
    * 0   binding resolved cleanly (or ``--all-sources`` returned both)
    * 2   commit has no binding (no trailer, no note)
    * 65  commit not found / git invocation failed
    * 70  trailer and note disagree on digest
    """
    _configure(common)
    fmt = _fmt(common)
    repo_path = repo.resolve() if repo is not None else None
    try:
        if all_sources:
            sources = _binding.resolve_all_sources(commit_sha, repo=repo_path)
            payload: dict[str, dict[str, str] | None] = {
                "trailer": sources["trailer"].to_dict() if sources["trailer"] is not None else None,
                "note": sources["note"].to_dict() if sources["note"] is not None else None,
            }
            emit_json(payload, fmt=fmt)
            return
        binding = _binding.resolve_commit_to_transcript(commit_sha, repo=repo_path)
    except _binding.BindingMismatchError as exc:
        err = ClassifiedError(
            kind="runtime_error",
            exit_code=EXIT_CODES["runtime_error"],
            message=str(exc),
            hint="run `claude-sql resolve <sha> --all-sources` to see both surfaces",
        )
        emit_error(err, fmt)
        sys.exit(err.exit_code)
    except LookupError as exc:
        err = ClassifiedError(
            kind="no_embeddings",  # re-uses the "absent-but-not-broken" kind
            exit_code=EXIT_CODES["no_embeddings"],
            message=str(exc),
            hint="commit has no Claude-Transcript-* trailer and no refs/notes/transcripts entry",
        )
        emit_error(err, fmt)
        sys.exit(err.exit_code)
    except _binding.GitInvocationError as exc:
        err = ClassifiedError(
            kind="catalog_error",
            exit_code=EXIT_CODES["catalog_error"],
            message=f"git invocation failed: {exc.stderr.strip()}",
            hint="check that the commit SHA exists in --repo",
        )
        emit_error(err, fmt)
        sys.exit(err.exit_code)

    emit_json(binding.to_dict(), fmt=fmt)


class RenderFormat(StrEnum):
    """``review-sheet`` render targets.

    Local to ``review-sheet`` because no other subcommand emits human prose.
    Keeping markdown out of the global :class:`OutputFormat` keeps
    ``--format`` honest on every other subcommand (only renderers that
    actually support markdown get to advertise it).
    """

    MARKDOWN = "markdown"
    JSON = "json"


def _review_sheet_format(common: Common | None) -> RenderFormat:
    """Pick the review-sheet effective render format.

    Default policy diverges from every other subcommand: review-sheet
    output is human-first prose, so ``--format auto`` resolves to
    ``MARKDOWN`` on a TTY (override of the global ``TABLE`` default) and
    ``JSON`` off-TTY. ``--format json`` always pins JSON; every other
    ``OutputFormat`` value resolves to ``MARKDOWN`` on a TTY and ``JSON``
    off-TTY (table/ndjson/csv are not meaningful for the prose shape).
    """
    fmt = _fmt(common)
    if fmt is OutputFormat.JSON:
        return RenderFormat.JSON
    return RenderFormat.MARKDOWN if sys.stdout.isatty() else RenderFormat.JSON


@app.command(name="review-sheet")
def review_sheet_cmd(
    commit_sha: str,
    /,
    *,
    repo: Path | None = None,
    no_thinking: bool = False,
    dry_run: bool = True,
    common: Common | None = None,
) -> None:
    """Render a compressed PR review sheet for a merged commit.

    Resolves the commit's bound transcript via
    :func:`claude_sql.binding.resolve_commit_to_transcript` (RFC 0001
    precedence: trailer first, note fallback, loud failure on
    disagreement), flattens the JSONL into a single review text, and
    asks Sonnet 4.6 — via ``output_config.format`` structured output —
    to populate the :class:`PRReviewSheet` schema.

    Defaults to ``--dry-run`` per the project cost-guard convention.
    Dry-run prints a plan dict (commit_sha, transcript_uri,
    transcript_digest, model_id, prompt_chars_estimate) and skips the
    Bedrock call.

    Output format
    -------------
    On a TTY ``--format auto`` resolves to ``markdown`` (the
    human-readable review-sheet shape). Off-TTY it resolves to
    ``json`` so agents get machine-readable output without a flag. Pass
    ``--format json`` / ``--format markdown`` explicitly to override.
    Dry-run always emits JSON regardless of the selected format —
    plan output is structured by design.

    Exit codes
    ----------
    * 0   review sheet rendered (or refused; refusal still exits 0 with
          ``{"refused": true}`` in the payload).
    * 2   commit has no binding (no trailer, no note).
    * 65  commit not found / git invocation failed.
    * 70  trailer and note disagree on digest.
    """
    _configure(common)
    fmt = _review_sheet_format(common)
    # Error output follows the global rule (TABLE on TTY, JSON off-TTY) — the
    # render format is only meaningful for the success-path narrative.
    error_fmt = _fmt(common)
    settings = _resolve_settings(common)
    repo_path = repo.resolve() if repo is not None else None

    try:
        # Resolve up-front so the worker's binding lookup uses the same repo
        # (the worker re-runs resolve internally when ``transcript_uri_override``
        # is unset; we pre-resolve so we can map LookupError / mismatch errors
        # to the canonical CLI exit codes before opening a DuckDB connection).
        binding = _binding.resolve_commit_to_transcript(commit_sha, repo=repo_path)
    except _binding.BindingMismatchError as exc:
        err = ClassifiedError(
            kind="runtime_error",
            exit_code=EXIT_CODES["runtime_error"],
            message=str(exc),
            hint="run `claude-sql resolve <sha> --all-sources` to see both surfaces",
        )
        emit_error(err, error_fmt)
        sys.exit(err.exit_code)
    except LookupError as exc:
        err = ClassifiedError(
            kind="no_embeddings",
            exit_code=EXIT_CODES["no_embeddings"],
            message=str(exc),
            hint="commit has no Claude-Transcript-* trailer and no refs/notes/transcripts entry",
        )
        emit_error(err, error_fmt)
        sys.exit(err.exit_code)
    except _binding.GitInvocationError as exc:
        err = ClassifiedError(
            kind="catalog_error",
            exit_code=EXIT_CODES["catalog_error"],
            message=f"git invocation failed: {exc.stderr.strip()}",
            hint="check that the commit SHA exists in --repo",
        )
        emit_error(err, error_fmt)
        sys.exit(err.exit_code)

    # Hand the resolved URI through the override so the worker doesn't
    # round-trip to git twice (and so it stays testable without a repo).
    result = generate_review_sheet(
        None,
        settings,
        commit_sha=commit_sha,
        transcript_uri_override=binding.uri,
        dry_run=dry_run,
        no_thinking=no_thinking,
    )

    if dry_run:
        # Plan output is structured regardless of --format choice; users
        # asking for markdown still get JSON for the plan because there's
        # no narrative to render yet.
        plan = result.get("plan", result)
        emit_json(plan, fmt=OutputFormat.JSON)
        return

    if result.get("refused"):
        if fmt is RenderFormat.MARKDOWN:
            metadata = result.get("metadata") or {"commit_sha": commit_sha}
            print(render_refusal_markdown(str(result.get("reason", "")), metadata))
            return
        emit_json(result, fmt=OutputFormat.JSON)
        return

    sheet = result.get("sheet") or {}
    metadata = result.get("metadata") or {}
    if fmt is RenderFormat.MARKDOWN:
        print(render_markdown(sheet, metadata))
        return
    emit_json({"sheet": sheet, "metadata": metadata}, fmt=OutputFormat.JSON)


@app.default
def _default(*, common: Common | None = None) -> None:
    """Print a hint when ``claude-sql`` is invoked without a subcommand."""
    del common
    print("claude-sql - pass a subcommand or --help")
    print("  schema | query | explain | shell | list-cache")
    print("  embed | search")
    print("  classify | trajectory | conflicts | friction | cluster | terms | community | analyze")
    print("  judges | freeze | replay | judge | ungrounded-claim | kappa | blind-handover")
    print("  bind | resolve | review-sheet")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point wired into ``[project.scripts]`` in ``pyproject.toml``."""
    app()


if __name__ == "__main__":
    main()

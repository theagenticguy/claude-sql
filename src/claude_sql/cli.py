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
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import duckdb
import polars as pl
from cyclopts import App, Parameter
from loguru import logger

from claude_sql import blind_handover as _blind_handover
from claude_sql import checkpointer
from claude_sql import freeze as _freeze
from claude_sql import judge_worker as _judge_worker
from claude_sql import judges as _judge_catalog
from claude_sql import kappa_worker as _kappa_worker
from claude_sql import ungrounded_worker as _ungrounded_worker
from claude_sql.cluster_worker import run_clustering
from claude_sql.community_worker import run_communities
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
from claude_sql.sql_views import (
    describe_all,
    list_macros,
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
  cluster / terms / community     UMAP+HDBSCAN, c-TF-IDF, Louvain
  analyze                         composite pipeline over every stage above

Flag placement (important for agents)
-------------------------------------
All flags attach to a SUBCOMMAND, not the top-level binary. Correct:
    claude-sql query --format json "SELECT 1"
    claude-sql classify --no-dry-run --limit 5
Incorrect (flag gets swallowed as the subcommand argument):
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


def _open_connection(settings: Settings) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with every claude-sql object wired."""
    con = duckdb.connect(":memory:")
    register_all(con, settings=settings)
    return con


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

    The row count is only read when the parquet looks healthy (>16 bytes --
    same sentinel used by :func:`register_analytics`).  Reading row counts is
    cheap (``read_parquet`` reads the footer only), but we still gate it
    because a zero-byte file is a sign of an aborted run we should not touch.
    """
    exists = path.exists() and path.is_file()
    entry: dict[str, object] = {"name": name, "path": str(path), "exists": exists}
    if not exists:
        return entry
    stat = path.stat()
    entry["bytes"] = stat.st_size
    entry["mtime"] = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
    if stat.st_size > 16:
        con = duckdb.connect(":memory:")
        try:
            row = con.execute(
                "SELECT count(*) FROM read_parquet(?)",
                [str(path)],
            ).fetchone()
            entry["rows"] = int(row[0]) if row else 0
        except duckdb.Error:
            # Do not let a corrupt parquet abort the whole list -- surface it
            # with rows=None so the caller still sees the metadata.
            entry["rows"] = None
        finally:
            con.close()
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

    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as tf:
        db_path = tf.name

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


@app.command
def query(sql: str, /, *, common: Common | None = None) -> None:
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
    con = _open_connection(settings)
    try:
        df = run_or_die(lambda: con.execute(sql).pl(), fmt=fmt)
        emit_dataframe(df, fmt)
    finally:
        con.close()


@app.command
def explain(
    sql: str,
    /,
    *,
    analyze: bool = False,
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
    con = _open_connection(settings)
    try:
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
    finally:
        con.close()


@app.command
def schema(*, common: Common | None = None) -> None:
    """List every registered view (with columns) and every macro in one pass.

    When to use
    -----------
    First thing an agent should call after ``--help``: it's the canonical
    catalog. Use it to discover column names before composing ``query``
    calls -- e.g., ``session_classifications`` uses both ``autonomy_tier``
    (canonical) and ``autonomy`` (alias), and the schema lists both.

    Output shape (non-TTY / JSON)
    -----------------------------
    ::

        {
          "views": {
            "sessions": [{"column": "session_id", "type": "VARCHAR"}, ...],
            "messages": [...],
            "session_classifications": [...],   // only if parquet exists
            ...
          },
          "macros": ["autonomy_trend", "conflict_rate", ...]
        }

    Missing analytics parquets are silently omitted (register_analytics
    skips them). Use ``list-cache`` to see which generators still need to
    run.
    """
    _configure(common)
    settings = _resolve_settings(common)
    fmt = resolve_format(_fmt(common))
    con = _open_connection(settings)
    try:
        views = describe_all(con)
        macros = list_macros(con)
        if fmt is OutputFormat.TABLE:
            for name, cols in views.items():
                print(f"\n\033[1m{name}\033[0m ({len(cols)} cols)")
                for col, col_type in cols:
                    print(f"  {col:<28} {col_type}")
            print(f"\n\033[1mMacros\033[0m ({len(macros)})")
            for macro in macros:
                print(f"  {macro}")
        else:
            payload = {
                "views": {
                    name: [{"column": c, "type": t} for c, t in cols]
                    for name, cols in views.items()
                },
                "macros": list(macros),
            }
            emit_json(payload, fmt)
    finally:
        con.close()


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
        _describe_cache_entry("user_friction", settings.user_friction_parquet_path),
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

        {"pipeline": "embed", "candidates": N, "batches": B,
         "batch_size": 96, "concurrency": 2, "model": "...",
         "since_days": null, "limit": null, "dry_run": true}

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
    con = _open_connection(settings)
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
    con = _open_connection(settings)
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
    con = _open_connection(settings)
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
    con = _open_connection(settings)
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
    con = _open_connection(settings)
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
    con = _open_connection(settings)
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
def community(*, force: bool = False, common: Common | None = None) -> None:
    """Session-level Louvain community detection over a cosine-similarity graph.

    Prereq: ``embed`` (needs the embeddings parquet).

    Output columns (``session_communities`` view)
    ---------------------------------------------
    session_id, community_id (int; -1 = isolated).

    Method: build a session-centroid-cosine KNN graph, then run
    ``networkx.community.louvain_communities`` (networkx ≥3.4).
    Cost: zero. Seeded by ``CLAUDE_SQL_SEED=42``.

    Flags
    -----
    --force   Re-detect even if session_communities.parquet exists.
    """
    _configure(common)
    settings = _resolve_settings(common)
    con = _open_connection(settings)
    try:
        stats = run_communities(con, settings, force=force)
        logger.info(
            "community: {} sessions grouped into {} communities",
            stats["sessions"],
            stats["communities"],
        )
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
    force_cluster: bool = False,
    force_community: bool = False,
    common: Common | None = None,
) -> None:
    """Run the full analytics pipeline end-to-end: embed → structure → LLM analytics.

    Stages (in order)
    -----------------
    1. embed        (Bedrock Cohere Embed v4; honors --dry-run)
    2. cluster      (UMAP+HDBSCAN; zero-cost; --force_cluster to rebuild)
    3. terms        (c-TF-IDF labels for clusters; zero-cost)
    4. community    (Louvain; zero-cost; --force_community to rebuild)
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

    # 1. Embed (reuses embed_worker).  Silently skipped if the parquet is up to date.
    if not skip_embed:
        con = _open_connection(settings)
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
        con = _open_connection(settings)
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
        con = _open_connection(settings)
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
        con = _open_connection(settings)
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
        con = _open_connection(settings)
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
        con = _open_connection(settings)
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
        con = _open_connection(settings)
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


@app.default
def _default(*, common: Common | None = None) -> None:
    """Print a hint when ``claude-sql`` is invoked without a subcommand."""
    del common
    print("claude-sql - pass a subcommand or --help")
    print("  schema | query | explain | shell | list-cache")
    print("  embed | search")
    print("  classify | trajectory | conflicts | friction | cluster | terms | community | analyze")
    print("  judges | freeze | replay | judge | ungrounded-claim | kappa | blind-handover")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point wired into ``[project.scripts]`` in ``pyproject.toml``."""
    app()


if __name__ == "__main__":
    main()

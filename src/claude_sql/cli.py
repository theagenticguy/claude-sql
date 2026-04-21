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

from claude_sql import checkpointer
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
    OutputFormat,
    emit_dataframe,
    emit_json,
    resolve_format,
    run_or_die,
)
from claude_sql.sql_views import (
    describe_all,
    list_macros,
    register_all,
    register_raw,
    register_views,
)
from claude_sql.terms_worker import run_terms

app = App(
    name="claude-sql",
    version=format_version,
    help=("Zero-copy SQL over ~/.claude/ JSONL transcripts with Cohere Embed v4 semantic search."),
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
    """Build :class:`Settings` from env then apply CLI overrides."""
    settings = Settings()
    if common is None:
        return settings
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
    """Launch the ``duckdb`` REPL with all views, macros, and VSS pre-registered.

    Writes a temporary on-disk DuckDB database, calls :func:`register_all` to
    materialize every view / macro / HNSW index into it, closes the Python
    connection, then execs the ``duckdb`` binary against that DB file.  The
    DB path is printed so the user can reopen it later or clean it up.
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
    """Run a SQL query and emit the result in the requested format.

    Default format is ``auto`` (table on TTY, JSON otherwise).  DuckDB errors
    are classified into parse / catalog / runtime and exit with codes 64 / 65
    / 70 respectively.
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
    """Show the EXPLAIN plan for ``sql`` and highlight pushdown markers.

    Defaults to a static plan (``EXPLAIN``) so probing slow or expensive
    queries doesn't execute them.  Pass ``--analyze`` to run ``EXPLAIN
    ANALYZE`` when you actually want timings.

    When ``--format=json``, emits ``{"plan": "<text>"}`` without the ANSI
    highlights so agents can consume the plan cleanly.
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
    """List every registered view with its columns, plus the macro inventory.

    ``--format table`` (default on TTY) prints a compact human layout; every
    other format emits ``{"views": {...}, "macros": [...]}`` so agents can
    parse the catalog without scraping.
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
    """Introspect parquet cache state -- which outputs exist, how fresh, how many rows.

    Exists so agents can decide, before issuing ``search`` or ``query``,
    whether the prerequisite pipeline stage (``embed`` / ``classify`` /
    ``cluster`` / ``community``) needs to be run.  Each entry reports
    ``{name, path, exists, bytes, mtime, rows}`` where the last three are
    omitted when the parquet is absent.
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
    """Embed unembedded messages to the configured parquet via Cohere Embed v4.

    Only registers raw + derived views (not VSS) because the backfill reads
    ``messages_text`` for discovery and the existing parquet for the anti-join;
    it does not need the HNSW index.
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
        count = asyncio.run(
            run_backfill(
                con=con,
                settings=settings,
                since_days=since_days,
                limit=limit,
                dry_run=dry_run,
            )
        )
        logger.info("Embedded {} messages (dry_run={})", count, dry_run)
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
    """Semantic search over ``message_embeddings`` using HNSW cosine distance.

    Embeds ``query_text`` with Cohere Embed v4 ``search_query`` mode, then
    returns the top-``k`` nearest messages along with a snippet of their text.
    Exits with code 2 (with a clear hint) when the embeddings parquet is empty.
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
    """Classify sessions via Sonnet 4.6 (autonomy, work category, success, goal).

    Default is ``--dry-run`` -- real Bedrock spend requires explicit
    ``--no-dry-run``.
    """
    _configure(common)
    settings = _resolve_settings(common)
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
        logger.info("classify: {} sessions processed (dry_run={})", n, dry_run)
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
    """Per-message sentiment + is_transition (regex prefilter -> Sonnet 4.6).

    Default is ``--dry-run`` -- real Bedrock spend requires explicit
    ``--no-dry-run``.
    """
    _configure(common)
    settings = _resolve_settings(common)
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
        logger.info("trajectory: {} messages processed (dry_run={})", n, dry_run)
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

    Default is ``--dry-run`` -- real Bedrock spend requires explicit
    ``--no-dry-run``.
    """
    _configure(common)
    settings = _resolve_settings(common)
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
        logger.info("conflicts: {} sessions processed (dry_run={})", n, dry_run)
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
    """Classify short user messages for friction signals (regex + Sonnet 4.6).

    Detects status pings, unmet expectations ("screenshot?"), confusion,
    interruptions, corrections, and frustration.  Default is ``--dry-run`` --
    real Bedrock spend requires explicit ``--no-dry-run``.
    """
    _configure(common)
    settings = _resolve_settings(common)
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
        logger.info("friction: {} rows written (dry_run={})", n, dry_run)
    finally:
        con.close()


@app.command
def cluster(*, force: bool = False, common: Common | None = None) -> None:
    """UMAP + HDBSCAN over message_embeddings; writes clusters.parquet."""
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
    """Compute c-TF-IDF term labels per cluster; writes cluster_terms.parquet."""
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
    """Session-level Louvain community detection from cosine-similarity graph."""
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
    """Run the full v2 analytics pipeline.

    Stages: embed -> classify + cluster + community -> trajectory -> conflicts -> friction.

    Default is ``--dry-run`` -- every LLM-touching stage just prints pending
    counts and cost estimates.  Pass ``--no-dry-run`` to execute for real.
    Use ``--skip-<stage>`` to drop a stage entirely.  ``--force-cluster`` /
    ``--force-community`` rebuild those parquet outputs even if they already
    exist.
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


@app.default
def _default(*, common: Common | None = None) -> None:
    """Print a hint when ``claude-sql`` is invoked without a subcommand."""
    del common
    print("claude-sql - pass a subcommand or --help")
    print("  schema | query | explain | shell | list-cache")
    print("  embed | search")
    print("  classify | trajectory | conflicts | friction | cluster | terms | community | analyze")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point wired into ``[project.scripts]`` in ``pyproject.toml``."""
    app()


if __name__ == "__main__":
    main()

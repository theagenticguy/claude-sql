"""Cyclopts CLI entry point for ``claude-sql``.

Wires the ``claude-sql`` console script to six subcommands: ``shell``,
``query``, ``explain``, ``schema``, ``embed``, and ``search``. Shared flags
(``--verbose`` / ``--quiet``, ``--glob``, ``--subagent-glob``) live on a
flattened :class:`Common` dataclass so callers write ``claude-sql query ...
--verbose`` instead of ``--common.verbose``.

``asyncio`` and subprocess imports are performed lazily inside the relevant
commands so that the fast path (``schema``, ``query``, ``explain``) does not
drag extra modules into startup.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Annotated

import duckdb
import polars as pl
from cyclopts import App, Parameter
from loguru import logger

from claude_sql.config import Settings
from claude_sql.embed_worker import embed_query, run_backfill
from claude_sql.logging_setup import configure_logging
from claude_sql.sql_views import (
    describe_all,
    list_macros,
    register_all,
    register_raw,
    register_views,
)

app = App(
    name="claude-sql",
    help=("Zero-copy SQL over ~/.claude/ JSONL transcripts with Cohere Embed v4 semantic search."),
)


@Parameter(name="*")
@dataclass
class Common:
    """Shared CLI flags flattened onto every subcommand."""

    verbose: Annotated[bool, Parameter(negative="--quiet")] = False
    glob: str | None = None
    subagent_glob: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_settings(common: Common | None) -> Settings:
    """Build :class:`Settings` from env then apply CLI overrides.

    Parameters
    ----------
    common
        Parsed shared flags, or ``None`` if the subcommand was invoked without
        any common options.

    Returns
    -------
    Settings
        A fresh settings instance with any CLI overrides applied via
        :meth:`pydantic.BaseModel.model_copy`.
    """
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
    """Open an in-memory DuckDB connection with every claude-sql object wired.

    Parameters
    ----------
    settings
        Fully resolved settings controlling globs, embedding dim, HNSW knobs.

    Returns
    -------
    duckdb.DuckDBPyConnection
        Connection with raw views, derived views, the VSS extension, the
        ``message_embeddings`` table, and all macros registered.
    """
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


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@app.command
def shell(*, common: Common | None = None) -> None:
    """Launch the ``duckdb`` REPL with all views, macros, and VSS pre-registered.

    Writes a temporary on-disk DuckDB database, calls :func:`register_all` to
    materialize every view / macro / HNSW index into it, closes the Python
    connection, then execs the ``duckdb`` binary against that DB file. The CLI
    is strictly a launcher; the real REPL is DuckDB's own.

    Parameters
    ----------
    common
        Shared flags (``--verbose`` / ``--quiet``, ``--glob``,
        ``--subagent-glob``).
    """
    configure_logging(common.verbose if common else False)
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
        sys.exit(127)


@app.command
def query(sql: str, /, *, common: Common | None = None) -> None:
    """Run a SQL query and print the result as a polars table.

    Parameters
    ----------
    sql
        A single DuckDB SQL statement.
    common
        Shared flags (``--verbose`` / ``--quiet``, ``--glob``,
        ``--subagent-glob``).
    """
    configure_logging(common.verbose if common else False, False)
    settings = _resolve_settings(common)
    con = _open_connection(settings)
    try:
        df = con.execute(sql).pl()
        with pl.Config(tbl_rows=100, tbl_cols=20, fmt_str_lengths=120):
            print(df)
    finally:
        con.close()


@app.command
def explain(
    sql: str,
    /,
    *,
    analyze: bool = True,
    common: Common | None = None,
) -> None:
    """Show the EXPLAIN plan for ``sql`` and highlight pushdown markers in green.

    Parameters
    ----------
    sql
        The SQL statement to analyze.
    analyze
        When true (default), run ``EXPLAIN ANALYZE`` so the plan includes
        timing. Pass ``--no-analyze`` for a static plan.
    common
        Shared flags (``--verbose`` / ``--quiet``, ``--glob``,
        ``--subagent-glob``).
    """
    configure_logging(common.verbose if common else False)
    settings = _resolve_settings(common)
    con = _open_connection(settings)
    try:
        prefix = "EXPLAIN ANALYZE " if analyze else "EXPLAIN "
        rows = con.execute(prefix + sql).fetchall()
        # EXPLAIN rows are (type, plan_text) tuples; the plan sits in the last
        # column regardless of row shape.
        text = "\n".join(str(r[-1]) for r in rows)
        for line in text.splitlines():
            if any(m in line for m in _EXPLAIN_MARKERS):
                print(f"\033[92m{line}\033[0m")
            else:
                print(line)
    finally:
        con.close()


@app.command
def schema(*, common: Common | None = None) -> None:
    """List every registered view with its columns, plus the macro inventory.

    Parameters
    ----------
    common
        Shared flags (``--verbose`` / ``--quiet``, ``--glob``,
        ``--subagent-glob``).
    """
    configure_logging(common.verbose if common else False)
    settings = _resolve_settings(common)
    con = _open_connection(settings)
    try:
        views = describe_all(con)
        macros = list_macros(con)
        for name, cols in views.items():
            print(f"\n\033[1m{name}\033[0m ({len(cols)} cols)")
            for col, col_type in cols:
                print(f"  {col:<28} {col_type}")
        print(f"\n\033[1mMacros\033[0m ({len(macros)})")
        for macro in macros:
            print(f"  {macro}")
    finally:
        con.close()


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

    Parameters
    ----------
    since_days
        Only embed messages newer than ``N`` days.
    limit
        Cap on total messages to embed this run.
    dry_run
        Print the batch plan without calling Bedrock.
    common
        Shared flags (``--verbose`` / ``--quiet``, ``--glob``,
        ``--subagent-glob``).
    """
    import asyncio

    configure_logging(common.verbose if common else False)
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
    Exits 2 (with a clear hint) when the embeddings parquet is empty.

    Parameters
    ----------
    query_text
        Natural-language search query.
    k
        Number of nearest neighbors to return.
    common
        Shared flags (``--verbose`` / ``--quiet``, ``--glob``,
        ``--subagent-glob``).
    """
    configure_logging(common.verbose if common else False)
    settings = _resolve_settings(common)
    con = _open_connection(settings)
    try:
        row = con.execute("SELECT count(*) FROM message_embeddings").fetchone()
        count = int(row[0]) if row else 0
        if count == 0:
            logger.error("No embeddings yet. Run: claude-sql embed --since-days 7")
            sys.exit(2)

        qv = embed_query(query_text, settings=settings)
        dim = int(settings.output_dimension)
        df = con.execute(
            f"""
            WITH qv AS (SELECT CAST(? AS FLOAT[{dim}]) AS v)
            SELECT m.uuid,
                   m.session_id,
                   m.role,
                   array_cosine_similarity(me.embedding, (SELECT v FROM qv)) AS sim,
                   substr(mt.text_content, 1, 200) AS snippet
            FROM message_embeddings me
            JOIN messages m USING (uuid)
            LEFT JOIN messages_text mt ON mt.uuid = m.uuid
            ORDER BY array_distance(me.embedding, (SELECT v FROM qv))
            LIMIT ?
            """,
            [qv, k],
        ).pl()
        with pl.Config(tbl_rows=k, tbl_cols=20, fmt_str_lengths=200):
            print(df)
    finally:
        con.close()


@app.default
def _default(*, common: Common | None = None) -> None:
    """Print a hint when ``claude-sql`` is invoked without a subcommand.

    Parameters
    ----------
    common
        Shared flags (unused here; kept for signature uniformity).
    """
    del common
    print("claude-sql - pass a subcommand or --help")
    print("  schema | query | explain | shell | embed | search")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point wired into ``[project.scripts]`` in ``pyproject.toml``."""
    app()


if __name__ == "__main__":
    main()

"""DuckDB view, macro, and VSS registry for claude-sql.

Wires a DuckDB connection to the on-disk ``~/.claude/`` JSONL transcript corpus
and exposes it as a stable set of zero-copy SQL views, analytical macros, and
an HNSW-indexed embeddings table.  v2 analytics outputs (classifications,
trajectory, conflicts, clusters, communities) are surfaced as parquet-backed
views alongside the transcript-derived views.

Design notes
------------
* Reads are zero-copy via ``read_json(..., filename=true)`` -- no intermediate
  parquet ingestion; the corpus is queried in place. ``filename`` unlocks
  file-level predicate pushdown (DuckDB 1.3+).
* Nested ``message.content`` is left as JSON and flattened at query time via
  ``UNNEST(json_extract(content_json, '$[*]'))``. This keeps views resilient
  to new content block types (``text``, ``tool_use``, ``tool_result``,
  ``thinking``, ...).
* Subagent transcripts live in sibling ``agent-<hex>.jsonl`` files under
  ``subagents/`` with ``*.meta.json`` partners; they surface via dedicated
  views so primary-session views stay pure.
* v2 analytics views (``session_classifications``, ``message_trajectory``,
  ``session_conflicts``, ``message_clusters``, ``cluster_terms``,
  ``session_communities``, and the derived ``session_goals``) are created by
  :func:`register_analytics` from the corresponding parquet files.  Each is
  skipped with a warning when its parquet is missing, so the function is
  idempotent on partially-populated systems.
* All views use ``CREATE OR REPLACE`` so callers may safely re-register.
* Globs are inlined into DDL (DuckDB rejects prepared parameters as
  table-function arguments).
* ``v_raw_events`` and ``v_raw_subagents`` are materialized as ``CREATE TEMP
  TABLE`` with an explicit ``columns={...}`` projection so JSON schema
  inference runs once instead of once per ``DESCRIBE``/view bind. The
  ``columns`` dict is a *strict projection filter* in DuckDB 1.5+: every
  field referenced by any downstream view or macro must appear in the dict
  or it silently disappears. ``union_by_name=true`` stays enabled to NULL-
  fill the listed fields across files with per-file drift.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
from loguru import logger

from claude_sql.core.config import DEFAULT_PRICING, Settings
from claude_sql.core.parquet_shards import iter_part_files
from claude_sql.core.s3_source import configure_s3, settings_need_s3

# ---------------------------------------------------------------------------
# Glob constants
# ---------------------------------------------------------------------------

DEFAULT_GLOB: str = os.path.expanduser("~/.claude/projects/*/*.jsonl")
SUBAGENT_GLOB: str = os.path.expanduser("~/.claude/projects/*/*/subagents/agent-*.jsonl")
SUBAGENT_META_GLOB: str = os.path.expanduser("~/.claude/projects/*/*/subagents/agent-*.meta.json")

# Business-level views emitted by ``register_views``. Used by the
# ``claude-sql schema`` subcommand for schema dumps.  Includes the v2
# analytics view names at the tail; the schema dump materializes only
# rows where :func:`register_analytics` has populated the matching parquets.
VIEW_NAMES: tuple[str, ...] = (
    "sessions",
    "messages",
    "content_blocks",
    "messages_text",
    "turn_window",
    "tool_calls",
    "tool_results",
    "todo_events",
    "todo_state_current",
    "subagent_spawns",
    "task_creations",
    "task_updates",
    "tasks_state_current",
    "skill_invocations",
    "subagent_sessions",
    "subagent_messages",
    # v2 analytics views (materialize when the matching parquet exists).
    "session_classifications",
    "session_goals",
    "message_trajectory",
    "session_conflicts",
    "conflicts_summary",
    "message_clusters",
    "cluster_terms",
    "session_communities",
    "community_profile",
    "user_friction",
    "skills_catalog",
    "skill_usage",
    "ingest_stamps",
)

# Hand-maintained column schema for the v1 (transcript-derived) views.
#
# Why a static dict, not catalog introspection?
# ---------------------------------------------
# ``DESCRIBE`` over every view re-binds ``v_raw_events`` and re-runs JSON
# schema inference per call -- the dominant cost in ``claude-sql schema``
# (14.5 s on the live corpus).  Hoisting the schema into a static dict
# lets the agent-facing ``schema`` command answer in <50 ms with zero
# DuckDB connection cost.
#
# Only v1 views appear here. v2 analytics views
# (``session_classifications``, ``message_trajectory``, etc.) are backed
# by parquet files whose schemas live in the parquet metadata itself --
# they're correctly omitted because the source of truth for those views
# is the parquet, not this dict.
#
# Drift is caught by :func:`tests.test_sql_views.test_view_schema_matches_describe_inline`,
# which registers the v1 views over the fixture corpus, runs ``DESCRIBE``
# inline per view, and asserts column-level equality with this dict. A
# contributor who edits view DDL without updating ``VIEW_SCHEMA`` gets a
# hard CI failure rather than a runtime mystery.
VIEW_SCHEMA: dict[str, tuple[tuple[str, str], ...]] = {
    "sessions": (
        ("session_id", "VARCHAR"),
        ("cwd", "VARCHAR"),
        ("git_branch", "VARCHAR"),
        ("started_at", "TIMESTAMP"),
        ("ended_at", "TIMESTAMP"),
        ("assistant_messages", "BIGINT"),
        ("record_count", "BIGINT"),
        ("transcript_path", "VARCHAR"),
    ),
    "messages": (
        ("uuid", "VARCHAR"),
        ("parent_uuid", "VARCHAR"),
        ("session_id", "VARCHAR"),
        ("ts", "TIMESTAMP"),
        ("type", "VARCHAR"),
        ("is_sidechain", "BOOLEAN"),
        ("is_compact_summary", "BOOLEAN"),
        ("role", "VARCHAR"),
        ("model", "VARCHAR"),
        ("stop_reason", "VARCHAR"),
        ("input_tokens", "BIGINT"),
        ("output_tokens", "BIGINT"),
        ("cache_read", "BIGINT"),
        ("cache_write", "BIGINT"),
        ("content_json", "JSON"),
        ("source_file", "VARCHAR"),
    ),
    "content_blocks": (
        ("session_id", "VARCHAR"),
        ("message_uuid", "VARCHAR"),
        ("ts", "TIMESTAMP"),
        ("role", "VARCHAR"),
        ("message_type", "VARCHAR"),
        ("is_compact_summary", "BOOLEAN"),
        ("block_type", "VARCHAR"),
        ("text", "VARCHAR"),
        ("tool_use_id_field", "VARCHAR"),
        ("tool_name", "VARCHAR"),
        ("tool_input", "JSON"),
        ("tool_use_id", "VARCHAR"),
        ("tool_result_content", "JSON"),
        ("thinking", "VARCHAR"),
    ),
    "messages_text": (
        ("uuid", "VARCHAR"),
        ("session_id", "VARCHAR"),
        ("ts", "TIMESTAMP"),
        ("role", "VARCHAR"),
        ("is_compact_summary", "BOOLEAN"),
        ("text_content", "VARCHAR"),
    ),
    "turn_window": (
        ("session_id", "VARCHAR"),
        ("prev_uuid", "VARCHAR"),
        ("prev_role", "VARCHAR"),
        ("prev_ts", "TIMESTAMP"),
        ("curr_uuid", "VARCHAR"),
        ("curr_role", "VARCHAR"),
        ("curr_ts", "TIMESTAMP"),
        ("gap_ms", "BIGINT"),
        ("window_idx", "BIGINT"),
    ),
    "tool_calls": (
        ("message_uuid", "VARCHAR"),
        ("session_id", "VARCHAR"),
        ("ts", "TIMESTAMP"),
        ("tool_name", "VARCHAR"),
        ("tool_use_id", "VARCHAR"),
        ("tool_input", "JSON"),
    ),
    "tool_results": (
        ("message_uuid", "VARCHAR"),
        ("session_id", "VARCHAR"),
        ("ts", "TIMESTAMP"),
        ("tool_use_id", "VARCHAR"),
        ("content", "JSON"),
    ),
    "todo_events": (
        ("session_id", "VARCHAR"),
        ("written_at", "TIMESTAMP"),
        ("message_uuid", "VARCHAR"),
        ("subject", "VARCHAR"),
        ("status", "VARCHAR"),
        ("active_form", "VARCHAR"),
        ("snapshot_ix", "BIGINT"),
    ),
    "todo_state_current": (
        ("session_id", "VARCHAR"),
        ("subject", "VARCHAR"),
        ("status", "VARCHAR"),
        ("active_form", "VARCHAR"),
        ("written_at", "TIMESTAMP"),
    ),
    "subagent_spawns": (
        ("session_id", "VARCHAR"),
        ("spawned_at", "TIMESTAMP"),
        ("message_uuid", "VARCHAR"),
        ("tool_use_id", "VARCHAR"),
        ("spawn_tool", "VARCHAR"),
        ("subagent_type", "VARCHAR"),
        ("description", "VARCHAR"),
        ("prompt", "VARCHAR"),
        ("run_in_background", "VARCHAR"),
    ),
    "task_creations": (
        ("session_id", "VARCHAR"),
        ("created_at", "TIMESTAMP"),
        ("message_uuid", "VARCHAR"),
        ("tool_use_id", "VARCHAR"),
        ("create_tool", "VARCHAR"),
        ("subject", "VARCHAR"),
        ("description", "VARCHAR"),
        ("active_form", "VARCHAR"),
        ("metadata", "JSON"),
    ),
    "task_updates": (
        ("session_id", "VARCHAR"),
        ("updated_at", "TIMESTAMP"),
        ("message_uuid", "VARCHAR"),
        ("tool_use_id", "VARCHAR"),
        ("update_tool", "VARCHAR"),
        ("task_id", "VARCHAR"),
        ("status", "VARCHAR"),
        ("add_blocked_by", "JSON"),
        ("owner", "VARCHAR"),
    ),
    "tasks_state_current": (
        ("session_id", "VARCHAR"),
        ("task_id", "VARCHAR"),
        ("subject", "VARCHAR"),
        ("active_form", "VARCHAR"),
        ("status", "VARCHAR"),
        ("created_at", "TIMESTAMP"),
        ("last_updated_at", "TIMESTAMP"),
    ),
    "skill_invocations": (
        ("session_id", "VARCHAR"),
        ("ts", "TIMESTAMP"),
        ("message_uuid", "VARCHAR"),
        ("source", "VARCHAR"),
        ("skill_id", "VARCHAR"),
        ("args", "VARCHAR"),
        ("tool_use_id", "VARCHAR"),
    ),
    "subagent_sessions": (
        ("parent_session_id", "VARCHAR"),
        ("agent_hex", "VARCHAR"),
        ("agent_type", "VARCHAR"),
        ("description", "VARCHAR"),
        ("started_at", "TIMESTAMP"),
        ("ended_at", "TIMESTAMP"),
        ("message_count", "BIGINT"),
        ("transcript_path", "VARCHAR"),
    ),
    "subagent_messages": (
        ("uuid", "VARCHAR"),
        ("parent_uuid", "VARCHAR"),
        ("session_id", "VARCHAR"),
        ("parent_session_id", "VARCHAR"),
        ("agent_hex", "VARCHAR"),
        ("ts", "TIMESTAMP"),
        ("type", "VARCHAR"),
        ("role", "VARCHAR"),
        ("model", "VARCHAR"),
        ("input_tokens", "BIGINT"),
        ("output_tokens", "BIGINT"),
        ("content_json", "JSON"),
        ("source_file", "VARCHAR"),
    ),
}


# Analytics-only view names -- the subset of :data:`VIEW_NAMES` backed by v2
# parquet outputs.  Exported so callers (``claude-sql`` subcommands, smoke
# tests) can enumerate analytics views without needing to filter out the
# transcript-derived views.
ANALYTICS_VIEW_NAMES: tuple[str, ...] = (
    "session_classifications",
    "session_goals",
    "message_trajectory",
    "session_conflicts",
    "conflicts_summary",
    "message_clusters",
    "cluster_terms",
    "session_communities",
    "community_profile",
    "user_friction",
    "skills_catalog",
    "ingest_stamps",
)

# Macro names registered by :func:`register_macros`.  ``ago`` is the
# always-on temporal helper.  The next seven are v1 macros that ship
# unconditionally; the remaining ten are v2 analytics macros, each
# registered via :func:`_safe_macro` so a missing backing view downgrades
# to a warning instead of an exception.
MACRO_NAMES: tuple[str, ...] = (
    "ago",
    "model_used",
    "cost_estimate",
    "tool_rank",
    "todo_velocity",
    "subagent_fanout",
    "semantic_search",
    "skill_rank",
    "skill_source_mix",
    # v2 analytics macros
    "autonomy_trend",
    "work_mix",
    "success_rate_by_work",
    "cluster_top_terms",
    "community_top_topics",
    "sentiment_arc",
    "friction_counts",
    "friction_rate",
    "friction_examples",
    "conflicts_over_time",
    "unused_skills",
    "canonical_uuid_resolve",
)


# Hand-maintained signatures for every macro in :data:`MACRO_NAMES`.
#
# Why a static dict, not catalog introspection?
# ---------------------------------------------
# DuckDB's ``duckdb_functions()`` returns a NULL ``parameters`` column for
# table macros (those defined as ``CREATE OR REPLACE MACRO foo(args) AS
# TABLE (...)``), so we can't recover signatures from the catalog at
# runtime.  The boring engineer's response: extract once into a static
# dict and gate drift behind a CI test.
#
# A regex-based drift test in ``tests/test_sql_views.py``
# (``test_macro_signatures_match_ddl``) parses the ``CREATE OR REPLACE
# MACRO <name>(<args>)`` strings out of :func:`register_macros`'s source
# and asserts equality with this dict, so a contributor who adds or
# renames a macro arg without updating ``MACRO_SIGNATURES`` gets a hard
# CI failure.
MACRO_SIGNATURES: dict[str, tuple[str, ...]] = {
    "ago": ("interval_text",),
    "model_used": ("sid",),
    "cost_estimate": ("sid",),
    "tool_rank": ("last_n_days",),
    "todo_velocity": ("sid",),
    "subagent_fanout": ("sid",),
    "semantic_search": ("query_vec", "k"),
    "skill_rank": ("last_n_days",),
    "skill_source_mix": ("last_n_days",),
    "autonomy_trend": ("window_days",),
    "work_mix": ("since_days",),
    "success_rate_by_work": ("since_days",),
    "cluster_top_terms": ("cid", "n"),
    "community_top_topics": ("cid", "n"),
    "sentiment_arc": ("sid",),
    "friction_counts": ("since_days",),
    "friction_rate": ("since_days",),
    "friction_examples": ("label_name", "n"),
    "conflicts_over_time": ("since_days",),
    "unused_skills": ("last_n_days",),
    "canonical_uuid_resolve": (),
}


def _sql_str(value: str) -> str:
    """Escape a Python string as a single-quoted SQL literal.

    Parameters
    ----------
    value
        Value to embed in a DDL statement.

    Returns
    -------
    str
        The value wrapped in single quotes with any embedded quotes doubled.
    """
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


# ---------------------------------------------------------------------------
# Raw readers
# ---------------------------------------------------------------------------


# Explicit projection for ``v_raw_events`` / ``v_raw_subagents``.
#
# DuckDB 1.5's ``read_json(columns={...})`` is a STRICT projection filter:
# every field not listed is silently dropped. Listing exactly what the
# downstream views and macros touch lets DuckDB skip JSON schema inference
# entirely — the dominant cost on the live ~10K-file / ~2GB corpus, which
# was previously paid every time any view bound to ``v_raw_events``
# (every ``DESCRIBE``, every cold-start ``SELECT count(*) FROM sessions``).
#
# Drift discipline: when a downstream view in :func:`register_views` adds a
# new top-level field reference, add it here too — otherwise the view will
# silently return NULL for that column. The
# ``test_view_schema_matches_describe_inline`` drift test catches the column
# disappearing from any of the 18 v1 views.
_MESSAGE_STRUCT_TYPE: str = (
    "STRUCT("
    '"role" VARCHAR, '
    '"content" JSON, '
    "model VARCHAR, "
    "stop_reason VARCHAR, "
    "usage STRUCT("
    "input_tokens BIGINT, "
    "output_tokens BIGINT, "
    "cache_read_input_tokens BIGINT, "
    "cache_creation_input_tokens BIGINT"
    ")"
    ")"
)
# Type choices: ``sessionId`` and ``uuid`` -> ``VARCHAR`` (not ``UUID``)
# because the live corpus has occasional non-canonical session/uuid strings
# (truncated trailing lines from in-flight transcripts, manually edited
# fixtures in tests, older Claude Code versions that emitted free-form
# session ids). Strict ``UUID`` typing in DuckDB silently nulls anything
# that doesn't match the canonical hex layout, which is invisible at
# read_json time but devastating at view-bind time. ``parentUuid`` ->
# ``JSON`` because the field is sometimes null and sometimes a UUID-shaped
# string and downstream views only stringify or json_extract over it.
_RAW_EVENT_COLUMNS: dict[str, str] = {
    "uuid": "VARCHAR",
    "sessionId": "VARCHAR",
    "parentUuid": "JSON",
    "type": "VARCHAR",
    "timestamp": "TIMESTAMP",
    "isSidechain": "BOOLEAN",
    "isCompactSummary": "BOOLEAN",
    "cwd": "VARCHAR",
    "gitBranch": "VARCHAR",
    "message": _MESSAGE_STRUCT_TYPE,
}
_RAW_SUBAGENT_COLUMNS: dict[str, str] = {
    "uuid": "VARCHAR",
    "sessionId": "VARCHAR",
    "parentUuid": "JSON",
    "type": "VARCHAR",
    "timestamp": "TIMESTAMP",
    "message": _MESSAGE_STRUCT_TYPE,
}
# Inlined ``read_json`` upper bound. Bumped from the 16 MB default to handle
# the rare jumbo transcript line (large tool_result blobs from web pages or
# repo dumps). Constant rather than a parameter — no caller has needed to
# override it in production.
_MAX_OBJECT_SIZE: int = 67_108_864


def _render_columns_clause(columns: dict[str, str]) -> str:
    """Render a ``columns={...}`` clause body for ``read_json``.

    The keys are bare DuckDB identifiers (no quoting) and the values are
    SQL type strings wrapped in single quotes. Both halves come from
    code-side constants — never from user input — so escaping is defensive
    only.
    """
    return ", ".join(f"{name}: {_sql_str(typ)}" for name, typ in columns.items())


def register_raw(
    con: duckdb.DuckDBPyConnection,
    *,
    glob: str | None = None,
    subagent_glob: str | None = None,
    subagent_meta_glob: str | None = None,
) -> None:
    """Create the low-level raw readers as TEMP TABLEs / a meta VIEW.

    ``v_raw_events`` and ``v_raw_subagents`` are materialized as
    ``CREATE TEMP TABLE`` with an explicit ``columns={...}`` projection over
    ``read_json``. This shape pays the JSON schema inference cost exactly
    once per connection — every downstream view bind is a cheap catalog
    lookup against a known-shape table instead of a re-inference of the
    underlying glob.

    The subagent ``meta.json`` files are tiny (one object per file) and
    bind so quickly that ``v_raw_subagent_meta`` stays as a ``CREATE OR
    REPLACE VIEW``.

    Parameters
    ----------
    con
        Open DuckDB connection.
    glob
        Glob for primary session transcripts. Defaults to :data:`DEFAULT_GLOB`.
    subagent_glob
        Glob for subagent transcripts. Defaults to :data:`SUBAGENT_GLOB`.
    subagent_meta_glob
        Glob for sibling ``*.meta.json`` files. Defaults to
        :data:`SUBAGENT_META_GLOB`.

    Raises
    ------
    duckdb.Error
        If any DDL fails. Logged via ``logger.exception`` before re-raise.
    """
    glob = glob if glob is not None else DEFAULT_GLOB
    subagent_glob = subagent_glob if subagent_glob is not None else SUBAGENT_GLOB
    subagent_meta_glob = (
        subagent_meta_glob if subagent_meta_glob is not None else SUBAGENT_META_GLOB
    )

    raw_event_cols = _render_columns_clause(_RAW_EVENT_COLUMNS)
    raw_subagent_cols = _render_columns_clause(_RAW_SUBAGENT_COLUMNS)

    try:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE v_raw_events AS
            SELECT *,
                   filename AS source_file,
                   -- Two on-disk layouts produce a session id from the path:
                   --   local : .../<session_id>.jsonl
                   --   S3SessionStore : .../<session_id>/part-<epochMs>-<rand>.jsonl
                   -- The part-file form keys the session on the parent
                   -- directory; the flat form keys on the basename. Detect the
                   -- part layout first so S3-mirrored corpora bind correctly.
                   CASE
                       WHEN regexp_full_match(filename, '.*/part-[^/]*\\.jsonl$')
                       THEN regexp_extract(filename, '/([^/]+)/part-[^/]*\\.jsonl$', 1)
                       ELSE regexp_extract(filename, '([^/]+)\\.jsonl$', 1)
                   END AS session_id_file
            FROM read_json(
                {_sql_str(glob)},
                format='newline_delimited',
                union_by_name=true,
                filename=true,
                ignore_errors=true,
                columns={{{raw_event_cols}}},
                maximum_object_size={_MAX_OBJECT_SIZE}
            );
            """
        )
        logger.debug(
            "Registered v_raw_events (TEMP TABLE) from glob {} with explicit columns",
            glob,
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE v_raw_subagents AS
            SELECT *,
                   filename AS source_file,
                   regexp_extract(
                       filename,
                       '/([0-9a-f-]{{36}})/subagents/agent-([a-f0-9]+)\\.jsonl$',
                       1
                   ) AS parent_session_id,
                   regexp_extract(
                       filename,
                       '/([0-9a-f-]{{36}})/subagents/agent-([a-f0-9]+)\\.jsonl$',
                       2
                   ) AS agent_hex
            FROM read_json(
                {_sql_str(subagent_glob)},
                format='newline_delimited',
                union_by_name=true,
                filename=true,
                ignore_errors=true,
                columns={{{raw_subagent_cols}}},
                maximum_object_size={_MAX_OBJECT_SIZE}
            );
            """
        )
        logger.debug("Registered v_raw_subagents (TEMP TABLE) from glob {}", subagent_glob)

        # meta.json files are one object per file (not NDJSON) -> format='auto'.
        # Tiny relative to the transcript globs; inference is cheap so we
        # keep the VIEW shape and let DuckDB infer the schema.
        con.execute(
            f"""
            CREATE OR REPLACE VIEW v_raw_subagent_meta AS
            SELECT *,
                   filename AS source_file,
                   regexp_extract(
                       filename,
                       '/([0-9a-f-]{{36}})/subagents/agent-([a-f0-9]+)\\.meta\\.json$',
                       1
                   ) AS parent_session_id,
                   regexp_extract(
                       filename,
                       '/([0-9a-f-]{{36}})/subagents/agent-([a-f0-9]+)\\.meta\\.json$',
                       2
                   ) AS agent_hex
            FROM read_json(
                {_sql_str(subagent_meta_glob)},
                format='auto',
                union_by_name=true,
                filename=true,
                ignore_errors=true
            );
            """
        )
        logger.debug("Registered v_raw_subagent_meta from glob {}", subagent_meta_glob)
    except Exception:
        # register-or-fail-loud — any DuckDB error must surface to the caller.
        logger.exception("Failed to register raw views")
        raise


# ---------------------------------------------------------------------------
# Derived views
# ---------------------------------------------------------------------------


def register_views(con: duckdb.DuckDBPyConnection) -> None:
    """Create logical business-level views on top of the raw readers.

    Must be called after :func:`register_raw`. Creates, in order:
    ``sessions``, ``messages``, ``content_blocks``, ``messages_text``,
    ``tool_calls``, ``tool_results``, ``todo_events``, ``todo_state_current``,
    ``subagent_spawns``, ``task_creations``, ``task_updates``,
    ``tasks_state_current``, ``subagent_sessions``, ``subagent_messages``.

    The split between ``subagent_spawns`` and ``task_creations`` reflects
    the Claude Code v2.1.63 ``Task``→``Agent`` rename and the v2.1.16
    (Jan 2026) split of interactive todo tracking from ``TodoWrite`` into
    the ``TaskCreate``/``TaskGet``/``TaskList``/``TaskUpdate`` family.
    Pre-2026 transcripts and Agent-SDK / ``--print`` runs still emit
    ``TodoWrite`` (covered by ``todo_events``).

    Parameters
    ----------
    con
        Open DuckDB connection with raw views already registered.

    Raises
    ------
    duckdb.Error
        If any view DDL fails. Logged via ``logger.exception`` before re-raise.
    """
    try:
        con.execute(
            """
            CREATE OR REPLACE VIEW sessions AS
            SELECT
                session_id_file                            AS session_id,
                any_value(cwd)                             AS cwd,
                any_value(gitBranch)                       AS git_branch,
                min(timestamp::TIMESTAMP)                  AS started_at,
                max(timestamp::TIMESTAMP)                  AS ended_at,
                count(*) FILTER (WHERE type = 'assistant') AS assistant_messages,
                count(*)                                   AS record_count,
                any_value(source_file)                     AS transcript_path
            FROM v_raw_events
            WHERE sessionId IS NOT NULL
            GROUP BY session_id_file;
            """
        )
        logger.debug("Registered view: sessions")

        # ``message.content`` is inferred by ``read_json`` as JSON. ``to_json``
        # is defensive: if a future schema infers LIST, ``to_json`` normalizes
        # it back to a JSON-typed column that ``json_extract`` understands.
        con.execute(
            """
            CREATE OR REPLACE VIEW messages AS
            SELECT
                uuid,
                -- ``parentUuid`` is inferred as JSON when the column carries
                -- both NULL and string-UUID values across files (DuckDB's
                -- read_json union picks the widest type). The recursive-CTE
                -- join in cookbook recipe 1.4 then fails with a Conversion
                -- Error because DuckDB tries to coerce the VARCHAR RHS to
                -- JSON. Casting at view-construction time exposes the
                -- column as a flat VARCHAR so user-facing recursive walks
                -- "just work" (GH #47).
                CAST(parentUuid AS VARCHAR)               AS parent_uuid,
                sessionId                                 AS session_id,
                timestamp::TIMESTAMP                      AS ts,
                type,
                isSidechain                               AS is_sidechain,
                coalesce(isCompactSummary, false)         AS is_compact_summary,
                message.role                              AS role,
                message.model                             AS model,
                message.stop_reason                       AS stop_reason,
                message.usage.input_tokens                AS input_tokens,
                message.usage.output_tokens               AS output_tokens,
                message.usage.cache_read_input_tokens     AS cache_read,
                message.usage.cache_creation_input_tokens AS cache_write,
                to_json(message.content)                  AS content_json,
                source_file
            FROM v_raw_events
            WHERE type IN ('user', 'assistant');
            """
        )
        logger.debug("Registered view: messages")

        con.execute(
            """
            CREATE OR REPLACE VIEW content_blocks AS
            SELECT
                m.session_id,
                m.uuid                                      AS message_uuid,
                m.ts,
                m.role,
                m.type                                      AS message_type,
                m.is_compact_summary,
                json_extract_string(block, '$.type')        AS block_type,
                json_extract_string(block, '$.text')        AS text,
                json_extract_string(block, '$.id')          AS tool_use_id_field,
                json_extract_string(block, '$.name')        AS tool_name,
                json_extract(block, '$.input')              AS tool_input,
                json_extract_string(block, '$.tool_use_id') AS tool_use_id,
                json_extract(block, '$.content')            AS tool_result_content,
                json_extract_string(block, '$.thinking')    AS thinking
            FROM messages m,
                 UNNEST(json_extract(m.content_json, '$[*]')) AS t(block);
            """
        )
        logger.debug("Registered view: content_blocks")

        # One row per *message*, not per text block.  Aggregating the text
        # blocks preserves enough context for useful embeddings; the per-block
        # fan-out made semantic search noisy (tiny fragments like
        # "Now run the tests" dominated results).  Messages with no text blocks
        # (tool-use-only, tool-result-only) are omitted.
        con.execute(
            """
            CREATE OR REPLACE VIEW messages_text AS
            SELECT
                cb.message_uuid AS uuid,
                any_value(cb.session_id) AS session_id,
                any_value(cb.ts)         AS ts,
                any_value(cb.role)       AS role,
                any_value(cb.is_compact_summary) AS is_compact_summary,
                string_agg(cb.text, '\n\n')  AS text_content
            FROM content_blocks cb
            WHERE cb.block_type = 'text'
              AND cb.text IS NOT NULL
              AND length(cb.text) > 0
              AND cb.message_type != 'attachment'
            GROUP BY cb.message_uuid
            HAVING length(string_agg(cb.text, '\n\n')) >= 32;
            """
        )
        logger.debug("Registered view: messages_text")

        # Adjacent-turn window over ``messages_text`` per session, ordered by
        # (ts, uuid) for stable tie-break.  Compact-summary rows are excluded
        # so the LAG() previous-turn pointer never lands on a synthetic
        # checkpoint row injected by Claude Code's own context-compaction
        # path.  ``gap_ms`` is the millisecond delta between turn timestamps,
        # NULL on the first row of each session (LAG() yields NULL there).
        # Used by v1.0 friction / trajectory pipelines that need adjacent
        # (prev, curr) pairs without re-deriving the LAG window in every
        # caller.
        con.execute(
            """
            CREATE OR REPLACE VIEW turn_window AS
            SELECT
                session_id,
                LAG(uuid) OVER w  AS prev_uuid,
                LAG(role) OVER w  AS prev_role,
                LAG(ts)   OVER w  AS prev_ts,
                uuid              AS curr_uuid,
                role              AS curr_role,
                ts                AS curr_ts,
                date_diff('millisecond', LAG(ts) OVER w, ts) AS gap_ms,
                row_number() OVER w AS window_idx
            FROM messages_text
            WHERE is_compact_summary = false
            WINDOW w AS (PARTITION BY session_id ORDER BY ts, uuid);
            """
        )
        logger.debug("Registered view: turn_window")

        con.execute(
            """
            CREATE OR REPLACE VIEW tool_calls AS
            SELECT
                cb.message_uuid,
                cb.session_id,
                cb.ts,
                cb.tool_name,
                cb.tool_use_id_field AS tool_use_id,
                cb.tool_input
            FROM content_blocks cb
            WHERE cb.block_type = 'tool_use';
            """
        )
        logger.debug("Registered view: tool_calls")

        con.execute(
            """
            CREATE OR REPLACE VIEW tool_results AS
            SELECT
                cb.message_uuid,
                cb.session_id,
                cb.ts,
                cb.tool_use_id,
                cb.tool_result_content AS content
            FROM content_blocks cb
            WHERE cb.block_type = 'tool_result';
            """
        )
        logger.debug("Registered view: tool_results")

        # DuckDB's ``UNNEST`` requires a LIST. ``json_extract(x, '$.todos')``
        # returns a JSON scalar (potentially an array) that UNNEST rejects.
        # The ``$.todos[*]`` wildcard path yields a ``JSON[]`` that UNNEST
        # accepts natively.
        con.execute(
            """
            CREATE OR REPLACE VIEW todo_events AS
            SELECT
                tc.session_id,
                tc.ts                                      AS written_at,
                tc.message_uuid,
                json_extract_string(todo, '$.content')     AS subject,
                json_extract_string(todo, '$.status')      AS status,
                json_extract_string(todo, '$.activeForm')  AS active_form,
                row_number() OVER (
                    PARTITION BY tc.session_id
                    ORDER BY tc.ts, tc.message_uuid
                ) AS snapshot_ix
            FROM tool_calls tc,
                 UNNEST(json_extract(tc.tool_input, '$.todos[*]')) AS t(todo)
            WHERE tc.tool_name = 'TodoWrite';
            """
        )
        logger.debug("Registered view: todo_events")

        con.execute(
            """
            CREATE OR REPLACE VIEW todo_state_current AS
            SELECT session_id, subject, status, active_form, written_at
            FROM (
                SELECT *,
                       row_number() OVER (
                           PARTITION BY session_id, subject
                           ORDER BY snapshot_ix DESC
                       ) AS rn
                FROM todo_events
            )
            WHERE rn = 1;
            """
        )
        logger.debug("Registered view: todo_state_current")

        # Subagent launchers: ``Task`` (pre-v2.1.63) and ``Agent`` (v2.1.63+).
        # Input shape: {subagent_type, description, prompt, run_in_background?}.
        con.execute(
            """
            CREATE OR REPLACE VIEW subagent_spawns AS
            SELECT
                session_id,
                ts AS spawned_at,
                message_uuid,
                tool_use_id,
                tool_name AS spawn_tool,
                json_extract_string(tool_input, '$.subagent_type')      AS subagent_type,
                json_extract_string(tool_input, '$.description')        AS description,
                json_extract_string(tool_input, '$.prompt')             AS prompt,
                json_extract_string(tool_input, '$.run_in_background')  AS run_in_background
            FROM tool_calls
            WHERE tool_name IN ('Task', 'Agent');
            """
        )
        logger.debug("Registered view: subagent_spawns")

        # Persistent task creation: ``TaskCreate`` (Claude Code v2.1.16+
        # interactive sessions) and the SDK-py mirror ``mcp__tasks__task_create``.
        # Input shape: {subject, description, activeForm?, metadata?}. Distinct
        # from subagent_spawns -- no subagent_type / prompt fields.
        con.execute(
            """
            CREATE OR REPLACE VIEW task_creations AS
            SELECT
                session_id,
                ts AS created_at,
                message_uuid,
                tool_use_id,
                tool_name                                              AS create_tool,
                json_extract_string(tool_input, '$.subject')           AS subject,
                json_extract_string(tool_input, '$.description')       AS description,
                json_extract_string(tool_input, '$.activeForm')        AS active_form,
                json_extract(tool_input, '$.metadata')                 AS metadata
            FROM tool_calls
            WHERE tool_name IN ('TaskCreate', 'mcp__tasks__task_create');
            """
        )
        logger.debug("Registered view: task_creations")

        # Task lifecycle updates: ``TaskUpdate`` (v2.1.16+) and the SDK-py
        # mirror ``mcp__tasks__task_update``. Native uses ``taskId`` (camel),
        # mcp variant uses ``id`` -- COALESCE to one column.
        con.execute(
            """
            CREATE OR REPLACE VIEW task_updates AS
            SELECT
                session_id,
                ts AS updated_at,
                message_uuid,
                tool_use_id,
                tool_name AS update_tool,
                COALESCE(
                    json_extract_string(tool_input, '$.taskId'),
                    json_extract_string(tool_input, '$.id')
                )                                                       AS task_id,
                json_extract_string(tool_input, '$.status')             AS status,
                json_extract(tool_input, '$.addBlockedBy')              AS add_blocked_by,
                json_extract_string(tool_input, '$.owner')              AS owner
            FROM tool_calls
            WHERE tool_name IN ('TaskUpdate', 'mcp__tasks__task_update');
            """
        )
        logger.debug("Registered view: task_updates")

        # Latest status per (session_id, task_id) by joining task_creations
        # to task_updates. The task_id on TaskCreate isn't carried in the
        # tool_input (the runtime assigns it), so we recover it from the
        # tool_result -- and fall back to row-position when the result is
        # missing. Mirrors ``todo_state_current`` for the v2.1.16+ family.
        con.execute(
            """
            CREATE OR REPLACE VIEW tasks_state_current AS
            WITH creates AS (
                SELECT
                    tc.session_id,
                    tc.created_at,
                    tc.subject,
                    tc.active_form,
                    tc.tool_use_id,
                    -- The runtime returns the assigned task id in tool_result.
                    -- Common shape: text content like "Task #N created..." or
                    -- a JSON {taskId: "N"}. Try both; fall back to per-session
                    -- creation order.
                    COALESCE(
                        regexp_extract(
                            CAST(tr.content AS VARCHAR), 'Task #(\\d+)', 1
                        ),
                        json_extract_string(tr.content, '$.taskId'),
                        CAST(row_number() OVER (
                            PARTITION BY tc.session_id ORDER BY tc.created_at
                        ) AS VARCHAR)
                    ) AS task_id
                FROM task_creations tc
                LEFT JOIN tool_results tr USING (tool_use_id)
            ),
            latest_status AS (
                SELECT session_id, task_id, status, updated_at,
                       row_number() OVER (
                           PARTITION BY session_id, task_id
                           ORDER BY updated_at DESC
                       ) AS rn
                FROM task_updates
                WHERE task_id IS NOT NULL
            )
            SELECT
                c.session_id,
                c.task_id,
                c.subject,
                c.active_form,
                COALESCE(ls.status, 'pending') AS status,
                c.created_at,
                ls.updated_at AS last_updated_at
            FROM creates c
            LEFT JOIN latest_status ls
              ON ls.session_id = c.session_id
             AND ls.task_id = c.task_id
             AND ls.rn = 1;
            """
        )
        logger.debug("Registered view: tasks_state_current")

        # Every Skill / slash-command invocation observable in the transcripts,
        # unioned across the two shapes they take:
        #
        # * ``tool`` — the assistant invokes the built-in ``Skill`` tool with
        #   ``tool_input.skill = '<name>'``.  Lives in ``tool_calls`` already.
        # * ``slash_command`` — the user types ``/<name>`` in chat, which
        #   Claude Code serializes into the text block as
        #   ``<command-name>/<name></command-name>`` (sometimes paired with
        #   ``<command-message>`` and ``<command-args>``).
        #
        # ``skill_id`` stays raw (``erpaval`` and ``personal-plugins:erpaval``
        # are distinct rows) — the ``skills_catalog`` seed emits both shapes
        # so the enriched ``skill_usage`` view joins cleanly either way.
        # ``<command-name>/<name></command-name>`` slash-command text lands in
        # two shapes across the corpus: inside a ``text`` block of a
        # list-typed ``content`` array (newer transcripts), and as a bare
        # VARCHAR ``message.content`` (older user turns).  We scan both
        # so the slash-command surface isn't biased toward one era.
        cmd_name_re = "<command-name>/([A-Za-z0-9_:.-]+)</command-name>"
        args_re = "<command-args>([^<]*)</command-args>"
        con.execute(
            f"""
            CREATE OR REPLACE VIEW skill_invocations AS
            SELECT
                tc.session_id,
                tc.ts,
                tc.message_uuid,
                'tool'                                        AS source,
                json_extract_string(tc.tool_input, '$.skill') AS skill_id,
                json_extract_string(tc.tool_input, '$.args')  AS args,
                tc.tool_use_id
            FROM tool_calls tc
            WHERE tc.tool_name = 'Skill'
              AND json_extract_string(tc.tool_input, '$.skill') IS NOT NULL
            UNION ALL
            SELECT
                cb.session_id,
                cb.ts,
                cb.message_uuid,
                'slash_command'                                    AS source,
                regexp_extract(cb.text, '{cmd_name_re}', 1)        AS skill_id,
                NULLIF(regexp_extract(cb.text, '{args_re}', 1), '') AS args,
                NULL                                                AS tool_use_id
            FROM content_blocks cb
            WHERE cb.role = 'user'
              AND cb.block_type = 'text'
              AND cb.text LIKE '%<command-name>/%'
              AND regexp_extract(cb.text, '{cmd_name_re}', 1) != ''
            UNION ALL
            SELECT
                m.session_id,
                m.ts,
                m.uuid                                                 AS message_uuid,
                'slash_command'                                        AS source,
                regexp_extract(raw.txt, '{cmd_name_re}', 1)            AS skill_id,
                NULLIF(regexp_extract(raw.txt, '{args_re}', 1), '')    AS args,
                NULL                                                   AS tool_use_id
            FROM messages m,
                 LATERAL (SELECT json_extract_string(m.content_json, '$') AS txt) raw
            WHERE m.role = 'user'
              AND json_type(m.content_json) = 'VARCHAR'
              AND raw.txt LIKE '%<command-name>/%'
              AND regexp_extract(raw.txt, '{cmd_name_re}', 1) != '';
            """
        )
        logger.debug("Registered view: skill_invocations")

        con.execute(
            """
            CREATE OR REPLACE VIEW subagent_sessions AS
            SELECT
                r.parent_session_id,
                r.agent_hex,
                any_value(m.agentType)      AS agent_type,
                any_value(m.description)    AS description,
                min(r.timestamp::TIMESTAMP) AS started_at,
                max(r.timestamp::TIMESTAMP) AS ended_at,
                count(*)                    AS message_count,
                any_value(r.source_file)    AS transcript_path
            FROM v_raw_subagents r
            LEFT JOIN v_raw_subagent_meta m
              ON m.parent_session_id = r.parent_session_id
             AND m.agent_hex = r.agent_hex
            GROUP BY r.parent_session_id, r.agent_hex;
            """
        )
        logger.debug("Registered view: subagent_sessions")

        con.execute(
            """
            CREATE OR REPLACE VIEW subagent_messages AS
            SELECT
                uuid,
                -- See ``messages`` view above (GH #47): keep parent_uuid
                -- VARCHAR-typed so recursive CTEs over subagent threads
                -- don't trip the same JSON-coercion landmine.
                CAST(parentUuid AS VARCHAR) AS parent_uuid,
                sessionId                   AS session_id,
                parent_session_id,
                agent_hex,
                timestamp::TIMESTAMP        AS ts,
                type,
                message.role                AS role,
                message.model               AS model,
                message.usage.input_tokens  AS input_tokens,
                message.usage.output_tokens AS output_tokens,
                to_json(message.content)    AS content_json,
                source_file
            FROM v_raw_subagents
            WHERE type IN ('user', 'assistant');
            """
        )
        logger.debug("Registered view: subagent_messages")
    except Exception:
        # register-or-fail-loud — any DuckDB error must surface to the caller.
        logger.exception("Failed to register derived views")
        raise


# ---------------------------------------------------------------------------
# Macros
# ---------------------------------------------------------------------------


def _pricing_values_clause(pricing: dict[str, tuple[float, float]]) -> str:
    """Render a pricing dict as an inline SQL ``VALUES`` row list.

    Parameters
    ----------
    pricing
        Mapping of ``model_name -> (input_rate, output_rate)`` per 1M tokens.

    Returns
    -------
    str
        Comma-separated ``('model', in, out)`` rows. Emits a sentinel row that
        matches no real model if ``pricing`` is empty (DuckDB rejects empty
        ``VALUES`` lists).
    """
    if not pricing:
        return "('__no_pricing__', 0.0, 0.0)"
    rows = [
        f"('{model}', {in_rate}, {out_rate})"
        for model, (in_rate, out_rate) in sorted(pricing.items())
    ]
    return ", ".join(rows)


# Per-macro parquet dependencies: each analytics macro is only registered
# when EVERY listed Settings ``*_parquet_path`` is populated on disk.
# Encoding the invariant here -- in the registration code, not in a log
# level on the failure path -- means a fresh install never flirts with
# ``CREATE OR REPLACE MACRO`` against a missing view.  ``_safe_macro``
# stays as defensive backstop for the rare gate-check / DDL-bind race.
#
# Drift catcher: ``test_register_macros_skips_friction_when_parquet_missing``
# checks that fresh-install registration produces zero analytics-macro
# entries in :func:`list_macros`.  When you add a new analytics macro,
# update this dict alongside :data:`MACRO_NAMES` and :data:`MACRO_SIGNATURES`
# (the existing drift tests catch those two).
_ANALYTICS_MACRO_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "autonomy_trend": ("classifications_parquet_path",),
    "work_mix": ("classifications_parquet_path",),
    "success_rate_by_work": ("classifications_parquet_path",),
    "cluster_top_terms": ("cluster_terms_parquet_path",),
    "community_top_topics": (
        "cluster_terms_parquet_path",
        "communities_parquet_path",
        "clusters_parquet_path",
    ),
    "sentiment_arc": ("trajectory_parquet_path",),
    "friction_counts": ("user_friction_parquet_path",),
    "friction_rate": ("user_friction_parquet_path",),
    "friction_examples": ("user_friction_parquet_path",),
    "conflicts_over_time": ("conflicts_parquet_path",),
    "unused_skills": ("skills_catalog_parquet_path",),
    "canonical_uuid_resolve": ("ingest_stamps_parquet_path",),
}


def _analytics_macro_ready(settings: Settings, macro_name: str) -> bool:
    """Return True when every parquet the macro binds against is populated.

    Looked up from :data:`_ANALYTICS_MACRO_REQUIREMENTS`. Macros not in the
    map (i.e. v1 always-on macros) return ``True`` -- they have no parquet
    dependency.
    """
    fields = _ANALYTICS_MACRO_REQUIREMENTS.get(macro_name)
    if not fields:
        return True
    return all(_parquet_is_populated(getattr(settings, field)) for field in fields)


def _safe_macro(con: duckdb.DuckDBPyConnection, name: str, ddl: str) -> None:
    """Execute a ``CREATE OR REPLACE MACRO`` DDL, swallowing bind failures.

    Defense in depth -- the parquet gate inside :func:`register_macros`
    (see :data:`_ANALYTICS_MACRO_REQUIREMENTS`) already prevents this code
    path from firing in normal operation. ``_safe_macro`` only catches the
    rare race where a parquet vanishes between gate-check and DDL-bind, or
    a future macro adds a different failure mode. The duckdb.Error is
    silently demoted to DEBUG so a routine read-only command never floods
    stderr with WARNING lines about analytics work the user hasn't yet
    asked for.

    Parameters
    ----------
    con
        Open DuckDB connection.
    name
        Macro name, used only for log messages.
    ddl
        Complete ``CREATE OR REPLACE MACRO`` statement.
    """
    try:
        con.execute(ddl)
        logger.debug("Registered analytics macro: {}", name)
    except duckdb.Error as exc:
        logger.debug("Skipped macro {} (backing view missing): {}", name, exc)


def register_macros(
    con: duckdb.DuckDBPyConnection,
    settings: Settings | None = None,
    *,
    skip_vss: bool = False,
) -> None:
    """Create SQL macros used by the CLI and analysts.

    v1 macros (always created): ``model_used``, ``cost_estimate``,
    ``tool_rank``, ``todo_velocity``, ``subagent_fanout``, ``semantic_search``.

    v2 analytics macros (created via :func:`_safe_macro`, skipped when their
    backing analytics view is missing): ``autonomy_trend``, ``work_mix``,
    ``success_rate_by_work``, ``cluster_top_terms``, ``community_top_topics``,
    ``sentiment_arc``, ``canonical_uuid_resolve``.

    ``semantic_search(query_vec, k)`` is a table macro that returns the top-k
    uuids by cosine distance to ``query_vec`` using the HNSW index.
    ``query_vec`` must be ``FLOAT[<dim>]`` matching the ``message_embeddings``
    column type.

    Parameters
    ----------
    con
        Open DuckDB connection with views (and the ``message_embeddings``
        table from :func:`register_vss`) already registered.  Analytics views
        should be registered first (via :func:`register_analytics`) so the
        analytics macros bind successfully; if they're not, those macros are
        skipped with a warning.
    settings
        Optional :class:`Settings` for pricing overrides; falls back to
        :data:`claude_sql.config.DEFAULT_PRICING`.
    skip_vss
        When ``True``, skip the ``semantic_search`` macro registration.
        DuckDB binds macro bodies at CREATE time, so the macro body's
        reference to ``message_embeddings`` would fail to bind whenever the
        VSS extension and the embeddings table are absent. Set this to match
        whatever was passed to :func:`register_all`'s ``skip_vss`` so the
        non-vector connection path stays self-consistent.
    """
    pricing = settings.pricing if settings is not None else DEFAULT_PRICING
    pricing_rows = _pricing_values_clause(pricing)

    # ``ago('14 days')`` -> ``current_timestamp - INTERVAL 14 DAY``.  The
    # CAST handles every interval-unit shape DuckDB recognizes ('30 minutes',
    # '2 weeks', '6 months', ...), so callers don't need to remember the
    # bare-INTERVAL syntax for the common ``WHERE ts >= ago('30 days')``
    # window queries.  Always-on (v1) macro -- no parquet dependency.
    con.execute(
        """
        CREATE OR REPLACE MACRO ago(interval_text) AS (
            current_timestamp - CAST(interval_text AS INTERVAL)
        );
        """
    )
    logger.debug("Registered v1 macro: ago")

    con.execute(
        """
        CREATE OR REPLACE MACRO model_used(sid) AS (
            SELECT any_value(model)
            FROM messages
            WHERE session_id = sid AND model IS NOT NULL
        );
        """
    )

    # Pricing join uses a prefix match so dated model IDs like
    # ``claude-haiku-4-5-20251001`` still resolve to the base entry
    # ``claude-haiku-4-5`` in ``DEFAULT_PRICING``.
    con.execute(
        f"""
        CREATE OR REPLACE MACRO cost_estimate(sid) AS (
            SELECT sum(
                (coalesce(m.input_tokens, 0) + coalesce(m.cache_write, 0)) * p.in_rate
                + coalesce(m.output_tokens, 0) * p.out_rate
            ) / 1e6
            FROM messages m
            JOIN (VALUES {pricing_rows}) p(model, in_rate, out_rate)
              ON regexp_replace(m.model, '-\\d{{8}}$', '') = p.model
            WHERE m.session_id = sid
        );
        """
    )

    con.execute(
        """
        CREATE OR REPLACE MACRO tool_rank(last_n_days) AS TABLE (
            SELECT tool_name, count(*) AS n
            FROM tool_calls
            WHERE ts >= current_timestamp - (last_n_days * INTERVAL 1 DAY)
              AND tool_name IS NOT NULL
            GROUP BY 1
            ORDER BY n DESC
        );
        """
    )

    con.execute(
        """
        CREATE OR REPLACE MACRO todo_velocity(sid) AS (
            SELECT count(*) FILTER (WHERE status = 'completed')::DOUBLE
                 / NULLIF(count(DISTINCT subject), 0)
            FROM todo_state_current
            WHERE session_id = sid
        );
        """
    )

    con.execute(
        """
        CREATE OR REPLACE MACRO subagent_fanout(sid) AS (
            SELECT count(*)
            FROM subagent_sessions
            WHERE parent_session_id = sid
        );
        """
    )

    # ``ORDER BY array_distance`` triggers the HNSW index rewrite; cosine
    # similarity and distance are both surfaced for human-readable ranking.
    # Skip when VSS was not registered: DuckDB binds macro bodies at CREATE
    # time, so referencing ``message_embeddings`` here would fail outright
    # whenever the table doesn't exist on the connection.
    if not skip_vss:
        con.execute(
            """
            CREATE OR REPLACE MACRO semantic_search(query_vec, k) AS TABLE (
                SELECT me.uuid,
                       array_cosine_similarity(me.embedding, query_vec) AS sim,
                       array_distance(me.embedding, query_vec)          AS distance
                FROM message_embeddings me
                ORDER BY array_distance(me.embedding, query_vec)
                LIMIT k
            );
            """
        )
    else:
        logger.debug("Skipped semantic_search macro (skip_vss=True)")

    # Skill / slash-command leaderboard over the last N days.  Resolves
    # against ``skill_usage``, which always exists (with or without the
    # catalog), so this macro is safe to register unconditionally.
    _safe_macro(
        con,
        "skill_rank",
        """
        CREATE OR REPLACE MACRO skill_rank(last_n_days) AS TABLE (
            SELECT skill_id,
                   skill_name,
                   plugin,
                   is_builtin,
                   count(*)                   AS n,
                   count(DISTINCT session_id) AS sessions
              FROM skill_usage
             WHERE ts >= current_timestamp - (last_n_days * INTERVAL 1 DAY)
             GROUP BY 1, 2, 3, 4
             ORDER BY n DESC
        );
        """,
    )

    # How is each skill invoked? ``n_tool`` comes from the ``Skill`` tool,
    # ``n_slash`` from user-typed ``/<name>`` in chat.  Built-ins are
    # excluded because they're almost always slash-only and would drown
    # everything else out.
    _safe_macro(
        con,
        "skill_source_mix",
        """
        CREATE OR REPLACE MACRO skill_source_mix(last_n_days) AS TABLE (
            SELECT skill_id,
                   skill_name,
                   count(*) FILTER (WHERE source = 'tool')          AS n_tool,
                   count(*) FILTER (WHERE source = 'slash_command') AS n_slash,
                   count(*)                                         AS n_total
              FROM skill_usage
             WHERE ts >= current_timestamp - (last_n_days * INTERVAL 1 DAY)
               AND NOT is_builtin
             GROUP BY 1, 2
             ORDER BY n_total DESC
        );
        """,
    )

    _semantic_search_part = "" if skip_vss else "semantic_search, "
    logger.debug(
        "Registered macros: ago, model_used, cost_estimate, tool_rank, "
        f"todo_velocity, subagent_fanout, {_semantic_search_part}skill_rank, "
        "skill_source_mix"
    )

    # ------------------------------------------------------------------
    # v2 analytics macros -- gated on parquet existence via
    # :data:`_ANALYTICS_MACRO_REQUIREMENTS`.  When a backing parquet is
    # missing, the macro DDL never runs, so a fresh install never
    # accidentally binds against a non-existent view.  ``_safe_macro``
    # remains the defensive backstop for the gate-check / DDL-bind race.
    # ------------------------------------------------------------------

    settings_for_gate = settings if settings is not None else Settings()
    analytics_macros: list[tuple[str, str]] = [
        # Time series: autonomy tier mix over rolling windows.
        #
        # Buckets by ``sessions.started_at`` (conversation time), NOT
        # ``classified_at`` (classifier run time). A one-shot backfill stamps
        # every row with the same ``classified_at = NOW()``, which collapsed
        # the "trend" to a single week (issue #49). The trend question is "how
        # has my autonomy mix evolved over time," which depends on when the
        # session happened. The inner join to ``sessions`` on ``session_id``
        # recovers that timestamp and — like ``conflicts_over_time`` (issue
        # #109) — keeps the time axis honest: a classification whose session
        # is not in the transcript-derived corpus has no conversation time to
        # place on the trend, so it is dropped rather than mis-bucketed.
        (
            "autonomy_trend",
            """
            CREATE OR REPLACE MACRO autonomy_trend(window_days) AS TABLE (
                SELECT
                    date_trunc('week', s.started_at) AS week,
                    sc.autonomy_tier,
                    count(*) AS n
                FROM session_classifications sc
                JOIN sessions s
                  ON s.session_id = sc.session_id
                WHERE s.started_at >= current_timestamp - (window_days * INTERVAL 1 DAY)
                GROUP BY 1, 2
                ORDER BY 1, 2
            );
            """,
        ),
        # Work-category mix in the last N days.
        (
            "work_mix",
            """
            CREATE OR REPLACE MACRO work_mix(since_days) AS TABLE (
                SELECT work_category, count(*) AS n
                FROM session_classifications
                WHERE classified_at >= current_timestamp - (since_days * INTERVAL 1 DAY)
                GROUP BY 1
                ORDER BY n DESC
            );
            """,
        ),
        # Success / failure / partial rate broken down by work category.
        (
            "success_rate_by_work",
            """
            CREATE OR REPLACE MACRO success_rate_by_work(since_days) AS TABLE (
                SELECT
                    work_category,
                    count(*) AS sessions,
                    count(*) FILTER (WHERE success = 'success')::DOUBLE
                        / NULLIF(count(*), 0) AS success_rate,
                    count(*) FILTER (WHERE success = 'failure')::DOUBLE
                        / NULLIF(count(*), 0) AS failure_rate,
                    count(*) FILTER (WHERE success = 'partial')::DOUBLE
                        / NULLIF(count(*), 0) AS partial_rate
                FROM session_classifications
                WHERE classified_at >= current_timestamp - (since_days * INTERVAL 1 DAY)
                GROUP BY 1
                ORDER BY sessions DESC
            );
            """,
        ),
        # Top-N TF-IDF terms for a single cluster.
        (
            "cluster_top_terms",
            """
            CREATE OR REPLACE MACRO cluster_top_terms(cid, n) AS TABLE (
                SELECT term, weight, rank
                FROM cluster_terms
                WHERE cluster_id = cid
                ORDER BY rank
                LIMIT n
            );
            """,
        ),
        # Top cluster_ids within a given community, ranked by the number
        # of messages each cluster contributes to the community.  Each
        # row carries its top 5 TF-IDF terms for human-readable context.
        (
            "community_top_topics",
            """
            CREATE OR REPLACE MACRO community_top_topics(cid, n) AS TABLE (
                WITH community_msgs AS (
                    SELECT CAST(m.uuid AS VARCHAR) AS uuid
                      FROM messages m
                      JOIN session_communities sc
                        ON CAST(m.session_id AS VARCHAR) = sc.session_id
                     WHERE sc.community_id = cid
                ),
                cluster_counts AS (
                    SELECT mc.cluster_id, count(*) AS n_msgs
                      FROM message_clusters mc
                      JOIN community_msgs cm USING (uuid)
                     WHERE mc.cluster_id >= 0
                     GROUP BY mc.cluster_id
                )
                SELECT cc.cluster_id, cc.n_msgs,
                       (SELECT string_agg(term, ', ' ORDER BY rank)
                          FROM cluster_terms ct
                         WHERE ct.cluster_id = cc.cluster_id
                           AND ct.rank <= 5) AS top_terms
                  FROM cluster_counts cc
                  ORDER BY n_msgs DESC
                  LIMIT n
            );
            """,
        ),
        # Sentiment arc for a single session: per-window (ts, role,
        # current-turn sentiment, numeric delta, transition_kind, filler
        # flag, confidence) in chronological order.
        #
        # v1.0 windowed rewrite (RFC 0002 §3.4): the macro now joins on
        # ``mt.curr_uuid`` (the per-window anchor turn) instead of the
        # pre-rewrite per-message ``mt.uuid`` column. Output columns are
        # ``(ts, role, curr_sentiment, delta, transition_kind,
        # is_transition, confidence)`` — ``sentiment_delta`` (the old
        # column name) is gone.
        (
            "sentiment_arc",
            """
            CREATE OR REPLACE MACRO sentiment_arc(sid) AS TABLE (
                SELECT m.ts,
                       m.role,
                       mt.curr_sentiment,
                       mt.delta,
                       mt.transition_kind,
                       mt.is_transition,
                       mt.confidence
                  FROM messages m
                  JOIN message_trajectory mt
                    ON CAST(m.uuid AS VARCHAR) = mt.curr_uuid
                 WHERE CAST(m.session_id AS VARCHAR) = sid
                 ORDER BY m.ts
            );
            """,
        ),
        # Counts per friction label, scoped to the last N days by message
        # ``ts`` (the user's actual utterance time, not detected_at).
        # Pass ``NULL`` to include the full corpus.  Excludes label='none'
        # because that is the majority sentinel class and would swamp the
        # output.
        (
            "friction_counts",
            """
            CREATE OR REPLACE MACRO friction_counts(since_days) AS TABLE (
                SELECT label,
                       count(*)                       AS n,
                       count(DISTINCT session_id)     AS sessions,
                       avg(confidence)                AS avg_confidence,
                       sum(CASE WHEN source='regex' THEN 1 ELSE 0 END) AS n_regex,
                       sum(CASE WHEN source='llm'   THEN 1 ELSE 0 END) AS n_llm
                  FROM user_friction
                 WHERE label != 'none'
                   AND (since_days IS NULL
                        OR ts >= current_timestamp - (since_days * INTERVAL 1 DAY))
                 GROUP BY label
                 ORDER BY n DESC
            );
            """,
        ),
        # Per-session friction pressure: how many non-'none' friction
        # messages fired vs the total user message count.  A high rate is
        # a strong proxy for a session where the agent repeatedly fell
        # short of what the user expected.
        (
            "friction_rate",
            """
            CREATE OR REPLACE MACRO friction_rate(since_days) AS TABLE (
                WITH hits AS (
                    SELECT session_id,
                           count(*) FILTER (WHERE label != 'none') AS n_friction,
                           count(*) FILTER (WHERE label = 'status_ping')        AS n_status,
                           count(*) FILTER (WHERE label = 'unmet_expectation')  AS n_unmet,
                           count(*) FILTER (WHERE label = 'confusion')          AS n_confusion,
                           count(*) FILTER (WHERE label = 'interruption')       AS n_interruption,
                           count(*) FILTER (WHERE label = 'correction')         AS n_correction,
                           count(*) FILTER (WHERE label = 'frustration')        AS n_frustration
                      FROM user_friction
                     WHERE since_days IS NULL
                        OR ts >= current_timestamp - (since_days * INTERVAL 1 DAY)
                     GROUP BY session_id
                ),
                user_msgs AS (
                    SELECT CAST(mt.session_id AS VARCHAR) AS session_id,
                           count(*) AS n_user_msgs
                      FROM messages_text mt
                     WHERE mt.role = 'user'
                       AND (since_days IS NULL
                            OR mt.ts >= current_timestamp - (since_days * INTERVAL 1 DAY))
                     GROUP BY 1
                )
                SELECT h.session_id,
                       h.n_friction,
                       h.n_status, h.n_unmet, h.n_confusion,
                       h.n_interruption, h.n_correction, h.n_frustration,
                       COALESCE(um.n_user_msgs, 0)                              AS n_user_msgs,
                       h.n_friction::DOUBLE / NULLIF(um.n_user_msgs, 0)         AS rate
                  FROM hits h
                  LEFT JOIN user_msgs um USING (session_id)
                 WHERE h.n_friction > 0
                 ORDER BY h.n_friction DESC
            );
            """,
        ),
        # Top-N example user messages for a given friction label, highest
        # confidence first.  ``label_name`` is a VARCHAR so DuckDB
        # callers don't have to quote-escape through the macro boundary.
        (
            "friction_examples",
            """
            CREATE OR REPLACE MACRO friction_examples(label_name, n) AS TABLE (
                SELECT session_id, ts, text_snippet, rationale, source, confidence
                  FROM user_friction
                 WHERE label = label_name
                 ORDER BY confidence DESC, ts DESC
                 LIMIT n
            );
            """,
        ),
        # Conflicts on a REAL conversation-time axis (issue #109).
        #
        # ``session_conflicts.detected_at`` is the *worker run-time* clock,
        # not when the user and agent actually disagreed -- a whole backfill
        # run collapses onto one timestamp, so any ``GROUP BY
        # date_trunc(detected_at)`` describes when the nightly job ran, not
        # the conversation.  This macro recovers the real moment by joining
        # the later turn of the pair (``turn_b_uuid``) back to
        # ``messages.uuid`` and surfacing ``m.ts`` -- the same recover-the-
        # timestamp-via-join shape as ``sentiment_arc`` (which joins
        # ``message_trajectory.curr_uuid``), and the same ``ts``-based
        # ``since_days`` window as ``friction_counts`` (pass NULL for the
        # full corpus).
        #
        # INNER JOIN on purpose: a conflict row only appears here when its
        # ``turn_b_uuid`` resolves to a real message, so the time axis is
        # never a fabricated stamp.  On the live corpus a large fraction of
        # stored ``turn_*_uuid`` values are tool-use ids or model-approximated
        # strings that are NOT verbatim message uuids (the whole-session
        # prompt shows ``[role ts]`` headers, not ``[uuid=...]`` ones), so the
        # recoverable subset is the trustworthy subset -- see the v1.1
        # pair-scanner note below for the source-side fix.
        #
        # ``conversation_ts`` is the real axis; ``detected_at`` is carried
        # through ONLY so a caller can see the run-time stamp it replaces.
        # ``root_session_id`` collapses an orchestrator + its subagent
        # sessions to one conversation: a single back-and-forth that fans
        # out across N ``session_id`` values dedupes to one root, so
        # ``count(DISTINCT root_session_id)`` counts distinct *conversations*
        # that disagreed, not raw rows.
        (
            "conflicts_over_time",
            """
            CREATE OR REPLACE MACRO conflicts_over_time(since_days) AS TABLE (
                SELECT m.ts                                          AS conversation_ts,
                       sc.session_id,
                       COALESCE(sm.root_sid, sc.session_id)          AS root_session_id,
                       sc.turn_a_uuid,
                       sc.turn_b_uuid,
                       sc.conflict_kind,
                       sc.severity,
                       sc.agent_position,
                       sc.user_position,
                       sc.confidence,
                       sc.detected_at
                  FROM session_conflicts sc
                  JOIN messages m
                    ON CAST(m.uuid AS VARCHAR) = sc.turn_b_uuid
                  LEFT JOIN (
                      -- child subagent session -> its orchestrator (parent)
                      -- session; aggregated to one row per child so the join
                      -- can never fan a conflict row out.
                      SELECT CAST(session_id AS VARCHAR)                  AS child_sid,
                             any_value(CAST(parent_session_id AS VARCHAR)) AS root_sid
                        FROM subagent_messages
                       GROUP BY CAST(session_id AS VARCHAR)
                  ) sm
                    ON sm.child_sid = sc.session_id
                 WHERE since_days IS NULL
                    OR m.ts >= current_timestamp - (since_days * INTERVAL 1 DAY)
                 ORDER BY m.ts DESC
            );
            """,
        ),
        # Catalog entries the user has NOT invoked in the last N days.
        # Pure catalog lookup; ``skills_catalog`` may be missing pre-sync.
        # ``source_kind`` filter keeps out the 'builtin' rows (users
        # don't install or uninstall ``/clear``).
        (
            "unused_skills",
            """
            CREATE OR REPLACE MACRO unused_skills(last_n_days) AS TABLE (
                SELECT cat.skill_id,
                       cat.name,
                       cat.plugin,
                       cat.plugin_version,
                       cat.source_kind,
                       cat.description
                  FROM skills_catalog cat
                  LEFT JOIN (
                      SELECT DISTINCT skill_id
                        FROM skill_invocations
                       WHERE ts >= current_timestamp - (last_n_days * INTERVAL 1 DAY)
                  ) used USING (skill_id)
                 WHERE used.skill_id IS NULL
                   AND cat.source_kind IN ('user-skill', 'plugin-skill', 'plugin-command')
                 ORDER BY cat.plugin NULLS FIRST, cat.name
            );
            """,
        ),
        # Canonical-UUID resolution over the ``ingest_stamps`` view.  Pairs
        # rows whose 64-bit SimHash differs by ≤ 3 bits (top-16-bit bucket
        # gates the self-join so it doesn't blow up on large corpora) and
        # picks the earliest-seen row as canonical.  ``xor`` is the bit-XOR
        # builtin (DuckDB's ``^`` is exponentiation, not XOR).  Materialised
        # as a table macro so callers can ``SELECT * FROM
        # canonical_uuid_resolve()`` without re-deriving the join.
        (
            "canonical_uuid_resolve",
            """
            CREATE OR REPLACE MACRO canonical_uuid_resolve() AS TABLE (
                SELECT a.uuid AS uuid,
                       MIN(b.uuid) AS canonical_uuid
                  FROM ingest_stamps a
                  JOIN ingest_stamps b
                    ON (a.simhash64 >> 48) = (b.simhash64 >> 48)
                   AND bit_count(xor(a.simhash64, b.simhash64)) <= 3
                   AND b.first_seen_ts <= a.first_seen_ts
                 GROUP BY a.uuid
            );
            """,
        ),
    ]

    for macro_name, ddl in analytics_macros:
        if not _analytics_macro_ready(settings_for_gate, macro_name):
            logger.debug(
                "Skipped analytics macro {} (parquet missing for {})",
                macro_name,
                _ANALYTICS_MACRO_REQUIREMENTS.get(macro_name, ()),
            )
            continue
        _safe_macro(con, macro_name, ddl)


# ---------------------------------------------------------------------------
# VSS
# ---------------------------------------------------------------------------


def register_vss(
    con: duckdb.DuckDBPyConnection,
    *,
    embeddings_parquet: Path | None = None,
    lance_uri: Path | None = None,
    dim: int = 1024,
    metric: str = "cosine",
) -> bool:
    """Bind ``message_embeddings`` over a LanceDB local dataset.

    LanceDB stores embeddings + its IVF_HNSW_SQ index in one place; reads
    come back through DuckDB via the lance core extension
    (``INSTALL lance; LOAD lance; ATTACH (TYPE LANCE)``).

    Parameters
    ----------
    con
        Open DuckDB connection.
    embeddings_parquet
        Legacy parquet shard directory. Used only for one-time migration
        when no Lance dataset exists yet.
    lance_uri
        Local LanceDB dataset directory. When ``None``, falls back to
        ``embeddings_parquet`` for the legacy path resolution.
    dim
        Fixed-length embedding dimension (matches Lance schema).
    metric
        Distance metric; ``cosine`` is the default and what the
        ``semantic_search`` macro expects.

    Returns
    -------
    bool
        ``True`` when the Lance table is reachable through the
        ``message_embeddings`` view; ``False`` when no embeddings exist
        yet (the view is created over an empty schema so downstream
        ``CREATE MACRO semantic_search`` can still bind).
    """
    if lance_uri is None and embeddings_parquet is not None:
        # Default Lance URI is a sibling directory of the legacy parquet path.
        lance_uri = embeddings_parquet.parent / "embeddings_lance"
    if lance_uri is None:
        raise ValueError("register_vss needs either lance_uri or embeddings_parquet")

    dim_i = int(dim)
    if metric not in {"cosine", "l2", "dot"}:
        raise ValueError(f"Unsupported Lance metric: {metric!r}")

    from claude_sql.core import lance_store

    # One-time migration: copy legacy parquet shards into Lance if Lance is
    # empty and legacy shards exist. Idempotent; logs only when work happens.
    if embeddings_parquet is not None:
        try:
            lance_store.migrate_from_parquet_shards(
                legacy_dir=embeddings_parquet,
                lance_uri=lance_uri,
                dim=dim_i,
                delete_legacy=False,
            )
        except Exception as exc:  # noqa: BLE001 — migration is best-effort
            logger.warning("Lance migration from {} skipped: {}", embeddings_parquet, exc)

    con.execute("INSTALL lance;")
    con.execute("LOAD lance;")

    # Empty-dataset gate: probe LanceDB for the embeddings table itself, not
    # the directory's filesystem state. A directory that exists with metadata
    # but no embeddings table (legitimate intermediate state, e.g. after a
    # `connect_db` call that never created a table) ATTACHes cleanly but then
    # blows up at view-bind time with `Catalog Error: Table … embeddings does
    # not exist`. The right gate is "is the table actually there?".
    db = lance_store.connect_db(lance_uri)
    if not lance_store._has_table(db, lance_store.TABLE_NAME):
        logger.warning(
            "No Lance embeddings table at {}; creating empty message_embeddings "
            "table so semantic_search binds. Run `claude-sql embed` to backfill.",
            lance_uri,
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE message_embeddings (
                uuid        VARCHAR PRIMARY KEY,
                model       VARCHAR,
                dim         INTEGER,
                embedding   FLOAT[{dim_i}],
                embedded_at TIMESTAMPTZ
            );
            """
        )
        return False

    # ATTACH the Lance directory and project the embeddings table as a
    # top-level view named ``message_embeddings``. Cast the embedding
    # column to FLOAT[dim] so existing macros that depend on the fixed-
    # size shape (``array_cosine_similarity``, ``array_distance``) keep
    # working without changes.
    con.execute(f"ATTACH '{lance_uri}' AS lance_store (TYPE LANCE);")
    con.execute(
        f"""
        CREATE OR REPLACE VIEW message_embeddings AS
        SELECT
            uuid,
            model,
            dim,
            CAST(embedding AS FLOAT[{dim_i}]) AS embedding,
            embedded_at
        FROM lance_store.main.embeddings;
        """
    )
    row = con.execute("SELECT count(*) FROM message_embeddings;").fetchone()
    count = int(row[0]) if row else 0
    logger.debug(
        "Bound message_embeddings over Lance ({} rows, metric={}, dim={})",
        count,
        metric,
        dim_i,
    )
    return True


# ---------------------------------------------------------------------------
# v2 analytics views
# ---------------------------------------------------------------------------


def _parquet_is_populated(path: Path | None) -> bool:
    """Return True when ``path`` has at least one usable parquet under it.

    Handles both legacy single-file caches (``<name>.parquet``) and the
    sharded directory layout (``<name>/part-<ts>.parquet``).  An empty
    directory or a single zero-byte file both count as "not populated"
    so an aborted run can't trick view registration into pointing at
    rubbish.
    """
    if path is None:
        return False
    parts = iter_part_files(path)
    return any(p.stat().st_size > 16 for p in parts)


def register_analytics(
    con: duckdb.DuckDBPyConnection,
    *,
    settings: Settings | None = None,
    classifications_parquet: Path | None = None,
    trajectory_parquet: Path | None = None,
    conflicts_parquet: Path | None = None,
    clusters_parquet: Path | None = None,
    cluster_terms_parquet: Path | None = None,
    communities_parquet: Path | None = None,
    community_profile_parquet: Path | None = None,
    user_friction_parquet: Path | None = None,
    skills_catalog_parquet: Path | None = None,
) -> None:
    """Register v2 analytics parquets as DuckDB views.

    Creates one ``CREATE OR REPLACE VIEW`` per parquet that exists on disk:
    ``session_classifications``, ``message_trajectory``, ``session_conflicts``,
    ``message_clusters``, ``cluster_terms``, ``session_communities``,
    ``user_friction``, plus the derived ``session_goals`` projection over
    ``session_classifications``.

    Each view is created only when its source parquet exists and is larger
    than an empty-file sentinel (>16 bytes).  Missing parquets are skipped
    with a ``logger.warning`` so the function is idempotent against a
    partially-populated system -- you can call it before, during, or after an
    analytics pipeline run and it will pick up whatever is on disk.

    Analytics macros (``autonomy_trend`` et al.) are **not** registered here
    -- they belong to :func:`register_macros`, which must be called
    afterwards so macro bodies bind against the just-created views.

    Parameters
    ----------
    con
        Open DuckDB connection.
    settings
        Optional :class:`Settings` whose ``*_parquet_path`` fields drive the
        per-view parquet locations.  If ``None``, explicit per-parquet
        keyword arguments take over (see below); if both are supplied, the
        explicit path wins.
    classifications_parquet, trajectory_parquet, conflicts_parquet,
    clusters_parquet, cluster_terms_parquet, communities_parquet
        Optional explicit paths, useful for tests and ad-hoc wiring.  Each
        defaults to the matching ``settings.*_parquet_path`` (or the
        :class:`Settings` defaults) when not provided.
    """
    resolved = settings if settings is not None else Settings()
    view_to_path: dict[str, Path] = {
        "session_classifications": classifications_parquet
        if classifications_parquet is not None
        else resolved.classifications_parquet_path,
        "message_trajectory": trajectory_parquet
        if trajectory_parquet is not None
        else resolved.trajectory_parquet_path,
        "session_conflicts": conflicts_parquet
        if conflicts_parquet is not None
        else resolved.conflicts_parquet_path,
        "message_clusters": clusters_parquet
        if clusters_parquet is not None
        else resolved.clusters_parquet_path,
        "cluster_terms": cluster_terms_parquet
        if cluster_terms_parquet is not None
        else resolved.cluster_terms_parquet_path,
        "session_communities": communities_parquet
        if communities_parquet is not None
        else resolved.communities_parquet_path,
        "community_profile": community_profile_parquet
        if community_profile_parquet is not None
        else resolved.community_profile_parquet_path,
        "user_friction": user_friction_parquet
        if user_friction_parquet is not None
        else resolved.user_friction_parquet_path,
        "skills_catalog": skills_catalog_parquet
        if skills_catalog_parquet is not None
        else resolved.skills_catalog_parquet_path,
        "ingest_stamps": resolved.ingest_stamps_parquet_path,
    }

    # View projections keyed by view name. A ``None`` projection means
    # ``SELECT *``; a string is spliced in verbatim so the wrapper view can
    # add convenience alias columns (e.g. ``autonomy`` alongside
    # ``autonomy_tier``).  These aliases are additive: the original column
    # names continue to work so existing queries never break.
    view_projections: dict[str, str | None] = {
        "session_classifications": (
            "*, autonomy_tier AS autonomy, success AS success_outcome, work_category AS category"
        ),
        # v1.0 windowed shape: ``(session_id, prev_uuid, curr_uuid,
        # prev_sentiment, curr_sentiment, delta, is_transition,
        # transition_kind, confidence, classified_at)``. The legacy
        # ``sentiment_delta`` alias is gone — see RFC 0002 §3.4.
        # ``sentiment`` is kept as a convenience alias for the *current*
        # turn's polarity (the load-bearing one for sentiment-arc plots).
        "message_trajectory": "*, curr_sentiment AS sentiment, is_transition AS transition",
        # v1.0 pair-keyed shape: ``(session_id, turn_a_uuid, turn_b_uuid,
        # conflict_kind, severity, agent_position, user_position,
        # confidence, detected_at)``.  Sessions with no conflicts produce
        # zero rows -- the legacy ``empty=True`` sentinel is gone, and
        # the legacy ``resolution AS conflict_resolution`` alias with it
        # (RFC 0002 §3.4).
        #
        # ⚠️ ``detected_at`` is the WORKER RUN-TIME clock, NOT when the
        # conflict happened in the conversation (issue #109).  A whole
        # backfill run stamps every row with one timestamp, so
        # ``GROUP BY date_trunc(detected_at)`` / ``WHERE detected_at >= ...``
        # describe *when the nightly job ran*, not when the user and agent
        # disagreed.  For a real conversation-time axis use the
        # ``conflicts_over_time(since_days)`` macro, which recovers ``ts`` by
        # joining ``turn_b_uuid`` back to ``messages.uuid`` (same caveat the
        # ``friction_counts`` macro already carries for ``user_friction``).
        "session_conflicts": None,
    }

    # Read-side dedup clauses. Belt-and-suspenders companion to the
    # writer-side replace-then-append in trajectory_worker (#54): if a
    # prior corpus state or a future write-path bug ever lands duplicate
    # ``(session_id, prev_uuid, curr_uuid)`` rows in the parquet shards,
    # the latest ``classified_at`` wins at read time. The fallback
    # ``SELECT *`` path below intentionally drops the QUALIFY because
    # legacy shards may not carry ``classified_at`` -- the fallback is
    # for read-only inspection of stale schemas, not analytics.
    view_qualify: dict[str, str] = {
        "message_trajectory": (
            "row_number() OVER ("
            "PARTITION BY session_id, prev_uuid, curr_uuid "
            "ORDER BY classified_at DESC NULLS LAST"
            ") = 1"
        ),
    }

    registered: set[str] = set()
    for view_name, path in view_to_path.items():
        if not _parquet_is_populated(path):
            # Missing analytics parquets are the default state until the user
            # runs the corresponding generator (classify / cluster / ...), so
            # they belong at DEBUG -- otherwise every query command floods the
            # terminal with warnings about work the user hasn't yet asked for.
            logger.debug(
                "register_analytics: skipping {} (parquet missing at {})",
                view_name,
                path,
            )
            continue
        projection = view_projections.get(view_name) or "*"
        qualify = view_qualify.get(view_name)
        qualify_clause = f" QUALIFY {qualify}" if qualify else ""
        # Sharded directories list every part file; legacy single-file paths
        # become a one-element list. ``read_parquet`` accepts both.
        parts = [p for p in iter_part_files(path) if p.stat().st_size > 16]
        path_literals = ", ".join(_sql_str(str(p)) for p in parts)
        try:
            con.execute(
                f"CREATE OR REPLACE VIEW {view_name} AS "
                f"SELECT {projection} FROM read_parquet([{path_literals}])"
                f"{qualify_clause};"
            )
            logger.debug("Registered analytics view: {} (source={})", view_name, path)
            registered.add(view_name)
        except duckdb.Error as exc:
            # Curated projection failed — most commonly because the on-disk
            # parquet predates a schema rewrite (e.g. v1.0 trajectory rip-
            # and-replace) and the column the projection aliases against
            # doesn't exist yet. Fall back to bare ``SELECT *`` so the
            # legacy view binds for read-only inspection; the next worker
            # run purges the stale shards and the curated projection
            # comes back on the following ``register_analytics``.
            if projection != "*":
                try:
                    con.execute(
                        f"CREATE OR REPLACE VIEW {view_name} AS "
                        f"SELECT * FROM read_parquet([{path_literals}]);"
                    )
                    logger.warning(
                        "register_analytics: {} bound with fallback SELECT * "
                        "(legacy schema detected at {}; run the matching worker "
                        "to refresh): {}",
                        view_name,
                        path,
                        exc,
                    )
                    registered.add(view_name)
                    continue
                except duckdb.Error:
                    # Fallback ``SELECT *`` itself raised — the parquet is
                    # corrupt or the file is gone. Fall through to the
                    # ``logger.exception`` below so the operator sees the
                    # original curated-projection failure with a stack trace.
                    pass
            logger.exception("Failed to register analytics view {} from {}", view_name, path)

    # ``session_goals`` is a thin projection of ``session_classifications``;
    # only materialize it when the upstream view exists.
    if "session_classifications" in registered:
        try:
            con.execute(
                """
                CREATE OR REPLACE VIEW session_goals AS
                SELECT session_id, goal, confidence, classified_at
                FROM session_classifications;
                """
            )
            logger.debug("Registered analytics view: session_goals")
        except duckdb.Error:
            logger.exception("Failed to register session_goals view")

    # ``conflicts_summary`` is the v1.0 replacement for the old
    # ``empty=True`` sentinel scheme.  Sessions with zero conflict rows
    # in ``session_conflicts`` simply do not appear here -- callers that
    # want every session in the result set must LEFT JOIN this view onto
    # ``sessions`` and coalesce the missing count to 0.  See RFC 0002
    # §3.4 for the rationale.
    if "session_conflicts" in registered:
        try:
            con.execute(
                """
                CREATE OR REPLACE VIEW conflicts_summary AS
                SELECT session_id, count(*) AS conflict_count
                FROM session_conflicts
                GROUP BY session_id;
                """
            )
            logger.debug("Registered analytics view: conflicts_summary")
        except duckdb.Error:
            logger.exception("Failed to register conflicts_summary view")

    # ``skill_usage`` joins ``skill_invocations`` (always-on) against the
    # catalog for human-readable labels + ``is_builtin`` tagging.  When the
    # catalog parquet is absent the view still works, but every row gets a
    # ``skill_name = skill_id`` pass-through and ``is_builtin = false``.
    try:
        if "skills_catalog" in registered:
            con.execute(
                """
                CREATE OR REPLACE VIEW skill_usage AS
                SELECT
                    si.session_id,
                    si.ts,
                    si.message_uuid,
                    si.source,
                    si.skill_id,
                    si.args,
                    si.tool_use_id,
                    coalesce(cat.name, si.skill_id)               AS skill_name,
                    cat.plugin                                    AS plugin,
                    cat.plugin_version                            AS plugin_version,
                    cat.description                               AS description,
                    cat.source_kind                               AS source_kind,
                    coalesce(cat.source_kind = 'builtin', false)  AS is_builtin
                FROM skill_invocations si
                LEFT JOIN skills_catalog cat ON cat.skill_id = si.skill_id;
                """
            )
        else:
            con.execute(
                """
                CREATE OR REPLACE VIEW skill_usage AS
                SELECT
                    si.session_id,
                    si.ts,
                    si.message_uuid,
                    si.source,
                    si.skill_id,
                    si.args,
                    si.tool_use_id,
                    si.skill_id                          AS skill_name,
                    CAST(NULL AS VARCHAR)                AS plugin,
                    CAST(NULL AS VARCHAR)                AS plugin_version,
                    CAST(NULL AS VARCHAR)                AS description,
                    CAST(NULL AS VARCHAR)                AS source_kind,
                    false                                AS is_builtin
                FROM skill_invocations si;
                """
            )
        logger.debug("Registered analytics view: skill_usage")
    except duckdb.Error:
        logger.exception("Failed to register skill_usage view")


def register_all(
    con: duckdb.DuckDBPyConnection,
    *,
    settings: Settings | None = None,
    include_analytics: bool = True,
    skip_vss: bool = False,
) -> None:
    """Register raw views, derived views, VSS, analytics, and macros in order.

    Parameters
    ----------
    con
        Open DuckDB connection.
    settings
        Optional :class:`Settings`; a default instance is created when absent.
    include_analytics
        When ``True`` (default), call :func:`register_analytics` before
        :func:`register_macros` so the v2 analytics macros can bind against
        the freshly-registered analytics views.  Set to ``False`` to skip
        analytics view registration entirely (useful in tests that only
        exercise v1 macros or when the caller will register analytics views
        out-of-band).
    skip_vss
        When ``True``, skip :func:`register_vss` entirely *and* skip the
        ``semantic_search`` macro registration inside :func:`register_macros`.
        Set this for connections that won't run vector queries — it spares
        the ``INSTALL vss; LOAD vss; ATTACH hnsw.duckdb`` round-trips that
        otherwise dominate a non-vector ``query`` cold start.

    Notes
    -----
    Order matters on two axes:

    1. ``register_vss`` must run before ``register_macros`` because the
       ``semantic_search`` macro body references the ``message_embeddings``
       table and DuckDB resolves macro bodies at creation time. When
       ``skip_vss=True`` the ``semantic_search`` macro is skipped too, so
       the ordering still holds: nothing references a missing table.
    2. ``register_analytics`` must also run before ``register_macros`` so
       the analytics macros (``autonomy_trend``, ``cluster_top_terms``, ...)
       bind against the analytics views at macro-creation time.  When a
       parquet is missing the macro is skipped with a warning rather than
       raising.
    """
    settings = settings or Settings()
    if settings_need_s3(settings):
        configure_s3(con, settings)
    register_raw(
        con,
        glob=settings.default_glob,
        subagent_glob=settings.subagent_glob,
        subagent_meta_glob=settings.subagent_meta_glob,
    )
    register_views(con)
    if not skip_vss:
        register_vss(
            con,
            embeddings_parquet=settings.embeddings_parquet_path,
            lance_uri=settings.lance_uri,
            dim=int(settings.output_dimension),
            metric=settings.hnsw_metric,
        )
    if include_analytics:
        register_analytics(con, settings=settings)
    register_macros(con, settings=settings, skip_vss=skip_vss)


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def list_macros(con: duckdb.DuckDBPyConnection) -> list[tuple[str, tuple[str, ...]]]:
    """Return ``(name, params)`` for every registered macro.

    Parameter signatures come from :data:`MACRO_SIGNATURES` rather than
    DuckDB's catalog because ``duckdb_functions()`` returns NULL
    ``parameters`` for table macros (those defined with ``AS TABLE
    (...)``).  This function joins the catalog query with our static
    signature table.

    Macros not in :data:`MACRO_SIGNATURES` get an empty params tuple --
    they're either ad-hoc test macros or new macros that haven't been
    registered in the dict yet (the regex drift test in
    ``test_sql_views.py`` catches the second case in CI).

    Parameters
    ----------
    con
        Open DuckDB connection.

    Returns
    -------
    list[tuple[str, tuple[str, ...]]]
        Sorted list of ``(macro_name, parameter_names)`` pairs.  Includes
        both scalar and table macros.
    """
    rows = con.execute(
        """
        SELECT DISTINCT function_name
        FROM duckdb_functions()
        WHERE schema_name = 'main'
          AND function_type IN ('macro', 'table_macro')
        ORDER BY function_name
        """
    ).fetchall()
    return [(str(r[0]), MACRO_SIGNATURES.get(str(r[0]), ())) for r in rows]

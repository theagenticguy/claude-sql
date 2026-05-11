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
  table-function arguments); ``sample_size`` and ``maximum_object_size`` are
  likewise inlined (guarded by Python ``int`` typing).
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import duckdb
from loguru import logger

from claude_sql.config import DEFAULT_PRICING, Settings
from claude_sql.parquet_shards import iter_part_files

# ---------------------------------------------------------------------------
# Glob constants
# ---------------------------------------------------------------------------

DEFAULT_GLOB: str = os.path.expanduser("~/.claude/projects/*/*.jsonl")
SUBAGENT_GLOB: str = os.path.expanduser("~/.claude/projects/*/*/subagents/agent-*.jsonl")
SUBAGENT_META_GLOB: str = os.path.expanduser("~/.claude/projects/*/*/subagents/agent-*.meta.json")

# Business-level views emitted by ``register_views``. Used by the
# ``claude-sql schema`` subcommand for schema dumps.  Includes the v2
# analytics view names at the tail so ``describe_all`` can enumerate them
# once :func:`register_analytics` has populated the corresponding parquets.
VIEW_NAMES: tuple[str, ...] = (
    "sessions",
    "messages",
    "content_blocks",
    "messages_text",
    "tool_calls",
    "tool_results",
    "todo_events",
    "todo_state_current",
    "subagent_spawns",
    "task_creations",
    "task_updates",
    "tasks_state_current",
    "task_spawns",
    "skill_invocations",
    "subagent_sessions",
    "subagent_messages",
    # v2 analytics views (materialize when the matching parquet exists).
    "session_classifications",
    "session_goals",
    "message_trajectory",
    "session_conflicts",
    "message_clusters",
    "cluster_terms",
    "session_communities",
    "community_profile",
    "user_friction",
    "skills_catalog",
    "skill_usage",
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
# Drift is caught by :func:`tests.test_sql_views.test_view_schema_matches_describe_all`,
# which registers the v1 views over the fixture corpus, runs
# :func:`describe_all`, and asserts column-level equality with this dict.
# A contributor who edits view DDL without updating ``VIEW_SCHEMA`` gets
# a hard CI failure rather than a runtime mystery.
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
        ("parent_uuid", "JSON"),
        ("session_id", "UUID"),
        ("ts", "TIMESTAMP"),
        ("type", "VARCHAR"),
        ("is_sidechain", "BOOLEAN"),
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
        ("session_id", "UUID"),
        ("message_uuid", "VARCHAR"),
        ("ts", "TIMESTAMP"),
        ("role", "VARCHAR"),
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
        ("session_id", "UUID"),
        ("ts", "TIMESTAMP"),
        ("role", "VARCHAR"),
        ("text_content", "VARCHAR"),
    ),
    "tool_calls": (
        ("message_uuid", "VARCHAR"),
        ("session_id", "UUID"),
        ("ts", "TIMESTAMP"),
        ("tool_name", "VARCHAR"),
        ("tool_use_id", "VARCHAR"),
        ("tool_input", "JSON"),
    ),
    "tool_results": (
        ("message_uuid", "VARCHAR"),
        ("session_id", "UUID"),
        ("ts", "TIMESTAMP"),
        ("tool_use_id", "VARCHAR"),
        ("content", "JSON"),
    ),
    "todo_events": (
        ("session_id", "UUID"),
        ("written_at", "TIMESTAMP"),
        ("message_uuid", "VARCHAR"),
        ("subject", "VARCHAR"),
        ("status", "VARCHAR"),
        ("active_form", "VARCHAR"),
        ("snapshot_ix", "BIGINT"),
    ),
    "todo_state_current": (
        ("session_id", "UUID"),
        ("subject", "VARCHAR"),
        ("status", "VARCHAR"),
        ("active_form", "VARCHAR"),
        ("written_at", "TIMESTAMP"),
    ),
    "subagent_spawns": (
        ("session_id", "UUID"),
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
        ("session_id", "UUID"),
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
        ("session_id", "UUID"),
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
        ("session_id", "UUID"),
        ("task_id", "VARCHAR"),
        ("subject", "VARCHAR"),
        ("active_form", "VARCHAR"),
        ("status", "VARCHAR"),
        ("created_at", "TIMESTAMP"),
        ("last_updated_at", "TIMESTAMP"),
    ),
    "task_spawns": (
        ("session_id", "UUID"),
        ("spawned_at", "TIMESTAMP"),
        ("message_uuid", "VARCHAR"),
        ("tool_use_id", "VARCHAR"),
        ("spawn_tool", "VARCHAR"),
        ("subagent_type", "VARCHAR"),
        ("description", "VARCHAR"),
        ("prompt", "VARCHAR"),
    ),
    "skill_invocations": (
        ("session_id", "UUID"),
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
        ("parent_uuid", "JSON"),
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
    "message_clusters",
    "cluster_terms",
    "session_communities",
    "community_profile",
    "user_friction",
    "skills_catalog",
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
    "unused_skills",
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
    "unused_skills": ("last_n_days",),
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


def register_raw(
    con: duckdb.DuckDBPyConnection,
    *,
    glob: str | None = None,
    subagent_glob: str | None = None,
    subagent_meta_glob: str | None = None,
    sample_size: int = -1,
    maximum_object_size: int = 67_108_864,
) -> None:
    """Create the low-level ``v_raw_events`` and ``v_raw_subagents`` views.

    Both views are glob-driven zero-copy scans of JSONL via ``read_json`` with
    ``filename=true`` for file-level predicate pushdown. The subagent
    ``meta.json`` files are registered separately as ``v_raw_subagent_meta``
    so ``subagent_sessions`` can join them in.

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
    sample_size
        ``read_json`` schema-inference sample size. ``-1`` forces a full scan.
    maximum_object_size
        Maximum JSON object size in bytes (``read_json`` option). Must be an
        int so we can inline it safely.

    Raises
    ------
    duckdb.Error
        If any view DDL fails. Logged via ``logger.exception`` before re-raise.
    """
    glob = glob if glob is not None else DEFAULT_GLOB
    subagent_glob = subagent_glob if subagent_glob is not None else SUBAGENT_GLOB
    subagent_meta_glob = (
        subagent_meta_glob if subagent_meta_glob is not None else SUBAGENT_META_GLOB
    )

    # Inline numeric literals; type-narrow via int() to neutralize injection.
    sample_size_i = int(sample_size)
    max_obj_i = int(maximum_object_size)

    try:
        con.execute(
            f"""
            CREATE OR REPLACE VIEW v_raw_events AS
            SELECT *,
                   filename AS source_file,
                   regexp_extract(filename, '([^/]+)\\.jsonl$', 1) AS session_id_file
            FROM read_json(
                {_sql_str(glob)},
                format='newline_delimited',
                union_by_name=true,
                filename=true,
                ignore_errors=true,
                sample_size={sample_size_i},
                maximum_object_size={max_obj_i}
            );
            """
        )
        logger.debug(
            "Registered v_raw_events from glob {} with sample_size={}",
            glob,
            sample_size_i,
        )

        con.execute(
            f"""
            CREATE OR REPLACE VIEW v_raw_subagents AS
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
                sample_size={sample_size_i},
                maximum_object_size={max_obj_i}
            );
            """
        )
        logger.debug("Registered v_raw_subagents from glob {}", subagent_glob)

        # meta.json files are one object per file (not NDJSON) -> format='auto'.
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
    ``tasks_state_current``, ``task_spawns`` (deprecated alias),
    ``subagent_sessions``, ``subagent_messages``.

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
                parentUuid                                AS parent_uuid,
                sessionId                                 AS session_id,
                timestamp::TIMESTAMP                      AS ts,
                type,
                isSidechain                               AS is_sidechain,
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
                string_agg(cb.text, '\n\n')  AS text_content
            FROM content_blocks cb
            WHERE cb.block_type = 'text'
              AND cb.text IS NOT NULL
              AND length(cb.text) > 0
            GROUP BY cb.message_uuid
            HAVING length(string_agg(cb.text, '\n\n')) >= 32;
            """
        )
        logger.debug("Registered view: messages_text")

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

        # DEPRECATED: ``task_spawns`` predates the Task→Agent rename (v2.1.63)
        # and the TodoWrite→TaskCreate split (v2.1.16). It conflated subagent
        # launchers with task-tracker creation. Kept as a UNION ALL alias for
        # one release; new analytics should use ``subagent_spawns`` or
        # ``task_creations`` directly. Removed in the next minor release.
        con.execute(
            """
            CREATE OR REPLACE VIEW task_spawns AS
            SELECT
                session_id, spawned_at, message_uuid, tool_use_id,
                spawn_tool, subagent_type, description, prompt
            FROM subagent_spawns
            UNION ALL
            SELECT
                session_id, created_at AS spawned_at, message_uuid, tool_use_id,
                create_tool AS spawn_tool,
                NULL AS subagent_type,
                description,
                NULL AS prompt
            FROM task_creations;
            """
        )
        logger.debug("Registered view: task_spawns (deprecated)")

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
                parentUuid                  AS parent_uuid,
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


def _safe_macro(con: duckdb.DuckDBPyConnection, name: str, ddl: str) -> None:
    """Execute a ``CREATE OR REPLACE MACRO`` DDL, downgrading failures to warnings.

    Analytics macros reference views (``session_classifications``,
    ``cluster_terms``, etc.) that only materialize once the corresponding
    parquet has been produced.  Wrapping creation in ``try/except
    duckdb.Error`` means a fresh install (pre-``claude-sql classify``) can
    still call :func:`register_macros` without blowing up: the macro simply
    doesn't get created and the caller gets a ``logger.warning`` pointing at
    the missing backing view.

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
        logger.warning("Skipped macro {} (backing view missing): {}", name, exc)


def register_macros(
    con: duckdb.DuckDBPyConnection,
    settings: Settings | None = None,
) -> None:
    """Create SQL macros used by the CLI and analysts.

    v1 macros (always created): ``model_used``, ``cost_estimate``,
    ``tool_rank``, ``todo_velocity``, ``subagent_fanout``, ``semantic_search``.

    v2 analytics macros (created via :func:`_safe_macro`, skipped when their
    backing analytics view is missing): ``autonomy_trend``, ``work_mix``,
    ``success_rate_by_work``, ``cluster_top_terms``, ``community_top_topics``,
    ``sentiment_arc``.

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

    logger.debug(
        "Registered macros: ago, model_used, cost_estimate, tool_rank, "
        "todo_velocity, subagent_fanout, semantic_search, skill_rank, "
        "skill_source_mix"
    )

    # ------------------------------------------------------------------
    # v2 analytics macros -- each wrapped in _safe_macro so a missing
    # backing view (pre-``claude-sql classify`` run) is a warning, not an
    # exception.
    # ------------------------------------------------------------------

    # Time series: autonomy tier mix over rolling windows.
    _safe_macro(
        con,
        "autonomy_trend",
        """
        CREATE OR REPLACE MACRO autonomy_trend(window_days) AS TABLE (
            SELECT
                date_trunc('week', classified_at) AS week,
                autonomy_tier,
                count(*) AS n
            FROM session_classifications
            WHERE classified_at >= current_timestamp - (window_days * INTERVAL 1 DAY)
            GROUP BY 1, 2
            ORDER BY 1, 2
        );
        """,
    )

    # Work-category mix in the last N days.
    _safe_macro(
        con,
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
    )

    # Success / failure / partial rate broken down by work category.
    _safe_macro(
        con,
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
    )

    # Top-N TF-IDF terms for a single cluster.
    _safe_macro(
        con,
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
    )

    # Top cluster_ids within a given community, ranked by the number of
    # messages each cluster contributes to the community.  Each row carries
    # its top 5 TF-IDF terms for human-readable context.
    _safe_macro(
        con,
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
    )

    # Sentiment arc for a single session: per-message (ts, role, delta,
    # transition flag, confidence) in chronological order.
    _safe_macro(
        con,
        "sentiment_arc",
        """
        CREATE OR REPLACE MACRO sentiment_arc(sid) AS TABLE (
            SELECT m.ts, m.role, mt.sentiment_delta, mt.is_transition, mt.confidence
              FROM messages m
              JOIN message_trajectory mt
                ON CAST(m.uuid AS VARCHAR) = mt.uuid
             WHERE CAST(m.session_id AS VARCHAR) = sid
             ORDER BY m.ts
        );
        """,
    )

    # Counts per friction label, scoped to the last N days by message ``ts``
    # (the user's actual utterance time, not detected_at).  Pass ``NULL`` to
    # include the full corpus.  Excludes label='none' because that is the
    # majority sentinel class and would swamp the output.
    _safe_macro(
        con,
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
    )

    # Per-session friction pressure: how many non-'none' friction messages
    # fired vs the total user message count.  A high rate is a strong proxy
    # for a session where the agent repeatedly fell short of what the user
    # expected.
    _safe_macro(
        con,
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
    )

    # Top-N example user messages for a given friction label, highest
    # confidence first.  ``label_name`` is a VARCHAR so DuckDB callers
    # don't have to quote-escape through the macro boundary.
    _safe_macro(
        con,
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
    )

    # Catalog entries the user has NOT invoked in the last N days.  Pure
    # catalog lookup; ``skills_catalog`` may be missing pre-sync, so this
    # is wrapped in ``_safe_macro`` and skipped cleanly in that case.
    # ``source_kind`` filter keeps out the 'builtin' rows (users don't
    # install or uninstall ``/clear``).
    _safe_macro(
        con,
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
    )


# ---------------------------------------------------------------------------
# VSS
# ---------------------------------------------------------------------------


def _hnsw_rebuild_needed(parquet: Path, hnsw_db: Path) -> bool:
    """Decide from filesystem state alone whether the parquet has shifted.

    Handles both legacy single-file caches and sharded directories: for a
    sharded directory we compare against the *latest* part file's mtime so
    a brand-new shard invalidates the persisted HNSW even when the
    directory's own mtime hasn't moved (some filesystems update dir mtime
    only on add/remove, not on touch of children).

    This is a *necessary* but not sufficient signal — even when the
    parquet hasn't moved, the attached store might be empty (for instance,
    DuckDB's ATTACH on a missing path creates a ~12 KB header-only file
    before any tables exist). Catalog existence is checked separately
    inside ``register_vss`` after the ATTACH.
    """
    if not hnsw_db.exists():
        return True
    parts = iter_part_files(parquet)
    if not parts:
        # No source-of-truth on disk yet. The attached store is whatever
        # was previously persisted; nothing to rebuild from.
        return False
    latest_ns = max(p.stat().st_mtime_ns for p in parts)
    return latest_ns > hnsw_db.stat().st_mtime_ns


def _attached_embeddings_table_present(con: duckdb.DuckDBPyConnection) -> bool:
    """Return True when ``hnsw_store.main.message_embeddings`` exists in the catalog."""
    row = con.execute(
        """
        SELECT count(*)
        FROM duckdb_tables()
        WHERE database_name = 'hnsw_store'
          AND schema_name = 'main'
          AND table_name = 'message_embeddings';
        """
    ).fetchone()
    return bool(row and row[0])


def register_vss(
    con: duckdb.DuckDBPyConnection,
    *,
    embeddings_parquet: Path,
    hnsw_db_path: Path | None = None,
    dim: int = 1024,
    metric: str = "cosine",
    ef_construction: int = 128,
    ef_search: int = 64,
    m: int = 16,
    m0: int = 32,
) -> bool:
    """Install + load VSS and bind ``message_embeddings`` over a persisted HNSW store.

    When ``hnsw_db_path`` is provided the embeddings table and its HNSW
    index live inside that DuckDB file (ATTACHed under the alias
    ``hnsw_store``) so reopening a CLI command reuses the index instead of
    rebuilding it from parquet. The store is rebuilt only when missing,
    suspiciously small, or older than the embeddings parquet on disk; an
    ``IOException`` during attach unlinks the store and rebuilds.

    When ``hnsw_db_path`` is ``None`` (legacy / tests) the table and index
    stay in the connection's main database, matching the original
    in-memory behavior.

    Parameters
    ----------
    con
        Open DuckDB connection.
    embeddings_parquet
        Path to the embeddings parquet produced by ``claude-sql embed``.
    hnsw_db_path
        Persistent DuckDB file that backs the HNSW index, or ``None`` to
        keep everything in the connection's main database.
    dim
        Fixed-length embedding dimension. Must match the parquet's
        ``embedding`` column. Defaults to 1024 (Cohere Embed v4 mid-tier).
    metric
        HNSW distance metric. One of ``cosine``, ``l2sq``, ``ip``.
    ef_construction, ef_search, m, m0
        Standard HNSW tuning knobs. ``m`` and ``m0`` map to DuckDB's ``M``
        and ``M0`` parameters.

    Returns
    -------
    bool
        ``True`` if the table was populated and the HNSW index is usable;
        ``False`` if the parquet file does not exist yet.

    Notes
    -----
    VSS only supports ``FLOAT`` element type. Embeddings persisted as
    ``DOUBLE[]`` are cast via ``CAST(embedding AS FLOAT[<dim>])``.
    Persistence rides on the experimental
    ``hnsw_enable_experimental_persistence`` flag — when corruption
    surfaces, ``rm`` the file and the next call rebuilds from parquet.
    """
    dim_i = int(dim)
    ef_c_i = int(ef_construction)
    ef_s_i = int(ef_search)
    m_i = int(m)
    m0_i = int(m0)
    if metric not in {"cosine", "l2sq", "ip"}:
        raise ValueError(f"Unsupported HNSW metric: {metric!r}")

    con.execute("INSTALL vss;")
    con.execute("LOAD vss;")
    con.execute("SET hnsw_enable_experimental_persistence = true;")

    use_persistence = hnsw_db_path is not None
    schema_qualifier = ""
    persisted_path: Path | None = hnsw_db_path
    if use_persistence and persisted_path is not None:
        persisted_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            con.execute(f"ATTACH '{persisted_path}' AS hnsw_store;")
        except duckdb.IOException as exc:
            logger.warning(
                "ATTACH on {} failed ({}); unlinking and rebuilding the HNSW store.",
                persisted_path,
                exc,
            )
            with contextlib.suppress(FileNotFoundError):
                persisted_path.unlink()
            con.execute(f"ATTACH '{persisted_path}' AS hnsw_store;")
        # ``message_embeddings`` lives inside the attached store. Macros and
        # readers reference it via a top-level VIEW so existing call sites
        # (cli.py, the ``semantic_search`` macro) keep working unchanged.
        schema_qualifier = "hnsw_store.main."

    parts = iter_part_files(embeddings_parquet)
    if not parts:
        logger.warning(
            "No embeddings parquet at {}; skipping HNSW index build. "
            "Run `claude-sql embed` to backfill.",
            embeddings_parquet,
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {schema_qualifier}message_embeddings (
                uuid      VARCHAR PRIMARY KEY,
                model     VARCHAR,
                dim       USMALLINT,
                embedding FLOAT[{dim_i}]
            );
            """
        )
        if use_persistence:
            con.execute(
                "CREATE OR REPLACE VIEW message_embeddings AS "
                "SELECT * FROM hnsw_store.main.message_embeddings;"
            )
        return False

    rebuild = not use_persistence
    if use_persistence and persisted_path is not None:
        # Two reasons to rebuild: parquet is newer than the on-disk store,
        # or the attached store is empty (newly created header-only file
        # from ``ATTACH`` on a missing path).
        rebuild = _hnsw_rebuild_needed(
            embeddings_parquet, persisted_path
        ) or not _attached_embeddings_table_present(con)

    if rebuild:
        # Drop any stale table+index in the target schema first so
        # CREATE TABLE doesn't trip on an existing index. DROP TABLE
        # cascades to dependent indexes.
        con.execute(f"DROP TABLE IF EXISTS {schema_qualifier}message_embeddings;")
        # ``parts`` may be a single legacy file or a list of shard files.
        # Inline-escape each path because DDL doesn't accept prepared params.
        path_literals = ", ".join(_sql_str(str(p)) for p in parts)
        con.execute(
            f"""
            CREATE TABLE {schema_qualifier}message_embeddings AS
            SELECT
                uuid,
                model,
                dim,
                CAST(embedding AS FLOAT[{dim_i}]) AS embedding
            FROM read_parquet([{path_literals}]);
            """
        )
        con.execute(
            f"""
            CREATE INDEX idx_msg_hnsw
            ON {schema_qualifier}message_embeddings
            USING HNSW (embedding)
            WITH (
                metric='{metric}',
                ef_construction={ef_c_i},
                ef_search={ef_s_i},
                M={m_i},
                M0={m0_i}
            );
            """
        )
        if use_persistence:
            con.execute("CHECKPOINT hnsw_store;")

    if use_persistence:
        con.execute(
            "CREATE OR REPLACE VIEW message_embeddings AS "
            "SELECT * FROM hnsw_store.main.message_embeddings;"
        )

    row = con.execute(f"SELECT count(*) FROM {schema_qualifier}message_embeddings;").fetchone()
    count = int(row[0]) if row else 0
    logger.debug(
        "{} {} embeddings (metric={}, M={}, ef_search={}, persistent={})",
        "Built" if rebuild else "Reused persisted",
        count,
        metric,
        m_i,
        ef_s_i,
        use_persistence,
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
        "message_trajectory": ("*, sentiment_delta AS sentiment, is_transition AS transition"),
        "session_conflicts": ("*, resolution AS conflict_resolution"),
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
        # Sharded directories list every part file; legacy single-file paths
        # become a one-element list. ``read_parquet`` accepts both.
        parts = [p for p in iter_part_files(path) if p.stat().st_size > 16]
        path_literals = ", ".join(_sql_str(str(p)) for p in parts)
        try:
            con.execute(
                f"CREATE OR REPLACE VIEW {view_name} AS "
                f"SELECT {projection} FROM read_parquet([{path_literals}]);"
            )
            logger.debug("Registered analytics view: {} (source={})", view_name, path)
            registered.add(view_name)
        except duckdb.Error:
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

    Notes
    -----
    Order matters on two axes:

    1. ``register_vss`` must run before ``register_macros`` because the
       ``semantic_search`` macro body references the ``message_embeddings``
       table and DuckDB resolves macro bodies at creation time.
    2. ``register_analytics`` must also run before ``register_macros`` so
       the analytics macros (``autonomy_trend``, ``cluster_top_terms``, ...)
       bind against the analytics views at macro-creation time.  When a
       parquet is missing the macro is skipped with a warning rather than
       raising.
    """
    settings = settings or Settings()
    register_raw(
        con,
        glob=settings.default_glob,
        subagent_glob=settings.subagent_glob,
        subagent_meta_glob=settings.subagent_meta_glob,
    )
    register_views(con)
    register_vss(
        con,
        embeddings_parquet=settings.embeddings_parquet_path,
        hnsw_db_path=settings.hnsw_db_path,
        dim=int(settings.output_dimension),
        metric=settings.hnsw_metric,
        ef_construction=settings.hnsw_ef_construction,
        ef_search=settings.hnsw_ef_search,
        m=settings.hnsw_m,
        m0=settings.hnsw_m0,
    )
    if include_analytics:
        register_analytics(con, settings=settings)
    register_macros(con, settings=settings)


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def describe_all(con: duckdb.DuckDBPyConnection) -> dict[str, list[tuple[str, str]]]:
    """Return the column schema of every business-level view.

    Parameters
    ----------
    con
        Open DuckDB connection with views registered.

    Returns
    -------
    dict
        ``{view_name: [(column_name, column_type), ...]}``. Views that fail to
        describe (e.g. missing because ``register_views`` was not called) map
        to an empty list and emit a warning.
    """
    out: dict[str, list[tuple[str, str]]] = {}
    for name in VIEW_NAMES:
        try:
            rows = con.execute(f"DESCRIBE {name}").fetchall()
            out[name] = [(str(r[0]), str(r[1])) for r in rows]
        except duckdb.Error as exc:
            logger.warning("Could not describe {}: {}", name, exc)
            out[name] = []
    return out


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

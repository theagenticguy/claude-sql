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

import os
from pathlib import Path

import duckdb
from loguru import logger

from claude_sql.config import DEFAULT_PRICING, Settings

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
    "task_spawns",
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
    "user_friction",
)

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
    "user_friction",
)

# Macro names registered by :func:`register_macros`.  The first six are the
# v1 macros that ship unconditionally; the remaining six are the v2 analytics
# macros, each registered via :func:`_safe_macro` so a missing backing view
# downgrades to a warning instead of an exception.
MACRO_NAMES: tuple[str, ...] = (
    "model_used",
    "cost_estimate",
    "tool_rank",
    "todo_velocity",
    "subagent_fanout",
    "semantic_search",
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
)


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
    ``task_spawns``, ``subagent_sessions``, ``subagent_messages``.

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

        con.execute(
            """
            CREATE OR REPLACE VIEW task_spawns AS
            SELECT
                session_id,
                ts AS spawned_at,
                message_uuid,
                tool_use_id,
                tool_name AS spawn_tool,
                json_extract_string(tool_input, '$.subagent_type') AS subagent_type,
                json_extract_string(tool_input, '$.description')   AS description,
                json_extract_string(tool_input, '$.prompt')        AS prompt
            FROM tool_calls
            WHERE tool_name IN ('Task', 'Agent', 'TaskCreate', 'mcp__tasks__task_create');
            """
        )
        logger.debug("Registered view: task_spawns")

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

    logger.debug(
        "Registered macros: model_used, cost_estimate, tool_rank, "
        "todo_velocity, subagent_fanout, semantic_search"
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


# ---------------------------------------------------------------------------
# VSS
# ---------------------------------------------------------------------------


def register_vss(
    con: duckdb.DuckDBPyConnection,
    *,
    embeddings_parquet: Path,
    dim: int = 1024,
    metric: str = "cosine",
    ef_construction: int = 128,
    ef_search: int = 64,
    m: int = 16,
    m0: int = 32,
) -> bool:
    """Install + load VSS and build the HNSW index over ``message_embeddings``.

    Creates ``message_embeddings`` from ``embeddings_parquet`` (when present)
    and builds the HNSW index over the ``embedding`` column. When the parquet
    file is missing, an empty, schema-correct table is created so macros that
    reference ``message_embeddings`` still resolve -- ``semantic_search`` will
    simply return no rows until the parquet is populated.

    Parameters
    ----------
    con
        Open DuckDB connection.
    embeddings_parquet
        Path to the embeddings parquet produced by ``claude-sql embed``.
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
        ``True`` if the table was populated from parquet and the HNSW index
        built; ``False`` if the parquet file does not exist yet.

    Notes
    -----
    VSS only supports ``FLOAT`` element type. Embeddings persisted as
    ``DOUBLE[]`` are cast via ``CAST(embedding AS FLOAT[<dim>])``.
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

    if not embeddings_parquet.exists():
        logger.warning(
            "No embeddings parquet at {}; skipping HNSW index build. "
            "Run `claude-sql embed` to backfill.",
            embeddings_parquet,
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE message_embeddings (
                uuid      VARCHAR PRIMARY KEY,
                model     VARCHAR,
                dim       USMALLINT,
                embedding FLOAT[{dim_i}]
            );
            """
        )
        return False

    con.execute(
        f"""
        CREATE OR REPLACE TABLE message_embeddings AS
        SELECT
            uuid,
            model,
            dim,
            CAST(embedding AS FLOAT[{dim_i}]) AS embedding
        FROM read_parquet(?);
        """,
        [str(embeddings_parquet)],
    )
    con.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_msg_hnsw
        ON message_embeddings
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
    row = con.execute("SELECT count(*) FROM message_embeddings;").fetchone()
    count = int(row[0]) if row else 0
    logger.debug(
        "Loaded {} embeddings and built HNSW index (metric={}, M={}, ef_search={})",
        count,
        metric,
        m_i,
        ef_s_i,
    )
    return True


# ---------------------------------------------------------------------------
# v2 analytics views
# ---------------------------------------------------------------------------


def _parquet_is_populated(path: Path | None) -> bool:
    """Return True when ``path`` exists on disk with more than header-only bytes."""
    if path is None:
        return False
    return path.exists() and path.stat().st_size > 16


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
    user_friction_parquet: Path | None = None,
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
        "user_friction": user_friction_parquet
        if user_friction_parquet is not None
        else resolved.user_friction_parquet_path,
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
        try:
            con.execute(
                f"CREATE OR REPLACE VIEW {view_name} AS "
                f"SELECT * FROM read_parquet({_sql_str(str(path))});"
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


def list_macros(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Return the macro names defined in this connection's ``main`` schema.

    Parameters
    ----------
    con
        Open DuckDB connection.

    Returns
    -------
    list[str]
        Sorted, deduplicated list of macro function names (includes both
        scalar and table macros).
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
    return [str(r[0]) for r in rows]

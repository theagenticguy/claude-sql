"""DuckDB view, macro, and VSS registry for claude-sql.

Wires a DuckDB connection to the on-disk ``~/.claude/`` JSONL transcript corpus
and exposes it as a stable set of zero-copy SQL views, analytical macros, and
an HNSW-indexed embeddings table.

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
# ``claude-sql schema`` subcommand for schema dumps.
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
        logger.info(
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
        logger.info("Registered v_raw_subagents from glob {}", subagent_glob)

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
        logger.info("Registered v_raw_subagent_meta from glob {}", subagent_meta_glob)
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
        logger.info("Registered view: sessions")

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
        logger.info("Registered view: messages")

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
        logger.info("Registered view: content_blocks")

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
        logger.info("Registered view: messages_text")

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
        logger.info("Registered view: tool_calls")

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
        logger.info("Registered view: tool_results")

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
        logger.info("Registered view: todo_events")

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
        logger.info("Registered view: todo_state_current")

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
        logger.info("Registered view: task_spawns")

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
        logger.info("Registered view: subagent_sessions")

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
        logger.info("Registered view: subagent_messages")
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


def register_macros(
    con: duckdb.DuckDBPyConnection,
    settings: Settings | None = None,
) -> None:
    """Create SQL macros used by the CLI and analysts.

    Macros created: ``model_used``, ``cost_estimate``, ``tool_rank``,
    ``todo_velocity``, ``subagent_fanout``, ``semantic_search``.

    ``semantic_search(query_vec, k)`` is a table macro that returns the top-k
    uuids by cosine distance to ``query_vec`` using the HNSW index.
    ``query_vec`` must be ``FLOAT[<dim>]`` matching the ``message_embeddings``
    column type.

    Parameters
    ----------
    con
        Open DuckDB connection with views (and the ``message_embeddings``
        table from :func:`register_vss`) already registered.
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

    logger.info(
        "Registered macros: model_used, cost_estimate, tool_rank, "
        "todo_velocity, subagent_fanout, semantic_search"
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
    logger.info(
        "Loaded {} embeddings and built HNSW index (metric={}, M={}, ef_search={})",
        count,
        metric,
        m_i,
        ef_s_i,
    )
    return True


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def register_all(
    con: duckdb.DuckDBPyConnection,
    *,
    settings: Settings | None = None,
) -> None:
    """Register raw views, derived views, VSS, and macros in order.

    Parameters
    ----------
    con
        Open DuckDB connection.
    settings
        Optional :class:`Settings`; a default instance is created when absent.

    Notes
    -----
    Order matters: ``register_vss`` must run before ``register_macros`` because
    the ``semantic_search`` macro body references the ``message_embeddings``
    table and DuckDB resolves macro bodies at creation time.
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

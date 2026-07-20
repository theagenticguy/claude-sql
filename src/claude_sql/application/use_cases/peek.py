"""One-shot session-summary use-case for the ``peek`` CLI command.

Owns the DuckDB I/O the ``peek`` command used to inline: materialize a single
session's messages into a temp table, fan out the content blocks (tools +
text samples) from that slice, and assemble the summary dict the CLI emits.
The CLI command becomes parse -> call -> emit (T-8-1).

The projection-scoping subtlety this module protects (verified via EXPLAIN
ANALYZE) is why the session is materialized ONCE into ``_peek_msgs`` before the
UNNEST: DuckDB does not push the ``session_id`` predicate below the
``content_blocks`` lateral UNNEST, so the view-based shape would scan + UNNEST
the *whole corpus* for ``tool_calls`` and again for ``messages_text``. Scoping
to one session first turns two full-corpus UNNEST passes into one cheap
per-session pass (~5x on a 300-session corpus, byte-identical output).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import duckdb


def _truncate(text: str | None, sample_chars: int) -> str | None:
    if text is None:
        return None
    return text if len(text) <= sample_chars else text[: sample_chars - 1] + "…"


def peek_session(
    con: duckdb.DuckDBPyConnection,
    session_id: str,
    *,
    sample_chars: int,
    top_tools: int,
) -> dict[str, Any] | None:
    """Return the ``peek`` summary dict for ``session_id`` (``None`` if unknown).

    Parameters
    ----------
    con
        An open DuckDB connection with the ``messages`` view registered.
    session_id
        The target session to summarize.
    sample_chars
        Truncation width for the first/last text samples.
    top_tools
        Number of top tools (by call count) to report.

    Returns
    -------
    dict | None
        ``{session_id, source_file, total_lines, first_ts, last_ts,
        roles{role: count}, top_tools[{name, count}],
        samples{first_user, last_user, first_assistant_text}}`` -- or ``None``
        when the session_id is absent from the corpus (the CLI maps ``None`` to
        a ``not_found`` catalog error, exit 65).
    """
    # Materialize this session's messages ONCE, then derive the content-block
    # fan-out (tools, samples) from that slice. DuckDB does not push the
    # ``session_id`` predicate below the ``content_blocks`` lateral UNNEST
    # (verified via EXPLAIN ANALYZE: the filter lands *above* the WINDOW /
    # HASH_GROUP_BY, so the view-based shape TABLE_SCANs + UNNESTs the *whole
    # corpus* once for ``tool_calls`` and again for ``messages_text``). Scoping
    # to one session first turns two full-corpus UNNEST passes into one cheap
    # per-session pass -- ~5x on a 300-session corpus, byte-identical output.
    # The ``messages`` filter below *does* push down, so this single scan is the
    # only corpus-wide read. The block-level columns mirror the
    # ``content_blocks`` / ``tool_calls`` / ``messages_text`` view definitions in
    # ``infrastructure/duckdb_views.py``; keep them in sync if those views change.
    con.execute(
        "CREATE OR REPLACE TEMP TABLE _peek_msgs AS "
        "SELECT uuid, ts, role, type, is_compact_summary, "
        "       source_file, content_json "
        "FROM messages WHERE session_id = ?",
        [session_id],
    )
    header_row = con.execute(
        "SELECT COUNT(*) AS total_lines, "
        "MIN(ts) AS first_ts, MAX(ts) AS last_ts, "
        "ANY_VALUE(source_file) AS source_file "
        "FROM _peek_msgs"
    ).fetchone()
    total_lines = int((header_row or (0,))[0] or 0)
    if total_lines == 0:
        return None

    roles_rows = con.execute(
        "SELECT role, COUNT(*) AS n FROM _peek_msgs GROUP BY role ORDER BY n DESC, role"
    ).fetchall()
    # Session-scoped ``content_blocks`` equivalent: one UNNEST over the
    # materialized slice, shared by the tools and samples queries below.
    con.execute(
        "CREATE OR REPLACE TEMP TABLE _peek_blocks AS SELECT "
        "  uuid, ts, role, type AS message_type, "
        "  json_extract_string(block, '$.type') AS block_type, "
        "  json_extract_string(block, '$.text') AS text, "
        "  json_extract_string(block, '$.name') AS tool_name "
        "FROM _peek_msgs, "
        "     UNNEST(json_extract(content_json, '$[*]')) AS t(block)"
    )
    tools_rows = con.execute(
        "SELECT tool_name, COUNT(*) AS n FROM _peek_blocks "
        "WHERE block_type = 'tool_use' AND tool_name IS NOT NULL "
        "GROUP BY tool_name ORDER BY n DESC, tool_name LIMIT ?",
        [top_tools],
    ).fetchall()
    samples_rows = con.execute(
        "WITH mt AS ("
        " SELECT uuid, any_value(ts) AS ts, any_value(role) AS role,"
        "        string_agg(text, '\n\n') AS text_content"
        " FROM _peek_blocks"
        " WHERE block_type = 'text' AND text IS NOT NULL"
        "   AND length(text) > 0 AND message_type != 'attachment'"
        " GROUP BY uuid"
        " HAVING length(string_agg(text, '\n\n')) >= 32"
        "), ordered AS ("
        " SELECT role, ts, text_content,"
        "        row_number() OVER (PARTITION BY role ORDER BY ts, uuid) AS rn_asc,"
        "        row_number() OVER ("
        "          PARTITION BY role ORDER BY ts DESC, uuid DESC"
        "        ) AS rn_desc"
        " FROM mt"
        ") "
        "SELECT 'first_user' AS slot, ts, text_content FROM ordered "
        "WHERE role = 'user' AND rn_asc = 1 "
        "UNION ALL "
        "SELECT 'last_user', ts, text_content FROM ordered "
        "WHERE role = 'user' AND rn_desc = 1 "
        "UNION ALL "
        "SELECT 'first_assistant_text', ts, text_content FROM ordered "
        "WHERE role = 'assistant' AND rn_asc = 1"
    ).fetchall()
    samples: dict[str, dict[str, str | None] | None] = {
        "first_user": None,
        "last_user": None,
        "first_assistant_text": None,
    }
    for slot, ts, text in samples_rows:
        samples[str(slot)] = {
            "ts": str(ts) if ts is not None else None,
            "text": _truncate(str(text), sample_chars) if text is not None else None,
        }

    return {
        "session_id": session_id,
        "source_file": header_row[3] if header_row else None,
        "total_lines": total_lines,
        "first_ts": (str(header_row[1]) if header_row and header_row[1] is not None else None),
        "last_ts": (str(header_row[2]) if header_row and header_row[2] is not None else None),
        "roles": {str(role): int(n) for role, n in roles_rows},
        "top_tools": [{"name": str(name), "count": int(n)} for name, n in tools_rows],
        "samples": samples,
    }


__all__ = ["peek_session"]

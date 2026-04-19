"""Assemble a per-session text window suitable for LLM classification.

Reads v1's ``messages_text``, ``tool_calls``, and ``tool_results`` views,
interleaves them chronologically with role markers, and clips per the
``session_text_*`` settings so the total fits within Claude Sonnet 4.6's 1M
context window.

Public API
----------
build_session_text(con, session_id, *, settings) -> str
    Assemble one session.

iter_session_texts(con, *, settings, since_days=None, limit=None) -> Iterator[tuple[str, str]]
    Stream (session_id, text) for every session with any messages_text rows,
    newest first.  Skips sessions that produce an empty body.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    import duckdb

    from claude_sql.config import Settings


def _tool_input_preview(tool_input_json: str | None, max_chars: int = 400) -> str:
    """Truncate a tool_input JSON blob to the first ``max_chars`` for display."""
    if not tool_input_json:
        return ""
    s = str(tool_input_json)
    return s if len(s) <= max_chars else s[:max_chars] + "…(truncated)"


def _tool_result_preview(content_json: str | None, max_chars: int) -> str:
    """Truncate a tool_result content blob.

    Uses ``settings.session_text_tool_result_max_chars``.
    """
    if not content_json:
        return ""
    s = str(content_json)
    if len(s) <= max_chars:
        return s
    dropped = len(s) - max_chars
    return s[:max_chars] + "\n…(truncated, " + str(dropped) + " chars dropped)"


def build_session_text(
    con: duckdb.DuckDBPyConnection,
    session_id: str,
    *,
    settings: Settings,
) -> str:
    """Return a single newline-separated timeline of a session.

    Interleaves text blocks, tool calls, and tool results chronologically.
    Format::

        [user 2026-04-19T10:00:00] <text>
        [assistant 2026-04-19T10:00:05] <text>
        [tool_use:Bash tu-xxx] <tool_input preview>
        [tool_result tu-xxx] <content preview>
        ...

    Applies per-tool-result clipping (``settings.session_text_tool_result_max_chars``)
    and a total-length cap (``settings.session_text_total_max_chars``).
    """
    # One UNION-ALL to order everything by ts, then project a display line.
    sql = """
        WITH timeline AS (
            SELECT ts, role, 'text' AS kind, text_content AS body, NULL AS aux
              FROM messages_text
             WHERE CAST(session_id AS VARCHAR) = ?
            UNION ALL
            SELECT ts, 'tool' AS role, 'tool_use' AS kind,
                   CAST(tool_input AS VARCHAR) AS body,
                   tool_name AS aux
              FROM tool_calls
             WHERE CAST(session_id AS VARCHAR) = ?
            UNION ALL
            SELECT ts, 'tool' AS role, 'tool_result' AS kind,
                   CAST(content AS VARCHAR) AS body,
                   tool_use_id AS aux
              FROM tool_results
             WHERE CAST(session_id AS VARCHAR) = ?
        )
        SELECT ts, role, kind, body, aux
          FROM timeline
         ORDER BY ts
    """
    try:
        rows = con.execute(sql, [session_id, session_id, session_id]).fetchall()
    except duckdb.IOException as exc:
        # A JSONL file on the glob can be deleted between view registration and
        # the first query that materializes it.  Skip the session rather than
        # aborting the whole pipeline.
        logger.warning("build_session_text: skipping {} (IO error: {})", session_id, exc)
        return ""
    if not rows:
        return ""

    lines: list[str] = []
    running = 0
    cap = settings.session_text_total_max_chars
    tool_cap = settings.session_text_tool_result_max_chars

    for ts, role, kind, body, aux in rows:
        if body is None:
            continue
        if kind == "text":
            line = f"[{role} {ts.isoformat()}] {body}"
        elif kind == "tool_use":
            name = aux or "tool"
            line = f"[tool_use:{name} {ts.isoformat()}] {_tool_input_preview(body)}"
        else:  # tool_result
            tu_id = aux or "?"
            line = f"[tool_result {tu_id} {ts.isoformat()}] {_tool_result_preview(body, tool_cap)}"

        # Total-length cap — stop appending if we'd blow the budget.
        if running + len(line) + 1 > cap:
            lines.append(f"…(session truncated at {cap} chars, {len(rows)} events total)")
            break
        lines.append(line)
        running += len(line) + 1

    return "\n".join(lines)


def iter_session_texts(
    con: duckdb.DuckDBPyConnection,
    *,
    settings: Settings,
    since_days: int | None = None,
    limit: int | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield ``(session_id, text)`` for every session with at least one text message.

    Newest-first. Filters by ``since_days`` on ``messages_text.ts`` when given.
    Skips sessions that produce an empty body.
    """
    where = ["mt.text_content IS NOT NULL"]
    if since_days is not None:
        where.append(f"mt.ts >= current_timestamp - INTERVAL {int(since_days)} DAY")

    sql = f"""
        SELECT DISTINCT CAST(mt.session_id AS VARCHAR) AS sid,
               max(mt.ts) OVER (PARTITION BY mt.session_id) AS last_ts
          FROM messages_text mt
         WHERE {" AND ".join(where)}
         ORDER BY last_ts DESC
    """
    if limit is not None:
        sql += f"\nLIMIT {int(limit)}"

    session_ids = [r[0] for r in con.execute(sql).fetchall()]
    logger.info("iter_session_texts: {} sessions pending", len(session_ids))
    for sid in session_ids:
        text = build_session_text(con, sid, settings=settings)
        if text:
            yield sid, text

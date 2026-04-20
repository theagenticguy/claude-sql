"""Assemble per-session text windows for LLM classification.

Reads the v1 ``messages_text``, ``tool_calls``, and ``tool_results`` views,
interleaves them chronologically with role markers, and clips per the
``session_text_*`` settings so each session fits within Claude Sonnet 4.6's
1M-token context.

Why this module exists
----------------------
The naïve shape — one SQL round-trip per session — is quadratic against the
zero-copy ``read_json`` glob: every ``SELECT ... WHERE session_id = ?`` rescans
every JSONL file in the corpus.  On ~6K sessions that's unusable.  We
materialize the three source views into in-memory arrow tables **once** per
pipeline run, then do per-session slicing in Python.  One glob scan instead of
6K.

Public API
----------
session_text_corpus(con, *, since_days=None, limit=None) -> SessionTextCorpus
    Build an in-memory corpus of per-session timelines.  Call this once at
    the start of each classification / conflict / trajectory pipeline.

iter_session_texts(con, *, settings, since_days=None, limit=None) -> Iterator[tuple[str, str]]
    Thin wrapper that constructs a corpus and yields assembled texts.  Kept
    for callers that only want the ``(session_id, text)`` stream.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import duckdb
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from claude_sql.config import Settings


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _tool_input_preview(tool_input_json: str | None, max_chars: int = 400) -> str:
    """Truncate a ``tool_input`` JSON blob to the first ``max_chars``."""
    if not tool_input_json:
        return ""
    s = str(tool_input_json)
    return s if len(s) <= max_chars else s[:max_chars] + "…(truncated)"


def _tool_result_preview(content_json: str | None, max_chars: int) -> str:
    """Truncate a ``tool_result`` content blob with a "bytes dropped" footer."""
    if not content_json:
        return ""
    s = str(content_json)
    if len(s) <= max_chars:
        return s
    dropped = len(s) - max_chars
    return s[:max_chars] + "\n…(truncated, " + str(dropped) + " chars dropped)"


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _TimelineRow:
    """One event in a session's chronological timeline."""

    ts_iso: str
    role: str
    kind: str  # "text" | "tool_use" | "tool_result"
    body: str | None
    aux: str | None  # tool_name for tool_use, tool_use_id for tool_result


@dataclass(slots=True)
class SessionTextCorpus:
    """An in-memory corpus of per-session timelines, built with one glob scan.

    ``texts_by_session`` maps ``session_id`` → pre-sorted list of
    :class:`_TimelineRow`.  ``order`` is the newest-first session list that
    defines iteration order for downstream pipelines.
    """

    texts_by_session: dict[str, list[_TimelineRow]]
    order: list[str]

    def __len__(self) -> int:
        return len(self.order)

    def assemble(self, session_id: str, *, settings: Settings) -> str:
        """Render one session as a single newline-separated transcript.

        Applies :attr:`Settings.session_text_tool_result_max_chars` and the
        total-length cap :attr:`Settings.session_text_total_max_chars`.
        """
        rows = self.texts_by_session.get(session_id)
        if not rows:
            return ""

        cap = settings.session_text_total_max_chars
        tool_cap = settings.session_text_tool_result_max_chars
        lines: list[str] = []
        running = 0

        for row in rows:
            if row.body is None:
                continue
            if row.kind == "text":
                line = f"[{row.role} {row.ts_iso}] {row.body}"
            elif row.kind == "tool_use":
                name = row.aux or "tool"
                line = f"[tool_use:{name} {row.ts_iso}] {_tool_input_preview(row.body)}"
            else:  # tool_result
                tu_id = row.aux or "?"
                line = (
                    f"[tool_result {tu_id} {row.ts_iso}] {_tool_result_preview(row.body, tool_cap)}"
                )

            if running + len(line) + 1 > cap:
                lines.append(f"…(session truncated at {cap} chars, {len(rows)} events total)")
                break
            lines.append(line)
            running += len(line) + 1

        return "\n".join(lines)


def session_text_corpus(
    con: duckdb.DuckDBPyConnection,
    *,
    since_days: int | None = None,
    limit: int | None = None,
) -> SessionTextCorpus:
    """Materialize the per-session timeline corpus in one DuckDB round-trip.

    Parameters
    ----------
    con
        Open DuckDB connection with the v1 views registered.
    since_days
        Optional recency filter applied to ``messages_text.ts``.  Sessions
        whose most-recent text message is older than ``since_days`` are
        excluded from the corpus entirely.
    limit
        Cap the number of sessions (newest-first) returned in ``order``.

    Notes
    -----
    We issue three queries — one per source view (``messages_text``,
    ``tool_calls``, ``tool_results``) — filtered by the session-id window
    resolved from ``messages_text``.  Each result set is ordered by
    ``(session_id, ts)`` so we can stream it into a dict of lists without a
    per-row Python sort.  IO errors from stale JSONLs are caught and logged
    once with the session list that fell out.
    """
    order = _load_session_order(con, since_days=since_days, limit=limit)
    if not order:
        logger.info("session_text_corpus: 0 sessions matched the window")
        return SessionTextCorpus(texts_by_session={}, order=[])

    session_set = set(order)
    texts_by_session: dict[str, list[_TimelineRow]] = {sid: [] for sid in order}

    try:
        _load_messages_text(con, session_set, texts_by_session)
        _load_tool_calls(con, session_set, texts_by_session)
        _load_tool_results(con, session_set, texts_by_session)
    except duckdb.IOException as exc:
        # A JSONL on the glob can be deleted between view registration and
        # the materializing query.  Log and return whatever landed.
        logger.warning("session_text_corpus: IO error while materializing ({})", exc)

    # DuckDB already ordered by (session_id, ts), but each list got fed across
    # three separate queries so we sort once per session to stitch the three
    # streams into true chronological order.
    for rows in texts_by_session.values():
        rows.sort(key=lambda r: r.ts_iso)

    # Drop sessions that ended up with no rows at all -- keeps iteration
    # clean for downstream pipelines.
    non_empty_order = [sid for sid in order if texts_by_session.get(sid)]
    texts_by_session = {sid: texts_by_session[sid] for sid in non_empty_order}

    logger.info(
        "session_text_corpus: materialized {} sessions ({} with content)",
        len(order),
        len(non_empty_order),
    )
    return SessionTextCorpus(texts_by_session=texts_by_session, order=non_empty_order)


def _load_session_order(
    con: duckdb.DuckDBPyConnection,
    *,
    since_days: int | None,
    limit: int | None,
) -> list[str]:
    """Return the newest-first session id list for the requested window."""
    where = ["mt.text_content IS NOT NULL"]
    if since_days is not None:
        where.append(f"mt.ts >= current_timestamp - INTERVAL {int(since_days)} DAY")
    sql = f"""
        SELECT CAST(mt.session_id AS VARCHAR) AS sid,
               max(mt.ts) AS last_ts
          FROM messages_text mt
         WHERE {" AND ".join(where)}
         GROUP BY 1
         ORDER BY last_ts DESC
    """
    if limit is not None:
        sql += f"\nLIMIT {int(limit)}"
    return [r[0] for r in con.execute(sql).fetchall()]


def session_bounds(
    con: duckdb.DuckDBPyConnection,
    *,
    since_days: int | None = None,
    limit: int | None = None,
) -> dict[str, tuple[datetime | None, datetime | None]]:
    """Return ``{session_id: (last_ts, transcript_mtime)}`` for the window.

    ``last_ts`` is ``max(messages_text.ts)`` for the session. ``transcript_mtime``
    is ``os.stat(transcript_path).st_mtime`` for the JSONL file backing the
    session — or ``None`` when the file is unreadable (stale glob entry, etc).

    Used by the LLM worker pipelines to drive mtime-based checkpoint skip.
    """
    where = ["mt.text_content IS NOT NULL"]
    if since_days is not None:
        where.append(f"mt.ts >= current_timestamp - INTERVAL {int(since_days)} DAY")
    sql = f"""
        SELECT CAST(mt.session_id AS VARCHAR) AS sid,
               max(mt.ts) AS last_ts,
               any_value(s.transcript_path) AS transcript_path
          FROM messages_text mt
          LEFT JOIN sessions s ON CAST(s.session_id AS VARCHAR) = CAST(mt.session_id AS VARCHAR)
         WHERE {" AND ".join(where)}
         GROUP BY 1
         ORDER BY last_ts DESC
    """
    if limit is not None:
        sql += f"\nLIMIT {int(limit)}"
    out: dict[str, tuple[datetime | None, datetime | None]] = {}
    for sid, last_ts, path in con.execute(sql).fetchall():
        mtime: datetime | None = None
        if path:
            try:
                st = os.stat(path)
                mtime = datetime.fromtimestamp(st.st_mtime, tz=UTC)
            except OSError:
                mtime = None
        out[str(sid)] = (last_ts, mtime)
    return out


def _load_messages_text(
    con: duckdb.DuckDBPyConnection,
    session_set: set[str],
    out: dict[str, list[_TimelineRow]],
) -> None:
    """Stream text blocks for every session in ``session_set`` into ``out``."""
    sql = """
        SELECT CAST(mt.session_id AS VARCHAR) AS sid,
               mt.ts,
               mt.role,
               mt.text_content
          FROM messages_text mt
         WHERE CAST(mt.session_id AS VARCHAR) IN (SELECT unnest(?))
         ORDER BY sid, mt.ts
    """
    sids = list(session_set)
    for sid, ts, role, body in con.execute(sql, [sids]).fetchall():
        out[sid].append(
            _TimelineRow(
                ts_iso=ts.isoformat() if ts is not None else "",
                role=role or "user",
                kind="text",
                body=body,
                aux=None,
            )
        )


def _load_tool_calls(
    con: duckdb.DuckDBPyConnection,
    session_set: set[str],
    out: dict[str, list[_TimelineRow]],
) -> None:
    """Stream tool_use events for every session in ``session_set`` into ``out``."""
    sql = """
        SELECT CAST(tc.session_id AS VARCHAR) AS sid,
               tc.ts,
               tc.tool_name,
               CAST(tc.tool_input AS VARCHAR) AS tool_input_json
          FROM tool_calls tc
         WHERE CAST(tc.session_id AS VARCHAR) IN (SELECT unnest(?))
         ORDER BY sid, tc.ts
    """
    sids = list(session_set)
    for sid, ts, tool_name, tool_input_json in con.execute(sql, [sids]).fetchall():
        # Defensive: a session id can appear in tool_calls without being in
        # messages_text (tool-use-only probe sessions).  Skip those.
        rows = out.get(sid)
        if rows is None:
            continue
        rows.append(
            _TimelineRow(
                ts_iso=ts.isoformat() if ts is not None else "",
                role="tool",
                kind="tool_use",
                body=tool_input_json,
                aux=tool_name,
            )
        )


def _load_tool_results(
    con: duckdb.DuckDBPyConnection,
    session_set: set[str],
    out: dict[str, list[_TimelineRow]],
) -> None:
    """Stream tool_result events for every session in ``session_set`` into ``out``."""
    sql = """
        SELECT CAST(tr.session_id AS VARCHAR) AS sid,
               tr.ts,
               tr.tool_use_id,
               CAST(tr.content AS VARCHAR) AS content_json
          FROM tool_results tr
         WHERE CAST(tr.session_id AS VARCHAR) IN (SELECT unnest(?))
         ORDER BY sid, tr.ts
    """
    sids = list(session_set)
    for sid, ts, tool_use_id, content_json in con.execute(sql, [sids]).fetchall():
        rows = out.get(sid)
        if rows is None:
            continue
        rows.append(
            _TimelineRow(
                ts_iso=ts.isoformat() if ts is not None else "",
                role="tool",
                kind="tool_result",
                body=content_json,
                aux=tool_use_id,
            )
        )


# ---------------------------------------------------------------------------
# Stream adapter
# ---------------------------------------------------------------------------


def iter_session_texts(
    con: duckdb.DuckDBPyConnection,
    *,
    settings: Settings,
    since_days: int | None = None,
    limit: int | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield ``(session_id, text)`` for every session with at least one text block.

    Newest-first.  Internally materializes a :class:`SessionTextCorpus` — one
    glob scan regardless of how many sessions match the window.
    """
    corpus = session_text_corpus(con, since_days=since_days, limit=limit)
    for sid in corpus.order:
        text = corpus.assemble(sid, settings=settings)
        if text:
            yield sid, text

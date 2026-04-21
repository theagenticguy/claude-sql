"""Per-(session_id, pipeline) checkpoint backed by a persistent DuckDB file.

Tracks when each LLM pipeline last processed each session so re-runs skip
sessions whose transcripts have not advanced. One row per
``(session_id, pipeline)``; ``INSERT OR REPLACE`` is the upsert primitive.

Schema::

    CREATE TABLE session_checkpoint (
        session_id            VARCHAR,
        pipeline              VARCHAR,
        last_ts_processed     TIMESTAMP,
        last_mtime_processed  TIMESTAMP,
        completed_at          TIMESTAMP NOT NULL,
        PRIMARY KEY (session_id, pipeline)
    );

All timestamps are UTC. Plain ``TIMESTAMP`` (not ``TIMESTAMP WITH TIME
ZONE``) because DuckDB's tz-aware type requires ``pytz`` at query time —
an extra dep we don't want. We stash tz-aware UTC datetimes by stripping
``tzinfo`` at the boundary and re-attaching ``UTC`` on read.

The file lives at ``~/.claude/claude_sql.duckdb`` (overridable via
``CLAUDE_SQL_CHECKPOINT_DB_PATH``).
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import duckdb

PIPELINE_NAMES: tuple[str, ...] = ("classify", "trajectory", "conflicts")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS session_checkpoint (
    session_id            VARCHAR   NOT NULL,
    pipeline              VARCHAR   NOT NULL,
    last_ts_processed     TIMESTAMP,
    last_mtime_processed  TIMESTAMP,
    completed_at          TIMESTAMP NOT NULL,
    PRIMARY KEY (session_id, pipeline)
);
"""


def _strip_tz(dt: datetime | None) -> datetime | None:
    """Drop tz so DuckDB's naive TIMESTAMP round-trips without pytz."""
    if dt is None:
        return None
    return dt.astimezone(UTC).replace(tzinfo=None)


def _attach_tz(dt: datetime | None) -> datetime | None:
    """Re-attach UTC on read so callers always get aware datetimes back."""
    if dt is None:
        return None
    return dt.replace(tzinfo=UTC)


def _connect(path: Path) -> duckdb.DuckDBPyConnection:
    """Open the checkpoint DB and ensure the table exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(_CREATE_TABLE_SQL)
    return con


def load_as_map(db_path: Path, pipeline: str) -> dict[str, tuple[datetime | None, datetime | None]]:
    """Return ``{session_id: (last_ts, last_mtime)}`` for one pipeline.

    Empty dict when the DB doesn't exist yet or the pipeline has no rows.
    """
    if not db_path.exists():
        return {}
    con = _connect(db_path)
    try:
        rows = con.execute(
            "SELECT session_id, last_ts_processed, last_mtime_processed "
            "FROM session_checkpoint WHERE pipeline = ?",
            [pipeline],
        ).fetchall()
    finally:
        con.close()
    return {
        str(sid): (_attach_tz(last_ts), _attach_tz(last_mtime)) for sid, last_ts, last_mtime in rows
    }


def filter_unchanged(
    candidates: Iterable[tuple[str, datetime | None, datetime | None]],
    *,
    pipeline: str,
    checkpoint_db_path: Path,
) -> tuple[list[str], int]:
    """Drop sessions whose ``(last_ts, last_mtime)`` has not advanced.

    ``candidates`` is an iterable of ``(session_id, current_last_ts,
    current_last_mtime)``. Returns ``(pending_session_ids, skipped_count)``.

    A session is skipped iff a checkpoint row exists for ``pipeline`` AND
    both ``current_last_ts <= ckpt.last_ts`` AND ``current_last_mtime <=
    ckpt.last_mtime``. Either bound moving forward invalidates the skip.
    """
    ckpt = load_as_map(checkpoint_db_path, pipeline)
    pending: list[str] = []
    skipped = 0
    for sid, cur_ts, cur_mtime in candidates:
        prev = ckpt.get(sid)
        if prev is None:
            pending.append(sid)
            continue
        prev_ts, prev_mtime = prev
        ts_ok = cur_ts is not None and prev_ts is not None and cur_ts <= prev_ts
        mtime_ok = cur_mtime is not None and prev_mtime is not None and cur_mtime <= prev_mtime
        if ts_ok and mtime_ok:
            skipped += 1
            continue
        pending.append(sid)
    return pending, skipped


def mark_completed(
    db_path: Path,
    *,
    pipeline: str,
    rows: Iterable[tuple[str, datetime | None, datetime | None]],
) -> int:
    """Upsert checkpoint rows for ``(session_id, pipeline)``.

    Each row is ``(session_id, last_ts_processed, last_mtime_processed)``.
    The ``completed_at`` column is stamped with ``datetime.now(UTC)``.

    Returns the number of upserted rows. When ``rows`` is empty, the DB is
    left untouched.
    """
    incoming = list(rows)
    if not incoming:
        return 0
    now = datetime.now(UTC).replace(tzinfo=None)
    payload = [
        (sid, pipeline, _strip_tz(last_ts), _strip_tz(last_mtime), now)
        for sid, last_ts, last_mtime in incoming
    ]
    con = _connect(db_path)
    try:
        con.executemany(
            "INSERT OR REPLACE INTO session_checkpoint "
            "(session_id, pipeline, last_ts_processed, last_mtime_processed, completed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            payload,
        )
    finally:
        con.close()
    return len(incoming)


def count_rows(db_path: Path) -> int:
    """Return the total number of checkpoint rows, or 0 when the DB is missing."""
    if not db_path.exists():
        return 0
    con = _connect(db_path)
    try:
        row = con.execute("SELECT count(*) FROM session_checkpoint").fetchone()
    finally:
        con.close()
    return int(row[0]) if row else 0

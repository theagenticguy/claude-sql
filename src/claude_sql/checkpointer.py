"""Per-(session_id, pipeline) checkpoint backed by a SQLite WAL file.

Tracks when each LLM pipeline last processed each session so re-runs skip
sessions whose transcripts have not advanced. One row per
``(session_id, pipeline)``; ``INSERT ... ON CONFLICT DO UPDATE`` is the upsert
primitive (UPSERT, supported in SQLite 3.24+).

Schema::

    CREATE TABLE session_checkpoint (
        session_id            TEXT NOT NULL,
        pipeline              TEXT NOT NULL,
        last_ts_processed     TEXT,
        last_mtime_processed  TEXT,
        completed_at          TEXT NOT NULL,
        PRIMARY KEY (session_id, pipeline)
    );

All timestamps are stored as ISO-8601 UTC strings (``2026-05-11T19:32:00.001Z``).
They sort lexicographically. We use the stdlib ``sqlite3`` module with WAL
journal mode so multiple readers can run concurrently with one writer — the
DuckDB single-writer file lock used to force a 20× retry storm under parallel
classify/trajectory/conflicts pipelines; SQLite's ``busy_timeout`` pragma
absorbs transient writer contention transparently in microseconds.

The file lives at ``~/.claude/state.db`` (overridable via
``CLAUDE_SQL_CHECKPOINT_DB_PATH``). On first connect, if the legacy
``~/.claude/claude_sql.duckdb`` exists, its contents are migrated once.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

PIPELINE_NAMES: tuple[str, ...] = ("classify", "trajectory", "conflicts", "user_friction")

_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS session_checkpoint (
    session_id            TEXT NOT NULL,
    pipeline              TEXT NOT NULL,
    last_ts_processed     TEXT,
    last_mtime_processed  TEXT,
    completed_at          TEXT NOT NULL,
    PRIMARY KEY (session_id, pipeline)
);
"""

_LEGACY_DUCKDB_FILENAME = "claude_sql.duckdb"
_MIGRATION_SENTINEL = ".migrated_from_duckdb"

# Process-local set of paths whose schema has already been bootstrapped this
# process. Skipping the redundant ``CREATE TABLE IF NOT EXISTS`` on every
# ``_connect`` call avoids racing the writer lock when concurrent threads
# open the same file. Guarded by a lock so the very first concurrent opens
# don't both win the "not bootstrapped yet" check and double-issue DDL.
_SCHEMA_BOOTSTRAPPED: set[str] = set()
_SCHEMA_BOOTSTRAP_LOCK = threading.Lock()


def _to_iso(dt: datetime | None) -> str | None:
    """ISO-8601 UTC string, or None."""
    if dt is None:
        return None
    return dt.astimezone(UTC).isoformat()


def _from_iso(s: str | None) -> datetime | None:
    """Parse an ISO-8601 string back to a tz-aware UTC datetime."""
    if s is None:
        return None
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _legacy_duckdb_path(new_path: Path) -> Path:
    return new_path.parent / _LEGACY_DUCKDB_FILENAME


def _migration_sentinel_path(new_path: Path) -> Path:
    return new_path.parent / _MIGRATION_SENTINEL


def _migrate_from_duckdb_if_present(new_path: Path) -> None:
    """One-time copy from legacy DuckDB file. Idempotent.

    Reads ``session_checkpoint`` and ``retry_queue`` from the legacy DuckDB,
    writes both into the SQLite file, drops a sentinel so we don't retry on
    every open. Failures log and skip — never block the caller.
    """
    sentinel = _migration_sentinel_path(new_path)
    if sentinel.exists():
        return
    legacy = _legacy_duckdb_path(new_path)
    if not legacy.exists():
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.touch()
        return
    try:
        import duckdb  # local import — only needed for one-time migration
    except ImportError:
        # No duckdb at runtime — the codebase needs it elsewhere, but if the
        # import ever fails here we still want SQLite to come up.
        sentinel.touch()
        return

    checkpoint_rows: list[tuple] = []
    retry_rows: list[tuple] = []
    try:
        old = duckdb.connect(str(legacy), read_only=True)
        try:
            try:
                checkpoint_rows = [
                    (
                        str(sid),
                        str(pipeline),
                        _to_iso(last_ts) if isinstance(last_ts, datetime) else last_ts,
                        _to_iso(last_mtime) if isinstance(last_mtime, datetime) else last_mtime,
                        _to_iso(completed_at)
                        if isinstance(completed_at, datetime)
                        else completed_at,
                    )
                    for sid, pipeline, last_ts, last_mtime, completed_at in old.execute(
                        "SELECT session_id, pipeline, last_ts_processed, "
                        "last_mtime_processed, completed_at FROM session_checkpoint"
                    ).fetchall()
                ]
            except duckdb.CatalogException:
                checkpoint_rows = []
            try:
                retry_rows = [
                    (
                        str(pipeline),
                        str(unit_id),
                        str(error),
                        int(attempts),
                        _to_iso(next_at) if isinstance(next_at, datetime) else next_at,
                        _to_iso(created_at) if isinstance(created_at, datetime) else created_at,
                        _to_iso(completed_at)
                        if isinstance(completed_at, datetime)
                        else completed_at,
                    )
                    for pipeline, unit_id, error, attempts, next_at, created_at, completed_at in old.execute(
                        "SELECT pipeline, unit_id, error, attempts, next_attempt_at, "
                        "created_at, completed_at FROM retry_queue"
                    ).fetchall()
                ]
            except duckdb.CatalogException:
                retry_rows = []
        finally:
            old.close()
    except Exception:  # noqa: BLE001 — migration is best-effort; any failure must drop the sentinel and let SQLite come up clean
        logger.exception("Failed to read legacy DuckDB at {} for migration", legacy)
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.touch()
        return

    new_path.parent.mkdir(parents=True, exist_ok=True)
    # Open in autocommit so the PRAGMA setup runs outside a transaction
    # (``PRAGMA journal_mode=WAL`` is a no-op inside one). After that, wrap
    # the bulk INSERTs in a single explicit BEGIN/COMMIT — without it,
    # autocommit treats every executemany row as its own transaction and
    # fsyncs the WAL per row, turning a 39k-row migration into a many-minute
    # hang. Single-transaction batch lands in <1s on the live corpus.
    con = sqlite3.connect(str(new_path), isolation_level=None)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute(_CREATE_TABLES_SQL)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS retry_queue (
                pipeline        TEXT    NOT NULL,
                unit_id         TEXT    NOT NULL,
                error           TEXT    NOT NULL,
                attempts        INTEGER NOT NULL DEFAULT 0,
                next_attempt_at TEXT    NOT NULL,
                created_at      TEXT    NOT NULL,
                completed_at    TEXT,
                PRIMARY KEY (pipeline, unit_id)
            );
            """
        )
        if checkpoint_rows or retry_rows:
            con.execute("BEGIN")
            try:
                if checkpoint_rows:
                    con.executemany(
                        "INSERT OR REPLACE INTO session_checkpoint "
                        "(session_id, pipeline, last_ts_processed, last_mtime_processed, completed_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        checkpoint_rows,
                    )
                if retry_rows:
                    con.executemany(
                        "INSERT OR REPLACE INTO retry_queue "
                        "(pipeline, unit_id, error, attempts, next_attempt_at, created_at, completed_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        retry_rows,
                    )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
    finally:
        con.close()
    sentinel.touch()
    logger.info(
        "Migrated {} checkpoint rows + {} retry rows from {} to {}",
        len(checkpoint_rows),
        len(retry_rows),
        legacy,
        new_path,
    )


def _connect(path: Path, *, max_attempts: int = 20) -> sqlite3.Connection:
    """Open the SQLite checkpoint DB and ensure tables exist.

    SQLite WAL mode allows many concurrent readers + one writer. Transient
    write contention is absorbed by ``PRAGMA busy_timeout=5000`` (5s wait),
    so the 20× exponential-backoff retry loop the DuckDB version needed is
    gone. ``max_attempts`` is kept as a no-op kwarg for back-compat.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _migrate_from_duckdb_if_present(path)
    # Connect in autocommit (isolation_level=None) so PRAGMAs run outside a
    # transaction (``PRAGMA journal_mode=WAL`` can't be set from within one)
    # and so concurrent writers serialize through ``busy_timeout`` rather
    # than tripping Python's implicit-transaction wrapping. Each `execute`
    # is then its own atomic statement; ``executemany`` is wrapped in an
    # implicit BEGIN/COMMIT by SQLite. WAL guarantees readers never block.
    # Open in autocommit for PRAGMA setup (WAL can't be set inside a
    # transaction), then switch to deferred mode so subsequent writes wrap
    # in implicit BEGIN/COMMIT and respect ``timeout=`` when waiting on the
    # writer lock.
    con = sqlite3.connect(str(path), isolation_level=None, timeout=30.0)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA busy_timeout=10000")
    con.execute("PRAGMA foreign_keys=ON")
    # Bootstrap schema only once per process per path — running the DDL on
    # every open serializes concurrent writers on the writer lock even with
    # ``CREATE TABLE IF NOT EXISTS``, which is the contention pattern that
    # made this whole module move off DuckDB. Cached set keeps the hot path
    # single-statement.
    key = str(path.resolve())
    with _SCHEMA_BOOTSTRAP_LOCK:
        if key not in _SCHEMA_BOOTSTRAPPED:
            con.execute(_CREATE_TABLES_SQL)
            _SCHEMA_BOOTSTRAPPED.add(key)
    con.isolation_level = (
        "DEFERRED"  # writes wrap in implicit BEGIN/COMMIT; lock contention waits on timeout
    )
    return con


def load_as_map(db_path: Path, pipeline: str) -> dict[str, tuple[datetime | None, datetime | None]]:
    """Return ``{session_id: (last_ts, last_mtime)}`` for one pipeline.

    Empty dict when the DB doesn't exist yet or the pipeline has no rows.
    """
    if not db_path.exists() and not _legacy_duckdb_path(db_path).exists():
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
        str(sid): (_from_iso(last_ts), _from_iso(last_mtime)) for sid, last_ts, last_mtime in rows
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
        if _stale_or_equal(cur_ts, prev_ts) and _stale_or_equal(cur_mtime, prev_mtime):
            skipped += 1
            continue
        pending.append(sid)
    return pending, skipped


def _stale_or_equal(cur: datetime | None, prev: datetime | None) -> bool:
    """True iff both are present and ``cur`` has not advanced past ``prev``.

    Both inputs are tz-aware UTC datetimes after the boundary helpers have
    run; we compare directly. None on either side returns False (advance).
    """
    if cur is None or prev is None:
        return False
    cur_aware = cur.astimezone(UTC) if cur.tzinfo else cur.replace(tzinfo=UTC)
    prev_aware = prev.astimezone(UTC) if prev.tzinfo else prev.replace(tzinfo=UTC)
    return cur_aware <= prev_aware


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
    now_iso = _to_iso(datetime.now(UTC))
    payload = [
        (sid, pipeline, _to_iso(last_ts), _to_iso(last_mtime), now_iso)
        for sid, last_ts, last_mtime in incoming
    ]
    con = _connect(db_path)
    try:
        con.executemany(
            "INSERT INTO session_checkpoint "
            "(session_id, pipeline, last_ts_processed, last_mtime_processed, completed_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(session_id, pipeline) DO UPDATE SET "
            "last_ts_processed = excluded.last_ts_processed, "
            "last_mtime_processed = excluded.last_mtime_processed, "
            "completed_at = excluded.completed_at",
            payload,
        )
        con.commit()
    finally:
        con.close()
    return len(incoming)


def count_rows(db_path: Path) -> int:
    """Return the total number of checkpoint rows, or 0 when the DB is missing."""
    if not db_path.exists() and not _legacy_duckdb_path(db_path).exists():
        return 0
    con = _connect(db_path)
    try:
        row = con.execute("SELECT count(*) FROM session_checkpoint").fetchone()
    finally:
        con.close()
    return int(row[0]) if row else 0

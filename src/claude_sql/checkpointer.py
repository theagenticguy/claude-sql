"""Per-session checkpoint parquet.

Tracks one row per ``(session_id, pipeline)`` recording the session's
``last_ts`` (max ``mt.ts`` we've seen) and ``last_mtime`` (transcript JSONL
filesystem mtime) at the moment the pipeline last finished. The worker
pipelines (``classify``, ``trajectory``, ``conflicts``) read the checkpoint
before enumerating pending work and skip any session whose
``(last_ts, last_mtime)`` has not advanced since the last run.

Schema::

    session_id         VARCHAR     # key
    pipeline           VARCHAR     # "classify" | "trajectory" | "conflicts"
    last_ts_processed  TIMESTAMP   # max messages_text.ts the pipeline saw
    last_mtime_processed TIMESTAMP # transcript JSONL mtime at run time
    completed_at       TIMESTAMP   # wall-clock of the run that wrote the row

Semantics are upsert-by-``(session_id, pipeline)``: a second call replaces
the prior row, so the file stays small (hundreds of rows, not historical
audit).
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from loguru import logger

PIPELINE_NAMES: tuple[str, ...] = ("classify", "trajectory", "conflicts")

_SCHEMA: dict[str, pl.DataType | type] = {
    "session_id": pl.Utf8,
    "pipeline": pl.Utf8,
    "last_ts_processed": pl.Datetime("us", "UTC"),
    "last_mtime_processed": pl.Datetime("us", "UTC"),
    "completed_at": pl.Datetime("us", "UTC"),
}


def _parquet_is_populated(path: Path) -> bool:
    """Match the ``>16 bytes`` sentinel used throughout the project."""
    return path.exists() and path.stat().st_size > 16


def load(path: Path) -> pl.DataFrame:
    """Return the checkpoint DataFrame, or an empty frame with the right schema."""
    if not _parquet_is_populated(path):
        return pl.DataFrame(schema=_SCHEMA)
    try:
        return pl.read_parquet(path)
    except (OSError, pl.exceptions.ComputeError):
        logger.warning("checkpoint: unreadable parquet at {} — treating as empty", path)
        return pl.DataFrame(schema=_SCHEMA)


def load_as_map(path: Path, pipeline: str) -> dict[str, tuple[datetime | None, datetime | None]]:
    """Return ``{session_id: (last_ts, last_mtime)}`` for one pipeline.

    Timestamps may be ``None`` for rows that were written with missing
    metadata — callers treat ``None`` as "unknown, re-process".
    """
    df = load(path)
    if df.is_empty():
        return {}
    mine = df.filter(pl.col("pipeline") == pipeline)
    if mine.is_empty():
        return {}
    out: dict[str, tuple[datetime | None, datetime | None]] = {}
    for row in mine.iter_rows(named=True):
        out[str(row["session_id"])] = (
            row["last_ts_processed"],
            row["last_mtime_processed"],
        )
    return out


def filter_unchanged(
    candidates: Iterable[tuple[str, datetime | None, datetime | None]],
    *,
    pipeline: str,
    checkpoint_path: Path,
) -> tuple[list[str], int]:
    """Drop sessions whose ``(last_ts, last_mtime)`` has not advanced.

    ``candidates`` is an iterable of ``(session_id, current_last_ts,
    current_last_mtime)``. Returns ``(pending_session_ids, skipped_count)``.

    A session is skipped iff a checkpoint row exists for ``pipeline`` AND
    both ``current_last_ts <= ckpt.last_ts`` AND ``current_last_mtime <=
    ckpt.last_mtime``. Either bound moving forward invalidates the skip.
    """
    ckpt = load_as_map(checkpoint_path, pipeline)
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
    path: Path,
    *,
    pipeline: str,
    rows: Iterable[tuple[str, datetime | None, datetime | None]],
) -> int:
    """Upsert checkpoint rows for ``(session_id, pipeline)``.

    Each row is ``(session_id, last_ts_processed, last_mtime_processed)``.
    The ``completed_at`` column is stamped with ``datetime.now(UTC)``.

    Returns the number of upserted rows. When ``rows`` is empty, the file is
    left untouched.
    """
    incoming = list(rows)
    if not incoming:
        return 0
    now = datetime.now(UTC)
    df = pl.DataFrame(
        [
            {
                "session_id": sid,
                "pipeline": pipeline,
                "last_ts_processed": last_ts,
                "last_mtime_processed": last_mtime,
                "completed_at": now,
            }
            for sid, last_ts, last_mtime in incoming
        ],
        schema=_SCHEMA,
    )
    existing = load(path)
    if existing.is_empty():
        merged = df
    else:
        # Anti-join + concat: stable across polars versions, same result as
        # ``existing.update(df, on=["session_id","pipeline"], how="full")``.
        keys = ["session_id", "pipeline"]
        kept = existing.join(df.select(keys), on=keys, how="anti")
        merged = pl.concat([kept, df], how="diagonal_relaxed")
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(path)
    return len(incoming)

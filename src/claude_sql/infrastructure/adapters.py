"""Concrete storage-port adapters + their default factories.

The four storage ports declared in :mod:`claude_sql.application.ports` —
:class:`~claude_sql.application.ports.CachePort`,
:class:`~claude_sql.application.ports.CheckpointPort`,
:class:`~claude_sql.application.ports.RetryQueuePort`, and
:class:`~claude_sql.application.ports.VectorStorePort` — are satisfied here by
thin adapters that bind the resource handle (a parquet cache target, the
SQLite ``state.db`` path, or the LanceDB uri/dim) in the constructor and
delegate every method to the existing module-level functions in
``parquet_cache`` / ``sqlite_state`` / ``lance_store``.

Delegation is by **call-time module-attribute lookup** (``retry_queue.enqueue``,
not a captured ``from ... import enqueue`` reference). That is load-bearing: the
existing test suite monkeypatches the module functions (e.g.
``monkeypatch.setattr(retry_queue, "enqueue", fake)``), and resolving the name
on the module object at call time keeps those patches biting through the port
seam. Matches the injectable-default pattern T-4-1 used for
:class:`~claude_sql.application.analyze.DuckDbReader` (default resolves the
module implementation lazily).

Import discipline: ``lance_store`` pulls in the ~2.6 s ``lancedb`` subtree, so
:class:`LanceVectorStore` imports it lazily inside each method — a bare
``import claude_sql.infrastructure.adapters`` stays lean, preserving the
``test_pr3_perf`` lazy-import guards. ``parquet_cache`` and ``sqlite_state`` are
cheap (polars/sqlite only) and imported at module top.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from claude_sql.infrastructure import parquet_cache
from claude_sql.infrastructure.sqlite_state import checkpointer, retry_queue

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime
    from pathlib import Path

    import polars as pl

    from claude_sql.infrastructure.settings import Settings

# NOTE: these adapters deliberately do NOT import the port Protocols from
# ``claude_sql.application.ports`` — infrastructure importing application is
# forbidden by the import-linter hexagonal contract, and Protocol conformance is
# structural (checked at the use-case's injection site), so the import is
# unnecessary. Factory return types are the concrete adapter classes; each one
# structurally satisfies its port. Mirrors ``transcript_reader.DuckDbTranscriptReader``,
# which likewise names its port only in prose.


class ParquetCache:
    """:class:`~claude_sql.application.ports.CachePort` over one sharded-parquet cache.

    The cache target (a sharded directory or a legacy single file) is bound in
    the constructor; every method delegates to the module-level
    :mod:`claude_sql.infrastructure.parquet_cache` functions at call time.
    """

    def __init__(self, target: Path) -> None:
        self._target = target

    def write_part(self, df: pl.DataFrame) -> Path:
        """Append ``df`` as a new shard (or legacy rewrite); return the path."""
        return parquet_cache.write_part(self._target, df)

    def read_all(self, *, columns: list[str] | None = None) -> pl.DataFrame | None:
        """Return the union of all parts (``None`` when the cache is empty)."""
        return parquet_cache.read_all(self._target, columns=columns)

    def count_rows(self) -> int:
        """Return the total row count across every part (footer read only)."""
        return parquet_cache.count_rows(self._target)

    def iter_part_files(self) -> list[Path]:
        """Return the sorted list of parquet files backing the cache."""
        return parquet_cache.iter_part_files(self._target)

    def replace_sessions(self, *, key_column: str, session_ids: Iterable[str]) -> int:
        """Drop rows whose ``key_column`` is in ``session_ids``; return removed."""
        return parquet_cache.replace_sessions(
            self._target, key_column=key_column, session_ids=session_ids
        )


class SqliteCheckpoint:
    """:class:`~claude_sql.application.ports.CheckpointPort` over ``state.db``.

    The SQLite path is bound in the constructor; delegates to the module-level
    :mod:`claude_sql.infrastructure.sqlite_state.checkpointer` functions. The
    staleness math (``filter_unchanged`` / ``_stale_or_equal``) stays a domain
    concern on the module and is NOT part of the port.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def load_as_map(self, pipeline: str) -> dict[str, tuple[datetime | None, datetime | None]]:
        """Return ``{session_id: (last_ts, last_mtime)}`` for one pipeline."""
        return checkpointer.load_as_map(self._db_path, pipeline)

    def mark_completed(
        self,
        *,
        pipeline: str,
        rows: Iterable[tuple[str, datetime | None, datetime | None]],
    ) -> int:
        """Upsert checkpoint rows; return the number upserted."""
        return checkpointer.mark_completed(self._db_path, pipeline=pipeline, rows=rows)

    def count_rows(self) -> int:
        """Return the total number of checkpoint rows (``0`` when absent)."""
        return checkpointer.count_rows(self._db_path)


class SqliteRetryQueue:
    """:class:`~claude_sql.application.ports.RetryQueuePort` over ``state.db``.

    The SQLite path is bound in the constructor; delegates to the module-level
    :mod:`claude_sql.infrastructure.sqlite_state.retry_queue` functions. The
    exponential-backoff math (``_backoff_delta``) stays on the module.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def enqueue(self, *, pipeline: str, unit_id: str, error: str) -> int:
        """Record a failure (increments the attempt counter); return attempts."""
        return retry_queue.enqueue(self._db_path, pipeline=pipeline, unit_id=unit_id, error=error)

    def drain(self, *, pipeline: str, max_attempts: int = 5, limit: int | None = None) -> list[str]:
        """Return unit_ids eligible for retry (not done, attempts<max, due)."""
        return retry_queue.drain(
            self._db_path, pipeline=pipeline, max_attempts=max_attempts, limit=limit
        )

    def mark_done(self, *, pipeline: str, unit_ids: Iterable[str]) -> int:
        """Mark the given unit_ids completed; return the count marked."""
        return retry_queue.mark_done(self._db_path, pipeline=pipeline, unit_ids=unit_ids)

    def pending_count(self, *, pipeline: str) -> int:
        """Count not-yet-completed rows for one pipeline."""
        return retry_queue.pending_count(self._db_path, pipeline=pipeline)


class LanceVectorStore:
    """:class:`~claude_sql.application.ports.VectorStorePort` over the LanceDB store.

    The ``lance_uri`` (and, for the write path, the vector ``dim``) is bound in
    the constructor. Write-side methods (:meth:`add_chunk` / :meth:`ensure_index`
    / :meth:`optimize`) reuse a single cached ``(db, table)`` handle across a
    backfill run — mirroring the pre-seam inline loop, which opened the table
    once and appended chunk-by-chunk. Read-side methods delegate to the module
    functions that open their own short-lived connection.

    ``lance_store`` (and its ~2.6 s ``lancedb`` subtree) is imported lazily
    inside each method so importing this module stays lean.
    """

    def __init__(self, lance_uri: Path, *, dim: int | None = None) -> None:
        self._uri = lance_uri
        self._dim = dim
        self._tbl: Any = None

    def _table(self) -> Any:
        """Open (once) and cache the embeddings table for the write path."""
        from claude_sql.infrastructure import lance_store

        if self._tbl is None:
            if self._dim is None:
                msg = "LanceVectorStore needs an explicit dim to open/create the table"
                raise ValueError(msg)
            db = lance_store.connect_db(self._uri)
            self._tbl = lance_store.open_or_create_table(db, dim=self._dim)
        return self._tbl

    def add_chunk(self, df: pl.DataFrame) -> None:
        """Append one chunk of embeddings (fixed-size ``Array(Float32, dim)``)."""
        from claude_sql.infrastructure import lance_store

        lance_store.add_chunk(self._table(), df)

    def ensure_index(self, *, metric: str = "cosine") -> None:
        """Create the IVF_HNSW_SQ vector index if absent (no-op otherwise)."""
        from typing import cast

        from claude_sql.infrastructure import lance_store
        from claude_sql.infrastructure.lance_store import DistanceMetric

        lance_store.ensure_index(self._table(), metric=cast("DistanceMetric", metric))

    def optimize(self) -> None:
        """Compact accumulated fragments + clean up old versions (no-op on error)."""
        from claude_sql.infrastructure import lance_store

        lance_store.optimize_if_needed(self._table())

    def count_rows(self) -> int:
        """Return the embeddings row count, or ``0`` when the store is empty."""
        from claude_sql.infrastructure import lance_store

        return lance_store.count_rows(self._uri)

    def table_identity(self) -> tuple[str, int] | None:
        """Return the stamped ``(model, dim)``, or ``None`` for an empty store."""
        from claude_sql.infrastructure import lance_store

        return lance_store.table_identity(self._uri)

    def get_embedded_uuids(self) -> set[str]:
        """Return the set of uuids that already have embeddings (anti-join key)."""
        from claude_sql.infrastructure import lance_store

        return lance_store.get_embedded_uuids(self._uri)


# ---------------------------------------------------------------------------
# Default factories — the composition-root wiring the use-cases fall back to
# when no port is injected. Each builds the module-backed adapter from Settings.
# ---------------------------------------------------------------------------


def build_cache(target: Path) -> ParquetCache:
    """Construct the default ``CachePort`` (``ParquetCache``) over ``target``."""
    return ParquetCache(target)


def build_checkpoint(settings: Settings) -> SqliteCheckpoint:
    """Construct the default ``CheckpointPort`` over ``settings.checkpoint_db_path``."""
    return SqliteCheckpoint(settings.checkpoint_db_path)


def build_retry_queue(settings: Settings) -> SqliteRetryQueue:
    """Construct the default ``RetryQueuePort`` over ``settings.checkpoint_db_path``."""
    return SqliteRetryQueue(settings.checkpoint_db_path)


def build_vector_store(settings: Settings, *, dim: int | None = None) -> LanceVectorStore:
    """Construct the default ``VectorStorePort`` over ``settings.lance_uri``."""
    return LanceVectorStore(settings.lance_uri, dim=dim)


__all__ = [
    "LanceVectorStore",
    "ParquetCache",
    "SqliteCheckpoint",
    "SqliteRetryQueue",
    "build_cache",
    "build_checkpoint",
    "build_retry_queue",
    "build_vector_store",
]

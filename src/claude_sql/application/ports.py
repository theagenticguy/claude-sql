"""The port Protocol surface for the v2 hexagonal reshape.

Every seam between a use-case and the outside world is a ``@runtime_checkable``
Protocol here. Concrete adapters (DuckDB, LanceDB, SQLite, Bedrock, parquet)
live in ``infrastructure`` and are injected by the composition root
(``interfaces/cli``). Use-cases depend on these Protocols, never on a concrete
adapter — that is the whole point of the port layer.

Signatures are modeled on the EXISTING concrete call shapes (verified against
source in T-1-1), so wrapping each adapter behind its port in later waves is a
lift, not a redesign. Where a concrete module function threads a resource handle
(``con``, ``db_path``, ``target``, ``lance_uri``) as its first argument, the
port binds that handle in the adapter's constructor and drops it from the method
signature — the resource is adapter state, invisible to the use-case.

Import discipline (pinned by ``tests/test_ports.py``): this module imports only
stdlib, typing, ``returns``, and the pure ``domain`` modules (``domain.errors``,
``domain.ports``, ``domain.retrieval``). Those are themselves lean (typing +
``TYPE_CHECKING`` only). NO duckdb/polars/lancedb/boto3 at module top; heavy
types (``duckdb`` connection, ``pl.DataFrame``, ``datetime``) are referenced
under ``TYPE_CHECKING`` only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from returns.result import Result

# Re-export the two provider ports. Their Protocols are defined in
# ``domain.ports`` (T-8-2 dissolved the transitional ``core`` package): a
# pure-Protocol home in the innermost hexagon so the concrete adapters in
# ``infrastructure`` can be typed against them without importing UP into
# ``application`` — the same pattern ``SearchHit`` uses. This is the single
# import site so use-cases name them from the ports module like every other port.
from claude_sql.domain.errors import DomainError
from claude_sql.domain.ports import EmbeddingProvider, LlmAnalyticsProvider

# ``SearchHit`` is a pure value-object; it lives in ``domain.retrieval`` so the
# concrete DuckDB+Lance adapter can build it without importing up into the
# application layer. Re-exported here so callers name the whole port surface
# (ports + the row type they return) from one module.
from claude_sql.domain.retrieval import SearchHit

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime
    from pathlib import Path

    import duckdb
    import polars as pl


#: The uniform result type crossing a port boundary. A ``Success[T]`` carries
#: the value; a ``Failure[DomainError]`` carries a domain error. See CLAUDE.md
#: "returns discipline": use ``Success``/``Failure``, ``is_successful``,
#: ``.unwrap``, ``.map``, ``.alt``, and ``match`` narrowing ONLY — never
#: ``.bind()``, ``@safe``, ``flow``, ``pipe``, or HKT features (they require the
#: mypy plugin and hard-error under ty).
type PortResult[T] = Result[T, DomainError]


@runtime_checkable
class Clock(Protocol):
    """Port: the current time.

    A seam over ``datetime.now(UTC)`` so use-cases (checkpoint watermarks, retry
    backoff, ``classified_at`` stamps) are deterministic under test. The single
    method returns an aware UTC ``datetime``.
    """

    def now(self) -> datetime:
        """Return the current time as an aware UTC ``datetime``."""
        ...


@runtime_checkable
class TranscriptReaderPort(Protocol):
    """Port: read assembled transcript text and session structure.

    Wraps the ``session_text`` seam (per-session timeline assembly) and the
    session-enumeration queries. The DuckDB connection and the ``read_json``
    glob are adapter state.
    """

    def session_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Return the ordered message rows for one session (chronological)."""
        ...

    def read_turn_text(self, ref: str) -> str:
        """Return the assembled text for one turn/session reference.

        Honors the consumer collapse contract: tool_use / tool_result blocks are
        interleaved chronologically with role-marked text turns and clipped per
        the ``session_text_*`` caps (see
        :meth:`claude_sql.domain.transcript.SessionTextCorpus.assemble`).
        """
        ...

    def session_bounds(
        self, *, since_days: int | None = None, limit: int | None = None
    ) -> dict[str, tuple[datetime | None, datetime | None]]:
        """Return ``{session_id: (last_ts, transcript_mtime)}`` for the window.

        Mirrors :func:`claude_sql.infrastructure.session_text_loader.session_bounds`; drives the
        mtime-based checkpoint skip in the LLM worker pipelines.
        """
        ...

    def session_ids(self, *, since_days: int | None = None, limit: int | None = None) -> list[str]:
        """Return the newest-first session ids matching the window."""
        ...


@runtime_checkable
class SessionSearchPort(Protocol):
    """Port: semantic top-k search over message embeddings.

    Wraps the ``search`` command's embed-query + cosine-kNN path. The embedder
    and the DuckDB/Lance-backed ``message_embeddings`` view are adapter state.
    """

    def search(self, query: str, *, k: int = 10, session_id: str | None = None) -> list[SearchHit]:
        """Return the top-``k`` nearest neighbors to ``query`` as typed hits."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Return the query embedding vector (always float; adapter's width)."""
        ...


@runtime_checkable
class VectorStorePort(Protocol):
    """Port: the LanceDB embeddings store read/write + kNN seam.

    Mirrors the ``lance_store`` module surface. The ``lance_uri`` / open table
    handle is adapter state. Keeps the empty-namespace gate (an absent table
    reports ``0`` rows / ``None`` identity rather than raising).
    """

    def add_chunk(self, df: pl.DataFrame) -> None:
        """Append one chunk of embeddings (fixed-size ``Array(Float32, dim)``)."""
        ...

    def ensure_index(self, *, metric: str = "cosine") -> None:
        """Create the IVF_HNSW_SQ vector index if absent (no-op otherwise)."""
        ...

    def optimize(self) -> None:
        """Compact accumulated fragments + clean up old versions (best-effort)."""
        ...

    def count_rows(self) -> int:
        """Return the embeddings row count, or ``0`` when the store is empty."""
        ...

    def table_identity(self) -> tuple[str, int] | None:
        """Return the stamped ``(model, dim)``, or ``None`` for an empty store."""
        ...

    def get_embedded_uuids(self) -> set[str]:
        """Return the set of uuids that already have embeddings (anti-join key)."""
        ...


@runtime_checkable
class CheckpointPort(Protocol):
    """Port: per-``(session_id, pipeline)`` processing watermark.

    Mirrors the ``checkpointer`` surface. The SQLite ``state.db`` path is
    adapter state; the staleness math (``filter_unchanged`` / ``_stale_or_equal``)
    is domain and stays out of the port.
    """

    def load_as_map(self, pipeline: str) -> dict[str, tuple[datetime | None, datetime | None]]:
        """Return ``{session_id: (last_ts, last_mtime)}`` for one pipeline."""
        ...

    def mark_completed(
        self,
        *,
        pipeline: str,
        rows: Iterable[tuple[str, datetime | None, datetime | None]],
    ) -> int:
        """Upsert checkpoint rows; return the number upserted."""
        ...

    def count_rows(self) -> int:
        """Return the total number of checkpoint rows (``0`` when absent)."""
        ...


@runtime_checkable
class RetryQueuePort(Protocol):
    """Port: durable retry queue for failed LLM units of work.

    Mirrors the ``retry_queue`` surface. The SQLite path and clock are adapter
    state; the exponential-backoff math (``_backoff_delta``) is domain.
    """

    def enqueue(self, *, pipeline: str, unit_id: str, error: str) -> int:
        """Record a failure (increments the attempt counter); return attempts."""
        ...

    def drain(self, *, pipeline: str, max_attempts: int = 5, limit: int | None = None) -> list[str]:
        """Return unit_ids eligible for retry (not done, attempts<max, due)."""
        ...

    def mark_done(self, *, pipeline: str, unit_ids: Iterable[str]) -> int:
        """Mark the given unit_ids completed; return the count marked."""
        ...

    def pending_count(self, *, pipeline: str) -> int:
        """Count not-yet-completed rows for one pipeline."""
        ...


@runtime_checkable
class CachePort(Protocol):
    """Port: the sharded-parquet artifact store for one worker cache.

    Mirrors the ``parquet_shards`` surface. The cache target (a sharded
    directory or a legacy single file) is adapter state. The parquet-existence
    gate — views register only caches that exist — is preserved by the adapter.
    """

    def write_part(self, df: pl.DataFrame) -> Path:
        """Append ``df`` as a new shard (or legacy rewrite); return the path."""
        ...

    def read_all(self, *, columns: list[str] | None = None) -> pl.DataFrame | None:
        """Return the union of all parts (``None`` when the cache is empty)."""
        ...

    def count_rows(self) -> int:
        """Return the total row count across every part (footer read only)."""
        ...

    def iter_part_files(self) -> list[Path]:
        """Return the sorted list of parquet files backing the cache."""
        ...

    def replace_sessions(self, *, key_column: str, session_ids: Iterable[str]) -> int:
        """Drop rows whose ``key_column`` is in ``session_ids``; return removed."""
        ...


@runtime_checkable
class ReaderPort(Protocol):
    """Port: the DuckDB query seam, WITH its rebind lifecycle made explicit.

    The DuckDB connection is stateful, long-lived, and shared across ``analyze``
    stages — NOT a stateless query executor. ``register_all`` binds 18 views +
    14 macros + VSS against parquet path lists that are captured and FROZEN at
    registration time. A stage that writes new parquet shards (``embed``,
    ``cluster``) or populates LanceDB mid-run does not see its own writes until
    the views are re-bound.

    This port therefore models the register -> write -> rebind cycle EXPLICITLY.
    A naive ``query(sql)`` that hides the connection silently reintroduces the
    RFC §9.6 analyze stale-connection bug (``community`` reads
    ``message_embeddings`` and gets 0 rows without an intervening rebind; see
    ``_rebind_vss`` at ``cli.py:438`` and ``_refresh_analytics_views`` at
    ``cli.py:418``). Use-cases that write mid-run MUST call
    :meth:`refresh_analytics_views` / :meth:`rebind_vss` between the write and
    the next read.

    Implementation lands in Wave 4; this wave defines the Protocol only.
    """

    def connection(self) -> duckdb.DuckDBPyConnection:
        """Return the underlying (shared, stateful) DuckDB connection handle."""
        ...

    def query(self, sql: str, params: list[Any] | None = None) -> pl.DataFrame:
        """Execute ``sql`` and return the result as a polars DataFrame."""
        ...

    def refresh_analytics_views(self) -> None:
        """Re-bind the analytics views so mid-run parquet writes become visible."""
        ...

    def rebind_vss(self, stage: str) -> None:
        """Re-bind ``message_embeddings`` against the (mutated) Lance namespace.

        ``stage`` names the pipeline stage that just wrote, for logging. See the
        class docstring and RFC §9.6 for why this cannot be folded into
        :meth:`query`.
        """
        ...


__all__ = [
    "CachePort",
    "CheckpointPort",
    "Clock",
    "EmbeddingProvider",
    "LlmAnalyticsProvider",
    "PortResult",
    "ReaderPort",
    "RetryQueuePort",
    "SearchHit",
    "SessionSearchPort",
    "TranscriptReaderPort",
    "VectorStorePort",
]

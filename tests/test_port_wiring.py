"""Port-wiring conformance: each use-case consumes an injected storage port.

Closes the architecture-validator conditional from T-7-1: ``CachePort``,
``CheckpointPort``, ``RetryQueuePort``, and ``VectorStorePort`` are no longer
merely adapter-satisfied — the LLM/analytics use-cases now accept them as
injectable seams and call through them. Each test injects an in-memory fake
port, drives the use-case down a short-circuit path (so no Bedrock is touched),
and asserts:

1. the injected port's methods were called (the seam is live), and
2. the real backing store (SQLite ``state.db`` / the on-disk parquet cache /
   the Lance dataset) was NOT created — proving the fake fully stands in for
   the resource.

The direct ``session_bounds`` / ``checkpointer.filter_unchanged`` calls (which
stay on the module by name so the existing monkeypatch seams keep biting) are
patched here to hold each pipeline on its short-circuit path without a Bedrock
client or a real SQLite read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import pytest

from claude_sql.application.use_cases import (
    classify as classify_worker,
    conflicts as conflicts_worker,
    friction as friction_worker,
    trajectory as trajectory_worker,
)
from claude_sql.application.use_cases.cluster import run_clustering
from claude_sql.application.use_cases.embed import discover_unembedded
from claude_sql.application.use_cases.ingest import stamp_messages

if TYPE_CHECKING:
    from pathlib import Path

    import duckdb

    from claude_sql.infrastructure.settings import Settings


# ---------------------------------------------------------------------------
# In-memory fakes — record calls; never touch disk / sqlite / lance.
# ---------------------------------------------------------------------------


class FakeCache:
    """:class:`~claude_sql.application.ports.CachePort` recording fake."""

    def __init__(self, read_all_result: pl.DataFrame | None = None) -> None:
        self._read_all_result = read_all_result
        self.read_all_calls = 0
        self.written: list[pl.DataFrame] = []
        self.replace_calls: list[tuple[str, list[str]]] = []

    def write_part(self, df: pl.DataFrame) -> Path:
        import pathlib

        self.written.append(df)
        return pathlib.Path("/fake/part-0.parquet")

    def read_all(self, *, columns: list[str] | None = None) -> pl.DataFrame | None:
        self.read_all_calls += 1
        return self._read_all_result

    def count_rows(self) -> int:
        return 0

    def iter_part_files(self) -> list[Path]:
        return []

    def replace_sessions(self, *, key_column: str, session_ids: Any) -> int:
        self.replace_calls.append((key_column, list(session_ids)))
        return 0


class FakeCheckpoint:
    """:class:`~claude_sql.application.ports.CheckpointPort` recording fake."""

    def __init__(self) -> None:
        self.mark_completed_calls = 0

    def load_as_map(self, pipeline: str) -> dict[str, Any]:
        return {}

    def mark_completed(self, *, pipeline: str, rows: Any) -> int:
        self.mark_completed_calls += 1
        return len(list(rows))

    def count_rows(self) -> int:
        return 0


class FakeRetryQueue:
    """:class:`~claude_sql.application.ports.RetryQueuePort` recording fake."""

    def __init__(self) -> None:
        self.drain_calls = 0
        self.enqueued: list[str] = []
        self.done: list[str] = []

    def enqueue(self, *, pipeline: str, unit_id: str, error: str) -> int:
        self.enqueued.append(unit_id)
        return 1

    def drain(self, *, pipeline: str, max_attempts: int = 5, limit: int | None = None) -> list[str]:
        self.drain_calls += 1
        return []

    def mark_done(self, *, pipeline: str, unit_ids: Any) -> int:
        self.done.extend(unit_ids)
        return len(list(unit_ids))

    def pending_count(self, *, pipeline: str) -> int:
        return 0


class FakeVectorStore:
    """:class:`~claude_sql.application.ports.VectorStorePort` recording fake."""

    def __init__(self, *, rows: int = 0, embedded: set[str] | None = None) -> None:
        self._rows = rows
        self._embedded = embedded or set()
        self.count_rows_calls = 0
        self.get_embedded_calls = 0
        self.added: list[pl.DataFrame] = []

    def add_chunk(self, df: pl.DataFrame) -> None:
        self.added.append(df)

    def ensure_index(self, *, metric: str = "cosine") -> None:
        pass

    def optimize(self) -> None:
        pass

    def count_rows(self) -> int:
        self.count_rows_calls += 1
        return self._rows

    def table_identity(self) -> tuple[str, int] | None:
        return None

    def get_embedded_uuids(self) -> set[str]:
        self.get_embedded_calls += 1
        return set(self._embedded)


def _no_real_state(settings: Settings) -> None:
    """Assert the real SQLite ``state.db`` was never created by the pipeline."""
    assert not settings.checkpoint_db_path.exists(), (
        "pipeline touched the real SQLite state.db instead of the injected ports"
    )


# ---------------------------------------------------------------------------
# LLM workers — cache + checkpoint + retry ports
# ---------------------------------------------------------------------------


def test_classify_uses_injected_ports(
    registered_con: duckdb.DuckDBPyConnection,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``classify_sessions`` reads/writes through the injected storage ports."""
    # Hold the pipeline on its short-circuit path: empty bounds -> no pending
    # sessions -> returns 0 before any Bedrock client is built.
    monkeypatch.setattr(classify_worker, "session_bounds", lambda *a, **kw: {})
    monkeypatch.setattr(classify_worker.checkpointer, "filter_unchanged", lambda *a, **kw: ([], 0))

    cache = FakeCache(read_all_result=None)
    checkpoint = FakeCheckpoint()
    retry = FakeRetryQueue()
    out = classify_worker.classify_sessions(
        registered_con,
        tmp_settings,
        cache=cache,
        checkpoint=checkpoint,
        retry=retry,
    )
    assert out == 0
    assert cache.read_all_calls >= 1  # anti-join against the parquet cache
    assert retry.drain_calls == 1  # drained the retry queue via the port
    _no_real_state(tmp_settings)


def test_conflicts_uses_injected_ports(
    registered_con: duckdb.DuckDBPyConnection,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``detect_conflicts`` reads/writes through the injected storage ports."""
    monkeypatch.setattr(conflicts_worker, "session_bounds", lambda *a, **kw: {})
    monkeypatch.setattr(conflicts_worker.checkpointer, "filter_unchanged", lambda *a, **kw: ([], 0))

    cache = FakeCache(read_all_result=None)
    retry = FakeRetryQueue()
    out = conflicts_worker.detect_conflicts(
        registered_con,
        tmp_settings,
        cache=cache,
        checkpoint=FakeCheckpoint(),
        retry=retry,
    )
    assert out == 0
    assert cache.read_all_calls >= 1
    assert retry.drain_calls == 1
    _no_real_state(tmp_settings)


def test_friction_uses_injected_ports(
    registered_con: duckdb.DuckDBPyConnection,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``detect_user_friction`` reads/writes through the injected storage ports.

    A non-empty ``session_bounds`` scoped to a session id absent from the
    corpus makes the candidate SQL return no rows, so the pipeline returns 0
    before the LLM path — hermetic, no Bedrock.
    """
    monkeypatch.setattr(
        friction_worker, "session_bounds", lambda *a, **kw: {"nonexistent-session": (None, None)}
    )
    monkeypatch.setattr(
        friction_worker.checkpointer,
        "filter_unchanged",
        lambda *a, **kw: (["nonexistent-session"], 0),
    )

    cache = FakeCache(read_all_result=None)
    retry = FakeRetryQueue()
    out = friction_worker.detect_user_friction(
        registered_con,
        tmp_settings,
        cache=cache,
        checkpoint=FakeCheckpoint(),
        retry=retry,
    )
    assert out == 0
    assert cache.read_all_calls >= 1
    assert retry.drain_calls == 1
    _no_real_state(tmp_settings)


def test_trajectory_uses_injected_ports(
    registered_con: duckdb.DuckDBPyConnection,
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``trajectory_messages`` drains the retry queue through the injected port."""
    # Empty bounds -> ``active_sessions`` empty and ``not bounds`` -> returns 0
    # before window loading / provider construction.
    monkeypatch.setattr(trajectory_worker, "session_bounds", lambda *a, **kw: {})
    monkeypatch.setattr(
        trajectory_worker.checkpointer, "filter_unchanged", lambda *a, **kw: ([], 0)
    )

    retry = FakeRetryQueue()
    out = trajectory_worker.trajectory_messages(
        registered_con,
        tmp_settings,
        cache=FakeCache(),
        checkpoint=FakeCheckpoint(),
        retry=retry,
    )
    assert out == 0
    assert retry.drain_calls == 1
    _no_real_state(tmp_settings)


# ---------------------------------------------------------------------------
# ingest — cache port
# ---------------------------------------------------------------------------


def test_ingest_stamp_uses_injected_cache(
    registered_con: duckdb.DuckDBPyConnection,
    tmp_settings: Settings,
) -> None:
    """``stamp_messages`` writes stamps through the injected ``CachePort``."""
    cache = FakeCache()
    written = stamp_messages(registered_con, tmp_settings, cache=cache)
    assert written > 0  # the fixture corpus has stampable messages
    assert cache.written, "stamp_messages did not write through the injected cache port"


# ---------------------------------------------------------------------------
# embed + cluster — vector-store port
# ---------------------------------------------------------------------------


def test_discover_unembedded_uses_injected_store(
    registered_con: duckdb.DuckDBPyConnection,
    tmp_path: Path,
) -> None:
    """``discover_unembedded`` reads the embedded-uuid set through the port."""
    store = FakeVectorStore(embedded=set())
    rows = discover_unembedded(
        registered_con,
        lance_uri=tmp_path / "no_lance_here",
        store=store,
    )
    assert store.get_embedded_calls == 1
    assert isinstance(rows, list)
    # No Lance dataset was created — the fake stood in for the store.
    assert not (tmp_path / "no_lance_here").exists()


def test_run_clustering_uses_injected_store(tmp_settings: Settings) -> None:
    """``run_clustering`` consults the vector-store row-count guard via the port."""
    store = FakeVectorStore(rows=0)
    with pytest.raises(FileNotFoundError):
        run_clustering(tmp_settings, force=True, store=store)
    assert store.count_rows_calls == 1
    # The real Lance dataset was never opened.
    assert not tmp_settings.lance_uri.exists()

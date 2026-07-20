"""The ``ClaudeSql`` composition facade — the one importable entry point.

This is the composition root downstream consumers import:
``from claude_sql.composition import ClaudeSql`` (or ``from claude_sql import
ClaudeSql``). It wires the retrieval ports (transcript reader + semantic search)
and offers a ``query`` convenience over the full-registration DuckDB connection.

Import discipline: this module's top level pulls in NOTHING heavy — no duckdb,
no adapters, no polars. Every heavy import lives inside a method or factory so a
bare ``import claude_sql.composition`` (or ``import claude_sql``) stays a
sub-millisecond, dependency-free operation. The port-typed return annotations
are referenced under ``TYPE_CHECKING`` only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl

    from claude_sql.application.ports import (
        CachePort,
        CheckpointPort,
        EmbeddingProvider,
        RetryQueuePort,
        SessionSearchPort,
        TranscriptReaderPort,
        VectorStorePort,
    )
    from claude_sql.infrastructure.settings import Settings


class ClaudeSql:
    """Facade over the claude-sql retrieval surface.

    Lazily constructs and caches a :class:`TranscriptReaderPort` and a
    :class:`SessionSearchPort`, and exposes a :meth:`query` convenience for
    ad-hoc SQL over the fully-registered corpus. Construction is cheap: the
    ports (and their DuckDB connections) are not built until the corresponding
    accessor is first called.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: EmbeddingProvider | None = None,
        config_root: str | Path | None = None,
    ):
        """Capture the wiring inputs without building anything heavy.

        Parameters
        ----------
        settings
            Active :class:`Settings`; a default instance is created lazily when
            first needed if left ``None``.
        embedder
            Optional :class:`EmbeddingProvider` injected into the search port.
            Defaults to ``build_embedder(settings)`` at first use.
        config_root
            Optional transcript-root override threaded into the reader port
            (see :class:`~claude_sql.infrastructure.transcript_reader.DuckDbTranscriptReader`).
        """
        self._settings = settings
        self._embedder = embedder
        self._config_root = config_root
        self._reader: TranscriptReaderPort | None = None
        self._search: SessionSearchPort | None = None

    def _resolved_settings(self) -> Settings:
        if self._settings is None:
            from claude_sql.infrastructure.settings import Settings

            self._settings = Settings()
        return self._settings

    def reader(self) -> TranscriptReaderPort:
        """Return the lazily-built, cached transcript reader port."""
        if self._reader is None:
            from claude_sql.infrastructure.transcript_reader import DuckDbTranscriptReader

            self._reader = DuckDbTranscriptReader(
                self._resolved_settings(), config_root=self._config_root
            )
        return self._reader

    def search(self) -> SessionSearchPort:
        """Return the lazily-built, cached semantic-search port."""
        if self._search is None:
            from claude_sql.infrastructure.session_search import DuckDbSessionSearch

            self._search = DuckDbSessionSearch(self._resolved_settings(), embedder=self._embedder)
        return self._search

    def query(self, sql: str) -> pl.DataFrame:
        """Run ``sql`` over a fully-registered connection, return a DataFrame.

        Opens a fresh in-memory DuckDB connection, registers the full object
        surface (raw + views + VSS + analytics + macros) via
        ``claude_sql.infrastructure.duckdb_views.register_all``, runs the
        statement, and closes it. Convenience for ad-hoc SQL; for repeated reads
        prefer the port objects.

        ``register_all`` — not cli's ``_open_connection_full`` — is the seam used
        here: it is the stable infrastructure registration entrypoint, whereas
        ``_open_connection_full`` is a private CLI helper. This keeps
        ``composition.py`` decoupled from the interfaces layer while giving
        identical full-registration semantics.
        """
        import duckdb

        from claude_sql.infrastructure.duckdb_views import register_all

        con = duckdb.connect(":memory:")
        try:
            register_all(con, settings=self._resolved_settings())
            return con.execute(sql).pl()
        finally:
            con.close()


def build_reader(
    settings: Settings | None = None, *, config_root: str | Path | None = None
) -> TranscriptReaderPort:
    """Construct a standalone :class:`TranscriptReaderPort` adapter."""
    from claude_sql.infrastructure.transcript_reader import DuckDbTranscriptReader

    return DuckDbTranscriptReader(settings, config_root=config_root)


def build_search(
    settings: Settings | None = None, *, embedder: EmbeddingProvider | None = None
) -> SessionSearchPort:
    """Construct a standalone :class:`SessionSearchPort` adapter."""
    from claude_sql.infrastructure.session_search import DuckDbSessionSearch

    return DuckDbSessionSearch(settings, embedder=embedder)


def build_cache(target: Path) -> CachePort:
    """Construct the default :class:`CachePort` over a parquet cache ``target``."""
    from claude_sql.infrastructure.adapters import build_cache as _build

    return _build(target)


def build_checkpoint(settings: Settings) -> CheckpointPort:
    """Construct the default :class:`CheckpointPort` over the SQLite ``state.db``."""
    from claude_sql.infrastructure.adapters import build_checkpoint as _build

    return _build(settings)


def build_retry_queue(settings: Settings) -> RetryQueuePort:
    """Construct the default :class:`RetryQueuePort` over the SQLite ``state.db``."""
    from claude_sql.infrastructure.adapters import build_retry_queue as _build

    return _build(settings)


def build_vector_store(settings: Settings, *, dim: int | None = None) -> VectorStorePort:
    """Construct the default :class:`VectorStorePort` over ``settings.lance_uri``."""
    from claude_sql.infrastructure.adapters import build_vector_store as _build

    return _build(settings, dim=dim)


__all__: list[str] = [
    "ClaudeSql",
    "build_cache",
    "build_checkpoint",
    "build_reader",
    "build_retry_queue",
    "build_search",
    "build_vector_store",
]


def __getattr__(name: str) -> Any:
    # Nothing deferred here today; kept as a hook so a future lazy re-export
    # doesn't force a module-shape change. Raise the standard AttributeError.
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

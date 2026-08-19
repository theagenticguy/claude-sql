"""The watch loop: keep one connection's snapshot near-live as transcripts land.

``claude-sql``'s SQL plane reads the JSONL corpus zero-copy, but the raw readers
materialize as TEMP TABLEs, so a *process* is only as fresh as its last
registration and a *cron* is only as fresh as its last tick. This use-case closes
that gap: it consumes a :class:`~claude_sql.application.ports.FileWatcherPort`,
debounces the events per source file, and calls
:func:`~claude_sql.infrastructure.duckdb_views.refresh_raw` on the files that
settled — turning "fresh as of the last full rebuild" into "fresh as of a few
seconds ago" at a cost proportional to what changed.

Why debounce rather than trigger on a session-lifecycle hook
------------------------------------------------------------
A ``SessionEnd``-style hook is both too late and not universal. Too late,
because an interactive session stays open for hours and its rows should be
queryable during it. Not universal, because a hook only fires where it is
installed: subagent sidecars, resumed sessions, and any writer that does not
load the hook config all land as plain file writes. Watching the filesystem sees
every writer, and the quiet-period debounce recovers the same "the turn finished"
signal a hook would have announced — observed instead of trusted. A hook, where
one exists, is a latency optimization on top of this loop, not a replacement for
it.

Cost posture
------------
The refresh path is free (local file reads). Embedding is not: it calls the
configured embedding provider, which for the default Cohere-on-Bedrock adapter
is real spend. So ``embed_limit`` defaults to ``0`` — **the loop does not embed
unless asked** — and when asked it is capped per flush, because "keep search
fresh" must not become an uncapped meter on a corpus backlog.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from claude_sql.domain.watch import (
    NANOS_PER_SECOND,
    drop_paths,
    due_paths,
    note_events,
)
from claude_sql.infrastructure.duckdb_views import refresh_raw

if TYPE_CHECKING:
    from collections.abc import Callable

    import duckdb

    from claude_sql.application.ports import FileWatcherPort
    from claude_sql.domain.watch import PendingFile
    from claude_sql.infrastructure.settings import Settings


@dataclass(frozen=True, slots=True)
class WatchConfig:
    """Tuning for one watch loop.

    Attributes
    ----------
    quiet_period_seconds
        How long a source file must go untouched before its rows are refreshed.
        This is the "turn finished writing" bound.
    max_wait_seconds
        Upper bound on how long a continuously-written file may go unrefreshed.
        The starvation guard; ``0`` disables it and restores the pure idle rule.
    embed_limit
        Messages to embed per flush. ``0`` (the default) disables embedding, so
        the loop spends nothing. Any positive value opts into provider calls.
    max_flushes
        Stop after this many flushes. ``0`` means run until the change source is
        exhausted, which for a live source is until the caller is interrupted.
    """

    quiet_period_seconds: float = 5.0
    max_wait_seconds: float = 60.0
    embed_limit: int = 0
    max_flushes: int = 0

    @property
    def quiet_period_ns(self) -> int:
        """:attr:`quiet_period_seconds` in nanoseconds (the debounce clock's unit)."""
        return int(self.quiet_period_seconds * NANOS_PER_SECOND)

    @property
    def max_wait_ns(self) -> int:
        """:attr:`max_wait_seconds` in nanoseconds (the debounce clock's unit)."""
        return int(self.max_wait_seconds * NANOS_PER_SECOND)


@dataclass(frozen=True, slots=True)
class FlushReport:
    """What one flush did. Emitted per flush so the loop is observable live.

    ``flush_index`` is a **1-based ordinal** for display and log correlation —
    never arithmetic input. Every other integer here is a quantity.
    """

    flush_index: int
    files_touched_count: int
    files_removed_count: int
    rows_inserted_count: int
    rows_deleted_count: int
    embedded_count: int
    rebuilt: bool
    rebuild_reason: str | None
    covers_through: str | None
    elapsed_seconds: float

    def to_payload(self) -> dict[str, object]:
        """A JSON-safe dict for the CLI's NDJSON stream."""
        return {
            "flush_index": self.flush_index,
            "files_touched_count": self.files_touched_count,
            "files_removed_count": self.files_removed_count,
            "rows_inserted_count": self.rows_inserted_count,
            "rows_deleted_count": self.rows_deleted_count,
            "embedded_count": self.embedded_count,
            "rebuilt": self.rebuilt,
            "rebuild_reason": self.rebuild_reason,
            "covers_through": self.covers_through,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


@dataclass(slots=True)
class WatchTotals:
    """Cumulative loop counters, returned when the loop stops."""

    flush_count: int = 0
    files_touched_count: int = 0
    rows_inserted_count: int = 0
    rows_deleted_count: int = 0
    embedded_count: int = 0
    rebuild_count: int = 0
    reports: list[FlushReport] = field(default_factory=list)

    def to_payload(self) -> dict[str, object]:
        """A JSON-safe summary dict for the CLI's final line."""
        return {
            "flush_count": self.flush_count,
            "files_touched_count": self.files_touched_count,
            "rows_inserted_count": self.rows_inserted_count,
            "rows_deleted_count": self.rows_deleted_count,
            "embedded_count": self.embedded_count,
            "rebuild_count": self.rebuild_count,
        }


def _embed_new_messages(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    limit: int,
) -> int:
    """Embed up to ``limit`` unembedded messages; return how many landed.

    Deferred import: the embed path pulls lancedb and the provider adapter, and
    a watch loop running with ``embed_limit=0`` must not pay for either. Errors
    are caught and logged rather than raised — a throttled provider must not
    take down a loop whose primary job (the free SQL refresh) is succeeding.
    """
    import asyncio

    from claude_sql.application.use_cases.embed import run_backfill

    try:
        result = asyncio.run(run_backfill(con=con, settings=settings, limit=limit, dry_run=False))
    except Exception:  # noqa: BLE001 — a provider fault must not stop the free refresh loop
        logger.exception("watch: embed step failed; the SQL refresh stands")
        return 0
    return result if isinstance(result, int) else 0


def run_watch(
    *,
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    watcher: FileWatcherPort,
    config: WatchConfig,
    on_flush: Callable[[FlushReport], None] | None = None,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
) -> WatchTotals:
    """Consume ``watcher``, debounce, and refresh ``con``'s snapshot per flush.

    Parameters
    ----------
    con
        A connection with the raw readers already registered. The loop advances
        THIS connection — a watch process is useful because it holds one.
    settings
        Supplies the globs the refresh re-reads and the embedding provider.
    watcher
        Change source; see :class:`~claude_sql.application.ports.FileWatcherPort`.
    config
        Debounce bounds, embed cap, and the flush ceiling.
    on_flush
        Called with each :class:`FlushReport` as it happens, so a caller can
        stream progress instead of waiting for the loop to end.
    monotonic_ns
        Injectable clock. Tests drive the debounce deterministically instead of
        sleeping; production passes ``time.monotonic_ns``.

    Returns
    -------
    WatchTotals
        Cumulative counters plus every :class:`FlushReport` in order.

    Notes
    -----
    Only the RAW tier is advanced. The analytics views bind frozen parquet path
    lists and are untouched by a transcript append, so refreshing them here
    would be work with no result; the structural and LLM stages remain the
    ``analyze`` chain's job on their own cadence.
    """
    pending: dict[str, PendingFile] = {}
    totals = WatchTotals()

    for batch in watcher.changed_batches():
        now_ns = monotonic_ns()
        if batch:
            pending = note_events(pending, batch, at_ns=now_ns)
        ready = due_paths(
            pending,
            now_ns=now_ns,
            quiet_period_ns=config.quiet_period_ns,
            max_wait_ns=config.max_wait_ns,
        )
        if not ready:
            continue

        started = time.monotonic()
        # ``refresh_raw`` rescans and diffs the watermark itself, so it is the
        # authority on what actually moved. ``ready`` decides WHEN to refresh,
        # never WHAT: a debounce built from filesystem events can miss a write
        # (dropped inotify event, a path the filter excluded), and re-deriving
        # the file set from the watermark makes that miss self-healing.
        stats = refresh_raw(
            con,
            glob=settings.default_glob,
            subagent_glob=settings.subagent_glob,
            subagent_meta_glob=settings.subagent_meta_glob,
        )
        pending = drop_paths(pending, ready)

        embedded = 0
        if config.embed_limit > 0 and (stats.rows_inserted_count > 0 or stats.rebuilt):
            embedded = _embed_new_messages(con, settings, limit=config.embed_limit)

        totals.flush_count += 1
        totals.files_touched_count += stats.files_touched_count
        totals.rows_inserted_count += stats.rows_inserted_count
        totals.rows_deleted_count += stats.rows_deleted_count
        totals.embedded_count += embedded
        totals.rebuild_count += 1 if stats.rebuilt else 0

        report = FlushReport(
            flush_index=totals.flush_count,
            files_touched_count=stats.files_touched_count,
            files_removed_count=stats.files_removed_count,
            rows_inserted_count=stats.rows_inserted_count,
            rows_deleted_count=stats.rows_deleted_count,
            embedded_count=embedded,
            rebuilt=stats.rebuilt,
            rebuild_reason=stats.rebuild_reason,
            covers_through=(
                None if stats.covers_through is None else stats.covers_through.isoformat()
            ),
            elapsed_seconds=time.monotonic() - started,
        )
        totals.reports.append(report)
        logger.info(
            "watch flush {}: {} files, +{} rows, -{} rows, {} embedded in {:.2f}s",
            report.flush_index,
            report.files_touched_count,
            report.rows_inserted_count,
            report.rows_deleted_count,
            report.embedded_count,
            report.elapsed_seconds,
        )
        if on_flush is not None:
            on_flush(report)

        if config.max_flushes and totals.flush_count >= config.max_flushes:
            logger.debug("watch: reached max_flushes={}", config.max_flushes)
            break

    return totals


__all__ = ["FlushReport", "WatchConfig", "WatchTotals", "run_watch"]

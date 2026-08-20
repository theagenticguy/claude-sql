"""Change-source adapters for :class:`~claude_sql.application.ports.FileWatcherPort`.

Two implementations, one contract:

* :class:`WatchfilesWatcher` — OS-native events (inotify on Linux) via the
  ``watchfiles`` package, under the optional ``watch`` extra. Near-zero idle
  cost; the right default for a resident daemon.
* :class:`PollingWatcher` — diffs an mtime scan on an interval. No extra
  dependency, works anywhere, and costs one sweep per tick (measured 88-101 ms
  for 6,744 files). The fallback, and the honest choice on a filesystem where
  inotify does not fire (some network mounts).

:func:`build_file_watcher` picks the event source when importable and falls back
to polling with a logged reason, so an install without the extra degrades in
latency rather than in function.

Both yield an empty batch on their heartbeat interval. That is not a
formality: the debounce is a function of elapsed time, so a file that stops
being written must still produce a wakeup for its quiet period to expire. A
source that only yielded on change would leave the last write of a session
pending until the *next* unrelated write.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from loguru import logger

from claude_sql.infrastructure.source_files import glob_roots, scan_globs

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

#: Suffix a changed path must carry to reach the loop. The corpus is JSONL; the
#: sibling ``*.meta.json`` files bind as a live VIEW that needs no refresh, and
#: DuckDB spill files under the same tree would otherwise wake the loop
#: constantly.
_TRANSCRIPT_SUFFIX: str = ".jsonl"


def _is_transcript(path: str) -> bool:
    """True when ``path`` is a transcript the raw readers would read."""
    return path.endswith(_TRANSCRIPT_SUFFIX)


class PollingWatcher:
    """Poll the transcript globs on an interval and yield what moved.

    Holds its own mtime watermark rather than asking the DuckDB connection for
    one: this adapter's job is to notice change, and coupling it to the
    connection's watermark would make the first poll after a refresh report
    nothing (the refresh having already advanced it) and miss a write that
    landed in between.
    """

    def __init__(
        self,
        patterns: Sequence[str],
        *,
        interval_seconds: float = 2.0,
        max_ticks: int = 0,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        """Configure the poller.

        Parameters
        ----------
        patterns
            Transcript globs to scan.
        interval_seconds
            Seconds between scans.
        max_ticks
            Stop after this many polls; ``0`` polls forever.
        sleep
            Injectable sleep (tests pass a no-op so the loop does not wait).
        """
        self._patterns = tuple(patterns)
        self._interval_seconds = interval_seconds
        self._max_ticks = max_ticks
        self._sleep = sleep if sleep is not None else time.sleep
        self._previous: dict[str, int] | None = None

    def changed_batches(self) -> Iterator[frozenset[str]]:
        """Yield the set of paths whose mtime moved since the previous poll."""
        from claude_sql.domain.watch import diff_source_mtimes

        ticks = 0
        while True:
            current = scan_globs(self._patterns)
            if current is None:
                logger.warning(
                    "polling watcher: {} is remote and cannot be watched; stopping",
                    self._patterns,
                )
                return
            if self._previous is None:
                # The first scan establishes the baseline. Yielding it as a
                # change set would flush the entire corpus on startup.
                self._previous = current
                yield frozenset[str]()
            else:
                delta = diff_source_mtimes(self._previous, current)
                self._previous = current
                yield frozenset(delta.touched)
            ticks += 1
            if self._max_ticks and ticks >= self._max_ticks:
                return
            self._sleep(self._interval_seconds)


class WatchfilesWatcher:
    """OS-event change source over the transcript roots (``watchfiles``).

    Watches the deepest wildcard-free ancestor of each glob, recursively, so one
    subscription covers a project's transcripts and the subagent sidecars nested
    beneath them.
    """

    def __init__(
        self,
        patterns: Sequence[str],
        *,
        heartbeat_seconds: float = 1.0,
        debounce_ms: int = 200,
    ) -> None:
        """Configure the event watcher.

        Parameters
        ----------
        patterns
            Transcript globs; their wildcard-free roots are what get watched.
        heartbeat_seconds
            How often to yield an empty batch when nothing changed, so the
            caller's debounce clock advances.
        debounce_ms
            ``watchfiles``' own coalescing window. Deliberately short: the real
            debounce is the caller's quiet period, and collapsing events here
            beyond a few hundred milliseconds would just make the reported
            change set lag.
        """
        self._roots: tuple[Path, ...] = glob_roots(patterns)
        self._heartbeat_seconds = heartbeat_seconds
        self._debounce_ms = debounce_ms

    @property
    def roots(self) -> tuple[Path, ...]:
        """The directories this watcher subscribes to."""
        return self._roots

    def changed_batches(self) -> Iterator[frozenset[str]]:
        """Yield transcript paths from OS events, plus empty heartbeat batches."""
        import watchfiles

        if not self._roots:
            logger.warning("watchfiles watcher: no watchable roots; stopping")
            return
        logger.info("watching {} via OS events", ", ".join(str(root) for root in self._roots))
        for changes in watchfiles.watch(
            *self._roots,
            watch_filter=lambda _change, path: _is_transcript(path),
            debounce=self._debounce_ms,
            rust_timeout=int(self._heartbeat_seconds * 1000),
            yield_on_timeout=True,
        ):
            yield frozenset(path for _change, path in changes)


def watchfiles_available() -> bool:
    """True when the ``watch`` extra is installed."""
    from importlib.util import find_spec

    return find_spec("watchfiles") is not None


def build_file_watcher(
    patterns: Sequence[str],
    *,
    poll_seconds: float = 2.0,
    force_polling: bool = False,
    max_ticks: int = 0,
) -> PollingWatcher | WatchfilesWatcher:
    """Pick a change source: OS events when available, else polling.

    ``force_polling`` exists for the network-mount case, where inotify is
    accepted by the kernel and then never fires — a watcher that reports nothing
    forever is worse than a poller that costs 90 ms a tick.
    """
    if force_polling:
        logger.info("watch: polling every {}s (forced)", poll_seconds)
        return PollingWatcher(patterns, interval_seconds=poll_seconds, max_ticks=max_ticks)
    if not watchfiles_available():
        logger.info(
            "watch: polling every {}s (install the 'watch' extra for OS events)",
            poll_seconds,
        )
        return PollingWatcher(patterns, interval_seconds=poll_seconds, max_ticks=max_ticks)
    return WatchfilesWatcher(patterns, heartbeat_seconds=min(poll_seconds, 1.0))


__all__ = [
    "PollingWatcher",
    "WatchfilesWatcher",
    "build_file_watcher",
    "watchfiles_available",
]

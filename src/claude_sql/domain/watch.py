"""Pure watermark + debounce math for incremental raw refresh (no I/O).

Two independent pieces of logic, both pure so they can be tested without a
filesystem and proven without a runtime:

**The watermark diff.** ``v_raw_events`` / ``v_raw_subagents`` are DuckDB TEMP
TABLEs materialized from a transcript glob, so a connection that holds them is
answering from a snapshot. Advancing that snapshot without re-reading the whole
corpus needs to know exactly which source files moved:
:func:`diff_source_mtimes` compares two ``{path: mtime_ns}`` maps and returns
the added / modified / removed partition. ``added`` and ``modified`` are rows to
re-read; ``removed`` are rows to drop.

**The debounce.** A transcript is appended to many times per turn, and every
append is a filesystem event. Refreshing per event would re-read the same file
dozens of times per assistant turn, so events are coalesced per source file and
a file flushes only once it has been *quiet* for ``quiet_period_ns`` — which is
the "the turn finished writing" signal, observed rather than announced.

A pure quiet-period rule starves under sustained activity: a file written to
every second with a 20-second quiet period never becomes due. :func:`due_paths`
therefore takes a second, disjunctive bound — ``max_wait_ns`` since the *first*
unflushed event — so a continuously-appending session still flushes on a fixed
cadence. That starvation guard is the invariant machine-checked in
``proofs/ClaudeSql/Debounce.lean``.

Clock discipline: every ``*_ns`` field here is **monotonic-clock nanoseconds**
(``time.monotonic_ns()``), not a wall-clock epoch — the debounce measures
elapsed intervals and must not move when the system clock is stepped. The
``mtime_ns`` values in the watermark maps are the *other* kind: filesystem
modification times in epoch nanoseconds (``os.stat().st_mtime_ns``). The two
spaces never meet; nothing in this module compares one to the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from datetime import datetime

#: Nanoseconds per second. Both bounds below are expressed in nanoseconds
#: because ``time.monotonic_ns`` is the clock source; callers configure in
#: seconds and convert once at the boundary.
NANOS_PER_SECOND: int = 1_000_000_000


@dataclass(frozen=True, slots=True)
class SourceDelta:
    """Which transcript source files moved between two watermark snapshots.

    Attributes
    ----------
    added
        Paths present in the current scan and absent from the previous one.
    modified
        Paths present in both whose ``mtime_ns`` advanced.
    removed
        Paths present in the previous scan and absent from the current one.

    All three are sorted tuples so a delta is deterministic and comparable —
    the refresh SQL embeds these paths, and a set-ordered list would make two
    identical refreshes emit different statements.
    """

    added: tuple[str, ...]
    modified: tuple[str, ...]
    removed: tuple[str, ...]

    @property
    def touched(self) -> tuple[str, ...]:
        """Paths whose rows must be re-read (``added`` then ``modified``)."""
        return (*self.added, *self.modified)

    @property
    def is_empty(self) -> bool:
        """True when nothing moved, so a refresh would be a no-op."""
        return not (self.added or self.modified or self.removed)

    @property
    def changed_count(self) -> int:
        """Total number of paths in the delta, across all three partitions."""
        return len(self.added) + len(self.modified) + len(self.removed)


def diff_source_mtimes(
    previous: Mapping[str, int],
    current: Mapping[str, int],
) -> SourceDelta:
    """Partition two ``{path: mtime_ns}`` maps into added / modified / removed.

    A path is ``modified`` only when its mtime **advanced**. An mtime that went
    backwards (a restored backup, a clock step on the writing host, a file
    replaced by an older copy) also counts as modified: the recorded watermark
    no longer describes the bytes on disk, so the rows must be re-read. Only an
    exactly-equal mtime is treated as unchanged.

    Parameters
    ----------
    previous
        The watermark recorded when the snapshot was last built or advanced.
    current
        A fresh scan of the same globs.

    Returns
    -------
    SourceDelta
        Sorted added / modified / removed partitions.
    """
    added = sorted(path for path in current if path not in previous)
    removed = sorted(path for path in previous if path not in current)
    modified = sorted(
        path
        for path, mtime_ns in current.items()
        if path in previous and previous[path] != mtime_ns
    )
    return SourceDelta(added=tuple(added), modified=tuple(modified), removed=tuple(removed))


def newest_mtime_ns(scan: Mapping[str, int]) -> int | None:
    """The newest ``mtime_ns`` in a scan, or ``None`` when the scan is empty.

    This is what a snapshot *covers through*: the instant of the most recently
    written transcript the snapshot has read. ``None`` for an empty corpus is
    deliberate — reporting 0 (the epoch) would read as "covers nothing since
    1970" and invite a comparison against a real timestamp.
    """
    if not scan:
        return None
    return max(scan.values())


@dataclass(frozen=True, slots=True)
class RawSnapshot:
    """What instant a connection's raw-reader snapshot is an answer about.

    The raw readers are TEMP TABLEs, so any process holding a connection is
    answering from a snapshot. This is the authoritative report of that
    snapshot's coverage — the accessor an embedding consumer needs so it can
    say "my answer covers transcripts through T" instead of inferring freshness
    from its own clock.

    Attributes
    ----------
    registered_at
        UTC wall clock at which the raw tables were last built or advanced.
        This is when the *work* happened, not what the data covers.
    covers_through
        UTC modification time of the newest transcript file included in the
        snapshot — what the data actually covers. ``None`` when unobservable
        (a remote ``s3://`` corpus) or when the corpus matched no files.
    files_scanned_count
        Number of local source files in the snapshot's watermark. ``None`` when
        the watermark is unobservable. Scope: one corpus, both transcript
        globs (primary + subagent); the ``meta.json`` glob is excluded because
        it binds as a live VIEW and is never snapshotted.
    refresh_count
        How many incremental refreshes have been applied since the full build.
        A quantity, not an ordinal: 0 means "never refreshed since register".
    """

    registered_at: datetime
    covers_through: datetime | None
    files_scanned_count: int | None
    refresh_count: int


@dataclass(frozen=True, slots=True)
class RawRefreshStats:
    """Outcome of one incremental raw refresh.

    Attributes
    ----------
    rebuilt
        True when the refresh degraded to a full re-materialization instead of
        an incremental one. Callers log this: a refresh that silently rebuilds
        every time is a performance bug wearing a success message.
    rebuild_reason
        Why it degraded, or ``None`` when the refresh was incremental.
    files_touched_count
        Source files re-read (added + modified).
    files_removed_count
        Source files whose rows were dropped because the file is gone.
    rows_deleted_count
        Raw rows deleted across both raw tables.
    rows_inserted_count
        Raw rows inserted across both raw tables.
    covers_through
        The snapshot's new coverage instant — see :class:`RawSnapshot`.
    """

    rebuilt: bool
    rebuild_reason: str | None
    files_touched_count: int
    files_removed_count: int
    rows_deleted_count: int
    rows_inserted_count: int
    covers_through: datetime | None

    @property
    def is_noop(self) -> bool:
        """True when nothing moved, so no SQL was issued against the raw tables."""
        return not self.rebuilt and self.files_touched_count == 0 and self.files_removed_count == 0


@dataclass(frozen=True, slots=True)
class PendingFile:
    """One dirty source file awaiting a flush.

    Attributes
    ----------
    path
        The source file that received events.
    first_event_ns
        Monotonic-clock nanoseconds of the first event since the last flush.
        Anchors the ``max_wait_ns`` starvation bound.
    last_event_ns
        Monotonic-clock nanoseconds of the most recent event. Anchors the
        ``quiet_period_ns`` idle bound.
    """

    path: str
    first_event_ns: int
    last_event_ns: int


def note_events(
    pending: Mapping[str, PendingFile],
    paths: Iterable[str],
    *,
    at_ns: int,
) -> dict[str, PendingFile]:
    """Fold new events into the pending set, returning a new mapping.

    A path already pending keeps its ``first_event_ns`` (so repeated writes
    cannot push its ``max_wait_ns`` deadline out) and advances its
    ``last_event_ns``. A path seen for the first time anchors both at ``at_ns``.

    ``last_event_ns`` never moves backwards: an out-of-order event carrying an
    older timestamp than one already recorded would otherwise make an idle file
    look freshly active and delay its flush.

    Parameters
    ----------
    pending
        The current pending set; not mutated.
    paths
        Source files that just received filesystem events.
    at_ns
        Monotonic-clock nanoseconds to stamp the events with.

    Returns
    -------
    dict[str, PendingFile]
        A new pending set including the events.
    """
    updated = dict(pending)
    for path in paths:
        prior = updated.get(path)
        if prior is None:
            updated[path] = PendingFile(path=path, first_event_ns=at_ns, last_event_ns=at_ns)
            continue
        updated[path] = PendingFile(
            path=path,
            first_event_ns=prior.first_event_ns,
            last_event_ns=max(prior.last_event_ns, at_ns),
        )
    return updated


def due_paths(
    pending: Mapping[str, PendingFile],
    *,
    now_ns: int,
    quiet_period_ns: int,
    max_wait_ns: int,
) -> tuple[str, ...]:
    """Which pending paths are ready to flush at ``now_ns``.

    A path is due when EITHER bound is met:

    * **idle** — it has been quiet for at least ``quiet_period_ns``, i.e. the
      writer appears to have finished the turn;
    * **starvation guard** — at least ``max_wait_ns`` has elapsed since its
      first unflushed event, so a file being appended to continuously still
      flushes on a fixed cadence instead of never.

    A non-positive ``max_wait_ns`` disables the starvation guard, leaving the
    pure idle rule. A non-positive ``quiet_period_ns`` makes every pending
    path due immediately, which is the ``--once`` path's behaviour.

    Returns
    -------
    tuple[str, ...]
        Sorted due paths, so a flush batch is deterministic.
    """
    due = [
        path
        for path, entry in pending.items()
        if now_ns - entry.last_event_ns >= quiet_period_ns
        or (max_wait_ns > 0 and now_ns - entry.first_event_ns >= max_wait_ns)
    ]
    return tuple(sorted(due))


def drop_paths(
    pending: Mapping[str, PendingFile],
    paths: Iterable[str],
) -> dict[str, PendingFile]:
    """Remove flushed paths from the pending set, returning a new mapping."""
    dropped = set(paths)
    return {path: entry for path, entry in pending.items() if path not in dropped}


__all__ = [
    "NANOS_PER_SECOND",
    "PendingFile",
    "RawRefreshStats",
    "RawSnapshot",
    "SourceDelta",
    "diff_source_mtimes",
    "drop_paths",
    "due_paths",
    "newest_mtime_ns",
    "note_events",
]

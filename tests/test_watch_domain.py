"""Pure watermark-diff and debounce math (``domain/watch.py``).

No filesystem, no DuckDB: every case here is a mapping in and a value out. The
starvation guard in :func:`due_paths` is the one behaviour with a Lean
counterpart (``proofs/ClaudeSql/Debounce.lean``); the test and the proof cover
the same rule from opposite directions — the test on concrete schedules, the
proof over all of them.
"""

from __future__ import annotations

from claude_sql.domain.watch import (
    NANOS_PER_SECOND,
    PendingFile,
    diff_source_mtimes,
    drop_paths,
    due_paths,
    newest_mtime_ns,
    note_events,
)

SECOND = NANOS_PER_SECOND


# ---------------------------------------------------------------------------
# diff_source_mtimes
# ---------------------------------------------------------------------------


def test_diff_partitions_added_modified_removed() -> None:
    delta = diff_source_mtimes(
        {"a.jsonl": 100, "b.jsonl": 200, "gone.jsonl": 300},
        {"a.jsonl": 100, "b.jsonl": 999, "new.jsonl": 400},
    )
    assert delta.added == ("new.jsonl",)
    assert delta.modified == ("b.jsonl",)
    assert delta.removed == ("gone.jsonl",)
    assert delta.touched == ("new.jsonl", "b.jsonl")
    assert delta.changed_count == 3
    assert not delta.is_empty


def test_diff_of_identical_scans_is_empty() -> None:
    scan = {"a.jsonl": 100, "b.jsonl": 200}
    delta = diff_source_mtimes(scan, dict(scan))
    assert delta.is_empty
    assert delta.changed_count == 0
    assert delta.touched == ()


def test_diff_treats_a_backwards_mtime_as_modified() -> None:
    """An mtime that went BACKWARDS still invalidates the snapshot.

    A restored backup or a file replaced by an older copy leaves the recorded
    watermark describing bytes that are no longer on disk. A ``>`` comparison
    would skip it and serve the stale rows forever.
    """
    delta = diff_source_mtimes({"a.jsonl": 500}, {"a.jsonl": 100})
    assert delta.modified == ("a.jsonl",)


def test_diff_output_is_sorted_for_determinism() -> None:
    delta = diff_source_mtimes({}, {"c": 1, "a": 1, "b": 1})
    assert delta.added == ("a", "b", "c")


def test_newest_mtime_is_none_for_an_empty_scan() -> None:
    """``None``, not 0 — a zero would read as "covers through 1970"."""
    assert newest_mtime_ns({}) is None
    assert newest_mtime_ns({"a": 7, "b": 42, "c": 12}) == 42


# ---------------------------------------------------------------------------
# note_events
# ---------------------------------------------------------------------------


def test_note_events_anchors_both_stamps_on_first_sighting() -> None:
    pending = note_events({}, ["a.jsonl"], at_ns=5 * SECOND)
    assert pending["a.jsonl"] == PendingFile(
        path="a.jsonl", first_event_ns=5 * SECOND, last_event_ns=5 * SECOND
    )


def test_note_events_keeps_first_stamp_and_advances_last() -> None:
    """The ``max_wait`` deadline must not be pushed out by later writes."""
    pending = note_events({}, ["a.jsonl"], at_ns=5 * SECOND)
    pending = note_events(pending, ["a.jsonl"], at_ns=9 * SECOND)
    assert pending["a.jsonl"].first_event_ns == 5 * SECOND
    assert pending["a.jsonl"].last_event_ns == 9 * SECOND


def test_note_events_never_regresses_the_last_stamp() -> None:
    pending = note_events({}, ["a.jsonl"], at_ns=9 * SECOND)
    pending = note_events(pending, ["a.jsonl"], at_ns=4 * SECOND)
    assert pending["a.jsonl"].last_event_ns == 9 * SECOND


def test_note_events_does_not_mutate_its_input() -> None:
    original = note_events({}, ["a.jsonl"], at_ns=SECOND)
    snapshot = dict(original)
    note_events(original, ["b.jsonl"], at_ns=2 * SECOND)
    assert original == snapshot


# ---------------------------------------------------------------------------
# due_paths
# ---------------------------------------------------------------------------


def test_a_file_still_being_written_is_not_due() -> None:
    pending = note_events({}, ["a.jsonl"], at_ns=10 * SECOND)
    assert (
        due_paths(
            pending,
            now_ns=15 * SECOND,
            quiet_period_ns=20 * SECOND,
            max_wait_ns=120 * SECOND,
        )
        == ()
    )


def test_a_quiet_file_is_due() -> None:
    pending = note_events({}, ["a.jsonl"], at_ns=10 * SECOND)
    assert due_paths(
        pending,
        now_ns=30 * SECOND,
        quiet_period_ns=20 * SECOND,
        max_wait_ns=120 * SECOND,
    ) == ("a.jsonl",)


def test_continuous_writes_still_flush_at_the_max_wait_bound() -> None:
    """The starvation guard: a file appended to forever must still flush.

    A session written to every second with a 20-second quiet period never
    satisfies the idle rule. Without the ``max_wait_ns`` disjunct its rows would
    never reach the snapshot — the failure mode a naive debounce ships with.
    """
    pending: dict[str, PendingFile] = {}
    now = 0
    for _ in range(60):
        now += 1 * SECOND
        pending = note_events(pending, ["busy.jsonl"], at_ns=now)
    # Never quiet — the last event is always 1s old.
    assert due_paths(pending, now_ns=now, quiet_period_ns=20 * SECOND, max_wait_ns=0) == ()
    # ...but the first event is 59s old, so a 30s max-wait bound fires.
    assert due_paths(pending, now_ns=now, quiet_period_ns=20 * SECOND, max_wait_ns=30 * SECOND) == (
        "busy.jsonl",
    )


def test_a_nonpositive_quiet_period_makes_everything_due() -> None:
    """The ``--once`` path: flush whatever is pending, immediately."""
    pending = note_events({}, ["a.jsonl", "b.jsonl"], at_ns=10 * SECOND)
    assert due_paths(pending, now_ns=10 * SECOND, quiet_period_ns=0, max_wait_ns=0) == (
        "a.jsonl",
        "b.jsonl",
    )


def test_due_paths_is_sorted() -> None:
    pending = note_events({}, ["c", "a", "b"], at_ns=0)
    assert due_paths(pending, now_ns=SECOND, quiet_period_ns=0, max_wait_ns=0) == ("a", "b", "c")


def test_due_paths_is_monotone_in_now() -> None:
    """Once due, always due — time passing never un-dues a pending file."""
    pending = note_events({}, ["a"], at_ns=0)
    first_due_at = next(
        n
        for n in range(60)
        if due_paths(
            pending, now_ns=n * SECOND, quiet_period_ns=20 * SECOND, max_wait_ns=120 * SECOND
        )
    )
    for later in range(first_due_at, 60):
        assert due_paths(
            pending,
            now_ns=later * SECOND,
            quiet_period_ns=20 * SECOND,
            max_wait_ns=120 * SECOND,
        ) == ("a",)


# ---------------------------------------------------------------------------
# drop_paths
# ---------------------------------------------------------------------------


def test_drop_paths_removes_only_the_flushed_entries() -> None:
    pending = note_events({}, ["a", "b", "c"], at_ns=0)
    remaining = drop_paths(pending, ["a", "c"])
    assert set(remaining) == {"b"}


def test_drop_paths_ignores_unknown_paths() -> None:
    pending = note_events({}, ["a"], at_ns=0)
    assert set(drop_paths(pending, ["nope"])) == {"a"}

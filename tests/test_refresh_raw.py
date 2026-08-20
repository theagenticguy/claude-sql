"""Incremental raw refresh + snapshot coverage (``duckdb_views.refresh_raw``).

These run against a real DuckDB connection over a real on-disk corpus, not a
fake. The bug class this tier exists to catch is state semantics: ``refresh_raw``
both mutates and re-reads the same TEMP TABLE across calls, and a stateless fake
would happily report success while leaving two generations of one transcript's
rows in the table.

Two cases carry the metarepo's hard-won lessons explicitly:

* ``test_appending_to_one_session_leaves_a_neighbours_rows_intact`` seeds a
  NEIGHBOUR's rows. A single-session corpus passes against a ``DELETE`` with a
  missing predicate, because there is nothing else to destroy.
* ``test_refresh_after_append_does_not_duplicate_rows`` pins the delete-then-
  insert order. Insert-only leaves the pre-append rows behind and every count in
  the corpus silently doubles for that session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import duckdb
import pytest

from claude_sql.infrastructure import duckdb_views, source_files
from conftest import _seed_subagent_stub, make_user_msg, write_session_jsonl

if TYPE_CHECKING:
    from pathlib import Path

SID_ONE = "11111111-1111-1111-1111-111111111111"
SID_TWO = "22222222-2222-2222-2222-222222222222"
SID_THREE = "33333333-3333-3333-3333-333333333333"


def _register(tmp_corpus: dict[str, Any]) -> duckdb.DuckDBPyConnection:
    """A connection with the raw readers + watermark built over ``tmp_corpus``."""
    con = duckdb.connect(":memory:")
    duckdb_views.register_raw(
        con,
        glob=tmp_corpus["glob"],
        subagent_glob=tmp_corpus["subagent_glob"],
        subagent_meta_glob=tmp_corpus["subagent_meta_glob"],
    )
    return con


def _refresh(
    con: duckdb.DuckDBPyConnection,
    tmp_corpus: dict[str, Any],
    **kwargs: Any,
) -> Any:
    return duckdb_views.refresh_raw(
        con,
        glob=tmp_corpus["glob"],
        subagent_glob=tmp_corpus["subagent_glob"],
        subagent_meta_glob=tmp_corpus["subagent_meta_glob"],
        **kwargs,
    )


def _rows_for(con: duckdb.DuckDBPyConnection, session_id: str) -> int:
    row = con.execute(
        "SELECT count(*) FROM v_raw_events WHERE sessionId = ?", [session_id]
    ).fetchone()
    assert row is not None
    return int(row[0])


def _total_rows(con: duckdb.DuckDBPyConnection) -> int:
    row = con.execute("SELECT count(*) FROM v_raw_events").fetchone()
    assert row is not None
    return int(row[0])


def _append_to_session(root: Path, session_id: str, *, extra: int) -> None:
    """Rewrite a session transcript with ``extra`` additional records.

    Rewrites rather than appends because the fixture writer owns the file
    format; the observable effect on disk (more records, newer mtime) is what
    ``refresh_raw`` reacts to either way.
    """
    existing = (root / f"{session_id}.jsonl").read_text().splitlines()
    messages = [
        make_user_msg(
            f"extra-{session_id}-{index}",
            session_id,
            f"appended record {index} long enough to clear the text filter",
            ts="2026-04-03T10:00:00.000Z",
        )
        for index in range(extra)
    ]
    path = root / f"{session_id}.jsonl"
    write_session_jsonl(path, messages=messages)
    # Prepend the original lines back so the file only ever grows.
    path.write_text("\n".join([*existing, *path.read_text().splitlines()]) + "\n")


# ---------------------------------------------------------------------------
# Incremental happy paths
# ---------------------------------------------------------------------------


def test_refresh_after_append_does_not_duplicate_rows(tmp_corpus: dict[str, Any]) -> None:
    con = _register(tmp_corpus)
    try:
        before = _rows_for(con, SID_ONE)
        _append_to_session(tmp_corpus["root"], SID_ONE, extra=3)

        stats = _refresh(con, tmp_corpus)

        assert not stats.rebuilt
        assert stats.files_touched_count == 1
        assert _rows_for(con, SID_ONE) == before + 3
    finally:
        con.close()


def test_appending_to_one_session_leaves_a_neighbours_rows_intact(
    tmp_corpus: dict[str, Any],
) -> None:
    """Seed a NEIGHBOUR's rows: a one-session corpus can't catch an over-broad DELETE."""
    con = _register(tmp_corpus)
    try:
        neighbour_before = _rows_for(con, SID_TWO)
        assert neighbour_before > 0, "fixture must supply a second session to contaminate with"
        _append_to_session(tmp_corpus["root"], SID_ONE, extra=2)

        _refresh(con, tmp_corpus)

        assert _rows_for(con, SID_TWO) == neighbour_before
    finally:
        con.close()


def test_refresh_picks_up_a_brand_new_session_file(tmp_corpus: dict[str, Any]) -> None:
    con = _register(tmp_corpus)
    try:
        before = _total_rows(con)
        write_session_jsonl(
            tmp_corpus["root"] / f"{SID_THREE}.jsonl",
            messages=[
                make_user_msg(
                    "n1",
                    SID_THREE,
                    "a new session that landed after the snapshot was built",
                    ts="2026-04-04T08:00:00.000Z",
                )
            ],
        )

        stats = _refresh(con, tmp_corpus)

        assert not stats.rebuilt
        assert stats.files_touched_count == 1
        assert _rows_for(con, SID_THREE) == 1
        assert _total_rows(con) == before + 1
    finally:
        con.close()


def test_refresh_drops_rows_for_a_deleted_file(tmp_corpus: dict[str, Any]) -> None:
    con = _register(tmp_corpus)
    try:
        doomed = _rows_for(con, SID_TWO)
        survivor = _rows_for(con, SID_ONE)
        (tmp_corpus["root"] / f"{SID_TWO}.jsonl").unlink()

        stats = _refresh(con, tmp_corpus)

        assert not stats.rebuilt
        assert stats.files_removed_count == 1
        assert stats.rows_deleted_count == doomed
        assert _rows_for(con, SID_TWO) == 0
        assert _rows_for(con, SID_ONE) == survivor
    finally:
        con.close()


def test_refresh_is_a_noop_when_nothing_moved(tmp_corpus: dict[str, Any]) -> None:
    con = _register(tmp_corpus)
    try:
        before = _total_rows(con)

        stats = _refresh(con, tmp_corpus)

        assert stats.is_noop
        assert not stats.rebuilt
        assert stats.rows_inserted_count == 0
        assert stats.rows_deleted_count == 0
        assert _total_rows(con) == before
    finally:
        con.close()


def test_refresh_reads_a_subagent_sidecar_append(tmp_corpus: dict[str, Any]) -> None:
    """The subagent table has its own delta — a refresh must not skip it."""
    con = _register(tmp_corpus)
    try:
        row = con.execute("SELECT count(*) FROM v_raw_subagents").fetchone()
        assert row is not None
        before = int(row[0])
        sa_dir = (
            tmp_corpus["tmp_path"]
            / "projects"
            / "proj-stub"
            / "00000000-0000-0000-0000-000000000000"
            / "subagents"
        )
        write_session_jsonl(
            sa_dir / "agent-second.jsonl",
            messages=[
                make_user_msg(
                    "sa-2",
                    "placeholder",
                    "a second subagent transcript written after registration",
                    ts="2026-04-05T08:00:00.000Z",
                )
            ],
        )

        stats = _refresh(con, tmp_corpus)

        assert not stats.rebuilt
        after = con.execute("SELECT count(*) FROM v_raw_subagents").fetchone()
        assert after is not None
        assert int(after[0]) == before + 1
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Column layout — the positional-INSERT hazard
# ---------------------------------------------------------------------------


def test_refresh_preserves_the_registered_column_layout(tmp_corpus: dict[str, Any]) -> None:
    """``INSERT INTO ... SELECT`` is positional; a drifted projection misfiles values.

    Both the ``DESCRIBE`` layout and one computed column's *value* are checked:
    an equal-length projection with two columns transposed would keep the layout
    identical and silently write ``source_file`` into ``session_id_file``.
    """
    con = _register(tmp_corpus)
    try:
        before_layout = con.execute("DESCRIBE SELECT * FROM v_raw_events").fetchall()
        write_session_jsonl(
            tmp_corpus["root"] / f"{SID_THREE}.jsonl",
            messages=[
                make_user_msg(
                    "n1",
                    SID_THREE,
                    "new session used to check the refreshed column layout",
                    ts="2026-04-04T08:00:00.000Z",
                )
            ],
        )

        _refresh(con, tmp_corpus)

        assert con.execute("DESCRIBE SELECT * FROM v_raw_events").fetchall() == before_layout
        row = con.execute(
            "SELECT session_id_file, source_file FROM v_raw_events WHERE sessionId = ?",
            [SID_THREE],
        ).fetchone()
        assert row is not None
        session_id_file, source_file = row
        assert session_id_file == SID_THREE
        assert str(source_file).endswith(f"{SID_THREE}.jsonl")
    finally:
        con.close()


# ---------------------------------------------------------------------------
# snapshot_as_of
# ---------------------------------------------------------------------------


def test_snapshot_is_none_before_any_registration() -> None:
    con = duckdb.connect(":memory:")
    try:
        assert duckdb_views.snapshot_as_of(con) is None
    finally:
        con.close()


def test_snapshot_reports_coverage_and_advances_on_refresh(
    tmp_corpus: dict[str, Any],
) -> None:
    con = _register(tmp_corpus)
    try:
        first = duckdb_views.snapshot_as_of(con)
        assert first is not None
        assert first.refresh_count == 0
        assert first.files_scanned_count is not None
        assert first.files_scanned_count > 0
        assert first.covers_through is not None

        write_session_jsonl(
            tmp_corpus["root"] / f"{SID_THREE}.jsonl",
            messages=[
                make_user_msg(
                    "n1",
                    SID_THREE,
                    "a session newer than the first snapshot's coverage",
                    ts="2026-04-04T08:00:00.000Z",
                )
            ],
        )
        _refresh(con, tmp_corpus)

        second = duckdb_views.snapshot_as_of(con)
        assert second is not None
        assert second.refresh_count == 1
        assert second.files_scanned_count == first.files_scanned_count + 1
        assert second.covers_through is not None
        assert second.covers_through >= first.covers_through
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Documented degradations to a full rebuild
# ---------------------------------------------------------------------------


def test_refresh_rebuilds_when_the_connection_has_no_snapshot(
    tmp_corpus: dict[str, Any],
) -> None:
    con = duckdb.connect(":memory:")
    try:
        stats = _refresh(con, tmp_corpus)

        assert stats.rebuilt
        assert stats.rebuild_reason is not None
        assert _total_rows(con) > 0
    finally:
        con.close()


def test_refresh_rebuilds_over_the_incremental_cap(tmp_corpus: dict[str, Any]) -> None:
    con = _register(tmp_corpus)
    try:
        _append_to_session(tmp_corpus["root"], SID_ONE, extra=1)

        stats = _refresh(con, tmp_corpus, max_incremental_files=0)

        assert stats.rebuilt
        assert stats.rebuild_reason is not None
        assert "cap" in stats.rebuild_reason
    finally:
        con.close()


def test_refresh_rebuilds_when_the_watermark_is_unobservable(
    tmp_corpus: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remote corpus has no local mtimes; an empty scan would delete everything."""
    con = _register(tmp_corpus)
    try:
        monkeypatch.setattr(source_files, "scan_globs", lambda _patterns: None)

        stats = _refresh(con, tmp_corpus)

        assert stats.rebuilt
        assert stats.rebuild_reason is not None
        assert "unobservable" in stats.rebuild_reason
        assert _total_rows(con) > 0
    finally:
        con.close()


def test_scan_globs_returns_none_for_a_remote_glob() -> None:
    assert source_files.scan_globs(["s3://bucket/prefix/*/*.jsonl"]) is None


def test_scan_globs_distinguishes_empty_from_unobservable(tmp_path: Path) -> None:
    """A local glob matching nothing is ``{}``; a remote one is ``None``."""
    assert source_files.scan_globs([str(tmp_path / "nothing" / "*.jsonl")]) == {}


def test_scan_glob_mtimes_skips_directories(tmp_path: Path) -> None:
    (tmp_path / "adir.jsonl").mkdir()
    (tmp_path / "afile.jsonl").write_text("{}\n")

    scanned = source_files.scan_glob_mtimes(str(tmp_path / "*.jsonl"))

    assert set(scanned) == {str(tmp_path / "afile.jsonl")}


def test_subagent_stub_helper_is_used(tmp_path: Path) -> None:
    """Guard the conftest import this module relies on (used by other cases too)."""
    sa_glob, sa_meta_glob = _seed_subagent_stub(tmp_path)
    assert source_files.scan_globs([sa_glob])
    assert sa_meta_glob.endswith(".meta.json")

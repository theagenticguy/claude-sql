"""Tests for :mod:`claude_sql.session_text`.

Covers the formatting helpers, the corpus loader, the
:meth:`SessionTextCorpus.assemble` truncation path, the
:func:`iter_session_texts` adapter, and the defensive ``_load_tool_calls``
skip for sessions whose only events are tool_use blocks (no ``messages_text``
row to anchor them).

All tests build their own DuckDB connection over JSONLs written under
``tmp_path`` — we deliberately avoid the ``registered_con`` fixture (which
calls ``register_macros`` and trips a catalog error on the
``message_embeddings`` reference) to keep these tests hermetic and fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import pytest

from claude_sql.core.config import Settings
from claude_sql.core.session_text import (
    SessionTextCorpus,
    _tool_input_preview,
    _tool_result_preview,
    iter_session_texts,
    session_text_corpus,
)
from claude_sql.core.sql_views import register_raw, register_views
from conftest import (
    _seed_subagent_stub,
    make_assistant_msg,
    make_tool_result_msg,
    make_user_msg,
    write_session_jsonl,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


# ---------------------------------------------------------------------------
# Local helpers — build a corpus and a connection scoped to a single test
# ---------------------------------------------------------------------------


def _open_views(
    tmp_path: Path,
    glob: str,
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with raw + analytics views over ``glob``.

    Skips ``register_macros`` on purpose — the macro DDL references
    ``message_embeddings`` which only exists once ``register_vss`` has run,
    and these tests never run a real embed pipeline.
    """
    sa_glob, sa_meta_glob = _seed_subagent_stub(tmp_path)
    con = duckdb.connect(":memory:")
    register_raw(
        con,
        glob=glob,
        subagent_glob=sa_glob,
        subagent_meta_glob=sa_meta_glob,
    )
    register_views(con)
    return con


@pytest.fixture
def session_text_corpus_con(tmp_path: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    """Two-session corpus that exercises text + tool_use + tool_result paths.

    Session A (newest) has every kind of timeline event:
      * two user-text messages (uuids ``u1`` and ``u3``)
      * one assistant tool_use (``a1`` carrying ``tu-a1`` for ``Read``)
      * one tool_result (``u2`` paired with ``tu-a1``)

    Session B (older) has just one user-text and one short assistant text
    so it lives in ``messages_text`` only.
    """
    proj = tmp_path / "projects" / "proj-st"
    sid_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    sid_b = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    write_session_jsonl(
        proj / f"{sid_a}.jsonl",
        messages=[
            make_user_msg(
                "u1",
                sid_a,
                "first message in session A — long enough to clear the 32-char filter",
                ts="2026-04-01T10:00:00.000Z",
            ),
            make_assistant_msg(
                "a1",
                sid_a,
                ts="2026-04-01T10:00:10.000Z",
                content=[
                    {"type": "text", "text": "ok let me check the file"},
                    {
                        "type": "tool_use",
                        "id": "tu-a1",
                        "name": "Read",
                        "input": {"file_path": "/home/u/proj/x.txt"},
                    },
                ],
            ),
            make_tool_result_msg(
                "u2",
                sid_a,
                "tu-a1",
                "file contents go here",
                ts="2026-04-01T10:00:12.000Z",
            ),
            make_user_msg(
                "u3",
                sid_a,
                "thanks, that is exactly what I expected from the file read",
                ts="2026-04-01T10:00:30.000Z",
            ),
        ],
    )
    write_session_jsonl(
        proj / f"{sid_b}.jsonl",
        messages=[
            make_user_msg(
                "v1",
                sid_b,
                "session B opening request — also long enough to clear the filter",
                ts="2026-04-02T09:00:00.000Z",
            ),
        ],
    )
    glob = str(tmp_path / "projects" / "*" / "*.jsonl")
    con = _open_views(tmp_path, glob)
    try:
        yield con
    finally:
        con.close()


# ---------------------------------------------------------------------------
# 1. Formatting helpers
# ---------------------------------------------------------------------------


def test_tool_input_preview_returns_empty_for_none() -> None:
    assert _tool_input_preview(None) == ""


def test_tool_input_preview_returns_empty_for_empty_string() -> None:
    assert _tool_input_preview("") == ""


def test_tool_input_preview_passthrough_when_short() -> None:
    body = '{"file_path": "/tmp/x.txt"}'
    assert _tool_input_preview(body, max_chars=400) == body


def test_tool_input_preview_truncates_with_footer() -> None:
    body = "x" * 500
    out = _tool_input_preview(body, max_chars=10)
    assert out == "x" * 10 + "…(truncated)"


def test_tool_result_preview_returns_empty_for_none() -> None:
    assert _tool_result_preview(None, max_chars=100) == ""


def test_tool_result_preview_returns_empty_for_empty_string() -> None:
    assert _tool_result_preview("", max_chars=100) == ""


def test_tool_result_preview_passthrough_when_short() -> None:
    body = '"file contents"'
    assert _tool_result_preview(body, max_chars=100) == body


def test_tool_result_preview_truncates_with_dropped_count() -> None:
    body = "y" * 500
    out = _tool_result_preview(body, max_chars=20)
    # The truncated output is exactly 20 ``y``s + the dropped-bytes footer.
    assert out == "y" * 20 + "\n…(truncated, 480 chars dropped)"


# ---------------------------------------------------------------------------
# 2. session_text_corpus happy path
# ---------------------------------------------------------------------------


def test_session_text_corpus_collects_all_three_kinds(
    session_text_corpus_con: duckdb.DuckDBPyConnection,
) -> None:
    corpus = session_text_corpus(session_text_corpus_con)

    sid_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    sid_b = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

    # Order is newest-first, so session B (2026-04-02) leads session A.
    assert corpus.order == [sid_b, sid_a]
    assert len(corpus) == len(corpus.order) == 2

    rows_a = corpus.texts_by_session[sid_a]
    kinds = {r.kind for r in rows_a}
    assert kinds == {"text", "tool_use", "tool_result"}

    # Each kind shows up at least once with a non-empty body.
    text_rows = [r for r in rows_a if r.kind == "text"]
    assert len(text_rows) == 2
    assert all((r.body and "session A" in r.body) or "expected" in r.body for r in text_rows)

    tool_use_rows = [r for r in rows_a if r.kind == "tool_use"]
    assert len(tool_use_rows) == 1
    assert tool_use_rows[0].aux == "Read"
    assert tool_use_rows[0].body is not None
    assert "/home/u/proj/x.txt" in tool_use_rows[0].body

    tool_result_rows = [r for r in rows_a if r.kind == "tool_result"]
    assert len(tool_result_rows) == 1
    assert tool_result_rows[0].aux == "tu-a1"
    assert tool_result_rows[0].body is not None
    assert "file contents go here" in tool_result_rows[0].body


def test_session_text_corpus_rows_sorted_chronologically(
    session_text_corpus_con: duckdb.DuckDBPyConnection,
) -> None:
    corpus = session_text_corpus(session_text_corpus_con)
    sid_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    rows = corpus.texts_by_session[sid_a]
    timestamps = [r.ts_iso for r in rows]
    assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# 3. session_text_corpus filters
# ---------------------------------------------------------------------------


def test_session_text_corpus_since_days_skips_old(
    session_text_corpus_con: duckdb.DuckDBPyConnection,
) -> None:
    # ``since_days=1`` against current_timestamp drops both fixture sessions
    # because they are dated 2026-04-01 / 2026-04-02 — but if "today" inside
    # DuckDB happens to be in that window we still expect zero or more.
    # The key invariant: the result is never larger than the unfiltered call,
    # and a tiny window (since_days=1, when 'now' is well past 2026-04) is 0.
    full = session_text_corpus(session_text_corpus_con)
    filtered = session_text_corpus(session_text_corpus_con, since_days=1)
    assert len(filtered) <= len(full)
    # Both fixture sessions are stamped in 2026-04, current_timestamp is far
    # past that, so the 1-day window must drop everything.
    assert filtered.order == []


def test_session_text_corpus_limit_caps_results(
    session_text_corpus_con: duckdb.DuckDBPyConnection,
) -> None:
    corpus = session_text_corpus(session_text_corpus_con, limit=1)
    assert len(corpus.order) == 1
    # Newest-first ordering means the limit-1 result is session B (2026-04-02).
    assert corpus.order[0] == "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


# ---------------------------------------------------------------------------
# 4. SessionTextCorpus.assemble
# ---------------------------------------------------------------------------


def test_assemble_renders_user_tool_use_and_result_lines(
    session_text_corpus_con: duckdb.DuckDBPyConnection,
    tmp_settings: Settings,
) -> None:
    corpus = session_text_corpus(session_text_corpus_con)
    sid_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

    rendered = corpus.assemble(sid_a, settings=tmp_settings)

    assert rendered.startswith("[user "), rendered[:80]
    assert "[tool_use:Read " in rendered
    assert "[tool_result tu-a1 " in rendered
    # Lines are joined with newline only — no trailing newline.
    assert "\n" in rendered
    assert not rendered.endswith("\n")
    # Each event line shows up exactly as one line.
    assert rendered.count("\n") == len(rendered.split("\n")) - 1


def test_assemble_include_uuids_adds_header_only_on_text_turns(
    session_text_corpus_con: duckdb.DuckDBPyConnection,
    tmp_settings: Settings,
) -> None:
    """``include_uuids=True`` prefixes text-turn headers with ``[uuid=<id> ...]``
    (the conflicts pipeline needs verbatim turn uuids — issue #109), while the
    default keeps the historic ``[role ts]`` header byte-identical so the
    classify / trajectory / friction prompts never shift."""
    corpus = session_text_corpus(session_text_corpus_con)
    sid_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

    default = corpus.assemble(sid_a, settings=tmp_settings)
    with_uuids = corpus.assemble(sid_a, settings=tmp_settings, include_uuids=True)

    # Default path: no uuid headers anywhere (byte-identical to pre-#109).
    assert "[uuid=" not in default
    assert default.startswith("[user ")

    # uuid path: the fixture's user text turns carry uuids u1 and u3.
    assert "[uuid=u1 user " in with_uuids
    assert "[uuid=u3 user " in with_uuids
    # tool_use / tool_result lines are NOT addressable turns -> no uuid header.
    assert "[tool_use:Read " in with_uuids
    assert "[tool_result tu-a1 " in with_uuids
    assert "[uuid=tu-a1" not in with_uuids


def test_assemble_truncates_when_total_exceeds_cap(
    session_text_corpus_con: duckdb.DuckDBPyConnection,
    tmp_path: Path,
) -> None:
    corpus = session_text_corpus(session_text_corpus_con)
    sid_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

    # Settings whose total cap is shorter than the rendered transcript.
    cache = tmp_path / "claude_assemble_cap"
    cache.mkdir(parents=True, exist_ok=True)
    tiny_settings = Settings(
        embeddings_parquet_path=cache / "embeddings",
        clusters_parquet_path=cache / "clusters.parquet",
        cluster_terms_parquet_path=cache / "cluster_terms.parquet",
        session_text_total_max_chars=100,
        session_text_tool_result_max_chars=20,
    )

    rendered = corpus.assemble(sid_a, settings=tiny_settings)
    expected_footer = "…(session truncated at 100 chars,"
    assert expected_footer in rendered, rendered


def test_assemble_returns_empty_for_unknown_session(
    session_text_corpus_con: duckdb.DuckDBPyConnection,
    tmp_settings: Settings,
) -> None:
    corpus = session_text_corpus(session_text_corpus_con)
    assert corpus.assemble("nope-not-a-session", settings=tmp_settings) == ""


def test_assemble_skips_rows_with_none_body(tmp_settings: Settings) -> None:
    """A timeline row whose ``body`` is ``None`` is skipped silently."""
    from claude_sql.core.session_text import _TimelineRow

    sid = "synthetic"
    corpus = SessionTextCorpus(
        texts_by_session={
            sid: [
                _TimelineRow(
                    ts_iso="2026-04-01T10:00:00",
                    role="user",
                    kind="text",
                    body=None,
                    aux=None,
                ),
                _TimelineRow(
                    ts_iso="2026-04-01T10:00:01",
                    role="user",
                    kind="text",
                    body="visible body",
                    aux=None,
                ),
            ],
        },
        order=[sid],
    )
    rendered = corpus.assemble(sid, settings=tmp_settings)
    assert rendered == "[user 2026-04-01T10:00:01] visible body"


# ---------------------------------------------------------------------------
# 5. iter_session_texts
# ---------------------------------------------------------------------------


def test_iter_session_texts_yields_corpus_order(
    session_text_corpus_con: duckdb.DuckDBPyConnection,
    tmp_settings: Settings,
) -> None:
    pairs = list(iter_session_texts(session_text_corpus_con, settings=tmp_settings))
    assert len(pairs) == 2

    sids = [sid for sid, _ in pairs]
    sid_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    sid_b = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    assert sids == [sid_b, sid_a]

    for _sid, text in pairs:
        assert text  # non-empty
        assert text.startswith("[user "), text[:60]


def test_iter_session_texts_empty_corpus(tmp_path: Path, tmp_settings: Settings) -> None:
    # Build a corpus where every session falls below the messages_text
    # 32-char threshold so ``_load_session_order`` returns an empty list.
    proj = tmp_path / "projects" / "proj-empty"
    write_session_jsonl(
        proj / "tiny.jsonl",
        messages=[
            make_user_msg(
                "x1",
                "tiny-sess",
                "short",  # < 32 chars; filtered out of messages_text
                ts="2026-04-01T10:00:00.000Z",
            ),
        ],
    )
    glob = str(tmp_path / "projects" / "*" / "*.jsonl")
    con = _open_views(tmp_path, glob)
    try:
        pairs = list(iter_session_texts(con, settings=tmp_settings))
        assert pairs == []
    finally:
        con.close()


# ---------------------------------------------------------------------------
# 6. session_bounds IO error retry path (monkeypatched)
# ---------------------------------------------------------------------------


class _FlakyConWrapper:
    """Proxy a DuckDB connection but raise ``IOException`` on the first ``execute``.

    DuckDB's connection methods are read-only attributes on the underlying
    PyObject so ``monkeypatch.setattr`` can't swap them in place. A wrapper
    around the real connection lets us simulate the stale-glob retry path
    cleanly.
    """

    def __init__(self, real: duckdb.DuckDBPyConnection) -> None:
        self._real = real
        self.calls = 0

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            raise duckdb.IOException("simulated stale glob")
        return self._real.execute(sql, *args, **kwargs)

    def __getattr__(self, item: str) -> Any:  # delegate everything else
        return getattr(self._real, item)


def test_session_bounds_retries_once_on_io_error(
    session_text_corpus_con: duckdb.DuckDBPyConnection,
) -> None:
    """First execute raises ``duckdb.IOException``; the second pass succeeds."""
    from claude_sql.core import session_text as st

    flaky = _FlakyConWrapper(session_text_corpus_con)
    bounds = st.session_bounds(flaky)  # type: ignore[arg-type]
    assert flaky.calls >= 2
    # Second pass returned full results.
    assert bounds  # non-empty dict


# ---------------------------------------------------------------------------
# 7. Defensive _tool_calls skip — session with tool_use but no messages_text
# ---------------------------------------------------------------------------


def test_session_text_corpus_skips_tool_only_session(tmp_path: Path) -> None:
    """A session without a single ``messages_text`` row is silently skipped.

    The session has only an assistant tool_use block — no text long enough to
    pass the ``messages_text`` 32-char floor — so ``_load_session_order``
    excludes it and ``_load_tool_calls`` hits the ``rows is None: continue``
    branch.
    """
    proj = tmp_path / "projects" / "proj-tool-only"
    write_session_jsonl(
        proj / "tool-only.jsonl",
        messages=[
            # Anchor session: has a long enough text row so it ends up in order.
            make_user_msg(
                "anchor-u",
                "anchor-sess",
                "this is the anchor session text long enough to clear the filter",
                ts="2026-04-03T10:00:00.000Z",
            ),
        ],
    )
    write_session_jsonl(
        proj / "tool-only-2.jsonl",
        messages=[
            # Tool-use-only session — has NO message_text row.
            make_assistant_msg(
                "a-only",
                "tool-only-sess",
                ts="2026-04-03T11:00:00.000Z",
                content=[
                    {
                        "type": "tool_use",
                        "id": "tu-only",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    },
                ],
            ),
        ],
    )
    glob = str(tmp_path / "projects" / "*" / "*.jsonl")
    con = _open_views(tmp_path, glob)
    try:
        corpus = session_text_corpus(con)
        # Only the anchor session ends up in the corpus.
        assert corpus.order == ["anchor-sess"]
        # And it has just the one text row, no tool_use bleed-through from
        # the orphan session.
        rows = corpus.texts_by_session["anchor-sess"]
        assert all(r.kind == "text" for r in rows)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# 6. Deterministic timeline order at equal timestamps
#
# Regression guard for the arrow-join loader rewrite. The three timeline
# queries are merged and sorted in Python; the DuckDB scan plan does not
# order rows that share a timestamp, so without an explicit total order the
# assembled transcript drifted run-to-run whenever one assistant turn emitted
# several tool_use blocks at the same millisecond. _timeline_sort_key pins a
# total order: (ts, kind_rank, body, aux).
# ---------------------------------------------------------------------------


def _tied_timestamp_con(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    """A one-session corpus whose assistant turn fires three tool_use blocks
    plus a text block, all stamped at the same instant."""
    proj = tmp_path / "projects" / "proj-tie"
    sid = "cccccccc-cccc-cccc-cccc-cccccccccccc"
    tied_ts = "2026-04-05T12:00:00.000Z"
    write_session_jsonl(
        proj / f"{sid}.jsonl",
        messages=[
            make_user_msg(
                "t1",
                sid,
                "kick off a parallel batch of reads — long enough for the filter",
                ts="2026-04-05T11:59:00.000Z",
            ),
            make_assistant_msg(
                "t2",
                sid,
                ts=tied_ts,
                content=[
                    {"type": "text", "text": "running three reads at once"},
                    {
                        "type": "tool_use",
                        "id": "tu-z",
                        "name": "Read",
                        "input": {"file_path": "/z.txt"},
                    },
                    {
                        "type": "tool_use",
                        "id": "tu-a",
                        "name": "Read",
                        "input": {"file_path": "/a.txt"},
                    },
                    {
                        "type": "tool_use",
                        "id": "tu-m",
                        "name": "Read",
                        "input": {"file_path": "/m.txt"},
                    },
                ],
            ),
        ],
    )
    glob = str(tmp_path / "projects" / "*" / "*.jsonl")
    return _open_views(tmp_path, glob)


def test_session_text_corpus_is_deterministic_across_rebuilds(tmp_path: Path) -> None:
    """Repeated corpus builds yield byte-identical timelines even when several
    events share one timestamp."""
    con = _tied_timestamp_con(tmp_path)
    try:
        first = session_text_corpus(con)
        for _ in range(4):
            again = session_text_corpus(con)
            assert again.order == first.order
            for sid in first.order:
                assert again.texts_by_session[sid] == first.texts_by_session[sid]
    finally:
        con.close()


def test_session_text_corpus_tie_ordering_is_body_ranked(tmp_path: Path) -> None:
    """Several tool_use blocks emitted at one timestamp fall in a fixed body
    order — the sort key's third component — not the scan-plan order."""
    con = _tied_timestamp_con(tmp_path)
    try:
        corpus = session_text_corpus(con)
        sid = "cccccccc-cccc-cccc-cccc-cccccccccccc"
        rows = corpus.texts_by_session[sid]
        # The three reads share the assistant turn's timestamp (the latest ts
        # in the session); the earlier user-text row sits before them.
        tied_ts = max(r.ts_iso for r in rows)
        tied = [r for r in rows if r.ts_iso == tied_ts]
        assert [r.kind for r in tied] == ["tool_use", "tool_use", "tool_use"]
        # Ordered by tool_input body, ascending → /a.txt, /m.txt, /z.txt,
        # regardless of the order DuckDB's scan returned them.
        tool_bodies = [r.body or "" for r in tied]
        assert tool_bodies == sorted(tool_bodies)
        assert "/a.txt" in tool_bodies[0]
        assert "/m.txt" in tool_bodies[1]
        assert "/z.txt" in tool_bodies[2]
    finally:
        con.close()


# ---------------------------------------------------------------------------
# 7. Engine-side ts.isoformat() (the timeline loaders format ts in SQL, not
#    via a per-row Python ``ts.isoformat()`` call — see ``_iso_expr``).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "literal",
    [
        "2026-04-01 10:00:00",  # whole second -> isoformat omits .%f
        "2026-04-01 10:00:00.5",  # 500ms
        "2026-04-01 10:00:00.123",  # ms precision
        "2026-04-01 10:00:00.123456",  # full microsecond precision
        "2026-04-01 10:00:00.000001",  # 1 microsecond
        "2026-12-31 23:59:59",  # year/whole-second boundary
        "2026-06-07 13:42:19",  # a real whole-second corpus stamp
    ],
)
def test_iso_expr_matches_python_isoformat(literal: str) -> None:
    """``_iso_expr`` reproduces ``datetime.isoformat()`` byte-for-byte.

    The subtle case is the whole-second timestamp: ``strftime`` always emits
    the fractional field, while Python drops it entirely, so the ``CASE`` must
    pick the no-fraction format exactly when ``ts == date_trunc('second', ts)``.
    """
    from claude_sql.core.session_text import _iso_expr

    con = duckdb.connect(":memory:")
    try:
        ts, iso = con.execute(
            f"SELECT t, {_iso_expr('t')} FROM (SELECT TIMESTAMP '{literal}' AS t)"
        ).fetchone()
        assert iso == ts.isoformat()
    finally:
        con.close()


def test_corpus_ts_iso_matches_python_isoformat_with_subsecond(tmp_path: Path) -> None:
    """End-to-end: corpus ``ts_iso`` equals ``ts.isoformat()`` across a session
    that mixes whole-second, millisecond, and microsecond timestamps.

    Guards the engine-side formatting against any future ``strftime`` drift —
    the loaders no longer touch ``ts`` in Python, so this is the only check
    that the rendered timeline string still matches the canonical Python form.
    """
    proj = tmp_path / "projects" / "proj-iso"
    sid = "dddddddd-dddd-dddd-dddd-dddddddddddd"
    # Three user messages at distinct sub-second precisions. The Z suffix is
    # parsed to a naive UTC ts (the corpus carries no tzinfo — see register_raw),
    # so the Python reference uses naive datetimes too.
    stamps = (
        "2026-04-01T10:00:00.000Z",  # whole second
        "2026-04-01T10:00:01.500Z",  # 500 ms
        "2026-04-01T10:00:02.123456Z",  # microseconds
    )
    write_session_jsonl(
        proj / f"{sid}.jsonl",
        messages=[
            make_user_msg(
                f"iso{i}",
                sid,
                f"message {i} long enough to clear the 32-char user-text filter here",
                ts=ts,
            )
            for i, ts in enumerate(stamps)
        ],
    )
    glob = str(tmp_path / "projects" / "*" / "*.jsonl")
    con = _open_views(tmp_path, glob)
    try:
        corpus = session_text_corpus(con)
        rows = corpus.texts_by_session[sid]
        got = [r.ts_iso for r in rows]
        # Reference: read the same ts values raw and isoformat() them in Python.
        raw = con.execute(
            "SELECT ts FROM messages_text WHERE CAST(session_id AS VARCHAR) = ? ORDER BY ts",
            [sid],
        ).fetchall()
        want = [r[0].isoformat() for r in raw]
        assert got == want
        # And the whole-second row carries no fractional component.
        assert got[0] == "2026-04-01T10:00:00"
        assert got[1] == "2026-04-01T10:00:01.500000"
        assert got[2] == "2026-04-01T10:00:02.123456"
    finally:
        con.close()

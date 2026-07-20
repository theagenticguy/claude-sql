"""Contract tests for the pure transcript domain module.

Covers:

* Domain purity — importing ``claude_sql.domain.transcript`` in a fresh
  interpreter must not pull in duckdb / pyarrow / polars / lancedb.
* ``render_turn_text`` collapse contract — role-rank ordering with uuid
  tiebreak, bare tool markers, per-turn + total caps, determinism.
* ``SessionTextCorpus.assemble`` byte-stability — the assembled transcript
  over a fixed fixture is byte-identical to the golden string the four LLM
  pipelines checkpoint on.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

from claude_sql.domain.transcript import (
    SessionTextCorpus,
    TranscriptRow,
    _TimelineRow,
    render_turn_text,
)


def _row(
    uuid: str,
    type_: str,
    ts: str,
    content: str | list[dict[str, object]],
    *,
    role: str | None = None,
) -> TranscriptRow:
    """Build a raw ``TranscriptRow`` with an inline message dict.

    ``role`` defaults to ``type_`` (the common case where envelope type and
    inner role agree); pass it explicitly to exercise the role-fidelity path.
    """
    message: dict[str, object] = {"role": role if role is not None else type_, "content": content}
    return TranscriptRow(uuid=uuid, type=type_, timestamp=ts, message=message)


@dataclass(slots=True)
class _CapsStub:
    """Minimal stand-in for ``Settings`` — only the two caps ``assemble`` reads."""

    session_text_total_max_chars: int = 800_000
    session_text_tool_result_max_chars: int = 50_000


# ---------------------------------------------------------------------------
# 1. Domain purity — no heavy imports
# ---------------------------------------------------------------------------


def test_domain_module_imports_nothing_heavy() -> None:
    """A fresh interpreter importing the domain module loads no adapter deps.

    duckdb / pyarrow / polars / lancedb are infrastructure concerns. The domain
    hexagon must stay dependency-free so it can be reasoned about (and unit
    tested) without a database or an Arrow runtime.
    """
    code = (
        "import sys\n"
        "import claude_sql.domain.transcript  # noqa: F401\n"
        "heavy = {'duckdb', 'pyarrow', 'polars', 'lancedb'}\n"
        "leaked = sorted(heavy & set(sys.modules))\n"
        "print(','.join(leaked))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    leaked = result.stdout.strip()
    assert leaked == "", f"domain.transcript leaked heavy imports: {leaked}"


# ---------------------------------------------------------------------------
# 2. render_turn_text — ordering (consumer ``_collapse`` contract)
# ---------------------------------------------------------------------------


def test_render_turn_text_kind_rank_ordering() -> None:
    """Same-timestamp rows sort by envelope-type rank: user<assistant<tool<system."""
    ts = "2026-04-01T10:00:00.000Z"
    rows = [
        _row("d", "system", ts, "sys"),
        _row("c", "tool", ts, "tl"),
        _row("b", "assistant", ts, "as"),
        _row("a", "user", ts, "us"),
    ]
    out = render_turn_text(rows)
    assert out == (f"[user {ts}] us\n[assistant {ts}] as\n[tool {ts}] tl\n[system {ts}] sys")


def test_render_turn_text_uuid_tiebreak() -> None:
    """Rows sharing ts AND type break the tie by uuid, ascending."""
    ts = "2026-04-01T10:00:00.000Z"
    rows = [
        _row("uuid-3", "user", ts, "third"),
        _row("uuid-1", "user", ts, "first"),
        _row("uuid-2", "user", ts, "second"),
    ]
    out = render_turn_text(rows)
    assert out == (f"[user {ts}] first\n[user {ts}] second\n[user {ts}] third")


def test_render_turn_text_timestamp_is_primary_key() -> None:
    """Timestamp dominates kind rank — an earlier system row precedes a later user row."""
    rows = [
        _row("b", "user", "2026-04-01T10:00:05.000Z", "late-user"),
        _row("a", "system", "2026-04-01T10:00:01.000Z", "early-system"),
    ]
    out = render_turn_text(rows)
    assert out == (
        "[system 2026-04-01T10:00:01.000Z] early-system\n[user 2026-04-01T10:00:05.000Z] late-user"
    )


# ---------------------------------------------------------------------------
# 3. render_turn_text — inline marker folding (DELTA-1) + role fidelity (DELTA-6)
# ---------------------------------------------------------------------------


def test_render_turn_text_folds_tool_markers_inline() -> None:
    """A message's tool_use / tool_result blocks fold INLINE into its own line.

    The consumer ``_collapse`` emits one line per message: the assistant's text body and
    its ``[tool_use:Read]`` marker share a line; the user tool_result carrier is
    its own ``[user ts] [tool_result]`` line — no standalone marker lines, no
    payload.
    """
    rows = [
        _row("a", "user", "2026-04-01T10:00:00.000Z", "hi"),
        _row(
            "b",
            "assistant",
            "2026-04-01T10:00:01.000Z",
            [
                {"type": "text", "text": "let me look"},
                {"type": "tool_use", "name": "Read", "input": {"path": "/etc/hosts"}},
            ],
        ),
        _row(
            "c",
            "user",
            "2026-04-01T10:00:02.000Z",
            [{"type": "tool_result", "tool_use_id": "t1", "content": "must not appear"}],
        ),
    ]
    out = render_turn_text(rows)
    assert out == (
        "[user 2026-04-01T10:00:00.000Z] hi\n"
        "[assistant 2026-04-01T10:00:01.000Z] let me look [tool_use:Read]\n"
        "[user 2026-04-01T10:00:02.000Z] [tool_result]"
    )
    assert "/etc/hosts" not in out
    assert "must not appear" not in out


def test_render_turn_text_role_prefers_inner_message_role() -> None:
    """DELTA-6: role prefers inner message.role, then envelope type, then unknown."""
    # Inner role wins over envelope type.
    inner = TranscriptRow(
        uuid="a",
        type="assistant",
        timestamp="2026-04-01T10:00:00.000Z",
        message={"role": "system", "content": "policy"},
    )
    assert render_turn_text([inner]) == "[system 2026-04-01T10:00:00.000Z] policy"
    # No inner role → fall back to envelope type.
    envelope = TranscriptRow(
        uuid="b",
        type="user",
        timestamp="2026-04-01T10:00:01.000Z",
        message={"content": "hello there general"},
    )
    assert render_turn_text([envelope]) == "[user 2026-04-01T10:00:01.000Z] hello there general"


def test_render_turn_text_tool_use_without_name_renders_empty_name() -> None:
    """A tool_use block with no ``name`` renders ``[tool_use:]`` (collapse parity)."""
    row = _row(
        "a",
        "assistant",
        "2026-04-01T10:00:00.000Z",
        [{"type": "text", "text": "go"}, {"type": "tool_use", "input": {}}],
    )
    assert render_turn_text([row]) == "[assistant 2026-04-01T10:00:00.000Z] go [tool_use:]"


def test_render_turn_text_drops_empty_body_rows() -> None:
    """A message whose collapsed body is empty (bare ack) is dropped entirely."""
    rows = [
        _row("a", "user", "2026-04-01T10:00:00.000Z", "real turn here"),
        _row("b", "assistant", "2026-04-01T10:00:01.000Z", []),  # no blocks → empty body
    ]
    assert render_turn_text(rows) == "[user 2026-04-01T10:00:00.000Z] real turn here"


# ---------------------------------------------------------------------------
# 4. render_turn_text — caps (DELTA-4: per-turn " …", total hard-slice)
# ---------------------------------------------------------------------------


def test_render_turn_text_per_turn_cap_appends_ellipsis() -> None:
    """A text body over ``per_turn_chars`` is clipped with a trailing `` …``."""
    rows = [_row("a", "user", "2026-04-01T10:00:00.000Z", "x" * 50)]
    out = render_turn_text(rows, per_turn_chars=10)
    assert out == "[user 2026-04-01T10:00:00.000Z] xxxxxxxxxx …"


def test_render_turn_text_total_cap_hard_slices_by_default() -> None:
    """Default total-cap semantics: hard-slice at ``total_chars``, NO notice."""
    rows = [
        _row("a", "user", "2026-04-01T10:00:00.000Z", "a" * 40),
        _row("b", "user", "2026-04-01T10:00:01.000Z", "b" * 40),
    ]
    out = render_turn_text(rows, total_chars=50)
    assert len(out) == 50
    assert "truncated" not in out
    # The slice keeps the head of the rendered transcript verbatim.
    assert out == ("[user 2026-04-01T10:00:00.000Z] " + "a" * 40)[:50]


def test_render_turn_text_total_cap_notice_is_opt_in() -> None:
    """``truncation_notice=True`` restores the older claude-sql notice style."""
    rows = [
        _row("a", "user", "2026-04-01T10:00:00.000Z", "a" * 40),
        _row("b", "user", "2026-04-01T10:00:01.000Z", "b" * 40),
    ]
    out = render_turn_text(rows, total_chars=60, truncation_notice=True)
    assert out.endswith("…(transcript truncated at 60 chars)")
    assert len(out) <= 60


def test_render_turn_text_no_truncation_when_under_cap() -> None:
    """A transcript comfortably under the total cap is untouched."""
    rows = [_row("a", "user", "2026-04-01T10:00:00.000Z", "short enough turn")]
    out = render_turn_text(rows)
    assert "truncated" not in out
    assert out == "[user 2026-04-01T10:00:00.000Z] short enough turn"


# ---------------------------------------------------------------------------
# 5. render_turn_text — determinism
# ---------------------------------------------------------------------------


def test_render_turn_text_deterministic() -> None:
    """Same input → byte-identical output across calls, regardless of input order."""
    ts = "2026-04-01T10:00:00.000Z"
    rows_a = [
        _row("2", "assistant", ts, "a"),
        _row("1", "user", ts, "u"),
        _row(
            "3",
            "assistant",
            "2026-04-01T10:00:01.000Z",
            [{"type": "text", "text": "run"}, {"type": "tool_use", "name": "Bash"}],
        ),
    ]
    rows_b = list(reversed(rows_a))
    assert render_turn_text(rows_a) == render_turn_text(rows_b)
    assert render_turn_text(rows_a) == render_turn_text(rows_a)


# ---------------------------------------------------------------------------
# 6. SessionTextCorpus.assemble — byte-stability regression
# ---------------------------------------------------------------------------


def _fixture_rows() -> list[_TimelineRow]:
    """A fixed timeline mixing text, tool_use, and tool_result across timestamps."""
    return [
        _TimelineRow(
            ts_iso="2026-04-01T10:00:00",
            role="user",
            kind="text",
            body="please read the file",
            aux=None,
            uuid="u-1",
        ),
        _TimelineRow(
            ts_iso="2026-04-01T10:00:01",
            role="assistant",
            kind="text",
            body="on it",
            aux=None,
            uuid="a-1",
        ),
        _TimelineRow(
            ts_iso="2026-04-01T10:00:02",
            role="tool",
            kind="tool_use",
            body='{"file_path": "/etc/hosts"}',
            aux="Read",
        ),
        _TimelineRow(
            ts_iso="2026-04-01T10:00:03",
            role="tool",
            kind="tool_result",
            body="127.0.0.1 localhost",
            aux="toolu_abc",
        ),
    ]


def test_assemble_byte_identical_default_header() -> None:
    """assemble() output over the fixture matches the frozen golden string.

    This is the hard gate: the four LLM pipelines checkpoint on this exact
    byte layout. Any drift in the header format, marker shape, or ordering is
    a regression.
    """
    corpus = SessionTextCorpus(texts_by_session={"s": _fixture_rows()}, order=["s"])
    out = corpus.assemble("s", settings=_CapsStub())
    expected = (
        "[user 2026-04-01T10:00:00] please read the file\n"
        "[assistant 2026-04-01T10:00:01] on it\n"
        '[tool_use:Read 2026-04-01T10:00:02] {"file_path": "/etc/hosts"}\n'
        "[tool_result toolu_abc 2026-04-01T10:00:03] 127.0.0.1 localhost"
    )
    assert out == expected


def test_assemble_byte_identical_with_uuids() -> None:
    """include_uuids=True stamps ``[uuid=<id> role ts]`` on text turns only."""
    corpus = SessionTextCorpus(texts_by_session={"s": _fixture_rows()}, order=["s"])
    out = corpus.assemble("s", settings=_CapsStub(), include_uuids=True)
    expected = (
        "[uuid=u-1 user 2026-04-01T10:00:00] please read the file\n"
        "[uuid=a-1 assistant 2026-04-01T10:00:01] on it\n"
        '[tool_use:Read 2026-04-01T10:00:02] {"file_path": "/etc/hosts"}\n'
        "[tool_result toolu_abc 2026-04-01T10:00:03] 127.0.0.1 localhost"
    )
    assert out == expected


def test_assemble_total_cap_notice() -> None:
    """The total-length cap emits the frozen ``…(session truncated…)`` notice."""
    rows = _fixture_rows()
    corpus = SessionTextCorpus(texts_by_session={"s": rows}, order=["s"])
    # A cap small enough that only the first line fits.
    out = corpus.assemble("s", settings=_CapsStub(session_text_total_max_chars=30))
    lines = out.split("\n")
    assert lines[-1] == f"…(session truncated at 30 chars, {len(rows)} events total)"

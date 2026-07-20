"""Property tests for the pure transcript domain (``domain/transcript.py``).

Invariant clusters:

* ``render_turn_text`` determinism — the render is a pure function of the row
  *set*: shuffling the input list never changes the bytes (the ``(ts, role,
  uuid)`` sort key is total over the generated rows).
* Per-turn + total caps — no text body exceeds ``per_turn_chars`` (plus the
  fixed truncation suffix); the whole render never exceeds ``total_chars`` plus
  one truncation-notice length.
* Marker invariants — every ``tool_use`` row renders ``[tool_use:{name}]``
  exactly; ``tool_result`` renders the bare ``[tool_result]`` marker.
* Role-rank ordering — same-timestamp text rows come out ordered
  ``user < assistant < tool < system`` with uuid as the final tiebreak.
* ``SessionTextCorpus.assemble`` determinism — the same corpus renders the same
  string every call and across identical corpus objects.
"""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import given, settings, strategies as st

from claude_sql.domain.transcript import (
    _COLLAPSE_KIND_RANK,
    SessionTextCorpus,
    TranscriptRow,
    _TimelineRow,
    render_turn_text,
)
from property.strategies import (
    SAFE_TEXT_NONBLANK,
    TIMESTAMPS,
    TYPES,
    same_ts_transcript_rows,
    timeline_rows,
    tool_use_transcript_rows,
    transcript_rows,
)

_PER_TURN_SUFFIX = " …"


def _text_row(uuid: str, type_: str, ts: str, text: str) -> TranscriptRow:
    """Single-text-block raw row (role == envelope type)."""
    return TranscriptRow(
        uuid=uuid,
        type=type_,
        timestamp=ts,
        message={"role": type_, "content": [{"type": "text", "text": text}]},
    )


# ---------------------------------------------------------------------------
# render_turn_text — determinism under input reordering
# ---------------------------------------------------------------------------


@given(data=st.data(), rows=transcript_rows())
@settings(max_examples=100)
def test_render_turn_text_shuffle_invariant(data: st.DataObject, rows: list[TranscriptRow]) -> None:
    """Shuffling the input list leaves the rendered bytes identical.

    The collapse key ``(ts, kind_rank, uuid)`` is total over the generated rows
    (uuids are globally unique), so the output depends only on the set of rows.
    """
    shuffled = data.draw(st.permutations(rows))
    assert render_turn_text(list(shuffled)) == render_turn_text(rows)


@given(rows=transcript_rows())
@settings(max_examples=100)
def test_render_turn_text_idempotent(rows: list[TranscriptRow]) -> None:
    """Calling twice on the same rows yields byte-identical output."""
    assert render_turn_text(rows) == render_turn_text(rows)


# ---------------------------------------------------------------------------
# render_turn_text — caps
# ---------------------------------------------------------------------------


@given(
    type_=TYPES,
    ts=TIMESTAMPS,
    body=st.text(alphabet="abcde", min_size=1, max_size=300),
    per_turn=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=100)
def test_render_turn_text_per_turn_cap(type_: str, ts: str, body: str, per_turn: int) -> None:
    """A single text turn's rendered body never exceeds per_turn_chars + suffix.

    The body alphabet has no whitespace, so ``str.strip()`` in the collapse is a
    no-op and the rendered body is exactly ``body`` (or its clipped form).
    """
    row = _text_row("0", type_, ts, body)
    out = render_turn_text([row], per_turn_chars=per_turn, total_chars=1_000_000)
    header = f"[{type_} {ts}] "
    assert out.startswith(header)
    rendered_body = out[len(header) :]
    assert len(rendered_body) <= per_turn + len(_PER_TURN_SUFFIX)
    if len(body) > per_turn:
        assert rendered_body == body[:per_turn] + _PER_TURN_SUFFIX
    else:
        assert rendered_body == body


@given(rows=transcript_rows(min_size=1), total=st.integers(min_value=1, max_value=400))
@settings(max_examples=150)
def test_render_turn_text_total_cap_hard_slice(rows: list[TranscriptRow], total: int) -> None:
    """Default total cap hard-slices at exactly total_chars — never more, no notice."""
    out = render_turn_text(rows, per_turn_chars=50, total_chars=total)
    assert len(out) <= total
    assert "truncated" not in out


@given(rows=transcript_rows(min_size=1), total=st.integers(min_value=64, max_value=400))
@settings(max_examples=100)
def test_render_turn_text_total_cap_notice_opt_in(rows: list[TranscriptRow], total: int) -> None:
    """With the opt-in notice, the render never exceeds total_chars.

    ``total`` is floored at 64 so the fixed truncation-notice line always fits —
    a cap smaller than the notice itself is a degenerate config the non-default
    notice mode does not promise to honor (the notice is a whole line).
    """
    out = render_turn_text(rows, per_turn_chars=50, total_chars=total, truncation_notice=True)
    assert len(out) <= total


# ---------------------------------------------------------------------------
# render_turn_text — marker invariants
# ---------------------------------------------------------------------------


@given(rows=tool_use_transcript_rows())
@settings(max_examples=100)
def test_tool_use_marker_folds_inline(rows: list[TranscriptRow]) -> None:
    """Each assistant row folds its ``[tool_use:{name}]`` inline after its text."""
    out = render_turn_text(rows, total_chars=1_000_000)
    lines = out.split("\n")
    expected = [
        f"[assistant {row.timestamp}] step [tool_use:{row.message['content'][1].get('name', '')}]"
        for row in rows
    ]
    # Distinct ascending timestamps → one line per row in row order.
    assert lines == expected


@given(
    ts=TIMESTAMPS,
    tuid=st.one_of(st.none(), st.text(alphabet="tool-id_", min_size=1, max_size=20)),
    payload=SAFE_TEXT_NONBLANK,
)
@settings(max_examples=50)
def test_tool_result_marker_is_bare(ts: str, tuid: str | None, payload: str) -> None:
    """A user tool_result carrier renders the bare ``[tool_result]`` — no payload."""
    block: dict[str, object] = {"type": "tool_result", "content": payload}
    if tuid is not None:
        block["tool_use_id"] = tuid
    row = TranscriptRow(
        uuid="0", type="user", timestamp=ts, message={"role": "user", "content": [block]}
    )
    assert render_turn_text([row], total_chars=1_000_000) == f"[user {ts}] [tool_result]"


# ---------------------------------------------------------------------------
# render_turn_text — kind-rank ordering for ts-ties
# ---------------------------------------------------------------------------


@given(rows=same_ts_transcript_rows())
@settings(max_examples=100)
def test_kind_rank_ordering_with_uuid_tiebreak(rows: list[TranscriptRow]) -> None:
    """Same-ts rows order by envelope-type rank then uuid (user<assistant<tool<system)."""
    out = render_turn_text(rows, total_chars=1_000_000)
    expected_order = sorted(rows, key=lambda r: (_COLLAPSE_KIND_RANK[r.type or ""], r.uuid or ""))
    expected_lines = [
        f"[{r.type} {r.timestamp}] {r.message['content'][0]['text']}" for r in expected_order
    ]
    assert out.split("\n") == expected_lines


# ---------------------------------------------------------------------------
# SessionTextCorpus.assemble — determinism
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Caps:
    """Minimal TranscriptCaps stand-in — the two int caps assemble reads."""

    session_text_total_max_chars: int = 800_000
    session_text_tool_result_max_chars: int = 50_000


@given(rows=timeline_rows(), total=st.integers(min_value=1, max_value=100_000))
@settings(max_examples=100)
def test_assemble_deterministic(rows: list[_TimelineRow], total: int) -> None:
    """The same corpus renders the same string every call and across copies."""
    caps = _Caps(session_text_total_max_chars=total)
    corpus_a = SessionTextCorpus(texts_by_session={"s": rows}, order=["s"])
    corpus_b = SessionTextCorpus(texts_by_session={"s": list(rows)}, order=["s"])
    out1 = corpus_a.assemble("s", settings=caps)
    out2 = corpus_a.assemble("s", settings=caps)
    out3 = corpus_b.assemble("s", settings=caps)
    assert out1 == out2 == out3

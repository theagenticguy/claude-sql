"""Pure transcript domain types: timeline rows, ordering, and text rendering.

This is the domain half of the ``TranscriptReaderPort`` seam. It holds the
dependency-free collapse/format logic that turns a session's chronological
timeline into a single transcript string. The DuckDB loaders that materialize
:class:`SessionTextCorpus` from the ``read_json`` glob live in the
infrastructure adapter (``infrastructure.session_text_loader``); nothing here
imports duckdb, pyarrow, polars, or any adapter.

Two rendering contracts live side by side:

:meth:`SessionTextCorpus.assemble`
    The historic per-session transcript used by the four LLM pipelines
    (classify / trajectory / conflicts / friction). Its output is
    **byte-stable**: those pipelines checkpoint on it, so the string literals
    and the tie-break ordering (``_timeline_sort_key``, kind-rank) must not
    drift.

:func:`render_turn_text`
    The consumer collapse contract — a pure function over a list of rows
    with a different, role-rank ordering and bare tool markers. Used by the
    read-turn seam where caps arrive as keyword arguments rather than from
    ``Settings``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    from claude_sql.domain.config import TranscriptCaps


#: Chronological precedence for events sharing a timestamp. Append order plus a
#: stable sort historically put text before tool_use before tool_result at a
#: tie; we encode that explicitly so the sort key fully determines order.
_KIND_RANK = {"text": 0, "tool_use": 1, "tool_result": 2}

#: Envelope-type precedence for :func:`render_turn_text`'s collapse ordering.
#: This is the deterministic within-timestamp tie-break the consumer's collapse
#: routine uses (``user < assistant < tool < system``), keyed on the
#: message *envelope* ``type`` — NOT the block-level kind-rank
#: :meth:`SessionTextCorpus.assemble` uses. Unknown types sort last but stably
#: (uuid is the final key), so ordering never depends on file read order.
_COLLAPSE_KIND_RANK = {"user": 0, "assistant": 1, "tool": 2, "system": 3}
_COLLAPSE_KIND_RANK_DEFAULT = 99


def _timeline_sort_key(row: _TimelineRow) -> tuple[str, int, str, str]:
    """Total order for a session's timeline rows.

    ``ts_iso`` is the primary key. The remaining components only break ties
    *within* one timestamp: ``kind`` preserves the text→tool_use→tool_result
    precedence, then ``body``/``aux`` canonicalize the order of multiple events
    emitted in the same millisecond (e.g. several ``tool_use`` blocks in one
    assistant turn). Without these, the row order at a tie was inherited from
    the DuckDB scan plan and varied run-to-run.
    """
    return (row.ts_iso, _KIND_RANK.get(row.kind, 9), row.body or "", row.aux or "")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _tool_input_preview(tool_input_json: str | None, max_chars: int = 400) -> str:
    """Truncate a ``tool_input`` JSON blob to the first ``max_chars``."""
    if not tool_input_json:
        return ""
    s = str(tool_input_json)
    return s if len(s) <= max_chars else s[:max_chars] + "…(truncated)"


def _tool_result_preview(content_json: str | None, max_chars: int) -> str:
    """Truncate a ``tool_result`` content blob with a "bytes dropped" footer."""
    if not content_json:
        return ""
    s = str(content_json)
    if len(s) <= max_chars:
        return s
    dropped = len(s) - max_chars
    return s[:max_chars] + "\n…(truncated, " + str(dropped) + " chars dropped)"


# ---------------------------------------------------------------------------
# Timeline row + corpus
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _TimelineRow:
    """One event in a session's chronological timeline."""

    ts_iso: str
    role: str
    kind: str  # "text" | "tool_use" | "tool_result"
    body: str | None
    aux: str | None  # tool_name for tool_use, tool_use_id for tool_result
    # The message uuid, populated only on ``kind == "text"`` rows (the turn
    # uuids the conflicts pipeline needs to copy verbatim). ``None`` on
    # tool_use / tool_result rows, which are not addressable turns. Surfaced
    # in the transcript only when ``assemble(include_uuids=True)``.
    uuid: str | None = None


@dataclass(slots=True)
class SessionTextCorpus:
    """An in-memory corpus of per-session timelines, built with one glob scan.

    ``texts_by_session`` maps ``session_id`` → pre-sorted list of
    :class:`_TimelineRow`.  ``order`` is the newest-first session list that
    defines iteration order for downstream pipelines.
    """

    texts_by_session: dict[str, list[_TimelineRow]]
    order: list[str]

    def __len__(self) -> int:
        return len(self.order)

    def assemble(
        self, session_id: str, *, settings: TranscriptCaps, include_uuids: bool = False
    ) -> str:
        """Render one session as a single newline-separated transcript.

        Applies :attr:`TranscriptCaps.session_text_tool_result_max_chars` and
        the total-length cap :attr:`TranscriptCaps.session_text_total_max_chars`.
        The ``settings`` parameter is a :class:`~claude_sql.domain.config.TranscriptCaps`
        value-object — the pure two-int caps slice, not the god-``Settings``
        (the infrastructure ``iter_session_texts`` boundary projects
        ``settings.transcript_caps()`` before calling in). The keyword name is
        kept as ``settings`` for call-site + test stability.

        When ``include_uuids`` is True, each text turn's header carries its
        message uuid as ``[uuid=<id> role ts]`` so a classifier can copy the
        turn's natural key verbatim (the conflicts pipeline needs this — see
        issue #109). The default (False) keeps the historic ``[role ts]``
        header so the classify / trajectory / friction prompts stay
        byte-identical run-to-run.
        """
        rows = self.texts_by_session.get(session_id)
        if not rows:
            return ""

        cap = settings.session_text_total_max_chars
        tool_cap = settings.session_text_tool_result_max_chars
        lines: list[str] = []
        running = 0

        for row in rows:
            if row.body is None:
                continue
            if row.kind == "text":
                if include_uuids and row.uuid:
                    line = f"[uuid={row.uuid} {row.role} {row.ts_iso}] {row.body}"
                else:
                    line = f"[{row.role} {row.ts_iso}] {row.body}"
            elif row.kind == "tool_use":
                name = row.aux or "tool"
                line = f"[tool_use:{name} {row.ts_iso}] {_tool_input_preview(row.body)}"
            else:  # tool_result
                tu_id = row.aux or "?"
                line = (
                    f"[tool_result {tu_id} {row.ts_iso}] {_tool_result_preview(row.body, tool_cap)}"
                )

            if running + len(line) + 1 > cap:
                lines.append(f"…(session truncated at {cap} chars, {len(rows)} events total)")
                break
            lines.append(line)
            running += len(line) + 1

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Collapse renderer
# ---------------------------------------------------------------------------
#
# ``render_turn_text`` is a faithful port of the downstream consumer's collapse
# routine: it folds a session's raw message-envelope rows into ONE line per
# message. Unlike :meth:`SessionTextCorpus.assemble` — which fans a message out
# into per-block ``text`` / ``tool_use`` / ``tool_result`` timeline rows — the
# collapse groups by the message envelope and appends the
# ``[tool_use:name]`` / ``[tool_result]`` markers inline after that message's
# text body, so ``[assistant ts] Let me list. [tool_use:Bash]`` and
# ``[user ts] [tool_result]`` render as single lines. This is the drop-in
# retrieval contract downstream consumers depend on; the acceptance proof is
# the ported-fixture parity test.


@dataclass(slots=True)
class TranscriptRow:
    """One raw message-envelope row, the input to :func:`render_turn_text`.

    Mirrors the consumer's projected ``read_json`` row: the four fields the
    collapse step reads. ``message`` is either the decoded message dict or the
    JSON text DuckDB hands back (``read_json(columns={message: 'JSON'})``);
    :func:`render_turn_text` decodes it. Deliberately NOT the per-block
    :class:`_TimelineRow`: the collapse is message-grained.
    """

    uuid: str | None
    type: str | None
    timestamp: str | None
    message: Any  # dict | JSON-text str | None


def _parse_message(value: Any) -> dict[str, Any] | None:
    """Decode a ``message`` field to a dict (DuckDB hands JSON back as text)."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed: Any = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _collapse_role_of(row: TranscriptRow) -> str:
    """Prefer the inner ``message.role``; fall back to the envelope ``type``.

    Makes ``system`` (and any future role) reachable straight from the source
    row, and keeps ``tool_result`` envelopes on their real message role (``user``
    for the tool_result carrier). ``unknown`` only when both are absent.
    """
    message = _parse_message(row.message)
    if message is not None:
        role = message.get("role")
        if isinstance(role, str) and role:
            return role
    return str(row.type or "unknown")


def _collapse_kind_rank(row: TranscriptRow) -> int:
    """Within-timestamp ordering key from the envelope ``type``."""
    return _COLLAPSE_KIND_RANK.get(str(row.type or ""), _COLLAPSE_KIND_RANK_DEFAULT)


def _collapse_row_body(row: TranscriptRow, *, per_turn_chars: int) -> str:
    """Collapse one message's ``content`` to a single body string.

    ``message.content`` is either a plain string (simple user text) or a list of
    content blocks. Text blocks contribute their (stripped) text; a ``tool_use``
    block becomes ``[tool_use:<name>]`` and a ``tool_result`` block
    ``[tool_result]`` — inline markers, so tool activity shows without the (often
    huge) payloads. The body is truncated to ``per_turn_chars`` with a trailing
    `` …`` marker (collapse parity — note the leading space).
    """
    message = _parse_message(row.message)
    if message is None:
        return ""
    content = message.get("content")

    if isinstance(content, str):
        body = content.strip()
    elif isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                txt = block.get("text")
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt.strip())
            elif btype == "tool_use":
                parts.append(f"[tool_use:{block.get('name', '')}]")
            elif btype == "tool_result":
                parts.append("[tool_result]")
        body = " ".join(parts)
    else:
        body = ""

    if len(body) > per_turn_chars:
        body = body[:per_turn_chars] + " …"
    return body


def render_turn_text(
    rows: Iterable[TranscriptRow],
    *,
    per_turn_chars: int = 8_000,
    total_chars: int = 200_000,
    truncation_notice: bool = False,
) -> str:
    """Collapse raw message rows into the consumer's transcript text.

    Faithful port of the consumer's collapse routine. ONE line per message:

    * Line shape is ``[{role} {ts}] {body}``. ``role`` prefers the inner
      ``message.role`` and falls back to the envelope ``type`` (so ``system`` is
      reachable and a ``tool_result`` carrier keeps its ``user`` role).
    * ``ts`` is the RAW timestamp string, verbatim (incl. ``.000`` / ``Z``) — no
      TIMESTAMP round-trip.
    * A message's ``tool_use`` / ``tool_result`` blocks fold INLINE into that
      message's body as ``[tool_use:{name}]`` / ``[tool_result]`` markers, after
      any text.
    * Ordering is ``(ts, kind_rank, uuid)`` — envelope-type rank
      (``user < assistant < tool < system``) with the message uuid as the final
      tiebreak. ISO-8601 timestamps sort chronologically as strings.
    * Rows whose collapsed body is empty (a bare tool ack) are dropped.
    * Per-turn cap appends `` …`` (collapse parity). The whole transcript is
      hard-sliced at ``total_chars`` with **no** trailing notice by default;
      pass ``truncation_notice=True`` for the older claude-sql notice style.

    Pure and deterministic: the same input always renders the same bytes. Caps
    arrive as keyword arguments — there is no ``Settings`` dependency.
    """
    turns: list[tuple[str, int, str, str]] = []
    for row in rows:
        body = _collapse_row_body(row, per_turn_chars=per_turn_chars)
        if not body:
            continue
        role = _collapse_role_of(row)
        ts = str(row.timestamp or "")
        line = f"[{role} {ts}] {body}"
        turns.append((ts, _collapse_kind_rank(row), str(row.uuid or ""), line))

    turns.sort(key=lambda t: (t[0], t[1], t[2]))
    text = "\n".join(line for *_, line in turns)

    if len(text) > total_chars:
        if truncation_notice:
            notice = f"\n…(transcript truncated at {total_chars} chars)"
            text = text[: max(0, total_chars - len(notice))] + notice
        else:
            text = text[:total_chars]
    return text


__all__ = [
    "SessionTextCorpus",
    "TranscriptRow",
    "render_turn_text",
]

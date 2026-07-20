"""Pure trajectory math: sentiment/delta arithmetic, window chunking, XML rendering.

Lifted out of the windowed trajectory worker during the v2 hexagonal reshape
(MIGRATION Phase C). Everything here is dependency-free stdlib — no duckdb,
polars, pyarrow, boto3, or any adapter — so it belongs in the innermost hexagon.

The windowed pipeline (:mod:`claude_sql.application.use_cases.trajectory`)
composes these helpers with the DuckDB window loader, the LLM-analytics
provider, and the parquet cache. The Sonnet system prompt and the polars
parquet schema stay with the use-case (the prompt is cache-floor-pinned and the
schema is a polars type map, neither of which is domain-pure).

A "window" tuple is ``(session_id, prev_uuid, curr_uuid, prev_role, curr_role,
prev_text, curr_text)`` — one adjacent text-turn pair from a single session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Honest ceiling on windows per Sonnet request body. RFC §4.1 — empirically
#: the per-window output budget under thinking=disabled stays comfortably
#: below the 2048-token classify_max_tokens cap up to ~16 windows. Past that
#: the model truncates the array and we end up running the missing-windows
#: retry path on every chunk.
MAX_WINDOWS_PER_CHUNK: int = 16

#: Numeric encoding of the three-label sentiment for the ``delta`` column.
_SENTIMENT_VAL: dict[str, int] = {"negative": -1, "neutral": 0, "positive": 1}

#: The six transition_kind labels — pinned in code so the count never drifts
#: from the schema. Tested in ``test_transition_kind_enum_values_are_six``.
TRANSITION_KINDS: tuple[str, ...] = (
    "frustration_spike",
    "resolution",
    "reset",
    "drift",
    "clarification",
    "none",
)


# ---------------------------------------------------------------------------
# Window assembly
# ---------------------------------------------------------------------------


def _chunk_windows(
    windows: list[tuple[str, str | None, str, str | None, str, str | None, str]],
    *,
    chunk_size: int = MAX_WINDOWS_PER_CHUNK,
) -> list[list[tuple[str, str | None, str, str | None, str, str | None, str]]]:
    """Split ``windows`` (already filtered to one session) into anchor-sharing chunks.

    Chunks of size ``chunk_size`` overlap by one *anchor turn*: chunk N's
    last window's ``curr_uuid`` equals chunk N+1's first window's
    ``prev_uuid`` — the natural shape since adjacent windows already share
    that turn by construction. We do NOT duplicate the anchor row; we just
    let the natural overlap fall out of slicing on contiguous windows.

    Empty input returns an empty list.
    """
    if not windows:
        return []
    return [windows[i : i + chunk_size] for i in range(0, len(windows), chunk_size)]


def _format_chunk_xml(
    chunk: list[tuple[str, str | None, str, str | None, str, str | None, str]],
    *,
    max_text_chars: int = 2000,
) -> str:
    """Render one chunk of windows as XML for the user content block.

    Each window block has a <prev> and <curr> with the role + uuid as
    attributes and the text as the body. Body is clipped at
    ``max_text_chars`` per turn so a single jumbo prompt doesn't blow the
    request size. Truncation is footer-marked so the model knows the rest
    was elided.
    """
    out: list[str] = []
    for idx, (_sid, prev_uuid, curr_uuid, prev_role, curr_role, prev_text, curr_text) in enumerate(
        chunk
    ):
        prev_attr_uuid = "" if prev_uuid is None else prev_uuid
        prev_attr_role = "" if prev_role is None else prev_role
        prev_body = ""
        if prev_text is not None:
            prev_body = prev_text
            if len(prev_body) > max_text_chars:
                prev_body = prev_body[:max_text_chars] + "…(truncated)"
        curr_body = curr_text
        if len(curr_body) > max_text_chars:
            curr_body = curr_body[:max_text_chars] + "…(truncated)"
        out.append(
            f"<window idx={idx}>\n"
            f'<prev role="{_xml_attr(prev_attr_role)}" uuid="{_xml_attr(prev_attr_uuid)}">'
            f"{_xml_text(prev_body)}</prev>\n"
            f'<curr role="{_xml_attr(curr_role or "")}" uuid="{_xml_attr(curr_uuid)}">'
            f"{_xml_text(curr_body)}</curr>\n"
            f"</window>"
        )
    return "\n".join(out)


def _xml_attr(value: str) -> str:
    """Escape a string for use inside an XML attribute value."""
    return (
        value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _xml_text(value: str) -> str:
    """Escape a string for use inside an XML element body."""
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# Sentiment / delta arithmetic
# ---------------------------------------------------------------------------


def _delta_value(prev: str | None, curr: str | None) -> float | None:
    """Compute the numeric delta (curr - prev) for the parquet row."""
    if prev is None or curr is None:
        return None
    return float(_SENTIMENT_VAL.get(curr, 0) - _SENTIMENT_VAL.get(prev, 0))


def _placeholder_row(
    session_id: str,
    prev_uuid: str | None,
    curr_uuid: str,
    classified_at: datetime,
) -> dict[str, Any]:
    """Build a neutral placeholder row when the model refuses or persistently misses a window."""
    return {
        "session_id": session_id,
        "prev_uuid": prev_uuid,
        "curr_uuid": curr_uuid,
        "prev_sentiment": None if prev_uuid is None else "neutral",
        "curr_sentiment": "neutral",
        "delta": None if prev_uuid is None else 0.0,
        "is_transition": False,
        "transition_kind": "none",
        "confidence": 0.0,
        "classified_at": classified_at,
    }


def _build_row(
    session_id: str,
    win: dict[str, Any],
    classified_at: datetime,
) -> dict[str, Any]:
    """Build one parquet row from a returned :class:`TrajectoryWindow` dict."""
    prev_uuid = win.get("prev_uuid")
    curr_uuid = win.get("curr_uuid")
    prev_sentiment = win.get("prev_sentiment")
    curr_sentiment = win.get("curr_sentiment") or "neutral"
    transition_kind = win.get("transition_kind") or "none"
    if transition_kind not in TRANSITION_KINDS:
        transition_kind = "none"
    # Trust the model's emitted delta when it parses to a number; otherwise
    # recompute from the prev/curr labels. Either path keeps the column
    # well-typed (Float64) and the recompute path is the audit trail when
    # the model returns garbage.
    raw_delta = win.get("delta")
    delta_val: float | None
    if raw_delta is None or prev_sentiment is None:
        delta_val = _delta_value(prev_sentiment, curr_sentiment) if prev_sentiment else None
    else:
        try:
            delta_val = float(raw_delta)
        except (TypeError, ValueError):
            delta_val = _delta_value(prev_sentiment, curr_sentiment)
    return {
        "session_id": session_id,
        "prev_uuid": prev_uuid,
        "curr_uuid": curr_uuid,
        "prev_sentiment": prev_sentiment,
        "curr_sentiment": curr_sentiment,
        "delta": delta_val,
        "is_transition": bool(win.get("is_transition", False)),
        "transition_kind": transition_kind,
        "confidence": float(win.get("confidence", 0.0)),
        "classified_at": classified_at,
    }


def _missing_keys(
    chunk: list[tuple[str, str | None, str, str | None, str, str | None, str]],
    indexed: dict[tuple[str | None, str], dict[str, Any]],
) -> list[tuple[str, str | None, str, str | None, str, str | None, str]]:
    """Return the chunk rows whose (prev_uuid, curr_uuid) was NOT in the response."""
    out = []
    for row in chunk:
        prev_uuid, curr_uuid = row[1], row[2]
        if (prev_uuid, curr_uuid) not in indexed:
            out.append(row)
    return out

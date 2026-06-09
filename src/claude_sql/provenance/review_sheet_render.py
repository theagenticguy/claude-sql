"""Markdown renderer for the PR review sheet.

Pure formatting — no schema validation, no Bedrock, no I/O. Takes the
structured dict produced by :func:`review_sheet_worker.generate_review_sheet`
and returns a Markdown string per strategy-memo §Coherent Actions #2.

The renderer is split into a separate module from the worker so the CLI
can decide whether to emit JSON (off-TTY default) or Markdown (TTY
default) without dragging Bedrock imports through the JSON path.
"""

from __future__ import annotations

from typing import Any


def _short_sha(commit_sha: str, *, length: int = 12) -> str:
    """Trim a commit SHA to the conventional review-sheet length."""
    return commit_sha[:length] if len(commit_sha) > length else commit_sha


def _short_digest(digest: str, *, length: int = 16) -> str:
    """Trim ``sha256:<hex>`` for header display.

    Preserves the ``sha256:`` prefix and the leading hex chars so the
    user can still spot-check the digest against the binding output;
    the rest is noise in a header line.
    """
    if ":" not in digest:
        return digest[:length]
    prefix, rest = digest.split(":", 1)
    return f"{prefix}:{rest[:length]}"


def _format_corrections(corrections: list[dict[str, Any]]) -> list[str]:
    """Render the corrections list. ``_None._`` placeholder if empty."""
    if not corrections:
        return ["_None._"]
    out: list[str] = []
    for entry in corrections:
        if not isinstance(entry, dict):
            continue
        what = entry.get("what_agent_did", "").strip()
        correction = entry.get("correction", "").strip()
        if what and correction:
            out.append(f"- *{what}* → {correction}")
        elif what:
            out.append(f"- *{what}*")
        elif correction:
            out.append(f"- {correction}")
    return out or ["_None._"]


def _format_inline_code_list(values: list[str]) -> str:
    """Render a list of tool names as backtick-wrapped inline code."""
    if not values:
        return "_None._"
    return ", ".join(f"`{v}`" for v in values if v)


def _format_refusal_lines(values: list[str]) -> list[str]:
    """Render ``tools_refused`` as one bullet per entry, "_None._" if empty."""
    if not values:
        return ["_None._"]
    return [f"- `{entry}`" for entry in values if entry]


def render_markdown(sheet: dict[str, Any], metadata: dict[str, Any]) -> str:
    """Render a PR review sheet dict into the canonical Markdown shape.

    Parameters
    ----------
    sheet
        The structured review-sheet dict — schema-shaped per
        :class:`claude_sql.schemas.PRReviewSheet`. Either the ``sheet``
        sub-key from :func:`generate_review_sheet`'s success return or
        an arbitrary dict in the same shape.
    metadata
        ``{commit_sha, transcript_uri, transcript_digest, model_id,
        captured_at}`` — populated by the worker on the success path.

    Returns
    -------
    str
        Markdown with a ``# PR Review Sheet — `<sha-short>`` header
        followed by the five canonical sections. The trailing newline
        keeps it pipe-friendly.
    """
    commit_sha = str(metadata.get("commit_sha", ""))
    transcript_uri = str(metadata.get("transcript_uri", ""))
    transcript_digest = str(metadata.get("transcript_digest", ""))
    runtime = str(metadata.get("model_id", ""))
    captured_at = str(metadata.get("captured_at", ""))

    human_intent = str(sheet.get("human_intent", "")).strip() or "_(missing)_"
    exploration = sheet.get("agent_exploration") or []
    corrections = sheet.get("corrections") or []
    tools_used = sheet.get("tools_used") or []
    tools_refused = sheet.get("tools_refused") or []
    diff_rationale = str(sheet.get("diff_rationale", "")).strip() or "_(missing)_"

    parts: list[str] = []
    parts.append(f"# PR Review Sheet — `{_short_sha(commit_sha)}`")
    parts.append("")
    parts.append(
        f"**Transcript:** `{transcript_uri}` (digest `{_short_digest(transcript_digest)}`)"
    )
    parts.append(f"**Agent runtime:** {runtime}")
    parts.append(f"**Generated:** {captured_at}")
    parts.append("")
    parts.append("## What the human asked for")
    parts.append(human_intent)
    parts.append("")
    parts.append("## What the agent explored")
    if exploration:
        parts.extend(f"- {str(bullet).strip()}" for bullet in exploration)
    else:
        parts.append("_None._")
    parts.append("")
    parts.append("## Corrections")
    parts.extend(_format_corrections(list(corrections)))
    parts.append("")
    parts.append("## Tools used")
    parts.append(_format_inline_code_list([str(v) for v in tools_used]))
    parts.append("")
    parts.append("## Tools refused")
    parts.extend(_format_refusal_lines([str(v) for v in tools_refused]))
    parts.append("")
    parts.append("## Why this diff")
    parts.append(diff_rationale)
    parts.append("")
    return "\n".join(parts)


def render_refusal_markdown(reason: str, metadata: dict[str, Any]) -> str:
    """Render the markdown footer for a refused review-sheet call.

    Keeps the same header so downstream consumers can still file the
    output by commit, then prints the canonical refusal note in place
    of the five sections.
    """
    commit_sha = str(metadata.get("commit_sha", ""))
    transcript_uri = str(metadata.get("transcript_uri", ""))
    transcript_digest = str(metadata.get("transcript_digest", ""))
    runtime = str(metadata.get("model_id", ""))
    captured_at = str(metadata.get("captured_at", ""))

    parts: list[str] = []
    parts.append(f"# PR Review Sheet — `{_short_sha(commit_sha)}`")
    parts.append("")
    parts.append(
        f"**Transcript:** `{transcript_uri}` (digest `{_short_digest(transcript_digest)}`)"
    )
    parts.append(f"**Agent runtime:** {runtime}")
    parts.append(f"**Generated:** {captured_at}")
    parts.append("")
    parts.append("_Review sheet refused; see metadata._")
    parts.append("")
    parts.append(f"> {reason.strip()}")
    parts.append("")
    return "\n".join(parts)


__all__ = [
    "render_markdown",
    "render_refusal_markdown",
]

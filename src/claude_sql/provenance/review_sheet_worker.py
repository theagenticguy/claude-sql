"""Single-shot PR review-sheet worker.

Compresses the bound transcript for one merged commit into a 1K-token PR
review sheet via Sonnet 4.6 with ``output_config.format`` structured
output. The motivating use case: a reviewer staring at a 400-line PR diff
wants a "what was the agent actually trying to do, and what did it
verify" digest without scrolling 200 KB of JSONL.

Pipeline shape
--------------
1. Resolve ``commit_sha`` → ``transcript_uri`` via
   :func:`claude_sql.binding.resolve_commit_to_transcript`. The override
   parameter ``transcript_uri_override`` is the test / direct-invocation
   bypass that loads a JSONL by path without touching git.
2. Compress the JSONL into a flat session text. We avoid the full DuckDB
   ``SessionTextCorpus.assemble`` round-trip on the override path so the
   worker is invokable from tests without a populated database.
3. Build a system prompt with Anthropic XML tags
   (``<instructions>``, ``<context>``, ``<examples><example>``,
   ``<anti_patterns>``) that frames the task as "compress the bound
   transcript into a 1K-token PR review sheet."
4. Call ``llm_shared._invoke_classifier_sync`` with
   :data:`PR_REVIEW_SHEET_SCHEMA`. Adaptive thinking is on by default;
   ``no_thinking=True`` disables it.
5. Return ``{"sheet": <PRReviewSheet dict>, "metadata": {...}}`` on
   success, ``{"refused": True, "reason": ...}`` on
   :class:`BedrockRefusalError`.

Out of scope: caching (single-shot, no parquet), batched processing,
DuckDB views. The worker is a thin wrapper around the structured-output
call so it can be re-used by future review-fleet code without a rewrite.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

from loguru import logger

from claude_sql.core.llm_shared import (
    BedrockRefusalError,
    _build_bedrock_client,
    _invoke_classifier_sync,
)
from claude_sql.core.schemas import PR_REVIEW_SHEET_SCHEMA
from claude_sql.provenance.binding import resolve_commit_to_transcript

if TYPE_CHECKING:
    import duckdb

    from claude_sql.core.config import Settings


# ---------------------------------------------------------------------------
# System prompt — Anthropic XML canonical tags only.
# ---------------------------------------------------------------------------


REVIEW_SHEET_SYSTEM_PROMPT = """\
<instructions>
You compress one Claude Code coding session — bound to a merged commit —
into a structured PR review sheet. The user message contains the bound
JSONL transcript, already flattened to one event per line in chronological
order (user turns, assistant turns, tool calls, tool results).

Your job is to populate the schema with a faithful, factual summary in
under ~1K rendered tokens. Six fields, no surrounding prose, no markdown
fences.
</instructions>

<context>
How to read the transcript:

- Opening user-role messages state the human's intent. Goal restatements
  later in the session refine it.
- Tool calls (``[tool_use:<name>...]``) are the strongest evidence of
  what the agent actually explored — read past chitchat to the actions.
- Tool results (``[tool_result ...]``) confirm what landed; pay attention
  to errors, refusals, and "blocked" markers — those become
  ``tools_refused`` entries.
- Closing exchanges state whether the goal was met; the diff_rationale
  field captures the WHY of the merged change.

Field semantics:

- human_intent: 1-3 sentences in present tense, paraphrased from the
  human's opening turn(s). Not a literal quote; not the agent's
  restatement.
- agent_exploration: 3-8 short bullets, each a noun phrase or short
  clause. Drawn from tool calls, search queries, and file reads.
  Examples: "Read src/auth/middleware.py", "Searched for token rotator
  call sites", "Inspected failing test fixture".
- corrections: human-redirected agent actions only. Up to 5 entries,
  empty list if none. Skip surface-level acknowledgements ("ok", "thanks").
  Each entry pairs what the agent did with the human's correction.
- tools_used: deduplicated tool names (Read, Edit, Write, Bash, Grep,
  Glob, etc.). Order by first use.
- tools_refused: tool calls the agent declined or that hooks/permissions
  blocked. Format: "ToolName: brief reason". Empty list when nothing
  was blocked.
- diff_rationale: 2-4 sentences naming the files / modules touched and
  the user-facing change. The WHY, not a line-by-line summary.
</context>

<examples>
<example>
<input>A 30-minute session: user asks "fix the off-by-one in the
pagination cursor". Agent reads src/api/pagination.py, runs the failing
test, edits the cursor math, re-runs tests (green), commits.</input>
<output>human_intent="Fix the off-by-one bug in the pagination cursor.";
agent_exploration=["Read src/api/pagination.py", "Ran the failing
pagination test", "Inspected cursor advance math"]; corrections=[];
tools_used=["Read", "Bash", "Edit"]; tools_refused=[];
diff_rationale="Adjusted the cursor advance math in
src/api/pagination.py from len(rows) to len(rows)-1 so the next page
starts on the correct row. Test pagination_test.py::test_cursor_edge
covers the regression."</output>
</example>
<example>
<input>User asks for a refactor; agent starts rewriting from scratch;
user says "no, keep the existing module and only update the rotator
call". Agent pivots, edits the rotator, ships.</input>
<output>human_intent="Refactor the auth middleware to use the new
token rotator without rewriting the existing module.";
agent_exploration=["Read src/auth/middleware.py", "Surveyed token
rotator call sites", "Reviewed the existing rotator interface"];
corrections=[{what_agent_did="Started rewriting the auth middleware
from scratch.", correction="Asked to keep the existing middleware and
only update the rotator call."}]; tools_used=["Read", "Grep", "Edit"];
tools_refused=[]; diff_rationale="Updated the rotator invocation in
src/auth/middleware.py to call rotate_v2() instead of rotate_v1(),
preserving the surrounding control flow per the user's redirection."
</output>
</example>
</examples>

<anti_patterns>
- Do not invent files or tool calls that aren't in the transcript. The
  review sheet is forensic; fabrication corrupts the audit trail.
- Do not restate the agent's narration as exploration. "I will now
  read the file" is narration; the matching ``[tool_use:Read ...]``
  block is the exploration evidence.
- Do not pad ``corrections`` with surface acknowledgements ("ok",
  "thanks", "sounds good"). Empty list is correct when nothing
  substantive was redirected.
- Do not echo the diff line-by-line in ``diff_rationale``. Reviewers
  read the diff itself; the field is the WHY in 2-4 sentences.
- Do not include subagent or sidecar transcripts unless they appear
  in the bound JSONL — the session text we hand you is the ground
  truth.
- ``tools_refused`` is for declined / blocked calls only. A
  successfully-executed Bash command does not belong here.
</anti_patterns>
"""


# ---------------------------------------------------------------------------
# Transcript loading
# ---------------------------------------------------------------------------


_MAX_LINE_PREVIEW: int = 800
"""Per-event preview cap when flattening a JSONL into review-sheet text."""


def _resolve_transcript_path(uri: str) -> Path:
    """Translate a ``file://`` URI back into an absolute :class:`Path`.

    The binding module emits ``Path.resolve().as_uri()``, which produces
    a percent-encoded ``file://`` URI. Round-trip through ``urlparse`` so
    spaces and other escaped characters survive.

    Raises
    ------
    ValueError
        If the URI scheme isn't ``file``. The review-sheet worker has
        no S3 / git-notes loader yet — those entry points are reserved
        for future emitters per RFC 0001.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(
            f"transcript URI scheme {parsed.scheme!r} not supported by review-sheet worker; "
            "expected file://"
        )
    return Path(unquote(parsed.path))


def _flatten_jsonl_to_text(jsonl_path: Path, *, total_max_chars: int = 32_000) -> str:
    """Compress a Claude Code JSONL into one event-per-line review text.

    Mirrors the shape of :meth:`SessionTextCorpus.assemble` but works
    directly off a single JSONL — no DuckDB connection required. This is
    what makes the worker testable in isolation: hand it a tmp_path
    JSONL and a ``file://`` URI override and it produces the same
    flattened format that the prompt expects.

    The output cap matches ``Settings.session_text_total_max_chars``
    defaults so a long session still fits inside the Sonnet 4.6 context
    after the system prompt and schema overhead.
    """
    lines: list[str] = []
    running = 0
    with jsonl_path.open(encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                # Skip malformed lines — same forgiving posture as
                # ``read_json(..., ignore_errors=true)`` upstream.
                continue
            line = _render_event_line(record)
            if not line:
                continue
            if running + len(line) + 1 > total_max_chars:
                lines.append(
                    f"…(transcript truncated at {total_max_chars} chars; "
                    f"{len(lines)} events rendered)"
                )
                break
            lines.append(line)
            running += len(line) + 1
    return "\n".join(lines)


def _render_event_line(record: dict[str, Any]) -> str:
    """Render one Claude Code JSONL record as a single review-text line.

    Falls back to ``""`` (skipped) for record types we don't surface in
    the review sheet — snapshots, permission updates, and other
    bookkeeping events. The renderer is deliberately conservative:
    review-sheet quality depends on the prompt seeing user/assistant
    turns plus tool calls, not internal CLI events.
    """
    rec_type = record.get("type")
    ts = record.get("timestamp") or record.get("ts") or ""
    if rec_type in ("user", "assistant"):
        return _render_message(record, ts)
    return ""


def _render_message(record: dict[str, Any], ts: str) -> str:
    """Format a user/assistant message record into the review-text shape."""
    message = record.get("message")
    if not isinstance(message, dict):
        return ""
    role = message.get("role") or record.get("type") or "?"
    content = message.get("content")
    if isinstance(content, str):
        body = content[:_MAX_LINE_PREVIEW]
        return f"[{role} {ts}] {body}"
    if not isinstance(content, list):
        return ""
    rendered: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_line = _render_content_block(block, ts)
        if block_line:
            rendered.append(block_line)
    if not rendered:
        return ""
    if len(rendered) == 1 and rendered[0].startswith("["):
        # Tool-only block — return as its own line so the prompt sees
        # the tool_use / tool_result framing the system prompt expects.
        return rendered[0]
    body = " ".join(rendered)[:_MAX_LINE_PREVIEW]
    return f"[{role} {ts}] {body}"


def _render_content_block(block: dict[str, Any], ts: str) -> str:
    """Format one content block (text / tool_use / tool_result)."""
    btype = block.get("type")
    if btype == "text":
        text = block.get("text", "")
        if not isinstance(text, str):
            return ""
        return text
    if btype == "tool_use":
        name = block.get("name") or "tool"
        tool_input = block.get("input")
        return f"[tool_use:{name} {ts}] {_safe_preview(tool_input)}"
    if btype == "tool_result":
        tu_id = block.get("tool_use_id") or "?"
        return f"[tool_result {tu_id} {ts}] {_safe_preview(block.get('content'))}"
    if btype == "thinking":
        # Skip thinking blocks — they're agent-internal, not review evidence.
        return ""
    return ""


def _safe_preview(value: Any) -> str:
    """Compact one-line preview of arbitrary JSON content."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:_MAX_LINE_PREVIEW]
    try:
        rendered = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        rendered = str(value)
    return rendered[:_MAX_LINE_PREVIEW]


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------


def _digest_transcript_text(text: str) -> str:
    """Stable digest of the flattened transcript for plan/metadata output.

    Uses a short SHA-256 prefix so dry-run plans are diff-friendly across
    runs. Distinct from :func:`claude_sql.binding.compute_digest`, which
    hashes the raw JSONL bytes — this digest covers what the LLM saw,
    not what's on disk.
    """
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{h[:16]}"


def _resolve_uri_for_commit(
    commit_sha: str,
    *,
    transcript_uri_override: str | None,
) -> str:
    """Pick the transcript URI: override if set, else the bound trailer."""
    if transcript_uri_override is not None:
        return transcript_uri_override
    binding = resolve_commit_to_transcript(commit_sha)
    return binding.uri


def generate_review_sheet(
    con: duckdb.DuckDBPyConnection | None,
    settings: Settings,
    *,
    commit_sha: str,
    transcript_uri_override: str | None = None,
    dry_run: bool = True,
    no_thinking: bool = False,
) -> dict[str, Any]:
    """Produce a PR review sheet for ``commit_sha``.

    Parameters
    ----------
    con
        DuckDB connection (unused on the override / file:// path; kept
        in the signature so future S3 / git-notes resolution can scan
        the views without changing the public API).
    settings
        :class:`Settings` driving region, model id, and concurrency.
    commit_sha
        Merged commit whose transcript should be summarized.
    transcript_uri_override
        Skip the binding lookup and load this URI directly. Reserved for
        tests and direct invocation; production callers pass ``None``.
    dry_run
        When ``True`` returns a plan dict without invoking Bedrock.
    no_thinking
        Force ``thinking_mode='disabled'``. Defaults to adaptive
        thinking, which adds ~10-30% latency but improves field
        synthesis quality on edge cases.

    Returns
    -------
    dict
        Under ``dry_run=True``: ``{"plan": {commit_sha, transcript_uri,
        transcript_digest, model_id, prompt_chars_estimate, dry_run}}``.

        Under successful Bedrock call: ``{"sheet": <PRReviewSheet dict>,
        "metadata": {commit_sha, transcript_uri, transcript_digest,
        model_id, captured_at}}``.

        Under :class:`BedrockRefusalError`: ``{"refused": True,
        "reason": <str>}``.
    """
    del con  # connection is not consumed on the file:// path; reserved for future loaders.
    transcript_uri = _resolve_uri_for_commit(
        commit_sha, transcript_uri_override=transcript_uri_override
    )
    transcript_path = _resolve_transcript_path(transcript_uri)
    if not transcript_path.is_file():
        raise FileNotFoundError(
            f"transcript JSONL not found at resolved path {transcript_path!s} "
            f"(uri={transcript_uri})"
        )
    transcript_text = _flatten_jsonl_to_text(transcript_path)
    transcript_digest = _digest_transcript_text(transcript_text)

    if dry_run:
        return {
            "plan": {
                "commit_sha": commit_sha,
                "transcript_uri": transcript_uri,
                "transcript_digest": transcript_digest,
                "model_id": settings.sonnet_model_id,
                "prompt_chars_estimate": len(transcript_text),
                "dry_run": True,
            }
        }

    thinking_mode = "disabled" if no_thinking else settings.classify_thinking
    user_text = (
        "Compress the following bound transcript into a PR review sheet "
        "matching the schema. Be faithful to the events; do not invent "
        "files, tools, or corrections.\n\n"
        f"COMMIT: {commit_sha}\n"
        f"TRANSCRIPT URI: {transcript_uri}\n"
        f"TRANSCRIPT DIGEST: {transcript_digest}\n\n"
        "TRANSCRIPT (chronological events):\n"
        "```\n"
        f"{transcript_text}\n"
        "```\n"
    )

    client = _build_bedrock_client(settings)
    try:
        sheet = _invoke_classifier_sync(
            client,
            settings.sonnet_model_id,
            PR_REVIEW_SHEET_SCHEMA,
            user_text,
            max_tokens=settings.classify_max_tokens,
            thinking_mode=thinking_mode,
            system=REVIEW_SHEET_SYSTEM_PROMPT,
        )
    except BedrockRefusalError as exc:
        logger.info("review-sheet: refused by Bedrock — {}", exc)
        return {
            "refused": True,
            "reason": str(exc),
            "metadata": {
                "commit_sha": commit_sha,
                "transcript_uri": transcript_uri,
                "transcript_digest": transcript_digest,
                "model_id": settings.sonnet_model_id,
                "captured_at": datetime.now(UTC).isoformat(),
            },
        }

    return {
        "sheet": sheet,
        "metadata": {
            "commit_sha": commit_sha,
            "transcript_uri": transcript_uri,
            "transcript_digest": transcript_digest,
            "model_id": settings.sonnet_model_id,
            "captured_at": datetime.now(UTC).isoformat(),
        },
    }


__all__ = [
    "REVIEW_SHEET_SYSTEM_PROMPT",
    "generate_review_sheet",
]

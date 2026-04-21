"""Blind-handover stripper for grader-safe session bundles.

Goal: hand a session transcript to an external grader (cross-provider
judge) with all identity markers removed, so the grader cannot use
"who sent this" as a cue.  The stripper removes:

* Slack IDs: users (``U[A-Z0-9]{8,}``), channels (``C[A-Z0-9]+``),
  teams (``T[A-Z0-9]+``), DM channels (``D[A-Z0-9]+``), workflows
  (``W[A-Z0-9]+``).
* Agent persona markers: ``:moyai:``, ``Bonk →``, ``Bonk:``,
  ``Clod:``, ``Clod →``, ``-- Bonk`` sign-offs.
* Protocol tokens: ``over :radio:``, ``out :radio:``.
* Tool names: ``mcp__<server>__<tool>`` calls in plain text.
* System IDs: OTel trace IDs, session UUIDs, work item IDs
  (``wi_[0-9a-f]+``), thread timestamps (``\\d{10}\\.\\d{6}``).
* Mrkdwn/markdown formatting: ``<@UXXX>``, ``<#CXXX|name>``,
  ``<!here>``, ``<!channel>``.

Original session_id is hashed (SHA256[:16]) so bundles remain
re-linkable for post-grading analysis without leaking identity.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Precompiled patterns
# ---------------------------------------------------------------------------

_SLACK_USER_RE = re.compile(r"\bU[A-Z0-9]{8,}\b")
_SLACK_CHANNEL_RE = re.compile(r"\bC[A-Z0-9]{8,}\b")
_SLACK_TEAM_RE = re.compile(r"\bT[A-Z0-9]{8,}\b")
_SLACK_DM_RE = re.compile(r"\bD[A-Z0-9]{8,}\b")
_SLACK_WORKFLOW_RE = re.compile(r"\bW[A-Z0-9]{8,}\b")

# Mrkdwn refs like <@UABC123>, <#CABC123|chan>, <!here>, <!channel>
_MRKDWN_USER_REF_RE = re.compile(r"<@U[A-Z0-9]{8,}>")
_MRKDWN_CHANNEL_REF_RE = re.compile(r"<#C[A-Z0-9]{8,}(?:\|[^>]*)?>")
_MRKDWN_BROADCAST_RE = re.compile(r"<!(?:here|channel|everyone)>")

# Agent persona markers (case-sensitive on purpose — these are brand tokens)
_PERSONA_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r":moyai:\s*"), ""),
    (re.compile(r"\bBonk\s*(?:→|->)\s*\w+:?\s*"), "[agent] "),
    (re.compile(r"\bClod\s*(?:→|->)\s*\w+:?\s*"), "[agent] "),
    (re.compile(r"(?<![a-zA-Z])Bonk\b"), "[agent]"),
    (re.compile(r"(?<![a-zA-Z])Clod\b"), "[agent]"),
    (re.compile(r"--\s*\[agent\]\s*"), ""),
)

# Protocol tokens
_RADIO_OVER_RE = re.compile(r"over\s*:radio:\s*", re.IGNORECASE)
_RADIO_OUT_RE = re.compile(r"out\s*:radio:\s*", re.IGNORECASE)

# System IDs
_WORK_ITEM_RE = re.compile(r"\bwi_[0-9a-f]{12}\b")
_THREAD_TS_RE = re.compile(r"\b\d{10}\.\d{6,}\b")
_OTEL_TRACE_ID_RE = re.compile(r"\b[0-9a-f]{32}\b")
_SESSION_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")

# MCP tool names
_MCP_TOOL_RE = re.compile(r"\bmcp__[a-z0-9_-]+__[a-z0-9_-]+\b")


@dataclass(frozen=True)
class BlindResult:
    """Output of ``strip_text``: the cleaned text + diagnostic counts."""

    text: str
    n_user_ids: int
    n_channel_ids: int
    n_persona_markers: int
    n_protocol_tokens: int
    n_system_ids: int
    n_tool_names: int


def strip_text(text: str) -> BlindResult:
    """Strip every identity marker from ``text``.

    Returns both the cleaned text and a count of each category, so
    callers can log per-strip diagnostics (e.g., "session X had 12
    persona markers, 4 thread timestamps removed").
    """
    # Count BEFORE replacement so diagnostic counts are accurate
    n_user_ids = len(_SLACK_USER_RE.findall(text)) + len(_MRKDWN_USER_REF_RE.findall(text))
    n_channel_ids = (
        len(_SLACK_CHANNEL_RE.findall(text))
        + len(_SLACK_TEAM_RE.findall(text))
        + len(_SLACK_DM_RE.findall(text))
        + len(_SLACK_WORKFLOW_RE.findall(text))
        + len(_MRKDWN_CHANNEL_REF_RE.findall(text))
        + len(_MRKDWN_BROADCAST_RE.findall(text))
    )
    n_tool_names = len(_MCP_TOOL_RE.findall(text))
    n_system_ids = (
        len(_WORK_ITEM_RE.findall(text))
        + len(_THREAD_TS_RE.findall(text))
        + len(_OTEL_TRACE_ID_RE.findall(text))
        + len(_SESSION_UUID_RE.findall(text))
    )
    n_protocol_tokens = len(_RADIO_OVER_RE.findall(text)) + len(_RADIO_OUT_RE.findall(text))
    n_persona_markers = sum(len(pat.findall(text)) for pat, _ in _PERSONA_PATTERNS)

    # Replacements. Order matters: longest-tokens-first for mrkdwn refs
    # before bare ID patterns, otherwise ``<@U...>`` would have the
    # inner ID stripped first and leave stray angle brackets.
    text = _MRKDWN_USER_REF_RE.sub("[user]", text)
    text = _MRKDWN_CHANNEL_REF_RE.sub("[channel]", text)
    text = _MRKDWN_BROADCAST_RE.sub("[broadcast]", text)
    text = _SLACK_USER_RE.sub("[user]", text)
    text = _SLACK_CHANNEL_RE.sub("[channel]", text)
    text = _SLACK_TEAM_RE.sub("[team]", text)
    text = _SLACK_DM_RE.sub("[dm]", text)
    text = _SLACK_WORKFLOW_RE.sub("[workflow]", text)

    for pat, replacement in _PERSONA_PATTERNS:
        text = pat.sub(replacement, text)

    text = _RADIO_OVER_RE.sub("[end-of-turn] ", text)
    text = _RADIO_OUT_RE.sub("[sign-off] ", text)

    text = _MCP_TOOL_RE.sub("[tool]", text)
    text = _WORK_ITEM_RE.sub("[work-item]", text)
    text = _THREAD_TS_RE.sub("[ts]", text)
    # OTel trace IDs before session UUIDs — they overlap in character set
    # but not in structure (dashed vs not).
    text = _OTEL_TRACE_ID_RE.sub("[trace]", text)
    text = _SESSION_UUID_RE.sub("[session]", text)

    # Collapse runs of whitespace introduced by strip operations
    text = re.sub(r"[ \t]{2,}", " ", text).strip()

    return BlindResult(
        text=text,
        n_user_ids=n_user_ids,
        n_channel_ids=n_channel_ids,
        n_persona_markers=n_persona_markers,
        n_protocol_tokens=n_protocol_tokens,
        n_system_ids=n_system_ids,
        n_tool_names=n_tool_names,
    )


def original_hash(session_id: str) -> str:
    """Return a stable 16-char hash of ``session_id`` for re-linkage.

    Used so the blinded bundle keeps a ``original_hash`` column that
    lets post-grading analysis re-associate scores with the true
    session without exposing the ID to the grader.
    """
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]

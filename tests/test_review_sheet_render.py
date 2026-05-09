"""Coverage top-up for :mod:`claude_sql.review_sheet_render`.

Targets the branches the existing test_review_sheet_worker.py doesn't
already exercise: digest-without-colon, non-dict correction entries,
asymmetric correction shapes, empty tools_used / exploration, and the
:func:`render_refusal_markdown` path.
"""

from __future__ import annotations

from claude_sql.review_sheet_render import (
    _format_corrections,
    _format_inline_code_list,
    _format_refusal_lines,
    _short_digest,
    render_markdown,
    render_refusal_markdown,
)


def test_short_digest_without_colon_truncates() -> None:
    """A digest missing the ``sha256:`` prefix is truncated head-on (line 30)."""
    raw = "abcdef0123456789abcdef0123456789"
    got = _short_digest(raw)
    assert got == raw[:16]
    assert ":" not in got


def test_short_digest_with_colon_preserves_prefix() -> None:
    """The standard form keeps ``sha256:`` and trims the hex tail."""
    digest = "sha256:" + "a" * 64
    got = _short_digest(digest)
    assert got.startswith("sha256:")
    assert got == "sha256:" + "a" * 16


def test_format_corrections_skips_non_dict_entries() -> None:
    """Non-dict items in the corrections list are silently skipped (line 42)."""
    out = _format_corrections(
        [
            "not a dict",  # type: ignore[list-item]
            {"what_agent_did": "Did the thing.", "correction": "Don't."},
        ]
    )
    # Non-dict skipped; dict entry rendered as bullet.
    assert len(out) == 1
    assert "Did the thing." in out[0]
    assert "Don't." in out[0]


def test_format_corrections_what_only_branch() -> None:
    """``what_agent_did`` set, ``correction`` empty → italic-only bullet (line 47-48)."""
    out = _format_corrections([{"what_agent_did": "Wrote draft.", "correction": ""}])
    assert out == ["- *Wrote draft.*"]


def test_format_corrections_correction_only_branch() -> None:
    """``correction`` set, ``what_agent_did`` empty → bare bullet (line 49-50)."""
    out = _format_corrections([{"what_agent_did": "", "correction": "Use the rotator."}])
    assert out == ["- Use the rotator."]


def test_format_corrections_all_empty_falls_through_to_none_placeholder() -> None:
    """When every entry has empty fields, the renderer drops back to ``_None._``."""
    out = _format_corrections([{"what_agent_did": "", "correction": ""}])
    assert out == ["_None._"]


def test_format_inline_code_list_empty_returns_none_placeholder() -> None:
    """Empty tools_used renders as ``_None._`` (line 57)."""
    assert _format_inline_code_list([]) == "_None._"


def test_format_refusal_lines_empty_returns_none_placeholder() -> None:
    """Empty refusal list falls through to the placeholder."""
    assert _format_refusal_lines([]) == ["_None._"]


def test_format_refusal_lines_skips_empty_strings() -> None:
    """Entries that are empty strings are filtered out by the comprehension."""
    assert _format_refusal_lines(["Bash: blocked", ""]) == ["- `Bash: blocked`"]


def test_render_markdown_empty_exploration_emits_none_placeholder() -> None:
    """When ``agent_exploration`` is missing/empty, the section body is ``_None._``
    (line 118).
    """
    sheet = {
        "human_intent": "Tweak something.",
        "agent_exploration": [],
        "corrections": [],
        "tools_used": ["Read"],
        "tools_refused": [],
        "diff_rationale": "Nothing exciting.",
    }
    metadata = {
        "commit_sha": "deadbeefcafe0000",
        "transcript_uri": "file:///tmp/x.jsonl",
        "transcript_digest": "sha256:deadbeefcafe",
        "model_id": "global.anthropic.claude-sonnet-4-6",
        "captured_at": "2026-05-09T11:00:00+00:00",
    }
    md = render_markdown(sheet, metadata)
    explored_idx = md.find("## What the agent explored")
    next_section = md.find("## Corrections", explored_idx)
    section = md[explored_idx:next_section]
    assert "_None._" in section


def test_render_markdown_handles_missing_strings_with_placeholders() -> None:
    """Missing human_intent / diff_rationale render as ``_(missing)_``."""
    sheet = {
        "human_intent": "",
        "agent_exploration": ["a"],
        "corrections": [],
        "tools_used": [],
        "tools_refused": [],
        "diff_rationale": "",
    }
    md = render_markdown(sheet, {})
    assert "_(missing)_" in md


def test_render_refusal_markdown_renders_header_and_reason() -> None:
    """``render_refusal_markdown`` covers lines 142-161 — the refusal output path."""
    metadata = {
        "commit_sha": "abc1234deadbeef0000",
        "transcript_uri": "file:///tmp/transcript.jsonl",
        "transcript_digest": "sha256:0123456789abcdef0123456789abcdef",
        "model_id": "global.anthropic.claude-sonnet-4-6",
        "captured_at": "2026-05-09T10:00:00+00:00",
    }
    out = render_refusal_markdown("model declined under content policy", metadata)
    # Header still keys off the short SHA.
    assert out.startswith("# PR Review Sheet — `abc1234deadb`")
    # Refusal note + quoted reason both present.
    assert "_Review sheet refused; see metadata._" in out
    assert "> model declined under content policy" in out
    # Header metadata is reproduced.
    assert "**Transcript:**" in out
    assert "**Agent runtime:** global.anthropic.claude-sonnet-4-6" in out
    assert "**Generated:** 2026-05-09T10:00:00+00:00" in out
    # Trailing newline so it's pipe-friendly.
    assert out.endswith("\n")


def test_render_refusal_markdown_with_empty_metadata_does_not_raise() -> None:
    """Robustness: missing metadata fields render as empty placeholders."""
    out = render_refusal_markdown("nope", {})
    assert "# PR Review Sheet —" in out
    assert "> nope" in out

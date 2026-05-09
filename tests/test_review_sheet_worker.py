"""Tests for the PR review-sheet worker, schema, and Markdown renderer.

The Bedrock invoke path is exercised with a fake client (the same
``_FakeBedrockClient`` shape used by ``test_friction_worker.py``) so the
tests never reach the network. JSONL fixtures live under ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from claude_sql import review_sheet_worker
from claude_sql.config import Settings
from claude_sql.review_sheet_render import render_markdown
from claude_sql.schemas import PR_REVIEW_SHEET_SCHEMA, PRReviewSheet

# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------


def _walk(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Yield every nested dict in a flattened JSON Schema (assertion helper)."""
    out: list[dict[str, Any]] = [schema]
    for v in schema.values():
        if isinstance(v, dict):
            out.extend(_walk(v))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    out.extend(_walk(item))
    return out


def test_pr_review_sheet_schema_flattened() -> None:
    """``PR_REVIEW_SHEET_SCHEMA`` must pass Bedrock's Draft 2020-12 subset.

    Specifically: no ``$ref`` / ``$defs`` (nested ``Correction`` should be
    inlined), ``additionalProperties: false`` on every object node, and
    none of the numeric/string/array constraints that Bedrock's
    validator rejects (``minLength``, ``maxLength``, ``minItems``, …).
    Field-level ``description`` strings must survive — they carry the
    semantics the model needs to populate the sheet.
    """
    s = PR_REVIEW_SHEET_SCHEMA
    dumped = json.dumps(s)

    assert "$ref" not in dumped, "nested Correction must be inlined"
    assert "$defs" not in dumped, "schema must not carry $defs"

    for node in _walk(s):
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False, (
                f"object node missing additionalProperties=false: {node}"
            )
        # Bedrock-rejected constraints stripped from every node.
        banned_string = {"minLength", "maxLength", "pattern", "format"}
        banned_number = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}
        banned_array = {"minItems", "maxItems", "uniqueItems"}
        for key in banned_string | banned_number | banned_array:
            assert key not in node, f"banned constraint {key!r} survived: {node}"

    # Required top-level fields cover the six PRReviewSheet fields.
    assert set(s["required"]) == {
        "human_intent",
        "agent_exploration",
        "corrections",
        "tools_used",
        "tools_refused",
        "diff_rationale",
    }

    # Field descriptions survive for both the top-level fields and the
    # nested Correction shape.
    assert s["properties"]["human_intent"].get("description")
    correction_schema = s["properties"]["corrections"]["items"]
    assert correction_schema["additionalProperties"] is False
    assert set(correction_schema["required"]) == {"what_agent_did", "correction"}
    assert correction_schema["properties"]["what_agent_did"].get("description")
    assert correction_schema["properties"]["correction"].get("description")


def test_pydantic_pr_review_sheet_round_trip() -> None:
    """Happy-path validation: the model accepts a fully populated dict."""
    sheet = PRReviewSheet(
        human_intent="Fix the off-by-one in the pagination cursor.",
        agent_exploration=[
            "Read src/api/pagination.py",
            "Ran the failing pagination test",
            "Inspected cursor advance math",
        ],
        corrections=[],
        tools_used=["Read", "Bash", "Edit"],
        tools_refused=[],
        diff_rationale=(
            "Adjusted the cursor advance math in src/api/pagination.py from "
            "len(rows) to len(rows)-1 so the next page starts on the correct "
            "row. Test pagination_test.py::test_cursor_edge covers the "
            "regression."
        ),
    )
    assert sheet.tools_used == ["Read", "Bash", "Edit"]
    assert sheet.corrections == []


# ---------------------------------------------------------------------------
# Bedrock invoke shape
# ---------------------------------------------------------------------------


class _FakeBody:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._payload


class _FakeBedrockClient:
    def __init__(self, response_payload: dict[str, Any]) -> None:
        self.response_payload = response_payload
        self.captured: dict[str, Any] = {}

    def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        self.captured["modelId"] = kwargs["modelId"]
        self.captured["body"] = json.loads(kwargs["body"])
        return {"body": _FakeBody(self.response_payload)}


def _write_minimal_jsonl(path: Path) -> None:
    """Write a JSONL fixture with one user turn + one assistant tool_use."""
    records = [
        {
            "type": "user",
            "timestamp": "2026-05-09T10:00:00Z",
            "sessionId": "fixture-session",
            "message": {
                "role": "user",
                "content": "fix the off-by-one in the pagination cursor",
            },
        },
        {
            "type": "assistant",
            "timestamp": "2026-05-09T10:00:05Z",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "Read",
                        "input": {"file_path": "/repo/src/api/pagination.py"},
                    }
                ],
            },
        },
        {
            "type": "user",
            "timestamp": "2026-05-09T10:00:06Z",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": "def cursor_advance(rows): return len(rows)",
                    }
                ],
            },
        },
    ]
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r))
            fh.write("\n")


def _settings_for_test(tmp_path: Path) -> Settings:
    """Build a Settings instance with all paths pinned under ``tmp_path``.

    Avoids touching the user's real ``~/.claude/`` parquet caches and
    keeps the test hermetic regardless of what the dev box has cached.
    """
    return Settings(
        user_friction_parquet_path=tmp_path / "user_friction.parquet",
        checkpoint_db_path=tmp_path / "checkpoint.duckdb",
    )


def test_review_sheet_invoke_uses_pr_review_sheet_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The non-dry-run path calls invoke_model with PR_REVIEW_SHEET_SCHEMA.

    Mirrors the friction-worker pattern: capture the request body, assert
    the structured-output schema is wired through ``output_config.format``
    and that no tool_use machinery is present.
    """
    jsonl = tmp_path / "transcript.jsonl"
    _write_minimal_jsonl(jsonl)

    fake_payload = {
        "output": {
            "human_intent": "Fix the off-by-one in the pagination cursor.",
            "agent_exploration": [
                "Read src/api/pagination.py",
            ],
            "corrections": [],
            "tools_used": ["Read"],
            "tools_refused": [],
            "diff_rationale": (
                "Adjusted cursor advance to use len(rows)-1 in "
                "src/api/pagination.py so the next page starts on the "
                "correct row."
            ),
        }
    }
    fake_client = _FakeBedrockClient(fake_payload)
    monkeypatch.setattr(review_sheet_worker, "_build_bedrock_client", lambda settings: fake_client)

    settings = _settings_for_test(tmp_path)
    result = review_sheet_worker.generate_review_sheet(
        None,
        settings,
        commit_sha="abc1234deadbeef",
        transcript_uri_override=jsonl.resolve().as_uri(),
        dry_run=False,
        no_thinking=True,
    )

    assert "sheet" in result
    body = fake_client.captured["body"]
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["output_config"]["format"]["schema"] == PR_REVIEW_SHEET_SCHEMA
    assert "tool_choice" not in body
    assert "tools" not in body
    # System prompt with Anthropic XML tags is sent as a content-block list.
    system = body["system"]
    assert isinstance(system, list)
    assert system[0]["type"] == "text"
    assert "<instructions>" in system[0]["text"]
    assert "<context>" in system[0]["text"]
    # Metadata round-trip survives.
    assert result["metadata"]["commit_sha"] == "abc1234deadbeef"
    assert result["metadata"]["transcript_uri"] == jsonl.resolve().as_uri()
    assert result["metadata"]["transcript_digest"].startswith("sha256:")


def test_review_sheet_dry_run_returns_plan_dict(tmp_path: Path) -> None:
    """Dry-run path skips Bedrock and returns the plan keys per spec."""
    jsonl = tmp_path / "transcript.jsonl"
    _write_minimal_jsonl(jsonl)

    settings = _settings_for_test(tmp_path)
    result = review_sheet_worker.generate_review_sheet(
        None,
        settings,
        commit_sha="abc1234deadbeef",
        transcript_uri_override=jsonl.resolve().as_uri(),
        dry_run=True,
    )

    plan = result["plan"]
    assert plan["commit_sha"] == "abc1234deadbeef"
    assert plan["transcript_uri"] == jsonl.resolve().as_uri()
    assert plan["transcript_digest"].startswith("sha256:")
    assert plan["model_id"] == settings.sonnet_model_id
    assert plan["dry_run"] is True
    assert plan["prompt_chars_estimate"] > 0
    # No "sheet" key on dry-run; otherwise downstream consumers conflate
    # "we ran" and "we didn't run".
    assert "sheet" not in result


def test_review_sheet_refusal_returns_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Bedrock refusal short-circuits to ``{"refused": True, ...}``.

    Mirrors the production behavior: ``BedrockRefusalError`` is terminal,
    we don't retry, and the caller gets a structured signal it can use
    to decide whether to re-prompt or skip.
    """
    jsonl = tmp_path / "transcript.jsonl"
    _write_minimal_jsonl(jsonl)

    refusal_payload = {"stop_reason": "refusal"}
    fake_client = _FakeBedrockClient(refusal_payload)
    monkeypatch.setattr(review_sheet_worker, "_build_bedrock_client", lambda settings: fake_client)

    settings = _settings_for_test(tmp_path)
    result = review_sheet_worker.generate_review_sheet(
        None,
        settings,
        commit_sha="abc1234deadbeef",
        transcript_uri_override=jsonl.resolve().as_uri(),
        dry_run=False,
        no_thinking=True,
    )

    assert result["refused"] is True
    assert "reason" in result
    assert "metadata" in result
    assert result["metadata"]["commit_sha"] == "abc1234deadbeef"


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def test_render_markdown_emits_canonical_sections() -> None:
    """``render_markdown`` produces the five-section shape from the plan.

    Spec lives in ``.erpaval/sessions/session-2293a5/plan.md`` §Action 2
    "Markdown rendering". The rendered string must contain the
    ``# PR Review Sheet`` header keyed off the short commit SHA and
    each of the five canonical section headers in order.
    """
    sheet = {
        "human_intent": "Fix the off-by-one in the pagination cursor.",
        "agent_exploration": [
            "Read src/api/pagination.py",
            "Ran the failing pagination test",
        ],
        "corrections": [
            {
                "what_agent_did": "Started rewriting from scratch.",
                "correction": "Asked to keep the existing module and only update the rotator.",
            }
        ],
        "tools_used": ["Read", "Bash", "Edit"],
        "tools_refused": ["Bash: blocked by allowlist"],
        "diff_rationale": (
            "Adjusted cursor advance math in src/api/pagination.py so the "
            "next page starts on the correct row."
        ),
    }
    metadata = {
        "commit_sha": "abc1234deadbeef0000",
        "transcript_uri": "file:///tmp/transcript.jsonl",
        "transcript_digest": "sha256:0123456789abcdef0123456789abcdef",
        "model_id": "global.anthropic.claude-sonnet-4-6",
        "captured_at": "2026-05-09T10:00:00+00:00",
    }
    md = render_markdown(sheet, metadata)

    # Header line — short SHA (12 hex chars is the conventional length),
    # not the full one.
    assert md.startswith("# PR Review Sheet — `abc1234deadb`")
    assert "abc1234deadbeef0000" not in md.splitlines()[0]

    # Five section headers in order.
    section_order = [
        "## What the human asked for",
        "## What the agent explored",
        "## Corrections",
        "## Tools used",
        "## Tools refused",
        "## Why this diff",
    ]
    last_idx = -1
    for header in section_order:
        idx = md.find(header)
        assert idx > last_idx, f"section out of order: {header}"
        last_idx = idx

    # Field bodies survive into the output.
    assert "Fix the off-by-one in the pagination cursor." in md
    assert "Read src/api/pagination.py" in md
    assert "Started rewriting from scratch." in md
    assert "`Read`" in md and "`Edit`" in md
    assert "Bash: blocked by allowlist" in md


def test_render_markdown_empty_corrections_emits_none_placeholder() -> None:
    """Empty corrections list → ``_None._`` per the plan markdown shape."""
    sheet = {
        "human_intent": "Tweak a docstring.",
        "agent_exploration": ["Read foo.py"],
        "corrections": [],
        "tools_used": ["Read", "Edit"],
        "tools_refused": [],
        "diff_rationale": "Updated the docstring on foo.bar to reflect new defaults.",
    }
    metadata = {
        "commit_sha": "deadbeef",
        "transcript_uri": "file:///tmp/x.jsonl",
        "transcript_digest": "sha256:abcd",
        "model_id": "global.anthropic.claude-sonnet-4-6",
        "captured_at": "2026-05-09T11:00:00+00:00",
    }
    md = render_markdown(sheet, metadata)
    # Find the Corrections section and assert "_None._" follows immediately.
    corrections_idx = md.find("## Corrections")
    assert corrections_idx >= 0
    next_section_idx = md.find("## Tools used", corrections_idx)
    section_body = md[corrections_idx:next_section_idx]
    assert "_None._" in section_body

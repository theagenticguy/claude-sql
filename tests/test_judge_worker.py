"""Unit tests for judge_worker (pure functions; Bedrock calls are mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest

from claude_sql import judge_worker as jw

RUBRIC_YAML = """
axes:
  - name: correction_required
    description: Did the user have to correct the agent this turn?
    detector_vs_grader: detector
    levels:
      0: No correction required.
      1: Correction required.
  - name: fabrication_present
    description: Did the agent state something unverified?
    detector_vs_grader: detector
    levels:
      0: No fabrication.
      1: Fabrication present.
"""


def test_parse_rubric_extracts_axes() -> None:
    axes = jw.parse_rubric(RUBRIC_YAML)
    assert len(axes) == 2
    assert axes[0].name == "correction_required"
    assert axes[0].levels == {0: "No correction required.", 1: "Correction required."}
    assert axes[1].detector_vs_grader == "detector"


def test_parse_rubric_rejects_empty() -> None:
    with pytest.raises(ValueError):
        jw.parse_rubric("axes: []")


def test_parse_rubric_rejects_missing_name() -> None:
    bad = "axes:\n  - description: no name here\n"
    with pytest.raises(ValueError):
        jw.parse_rubric(bad)


def test_render_prompt_contains_axis_and_transcript() -> None:
    axis = jw.parse_rubric(RUBRIC_YAML)[0]
    prompt = jw.render_prompt("hello world", axis)
    assert "correction_required" in prompt
    assert "hello world" in prompt
    assert "score=" in prompt
    assert "rationale=" in prompt


def test_parse_judge_response_ideal_format() -> None:
    text = "score=1\nrationale=The agent fabricated a stop-hook."
    score, rationale = jw.parse_judge_response(text)
    assert score == 1
    assert "fabricated" in rationale


def test_parse_judge_response_json_fallback() -> None:
    text = '{"score": 0, "rationale": "clean turn"}'
    score, rationale = jw.parse_judge_response(text)
    assert score == 0


def test_parse_judge_response_no_score_returns_none() -> None:
    text = "I cannot score this."
    score, rationale = jw.parse_judge_response(text)
    assert score is None
    # No truncation: the full malformed response is preserved so a human
    # reviewer can see exactly what the judge emitted.
    assert rationale == "I cannot score this."


def test_parse_judge_response_preserves_long_malformed_text() -> None:
    """Regression: the fallback path previously truncated to 1000 chars."""
    text = "no-score-here " + ("x" * 5000)
    _, rationale = jw.parse_judge_response(text)
    assert len(rationale) > 4000


def test_plan_estimates_calls_and_dollars() -> None:
    axes = jw.parse_rubric(RUBRIC_YAML)
    panel = jw.judge_catalog.panel(["kimi-k2.5", "deepseek-v3.2"])
    sessions = [("s1", "a" * 4000), ("s2", "b" * 4000)]
    p = jw.plan(sessions, panel, axes)
    assert p.n_sessions == 2
    assert p.n_judges == 2
    assert p.n_axes == 2
    assert p.n_calls == 2 * 2 * 2
    assert p.est_input_tokens > 0
    assert p.est_usd > 0


def test_run_dry_run_skips_bedrock(tmp_path: Path) -> None:
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text(RUBRIC_YAML)
    sessions = [("s1", "transcript one")]
    result = jw.run(
        sessions=sessions,
        panel_shortnames=["kimi-k2.5"],
        rubric_yaml_path=rubric,
        freeze_sha="deadbeefcafe0000",
        out_parquet=tmp_path / "out.parquet",
        dry_run=True,
    )
    assert isinstance(result, jw.GradePlan)
    assert result.n_calls == 2  # 1 session x 1 judge x 2 axes


def test_to_parquet_roundtrip(tmp_path: Path) -> None:
    scores = [
        jw.JudgeScore(
            session_id="s1",
            axis="correction_required",
            judge_shortname="kimi-k2.5",
            judge_model_id="moonshotai.kimi-k2.5",
            score=0,
            rationale="clean",
            freeze_sha="sha0000000000000",
        ),
        jw.JudgeScore(
            session_id="s1",
            axis="fabrication_present",
            judge_shortname="kimi-k2.5",
            judge_model_id="moonshotai.kimi-k2.5",
            score=1,
            rationale="fab",
            freeze_sha="sha0000000000000",
        ),
    ]
    p = tmp_path / "scores.parquet"
    jw.to_parquet(scores, p)
    df = pl.read_parquet(p)
    assert df.height == 2
    assert set(df.columns) == {
        "session_id",
        "axis",
        "judge_shortname",
        "judge_model_id",
        "score",
        "rationale",
        "freeze_sha",
    }
    assert df.filter(pl.col("axis") == "fabrication_present")["score"][0] == 1


def test_converse_once_parses_text_block() -> None:
    mock_client = MagicMock()
    mock_client.converse.return_value = {
        "output": {"message": {"content": [{"text": "score=1\nrationale=ok"}]}}
    }
    got = jw._converse_once(mock_client, "anything", "prompt")
    assert got == "score=1\nrationale=ok"

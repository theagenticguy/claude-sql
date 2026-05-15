"""Coverage top-up for :mod:`claude_sql.judge_worker`.

Targets:

* ``_is_retryable`` for both retryable and non-retryable shapes.
* ``parse_rubric`` shape errors (axes-not-list, axes[i]-not-mapping).
* ``_bedrock_client`` returns a configured client.
* ``_converse_once`` empty-content fallback.
* ``_grade_one`` via :func:`run_async` — covers the unparseable-score
  path and the LLM-failure-skip path.
* :func:`to_parquet` empty-list warning.
* :func:`run` non-dry-run path → parquet roundtrip.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest
from botocore.exceptions import ClientError, SSLError

from claude_sql.evals import judge_worker as jw

RUBRIC = """
axes:
  - name: correction_required
    description: Detector axis.
    detector_vs_grader: detector
    levels:
      0: clean
      1: corrected
"""


# ---------------------------------------------------------------------------
# _is_retryable (lines 69-74)
# ---------------------------------------------------------------------------


def test_is_retryable_picks_up_ssl_error() -> None:
    """Network-class exceptions are retryable."""
    assert jw._is_retryable(SSLError(endpoint_url="x", error="boom")) is True


def test_is_retryable_picks_up_throttling_client_error() -> None:
    """A ClientError with a throttling code is retryable."""
    exc = ClientError(
        error_response={"Error": {"Code": "ThrottlingException", "Message": "slow"}},
        operation_name="Converse",
    )
    assert jw._is_retryable(exc) is True


def test_is_retryable_skips_non_throttling_client_error() -> None:
    """A ClientError with a non-retryable code is *not* retryable."""
    exc = ClientError(
        error_response={"Error": {"Code": "ValidationException", "Message": "bad"}},
        operation_name="Converse",
    )
    assert jw._is_retryable(exc) is False


def test_is_retryable_skips_value_error() -> None:
    """Plain unrelated exceptions fall through to ``False``."""
    assert jw._is_retryable(ValueError("nope")) is False


# ---------------------------------------------------------------------------
# parse_rubric shape errors (lines 143, 147)
# ---------------------------------------------------------------------------


def test_parse_rubric_rejects_non_list_axes() -> None:
    """``axes`` keyed at a non-list raises ``TypeError`` (line 143)."""
    bad = "axes:\n  not_a_list: yes\n"
    with pytest.raises(TypeError, match="axes must be a list"):
        jw.parse_rubric(bad)


def test_parse_rubric_rejects_non_mapping_axis_entry() -> None:
    """An axis entry that isn't a mapping raises ``TypeError`` (line 147)."""
    bad = "axes:\n  - just_a_string\n"
    with pytest.raises(TypeError, match="must be a mapping"):
        jw.parse_rubric(bad)


# ---------------------------------------------------------------------------
# _bedrock_client (lines 222-228)
# ---------------------------------------------------------------------------


def test_bedrock_client_constructs(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_bedrock_client`` calls ``boto3.client('bedrock-runtime', ...)``."""
    captured: dict[str, Any] = {}

    def fake_client(name: str, *, config: Any) -> Any:
        captured["name"] = name
        captured["config"] = config
        return MagicMock()

    monkeypatch.setattr(jw.boto3, "client", fake_client)
    out = jw._bedrock_client(region="us-west-2")
    assert captured["name"] == "bedrock-runtime"
    # Config carries our zeroed retries; tenacity owns the policy.
    assert captured["config"].retries == {"max_attempts": 0, "mode": "standard"}
    assert out is not None


# ---------------------------------------------------------------------------
# _converse_once empty-content fallback (line 260)
# ---------------------------------------------------------------------------


def test_converse_once_empty_content_returns_empty_string() -> None:
    """When the model returns no ``text`` block, ``_converse_once`` returns ``""``."""
    mock_client = MagicMock()
    mock_client.converse.return_value = {"output": {"message": {"content": []}}}
    assert jw._converse_once(mock_client, "anything", "prompt") == ""


# ---------------------------------------------------------------------------
# _grade_one via run_async (lines 333-361)
# ---------------------------------------------------------------------------


def test_run_async_handles_unparseable_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unparseable judge response stamps ``score=-1`` with ``[unparseable]`` rationale.

    Covers lines 343-360 — the no-score branch in ``_grade_one``.
    """
    monkeypatch.setattr(jw, "_bedrock_client", lambda region="us-east-1": MagicMock())
    monkeypatch.setattr(jw, "_converse_once", lambda *a, **k: "I refuse to score this.")

    axes = jw.parse_rubric(RUBRIC)
    panel = jw.judge_catalog.panel(["kimi-k2.5"])
    out = asyncio.run(
        jw.run_async(
            sessions=[("s1", "the transcript")],
            panel=panel,
            axes=axes,
            freeze_sha="freeze-1",
            concurrency=1,
        )
    )
    assert len(out) == 1
    score = out[0]
    assert score.score == -1
    assert score.rationale.startswith("[unparseable]")


def test_run_async_skips_failed_judge_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception thrown by ``_converse_once`` makes ``_grade_one`` return ``None``.

    Covers lines 338-340 — the LLM-failure-skip branch.
    """
    monkeypatch.setattr(jw, "_bedrock_client", lambda region="us-east-1": MagicMock())

    def boom(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("network is dead")

    monkeypatch.setattr(jw, "_converse_once", boom)

    axes = jw.parse_rubric(RUBRIC)
    panel = jw.judge_catalog.panel(["kimi-k2.5"])
    out = asyncio.run(
        jw.run_async(
            sessions=[("s1", "the transcript")],
            panel=panel,
            axes=axes,
            freeze_sha="freeze-1",
            concurrency=1,
        )
    )
    # The failing call is skipped; nothing makes it into the output list.
    assert out == []


def test_run_async_happy_path_produces_score(monkeypatch: pytest.MonkeyPatch) -> None:
    """A well-formed response round-trips through ``_grade_one`` successfully.

    Covers lines 361-369 — the happy-path return.
    """
    monkeypatch.setattr(jw, "_bedrock_client", lambda region="us-east-1": MagicMock())
    monkeypatch.setattr(
        jw,
        "_converse_once",
        lambda *a, **k: "score=1\nrationale=correction was needed",
    )

    axes = jw.parse_rubric(RUBRIC)
    panel = jw.judge_catalog.panel(["kimi-k2.5"])
    out = asyncio.run(
        jw.run_async(
            sessions=[("s1", "transcript")],
            panel=panel,
            axes=axes,
            freeze_sha="freeze-1",
            concurrency=1,
        )
    )
    assert len(out) == 1
    assert out[0].score == 1
    assert "correction" in out[0].rationale


# ---------------------------------------------------------------------------
# to_parquet empty-list warning (line 405)
# ---------------------------------------------------------------------------


def test_to_parquet_empty_list_emits_warning_and_writes_empty_parquet(
    tmp_path: Path,
) -> None:
    """An empty score list still writes a schema-stable parquet."""
    out = tmp_path / "scores.parquet"
    jw.to_parquet([], out)
    assert out.exists()
    df = pl.read_parquet(out)
    assert df.height == 0
    assert set(df.columns) == {
        "session_id",
        "axis",
        "judge_shortname",
        "judge_model_id",
        "score",
        "rationale",
        "freeze_sha",
    }


# ---------------------------------------------------------------------------
# run() non-dry-run path (lines 457-461)
# ---------------------------------------------------------------------------


def test_run_non_dry_run_writes_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The non-dry-run branch dispatches ``run_async`` and writes a parquet."""
    monkeypatch.setattr(jw, "_bedrock_client", lambda region="us-east-1": MagicMock())
    monkeypatch.setattr(jw, "_converse_once", lambda *a, **k: "score=0\nrationale=clean")

    rubric_path = tmp_path / "rubric.yaml"
    rubric_path.write_text(RUBRIC)
    out_parquet = tmp_path / "scores.parquet"
    result = jw.run(
        sessions=[("s1", "transcript")],
        panel_shortnames=["kimi-k2.5"],
        rubric_yaml_path=rubric_path,
        freeze_sha="sha000000",
        out_parquet=out_parquet,
        dry_run=False,
        concurrency=1,
    )
    assert isinstance(result, list)
    assert len(result) == 1
    df = pl.read_parquet(out_parquet)
    assert df.height == 1
    assert df["score"][0] == 0

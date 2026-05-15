"""Shared pytest fixtures for the claude-sql test suite.

Exposes three high-leverage primitives so individual test modules don't
re-derive corpus scaffolding or Bedrock-client mocks:

* :func:`make_session_jsonl` — write one JSONL transcript with sane defaults.
* :func:`fake_bedrock_client` factory — captures ``invoke_model`` body and
  returns a configurable JSON payload (covers Cohere Embed v4 + Sonnet 4.6).
* :func:`tmp_corpus` / :func:`registered_con` — temp on-disk corpus + an
  open DuckDB connection with the v1 views registered against it.

Plus :func:`tmp_settings` for an ergonomic, side-effect-free
:class:`claude_sql.config.Settings` whose every cache path lives under
``tmp_path``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import duckdb
import pytest

from claude_sql.core.config import Settings
from claude_sql.core.sql_views import register_raw, register_views

# ---------------------------------------------------------------------------
# JSONL fixture builders
# ---------------------------------------------------------------------------


def _msg(
    *,
    uuid: str,
    session_id: str,
    ts: str,
    role: str = "user",
    type_: str | None = None,
    model: str | None = None,
    content: list[dict] | None = None,
    parent: str | None = None,
    usage: dict | None = None,
) -> dict[str, Any]:
    """Build one Claude Code transcript record in the v1 JSONL shape."""
    return {
        "parentUuid": parent,
        "isSidechain": False,
        "type": type_ or role,
        "uuid": uuid,
        "timestamp": ts,
        "sessionId": session_id,
        "version": "2.0.0",
        "gitBranch": "main",
        "cwd": "/home/u/proj",
        "userType": "external",
        "entrypoint": "cli",
        "permissionMode": "acceptEdits",
        "promptId": f"p-{uuid}",
        "message": {
            "id": f"m-{uuid}",
            "type": "message",
            "role": role,
            "model": model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "content": content or [{"type": "text", "text": "hello"}],
            "usage": usage
            or {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    }


def make_user_msg(
    uuid: str,
    session_id: str,
    text: str,
    *,
    ts: str | None = None,
) -> dict[str, Any]:
    """User-role text message (passes the 32-char filter when ``len(text)>=32``)."""
    return _msg(
        uuid=uuid,
        session_id=session_id,
        ts=ts or "2026-04-01T10:00:00.000Z",
        role="user",
        type_="user",
        content=[{"type": "text", "text": text}],
    )


def make_assistant_msg(
    uuid: str,
    session_id: str,
    *,
    ts: str | None = None,
    content: list[dict] | None = None,
    model: str = "claude-sonnet-4-6",
) -> dict[str, Any]:
    """Assistant-role message; supply your own ``content`` blocks for tool_use."""
    return _msg(
        uuid=uuid,
        session_id=session_id,
        ts=ts or "2026-04-01T10:00:10.000Z",
        role="assistant",
        type_="assistant",
        model=model,
        content=content or [{"type": "text", "text": "ack"}],
    )


def make_tool_result_msg(
    uuid: str,
    session_id: str,
    tool_use_id: str,
    text: str,
    *,
    ts: str | None = None,
) -> dict[str, Any]:
    """User-role tool_result message paired with a prior tool_use by id."""
    return _msg(
        uuid=uuid,
        session_id=session_id,
        ts=ts or "2026-04-01T10:00:11.000Z",
        role="user",
        type_="user",
        content=[
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": text,
            }
        ],
    )


def write_session_jsonl(
    path: Path,
    *,
    messages: list[dict[str, Any]],
) -> Path:
    """Write ``messages`` one-per-line at ``path`` and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for msg in messages:
            fh.write(json.dumps(msg))
            fh.write("\n")
    return path


# ---------------------------------------------------------------------------
# Subagent stub — ensures register_raw doesn't crash on empty subagent globs
# ---------------------------------------------------------------------------


def _seed_subagent_stub(tmp_path: Path) -> tuple[str, str]:
    """Drop a placeholder subagent under ``tmp_path`` so the globs resolve.

    Returns ``(subagent_glob, subagent_meta_glob)`` matching the stub.
    """
    sa_dir = (
        tmp_path / "projects" / "proj-stub" / "00000000-0000-0000-0000-000000000000" / "subagents"
    )
    sa_dir.mkdir(parents=True, exist_ok=True)
    write_session_jsonl(
        sa_dir / "agent-placeholder.jsonl",
        messages=[
            make_user_msg(
                "sa-stub",
                "placeholder",
                "subagent stub record so duckdb read_json infers a schema",
                ts="2026-01-01T00:00:00.000Z",
            )
        ],
    )
    (sa_dir / "agent-placeholder.meta.json").write_text(
        json.dumps({"agentType": "stub", "description": "stub subagent"})
    )
    sa_glob = str(tmp_path / "projects" / "*" / "*" / "subagents" / "agent-*.jsonl")
    sa_meta_glob = str(tmp_path / "projects" / "*" / "*" / "subagents" / "agent-*.meta.json")
    return sa_glob, sa_meta_glob


# ---------------------------------------------------------------------------
# Public fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def make_session_jsonl() -> Callable[..., Path]:
    """Factory the tests can call to write extra session JSONLs into the corpus."""

    def _make(path: Path, *, messages: list[dict[str, Any]]) -> Path:
        return write_session_jsonl(path, messages=messages)

    return _make


@pytest.fixture
def tmp_corpus(tmp_path: Path) -> dict[str, Any]:
    """Two-session on-disk corpus rooted at ``tmp_path/projects/proj-a/``.

    Returns a dict with keys ``glob``, ``subagent_glob``, ``subagent_meta_glob``,
    ``root`` (the project directory), and ``session_ids`` (list of ids written).
    """
    proj = tmp_path / "projects" / "proj-a"
    sid_one = "11111111-1111-1111-1111-111111111111"
    sid_two = "22222222-2222-2222-2222-222222222222"

    write_session_jsonl(
        proj / f"{sid_one}.jsonl",
        messages=[
            make_user_msg(
                "u1",
                sid_one,
                "first message in session one — long enough to clear the filter",
                ts="2026-04-01T10:00:00.000Z",
            ),
            make_assistant_msg(
                "a1",
                sid_one,
                ts="2026-04-01T10:00:10.000Z",
                content=[
                    {"type": "text", "text": "ok let me check the file"},
                    {
                        "type": "tool_use",
                        "id": "tu-a1",
                        "name": "Read",
                        "input": {"file_path": "/workspace/x.txt"},
                    },
                ],
            ),
            make_tool_result_msg(
                "u2",
                sid_one,
                "tu-a1",
                "file contents go here",
                ts="2026-04-01T10:00:12.000Z",
            ),
            make_user_msg(
                "u3",
                sid_one,
                "thanks, that's exactly what I expected from the file read",
                ts="2026-04-01T10:00:30.000Z",
            ),
        ],
    )
    write_session_jsonl(
        proj / f"{sid_two}.jsonl",
        messages=[
            make_user_msg(
                "v1",
                sid_two,
                "second session opening request — also long enough to clear",
                ts="2026-04-02T09:00:00.000Z",
            ),
            make_assistant_msg(
                "b1",
                sid_two,
                ts="2026-04-02T09:00:05.000Z",
                content=[{"type": "text", "text": "done"}],
            ),
        ],
    )

    sa_glob, sa_meta_glob = _seed_subagent_stub(tmp_path)
    return {
        "glob": str(tmp_path / "projects" / "*" / "*.jsonl"),
        "subagent_glob": sa_glob,
        "subagent_meta_glob": sa_meta_glob,
        "root": proj,
        "session_ids": [sid_one, sid_two],
        "tmp_path": tmp_path,
    }


@pytest.fixture
def registered_con(tmp_corpus: dict[str, Any]) -> Iterator[duckdb.DuckDBPyConnection]:
    """In-memory DuckDB with raw + analytics view registration over ``tmp_corpus``.

    Skips ``register_macros`` because the ``semantic_search`` macro depends on
    ``message_embeddings`` (created by ``register_vss`` only after a real
    embeddings parquet exists) and would otherwise crash registration on a
    bare test corpus. Tests that need macros should register VSS first.
    """
    con = duckdb.connect(":memory:")
    register_raw(
        con,
        glob=tmp_corpus["glob"],
        subagent_glob=tmp_corpus["subagent_glob"],
        subagent_meta_glob=tmp_corpus["subagent_meta_glob"],
    )
    register_views(con)
    try:
        yield con
    finally:
        con.close()


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    """A :class:`Settings` whose every cache path lives under ``tmp_path``.

    Useful for any test that constructs a worker without poking the user's
    real ``~/.claude/`` cache dir.
    """
    cache = tmp_path / "claude"
    cache.mkdir(parents=True, exist_ok=True)
    return Settings(
        embeddings_parquet_path=cache / "embeddings",
        lance_uri=cache / "embeddings_lance",
        classifications_parquet_path=cache / "classifications",
        trajectory_parquet_path=cache / "trajectory",
        conflicts_parquet_path=cache / "conflicts",
        clusters_parquet_path=cache / "clusters.parquet",
        cluster_terms_parquet_path=cache / "cluster_terms.parquet",
        communities_parquet_path=cache / "communities.parquet",
        user_friction_parquet_path=cache / "user_friction",
        skills_catalog_parquet_path=cache / "skills_catalog.parquet",
        checkpoint_db_path=cache / "state.db",
        duckdb_temp_dir=cache / "duckdb_tmp",
        user_skills_dir=cache / "skills",
        plugins_cache_dir=cache / "plugins_cache",
        embed_concurrency=2,
        llm_concurrency=2,
        batch_size=4,
    )


# ---------------------------------------------------------------------------
# Bedrock fake client
# ---------------------------------------------------------------------------


class FakeBedrockClient:
    """Drop-in for boto3 ``bedrock-runtime`` that captures the request body.

    ``payload`` is the JSON object the fake will return as ``response['body']``.
    Tests assert against ``self.captured`` to verify the request shape.

    For Cohere Embed v4 callers, set ``payload={"embeddings": {"int8": [[...], ...]}}``.
    For Sonnet 4.6 callers, set ``payload={"output": {...}}`` or one of the
    other shapes tolerated by ``llm_worker._parse_structured_payload``.
    """

    def __init__(self, payload: dict[str, Any] | list[dict[str, Any]] | None = None) -> None:
        self.payload: dict[str, Any] | list[dict[str, Any]] = payload or {}
        self.captured: list[dict[str, Any]] = []
        # Allow tests to flip behaviour mid-test (e.g., raise on the 2nd call).
        self.exc_to_raise: list[BaseException | None] = []

    def queue_exception(self, exc: BaseException) -> None:
        """Raise ``exc`` on the next ``invoke_model`` call (one-shot)."""
        self.exc_to_raise.append(exc)

    def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        body = (
            json.loads(kwargs["body"])
            if isinstance(kwargs.get("body"), (str, bytes))
            else kwargs.get("body")
        )
        record = {
            "modelId": kwargs.get("modelId"),
            "body": body,
            "contentType": kwargs.get("contentType"),
            "accept": kwargs.get("accept"),
        }
        self.captured.append(record)
        if self.exc_to_raise:
            exc = self.exc_to_raise.pop(0)
            if exc is not None:
                raise exc
        # Pop the next payload off a list, or always return the same dict.
        if isinstance(self.payload, list):
            payload = self.payload[len(self.captured) - 1] if self.payload else {}
        else:
            payload = self.payload
        return {"body": SimpleNamespace(read=lambda p=payload: json.dumps(p).encode())}


@pytest.fixture
def fake_bedrock_client() -> Callable[..., FakeBedrockClient]:
    """Factory: build a :class:`FakeBedrockClient` with a configurable payload."""

    def _make(payload: dict[str, Any] | list[dict[str, Any]] | None = None) -> FakeBedrockClient:
        return FakeBedrockClient(payload)

    return _make


# ---------------------------------------------------------------------------
# Generic capture-bound MagicMock fallback
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_invoke_model() -> Callable[[dict[str, Any]], MagicMock]:
    """Return a MagicMock with ``invoke_model`` wired to a single payload.

    Cheaper than :class:`FakeBedrockClient` when the test only cares about
    a single response and uses ``call_args`` directly.
    """

    def _make(return_payload: dict[str, Any]) -> MagicMock:
        client = MagicMock()
        body_bytes = json.dumps(return_payload).encode()
        client.invoke_model.return_value = {"body": SimpleNamespace(read=lambda: body_bytes)}
        return client

    return _make


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fixed_now() -> datetime:
    """A reproducible UTC ``datetime.now()`` substitute the tests can lean on."""
    return datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC)


def iso_at(offset_seconds: int = 0) -> str:
    """Helper: ISO-8601 UTC timestamp at ``2026-04-01T10:00:00Z + offset``."""
    base = datetime(2026, 4, 1, 10, 0, 0, tzinfo=UTC)
    return (base + timedelta(seconds=offset_seconds)).isoformat().replace("+00:00", "Z")

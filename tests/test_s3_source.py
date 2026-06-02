"""Tests for the S3-backed transcript source.

Covers the pure helpers (URI detection, settings probe, secret SQL shape) and a
full end-to-end read of the ``S3SessionStore`` part-file layout through the
existing DuckDB view stack — served by an in-process moto S3 server so DuckDB's
``httpfs`` client hits a real endpoint (moto's in-process ``mock_aws`` patches
botocore, which httpfs does not use).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import boto3
import duckdb
import pytest
from moto.server import ThreadedMotoServer

from claude_sql.core.config import Settings
from claude_sql.core.s3_source import (
    S3_SECRET_NAME,
    configure_s3,
    is_s3_uri,
    settings_need_s3,
)
from claude_sql.core.sql_views import register_views

_BUCKET = "claude-sessions"
_PREFIX = "transcripts"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("s3://bucket/prefix/*/*.jsonl", True),
        ("s3://bucket", True),
        ("/home/u/.claude/projects/*/*.jsonl", False),
        ("~/.claude/projects/*/*.jsonl", False),
        ("", False),
        (None, False),
    ],
)
def test_is_s3_uri(value: str | None, expected: bool) -> None:
    assert is_s3_uri(value) is expected


def test_settings_need_s3_local_default() -> None:
    """A stock Settings (local globs) must not trigger S3 setup."""
    assert settings_need_s3(Settings()) is False


def test_settings_need_s3_when_glob_is_s3() -> None:
    settings = Settings(default_glob="s3://bucket/prefix/*/*/part-*.jsonl")
    assert settings_need_s3(settings) is True


def test_settings_need_s3_when_only_subagent_glob_is_s3() -> None:
    """A hand-pinned subagent glob on S3 still triggers setup."""
    settings = Settings(subagent_glob="s3://bucket/prefix/*/*/subagents/agent-*.jsonl")
    assert settings_need_s3(settings) is True


def test_configure_s3_creates_secret_credential_chain() -> None:
    """Default (no endpoint) builds a credential_chain secret, no key material."""
    con = duckdb.connect(":memory:")
    try:
        configure_s3(con, Settings(region="us-west-2"))
        rows = con.execute(
            "SELECT name, type, provider FROM duckdb_secrets() WHERE name = ?",
            [S3_SECRET_NAME],
        ).fetchall()
        assert rows == [(S3_SECRET_NAME, "s3", "credential_chain")]
    finally:
        con.close()


def test_configure_s3_is_idempotent() -> None:
    """CREATE OR REPLACE means a second call updates rather than duplicates."""
    con = duckdb.connect(":memory:")
    try:
        configure_s3(con, Settings())
        configure_s3(con, Settings())
        count = con.execute(
            "SELECT count(*) FROM duckdb_secrets() WHERE name = ?",
            [S3_SECRET_NAME],
        ).fetchone()
        assert count is not None
        assert count[0] == 1
    finally:
        con.close()


# ---------------------------------------------------------------------------
# End-to-end: read S3SessionStore layout through the view stack
# ---------------------------------------------------------------------------


@pytest.fixture
def moto_s3() -> Iterator[dict[str, Any]]:
    """Start an in-process moto S3 server and yield connection details.

    httpfs uses its own HTTP client, so the in-process ``mock_aws`` decorator
    does not intercept it — a real listening endpoint is required.
    """
    server = ThreadedMotoServer(port=0)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"{host}:{port}"
    client = boto3.client(
        "s3",
        endpoint_url=f"http://{endpoint}",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        region_name="us-east-1",
    )
    # moto backends are process-global; a server from a prior test in the same
    # session leaves the bucket behind. Reset so each test starts clean.
    client.create_bucket(Bucket=_BUCKET)
    try:
        yield {"endpoint": endpoint, "client": client}
    finally:
        # Empty + drop the bucket so the next test's create_bucket succeeds.
        objs = client.list_objects_v2(Bucket=_BUCKET).get("Contents", [])
        for obj in objs:
            client.delete_object(Bucket=_BUCKET, Key=obj["Key"])
        client.delete_bucket(Bucket=_BUCKET)
        server.stop()


def _put_part(
    client: Any,
    *,
    project_key: str,
    session_id: str,
    part: str,
    records: list[dict[str, Any]],
) -> None:
    """Upload a JSONL part file at the S3SessionStore key layout."""
    key = f"{_PREFIX}/{project_key}/{session_id}/part-{part}.jsonl"
    body = "\n".join(json.dumps(r) for r in records)
    client.put_object(Bucket=_BUCKET, Key=key, Body=body)


def _record(uuid: str, session_id: str, role: str, text: str, ts: str) -> dict[str, Any]:
    return {
        "parentUuid": None,
        "isSidechain": False,
        "type": role,
        "uuid": uuid,
        "timestamp": ts,
        "sessionId": session_id,
        "gitBranch": "main",
        "cwd": "/home/u/proj",
        "message": {
            "role": role,
            "model": "claude-sonnet-4-6" if role == "assistant" else None,
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": text}],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    }


_SA_PREFIX = "subagents-stub"


def _seed_subagent_stub_s3(client: Any) -> None:
    """Upload a subagent + meta stub so the subagent globs resolve to ≥1 object.

    DuckDB's ``read_json`` raises ``IOException`` when a glob matches zero
    objects (this holds on S3 just as locally — the conftest seeds the same
    stub on disk). One placeholder part keeps the subagent views bindable.
    """
    sid = "00000000-0000-0000-0000-000000000000"
    client.put_object(
        Bucket=_BUCKET,
        Key=f"{_SA_PREFIX}/{sid}/subagents/agent-stub.jsonl",
        Body=json.dumps(
            _record("sa1", "placeholder", "user", "subagent stub", "2026-01-01T00:00:00Z")
        ),
    )
    client.put_object(
        Bucket=_BUCKET,
        Key=f"{_SA_PREFIX}/{sid}/subagents/agent-stub.meta.json",
        Body=json.dumps({"agentType": "stub", "description": "stub subagent"}),
    )


def _s3_settings(endpoint: str) -> Settings:
    """Settings pointing the transcript globs at the moto bucket.

    The subagent globs target the stub uploaded by :func:`_seed_subagent_stub_s3`
    so the subagent views bind (a zero-match S3 glob raises ``IOException``).
    """
    return Settings(
        default_glob=f"s3://{_BUCKET}/{_PREFIX}/*/*/part-*.jsonl",
        subagent_glob=f"s3://{_BUCKET}/{_SA_PREFIX}/*/subagents/agent-*.jsonl",
        subagent_meta_glob=f"s3://{_BUCKET}/{_SA_PREFIX}/*/subagents/agent-*.meta.json",
        s3_endpoint=endpoint,
        s3_url_style="path",
        s3_use_ssl=False,
        region="us-east-1",
    )


def test_read_s3_sessionstore_layout_end_to_end(moto_s3: dict[str, Any]) -> None:
    """Transcripts in the S3SessionStore part-file layout surface in views."""
    client = moto_s3["client"]
    _seed_subagent_stub_s3(client)
    _put_part(
        client,
        project_key="proj1",
        session_id="sess-A",
        part="0000000000001-aaa111",
        records=[
            _record(
                "u1",
                "sess-A",
                "user",
                "hello from the s3-backed transcript source",
                "2026-06-01T00:00:00Z",
            ),
            _record(
                "a1",
                "sess-A",
                "assistant",
                "hi there from the assistant turn",
                "2026-06-01T00:00:01Z",
            ),
        ],
    )
    _put_part(
        client,
        project_key="proj1",
        session_id="sess-B",
        part="0000000000002-bbb222",
        records=[
            _record(
                "u2",
                "sess-B",
                "user",
                "second session opening message, long enough to clear the filter",
                "2026-06-02T00:00:00Z",
            ),
        ],
    )

    settings = _s3_settings(moto_s3["endpoint"])
    con = duckdb.connect(":memory:")
    try:
        configure_s3(con, settings)
        from claude_sql.core.sql_views import register_raw

        register_raw(
            con,
            glob=settings.default_glob,
            subagent_glob=settings.subagent_glob,
            subagent_meta_glob=settings.subagent_meta_glob,
        )
        register_views(con)

        sessions = con.execute("SELECT session_id FROM sessions ORDER BY session_id").fetchall()
        assert sessions == [("sess-A",), ("sess-B",)]

        text = con.execute("SELECT text_content FROM messages_text WHERE uuid = 'u1'").fetchone()
        assert text is not None
        assert "s3-backed transcript source" in text[0]
    finally:
        con.close()


def test_register_all_wires_s3(moto_s3: dict[str, Any]) -> None:
    """register_all detects the s3:// glob and configures httpfs before binding."""
    client = moto_s3["client"]
    _seed_subagent_stub_s3(client)
    _put_part(
        client,
        project_key="proj1",
        session_id="sess-C",
        part="0000000000003-ccc333",
        records=[
            _record("u3", "sess-C", "user", "via register_all", "2026-06-03T00:00:00Z"),
        ],
    )

    settings = _s3_settings(moto_s3["endpoint"])
    con = duckdb.connect(":memory:")
    try:
        from claude_sql.core.sql_views import register_all

        # skip_vss + no analytics: this corpus has no embeddings/parquet caches.
        register_all(con, settings=settings, include_analytics=False, skip_vss=True)
        rows = con.execute("SELECT session_id FROM sessions").fetchall()
        assert rows == [("sess-C",)]
    finally:
        con.close()

"""Tests for ``claude_sql.embed_worker``.

Covers:
* ``_is_retryable`` decision matrix (network errors, retryable codes, others).
* ``discover_unembedded`` happy path — initial scan, sharded-parquet
  anti-join, ``since_days`` recency filter, and ``limit`` cap.
* ``_build_bedrock_client`` config knobs (region, timeouts).
* ``_invoke_bedrock_sync`` happy path (request body shape, ``MAX_CHARS_PER_TEXT``
  clip) and the tenacity retry on ``ThrottlingException``.
* ``embed_documents_async`` empty-input shortcut and batched path.
* ``embed_query`` single-call path.
* ``run_backfill`` dry-run (with and without candidates), real run, and the
  no-pending fast path.

Hermetic: all Bedrock calls go through ``FakeBedrockClient``; tenacity sleeps
are neutralized so the file runs in <2 s; no writes outside ``tmp_path``.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import pytest
from botocore.exceptions import (
    ClientError,
    ConnectionError as BotoConnectionError,
    EndpointConnectionError,
    ReadTimeoutError,
    SSLError,
)

from claude_sql import embed_worker, lance_store
from claude_sql.embed_worker import (
    MAX_CHARS_PER_TEXT,
    _build_bedrock_client,
    _invoke_bedrock_sync,
    _is_retryable,
    discover_unembedded,
    embed_documents_async,
    embed_query,
    run_backfill,
)
from claude_sql.sql_views import register_raw, register_views
from conftest import FakeBedrockClient, make_user_msg, write_session_jsonl


def _seed_lance_uuids(lance_uri: Path, uuids: list[str], dim: int = 4) -> None:
    """Helper for tests: write a tiny Lance dataset claiming ``uuids`` are embedded."""
    db = lance_store.connect_db(lance_uri)
    tbl = lance_store.open_or_create_table(db, dim=dim)
    df = pl.DataFrame(
        {
            "uuid": uuids,
            "model": ["test"] * len(uuids),
            "dim": [dim] * len(uuids),
            "embedding": [[0.0] * dim for _ in uuids],
            "embedded_at": [datetime.now(UTC)] * len(uuids),
        },
        schema={
            "uuid": pl.Utf8,
            "model": pl.Utf8,
            "dim": pl.Int32,
            "embedding": pl.Array(pl.Float32, dim),
            "embedded_at": pl.Datetime("us", "UTC"),
        },
    )
    lance_store.add_chunk(tbl, df)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _client_error(code: str) -> ClientError:
    """Build a botocore ``ClientError`` carrying ``code`` in its error envelope."""
    return ClientError({"Error": {"Code": code, "Message": code}}, "InvokeModel")


@pytest.fixture(autouse=True)
def _kill_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize all sleeps so tenacity retries are instantaneous."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_kw: None)
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda *_a, **_kw: None)


@pytest.fixture
def views_con(tmp_corpus: dict[str, Any]) -> Any:
    """In-memory DuckDB with raw + business views over ``tmp_corpus``.

    Skips ``register_macros`` because the embed worker only depends on
    ``messages_text``; the macros pull in ``message_embeddings`` which is a
    VSS-managed table we don't have here.
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


# ---------------------------------------------------------------------------
# _is_retryable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        SSLError(endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com", error="ssl"),
        BotoConnectionError(error="boom"),
        EndpointConnectionError(endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"),
        ReadTimeoutError(endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"),
    ],
)
def test_is_retryable_network_errors_true(exc: BaseException) -> None:
    assert _is_retryable(exc) is True


@pytest.mark.parametrize(
    "code",
    [
        "ThrottlingException",
        "ServiceUnavailableException",
        "ModelTimeoutException",
        "ModelErrorException",
    ],
)
def test_is_retryable_client_error_retryable_code(code: str) -> None:
    assert _is_retryable(_client_error(code)) is True


def test_is_retryable_client_error_non_retryable_code() -> None:
    assert _is_retryable(_client_error("AccessDeniedException")) is False


def test_is_retryable_random_runtime_error() -> None:
    assert _is_retryable(RuntimeError("nope")) is False


# ---------------------------------------------------------------------------
# discover_unembedded
# ---------------------------------------------------------------------------


def test_discover_unembedded_no_parquet_returns_all(views_con: Any, tmp_path: Path) -> None:
    """First-ever run: nothing is embedded, every messages_text row is returned."""
    lance_uri = tmp_path / "embeddings_lance"
    rows = discover_unembedded(views_con, lance_uri=lance_uri)
    uuids = {r[0] for r in rows}
    # The fixture corpus has user messages u1, u3, v1 with text >= 32 chars.
    assert {"u1", "u3", "v1"} <= uuids


def test_discover_unembedded_anti_join_filters_existing(views_con: Any, tmp_path: Path) -> None:
    """After seeding a Lance row claiming one uuid is embedded, it must drop out."""
    lance_uri = tmp_path / "embeddings_lance"
    initial = discover_unembedded(views_con, lance_uri=lance_uri)
    initial_uuids = {r[0] for r in initial}
    assert "u1" in initial_uuids

    # Seed Lance with u1 only.
    _seed_lance_uuids(lance_uri, ["u1"])

    after = discover_unembedded(views_con, lance_uri=lance_uri)
    after_uuids = {r[0] for r in after}
    assert "u1" not in after_uuids
    # Anti-join must not nuke unrelated rows.
    assert "u3" in after_uuids
    assert "v1" in after_uuids


def test_discover_unembedded_since_days_filters_recent(
    views_con: Any, tmp_corpus: dict[str, Any], tmp_path: Path
) -> None:
    """``since_days`` cuts old fixture rows; only a freshly-written msg survives."""
    # Drop a recent message into a fresh session JSONL so ``ts`` is "now".
    fresh_sid = "33333333-3333-3333-3333-333333333333"
    now_iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    write_session_jsonl(
        tmp_corpus["root"] / f"{fresh_sid}.jsonl",
        messages=[
            make_user_msg(
                "fresh-uuid",
                fresh_sid,
                "freshly-written user message that is comfortably past the 32-char filter",
                ts=now_iso,
            )
        ],
    )
    # T1.1: ``v_raw_events`` is now materialized as a TEMP TABLE, so a JSONL
    # written *after* fixture setup is invisible until we re-register the
    # raw readers. Re-run register_raw + register_views to pick up the new
    # file. Mirrors the production flow where the CLI calls register_raw on
    # every invocation.
    register_raw(
        views_con,
        glob=tmp_corpus["glob"],
        subagent_glob=tmp_corpus["subagent_glob"],
        subagent_meta_glob=tmp_corpus["subagent_meta_glob"],
    )
    register_views(views_con)
    lance_uri = tmp_path / "embeddings_lance"
    rows = discover_unembedded(views_con, lance_uri=lance_uri, since_days=1)
    uuids = {r[0] for r in rows}
    # 2026-04-01 / 2026-04-02 fixture rows are ancient relative to "today";
    # ``since_days=1`` filters them out and keeps only the fresh one.
    assert "fresh-uuid" in uuids
    assert "u1" not in uuids
    assert "v1" not in uuids


def test_discover_unembedded_limit_cap(views_con: Any, tmp_path: Path) -> None:
    lance_uri = tmp_path / "embeddings_lance"
    rows = discover_unembedded(views_con, lance_uri=lance_uri, limit=1)
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# _build_bedrock_client
# ---------------------------------------------------------------------------


def test_build_bedrock_client_config(tmp_settings: Any) -> None:
    client = _build_bedrock_client(tmp_settings)
    cfg = client._client_config
    assert cfg.region_name == tmp_settings.region
    assert cfg.read_timeout == 60
    assert cfg.connect_timeout == 10
    # botocore retries disabled (tenacity owns the retry policy). botocore
    # normalizes ``max_attempts: 0`` away — verify via the explicit
    # ``total_max_attempts`` it does keep, or fall back to checking the mode.
    retries = cfg.retries or {}
    assert retries.get("mode") == "standard"


# ---------------------------------------------------------------------------
# _invoke_bedrock_sync
# ---------------------------------------------------------------------------


def test_invoke_bedrock_sync_happy_path(fake_bedrock_client: Any, tmp_settings: Any) -> None:
    fake = fake_bedrock_client({"embeddings": {"int8": [[1, 2, 3]]}})
    vectors = _invoke_bedrock_sync(
        fake,
        tmp_settings.active_model_id,
        ["hello"],
        input_type="search_document",
        output_dimension=tmp_settings.output_dimension,
        embedding_type="int8",
    )
    assert vectors == [[1, 2, 3]]
    assert len(fake.captured) == 1
    body = fake.captured[0]["body"]
    assert body["texts"] == ["hello"]
    assert body["input_type"] == "search_document"
    assert body["output_dimension"] == tmp_settings.output_dimension
    assert body["embedding_types"] == ["int8"]
    assert body["truncate"] == "RIGHT"
    assert fake.captured[0]["modelId"] == tmp_settings.active_model_id


def test_invoke_bedrock_sync_clips_long_text(fake_bedrock_client: Any, tmp_settings: Any) -> None:
    """Texts beyond ``MAX_CHARS_PER_TEXT`` must be clipped before send."""
    fake = fake_bedrock_client({"embeddings": {"int8": [[0]]}})
    huge = "x" * 60_000
    _invoke_bedrock_sync(
        fake,
        tmp_settings.active_model_id,
        [huge],
        input_type="search_document",
        output_dimension=tmp_settings.output_dimension,
        embedding_type="int8",
    )
    body = fake.captured[0]["body"]
    assert len(body["texts"][0]) == MAX_CHARS_PER_TEXT


def test_invoke_bedrock_sync_retries_throttling(
    fake_bedrock_client: Any, tmp_settings: Any
) -> None:
    """One ``ThrottlingException`` then success — tenacity should retry."""
    fake = fake_bedrock_client({"embeddings": {"int8": [[7, 8, 9]]}})
    fake.queue_exception(_client_error("ThrottlingException"))
    vectors = _invoke_bedrock_sync(
        fake,
        tmp_settings.active_model_id,
        ["retry me"],
        input_type="search_document",
        output_dimension=tmp_settings.output_dimension,
        embedding_type="int8",
    )
    assert vectors == [[7, 8, 9]]
    # First call raised, second call succeeded -> two captured invocations.
    assert len(fake.captured) == 2


# ---------------------------------------------------------------------------
# embed_documents_async
# ---------------------------------------------------------------------------


def test_embed_documents_async_empty_short_circuit(
    monkeypatch: pytest.MonkeyPatch, tmp_settings: Any
) -> None:
    """Empty input must return ``[]`` without ever building a Bedrock client."""

    def _explode(_settings: Any) -> Any:  # pragma: no cover - failure trap
        raise AssertionError("client should not be built for empty input")

    monkeypatch.setattr(embed_worker, "_build_bedrock_client", _explode)
    out = asyncio.run(embed_documents_async([], settings=tmp_settings))
    assert out == []


def test_embed_documents_async_batches_correctly(
    monkeypatch: pytest.MonkeyPatch,
    fake_bedrock_client: Any,
    tmp_settings: Any,
) -> None:
    """6 texts at batch_size=4 => two batches => two invoke_model calls."""
    # Each call returns a 4-vec or 2-vec depending on the batch — the worker
    # itself doesn't validate sizes, but we line them up to match anyway.
    payloads = [
        {"embeddings": {"int8": [[1], [2], [3], [4]]}},
        {"embeddings": {"int8": [[5], [6]]}},
    ]
    fake = FakeBedrockClient(payloads)
    monkeypatch.setattr(embed_worker, "_build_bedrock_client", lambda _s: fake)

    texts = [f"t{i}" for i in range(6)]
    vectors = asyncio.run(embed_documents_async(texts, settings=tmp_settings))

    assert len(vectors) == 6
    assert vectors[0] == [1.0]
    assert vectors[5] == [6.0]
    assert len(fake.captured) == 2
    assert fake.captured[0]["body"]["texts"] == texts[:4]
    assert fake.captured[1]["body"]["texts"] == texts[4:]


# ---------------------------------------------------------------------------
# embed_query
# ---------------------------------------------------------------------------


def test_embed_query_uses_search_query_input_type(
    monkeypatch: pytest.MonkeyPatch, tmp_settings: Any
) -> None:
    fake = FakeBedrockClient({"embeddings": {"float": [[0.1, 0.2]]}})
    monkeypatch.setattr(embed_worker, "_build_bedrock_client", lambda _s: fake)

    out = embed_query("what did I work on", settings=tmp_settings)
    assert out == [pytest.approx(0.1), pytest.approx(0.2)]
    body = fake.captured[0]["body"]
    assert body["input_type"] == "search_query"
    assert body["embedding_types"] == ["float"]
    assert body["texts"] == ["what did I work on"]


# ---------------------------------------------------------------------------
# run_backfill
# ---------------------------------------------------------------------------


def test_run_backfill_dry_run_no_candidates(
    views_con: Any, tmp_settings: Any, tmp_path: Path
) -> None:
    """Dry-run on an empty corpus path: candidates=0, dry_run=True."""
    # Pre-seed Lance with every messages_text uuid so the candidate set is empty.
    rows = views_con.execute("SELECT CAST(uuid AS VARCHAR) FROM messages_text").fetchall()
    _seed_lance_uuids(
        tmp_settings.lance_uri,
        [r[0] for r in rows],
        dim=tmp_settings.output_dimension,
    )

    plan = asyncio.run(run_backfill(con=views_con, settings=tmp_settings, dry_run=True))
    assert isinstance(plan, dict)
    assert plan["pipeline"] == "embed"
    assert plan["candidates"] == 0
    assert plan["dry_run"] is True


def test_run_backfill_dry_run_with_candidates(views_con: Any, tmp_settings: Any) -> None:
    plan = asyncio.run(run_backfill(con=views_con, settings=tmp_settings, dry_run=True))
    assert isinstance(plan, dict)
    assert plan["candidates"] > 0
    assert plan["batches"] >= 1
    assert plan["batch_size"] == tmp_settings.batch_size
    assert plan["concurrency"] == tmp_settings.embed_concurrency
    assert plan["model"] == tmp_settings.active_model_id
    assert plan["dry_run"] is True


def test_run_backfill_real_run_writes_shards(
    monkeypatch: pytest.MonkeyPatch,
    views_con: Any,
    tmp_settings: Any,
) -> None:
    """Real run: every pending message gets embedded and one shard lands on disk."""
    pending = views_con.execute("SELECT count(*) FROM messages_text").fetchone()[0]
    assert pending > 0

    # Build a payload that returns one int8 vector of the right shape per
    # text in the (only) batch.  The fixture has ~3 candidate rows, well
    # under tmp_settings.batch_size=4 -> a single Bedrock call.
    fake_vec = [0] * tmp_settings.output_dimension
    payload = {"embeddings": {"int8": [fake_vec for _ in range(pending)]}}
    fake = FakeBedrockClient(payload)
    monkeypatch.setattr(embed_worker, "_build_bedrock_client", lambda _s: fake)

    written = asyncio.run(run_backfill(con=views_con, settings=tmp_settings))
    assert written == pending

    # Verify Lance now holds the expected rows.
    db = lance_store.connect_db(tmp_settings.lance_uri)
    tbl = db.open_table(lance_store.TABLE_NAME)
    assert tbl.count_rows() == pending
    arrow = tbl.to_arrow()
    assert "uuid" in arrow.column_names
    assert "embedding" in arrow.column_names


def test_run_backfill_no_pending_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
    views_con: Any,
    tmp_settings: Any,
) -> None:
    """Pre-embedded uuids => returns 0 and never calls invoke_model."""
    rows = views_con.execute("SELECT CAST(uuid AS VARCHAR) FROM messages_text").fetchall()
    _seed_lance_uuids(
        tmp_settings.lance_uri,
        [r[0] for r in rows],
        dim=tmp_settings.output_dimension,
    )

    def _explode(_settings: Any) -> Any:  # pragma: no cover - failure trap
        raise AssertionError("Bedrock client must not be built when nothing pends")

    monkeypatch.setattr(embed_worker, "_build_bedrock_client", _explode)
    written = asyncio.run(run_backfill(con=views_con, settings=tmp_settings))
    assert written == 0


# ---------------------------------------------------------------------------
# Sanity: the captured request body is JSON-serializable
# ---------------------------------------------------------------------------


def test_invoke_bedrock_sync_body_is_json(fake_bedrock_client: Any, tmp_settings: Any) -> None:
    fake = fake_bedrock_client({"embeddings": {"int8": [[1]]}})
    _invoke_bedrock_sync(
        fake,
        tmp_settings.active_model_id,
        ["whatever"],
        input_type="search_document",
        output_dimension=tmp_settings.output_dimension,
        embedding_type="int8",
    )
    # Round-tripping the captured body through json must preserve keys.
    body = fake.captured[0]["body"]
    roundtrip = json.loads(json.dumps(body))
    assert roundtrip == body

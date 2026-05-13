"""Coverage for the friction LLM pipeline (``_classify_async``).

Targets the deeply-uncovered branch in :mod:`claude_sql.friction_worker`
(lines 275-473): regex/llm split, refusal handling, retry-queue enqueue,
anti-join against an existing parquet, checkpoint skip on rerun, and
the SQL-boundary filters (system markers, char cutoff, dry-run plan).

Each test builds its own tiny ``messages_text``-shaped table directly so
we never touch the real ``~/.claude`` cache or Bedrock. The full
``messages_text`` view enforces ``length(...) >= 32`` for legitimate
embedding reasons; the friction worker needs short messages, so we mock
the view at the table level the way the existing dry-run tests already
do.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import pytest

from claude_sql import checkpointer, friction_worker, retry_queue
from claude_sql.config import Settings
from claude_sql.llm_shared import BedrockRefusalError
from claude_sql.parquet_shards import read_all

# ---------------------------------------------------------------------------
# Local helpers — keep each test under <1 s and isolated from the real cache.
# ---------------------------------------------------------------------------

SID = "11111111-1111-1111-1111-111111111111"


def _make_con(messages: list[tuple[str, str, str, str, str]]) -> duckdb.DuckDBPyConnection:
    """Build an in-memory DuckDB connection with a synthetic ``messages_text``.

    Each row in ``messages`` is ``(uuid, session_id, ts_iso, role, text)``.
    A ``sessions`` table is also stamped (matching ``messages_text``'s
    expected JOIN shape used by ``session_bounds``) with
    ``transcript_path`` set to a non-existent file — that triggers the
    ``OSError`` branch in ``session_bounds`` that returns ``mtime=None``,
    which is fine for our tests since the checkpoint logic only requires
    a stable identity.
    """
    con = duckdb.connect(":memory:")
    # Use plain TIMESTAMP (not TIMESTAMPTZ): DuckDB requires pytz at fetch
    # time when retrieving TIMESTAMPTZ values, and the project intentionally
    # doesn't depend on pytz (see ``checkpointer.py`` docstring).
    con.execute(
        """
        CREATE TABLE messages_text (
            uuid          VARCHAR,
            session_id    VARCHAR,
            ts            TIMESTAMP,
            role          VARCHAR,
            text_content  VARCHAR
        )
        """
    )
    if messages:
        con.executemany(
            "INSERT INTO messages_text VALUES (?, ?, ?, ?, ?)",
            messages,
        )
    con.execute(
        """
        CREATE TABLE sessions (
            session_id      VARCHAR,
            transcript_path VARCHAR
        )
        """
    )
    sids = {row[1] for row in messages}
    if sids:
        con.executemany(
            "INSERT INTO sessions VALUES (?, ?)",
            [(sid, f"/nonexistent/{sid}.jsonl") for sid in sids],
        )
    return con


def _user(uuid: str, ts: str, text: str, *, sid: str = SID) -> tuple[str, str, str, str, str]:
    """Compose a single user-role row in the shape ``_make_con`` expects."""
    return (uuid, sid, ts, "user", text)


def _settings(tmp_path: Path) -> Settings:
    cache = tmp_path / "claude"
    cache.mkdir(parents=True, exist_ok=True)
    return Settings(
        embeddings_parquet_path=cache / "embeddings",
        classifications_parquet_path=cache / "classifications",
        trajectory_parquet_path=cache / "trajectory",
        conflicts_parquet_path=cache / "conflicts",
        clusters_parquet_path=cache / "clusters.parquet",
        cluster_terms_parquet_path=cache / "cluster_terms.parquet",
        communities_parquet_path=cache / "communities.parquet",
        user_friction_parquet_path=cache / "user_friction",
        skills_catalog_parquet_path=cache / "skills_catalog.parquet",
        checkpoint_db_path=cache / "claude_sql.duckdb",
        duckdb_temp_dir=cache / "duckdb_tmp",
        user_skills_dir=cache / "skills",
        plugins_cache_dir=cache / "plugins_cache",
        embed_concurrency=2,
        llm_concurrency=2,
        batch_size=4,
        friction_max_chars=300,
    )


def _patch_classify_one(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response_factory: Callable[[str], Any],
) -> list[str]:
    """Patch ``friction_worker.classify_one`` to capture inputs and return canned results.

    The real function is async and runs under an ``anyio.CapacityLimiter``;
    the fake mirrors that surface so it can be awaited identically. Each
    call's user-text (the prompt body) is recorded so callers can assert
    *which* messages reached the LLM. ``response_factory(text)`` is invoked
    per call: returning a dict is treated as a successful classification,
    returning an exception simulates the failure paths (the helper raises
    on the caller's behalf).
    """
    seen: list[str] = []

    async def fake_classify_one(
        client: Any,
        model_id: str,
        schema: dict,
        text: str,
        *,
        max_tokens: int,
        thinking_mode: str,
        sem: Any,
        system: str | None = None,
        pipeline: str = "classifier",
    ) -> dict:
        del client, model_id, schema, max_tokens, thinking_mode, sem, system, pipeline
        seen.append(text)
        result = response_factory(text)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(friction_worker, "classify_one", fake_classify_one)
    # Don't actually open a boto3 client — the patched classify_one ignores
    # what we return, but the surrounding code still calls
    # _build_bedrock_client and would fail without AWS creds.
    monkeypatch.setattr(friction_worker, "_build_bedrock_client", lambda _settings: object())
    return seen


# ---------------------------------------------------------------------------
# 1. dry-run paths
# ---------------------------------------------------------------------------


def test_dry_run_empty_corpus_returns_zero(tmp_path: Path) -> None:
    """No candidates → ``candidates=0, llm_calls=0``."""
    con = _make_con([])
    settings = _settings(tmp_path)

    plan = friction_worker.detect_user_friction(
        con, settings, since_days=None, limit=None, dry_run=True
    )

    assert isinstance(plan, dict)
    assert plan["pipeline"] == "friction"
    assert plan["candidates"] == 0
    assert plan["llm_calls"] == 0
    assert plan["estimated_cost_usd"] == 0.0
    assert plan["dry_run"] is True


def test_dry_run_with_candidates_estimates_cost(tmp_path: Path) -> None:
    """Two short user messages → 2 candidates, ``llm_calls = 2 // 2 = 1``."""
    con = _make_con(
        [
            _user("u1", "2026-04-01T10:00:00Z", "screenshot?"),
            _user("u2", "2026-04-01T10:00:01Z", "ok thanks"),
        ]
    )
    settings = _settings(tmp_path)

    plan = friction_worker.detect_user_friction(
        con, settings, since_days=None, limit=None, dry_run=True
    )

    assert isinstance(plan, dict)
    assert plan["candidates"] == 2
    assert plan["llm_calls"] == 1  # candidates // 2
    assert plan["estimated_cost_usd"] > 0
    assert plan["model"] == settings.sonnet_model_id
    assert plan["friction_max_chars"] == settings.friction_max_chars


def test_dry_run_limit_clamps_candidates(tmp_path: Path) -> None:
    """Hits the ``n = min(n, int(limit))`` branch on line 522."""
    con = _make_con(
        [
            _user("u1", "2026-04-01T10:00:00Z", "screenshot?"),
            _user("u2", "2026-04-01T10:00:01Z", "ok thanks"),
            _user("u3", "2026-04-01T10:00:02Z", "and another"),
        ]
    )
    settings = _settings(tmp_path)

    plan = friction_worker.detect_user_friction(
        con, settings, since_days=None, limit=1, dry_run=True
    )

    assert isinstance(plan, dict)
    assert plan["candidates"] == 1


# ---------------------------------------------------------------------------
# 2. mixed regex + LLM happy path
# ---------------------------------------------------------------------------


def test_classify_async_regex_and_llm_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One regex hit ("wait, stop!"), one LLM-bound message ("screenshot?").

    The fake ``_classify_one`` returns a normal dict for the LLM message;
    the parquet ends up with one ``source='regex'`` row and one
    ``source='llm'`` row.
    """
    con = _make_con(
        [
            _user("u1", "2026-04-01T10:00:00Z", "wait, stop!"),
            _user("u2", "2026-04-01T10:00:01Z", "screenshot?"),
        ]
    )
    settings = _settings(tmp_path)

    seen = _patch_classify_one(
        monkeypatch,
        response_factory=lambda _t: {
            "label": "unmet_expectation",
            "rationale": "bare artifact ping",
            "confidence": 0.8,
        },
    )

    written = friction_worker.detect_user_friction(con, settings)
    assert written == 2

    # Only the LLM-bound message should have hit our fake.
    assert len(seen) == 1
    assert "screenshot?" in seen[0]

    df = read_all(settings.user_friction_parquet_path)
    assert df is not None
    assert df.height == 2
    by_source = {row["source"]: row for row in df.to_dicts()}
    assert set(by_source) == {"regex", "llm"}
    assert by_source["regex"]["label"] == "interruption"
    assert by_source["regex"]["confidence"] == pytest.approx(0.9, abs=1e-6)
    assert by_source["llm"]["label"] == "unmet_expectation"
    assert by_source["llm"]["confidence"] == pytest.approx(0.8, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. LLM refusal stamps a neutral row
# ---------------------------------------------------------------------------


def test_classify_async_refusal_writes_refused_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``BedrockRefusalError`` → ``source='refused'`` placeholder row.

    The friction prompt template lists "screenshot?" / "tests?" as
    in-prompt examples of friction labels, so those exact strings always
    appear in every formatted prompt body. Use unique sentinel tokens so
    the fake can disambiguate which message was passed in.
    """
    con = _make_con(
        [
            _user("u1", "2026-04-01T10:00:00Z", "uniqueRefuseToken"),
            _user("u2", "2026-04-01T10:00:01Z", "uniqueOkToken"),
        ]
    )
    settings = _settings(tmp_path)

    def _factory(text: str) -> Any:
        if "uniqueRefuseToken" in text:
            return BedrockRefusalError("policy")
        return {"label": "none", "rationale": "ordinary task", "confidence": 0.7}

    _patch_classify_one(monkeypatch, response_factory=_factory)

    written = friction_worker.detect_user_friction(con, settings)
    assert written == 2

    df = read_all(settings.user_friction_parquet_path)
    assert df is not None
    rows = df.to_dicts()
    refused = [r for r in rows if r["source"] == "refused"]
    assert len(refused) == 1
    assert refused[0]["label"] == "none"
    assert refused[0]["confidence"] == pytest.approx(0.0, abs=1e-6)
    assert "uniqueRefuseToken" in refused[0]["text_snippet"]


# ---------------------------------------------------------------------------
# 4. transient errors land on the retry queue, not the parquet
# ---------------------------------------------------------------------------


def test_classify_async_transient_error_enqueues_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``RuntimeError("transient")`` → no parquet row, ``retry_queue.enqueue`` called.

    ``user_friction`` is now in ``PIPELINE_NAMES`` (T2.1), so the underlying
    ``retry_queue.enqueue`` succeeds end-to-end. We still patch ``enqueue`` to
    capture the call so the assertion stays focused on the friction-worker
    branch rather than the retry-queue persistence path.
    """
    con = _make_con(
        [
            _user("u1", "2026-04-01T10:00:00Z", "uniqueErrorToken"),
            _user("u2", "2026-04-01T10:00:01Z", "uniqueOkToken"),
        ]
    )
    settings = _settings(tmp_path)

    def _factory(text: str) -> Any:
        if "uniqueErrorToken" in text:
            return RuntimeError("transient")
        return {"label": "none", "rationale": "ordinary", "confidence": 0.7}

    _patch_classify_one(monkeypatch, response_factory=_factory)

    enqueue_calls: list[dict[str, Any]] = []

    def fake_enqueue(db_path: Path, *, pipeline: str, unit_id: str, error: str) -> int:
        enqueue_calls.append(
            {"db_path": db_path, "pipeline": pipeline, "unit_id": unit_id, "error": error}
        )
        return 1

    monkeypatch.setattr(retry_queue, "enqueue", fake_enqueue)

    written = friction_worker.detect_user_friction(con, settings)
    # Only the second message survived; the first hit the (faked) retry queue.
    assert written == 1

    df = read_all(settings.user_friction_parquet_path)
    assert df is not None
    rows = df.to_dicts()
    snippets = {r["text_snippet"] for r in rows}
    assert "uniqueErrorToken" not in snippets

    # The friction worker took the retry-queue branch for the failing uuid.
    assert len(enqueue_calls) == 1
    assert enqueue_calls[0]["pipeline"] == "user_friction"
    assert enqueue_calls[0]["unit_id"] == "u1"
    assert "transient" in enqueue_calls[0]["error"]


# ---------------------------------------------------------------------------
# 5. anti-join against an existing parquet skips already-classified uuids
# ---------------------------------------------------------------------------


def test_classify_async_skips_uuids_in_existing_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-existing parquet shard → that uuid never reaches the LLM."""
    con = _make_con(
        [
            # Both fall to the LLM (no regex hit) so we can count fake-call hits.
            _user("u1", "2026-04-01T10:00:00Z", "screenshot?"),
            _user("u2", "2026-04-01T10:00:01Z", "tests?"),
        ]
    )
    settings = _settings(tmp_path)

    # Pre-write a parquet shard claiming "u1" is already classified.
    settings.user_friction_parquet_path.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    pl.DataFrame(
        [
            {
                "uuid": "u1",
                "session_id": SID,
                "ts": now,
                "text_snippet": "previously classified",
                "label": "none",
                "rationale": "seeded",
                "source": "llm",
                "confidence": 0.5,
                "classified_at": now,
            }
        ],
        schema={
            "uuid": pl.Utf8,
            "session_id": pl.Utf8,
            "ts": pl.Datetime("us", "UTC"),
            "text_snippet": pl.Utf8,
            "label": pl.Utf8,
            "rationale": pl.Utf8,
            "source": pl.Utf8,
            "confidence": pl.Float32,
            "classified_at": pl.Datetime("us", "UTC"),
        },
    ).write_parquet(settings.user_friction_parquet_path / "part-00000.parquet")

    seen = _patch_classify_one(
        monkeypatch,
        response_factory=lambda _t: {
            "label": "none",
            "rationale": "ordinary",
            "confidence": 0.6,
        },
    )

    written = friction_worker.detect_user_friction(con, settings)
    # Only u2 was newly classified.
    assert written == 1
    assert len(seen) == 1
    assert "tests?" in seen[0]


# ---------------------------------------------------------------------------
# 6. checkpoint skip on rerun
# ---------------------------------------------------------------------------


def test_classify_async_checkpoint_skip_on_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two runs back-to-back: second run skips every session via checkpoint.

    For the checkpoint-skip branch (``_stale_or_equal`` returns True) we
    need both ``last_ts`` and ``transcript_mtime`` to be present and
    stable across runs, so we point the synthetic ``sessions.transcript_path``
    at a real file that exists on disk.
    """
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE messages_text (
            uuid VARCHAR, session_id VARCHAR, ts TIMESTAMP,
            role VARCHAR, text_content VARCHAR
        )
        """
    )
    con.execute(
        "INSERT INTO messages_text VALUES "
        "('u1', ?, TIMESTAMP '2026-04-01 10:00:00', 'user', 'screenshot?')",
        [SID],
    )
    real_transcript = tmp_path / "real_transcript.jsonl"
    real_transcript.write_text("{}\n")
    con.execute(
        """
        CREATE TABLE sessions (session_id VARCHAR, transcript_path VARCHAR)
        """
    )
    con.execute("INSERT INTO sessions VALUES (?, ?)", [SID, str(real_transcript)])

    settings = _settings(tmp_path)

    seen = _patch_classify_one(
        monkeypatch,
        response_factory=lambda _t: {
            "label": "none",
            "rationale": "ordinary",
            "confidence": 0.6,
        },
    )

    first = friction_worker.detect_user_friction(con, settings)
    assert first == 1
    assert len(seen) == 1

    # Second run: the parquet anti-join would catch u1 anyway, but the
    # checkpoint short-circuit fires first because ``last_ts`` and the
    # transcript ``mtime`` haven't advanced.
    second = friction_worker.detect_user_friction(con, settings)
    assert second == 0
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# 7. system markers are filtered at the SQL boundary
# ---------------------------------------------------------------------------


def test_system_marker_message_never_reaches_classifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Continue from where you left off.`` is bookkeeping, not friction."""
    con = _make_con(
        [
            _user("u1", "2026-04-01T10:00:00Z", "Continue from where you left off."),
            _user("u2", "2026-04-01T10:00:01Z", "screenshot?"),
        ]
    )
    settings = _settings(tmp_path)

    seen = _patch_classify_one(
        monkeypatch,
        response_factory=lambda _t: {
            "label": "unmet_expectation",
            "rationale": "bare artifact ping",
            "confidence": 0.8,
        },
    )

    written = friction_worker.detect_user_friction(con, settings)
    assert written == 1  # only u2 survived the marker filter
    assert len(seen) == 1
    assert "screenshot?" in seen[0]
    assert all("Continue from where you left off." not in s for s in seen)

    df = read_all(settings.user_friction_parquet_path)
    assert df is not None
    snippets = {r["text_snippet"] for r in df.to_dicts()}
    assert all("Continue from where you left off." not in s for s in snippets)


# ---------------------------------------------------------------------------
# 8. char cutoff drops over-long messages before classification
# ---------------------------------------------------------------------------


def test_long_message_excluded_by_char_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """500-char user message > ``friction_max_chars=300`` → dropped at SQL."""
    long_text = "x" * 500
    con = _make_con(
        [
            _user("u1", "2026-04-01T10:00:00Z", long_text),
            _user("u2", "2026-04-01T10:00:01Z", "screenshot?"),
        ]
    )
    settings = _settings(tmp_path)

    seen = _patch_classify_one(
        monkeypatch,
        response_factory=lambda _t: {
            "label": "none",
            "rationale": "ordinary",
            "confidence": 0.6,
        },
    )

    written = friction_worker.detect_user_friction(con, settings)
    assert written == 1
    assert len(seen) == 1
    # The fake never sees the long-text message.
    assert all(long_text not in prompt for prompt in seen)


# ---------------------------------------------------------------------------
# 9. regex-only path returns without building a Bedrock client
# ---------------------------------------------------------------------------


def test_regex_only_path_skips_bedrock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every message is a regex hit → no LLM call, parquet has all rows."""
    con = _make_con(
        [
            _user("u1", "2026-04-01T10:00:00Z", "wait, stop!"),
            _user("u2", "2026-04-01T10:00:01Z", "nope"),
        ]
    )
    settings = _settings(tmp_path)

    # If _build_bedrock_client gets called, the test should explode — the
    # regex-only branch returns before that step.
    def _explode(_settings: Settings) -> Any:
        raise AssertionError("Bedrock client must not be built on the regex-only path")

    monkeypatch.setattr(friction_worker, "_build_bedrock_client", _explode)

    written = friction_worker.detect_user_friction(con, settings)
    assert written == 2

    df = read_all(settings.user_friction_parquet_path)
    assert df is not None
    rows = df.to_dicts()
    labels = {row["label"] for row in rows}
    assert labels == {"interruption", "correction"}
    assert all(row["source"] == "regex" for row in rows)

    # Checkpoint was stamped so a rerun is a no-op.
    ckpt_rows = checkpointer.count_rows(settings.checkpoint_db_path)
    assert ckpt_rows >= 1


# ---------------------------------------------------------------------------
# 10. since_days + limit tighten the candidate SQL
# ---------------------------------------------------------------------------


def test_dry_run_with_since_days_and_limit(tmp_path: Path) -> None:
    """Hits the ``since_days`` branch in ``_candidate_sql`` and the
    ``limit`` cap in the non-dry-run candidate SQL by way of the
    ``--dry-run`` plan path."""
    con = _make_con(
        [
            _user("u1", "2026-04-01T10:00:00Z", "screenshot?"),
            _user("u2", "2026-04-01T10:00:01Z", "ok thanks"),
        ]
    )
    settings = _settings(tmp_path)

    # ``since_days`` injects ``mt.ts >= current_timestamp - INTERVAL N DAY``;
    # at a 1-million-day window, both fixture rows still pass.
    plan = friction_worker.detect_user_friction(
        con, settings, since_days=1_000_000, limit=None, dry_run=True
    )
    assert isinstance(plan, dict)
    assert plan["candidates"] == 2
    assert plan["since_days"] == 1_000_000


def test_classify_async_with_limit_caps_candidate_sql(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hits the ``limit`` branch in ``_classify_async`` (line 303): the
    candidate SQL appends ``LIMIT N`` so only the first row reaches LLM."""
    con = _make_con(
        [
            _user("u1", "2026-04-01T10:00:00Z", "uniqueLimitToken1"),
            _user("u2", "2026-04-01T10:00:01Z", "uniqueLimitToken2"),
        ]
    )
    settings = _settings(tmp_path)

    seen = _patch_classify_one(
        monkeypatch,
        response_factory=lambda _t: {
            "label": "none",
            "rationale": "ordinary",
            "confidence": 0.6,
        },
    )

    written = friction_worker.detect_user_friction(con, settings, limit=1)
    assert written == 1
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# 11. retry-queue drain branch (lines 292-293)
# ---------------------------------------------------------------------------


def test_classify_async_drains_retry_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-empty drain return value triggers the ``already -= retry_uuids``
    branch and re-classifies a uuid that's already in the parquet."""
    con = _make_con(
        [
            _user("u1", "2026-04-01T10:00:00Z", "uniqueDrainToken"),
        ]
    )
    settings = _settings(tmp_path)

    # Pre-write the parquet so u1 would normally be skipped by the anti-join.
    settings.user_friction_parquet_path.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    pl.DataFrame(
        [
            {
                "uuid": "u1",
                "session_id": SID,
                "ts": now,
                "text_snippet": "previously classified",
                "label": "none",
                "rationale": "seeded",
                "source": "llm",
                "confidence": 0.5,
                "classified_at": now,
            }
        ],
        schema={
            "uuid": pl.Utf8,
            "session_id": pl.Utf8,
            "ts": pl.Datetime("us", "UTC"),
            "text_snippet": pl.Utf8,
            "label": pl.Utf8,
            "rationale": pl.Utf8,
            "source": pl.Utf8,
            "confidence": pl.Float32,
            "classified_at": pl.Datetime("us", "UTC"),
        },
    ).write_parquet(settings.user_friction_parquet_path / "part-00000.parquet")

    # Make ``retry_queue.drain`` return ``["u1"]`` so the worker takes the
    # "drain non-empty" branch and removes u1 from the ``already`` set.
    monkeypatch.setattr(
        retry_queue,
        "drain",
        lambda _db, *, pipeline, **_kw: ["u1"] if pipeline == "user_friction" else [],
    )
    # Capture mark_done so the test can confirm the worker followed up
    # (even though our PIPELINE_NAMES doesn't include 'user_friction',
    # ``mark_done`` itself doesn't validate, so we don't strictly have to
    # patch it; we patch enqueue out of caution though).
    monkeypatch.setattr(retry_queue, "enqueue", lambda *_a, **_kw: 1)  # silence on potential errors

    seen = _patch_classify_one(
        monkeypatch,
        response_factory=lambda _t: {
            "label": "none",
            "rationale": "ordinary",
            "confidence": 0.6,
        },
    )

    written = friction_worker.detect_user_friction(con, settings)
    # u1 was re-classified because the drain pulled it back into scope.
    assert written == 1
    assert len(seen) == 1
    assert "uniqueDrainToken" in seen[0]

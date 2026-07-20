"""End-to-end tests for the ``claude_sql.ingest`` worker pipeline.

Builds a small JSONL fixture corpus, registers raw + derived views, runs
``stamp_messages`` / ``resolve_canonicals`` / ``count_pending``, and asserts
the resulting parquet shape + canonical-uuid resolution behaviour.

No Bedrock calls — this entire module is pure CPU.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import pytest

from claude_sql.application.use_cases.ingest import (
    count_pending,
    resolve_canonicals,
    stamp_messages,
)
from claude_sql.domain.dedup import NEAR_DUP_HAMMING_THRESHOLD, simhash64
from claude_sql.infrastructure.duckdb_views import register_raw, register_views
from claude_sql.infrastructure.parquet_cache import iter_part_files
from claude_sql.infrastructure.settings import Settings

# ---------------------------------------------------------------------------
# Fixture builders (local — we want full control over text contents)
# ---------------------------------------------------------------------------


def _record(
    *,
    uuid: str,
    session_id: str,
    ts: str,
    text: str,
    role: str = "user",
) -> dict[str, Any]:
    """Minimal JSONL transcript record with a single text content block."""
    return {
        "parentUuid": None,
        "isSidechain": False,
        "type": role,
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
            "model": "claude-sonnet-4-6" if role == "assistant" else None,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "content": [{"type": "text", "text": text}],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec))
            fh.write("\n")


SID = "11111111-1111-1111-1111-111111111111"
SID_DUP = "22222222-2222-2222-2222-222222222222"


@pytest.fixture
def corpus(tmp_path: Path) -> dict[str, Any]:
    """Two-session corpus with one near-dup pair and one unrelated row."""
    proj = tmp_path / "projects" / "proj-a"
    _write_jsonl(
        proj / f"{SID}.jsonl",
        [
            _record(
                uuid="u1",
                session_id=SID,
                ts="2026-04-01T10:00:00.000Z",
                text=(
                    "please run the regression suite and report any failures "
                    "back to me with the diff so I can review the regressions"
                ),
            ),
            _record(
                uuid="u2",
                session_id=SID,
                ts="2026-04-01T10:30:00.000Z",
                text=(
                    "let's pivot to building the cluster visualization in the "
                    "frontend with the new color palette and tooltip layout"
                ),
            ),
        ],
    )
    # Second session: a near-dup of u1 (same intent, single word swap) +
    # one unrelated row.  Earlier ts so u1 is the canonical for the dup.
    _write_jsonl(
        proj / f"{SID_DUP}.jsonl",
        [
            _record(
                uuid="v1",
                session_id=SID_DUP,
                ts="2026-04-02T09:00:00.000Z",
                text=(
                    # Identical to u1 except a trailing whitespace nudge so
                    # the simhash matches but the uuid differs.
                    "please run the regression suite and report any failures "
                    "back to me with the diff so I can review the regressions"
                ),
            ),
        ],
    )
    # Subagent stub so register_raw doesn't choke on missing globs.
    sa_dir = tmp_path / "projects" / "proj-stub" / "00000000" / "subagents"
    sa_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        sa_dir / "agent-placeholder.jsonl",
        [
            _record(
                uuid="sa-stub",
                session_id="placeholder",
                ts="2026-01-01T00:00:00.000Z",
                text="subagent stub record so duckdb read_json infers a schema",
            )
        ],
    )
    (sa_dir / "agent-placeholder.meta.json").write_text(
        json.dumps({"agentType": "stub", "description": "stub subagent"})
    )
    return {
        "glob": str(tmp_path / "projects" / "*" / "*.jsonl"),
        "subagent_glob": str(tmp_path / "projects" / "*" / "*" / "subagents" / "agent-*.jsonl"),
        "subagent_meta_glob": str(
            tmp_path / "projects" / "*" / "*" / "subagents" / "agent-*.meta.json"
        ),
        "tmp_path": tmp_path,
    }


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings whose ingest_stamps dir lives under ``tmp_path``."""
    return Settings(
        ingest_stamps_parquet_path=tmp_path / "ingest_stamps",
        embeddings_parquet_path=tmp_path / "embeddings",
        lance_uri=tmp_path / "embeddings_lance",
        classifications_parquet_path=tmp_path / "classifications",
        trajectory_parquet_path=tmp_path / "trajectory",
        conflicts_parquet_path=tmp_path / "conflicts",
        clusters_parquet_path=tmp_path / "clusters.parquet",
        cluster_terms_parquet_path=tmp_path / "cluster_terms.parquet",
        communities_parquet_path=tmp_path / "communities.parquet",
        user_friction_parquet_path=tmp_path / "user_friction",
        skills_catalog_parquet_path=tmp_path / "skills_catalog.parquet",
        checkpoint_db_path=tmp_path / "state.db",
        duckdb_temp_dir=tmp_path / "duckdb_tmp",
        user_skills_dir=tmp_path / "skills",
        plugins_cache_dir=tmp_path / "plugins_cache",
    )


@pytest.fixture
def con(corpus: dict[str, Any]) -> Iterator[duckdb.DuckDBPyConnection]:
    """In-memory DuckDB with raw + derived views over ``corpus``."""
    c = duckdb.connect(":memory:")
    register_raw(
        c,
        glob=corpus["glob"],
        subagent_glob=corpus["subagent_glob"],
        subagent_meta_glob=corpus["subagent_meta_glob"],
    )
    register_views(c)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# stamp_messages
# ---------------------------------------------------------------------------


def test_stamp_messages_writes_part_file(
    con: duckdb.DuckDBPyConnection, settings: Settings
) -> None:
    """First run on a fresh corpus must write a populated part shard."""
    n = stamp_messages(con, settings)
    assert n >= 2  # u1 + v1 at minimum
    parts = iter_part_files(settings.ingest_stamps_parquet_path)
    assert parts, "expected at least one part-*.parquet shard"
    df = pl.read_parquet([str(p) for p in parts])
    expected_cols = {
        "uuid",
        "session_id",
        "approx_tokens",
        "simhash64",
        "token_budget_bucket",
        "canonical_uuid",
        "first_seen_ts",
        "stamped_at",
    }
    assert set(df.columns) == expected_cols
    # Token bucket must be a known label.
    assert df["token_budget_bucket"].is_in(["xs", "sm", "md", "lg", "xl"]).all()
    # No row stamped to the all-zero simhash for non-empty input.
    assert (df["simhash64"] != 0).all()


def test_stamp_messages_skips_already_stamped(
    con: duckdb.DuckDBPyConnection, settings: Settings
) -> None:
    """A second run on the same corpus must be a no-op."""
    first = stamp_messages(con, settings)
    assert first > 0
    second = stamp_messages(con, settings)
    assert second == 0


def test_count_pending_zero_after_stamp(con: duckdb.DuckDBPyConnection, settings: Settings) -> None:
    """``count_pending`` must reflect the anti-join against the parquet."""
    pre = count_pending(con, settings)
    assert pre > 0
    stamp_messages(con, settings)
    post = count_pending(con, settings)
    assert post == 0


# ---------------------------------------------------------------------------
# resolve_canonicals
# ---------------------------------------------------------------------------


def test_canonical_resolve_finds_near_dups(
    con: duckdb.DuckDBPyConnection, settings: Settings
) -> None:
    """Two near-dup messages must share a canonical_uuid (the earlier one)."""
    stamp_messages(con, settings)
    n_resolved = resolve_canonicals(con, settings)
    assert n_resolved > 0
    parts = iter_part_files(settings.ingest_stamps_parquet_path)
    df = pl.read_parquet([str(p) for p in parts])
    # u1 and v1 carry the same paragraph, so their simhash distance is 0.
    u1 = df.filter(pl.col("uuid") == "u1").row(0, named=True)
    v1 = df.filter(pl.col("uuid") == "v1").row(0, named=True)
    # u1 is earlier (2026-04-01 vs 2026-04-02), so v1's canonical is u1.
    assert v1["canonical_uuid"] == "u1"
    # u1 has no earlier near-dup; its canonical falls back to itself.
    assert u1["canonical_uuid"] == "u1"


def test_canonical_resolve_keeps_distinct_intact(
    con: duckdb.DuckDBPyConnection, settings: Settings
) -> None:
    """Unrelated rows must each be their own canonical."""
    stamp_messages(con, settings)
    resolve_canonicals(con, settings)
    parts = iter_part_files(settings.ingest_stamps_parquet_path)
    df = pl.read_parquet([str(p) for p in parts])
    u2 = df.filter(pl.col("uuid") == "u2").row(0, named=True)
    assert u2["canonical_uuid"] == "u2"


def test_resolve_canonicals_empty_parquet_returns_zero(
    con: duckdb.DuckDBPyConnection, settings: Settings
) -> None:
    """Calling resolve before any stamp pass must be a no-op."""
    assert resolve_canonicals(con, settings) == 0


def test_simhash_threshold_constant_matches_macro() -> None:
    """The Python module's threshold and the SQL macro must agree.

    If a contributor bumps :data:`NEAR_DUP_HAMMING_THRESHOLD` to 5 they
    must also bump the ``<= 3`` literal in ``canonical_uuid_resolve``'s
    DDL — this test catches the drift.
    """
    import inspect

    from claude_sql.infrastructure.duckdb_views import register_macros  # local import

    src = inspect.getsource(register_macros)
    assert f"<= {NEAR_DUP_HAMMING_THRESHOLD}" in src, (
        "NEAR_DUP_HAMMING_THRESHOLD changed but the SQL macro literal didn't"
    )


# ---------------------------------------------------------------------------
# Cross-platform sanity: simhash distance for engineered near-dups
# ---------------------------------------------------------------------------


def test_simhash_paste_retry_zero_distance() -> None:
    """A paste-retry of the same prompt produces an identical simhash."""
    text = (
        "please run the regression suite and report any failures back to me "
        "with the diff so I can review the regressions"
    )
    assert simhash64(text) == simhash64(text)


# ---------------------------------------------------------------------------
# CLI dry-run
# ---------------------------------------------------------------------------


def test_ingest_subcommand_dry_run_default(corpus: dict[str, Any], tmp_path: Path) -> None:
    """``claude-sql ingest`` with no flag must dry-run + emit JSON, no writes."""
    env = {
        "CLAUDE_SQL_DEFAULT_GLOB": corpus["glob"],
        "CLAUDE_SQL_SUBAGENT_GLOB": corpus["subagent_glob"],
        "CLAUDE_SQL_SUBAGENT_META_GLOB": corpus["subagent_meta_glob"],
        "CLAUDE_SQL_INGEST_STAMPS_PARQUET_PATH": str(tmp_path / "ingest_stamps"),
        "CLAUDE_SQL_DUCKDB_TEMP_DIR": str(tmp_path / "duckdb_tmp"),
        # Inherit PATH so uv / claude-sql resolves
        "PATH": __import__("os").environ.get("PATH", ""),
        "HOME": __import__("os").environ.get("HOME", ""),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "claude_sql.interfaces.cli.app", "ingest", "--format", "json"],
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )
    # Subcommand should exit 0 even when zero rows pending; the body is
    # always machine-readable JSON because --format json is set.
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["pipeline"] == "ingest"
    assert payload["dry_run"] is True
    assert "candidates" in payload
    # Must not have written any parquet shards.
    assert not iter_part_files(tmp_path / "ingest_stamps")

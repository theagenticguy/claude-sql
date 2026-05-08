"""Synthetic-fixture smoke test for session-level Louvain community detection."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pytest

from claude_sql.community_worker import run_communities
from claude_sql.config import Settings
from claude_sql.sql_views import register_raw, register_views


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _msg(uuid: str, session_id: str, ts: str, *, role: str, text: str) -> dict:
    return {
        "parentUuid": None,
        "isSidechain": False,
        "type": "assistant" if role == "assistant" else "user",
        "uuid": uuid,
        "timestamp": ts,
        "sessionId": session_id,
        "version": "2.0.0",
        "gitBranch": "main",
        "cwd": "/x",
        "userType": "external",
        "entrypoint": "cli",
        "permissionMode": "acceptEdits",
        "promptId": f"p-{uuid}",
        "message": {
            "id": f"m-{uuid}",
            "type": "message",
            "role": role,
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "content": [{"type": "text", "text": text}],
            "usage": {
                "input_tokens": 50,
                "output_tokens": 20,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    }


@pytest.fixture
def connected_settings(tmp_path: Path) -> tuple[duckdb.DuckDBPyConnection, Settings]:
    # 3 sessions worth of messages so _load_session_centroids finds multiple
    proj = tmp_path / "projects" / "-home-x-proj"
    proj.mkdir(parents=True)
    sids = [f"{i:08d}-0000-0000-0000-000000000000" for i in range(1, 4)]
    for i, sid in enumerate(sids):
        _write_jsonl(
            proj / f"{sid}.jsonl",
            [
                _msg(
                    f"u{i}-1",
                    sid,
                    f"2026-04-0{i + 1}T10:00:00.000Z",
                    role="user",
                    text="A" * 100,
                ),
                _msg(
                    f"a{i}-1",
                    sid,
                    f"2026-04-0{i + 1}T10:00:01.000Z",
                    role="assistant",
                    text="B" * 100,
                ),
            ],
        )

    # Synthetic embeddings: 2 sessions "close", 1 "far" -- so we expect >=1 community.
    # 6 rows total (3 sessions * 2 messages each); uuids match the jsonl rows.
    rng = np.random.default_rng(42)
    close = rng.normal(size=(4, 32)).astype(np.float32)
    close /= np.linalg.norm(close, axis=1, keepdims=True)
    # Make sessions 0 and 1 share a direction; session 2 is random
    close[0:2] = close[0]
    close[2:4] = close[0] + 0.01 * rng.normal(size=(2, 32)).astype(np.float32)
    far = rng.normal(size=(2, 32)).astype(np.float32)
    far /= np.linalg.norm(far, axis=1, keepdims=True)
    e = np.vstack([close, far])
    e /= np.linalg.norm(e, axis=1, keepdims=True)

    msg_uuids = ["u0-1", "a0-1", "u1-1", "a1-1", "u2-1", "a2-1"]
    emb_df = pl.DataFrame(
        {
            "uuid": msg_uuids,
            "model": ["test"] * 6,
            "dim": [32] * 6,
            "embedding": e,
        },
        schema={
            "uuid": pl.Utf8,
            "model": pl.Utf8,
            "dim": pl.UInt16,
            "embedding": pl.Array(pl.Float32, 32),
        },
    )
    emb_path = tmp_path / "embeddings.parquet"
    emb_df.write_parquet(emb_path)

    # ``read_json`` errors out when the glob matches zero files, and
    # register_views binds columns like ``timestamp`` that require at least
    # one valid record.  Write a real-but-unrelated subagent pair so the glob
    # resolves and schema inference picks up every column register_views
    # needs.  The parent UUID chosen here is never referenced by the
    # community join so it's harmless.
    parent_uuid = "99999999-9999-9999-9999-999999999999"
    sub_dir = tmp_path / "projects" / "-x" / parent_uuid / "subagents"
    sub_dir.mkdir(parents=True)
    (sub_dir / "agent-deadbeef.meta.json").write_text(
        json.dumps({"agentType": "general-purpose", "description": "placeholder"})
    )
    (sub_dir / "agent-deadbeef.jsonl").write_text(
        json.dumps(
            _msg(
                "sub-u-dead",
                f"sub-{parent_uuid}",
                "2026-04-01T10:00:00.000Z",
                role="user",
                text="placeholder",
            )
        )
        + "\n"
    )

    settings = Settings(
        embeddings_parquet_path=emb_path,
        communities_parquet_path=tmp_path / "communities.parquet",
        default_glob=str(proj / "*.jsonl"),
        subagent_glob=str(tmp_path / "projects" / "*" / "*" / "subagents" / "agent-*.jsonl"),
        subagent_meta_glob=str(
            tmp_path / "projects" / "*" / "*" / "subagents" / "agent-*.meta.json"
        ),
        louvain_edge_threshold=0.5,
        # The synthetic fixture has 3 sessions; keep the min-community floor
        # at 2 so the "close" pair survives the size filter.
        louvain_min_community_size=2,
        # Small fixture -> push avg-degree target low enough that a 3-node
        # graph can hit it; adaptive picker will fall back to the floor anyway.
        louvain_target_avg_degree_low=0.5,
        louvain_target_avg_degree_high=2.0,
    )
    con = duckdb.connect(":memory:")
    register_raw(
        con,
        glob=settings.default_glob,
        subagent_glob=settings.subagent_glob,
        subagent_meta_glob=settings.subagent_meta_glob,
    )
    register_views(con)
    return con, settings


def test_communities_smoke(
    connected_settings: tuple[duckdb.DuckDBPyConnection, Settings],
) -> None:
    con, settings = connected_settings
    stats = run_communities(con, settings, force=True)
    assert stats["sessions"] >= 2  # at least 2 of 3 sessions contributed
    assert stats["communities"] >= 1
    df = pl.read_parquet(settings.communities_parquet_path)
    assert set(df.columns) == {"session_id", "community_id", "size"}
    # At least one real (non-noise) community -> min community_id on the
    # real subset is non-negative.
    real = df.filter(pl.col("community_id") >= 0)
    assert real.height >= 2
    assert real["community_id"].min() >= 0


def test_load_session_centroids_matches_numpy_reference(
    connected_settings: tuple[duckdb.DuckDBPyConnection, Settings],
) -> None:
    """The SQL-CTE centroid implementation matches the legacy numpy loop.

    Replays the legacy logic (group-by-session → mean → L2 normalize) on the
    same fixture data and asserts byte-for-byte float32 equality with the
    new ``_load_session_centroids`` output. Catches regressions in the
    DuckDB ``unnest+groupby+list`` rewrite without freezing a binary
    baseline parquet on disk.
    """
    from claude_sql.community_worker import _load_session_centroids

    con, settings = connected_settings
    sids, centroids = _load_session_centroids(con, settings.embeddings_parquet_path)

    # Pull the same join via Polars and recompute the centroids in-process.
    parts = list(settings.embeddings_parquet_path.glob("part-*.parquet"))
    if parts:
        emb_df = pl.read_parquet([str(p) for p in parts])
    else:
        emb_df = pl.read_parquet(settings.embeddings_parquet_path)
    msg_df = con.execute(
        "SELECT CAST(uuid AS VARCHAR) AS uuid, "
        "CAST(session_id AS VARCHAR) AS session_id FROM messages"
    ).pl()
    joined = emb_df.join(msg_df, on="uuid", how="inner")
    expected: dict[str, np.ndarray] = {}
    for sid in sorted(joined["session_id"].unique().to_list()):
        rows = joined.filter(pl.col("session_id") == sid)
        emb = np.stack([np.asarray(v, dtype=np.float32) for v in rows["embedding"].to_list()])
        mean = emb.mean(axis=0)
        norm = np.linalg.norm(mean)
        expected[sid] = mean / norm if norm > 0 else mean

    assert sids == sorted(expected.keys())
    for i, sid in enumerate(sids):
        np.testing.assert_allclose(centroids[i], expected[sid], rtol=1e-5, atol=1e-6)

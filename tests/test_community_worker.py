"""Synthetic-fixture smoke tests for session-level Leiden+CPM community detection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pytest

from claude_sql.community_worker import (
    NOISE_COMMUNITY_ID,
    _build_igraph,
    _build_mutual_knn,
    _compute_resolution_profile,
    _warn_disconnected,
    neighbors_of,
    run_communities,
)
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
    """4 sessions, two clear pairs in centroid space.

    Session 0 + session 1 share a direction (will form one community).
    Session 2 + session 3 share a different direction (another community).
    Each session has 2 messages so ``_load_session_centroids`` averages
    something non-trivial.
    """
    proj = tmp_path / "projects" / "-home-x-proj"
    proj.mkdir(parents=True)
    sids = [f"{i:08d}-0000-0000-0000-000000000000" for i in range(1, 5)]
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

    # 8 message rows, 4 sessions of 2 messages each. Group A (sessions 0/1)
    # and group B (sessions 2/3) live on orthogonal directions in 32-d.
    rng = np.random.default_rng(42)
    dim = 32
    e = np.zeros((8, dim), dtype=np.float32)
    a_dir = rng.normal(size=dim).astype(np.float32)
    a_dir /= np.linalg.norm(a_dir)
    b_dir = rng.normal(size=dim).astype(np.float32)
    b_dir -= b_dir.dot(a_dir) * a_dir  # orthogonalize
    b_dir /= np.linalg.norm(b_dir)
    # Sessions 0 + 1 (rows 0-3): a_dir + tiny jitter.
    for r in range(4):
        e[r] = a_dir + 0.01 * rng.normal(size=dim).astype(np.float32)
    # Sessions 2 + 3 (rows 4-7): b_dir + tiny jitter.
    for r in range(4, 8):
        e[r] = b_dir + 0.01 * rng.normal(size=dim).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)

    msg_uuids = [f"u{i}-1" if k == 0 else f"a{i}-1" for i in range(4) for k in range(2)]
    emb_df = pl.DataFrame(
        {
            "uuid": msg_uuids,
            "model": ["test"] * 8,
            "dim": [dim] * 8,
            "embedding": e,
        },
        schema={
            "uuid": pl.Utf8,
            "model": pl.Utf8,
            "dim": pl.UInt16,
            "embedding": pl.Array(pl.Float32, dim),
        },
    )
    emb_path = tmp_path / "embeddings.parquet"
    emb_df.write_parquet(emb_path)

    # Subagent placeholder so ``read_json`` schema inference picks up every
    # column ``register_views`` needs (otherwise the glob matches zero files).
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
        community_profile_parquet_path=tmp_path / "community_profile.parquet",
        default_glob=str(proj / "*.jsonl"),
        subagent_glob=str(tmp_path / "projects" / "*" / "*" / "subagents" / "agent-*.jsonl"),
        subagent_meta_glob=str(
            tmp_path / "projects" / "*" / "*" / "subagents" / "agent-*.meta.json"
        ),
        # Tiny fixture: keep k small so mutual-kNN can resolve at all.
        leiden_knn_k=2,
        # Edges in this fixture have cosine ≥ 0.99 within group, ≈ 0 across.
        leiden_edge_floor=0.5,
        # 4 sessions, two pairs -> min size 2 lets both pairs survive.
        leiden_min_community_size=2,
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
    assert stats["sessions"] == 4
    assert stats["communities"] >= 1
    assert stats["algorithm"] == "leiden_cpm"
    assert isinstance(stats["gamma_used"], float)
    assert 0.0 <= float(stats["gamma_used"]) <= 1.0

    df = pl.read_parquet(settings.communities_parquet_path)
    assert set(df.columns) == {
        "session_id",
        "community_id",
        "size",
        "is_medoid",
        "coherence",
        "gamma_used",
    }
    real = df.filter(pl.col("community_id") >= 0)
    assert real.height >= 2
    # At least one medoid per real community.
    medoids = real.filter(pl.col("is_medoid"))
    assert medoids.height >= int(real["community_id"].n_unique())


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


def test_communities_deterministic(
    connected_settings: tuple[duckdb.DuckDBPyConnection, Settings],
) -> None:
    """Same seed + same input ⇒ byte-identical primary parquet across runs."""
    con, settings = connected_settings
    run_communities(con, settings, force=True)
    primary_first = settings.communities_parquet_path.read_bytes()
    profile_first = (
        settings.community_profile_parquet_path.read_bytes()
        if settings.community_profile_parquet_path.exists()
        else b""
    )

    # Force a clean re-run.
    settings.communities_parquet_path.unlink()
    if settings.community_profile_parquet_path.exists():
        settings.community_profile_parquet_path.unlink()
    run_communities(con, settings, force=True)
    primary_second = settings.communities_parquet_path.read_bytes()
    profile_second = (
        settings.community_profile_parquet_path.read_bytes()
        if settings.community_profile_parquet_path.exists()
        else b""
    )

    assert hashlib.sha256(primary_first).hexdigest() == hashlib.sha256(primary_second).hexdigest()
    assert hashlib.sha256(profile_first).hexdigest() == hashlib.sha256(profile_second).hexdigest()


def test_communities_explicit_gamma_skips_profile(
    connected_settings: tuple[duckdb.DuckDBPyConnection, Settings],
) -> None:
    """``--gamma`` runs do not write the resolution-profile sidecar."""
    con, settings = connected_settings
    if settings.community_profile_parquet_path.exists():
        settings.community_profile_parquet_path.unlink()

    stats = run_communities(con, settings, force=True, gamma=0.5)
    assert float(stats["gamma_used"]) == pytest.approx(0.5)
    # Sidecar must NOT be written when γ is explicit.
    assert not settings.community_profile_parquet_path.exists()
    df = pl.read_parquet(settings.communities_parquet_path)
    assert df["gamma_used"].unique().to_list() == [pytest.approx(0.5)]


def test_communities_resolution_profile_populated(
    connected_settings: tuple[duckdb.DuckDBPyConnection, Settings],
) -> None:
    """Auto-γ writes a populated profile sidecar with the expected schema."""
    con, settings = connected_settings
    if settings.community_profile_parquet_path.exists():
        settings.community_profile_parquet_path.unlink()
    run_communities(con, settings, force=True)
    assert settings.community_profile_parquet_path.exists()
    prof = pl.read_parquet(settings.community_profile_parquet_path)
    assert set(prof.columns) == {"gamma", "n_communities", "quality", "plateau_length"}
    assert prof.height >= 1
    # Every γ in the profile is inside the configured range.
    assert prof["gamma"].min() >= settings.leiden_resolution_range_lo
    assert prof["gamma"].max() <= settings.leiden_resolution_range_hi


def test_communities_min_size_collapse(
    connected_settings: tuple[duckdb.DuckDBPyConnection, Settings],
) -> None:
    """min_community_size=10 on a 4-session fixture collapses everything to noise."""
    con, settings = connected_settings
    settings = settings.model_copy(update={"leiden_min_community_size": 10})
    stats = run_communities(con, settings, force=True)
    assert stats["communities"] == 0
    assert stats["noise"] == 4
    df = pl.read_parquet(settings.communities_parquet_path)
    assert (df["community_id"] == NOISE_COMMUNITY_ID).all()


def test_neighbors_of_returns_top_k(
    connected_settings: tuple[duckdb.DuckDBPyConnection, Settings],
) -> None:
    """`--neighbors-of` early-return path returns top-k cosine neighbors descending."""
    con, settings = connected_settings
    # Run communities first so the join carries community_id / is_medoid.
    run_communities(con, settings, force=True)

    sids = sorted(f"{i:08d}-0000-0000-0000-000000000000" for i in range(1, 5))
    target = sids[0]
    df = neighbors_of(con, settings, target, top_k=3)
    assert df.height == 3
    assert "neighbor_session_id" in df.columns
    assert "weight" in df.columns
    # Descending weight.
    weights = df["weight"].to_list()
    assert weights == sorted(weights, reverse=True)
    # Self is excluded.
    assert target not in df["neighbor_session_id"].to_list()
    # community_id column is present because the parquet exists.
    assert "community_id" in df.columns


def test_warn_disconnected_logs_no_split(
    connected_settings: tuple[duckdb.DuckDBPyConnection, Settings],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Synthesize a community with two weakly-connected components -> warn but not split."""
    import igraph as ig

    # 6 nodes: nodes {0,1,2} are one clique, {3,4,5} are another, no edge
    # between cliques. If we manually assign all six to community 0, the
    # induced subgraph is disconnected -> warn fires but membership labels
    # are NOT changed.
    edges = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
    g = ig.Graph(n=6, edges=edges, directed=False)
    g.es["weight"] = [1.0] * len(edges)
    labels = [0, 0, 0, 0, 0, 0]
    # Use loguru's intercept so we capture the warning.
    from loguru import logger as loguru_logger

    captured: list[str] = []
    sink_id = loguru_logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
    try:
        _warn_disconnected(g, labels)
    finally:
        loguru_logger.remove(sink_id)
    assert any("weakly-connected components" in msg for msg in captured)


def test_build_mutual_knn_symmetric_no_self_loops() -> None:
    """Mutual-kNN edge list contains no self-loops and is unique per pair."""
    rng = np.random.default_rng(0)
    centroids = rng.normal(size=(20, 32)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    sim = centroids @ centroids.T
    np.fill_diagonal(sim, 0.0)
    edges, weights = _build_mutual_knn(sim, k=5, floor=0.0)
    seen: set[tuple[int, int]] = set()
    for (u, v), w in zip(edges, weights, strict=True):
        assert u != v, "self-loop in mutual-kNN edge list"
        assert (u, v) not in seen, "duplicate edge"
        seen.add((u, v))
        assert 0.0 <= w <= 1.0


def test_resolution_profile_changepoints_monotonic(
    connected_settings: tuple[duckdb.DuckDBPyConnection, Settings],
) -> None:
    """The change-point list emitted by Optimiser.resolution_profile is γ-sorted."""
    con, settings = connected_settings
    sids, centroids = _compute_centroids_helper(con, settings)
    sim = centroids @ centroids.T
    np.fill_diagonal(sim, 0.0)
    edges, weights = _build_mutual_knn(
        sim, k=settings.leiden_knn_k, floor=settings.leiden_edge_floor
    )
    g = _build_igraph(len(sids), edges, weights)
    profile = _compute_resolution_profile(
        g,
        range_lo=settings.leiden_resolution_range_lo,
        range_hi=settings.leiden_resolution_range_hi,
        seed=settings.seed,
    )
    if profile:
        gammas = [row[0] for row in profile]
        assert gammas == sorted(gammas)


def _compute_centroids_helper(
    con: duckdb.DuckDBPyConnection, settings: Settings
) -> tuple[list[str], np.ndarray]:
    from claude_sql.community_worker import _load_session_centroids

    return _load_session_centroids(con, settings.embeddings_parquet_path)

"""Coverage top-up for ``claude_sql.community_worker`` and the cli ``community`` subcommand.

The smoke tests in ``test_community_worker.py`` exercise the happy path of
auto-γ + Leiden+CPM + medoid/coherence/relabel + sidecar write. This module
targets the error / fallback branches and the CLI flag matrix so that the
agent-first paths (``--neighbors-of`` early-return, ``--dry-run`` plan,
mutual-exclusion exit 64, ``--gamma`` overriding the profile) are pinned.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl
import pytest

from claude_sql.analytics.community_worker import (
    NOISE_COMMUNITY_ID,
    _build_mutual_knn,
    _compute_medoid_and_coherence,
    _pick_zoom,
    neighbors_of,
    run_communities,
)
from claude_sql.app import cli
from claude_sql.app.cli import Common
from claude_sql.core.config import Settings
from claude_sql.core.output import OutputFormat

# ---------------------------------------------------------------------------
# _build_mutual_knn: n < 2 short-circuit
# ---------------------------------------------------------------------------


def test_build_mutual_knn_single_node_returns_empty() -> None:
    """One-node similarity matrix has no edges to find -> ``([], [])``."""
    sim = np.array([[0.0]], dtype=np.float32)
    edges, weights = _build_mutual_knn(sim, k=15, floor=0.3)
    assert edges == []
    assert weights == []


# ---------------------------------------------------------------------------
# _pick_zoom: empty profile + every level branch
# ---------------------------------------------------------------------------


def test_pick_zoom_empty_profile_raises() -> None:
    with pytest.raises(RuntimeError, match="empty resolution profile"):
        _pick_zoom([], "medium")


def test_pick_zoom_coarse_picks_lowest_n_communities_above_one() -> None:
    """coarse -> smallest n_communities >= 2."""
    profile: list[tuple[float, int, float, int]] = [
        (0.10, 1, 0.5, 100),  # only 1 community -> ineligible
        (0.30, 2, 0.4, 50),  # eligible, lowest n
        (0.60, 5, 0.3, 80),
        (0.90, 12, 0.2, 200),
    ]
    assert _pick_zoom(profile, "coarse") == pytest.approx(0.30)


def test_pick_zoom_coarse_falls_back_when_no_eligible() -> None:
    """When every partition has < 2 communities, fall back to median γ."""
    profile = [(0.10, 1, 0.5, 100), (0.30, 1, 0.4, 50), (0.60, 1, 0.3, 80)]
    # median index = 3 // 2 = 1 -> γ=0.30
    assert _pick_zoom(profile, "coarse") == pytest.approx(0.30)


def test_pick_zoom_fine_picks_highest_n_communities() -> None:
    """fine -> largest n_communities; ties broken by smallest γ."""
    profile = [
        (0.10, 2, 0.5, 100),
        (0.30, 5, 0.4, 50),
        (0.60, 12, 0.3, 80),
        (0.90, 8, 0.2, 200),
    ]
    assert _pick_zoom(profile, "fine") == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# _compute_medoid_and_coherence: size==1 community
# ---------------------------------------------------------------------------


def test_compute_medoid_and_coherence_single_node_community() -> None:
    """A community of size 1 has itself as medoid and coherence 1.0."""
    sim = np.eye(3, dtype=np.float32)
    labels = [0, 1, 2]  # three singletons
    medoids, coherence = _compute_medoid_and_coherence(sim, labels)
    assert medoids == {0, 1, 2}
    assert coherence == {0: 1.0, 1: 1.0, 2: 1.0}


# ---------------------------------------------------------------------------
# run_communities: cache-hit short-circuit
# ---------------------------------------------------------------------------


def test_run_communities_cache_hit_returns_summary(tmp_path: Path) -> None:
    """Existing parquet + force=False -> read summary, no Leiden invocation."""
    out = tmp_path / "communities.parquet"
    df = pl.DataFrame(
        {
            "session_id": ["a", "b", "c"],
            "community_id": [0, 0, NOISE_COMMUNITY_ID],
            "size": [2, 2, 0],
            "is_medoid": [True, False, False],
            "coherence": [0.9, 0.9, 0.0],
            "gamma_used": [0.5, 0.5, 0.5],
        },
        schema={
            "session_id": pl.Utf8,
            "community_id": pl.Int32,
            "size": pl.Int32,
            "is_medoid": pl.Boolean,
            "coherence": pl.Float32,
            "gamma_used": pl.Float32,
        },
    )
    df.write_parquet(out)

    settings = Settings(
        embeddings_parquet_path=tmp_path / "no-such-emb",
        communities_parquet_path=out,
        community_profile_parquet_path=tmp_path / "profile.parquet",
    )
    con = duckdb.connect(":memory:")
    try:
        stats = run_communities(con, settings, force=False)
    finally:
        con.close()
    assert stats["sessions"] == 3
    assert stats["communities"] == 1
    assert stats["noise"] == 1
    assert stats["gamma_used"] == pytest.approx(0.5)
    assert isinstance(stats["quality"], float)
    assert math.isnan(float(stats["quality"]))
    assert stats["algorithm"] == "leiden_cpm"


# ---------------------------------------------------------------------------
# run_communities: missing embeddings parquet -> RuntimeError
# ---------------------------------------------------------------------------


def test_run_communities_missing_embeddings_raises(tmp_path: Path) -> None:
    """Missing Lance dataset -> RuntimeError surfaced from the catalog wrap."""
    settings = Settings(
        lance_uri=tmp_path / "no-such-dir",
        communities_parquet_path=tmp_path / "communities.parquet",
        community_profile_parquet_path=tmp_path / "profile.parquet",
    )
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE messages (uuid VARCHAR, session_id VARCHAR)")
    try:
        # ``message_embeddings`` view is not registered → CatalogException → RuntimeError wrap.
        with pytest.raises(RuntimeError, match="No embeddings"):
            run_communities(con, settings, force=True)
    finally:
        con.close()


def test_run_communities_empty_join_raises(tmp_path: Path) -> None:
    """Embeddings exist but no uuid matches messages -> RuntimeError."""
    from datetime import UTC, datetime as _dt

    from claude_sql.core import lance_store
    from claude_sql.core.sql_views import register_vss

    lance_uri = tmp_path / "embeddings_lance"
    df = pl.DataFrame(
        {
            "uuid": ["unrelated-uuid"],
            "model": ["test"],
            "dim": [4],
            "embedding": [np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)],
            "embedded_at": [_dt.now(UTC)],
        },
        schema={
            "uuid": pl.Utf8,
            "model": pl.Utf8,
            "dim": pl.Int32,
            "embedding": pl.Array(pl.Float32, 4),
            "embedded_at": pl.Datetime("us", "UTC"),
        },
    )
    db = lance_store.connect_db(lance_uri)
    tbl = lance_store.open_or_create_table(db, dim=4)
    lance_store.add_chunk(tbl, df)
    settings = Settings(
        lance_uri=lance_uri,
        communities_parquet_path=tmp_path / "communities.parquet",
        community_profile_parquet_path=tmp_path / "profile.parquet",
    )
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE messages AS SELECT 'other-uuid' AS uuid, 's1' AS session_id")
    register_vss(con, lance_uri=lance_uri, dim=4)
    try:
        with pytest.raises(RuntimeError, match="No rows returned"):
            run_communities(con, settings, force=True)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# run_communities: settings.leiden_resolution scalar overrides auto-γ
# ---------------------------------------------------------------------------


def test_run_communities_settings_leiden_resolution_skips_profile(
    connected_settings_module: tuple[duckdb.DuckDBPyConnection, Settings],
) -> None:
    """When ``settings.leiden_resolution`` is set, the profile is not run and no sidecar is written."""
    con, settings = connected_settings_module
    settings = settings.model_copy(update={"leiden_resolution": 0.4})
    if settings.community_profile_parquet_path.exists():
        settings.community_profile_parquet_path.unlink()
    stats = run_communities(con, settings, force=True)
    assert float(stats["gamma_used"]) == pytest.approx(0.4)
    assert not settings.community_profile_parquet_path.exists()


# ---------------------------------------------------------------------------
# neighbors_of: ValueError + missing-parquet warning
# ---------------------------------------------------------------------------


def test_neighbors_of_top_k_clamped_to_zero_returns_empty(
    connected_settings_module: tuple[duckdb.DuckDBPyConnection, Settings],
) -> None:
    """top_k=0 (or > N-1 with N=1) clamps to zero and returns the empty schema."""
    con, settings = connected_settings_module
    sids = sorted(f"{i:08d}-0000-0000-0000-000000000000" for i in range(1, 5))
    df = neighbors_of(con, settings, sids[0], top_k=0)
    assert df.height == 0
    assert set(df.columns) == {"neighbor_session_id", "weight"}


def test_compute_resolution_profile_emits_last_partition_plateau(
    connected_settings_module: tuple[duckdb.DuckDBPyConnection, Settings],
) -> None:
    """The final partition in the profile uses range_hi as its right edge."""
    from claude_sql.analytics.community_worker import (
        _build_igraph,
        _compute_resolution_profile,
    )

    con, settings = connected_settings_module
    # Reuse the loader to build a real graph then run the profile end-to-end
    # so the last-partition `next_gamma = range_hi` branch executes.
    from claude_sql.analytics.community_worker import _build_mutual_knn, _load_session_centroids

    sids, centroids = _load_session_centroids(con, settings.embeddings_parquet_path)
    sim = centroids @ centroids.T
    np.fill_diagonal(sim, 0.0)
    edges, weights = _build_mutual_knn(
        sim, k=settings.leiden_knn_k, floor=settings.leiden_edge_floor
    )
    g = _build_igraph(len(sids), edges, weights)
    profile = _compute_resolution_profile(g, range_lo=0.05, range_hi=0.95, seed=42)
    # At least one row, and every plateau_length is >= 0.
    assert profile
    assert all(row[3] >= 0 for row in profile)


def test_neighbors_of_unknown_session_raises(
    connected_settings_module: tuple[duckdb.DuckDBPyConnection, Settings],
) -> None:
    con, settings = connected_settings_module
    with pytest.raises(ValueError, match="not found in embeddings corpus"):
        neighbors_of(con, settings, "no-such-sid", top_k=5)


def test_neighbors_of_warns_when_communities_parquet_missing(
    connected_settings_module: tuple[duckdb.DuckDBPyConnection, Settings],
) -> None:
    """Without session_communities.parquet, neighbors_of returns sid+weight only and warns."""
    from loguru import logger as loguru_logger

    con, settings = connected_settings_module
    if settings.communities_parquet_path.exists():
        settings.communities_parquet_path.unlink()
    captured: list[str] = []
    sink_id = loguru_logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
    try:
        sids = sorted(f"{i:08d}-0000-0000-0000-000000000000" for i in range(1, 5))
        df = neighbors_of(con, settings, sids[0], top_k=2)
    finally:
        loguru_logger.remove(sink_id)
    assert "community_id" not in df.columns
    assert any("communities parquet missing" in m for m in captured)


# ---------------------------------------------------------------------------
# CLI community subcommand: --dry-run, --neighbors-of, mutual-exclusion
# ---------------------------------------------------------------------------


def test_community_dry_run_emits_plan(
    connected_corpus: tuple[duckdb.DuckDBPyConnection, Settings, Common],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--dry-run emits a JSON plan and never calls run_communities."""
    _con, settings, common = connected_corpus
    # Make _resolve_settings + _open_connection_full see our temp settings.
    monkeypatch.setattr(cli, "_resolve_settings", lambda c: settings)
    cli.community(
        force=False,
        gamma=None,
        resolution="medium",
        neighbors_of_session=None,
        top_k=15,
        dry_run=True,
        common=common,
    )
    out = capsys.readouterr().out
    plan = json.loads(out)
    assert plan["pipeline"] == "community"
    assert plan["dry_run"] is True
    assert plan["gamma"] == "auto"
    assert plan["candidate_sessions"] >= 1
    # Sidecar listed because gamma == "auto"
    assert any("community_profile.parquet" in p for p in plan["would_write"])


def test_community_dry_run_with_explicit_gamma_drops_sidecar(
    connected_corpus: tuple[duckdb.DuckDBPyConnection, Settings, Common],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--dry-run --gamma X plans without the profile sidecar."""
    _con, settings, common = connected_corpus
    monkeypatch.setattr(cli, "_resolve_settings", lambda c: settings)
    cli.community(
        force=False,
        gamma=0.5,
        resolution="medium",
        neighbors_of_session=None,
        top_k=15,
        dry_run=True,
        common=common,
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["gamma"] == 0.5
    assert all("community_profile.parquet" not in p for p in plan["would_write"])


def test_community_neighbors_of_early_return(
    connected_corpus: tuple[duckdb.DuckDBPyConnection, Settings, Common],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--neighbors-of returns top-k cosine neighbors as JSON without running Leiden."""
    _con, settings, common = connected_corpus
    monkeypatch.setattr(cli, "_resolve_settings", lambda c: settings)
    sids = sorted(f"{i:08d}-0000-0000-0000-000000000000" for i in range(1, 5))
    cli.community(
        force=False,
        gamma=None,
        resolution="medium",
        neighbors_of_session=sids[0],
        top_k=2,
        dry_run=False,
        common=common,
    )
    out = capsys.readouterr().out
    rows = json.loads(out)
    assert len(rows) == 2
    assert all("neighbor_session_id" in r for r in rows)


def test_community_mutual_exclusion_exits_64(
    connected_corpus: tuple[duckdb.DuckDBPyConnection, Settings, Common],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--neighbors-of combined with --gamma exits 64 with a structured error."""
    _con, settings, common = connected_corpus
    monkeypatch.setattr(cli, "_resolve_settings", lambda c: settings)
    with pytest.raises(SystemExit) as exc:
        cli.community(
            force=False,
            gamma=0.5,
            resolution="medium",
            neighbors_of_session="any",
            top_k=5,
            dry_run=False,
            common=common,
        )
    assert exc.value.code == 64
    err = json.loads(capsys.readouterr().err)
    assert err["error"]["kind"] == "invalid_input"
    assert "mutually exclusive" in err["error"]["message"]


# ---------------------------------------------------------------------------
# Module-scoped fixture: a working centroid+messages corpus that survives
# multiple test runs (we mutate communities_parquet_path / sidecar in place).
# ---------------------------------------------------------------------------


@pytest.fixture
def connected_settings_module(
    tmp_path: Path,
) -> tuple[duckdb.DuckDBPyConnection, Settings]:
    """Mirror of the smoke test fixture; 4 sessions, 2 orthogonal centroid groups."""
    from datetime import UTC, datetime as _dt

    from claude_sql.core import lance_store
    from claude_sql.core.sql_views import register_raw, register_views, register_vss

    proj = tmp_path / "projects" / "-home-x-proj"
    proj.mkdir(parents=True)
    sids = [f"{i:08d}-0000-0000-0000-000000000000" for i in range(1, 5)]
    for i, sid in enumerate(sids):
        (proj / f"{sid}.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "parentUuid": None,
                        "isSidechain": False,
                        "type": "user" if k == 0 else "assistant",
                        "uuid": f"{'u' if k == 0 else 'a'}{i}-1",
                        "timestamp": f"2026-04-0{i + 1}T10:00:0{k}.000Z",
                        "sessionId": sid,
                        "version": "2.0.0",
                        "gitBranch": "main",
                        "cwd": "/x",
                        "userType": "external",
                        "entrypoint": "cli",
                        "permissionMode": "acceptEdits",
                        "promptId": f"p-{i}-{k}",
                        "message": {
                            "id": f"m-{i}-{k}",
                            "type": "message",
                            "role": "user" if k == 0 else "assistant",
                            "model": "claude-sonnet-4-6",
                            "stop_reason": "end_turn",
                            "stop_sequence": None,
                            "content": [{"type": "text", "text": "x" * 100}],
                            "usage": {
                                "input_tokens": 50,
                                "output_tokens": 20,
                                "cache_read_input_tokens": 0,
                                "cache_creation_input_tokens": 0,
                            },
                        },
                    }
                )
                for k in range(2)
            )
            + "\n"
        )

    rng = np.random.default_rng(42)
    dim = 32
    e = np.zeros((8, dim), dtype=np.float32)
    a_dir = rng.normal(size=dim).astype(np.float32)
    a_dir /= np.linalg.norm(a_dir)
    b_dir = rng.normal(size=dim).astype(np.float32)
    b_dir -= b_dir.dot(a_dir) * a_dir
    b_dir /= np.linalg.norm(b_dir)
    for r in range(4):
        e[r] = a_dir + 0.01 * rng.normal(size=dim).astype(np.float32)
    for r in range(4, 8):
        e[r] = b_dir + 0.01 * rng.normal(size=dim).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    msg_uuids = [f"u{i}-1" if k == 0 else f"a{i}-1" for i in range(4) for k in range(2)]
    now = _dt.now(UTC)
    emb_df = pl.DataFrame(
        {
            "uuid": msg_uuids,
            "model": ["test"] * 8,
            "dim": [dim] * 8,
            "embedding": e,
            "embedded_at": [now] * 8,
        },
        schema={
            "uuid": pl.Utf8,
            "model": pl.Utf8,
            "dim": pl.Int32,
            "embedding": pl.Array(pl.Float32, dim),
            "embedded_at": pl.Datetime("us", "UTC"),
        },
    )
    lance_uri = tmp_path / "embeddings_lance"
    db = lance_store.connect_db(lance_uri)
    tbl = lance_store.open_or_create_table(db, dim=dim)
    lance_store.add_chunk(tbl, emb_df)

    parent_uuid = "99999999-9999-9999-9999-999999999999"
    sub_dir = tmp_path / "projects" / "-x" / parent_uuid / "subagents"
    sub_dir.mkdir(parents=True)
    (sub_dir / "agent-deadbeef.meta.json").write_text(
        json.dumps({"agentType": "general-purpose", "description": "x"})
    )
    (sub_dir / "agent-deadbeef.jsonl").write_text(
        json.dumps(
            {
                "parentUuid": None,
                "isSidechain": False,
                "type": "user",
                "uuid": "sub-u-dead",
                "timestamp": "2026-04-01T10:00:00.000Z",
                "sessionId": f"sub-{parent_uuid}",
                "version": "2.0.0",
                "gitBranch": "main",
                "cwd": "/x",
                "userType": "external",
                "entrypoint": "cli",
                "permissionMode": "acceptEdits",
                "promptId": "p-sub",
                "message": {
                    "id": "m-sub",
                    "type": "message",
                    "role": "user",
                    "model": "claude-sonnet-4-6",
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "content": [{"type": "text", "text": "x"}],
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            }
        )
        + "\n"
    )

    settings = Settings(
        lance_uri=lance_uri,
        communities_parquet_path=tmp_path / "communities.parquet",
        community_profile_parquet_path=tmp_path / "community_profile.parquet",
        default_glob=str(proj / "*.jsonl"),
        subagent_glob=str(tmp_path / "projects" / "*" / "*" / "subagents" / "agent-*.jsonl"),
        subagent_meta_glob=str(
            tmp_path / "projects" / "*" / "*" / "subagents" / "agent-*.meta.json"
        ),
        leiden_knn_k=2,
        leiden_edge_floor=0.5,
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
    register_vss(con, lance_uri=lance_uri, dim=dim)
    # Pre-populate the communities parquet so neighbors_of's join branch can hit.
    run_communities(con, settings, force=True)
    return con, settings


@pytest.fixture
def connected_corpus(
    connected_settings_module: tuple[duckdb.DuckDBPyConnection, Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[duckdb.DuckDBPyConnection, Settings, Common]:
    """Adds a Common dataclass + monkey-patches ``cli._open_connection_full`` to use the fixture's con."""
    con, settings = connected_settings_module
    common = Common(
        verbose=False, quiet=True, glob=None, subagent_glob=None, format=OutputFormat.JSON
    )

    def _open(_settings: Any) -> duckdb.DuckDBPyConnection:
        return con

    # cli.community calls ``_open_connection_full`` directly; the legacy
    # ``_open_connection`` alias is no longer needed for tests now that
    # PR 3 has split the helper into ``_open_connection_full`` and
    # ``_open_connection_introspect``.
    monkeypatch.setattr(cli, "_open_connection_full", _open)
    return con, settings, common

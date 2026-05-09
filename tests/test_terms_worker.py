"""Tests for :mod:`claude_sql.terms_worker`.

Covers the missing-input guard, the cached-output short-circuit, the
end-to-end c-TF-IDF happy path, the noise-cluster (``-1``) filter, and the
``force=True`` override that ignores existing output.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import polars as pl
import pytest

from claude_sql.config import Settings
from claude_sql.sql_views import register_raw, register_views
from claude_sql.terms_worker import run_terms
from conftest import (
    _seed_subagent_stub,
    make_user_msg,
    write_session_jsonl,
)

# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _build_corpus_with_clusters(tmp_path: Path) -> tuple[duckdb.DuckDBPyConnection, list[str]]:
    """Write a tiny JSONL corpus and return ``(con, uuids)`` for two clusters.

    Six messages total — three "alpha"-heavy, three "beta"-heavy — each long
    enough to clear the 32-char ``messages_text`` filter. The returned uuid
    list is exactly the order written, six entries.
    """
    proj = tmp_path / "projects" / "proj-terms"
    sid = "tssid-1111-1111-1111-111111111111"

    alpha_texts = [
        "alpha alpha alpha gamma delta epsilon: long enough to clear filter",
        "alpha alpha gamma delta zeta eta theta long phrase clears filter ok",
        "alpha alpha gamma iota kappa lambda mu nu xi omicron pi long enough",
    ]
    beta_texts = [
        "beta beta beta sigma tau upsilon phi chi long enough to clear floor",
        "beta beta sigma tau psi omega long phrase clears the 32-char floor",
        "beta beta sigma tau aa bb cc dd ee ff gg hh ii jj long phrase pass",
    ]

    messages = []
    uuids: list[str] = []
    base = "2026-04-01T10:00:"
    for i, text in enumerate(alpha_texts + beta_texts):
        uid = f"u{i:02d}"
        uuids.append(uid)
        messages.append(
            make_user_msg(
                uid,
                sid,
                text,
                ts=f"{base}{i:02d}.000Z",
            )
        )
    write_session_jsonl(proj / f"{sid}.jsonl", messages=messages)

    sa_glob, sa_meta_glob = _seed_subagent_stub(tmp_path)
    con = duckdb.connect(":memory:")
    register_raw(
        con,
        glob=str(tmp_path / "projects" / "*" / "*.jsonl"),
        subagent_glob=sa_glob,
        subagent_meta_glob=sa_meta_glob,
    )
    register_views(con)
    return con, uuids


def _settings_with_minimal_tfidf(tmp_path: Path) -> Settings:
    """Settings whose paths live under ``tmp_path`` and whose TF-IDF params
    let a tiny synthetic corpus produce a non-empty vocabulary.
    """
    cache = tmp_path / "claude"
    cache.mkdir(parents=True, exist_ok=True)
    return Settings(
        embeddings_parquet_path=cache / "embeddings",
        clusters_parquet_path=cache / "clusters.parquet",
        cluster_terms_parquet_path=cache / "cluster_terms.parquet",
        # min_df=1 is required because tiny synthetic corpora don't give any
        # term enough cross-cluster repetition to clear the default min_df=2.
        tfidf_min_df=1,
        # max_df=1.0 lets terms that appear in every pseudo-document survive,
        # which is fine for a 2-cluster fixture.
        tfidf_max_df=1.0,
        tfidf_ngram_min=1,
        tfidf_ngram_max=1,
        tfidf_top_n_terms=5,
    )


def _write_clusters_parquet(
    path: Path,
    pairs: list[tuple[str, int]],
) -> None:
    """Write a clusters.parquet with the cluster_worker's column shape."""
    df = pl.DataFrame(
        {
            "uuid": [u for u, _ in pairs],
            "cluster_id": [c for _, c in pairs],
            "x": [0.0] * len(pairs),
            "y": [0.0] * len(pairs),
            "is_noise": [c < 0 for _, c in pairs],
        },
        schema={
            "uuid": pl.Utf8,
            "cluster_id": pl.Int32,
            "x": pl.Float32,
            "y": pl.Float32,
            "is_noise": pl.Boolean,
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


# ---------------------------------------------------------------------------
# Connection fixture (per-test so the corpus stays small and isolated)
# ---------------------------------------------------------------------------


@pytest.fixture
def corpus_con(tmp_path: Path) -> Iterator[tuple[duckdb.DuckDBPyConnection, list[str]]]:
    con, uuids = _build_corpus_with_clusters(tmp_path)
    try:
        yield con, uuids
    finally:
        con.close()


# ---------------------------------------------------------------------------
# 1. FileNotFoundError when clusters parquet missing
# ---------------------------------------------------------------------------


def test_run_terms_raises_when_clusters_missing(tmp_path: Path) -> None:
    settings = _settings_with_minimal_tfidf(tmp_path)
    # clusters_parquet_path defaults to tmp_path/claude/clusters.parquet — and
    # that path does not exist.
    assert not settings.clusters_parquet_path.exists()
    con = duckdb.connect(":memory:")
    try:
        with pytest.raises(FileNotFoundError, match="Clusters parquet missing"):
            run_terms(con, settings, force=False)
    finally:
        con.close()


def test_run_terms_raises_when_clusters_too_small(tmp_path: Path) -> None:
    """A truncated <16-byte parquet file is treated as missing."""
    settings = _settings_with_minimal_tfidf(tmp_path)
    settings.clusters_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    settings.clusters_parquet_path.write_bytes(b"x")  # 1 byte — well under 16
    con = duckdb.connect(":memory:")
    try:
        with pytest.raises(FileNotFoundError, match="Clusters parquet missing"):
            run_terms(con, settings, force=False)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# 2. Cached short-circuit when output exists and force=False
# ---------------------------------------------------------------------------


def test_run_terms_returns_cached_result(tmp_path: Path) -> None:
    settings = _settings_with_minimal_tfidf(tmp_path)
    # Existing clusters parquet (must pass the 16-byte sniff so the
    # short-circuit isn't pre-empted by FileNotFoundError).
    _write_clusters_parquet(
        settings.clusters_parquet_path,
        [("a", 0), ("b", 0), ("c", 1)],
    )
    # Pre-write a fake cluster_terms parquet — three rows over two clusters.
    cached_rows = [
        {"cluster_id": 0, "term": "alpha", "weight": 0.7, "rank": 1},
        {"cluster_id": 0, "term": "beta", "weight": 0.3, "rank": 2},
        {"cluster_id": 1, "term": "gamma", "weight": 0.9, "rank": 1},
    ]
    pl.DataFrame(
        cached_rows,
        schema={
            "cluster_id": pl.Int32,
            "term": pl.Utf8,
            "weight": pl.Float32,
            "rank": pl.Int32,
        },
    ).write_parquet(settings.cluster_terms_parquet_path)

    con = duckdb.connect(":memory:")
    try:
        out = run_terms(con, settings, force=False)
    finally:
        con.close()

    assert out == {"clusters": 2, "terms": 3}
    # Output parquet was not rewritten — still has our exact synthetic rows
    # (float32 round-trip means we compare structure, not bit-equal weights).
    df = pl.read_parquet(settings.cluster_terms_parquet_path)
    assert len(df) == 3
    assert sorted(df["term"].to_list()) == ["alpha", "beta", "gamma"]
    assert df["rank"].to_list() == [1, 2, 1]
    assert sorted(df["cluster_id"].unique().to_list()) == [0, 1]


# ---------------------------------------------------------------------------
# 3. End-to-end happy path
# ---------------------------------------------------------------------------


def test_run_terms_happy_path(
    tmp_path: Path,
    corpus_con: tuple[duckdb.DuckDBPyConnection, list[str]],
) -> None:
    con, uuids = corpus_con
    settings = _settings_with_minimal_tfidf(tmp_path)

    # First three uuids → cluster 0 (alpha-heavy), last three → cluster 1.
    _write_clusters_parquet(
        settings.clusters_parquet_path,
        [(uid, 0 if i < 3 else 1) for i, uid in enumerate(uuids)],
    )

    out = run_terms(con, settings, force=True)
    assert out["clusters"] >= 1
    assert out["terms"] > 0
    assert settings.cluster_terms_parquet_path.exists()

    df = pl.read_parquet(settings.cluster_terms_parquet_path)
    assert set(df.columns) == {"cluster_id", "term", "weight", "rank"}
    assert df["rank"].min() == 1
    # Both cluster ids must show up — neither dropped to zero rows.
    assert set(df["cluster_id"].unique().to_list()) <= {0, 1}
    assert (df["weight"] > 0).all()

    # The dominant per-cluster terms reflect the synthetic vocabulary.
    top_zero = df.filter(pl.col("cluster_id") == 0).sort("rank")["term"].to_list()
    top_one = df.filter(pl.col("cluster_id") == 1).sort("rank")["term"].to_list()
    assert "alpha" in top_zero
    assert "beta" in top_one


# ---------------------------------------------------------------------------
# 4. Noise cluster (-1) is filtered out
# ---------------------------------------------------------------------------


def test_run_terms_skips_noise_cluster(
    tmp_path: Path,
    corpus_con: tuple[duckdb.DuckDBPyConnection, list[str]],
) -> None:
    con, uuids = corpus_con
    settings = _settings_with_minimal_tfidf(tmp_path)

    # Mix: first uuid noise (-1), next two cluster 0, next three cluster 1.
    pairs: list[tuple[str, int]] = [
        (uuids[0], -1),
        (uuids[1], 0),
        (uuids[2], 0),
        (uuids[3], 1),
        (uuids[4], 1),
        (uuids[5], 1),
    ]
    _write_clusters_parquet(settings.clusters_parquet_path, pairs)

    out = run_terms(con, settings, force=True)
    assert out["terms"] > 0

    df = pl.read_parquet(settings.cluster_terms_parquet_path)
    assert (df["cluster_id"] >= 0).all(), df
    # Specifically: -1 is absent.
    assert -1 not in df["cluster_id"].unique().to_list()


# ---------------------------------------------------------------------------
# 5. force=True overrides existing output
# ---------------------------------------------------------------------------


def test_run_terms_force_overrides_existing(
    tmp_path: Path,
    corpus_con: tuple[duckdb.DuckDBPyConnection, list[str]],
) -> None:
    con, uuids = corpus_con
    settings = _settings_with_minimal_tfidf(tmp_path)

    _write_clusters_parquet(
        settings.clusters_parquet_path,
        [(uid, 0 if i < 3 else 1) for i, uid in enumerate(uuids)],
    )

    # Pre-seed with garbage rows — a single dummy cluster id no caller would
    # ever produce — so we can detect that ``force=True`` rewrote the file.
    pl.DataFrame(
        [{"cluster_id": 999, "term": "garbage", "weight": 0.01, "rank": 1}],
        schema={
            "cluster_id": pl.Int32,
            "term": pl.Utf8,
            "weight": pl.Float32,
            "rank": pl.Int32,
        },
    ).write_parquet(settings.cluster_terms_parquet_path)

    out = run_terms(con, settings, force=True)
    assert out["terms"] > 0

    df = pl.read_parquet(settings.cluster_terms_parquet_path)
    # Garbage row would have cluster_id 999 — the recompute drops it entirely.
    assert 999 not in df["cluster_id"].unique().to_list()
    assert "garbage" not in df["term"].to_list()
    # The recompute populated cluster 0 and/or 1.
    assert set(df["cluster_id"].unique().to_list()) <= {0, 1}

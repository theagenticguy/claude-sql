"""Tests for canonical-uuid-based dedup in ``embed_worker.discover_unembedded``.

Act-2 added an ``ingest_stamps`` parquet whose ``canonical_uuid`` column
points each near-duplicate at the canonical row of its cluster. Act-2b
(the embed_worker change under test) extends ``discover_unembedded`` to
LEFT JOIN that view and skip rows where ``canonical_uuid`` points
elsewhere — so every cluster's canonical gets embedded exactly once and
queries against a near-dup's content fall back to the canonical's vector.

Five tests cover the LEFT JOIN truth table:

* canonical row that points at itself → embed
* near-dup row (canonical_uuid != uuid) → skip
* unstamped row (no ingest_stamps entry) → embed
* ``ingest_stamps`` view absent (fresh install) → embed everything
* canonical pointer flipped after first embed → no-op on rerun

Hermetic: no Bedrock calls (we only exercise the SQL planner), no writes
outside ``tmp_path``, no dependency on the user's live corpus.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from claude_sql.analytics.embed_worker import discover_unembedded
from claude_sql.core import lance_store
from claude_sql.core.sql_views import register_raw, register_views

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stamp_row(
    *,
    uuid: str,
    session_id: str,
    canonical_uuid: str | None,
    ts: datetime,
) -> dict[str, Any]:
    """Build one ``ingest_stamps`` parquet row matching the production schema."""
    return {
        "uuid": uuid,
        "session_id": session_id,
        "approx_tokens": 16,
        "simhash64": 0,
        "token_budget_bucket": "xs",
        "canonical_uuid": canonical_uuid,
        "first_seen_ts": ts,
        "stamped_at": ts,
    }


def _write_ingest_stamps(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write ``rows`` as a single parquet at ``path`` matching the prod schema."""
    schema: dict[str, pl.DataType | type[pl.DataType]] = {
        "uuid": pl.Utf8,
        "session_id": pl.Utf8,
        "approx_tokens": pl.Int32,
        "simhash64": pl.Int64,
        "token_budget_bucket": pl.Utf8,
        "canonical_uuid": pl.Utf8,
        "first_seen_ts": pl.Datetime("us", "UTC"),
        "stamped_at": pl.Datetime("us", "UTC"),
    }
    df = pl.DataFrame(rows, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def _bind_ingest_stamps_view(con: duckdb.DuckDBPyConnection, parquet: Path) -> None:
    """Register the ``ingest_stamps`` view directly from a single parquet file.

    Bypasses ``register_analytics_views`` so the test doesn't need to wire
    a full Settings object — the only contract ``discover_unembedded``
    relies on is that a view named ``ingest_stamps`` with columns
    ``(uuid, canonical_uuid)`` exists.
    """
    con.execute(
        f"CREATE OR REPLACE VIEW ingest_stamps AS "
        f"SELECT * FROM read_parquet('{parquet.as_posix()}')"
    )


def _seed_lance_uuids(lance_uri: Path, uuids: list[str], dim: int = 4) -> None:
    """Seed Lance so the listed uuids look 'already embedded' to the worker."""
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


def _open_views(tmp_corpus: dict[str, Any]) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB with raw + business views registered."""
    con = duckdb.connect(":memory:")
    register_raw(
        con,
        glob=tmp_corpus["glob"],
        subagent_glob=tmp_corpus["subagent_glob"],
        subagent_meta_glob=tmp_corpus["subagent_meta_glob"],
    )
    register_views(con)
    return con


# ---------------------------------------------------------------------------
# Tests — all use the shared ``tmp_corpus`` fixture which seeds u1, u3, v1
# (user-role messages with text >= 32 chars) across two sessions.
# ---------------------------------------------------------------------------


def test_canonical_row_is_embedded(tmp_corpus: dict[str, Any], tmp_path: Path) -> None:
    """A self-pointing canonical (canonical_uuid = uuid) must stay in the candidate set."""
    sid = tmp_corpus["session_ids"][0]
    con = _open_views(tmp_corpus)
    try:
        ts = datetime(2026, 4, 1, 10, 0, 0, tzinfo=UTC)
        stamps = tmp_path / "ingest_stamps.parquet"
        _write_ingest_stamps(
            stamps, [_stamp_row(uuid="u1", session_id=sid, canonical_uuid="u1", ts=ts)]
        )
        _bind_ingest_stamps_view(con, stamps)

        rows = discover_unembedded(con, lance_uri=tmp_path / "lance_empty")
        uuids = {r[0] for r in rows}
        assert "u1" in uuids
    finally:
        con.close()


def test_near_dup_row_is_skipped(tmp_corpus: dict[str, Any], tmp_path: Path) -> None:
    """Near-dup (canonical_uuid points at *another* uuid) must be filtered out."""
    sid = tmp_corpus["session_ids"][0]
    con = _open_views(tmp_corpus)
    try:
        ts = datetime(2026, 4, 1, 10, 0, 0, tzinfo=UTC)
        stamps = tmp_path / "ingest_stamps.parquet"
        # Pretend u3 is a near-dup of u1; canonical points at u1.
        _write_ingest_stamps(
            stamps,
            [
                _stamp_row(uuid="u1", session_id=sid, canonical_uuid="u1", ts=ts),
                _stamp_row(uuid="u3", session_id=sid, canonical_uuid="u1", ts=ts),
            ],
        )
        _bind_ingest_stamps_view(con, stamps)

        rows = discover_unembedded(con, lance_uri=tmp_path / "lance_empty")
        uuids = {r[0] for r in rows}
        assert "u1" in uuids
        assert "u3" not in uuids
    finally:
        con.close()


def test_unstamped_row_is_embedded(tmp_corpus: dict[str, Any], tmp_path: Path) -> None:
    """messages_text row missing from ingest_stamps must still be a candidate.

    LEFT JOIN keeps the unstamped row (canonical_uuid IS NULL → keep). This
    is the partial-ingest path: a fresh embed run on a corpus where the
    stamper hasn't yet caught up to every message must not starve.
    """
    sid = tmp_corpus["session_ids"][0]
    con = _open_views(tmp_corpus)
    try:
        ts = datetime(2026, 4, 1, 10, 0, 0, tzinfo=UTC)
        stamps = tmp_path / "ingest_stamps.parquet"
        # Only u1 stamped; u3 deliberately absent from ingest_stamps.
        _write_ingest_stamps(
            stamps, [_stamp_row(uuid="u1", session_id=sid, canonical_uuid="u1", ts=ts)]
        )
        _bind_ingest_stamps_view(con, stamps)

        rows = discover_unembedded(con, lance_uri=tmp_path / "lance_empty")
        uuids = {r[0] for r in rows}
        assert "u1" in uuids
        assert "u3" in uuids  # unstamped → LEFT JOIN keeps it
    finally:
        con.close()


def test_ingest_stamps_view_absent_falls_through(
    tmp_corpus: dict[str, Any], tmp_path: Path
) -> None:
    """Without registering ingest_stamps the candidate set must be the full corpus.

    This is the fresh-install default (analytics not yet run); the JOIN
    must not be injected against an unregistered view.
    """
    con = _open_views(tmp_corpus)
    try:
        # Sanity: ingest_stamps is NOT bound on this connection.
        probe = con.execute(
            "SELECT count(*) FROM duckdb_views() WHERE view_name = 'ingest_stamps'"
        ).fetchone()
        assert probe is not None
        assert int(probe[0]) == 0

        rows = discover_unembedded(con, lance_uri=tmp_path / "lance_empty")
        uuids = {r[0] for r in rows}
        # Fixture seeds u1, u3 in session one and v1 in session two.
        assert {"u1", "u3", "v1"} <= uuids
    finally:
        con.close()


def test_canonical_pointer_changes_re_runs_correctly(
    tmp_corpus: dict[str, Any], tmp_path: Path
) -> None:
    """After the canonical pointer flips on u1, the rerun is a no-op for the cluster.

    Sequence:
    1. Stamp u1 as canonical (canonical_uuid = u1), u3 as near-dup
       (canonical_uuid = u1).
    2. Embed u1 (seeded into Lance).
    3. Flip u1's pointer so u1.canonical_uuid = u3 (e.g. ``resolve_canonicals``
       picked a different cluster representative on a later run).
    4. ``discover_unembedded`` must return *neither* u1 nor u3:
       - u1 is already embedded (Lance anti-join filters it out) and its
         stamp now says canonical_uuid != uuid (skip-near-dup branch).
       - u3's stamp still has canonical_uuid = u1, so u3 stays in the
         skip-near-dup branch.

    The behaviour pinned: a canonical-pointer flip on the already-embedded
    side does not generate fresh embed work. The cluster's vector lives at
    u1 and the canonical-resolve macro routes reads accordingly.
    """
    sid = tmp_corpus["session_ids"][0]
    con = _open_views(tmp_corpus)
    try:
        ts = datetime(2026, 4, 1, 10, 0, 0, tzinfo=UTC)
        stamps = tmp_path / "ingest_stamps.parquet"

        # Phase 1: u1 canonical, u3 near-dup -> only u1 embeddable.
        _write_ingest_stamps(
            stamps,
            [
                _stamp_row(uuid="u1", session_id=sid, canonical_uuid="u1", ts=ts),
                _stamp_row(uuid="u3", session_id=sid, canonical_uuid="u1", ts=ts),
            ],
        )
        _bind_ingest_stamps_view(con, stamps)

        lance_uri = tmp_path / "lance_dataset"
        first = {r[0] for r in discover_unembedded(con, lance_uri=lance_uri)}
        # u1 is the only canonical in the cluster; v1 is in a different
        # session and unstamped, so it also surfaces as a candidate.
        assert "u1" in first
        assert "u3" not in first

        # Simulate the embed having run: u1 (and v1) now in Lance.
        _seed_lance_uuids(lance_uri, ["u1", "v1"])

        # Phase 2: u1's pointer flips to u3 (resolve_canonicals re-shuffled the
        # cluster representative). u3's stamp is unchanged: canonical_uuid still
        # points at u1, so u3 still falls into the skip branch.
        _write_ingest_stamps(
            stamps,
            [
                _stamp_row(uuid="u1", session_id=sid, canonical_uuid="u3", ts=ts),
                _stamp_row(uuid="u3", session_id=sid, canonical_uuid="u1", ts=ts),
            ],
        )

        second = {r[0] for r in discover_unembedded(con, lance_uri=lance_uri)}
        # u1: canonical_uuid != uuid AND already embedded -> skipped twice over.
        # u3: canonical_uuid != uuid -> skip-near-dup branch fires.
        # v1: already embedded (seeded above) -> Lance anti-join filters it.
        assert "u1" not in second
        assert "u3" not in second
        assert "v1" not in second
    finally:
        con.close()

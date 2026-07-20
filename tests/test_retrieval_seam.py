"""Tests for the T-4-2 retrieval seam: reader + search adapters + facade.

Covers the surfaces downstream consumers import:

* :class:`DuckDbTranscriptReader` — session_messages ordering over both the
  flat ``{sid}.jsonl`` and the part-dir ``{sid}/part-*.jsonl`` layouts, plus
  session_ids / session_bounds delegation.
* ``read_turn_text`` renders the collapse contract (role-rank ordering,
  bare tool markers, the per-turn / total caps).
* :class:`DuckDbSessionSearch` — a fake embedder + a tiny real Lance store,
  including the ``session_id`` filter and the empty-store ``[]`` case.
* :class:`ClaudeSql` facade — lazily builds the ports, and a bare import of
  ``claude_sql.composition`` pulls in no duckdb.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from claude_sql.application.ports import SearchHit
from claude_sql.infrastructure import lance_store
from claude_sql.infrastructure.session_search import DuckDbSessionSearch
from claude_sql.infrastructure.settings import Settings
from claude_sql.infrastructure.transcript_reader import DuckDbTranscriptReader

# The conftest builders are module-level functions; reuse them directly.
from conftest import (
    make_assistant_msg,
    make_user_msg,
    write_session_jsonl,
)


def _settings_for(tmp_corpus: dict[str, Any], *, lance_uri: Path | None = None) -> Settings:
    """A Settings pinned to the tmp_corpus globs (+ optional lance store)."""
    kwargs: dict[str, Any] = {
        "default_glob": tmp_corpus["glob"],
        "subagent_glob": tmp_corpus["subagent_glob"],
        "subagent_meta_glob": tmp_corpus["subagent_meta_glob"],
    }
    if lance_uri is not None:
        kwargs["lance_uri"] = lance_uri
    return Settings(**kwargs)


# ---------------------------------------------------------------------------
# TranscriptReader — session_messages ordering + both layouts
# ---------------------------------------------------------------------------


def test_session_messages_flat_layout_ordering(tmp_corpus: dict[str, Any]) -> None:
    """Flat ``{sid}.jsonl`` session returns rows in chronological timeline order."""
    reader = DuckDbTranscriptReader(_settings_for(tmp_corpus))
    try:
        sid_one = tmp_corpus["session_ids"][0]
        rows = reader.session_messages(sid_one)
    finally:
        reader.close()

    # Session one: user text, assistant text, tool_use, tool_result, user text.
    kinds = [(r["kind"], r["role"]) for r in rows]
    assert ("text", "user") in kinds
    assert ("tool_use", "tool") in kinds
    assert ("tool_result", "tool") in kinds
    # Chronological: ts_iso is non-decreasing across the merged streams.
    ts_values = [r["ts"] for r in rows]
    assert ts_values == sorted(ts_values)
    # The tool_use carries its tool name in aux; tool_result its tool_use_id.
    tu = next(r for r in rows if r["kind"] == "tool_use")
    assert tu["aux"] == "Read"
    tr = next(r for r in rows if r["kind"] == "tool_result")
    assert tr["aux"] == "tu-a1"


def test_session_messages_part_dir_layout(tmp_path: Path, tmp_corpus: dict[str, Any]) -> None:
    """A part-dir ``{sid}/part-*.jsonl`` session is discovered via the pruned glob."""
    proj = tmp_path / "projects" / "proj-a"
    sid = "33333333-3333-3333-3333-333333333333"
    # S3SessionStore-style layout: the session id keys the *directory*.
    write_session_jsonl(
        proj / sid / "part-1700000000000-abc123.jsonl",
        messages=[
            make_user_msg(
                "p1",
                sid,
                "the part-dir session opening message is plenty long",
                ts="2026-05-01T08:00:00.000Z",
            ),
            make_assistant_msg(
                "p2",
                sid,
                ts="2026-05-01T08:00:05.000Z",
                content=[
                    {"type": "text", "text": "acknowledged and ready for the next instruction"}
                ],
            ),
        ],
    )
    reader = DuckDbTranscriptReader(_settings_for(tmp_corpus))
    try:
        rows = reader.session_messages(sid)
    finally:
        reader.close()
    assert len(rows) == 2
    assert [r["role"] for r in rows] == ["user", "assistant"]


def test_session_messages_unknown_session_is_empty(tmp_corpus: dict[str, Any]) -> None:
    """An absent session id yields an empty list, not an error."""
    reader = DuckDbTranscriptReader(_settings_for(tmp_corpus))
    try:
        assert reader.session_messages("does-not-exist-0000") == []
    finally:
        reader.close()


def test_session_ids_and_bounds(tmp_corpus: dict[str, Any]) -> None:
    """session_ids and session_bounds delegate to the loader over the corpus."""
    reader = DuckDbTranscriptReader(_settings_for(tmp_corpus))
    try:
        ids = reader.session_ids()
        bounds = reader.session_bounds()
    finally:
        reader.close()
    for sid in tmp_corpus["session_ids"]:
        assert sid in ids
        assert sid in bounds
        last_ts, _mtime = bounds[sid]
        assert last_ts is not None


def test_session_enumeration_includes_part_dir_sessions(
    tmp_path: Path, tmp_corpus: dict[str, Any]
) -> None:
    """DELTA-5: session_ids / session_bounds see ``{sid}/part-*.jsonl`` sessions.

    ``register_raw`` binds ``v_raw_events`` over the single flat glob; the
    full-corpus connection folds the part-dir layout in so enumeration surfaces
    both layouts, not just the flat ``{sid}.jsonl`` sessions.
    """
    proj = tmp_path / "projects" / "proj-a"
    part_sid = "77777777-7777-7777-7777-777777777777"
    write_session_jsonl(
        proj / part_sid / "part-1700000000000-abc.jsonl",
        messages=[
            make_user_msg(
                "pd1",
                part_sid,
                "a part-dir session that enumeration must discover too",
                ts="2026-05-02T08:00:00.000Z",
            ),
        ],
    )
    reader = DuckDbTranscriptReader(_settings_for(tmp_corpus))
    try:
        ids = reader.session_ids()
        bounds = reader.session_bounds()
    finally:
        reader.close()
    # Both the flat fixture sessions AND the new part-dir session appear.
    assert part_sid in ids
    assert part_sid in bounds
    for sid in tmp_corpus["session_ids"]:
        assert sid in ids


def test_config_root_override_derives_globs(tmp_path: Path) -> None:
    """config_root overrides the transcript root and derives the flat glob."""
    root = tmp_path / "altroot"
    sid = "44444444-4444-4444-4444-444444444444"
    write_session_jsonl(
        root / "projects" / "projX" / f"{sid}.jsonl",
        messages=[
            make_user_msg(
                "c1",
                sid,
                "config-root override session message that is long",
                ts="2026-06-01T00:00:00.000Z",
            ),
        ],
    )
    reader = DuckDbTranscriptReader(config_root=root)
    try:
        rows = reader.session_messages(sid)
    finally:
        reader.close()
    assert len(rows) == 1
    assert rows[0]["role"] == "user"


# ---------------------------------------------------------------------------
# read_turn_text — collapse contract
# ---------------------------------------------------------------------------


def test_read_turn_text_collapse_contract(tmp_corpus: dict[str, Any]) -> None:
    """read_turn_text folds tool markers inline into their parent message line.

    The consumer collapse contract: ONE line per message, with
    ``[tool_use:name]`` / ``[tool_result]`` appended inline after the text body
    of the SAME message; the raw ISO timestamp string (incl. ``.000Z``) is
    preserved verbatim.
    """
    reader = DuckDbTranscriptReader(_settings_for(tmp_corpus))
    try:
        text = reader.read_turn_text(tmp_corpus["session_ids"][0])
    finally:
        reader.close()
    lines = text.splitlines()
    # DELTA-1: the assistant's tool_use folds into its own message line, after
    # the text body — not a standalone marker line.
    assert lines == [
        "[user 2026-04-01T10:00:00.000Z] "
        "first message in session one — long enough to clear the filter",
        "[assistant 2026-04-01T10:00:10.000Z] ok let me check the file [tool_use:Read]",
        "[user 2026-04-01T10:00:12.000Z] [tool_result]",
        "[user 2026-04-01T10:00:30.000Z] thanks, that's exactly what I expected from the file read",
    ]
    # The tool_use marker must NOT carry the input payload.
    assert "/workspace/x.txt" not in text
    # DELTA-3: raw timestamp string carried through verbatim (no TIMESTAMP
    # round-trip that would drop the ``.000Z``).
    assert "[user 2026-04-01T10:00:00.000Z]" in text


def test_read_turn_text_short_turn_bypasses_floor(
    tmp_path: Path, tmp_corpus: dict[str, Any]
) -> None:
    """DELTA-2: a sub-32-char user turn ("screenshot?") renders in the seam.

    The ``messages_text`` view drops it (32-char HAVING floor for analytics),
    but ``read_turn_text`` reads the rawer projection so short turns survive.
    """
    proj = tmp_path / "projects" / "proj-a"
    sid = "66666666-6666-6666-6666-666666666666"
    write_session_jsonl(
        proj / f"{sid}.jsonl",
        messages=[make_user_msg("s1", sid, "screenshot?", ts="2026-04-04T10:00:00.000Z")],
    )
    reader = DuckDbTranscriptReader(_settings_for(tmp_corpus))
    try:
        text = reader.read_turn_text(sid)
    finally:
        reader.close()
    assert text == "[user 2026-04-04T10:00:00.000Z] screenshot?"


def test_read_turn_text_caps_are_applied(tmp_path: Path, tmp_corpus: dict[str, Any]) -> None:
    """DELTA-4: per-turn cap appends `` …`` (collapse parity), no notice text."""
    proj = tmp_path / "projects" / "proj-a"
    sid = "55555555-5555-5555-5555-555555555555"
    huge = "x" * 20_000
    write_session_jsonl(
        proj / f"{sid}.jsonl",
        messages=[make_user_msg("h1", sid, huge, ts="2026-04-03T10:00:00.000Z")],
    )
    reader = DuckDbTranscriptReader(_settings_for(tmp_corpus))
    try:
        # Default caps: per_turn 8000 clips the 20k body.
        text = reader.read_turn_text(sid)
    finally:
        reader.close()
    # The collapse per-turn marker is a trailing " …" (leading space), not "…(truncated)".
    assert text.endswith(" …")
    assert "(truncated)" not in text
    # The clipped body is at most ~8000 chars + the marker + the header.
    assert len(text) < 9000


# ---------------------------------------------------------------------------
# SessionSearch — fake embedder + tiny Lance store
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Deterministic 4-dim embedder: vector points at the query's first char."""

    model_id = "test-fake"
    provider = "test"
    dimension = 4

    def embed_query(self, text: str) -> list[float]:
        # Map by a simple keyword so nearest-neighbor ranking is predictable.
        if "read" in text.lower():
            return [1.0, 0.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0, 0.0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        return [[0.0, 0.0, 0.0, 0.0] for _ in texts]


def _seed_lance(lance_uri: Path, rows: list[tuple[str, list[float]]], *, dim: int = 4) -> None:
    """Seed a tiny Lance embeddings store with ``(uuid, vector)`` rows."""
    db = lance_store.connect_db(lance_uri)
    tbl = lance_store.open_or_create_table(db, dim=dim)
    df = pl.DataFrame(
        {
            "uuid": [u for u, _ in rows],
            "model": ["test-fake"] * len(rows),
            "dim": [dim] * len(rows),
            "embedding": [v for _, v in rows],
            "embedded_at": [datetime.now(UTC)] * len(rows),
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


def _uuid_of(reader: DuckDbTranscriptReader, sid: str, role: str) -> str:
    """Fetch a real message uuid from the corpus so the join lands."""
    rows = reader.session_messages(sid)
    return next(r["uuid"] for r in rows if r["role"] == role and r["kind"] == "text")


def test_search_returns_ranked_hits(tmp_path: Path, tmp_corpus: dict[str, Any]) -> None:
    """search embeds the query and returns cosine-ranked SearchHits."""
    lance_uri = tmp_path / "lance"
    settings = _settings_for(tmp_corpus, lance_uri=lance_uri)
    reader = DuckDbTranscriptReader(settings)
    try:
        sid_one = tmp_corpus["session_ids"][0]
        sid_two = tmp_corpus["session_ids"][1]
        u_one = _uuid_of(reader, sid_one, "user")
        u_two = _uuid_of(reader, sid_two, "user")
    finally:
        reader.close()

    # Row for sid_one points at axis 0; sid_two points at axis 1.
    _seed_lance(lance_uri, [(u_one, [1.0, 0.0, 0.0, 0.0]), (u_two, [0.0, 1.0, 0.0, 0.0])])

    search = DuckDbSessionSearch(settings, embedder=_FakeEmbedder())
    try:
        hits = search.search("please read the file", k=2)
    finally:
        search.close()

    assert len(hits) == 2
    assert all(isinstance(h, SearchHit) for h in hits)
    # "read" -> axis 0 -> sid_one's row is the nearest neighbor.
    assert hits[0].uuid == u_one
    assert hits[0].cosine_sim > hits[1].cosine_sim


def test_search_session_filter(tmp_path: Path, tmp_corpus: dict[str, Any]) -> None:
    """The session_id filter confines results to that session only."""
    lance_uri = tmp_path / "lance"
    settings = _settings_for(tmp_corpus, lance_uri=lance_uri)
    reader = DuckDbTranscriptReader(settings)
    try:
        sid_one = tmp_corpus["session_ids"][0]
        sid_two = tmp_corpus["session_ids"][1]
        u_one = _uuid_of(reader, sid_one, "user")
        u_two = _uuid_of(reader, sid_two, "user")
    finally:
        reader.close()

    _seed_lance(lance_uri, [(u_one, [1.0, 0.0, 0.0, 0.0]), (u_two, [0.0, 1.0, 0.0, 0.0])])

    search = DuckDbSessionSearch(settings, embedder=_FakeEmbedder())
    try:
        hits = search.search("please read the file", k=10, session_id=sid_two)
    finally:
        search.close()

    assert [h.session_id for h in hits] == [sid_two]
    assert hits[0].uuid == u_two


def test_search_empty_store_returns_empty(tmp_path: Path, tmp_corpus: dict[str, Any]) -> None:
    """An empty embeddings store yields ``[]`` (library semantics, no exit)."""
    lance_uri = tmp_path / "lance_empty"
    settings = _settings_for(tmp_corpus, lance_uri=lance_uri)
    search = DuckDbSessionSearch(settings, embedder=_FakeEmbedder())
    try:
        assert search.search("anything", k=5) == []
    finally:
        search.close()


def test_embed_query_delegates_to_embedder(tmp_corpus: dict[str, Any]) -> None:
    """embed_query returns the injected embedder's vector without opening a conn."""
    search = DuckDbSessionSearch(_settings_for(tmp_corpus), embedder=_FakeEmbedder())
    assert search.embed_query("please read") == [1.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# ClaudeSql facade
# ---------------------------------------------------------------------------


def test_facade_lazily_builds_ports(tmp_corpus: dict[str, Any]) -> None:
    """ClaudeSql builds reader/search lazily and caches each instance."""
    from claude_sql.composition import ClaudeSql

    facade = ClaudeSql(_settings_for(tmp_corpus))
    reader = facade.reader()
    assert isinstance(reader, DuckDbTranscriptReader)
    assert facade.reader() is reader  # cached
    search = facade.search()
    assert isinstance(search, DuckDbSessionSearch)
    assert facade.search() is search  # cached
    reader.close()
    search.close()


def test_facade_build_factories(tmp_corpus: dict[str, Any]) -> None:
    """The module-level factories return standalone port adapters."""
    from claude_sql.composition import build_reader, build_search

    reader = build_reader(_settings_for(tmp_corpus))
    assert isinstance(reader, DuckDbTranscriptReader)
    reader.close()
    search = build_search(_settings_for(tmp_corpus), embedder=_FakeEmbedder())
    assert isinstance(search, DuckDbSessionSearch)
    search.close()


def test_facade_query_returns_dataframe(tmp_corpus: dict[str, Any]) -> None:
    """ClaudeSql.query runs SQL over a full-registration connection."""
    from claude_sql.composition import ClaudeSql

    facade = ClaudeSql(_settings_for(tmp_corpus))
    df = facade.query("SELECT CAST(session_id AS VARCHAR) AS sid FROM sessions")
    assert isinstance(df, pl.DataFrame)
    sids = set(df["sid"].to_list())
    for sid in tmp_corpus["session_ids"]:
        assert sid in sids


def test_importing_composition_pulls_no_duckdb() -> None:
    """A bare ``import claude_sql.composition`` must not import duckdb.

    The facade is the sibling import surface; keeping its module top light means
    ``import claude_sql`` / ``import claude_sql.composition`` never drags in the
    heavy DuckDB / adapter stack until a method actually runs.
    """
    code = (
        "import sys\n"
        "import claude_sql\n"
        "import claude_sql.composition\n"
        "assert 'duckdb' not in sys.modules, 'duckdb imported at composition import time'\n"
        "assert hasattr(claude_sql, 'ClaudeSql'), 'lazy ClaudeSql export missing'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout

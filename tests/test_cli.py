"""Coverage tests for ``claude_sql.cli``.

Strategy: invoke each subcommand's underlying function directly with a
``Common`` instance, point every cache path at ``tmp_path`` via
monkeypatched env vars (so the user's real ``~/.claude/`` is never
touched), and stay in the cost-guarded ``--dry-run`` path for any
Bedrock-hitting commands.

Tests must be hermetic — no real Bedrock, no network, no real ``~/.claude``
writes — and finish in well under three seconds each.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from claude_sql import cli
from claude_sql.cli import Common
from claude_sql.config import Settings
from claude_sql.output import OutputFormat

# ---------------------------------------------------------------------------
# Cache redirection — auto-applied to every test in this module.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def cache_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect every claude-sql cache path under ``tmp_path``.

    Touches the env vars Settings reads from so neither the test nor any
    code path it exercises ends up writing to the real ``~/.claude``.
    Returns the cache root so individual tests can poke at it directly.
    """
    cache = tmp_path / "claude_home"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "duckdb_tmp").mkdir(parents=True, exist_ok=True)
    (cache / "profiling").mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("CLAUDE_SQL_EMBEDDINGS_PARQUET_PATH", str(cache / "embeddings"))
    monkeypatch.setenv("CLAUDE_SQL_CLASSIFICATIONS_PARQUET_PATH", str(cache / "classifications"))
    monkeypatch.setenv("CLAUDE_SQL_TRAJECTORY_PARQUET_PATH", str(cache / "trajectory"))
    monkeypatch.setenv("CLAUDE_SQL_CONFLICTS_PARQUET_PATH", str(cache / "conflicts"))
    monkeypatch.setenv("CLAUDE_SQL_USER_FRICTION_PARQUET_PATH", str(cache / "user_friction"))
    monkeypatch.setenv("CLAUDE_SQL_CLUSTERS_PARQUET_PATH", str(cache / "clusters.parquet"))
    monkeypatch.setenv(
        "CLAUDE_SQL_CLUSTER_TERMS_PARQUET_PATH", str(cache / "cluster_terms.parquet")
    )
    monkeypatch.setenv("CLAUDE_SQL_COMMUNITIES_PARQUET_PATH", str(cache / "communities.parquet"))
    monkeypatch.setenv(
        "CLAUDE_SQL_SKILLS_CATALOG_PARQUET_PATH",
        str(cache / "skills_catalog.parquet"),
    )
    monkeypatch.setenv("CLAUDE_SQL_USER_SKILLS_DIR", str(cache / "skills"))
    monkeypatch.setenv("CLAUDE_SQL_PLUGINS_CACHE_DIR", str(cache / "plugins_cache"))
    monkeypatch.setenv("CLAUDE_SQL_CHECKPOINT_DB_PATH", str(cache / "claude_sql.duckdb"))
    monkeypatch.setenv("CLAUDE_SQL_HNSW_DB_PATH", str(cache / "hnsw.duckdb"))
    monkeypatch.setenv("CLAUDE_SQL_DUCKDB_TEMP_DIR", str(cache / "duckdb_tmp"))
    # Trim concurrency to keep things fast and deterministic in tests.
    monkeypatch.setenv("CLAUDE_SQL_EMBED_CONCURRENCY", "2")
    monkeypatch.setenv("CLAUDE_SQL_LLM_CONCURRENCY", "2")
    monkeypatch.setenv("CLAUDE_SQL_BATCH_SIZE", "4")
    monkeypatch.setenv("CLAUDE_SQL_DUCKDB_THREADS", "2")
    monkeypatch.setenv("CLAUDE_SQL_DUCKDB_MEMORY_LIMIT", "1GB")
    # Force HOME to tmp_path so binding helpers and ``_profile_path_for``
    # don't write under the real user home.
    monkeypatch.setenv("HOME", str(tmp_path / "fake_home"))
    (tmp_path / "fake_home" / ".claude").mkdir(parents=True, exist_ok=True)
    return cache


@pytest.fixture(autouse=True)
def _purge_meta_glob_env() -> Iterator[None]:
    """Snapshot/restore ``CLAUDE_SQL_SUBAGENT_META_GLOB`` around every test.

    ``_common`` writes the env var directly (no CLI flag exists for it), so
    we must reset on teardown to avoid leaking the fixture path into other
    test modules — notably ``test_team_corpus`` re-reads ``Settings()``.
    """
    import os as _os

    prior = _os.environ.get("CLAUDE_SQL_SUBAGENT_META_GLOB")
    yield
    if prior is None:
        _os.environ.pop("CLAUDE_SQL_SUBAGENT_META_GLOB", None)
    else:
        _os.environ["CLAUDE_SQL_SUBAGENT_META_GLOB"] = prior


def _common(tmp_corpus: dict[str, Any], fmt: OutputFormat = OutputFormat.JSON) -> Common:
    """Construct a ``Common`` pointing at the fixture corpus.

    Side effect: stamps ``CLAUDE_SQL_SUBAGENT_META_GLOB`` into the env via
    :data:`os.environ` so :class:`Settings` constructed inside the CLI sees
    the fixture's stub directory and not the user's real ``~/.claude``.
    ``Common`` itself only carries the two transcript globs that have CLI
    flags; the meta glob has no flag, hence the env-var route. The
    autouse ``_purge_meta_glob_env`` fixture restores the prior value on
    teardown so the env var doesn't bleed into unrelated test modules.
    """
    import os

    os.environ["CLAUDE_SQL_SUBAGENT_META_GLOB"] = tmp_corpus["subagent_meta_glob"]
    return Common(
        verbose=False,
        quiet=True,
        glob=tmp_corpus["glob"],
        subagent_glob=tmp_corpus["subagent_glob"],
        format=fmt,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def test_resolve_settings_no_common_returns_defaults() -> None:
    settings = cli._resolve_settings(None)
    assert isinstance(settings, Settings)


def test_resolve_settings_applies_common_overrides(tmp_corpus: dict[str, Any]) -> None:
    common = _common(tmp_corpus)
    settings = cli._resolve_settings(common)
    assert settings.default_glob == tmp_corpus["glob"]
    assert settings.subagent_glob == tmp_corpus["subagent_glob"]


def test_resolve_settings_rejects_multistar_glob(
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad = Common(glob="/a/**/b/**/c.jsonl", format=OutputFormat.JSON)
    with pytest.raises(SystemExit) as exc:
        cli._resolve_settings(bad)
    assert exc.value.code == 64
    err = capsys.readouterr().err
    assert "more than one" in err or "**" in err


def test_resolve_memory_limit_passes_absolute_through() -> None:
    assert cli._resolve_memory_limit("4GB") == "4GB"
    assert cli._resolve_memory_limit("  2GiB  ") == "2GiB"


def test_resolve_memory_limit_handles_percent_strings() -> None:
    out = cli._resolve_memory_limit("50%")
    assert out.endswith("MiB")


def test_emit_worker_result_dict(capsys: pytest.CaptureFixture[str]) -> None:
    plan = {"pipeline": "embed", "candidates": 5, "dry_run": True}
    cli._emit_worker_result(plan, Common(format=OutputFormat.JSON), pipeline="embed")
    out = capsys.readouterr().out
    assert json.loads(out) == plan


def test_emit_worker_result_int(capsys: pytest.CaptureFixture[str]) -> None:
    cli._emit_worker_result(7, Common(format=OutputFormat.JSON), pipeline="classify")
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"pipeline": "classify", "rows_processed": 7, "dry_run": False}


def test_describe_cache_entry_for_missing_path(tmp_path: Path) -> None:
    entry = cli._describe_cache_entry("missing", tmp_path / "nope")
    assert entry == {"name": "missing", "path": str(tmp_path / "nope"), "exists": False}


def test_describe_cache_entry_for_sharded_dir(tmp_path: Path) -> None:
    cache_dir = tmp_path / "embeddings"
    cache_dir.mkdir()
    df = pl.DataFrame({"uuid": ["a", "b", "c"]})
    df.write_parquet(cache_dir / "part-1.parquet")
    entry = cli._describe_cache_entry("embeddings", cache_dir)
    assert entry["exists"] is True
    assert entry["rows"] == 3
    assert entry["bytes"] > 0


def test_describe_cache_entry_for_legacy_file(tmp_path: Path) -> None:
    target = tmp_path / "legacy.parquet"
    pl.DataFrame({"uuid": ["x"]}).write_parquet(target)
    entry = cli._describe_cache_entry("legacy", target)
    assert entry["exists"] is True
    assert entry["rows"] == 1


def test_describe_cache_entry_empty_directory(tmp_path: Path) -> None:
    cache_dir = tmp_path / "blank_cache"
    cache_dir.mkdir()
    entry = cli._describe_cache_entry("blank", cache_dir)
    assert entry["exists"] is True
    assert entry["rows"] == 0
    assert entry["bytes"] == 0


def test_describe_checkpoint_entry_missing(tmp_path: Path) -> None:
    entry = cli._describe_checkpoint_entry(tmp_path / "missing.duckdb")
    assert entry["exists"] is False


def test_describe_checkpoint_entry_existing(tmp_path: Path) -> None:
    db = tmp_path / "ckpt.duckdb"
    db.write_bytes(b"\x00" * 64)  # not a real DuckDB file but still has a stat
    entry = cli._describe_checkpoint_entry(db)
    assert entry["exists"] is True
    assert "bytes" in entry
    # rows is None when DuckDB can't open the corrupt body — that's fine,
    # the helper catches duckdb.Error.
    assert "rows" in entry


def test_capture_profile_returns_path(tmp_corpus: dict[str, Any]) -> None:
    settings = cli._resolve_settings(_common(tmp_corpus))
    con = cli._open_connection_full(settings)
    try:
        out = cli._capture_profile(con, label="unit-test")
        assert out.parent.exists()
        assert out.suffix == ".json"
    finally:
        con.close()


def test_review_sheet_format_explicit_json_passes_through() -> None:
    assert cli._review_sheet_format(Common(format=OutputFormat.JSON)) is cli.RenderFormat.JSON


def test_review_sheet_format_auto_offtty_resolves_to_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    assert cli._review_sheet_format(Common(format=OutputFormat.AUTO)) is cli.RenderFormat.JSON


def test_review_sheet_format_auto_tty_resolves_to_markdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    assert cli._review_sheet_format(Common(format=OutputFormat.AUTO)) is cli.RenderFormat.MARKDOWN


# ---------------------------------------------------------------------------
# Read-only commands: query / explain / schema / list-cache
# ---------------------------------------------------------------------------


def test_query_happy_path(tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    cli.query("SELECT 1 AS one", common=_common(tmp_corpus))
    out = capsys.readouterr().out
    assert "1" in out


def test_query_against_messages_view(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    cli.query("SELECT COUNT(*) AS n FROM messages", common=_common(tmp_corpus))
    payload = json.loads(capsys.readouterr().out)
    # JSON shape is a list of row dicts.
    assert isinstance(payload, list)
    assert payload and "n" in payload[0]


def test_query_parse_error_exits_64(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.query("SELEKT broken", common=_common(tmp_corpus))
    assert exc.value.code == 64
    assert "error" in capsys.readouterr().err


def test_query_unknown_table_exits_65(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.query(
            "SELECT * FROM does_not_exist_anywhere_xyz",
            common=_common(tmp_corpus),
        )
    assert exc.value.code == 65


def test_query_with_profile_json(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    cli.query("SELECT 1", profile_json=True, common=_common(tmp_corpus))
    capsys.readouterr()  # drain — we just need the command to run cleanly


def test_explain_default_static(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    cli.explain("SELECT 1", common=_common(tmp_corpus))
    payload = json.loads(capsys.readouterr().out)
    assert "plan" in payload


def test_explain_analyze_path(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    cli.explain("SELECT 1", analyze=True, common=_common(tmp_corpus))
    payload = json.loads(capsys.readouterr().out)
    assert "plan" in payload


def test_explain_with_profile_json(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    cli.explain("SELECT 1", profile_json=True, common=_common(tmp_corpus))
    capsys.readouterr()


def test_explain_table_format_highlights_markers(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    cli.explain("SELECT 1", common=_common(tmp_corpus, fmt=OutputFormat.TABLE))
    capsys.readouterr()  # exercised the TTY branch


def test_schema_json_output(tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    cli.schema(common=_common(tmp_corpus))
    payload = json.loads(capsys.readouterr().out)
    assert "views" in payload
    assert "macros" in payload


def test_schema_table_output(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    cli.schema(common=_common(tmp_corpus, fmt=OutputFormat.TABLE))
    out = capsys.readouterr().out
    assert "Macros" in out


def test_list_cache_json(tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    cli.list_cache(common=_common(tmp_corpus))
    payload = json.loads(capsys.readouterr().out)
    names = {entry["name"] for entry in payload}
    assert "embeddings" in names
    assert "session_checkpoint" in names


def test_list_cache_ndjson(tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    cli.list_cache(common=_common(tmp_corpus, fmt=OutputFormat.NDJSON))
    out = capsys.readouterr().out.strip().splitlines()
    parsed = [json.loads(line) for line in out]
    assert any(entry["name"] == "embeddings" for entry in parsed)


def test_list_cache_csv(tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    cli.list_cache(common=_common(tmp_corpus, fmt=OutputFormat.CSV))
    assert "embeddings" in capsys.readouterr().out


def test_list_cache_table(tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    cli.list_cache(common=_common(tmp_corpus, fmt=OutputFormat.TABLE))
    capsys.readouterr()


# ---------------------------------------------------------------------------
# Cache sub-app: compact / migrate
# ---------------------------------------------------------------------------


def test_cache_compact_dry_run_skips_when_no_parts(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    cli.cache_compact(common=_common(tmp_corpus))
    payload = json.loads(capsys.readouterr().out)
    actions = {entry["action"] for entry in payload}
    assert actions == {"skip"}


def test_cache_compact_unknown_name_exits_64(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.cache_compact(name="not_a_cache", common=_common(tmp_corpus))
    assert exc.value.code == 64
    assert "Unknown cache name" in capsys.readouterr().err


def test_cache_compact_dry_run_with_two_parts(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str], cache_root: Path
) -> None:
    cache_dir = cache_root / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"uuid": ["a"]}).write_parquet(cache_dir / "part-1.parquet")
    pl.DataFrame({"uuid": ["b"]}).write_parquet(cache_dir / "part-2.parquet")

    cli.cache_compact(name="embeddings", common=_common(tmp_corpus))
    payload = json.loads(capsys.readouterr().out)
    [entry] = payload
    assert entry["action"] == "would_compact"
    assert entry["parts"] == 2


def test_cache_compact_real_run_consolidates(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str], cache_root: Path
) -> None:
    cache_dir = cache_root / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"uuid": ["a"]}).write_parquet(cache_dir / "part-1.parquet")
    pl.DataFrame({"uuid": ["b"]}).write_parquet(cache_dir / "part-2.parquet")

    cli.cache_compact(name="embeddings", dry_run=False, common=_common(tmp_corpus))
    payload = json.loads(capsys.readouterr().out)
    [entry] = payload
    assert entry["action"] == "compacted"
    # Original parts were unlinked; one compacted shard remains.
    remaining = sorted(cache_dir.glob("part-*.parquet"))
    assert len(remaining) == 1


def test_cache_compact_table_format(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    cli.cache_compact(common=_common(tmp_corpus, fmt=OutputFormat.TABLE))
    capsys.readouterr()


def test_cache_compact_csv_and_ndjson(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    cli.cache_compact(common=_common(tmp_corpus, fmt=OutputFormat.NDJSON))
    capsys.readouterr()
    cli.cache_compact(common=_common(tmp_corpus, fmt=OutputFormat.CSV))
    capsys.readouterr()


def test_cache_migrate_no_legacy_files_skips(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    cli.cache_migrate(common=_common(tmp_corpus))
    payload = json.loads(capsys.readouterr().out)
    assert payload
    assert {entry["action"] for entry in payload} == {"skip"}


def test_cache_migrate_dry_run_finds_legacy(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str], cache_root: Path
) -> None:
    legacy = cache_root / "embeddings.parquet"
    pl.DataFrame({"uuid": ["a"]}).write_parquet(legacy)
    cli.cache_migrate(common=_common(tmp_corpus))
    payload = json.loads(capsys.readouterr().out)
    embed_entry = next(e for e in payload if e["name"] == "embeddings")
    assert embed_entry["action"] == "would_move"


def test_cache_migrate_real_run_moves_legacy(tmp_corpus: dict[str, Any], cache_root: Path) -> None:
    legacy = cache_root / "embeddings.parquet"
    pl.DataFrame({"uuid": ["a"]}).write_parquet(legacy)
    cli.cache_migrate(dry_run=False, common=_common(tmp_corpus))
    assert not legacy.exists()
    moved = list((cache_root / "embeddings").glob("part-*.parquet"))
    assert len(moved) == 1


def test_cache_migrate_table_csv_ndjson(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    cli.cache_migrate(common=_common(tmp_corpus, fmt=OutputFormat.TABLE))
    capsys.readouterr()
    cli.cache_migrate(common=_common(tmp_corpus, fmt=OutputFormat.NDJSON))
    capsys.readouterr()
    cli.cache_migrate(common=_common(tmp_corpus, fmt=OutputFormat.CSV))
    capsys.readouterr()


# ---------------------------------------------------------------------------
# Skills sub-app: sync / ls
# ---------------------------------------------------------------------------


def test_skills_sync_dry_run(tmp_corpus: dict[str, Any]) -> None:
    cli.skills_sync(dry_run=True, common=_common(tmp_corpus))


def test_skills_sync_writes_parquet(tmp_corpus: dict[str, Any], cache_root: Path) -> None:
    cli.skills_sync(common=_common(tmp_corpus))
    assert (cache_root / "skills_catalog.parquet").exists()


def test_skills_ls_missing_parquet_exits_2(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.skills_ls(common=_common(tmp_corpus))
    assert exc.value.code == 2
    assert "skills catalog parquet missing" in capsys.readouterr().err


def test_skills_ls_after_sync_emits_rows(
    tmp_corpus: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
    cache_root: Path,
) -> None:
    cli.skills_sync(common=_common(tmp_corpus))
    capsys.readouterr()
    cli.skills_ls(common=_common(tmp_corpus))
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)


def test_skills_ls_with_filters(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    cli.skills_sync(common=_common(tmp_corpus))
    capsys.readouterr()
    cli.skills_ls(kind="builtin", plugin=None, common=_common(tmp_corpus, fmt=OutputFormat.NDJSON))
    capsys.readouterr()
    cli.skills_ls(kind="builtin", common=_common(tmp_corpus, fmt=OutputFormat.CSV))
    capsys.readouterr()
    cli.skills_ls(common=_common(tmp_corpus, fmt=OutputFormat.TABLE))
    capsys.readouterr()


# ---------------------------------------------------------------------------
# Bedrock-touching commands — exercised in dry-run.
# ---------------------------------------------------------------------------


def test_embed_dry_run(tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    cli.embed(dry_run=True, common=_common(tmp_corpus))
    payload = json.loads(capsys.readouterr().out)
    assert payload["pipeline"] == "embed"
    assert payload["dry_run"] is True


def test_classify_dry_run(tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    cli.classify(common=_common(tmp_corpus))
    payload = json.loads(capsys.readouterr().out)
    assert payload["pipeline"] == "classify"
    assert payload["dry_run"] is True


def test_trajectory_dry_run(tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    cli.trajectory(common=_common(tmp_corpus))
    payload = json.loads(capsys.readouterr().out)
    assert payload["pipeline"] == "trajectory"


def test_conflicts_dry_run(tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    cli.conflicts(common=_common(tmp_corpus))
    payload = json.loads(capsys.readouterr().out)
    assert payload["pipeline"] == "conflicts"


def test_friction_dry_run(tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    cli.friction(common=_common(tmp_corpus))
    payload = json.loads(capsys.readouterr().out)
    assert payload["pipeline"] == "friction"


def test_search_no_embeddings_exits_2(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.search("anything", common=_common(tmp_corpus))
    assert exc.value.code == 2
    assert "No embeddings yet" in capsys.readouterr().err


def test_search_with_mocked_embed(
    tmp_corpus: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    cache_root: Path,
) -> None:
    """Seed the embeddings parquet manually, mock embed_query, run search."""
    settings = cli._resolve_settings(_common(tmp_corpus))
    dim = int(settings.output_dimension)

    # Pull message uuids from the fixture corpus.
    con = cli._open_connection_full(settings)
    try:
        rows = con.execute("SELECT uuid FROM messages_text LIMIT 4").fetchall()
    finally:
        con.close()
    assert rows, "fixture must produce at least one message"

    embed_dir = settings.embeddings_parquet_path
    embed_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {
            "uuid": [str(r[0]) for r in rows],
            "model": [settings.active_model_id] * len(rows),
            "dim": [dim] * len(rows),
            "embedding": [[float(i % 5) for i in range(dim)] for _ in range(len(rows))],
        },
        schema={
            "uuid": pl.String,
            "model": pl.String,
            "dim": pl.UInt16,
            "embedding": pl.Array(pl.Float32, dim),
        },
    )
    df.write_parquet(embed_dir / "part-1.parquet")

    # Drop any pre-built HNSW store from the first connection — the
    # subsequent ``search`` call rebuilds against the parquet we just wrote.
    if settings.hnsw_db_path.exists():
        settings.hnsw_db_path.unlink()

    fake_qv = [float(i % 3) for i in range(dim)]
    monkeypatch.setattr(cli, "embed_query", lambda text, *, settings: fake_qv)

    cli.search("hello world", k=2, common=_common(tmp_corpus))
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert isinstance(payload, list)
    assert all("sim" in row for row in payload)


# ---------------------------------------------------------------------------
# Cluster / terms / community — error paths only (real fits cost ~40 s).
# ---------------------------------------------------------------------------


def test_cluster_without_embeddings_raises(tmp_corpus: dict[str, Any]) -> None:
    with pytest.raises(FileNotFoundError):
        cli.cluster(common=_common(tmp_corpus))


def test_terms_without_clusters_raises(tmp_corpus: dict[str, Any]) -> None:
    with pytest.raises(FileNotFoundError):
        cli.terms(common=_common(tmp_corpus))


def test_community_without_embeddings_raises(tmp_corpus: dict[str, Any]) -> None:
    # Community detection needs the embeddings parquet — without it we get
    # FileNotFoundError or a duckdb / runtime error.
    import duckdb as _duckdb

    with pytest.raises((FileNotFoundError, _duckdb.Error, RuntimeError, ValueError)):
        cli.community(common=_common(tmp_corpus))


# ---------------------------------------------------------------------------
# Composite analyze pipeline — dry-run chains every stage.
# ---------------------------------------------------------------------------


def test_analyze_dry_run_skip_heavy_stages(tmp_corpus: dict[str, Any]) -> None:
    """All Bedrock-bearing stages stay in dry-run; heavy clustering is skipped."""
    cli.analyze(
        dry_run=True,
        skip_cluster=True,
        skip_community=True,
        common=_common(tmp_corpus),
    )


def test_analyze_skip_everything_runs_skills_only(tmp_corpus: dict[str, Any]) -> None:
    cli.analyze(
        dry_run=True,
        skip_embed=True,
        skip_classify=True,
        skip_trajectory=True,
        skip_conflicts=True,
        skip_friction=True,
        skip_cluster=True,
        skip_community=True,
        common=_common(tmp_corpus),
    )


def test_analyze_skip_all_including_skills(tmp_corpus: dict[str, Any]) -> None:
    cli.analyze(
        dry_run=True,
        skip_embed=True,
        skip_classify=True,
        skip_trajectory=True,
        skip_conflicts=True,
        skip_friction=True,
        skip_cluster=True,
        skip_community=True,
        skip_skills_sync=True,
        common=_common(tmp_corpus),
    )


# ---------------------------------------------------------------------------
# judges / freeze / replay / blind-handover / kappa
# ---------------------------------------------------------------------------


def test_judges_cmd_lists_catalog(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    cli.judges_cmd(common=_common(tmp_corpus, fmt=OutputFormat.JSON))
    out = capsys.readouterr().out
    payload = json.loads(out)
    # JSON shape from emit_dataframe is a list of row dicts.
    assert isinstance(payload, list)
    shortnames = {row["shortname"] for row in payload}
    assert "kimi-k2.5" in shortnames


def test_freeze_then_replay_roundtrip(
    tmp_corpus: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``freeze`` writes to ~/.claude/studies/<sha>/ — HOME is already
    # pinned to tmp_path/fake_home by the autouse fixture.
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text(
        "axes:\n"
        "  - name: clarity\n"
        "    description: is the answer clear\n"
        "    detector_vs_grader: grader\n"
        "    levels:\n"
        "      0: not clear\n"
        "      1: clear\n",
        encoding="utf-8",
    )
    cli.freeze_cmd(rubric, panel="kimi-k2.5,deepseek-v3.2", common=_common(tmp_corpus))
    out = capsys.readouterr().out
    payload = json.loads(out)
    sha = payload["manifest_sha"]

    cli.replay_cmd(sha, common=_common(tmp_corpus))
    replayed = json.loads(capsys.readouterr().out)
    assert replayed["manifest_sha"] == sha


def test_freeze_empty_panel_raises(tmp_corpus: dict[str, Any], tmp_path: Path) -> None:
    from claude_sql.output import InputValidationError

    rubric = tmp_path / "rubric.yaml"
    rubric.write_text("axes:\n  - name: x\n    levels:\n      0: a\n", encoding="utf-8")
    with pytest.raises(InputValidationError):
        cli.freeze_cmd(rubric, panel="", common=_common(tmp_corpus))


def test_blind_handover_cmd_strips_text(tmp_corpus: dict[str, Any], tmp_path: Path) -> None:
    in_path = tmp_path / "in.parquet"
    out_path = tmp_path / "out.parquet"
    pl.DataFrame(
        {
            "session_id": ["s1", "s2"],
            "text": ["U12345678 said hello", "plain text"],
        }
    ).write_parquet(in_path)

    cli.blind_handover_cmd(in_path, out_path, common=_common(tmp_corpus))
    assert out_path.exists()
    df = pl.read_parquet(out_path)
    assert "original_hash" in df.columns
    assert "[user]" in df["text"][0]


def test_blind_handover_missing_columns_raises(tmp_corpus: dict[str, Any], tmp_path: Path) -> None:
    from claude_sql.output import InputValidationError

    in_path = tmp_path / "bad.parquet"
    out_path = tmp_path / "out.parquet"
    pl.DataFrame({"session_id": ["s1"]}).write_parquet(in_path)
    with pytest.raises(InputValidationError):
        cli.blind_handover_cmd(in_path, out_path, common=_common(tmp_corpus))


def test_judge_cmd_dry_run(
    tmp_corpus: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Set up a frozen study under HOME-pinned ~/.claude/studies/.
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text(
        "axes:\n"
        "  - name: clarity\n"
        "    description: is it clear\n"
        "    detector_vs_grader: grader\n"
        "    levels:\n"
        "      0: no\n"
        "      1: yes\n",
        encoding="utf-8",
    )
    cli.freeze_cmd(rubric, panel="kimi-k2.5", common=_common(tmp_corpus))
    sha = json.loads(capsys.readouterr().out)["manifest_sha"]

    sessions = tmp_path / "sessions.parquet"
    pl.DataFrame({"session_id": ["s1"], "text": ["hello world"]}).write_parquet(sessions)

    out_pq = tmp_path / "scores.parquet"
    cli.judge_cmd(
        sha,
        sessions_parquet=sessions,
        output_parquet=out_pq,
        dry_run=True,
        common=_common(tmp_corpus),
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["n_sessions"] == 1


def test_judge_cmd_missing_columns_raises(
    tmp_corpus: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text("axes:\n  - name: x\n    levels:\n      0: a\n      1: b\n", encoding="utf-8")
    cli.freeze_cmd(rubric, panel="kimi-k2.5", common=_common(tmp_corpus))
    sha = json.loads(capsys.readouterr().out)["manifest_sha"]

    from claude_sql.output import InputValidationError

    bad = tmp_path / "bad.parquet"
    pl.DataFrame({"session_id": ["s1"]}).write_parquet(bad)
    with pytest.raises(InputValidationError):
        cli.judge_cmd(
            sha,
            sessions_parquet=bad,
            output_parquet=tmp_path / "out.parquet",
            common=_common(tmp_corpus),
        )


def test_ungrounded_cmd_emits_summary(
    tmp_corpus: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text("axes:\n  - name: x\n    levels:\n      0: a\n      1: b\n", encoding="utf-8")
    cli.freeze_cmd(rubric, panel="kimi-k2.5", common=_common(tmp_corpus))
    sha = json.loads(capsys.readouterr().out)["manifest_sha"]

    turns = tmp_path / "turns.parquet"
    pl.DataFrame(
        {
            "session_id": ["s1"],
            "turn_idx": [0],
            "assistant_text": ["I read /tmp/x.txt"],
            "tool_output_text": ["contents"],
        }
    ).write_parquet(turns)

    cli.ungrounded_cmd(
        sha,
        turns_parquet=turns,
        output_parquet=tmp_path / "ungrounded.parquet",
        common=_common(tmp_corpus),
    )
    capsys.readouterr()  # summary printed via emit_dataframe


def test_ungrounded_cmd_missing_columns_raises(
    tmp_corpus: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text("axes:\n  - name: x\n    levels:\n      0: a\n      1: b\n", encoding="utf-8")
    cli.freeze_cmd(rubric, panel="kimi-k2.5", common=_common(tmp_corpus))
    sha = json.loads(capsys.readouterr().out)["manifest_sha"]

    from claude_sql.output import InputValidationError

    bad = tmp_path / "bad.parquet"
    pl.DataFrame({"session_id": ["s1"]}).write_parquet(bad)
    with pytest.raises(InputValidationError):
        cli.ungrounded_cmd(
            sha,
            turns_parquet=bad,
            output_parquet=tmp_path / "out.parquet",
            common=_common(tmp_corpus),
        )


def test_kappa_cmd_under_floor_exits_66(
    tmp_corpus: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 3 judges scoring 4 sessions on one axis with deliberate disagreement.
    rows = []
    for i, sid in enumerate(["s1", "s2", "s3", "s4"]):
        for j, judge in enumerate(["jA", "jB", "jC"]):
            rows.append(
                {
                    "session_id": sid,
                    "axis": "clarity",
                    "judge_shortname": judge,
                    "score": (i + j) % 2,
                }
            )
    scores = tmp_path / "scores.parquet"
    pl.DataFrame(rows).write_parquet(scores)

    with pytest.raises(SystemExit) as exc:
        cli.kappa_cmd(
            scores,
            bootstrap=20,
            floor=0.99,
            common=_common(tmp_corpus),
        )
    # 66 only fires when at least one axis has below_floor=true; the
    # disagreement above should drive Fleiss kappa under 0.99.
    payload = json.loads(capsys.readouterr().out)
    assert any(row["below_floor"] for row in payload["fleiss"])
    assert exc.value.code == 66


def test_kappa_cmd_passes_when_floor_zero(
    tmp_corpus: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = [
        {
            "session_id": sid,
            "axis": "clarity",
            "judge_shortname": judge,
            "score": 1,
        }
        for sid in ["s1", "s2", "s3"]
        for judge in ["jA", "jB", "jC"]
    ]
    scores = tmp_path / "scores.parquet"
    pl.DataFrame(rows).write_parquet(scores)
    cli.kappa_cmd(scores, bootstrap=10, floor=-1.0, common=_common(tmp_corpus))
    payload = json.loads(capsys.readouterr().out)
    assert "fleiss" in payload


# ---------------------------------------------------------------------------
# bind / resolve / review-sheet
# ---------------------------------------------------------------------------


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=repo, check=True)
    (repo / "README").write_text("hello")
    subprocess.run(["git", "add", "README"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return repo


def test_bind_cmd_no_active_transcript(
    tmp_corpus: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No transcript under HOME-pinned ~/.claude/projects → emits {bound:false}."""
    repo = _init_repo(tmp_path)
    monkey_cwd = repo
    # Run from inside the repo so binding.find_active_transcript looks for
    # transcripts under the projectified repo path — none exist, so the
    # command should emit {"bound": false, "reason": "no-active-transcript"}.
    import os as _os

    prev = Path.cwd()
    _os.chdir(monkey_cwd)
    try:
        cli.bind_cmd(repo=repo, common=_common(tmp_corpus))
    finally:
        _os.chdir(prev)
    payload = json.loads(capsys.readouterr().out)
    assert payload["bound"] is False
    assert payload["reason"] == "no-active-transcript"


def test_resolve_cmd_no_binding_exits_2(
    tmp_corpus: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _init_repo(tmp_path)
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    with pytest.raises(SystemExit) as exc:
        cli.resolve_cmd(sha, repo=repo, common=_common(tmp_corpus))
    assert exc.value.code == 2
    capsys.readouterr()


def test_resolve_cmd_unknown_sha_exits_65(
    tmp_corpus: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _init_repo(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli.resolve_cmd(
            "deadbeefdeadbeef",
            repo=repo,
            common=_common(tmp_corpus),
        )
    # The sha doesn't exist; git invocation fails → catalog_error 65.
    assert exc.value.code == 65
    capsys.readouterr()


def test_resolve_cmd_all_sources_dict(
    tmp_corpus: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _init_repo(tmp_path)
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    cli.resolve_cmd(sha, repo=repo, all_sources=True, common=_common(tmp_corpus))
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"trailer": None, "note": None}


def test_review_sheet_cmd_no_binding_exits_2(
    tmp_corpus: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _init_repo(tmp_path)
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    with pytest.raises(SystemExit) as exc:
        cli.review_sheet_cmd(sha, repo=repo, common=_common(tmp_corpus))
    assert exc.value.code == 2
    capsys.readouterr()


def test_review_sheet_cmd_unknown_sha_exits_65(
    tmp_corpus: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _init_repo(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli.review_sheet_cmd("deadbeef", repo=repo, common=_common(tmp_corpus))
    assert exc.value.code == 65
    capsys.readouterr()


def test_review_sheet_cmd_dry_run(
    tmp_corpus: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the dry-run plan emission by mocking the binding lookup."""
    from claude_sql import binding as _binding_mod

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text('{"sessionId": "s1"}\n', encoding="utf-8")
    fake_binding = _binding_mod.TranscriptBinding(
        digest="sha256:" + "0" * 64,
        uri=transcript.resolve().as_uri(),
        agent_runtime="claude-code/test",
        transcript_id="s1",
        captured_at="2026-01-01T00:00:00+00:00",
    )
    monkeypatch.setattr(
        cli._binding, "resolve_commit_to_transcript", lambda sha, repo=None: fake_binding
    )
    cli.review_sheet_cmd(
        "abcdef0",
        dry_run=True,
        common=_common(tmp_corpus, fmt=OutputFormat.JSON),
    )
    payload = json.loads(capsys.readouterr().out)
    assert "commit_sha" in payload
    assert payload["dry_run"] is True


def test_review_sheet_cmd_renders_markdown_on_tty(
    tmp_corpus: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Success path: TTY + AUTO format → render_markdown branch fires."""
    from claude_sql import binding as _binding_mod

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text('{"sessionId": "s1"}\n', encoding="utf-8")
    fake_binding = _binding_mod.TranscriptBinding(
        digest="sha256:" + "0" * 64,
        uri=transcript.resolve().as_uri(),
        agent_runtime="claude-code/test",
        transcript_id="s1",
        captured_at="2026-01-01T00:00:00+00:00",
    )
    monkeypatch.setattr(
        cli._binding, "resolve_commit_to_transcript", lambda sha, repo=None: fake_binding
    )
    # Force MARKDOWN render-format resolution: AUTO + isatty=True.
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    fake_result = {
        "sheet": {
            "human_intent": "fix the bug",
            "agent_exploration": ["read foo.py", "ran pytest"],
            "corrections": [],
            "tools_used": ["Read", "Bash"],
            "tools_refused": [],
            "diff_rationale": "smallest change that passes tests",
        },
        "metadata": {
            "commit_sha": "abcdef0123456789",
            "transcript_uri": fake_binding.uri,
            "transcript_digest": fake_binding.digest,
            "model_id": "claude-sonnet-4-6",
            "captured_at": "2026-01-01T00:00:00+00:00",
        },
    }
    monkeypatch.setattr(cli, "generate_review_sheet", lambda *a, **kw: fake_result)
    cli.review_sheet_cmd(
        "abcdef0",
        dry_run=False,
        common=_common(tmp_corpus, fmt=OutputFormat.AUTO),
    )
    out = capsys.readouterr().out
    # Markdown shape — header from render_markdown, plus body content.
    assert "# PR Review Sheet" in out
    assert "abcdef012345" in out
    assert "fix the bug" in out
    assert "smallest change that passes tests" in out


def test_review_sheet_cmd_renders_refusal_markdown(
    tmp_corpus: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refusal path: TTY + AUTO format → render_refusal_markdown branch fires."""
    from claude_sql import binding as _binding_mod

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text('{"sessionId": "s1"}\n', encoding="utf-8")
    fake_binding = _binding_mod.TranscriptBinding(
        digest="sha256:" + "0" * 64,
        uri=transcript.resolve().as_uri(),
        agent_runtime="claude-code/test",
        transcript_id="s1",
        captured_at="2026-01-01T00:00:00+00:00",
    )
    monkeypatch.setattr(
        cli._binding, "resolve_commit_to_transcript", lambda sha, repo=None: fake_binding
    )
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    fake_result = {
        "refused": True,
        "reason": "test refusal: model declined to summarize",
        "metadata": {
            "commit_sha": "abcdef0123456789",
            "transcript_uri": fake_binding.uri,
            "transcript_digest": fake_binding.digest,
            "model_id": "claude-sonnet-4-6",
            "captured_at": "2026-01-01T00:00:00+00:00",
        },
    }
    monkeypatch.setattr(cli, "generate_review_sheet", lambda *a, **kw: fake_result)
    cli.review_sheet_cmd(
        "abcdef0",
        dry_run=False,
        common=_common(tmp_corpus, fmt=OutputFormat.AUTO),
    )
    out = capsys.readouterr().out
    # Markdown refusal shape — header + the canonical refusal note + reason.
    assert "# PR Review Sheet" in out
    assert "Review sheet refused" in out
    assert "test refusal: model declined to summarize" in out


# ---------------------------------------------------------------------------
# Default + main entry points
# ---------------------------------------------------------------------------


def test_default_prints_hint(capsys: pytest.CaptureFixture[str]) -> None:
    cli._default()
    out = capsys.readouterr().out
    assert "claude-sql" in out
    assert "subcommand" in out


def test_main_with_help_exits_zero(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import contextlib

    monkeypatch.setattr("sys.argv", ["claude-sql", "--help"])
    # cyclopts may exit cleanly via SystemExit on --help; either path is fine.
    with contextlib.suppress(SystemExit):
        cli.main()
    capsys.readouterr()


def test_main_no_args_invokes_default(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import contextlib

    monkeypatch.setattr("sys.argv", ["claude-sql"])
    with contextlib.suppress(SystemExit):
        cli.main()
    out = capsys.readouterr().out
    # default printer outputs the subcommand list.
    assert "subcommand" in out or "claude-sql" in out

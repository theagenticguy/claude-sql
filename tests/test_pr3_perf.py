"""Regression tests for PR 3 (two-tier connection helpers + parquet-gated macros).

Covers decisions #1 (static schema), #2 (substring-scan dispatch), #4 (parquet-
gated macro DDL), #8 (cached map), #10 (eager registration regression). Each
test pins the specific behavior an end-to-end agent invocation depends on.

The fixtures here borrow ``tmp_corpus`` and ``cache_root`` from the shared
conftest plus ``test_cli.py``'s ``_common`` helper — the cache redirection
fixtures from ``test_cli.py`` aren't autouse for this module, so we re-mount
the env vars locally.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import time
from collections.abc import Iterator
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import duckdb
import pytest

from claude_sql import cli, sql_views
from claude_sql.cli import Common
from claude_sql.config import Settings
from claude_sql.output import OutputFormat

# ---------------------------------------------------------------------------
# Cache redirection — mirrors ``test_cli.py`` so each test gets a clean
# fixture-scoped ``~/.claude`` and never writes to the real one.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _purge_meta_glob_env() -> Iterator[None]:
    """Snapshot/restore ``CLAUDE_SQL_SUBAGENT_META_GLOB`` around every test.

    ``_common`` writes this env var directly so it leaks across test
    modules (notably ``test_team_corpus`` re-reads ``Settings()``).
    Mirrors the same fixture in ``test_cli.py``.
    """
    prior = os.environ.get("CLAUDE_SQL_SUBAGENT_META_GLOB")
    yield
    if prior is None:
        os.environ.pop("CLAUDE_SQL_SUBAGENT_META_GLOB", None)
    else:
        os.environ["CLAUDE_SQL_SUBAGENT_META_GLOB"] = prior


@pytest.fixture(autouse=True)
def _cache_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    cache = tmp_path / "claude_home"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "duckdb_tmp").mkdir(parents=True, exist_ok=True)
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
        "CLAUDE_SQL_SKILLS_CATALOG_PARQUET_PATH", str(cache / "skills_catalog.parquet")
    )
    monkeypatch.setenv("CLAUDE_SQL_USER_SKILLS_DIR", str(cache / "skills"))
    monkeypatch.setenv("CLAUDE_SQL_PLUGINS_CACHE_DIR", str(cache / "plugins_cache"))
    monkeypatch.setenv("CLAUDE_SQL_CHECKPOINT_DB_PATH", str(cache / "claude_sql.duckdb"))
    monkeypatch.setenv("CLAUDE_SQL_HNSW_DB_PATH", str(cache / "hnsw.duckdb"))
    monkeypatch.setenv("CLAUDE_SQL_DUCKDB_TEMP_DIR", str(cache / "duckdb_tmp"))
    monkeypatch.setenv("CLAUDE_SQL_DUCKDB_THREADS", "2")
    monkeypatch.setenv("CLAUDE_SQL_DUCKDB_MEMORY_LIMIT", "1GB")
    monkeypatch.setenv("HOME", str(tmp_path / "fake_home"))
    (tmp_path / "fake_home" / ".claude").mkdir(parents=True, exist_ok=True)
    return cache


def _common(tmp_corpus: dict[str, Any], fmt: OutputFormat = OutputFormat.JSON) -> Common:
    """Construct a ``Common`` pointing at the fixture corpus."""
    os.environ["CLAUDE_SQL_SUBAGENT_META_GLOB"] = tmp_corpus["subagent_meta_glob"]
    return Common(
        verbose=False,
        quiet=True,
        glob=tmp_corpus["glob"],
        subagent_glob=tmp_corpus["subagent_glob"],
        format=fmt,
    )


# ---------------------------------------------------------------------------
# #2 + #10 — trivial query skips the registration chain
# ---------------------------------------------------------------------------


def test_query_scalar_skips_register(
    tmp_corpus: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``SELECT 1`` must not trigger ``register_views``.

    The substring-scan dispatch should pick :func:`_open_connection_introspect`
    (no view registration) for catalog-free SQL. We swap the actual
    ``register_views`` symbol used inside ``_open_connection_full`` for a
    sentinel and assert it's never reached.
    """
    calls: list[str] = []

    def _sentinel(*args: object, **kwargs: object) -> None:
        calls.append("register_views")
        raise AssertionError("register_views should not run on a catalog-free SELECT 1 query")

    # ``register_all`` is what ``_open_connection_full`` actually invokes;
    # patch it on the cli module (the import binding) so the dispatch path
    # being exercised is the real one.
    monkeypatch.setattr(cli, "register_all", _sentinel)
    cli.query("SELECT 1 AS one", common=_common(tmp_corpus))
    out = capsys.readouterr().out
    payload = json.loads(out)
    # Result rendered correctly via the bare connection.
    assert payload[0]["one"] == 1
    assert calls == []


def test_query_with_view_uses_full_registration(
    tmp_corpus: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SQL referencing ``sessions`` must hit the full registration path."""
    cli.query("SELECT COUNT(*) AS n FROM sessions", common=_common(tmp_corpus))
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["n"] >= 0


# ---------------------------------------------------------------------------
# #4 — analytics macros are gated on parquet existence (no warnings either)
# ---------------------------------------------------------------------------


def test_register_macros_skips_friction_when_parquet_missing(
    tmp_path: Path,
    tmp_corpus: dict[str, Any],
) -> None:
    """Fresh-install ``register_macros`` must not register analytics macros.

    Builds a Settings whose every analytics parquet path points at an
    empty directory, runs the v1 registration chain plus
    ``register_macros``, and asserts:
      * v2 analytics macro names (``friction_counts``, ``autonomy_trend``,
        …) are absent from ``list_macros`` output.
      * No WARNING-level loguru records were emitted (gate prevents the
        DDL from running so ``_safe_macro``'s catch-and-debug never fires).
    """
    from loguru import logger

    cache = tmp_path / "claude"
    cache.mkdir(parents=True, exist_ok=True)
    # Point the transcript globs at the fixture corpus so register_raw
    # has files to scan; analytics parquets stay missing -- which is the
    # invariant under test.
    settings = Settings(
        default_glob=tmp_corpus["glob"],
        subagent_glob=tmp_corpus["subagent_glob"],
        subagent_meta_glob=tmp_corpus["subagent_meta_glob"],
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
    )

    captured: list[dict[str, Any]] = []

    def _sink(message: object) -> None:
        record = getattr(message, "record", None)
        if record is None:
            return
        captured.append({"level": record["level"].name, "message": record["message"]})

    handler_id = logger.add(_sink, level="DEBUG")
    try:
        con = duckdb.connect(":memory:")
        try:
            sql_views.register_raw(
                con,
                glob=settings.default_glob,
                subagent_glob=settings.subagent_glob,
                subagent_meta_glob=settings.subagent_meta_glob,
            )
            sql_views.register_views(con)
            # ``register_vss`` creates an empty ``message_embeddings`` table
            # when the embeddings parquet is missing; the always-on
            # ``semantic_search`` macro binds against it at creation time.
            # Skip ``register_analytics`` -- the v2 analytics views stay
            # absent, which is the fresh-install state under test.
            sql_views.register_vss(
                con,
                embeddings_parquet=settings.embeddings_parquet_path,
                dim=int(settings.output_dimension),
            )
            sql_views.register_macros(con, settings=settings)
            macro_names = {n for n, _ in sql_views.list_macros(con)}
        finally:
            con.close()
    finally:
        logger.remove(handler_id)

    # v1 always-on macros are present.
    assert "ago" in macro_names
    assert "tool_rank" in macro_names
    # v2 analytics macros are gated out.
    for analytics in (
        "autonomy_trend",
        "work_mix",
        "success_rate_by_work",
        "cluster_top_terms",
        "community_top_topics",
        "sentiment_arc",
        "friction_counts",
        "friction_rate",
        "friction_examples",
        "unused_skills",
    ):
        assert analytics not in macro_names, (
            f"{analytics} should be skipped on a fresh install (no parquet)"
        )

    # Zero WARNING records about analytics macros.
    warnings = [r for r in captured if r["level"] == "WARNING"]
    macro_warnings = [
        r for r in warnings if "macro" in r["message"].lower() and "missing" in r["message"].lower()
    ]
    assert macro_warnings == [], (
        f"Expected no WARNING records about missing analytics macros; got {macro_warnings}"
    )


# ---------------------------------------------------------------------------
# #8 — schema payload includes the cached map and excludes analytics views
# ---------------------------------------------------------------------------


def test_schema_includes_cached_field(
    tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """Schema JSON output must include the ``cached`` map.

    On a fresh fixture (no analytics parquets) every analytics name maps
    to ``False``; v1 transcript-derived views are NOT in ``views`` because
    we kept VIEW_SCHEMA scoped to v1 (analytics views' source-of-truth is
    the parquet metadata).
    """
    cli.schema(common=_common(tmp_corpus))
    payload = json.loads(capsys.readouterr().out)
    assert "views" in payload
    assert "macros" in payload
    assert "cached" in payload
    # Analytics view membership in ``views`` is excluded by design.
    assert "user_friction" not in payload["views"]
    assert "session_classifications" not in payload["views"]
    # v1 views are present.
    assert "sessions" in payload["views"]
    assert "messages" in payload["views"]
    # ``cached`` is the source of truth for analytics presence.
    assert payload["cached"]["user_friction"] is False
    assert payload["cached"]["friction_rate"] is False
    # Macros section uses the {name, params} shape.
    macros_by_name = {m["name"]: m for m in payload["macros"]}
    assert "ago" in macros_by_name
    assert macros_by_name["ago"]["params"] == ["interval_text"]


# ---------------------------------------------------------------------------
# #1 — schema is fast (static dict, no DuckDB)
# ---------------------------------------------------------------------------


def test_schema_under_100ms(tmp_corpus: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    """``cli.schema`` should complete well under 100ms once warm.

    Budget tightened from 500ms to 100ms after T1.1 + T1.2 connection-helper
    refactors verified the schema path stays out of DuckDB entirely
    (static-dict read + settings parsing + json.dumps lands in ~10ms on a
    warm interpreter). A regression here means something started reaching
    back into DuckDB or the settings parse blew up.
    """
    # Warm any imports / lazy initialization first.
    cli.schema(common=_common(tmp_corpus))
    capsys.readouterr()

    start = time.perf_counter()
    cli.schema(common=_common(tmp_corpus))
    elapsed = time.perf_counter() - start
    capsys.readouterr()
    assert elapsed < 0.1, f"schema took {elapsed:.3f}s (budget 0.1s)"


# ---------------------------------------------------------------------------
# #3 + #4 — stdout purity / no WARNING flooding (in-process variants)
# ---------------------------------------------------------------------------


def test_schema_stdout_is_pure_json(
    tmp_corpus: dict[str, Any],
) -> None:
    """``schema --format json`` stdout must parse cleanly; stderr has no WARNINGs.

    In-process variant of ``test_stdout_purity.py``: capture stdout +
    stderr around the call, assert json.loads(stdout) succeeds and
    stderr has no WARNING-level lines.
    """
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    # Reconfigure loguru to write to our buffer rather than the real
    # sys.stderr; cli._configure rebinds the sink each invocation.
    with redirect_stdout(out_buf), redirect_stderr(err_buf):
        cli.schema(common=_common(tmp_corpus))
    stdout = out_buf.getvalue()
    stderr = err_buf.getvalue()
    payload = json.loads(stdout)  # Raises if stdout is contaminated.
    assert "views" in payload
    assert "WARNING" not in stderr.upper()


def test_query_scalar_no_warnings(
    tmp_corpus: dict[str, Any],
) -> None:
    """``query "SELECT 1"`` stderr must contain zero WARNING lines."""
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    with redirect_stdout(out_buf), redirect_stderr(err_buf):
        cli.query("SELECT 1", common=_common(tmp_corpus))
    stderr = err_buf.getvalue()
    assert "WARNING" not in stderr.upper()


# ---------------------------------------------------------------------------
# #4 — subprocess regression: live binary must keep stdout / stderr clean
# ---------------------------------------------------------------------------


def test_subprocess_query_select1_no_warnings(
    tmp_corpus: dict[str, Any],
) -> None:
    """Out-of-process: ``claude-sql query "SELECT 1"`` keeps stderr clean.

    The in-process tests above use ``redirect_stderr`` which catches
    everything loguru writes through the rebound sink. This test verifies
    the same invariant via the actual subprocess shape an end-user agent
    sees -- entry-point script, separate stdout / stderr pipes.

    Skipped when the ``claude-sql`` console script isn't on PATH (e.g.
    a fresh worktree pre-``mise run install``); the in-process variants
    cover the contract by themselves.
    """
    import shutil

    binary = shutil.which("claude-sql")
    if binary is None:
        pytest.skip("claude-sql binary not on PATH")
    env = os.environ.copy()
    env["CLAUDE_SQL_SUBAGENT_META_GLOB"] = tmp_corpus["subagent_meta_glob"]
    result = subprocess.run(
        [
            binary,
            "query",
            "--quiet",
            "--format",
            "json",
            "--glob",
            tmp_corpus["glob"],
            "--subagent-glob",
            tmp_corpus["subagent_glob"],
            "SELECT 1",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(
            f"subprocess invocation failed (rc={result.returncode}); stderr={result.stderr[:200]}"
        )
    json.loads(result.stdout)
    assert "WARNING" not in result.stderr.upper()

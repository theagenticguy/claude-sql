"""Tests for :mod:`claude_sql.config`.

Covers the per-pipeline concurrency split, the ``concurrency`` deprecation
alias, and the DuckDB tuning PRAGMAs that ``cli._open_connection_full`` applies.
The PRAGMA test exercises only the snippet that runs before ``register_all``
so we avoid scanning the live ``~/.claude/projects`` corpus during the unit
suite. The full integration with ``register_all`` is already covered by
``test_sql_views.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pytest

from claude_sql.app.cli import _resolve_memory_limit
from claude_sql.core.config import Settings


def test_default_per_pipeline_concurrency() -> None:
    s = Settings()
    assert s.embed_concurrency == 8
    assert s.llm_concurrency == 16


def test_env_override_embed_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_SQL_EMBED_CONCURRENCY", "16")
    s = Settings()
    assert s.embed_concurrency == 16
    assert s.llm_concurrency == 16


def test_env_override_llm_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_SQL_LLM_CONCURRENCY", "4")
    s = Settings()
    assert s.embed_concurrency == 8
    assert s.llm_concurrency == 4


def test_concurrency_field_removed_in_v1_0_1() -> None:
    """``Settings.concurrency`` was deprecated in v0.x and removed in v1.0.1."""
    s = Settings()
    assert not hasattr(s, "concurrency")


def test_duckdb_threads_default_matches_cpu_count() -> None:
    s = Settings()
    assert s.duckdb_threads == (os.cpu_count() or 4)


def test_duckdb_threads_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_SQL_DUCKDB_THREADS", "1")
    s = Settings()
    assert s.duckdb_threads == 1


def test_duckdb_memory_limit_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_SQL_DUCKDB_MEMORY_LIMIT", "256MB")
    s = Settings()
    assert s.duckdb_memory_limit == "256MB"


def test_duckdb_temp_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "spill"
    monkeypatch.setenv("CLAUDE_SQL_DUCKDB_TEMP_DIR", str(target))
    s = Settings()
    assert s.duckdb_temp_dir == target


def test_settings_field_smoke() -> None:
    """Quick sanity check that the new fields are wired on the type."""
    fields = Settings.model_fields.keys()
    assert "embed_concurrency" in fields
    assert "llm_concurrency" in fields
    assert "concurrency" not in fields  # removed in v1.0.1
    assert "duckdb_threads" in fields
    assert "duckdb_memory_limit" in fields
    assert "duckdb_temp_dir" in fields


def test_resolve_memory_limit_passes_absolute_size_through() -> None:
    """Absolute byte specs pass through ``_resolve_memory_limit`` unchanged."""
    assert _resolve_memory_limit("512MB") == "512MB"
    assert _resolve_memory_limit("4GiB") == "4GiB"
    assert _resolve_memory_limit("  2GB  ") == "2GB"


def test_resolve_memory_limit_translates_percent_to_mib() -> None:
    """Percentage specs become an absolute MiB value DuckDB accepts."""
    resolved = _resolve_memory_limit("70%")
    assert resolved.endswith("MiB")
    mib = int(resolved.removesuffix("MiB"))
    assert mib > 0


def test_duckdb_pragmas_take_effect_on_a_fresh_connection(tmp_path: Path) -> None:
    """The PRAGMA bundle ``_open_connection`` issues is accepted by DuckDB
    and round-trips through ``current_setting`` exactly as configured.

    We exercise the PRAGMA snippet in isolation (no view registration) so
    the unit suite stays fast — the integration with ``register_all`` is
    covered by the broader sql_views tests.
    """
    temp_dir = tmp_path / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(":memory:")
    try:
        memory_limit = _resolve_memory_limit("70%")
        con.execute("SET threads = 2")
        con.execute(f"SET memory_limit = '{memory_limit}'")
        con.execute(f"SET temp_directory = '{temp_dir}'")
        con.execute("SET enable_object_cache = true")
        con.execute("SET preserve_insertion_order = false")

        threads = con.execute("SELECT current_setting('threads')").fetchone()
        memory = con.execute("SELECT current_setting('memory_limit')").fetchone()
        temp = con.execute("SELECT current_setting('temp_directory')").fetchone()
        object_cache = con.execute("SELECT current_setting('enable_object_cache')").fetchone()
        order = con.execute("SELECT current_setting('preserve_insertion_order')").fetchone()
        assert threads is not None and int(threads[0]) == 2
        # DuckDB renders the limit in human-friendly units (GiB / MiB),
        # so assert only that it parsed our value rather than rejected it.
        assert memory is not None and str(memory[0]).strip() != ""
        assert temp is not None and str(temp_dir) in str(temp[0])
        assert object_cache is not None and str(object_cache[0]).lower() == "true"
        assert order is not None and str(order[0]).lower() == "false"
    finally:
        con.close()

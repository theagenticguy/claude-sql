"""Tests for output.py -- format resolution, DataFrame emission, error classification."""

from __future__ import annotations

import json
import sys

import duckdb
import polars as pl
import pytest

from claude_sql.output import (
    EXIT_CODES,
    ClassifiedError,
    OutputFormat,
    classify_duckdb_error,
    emit_dataframe,
    emit_error,
    resolve_format,
    run_or_die,
)


class _FakeTTY:
    """File-like wrapper that declares itself a TTY.

    Wraps pytest's real captured stdout so everything we print still lands in
    ``capsys.readouterr()``; we only override ``isatty()`` to flip the
    ``resolve_format`` branch.
    """

    def __init__(self, real: object, tty: bool) -> None:
        self._real = real
        self._tty = tty

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)

    def isatty(self) -> bool:
        return self._tty


def test_resolve_auto_picks_json_off_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdout", _FakeTTY(sys.stdout, tty=False))
    assert resolve_format(OutputFormat.AUTO) is OutputFormat.JSON


def test_resolve_auto_picks_table_on_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdout", _FakeTTY(sys.stdout, tty=True))
    assert resolve_format(OutputFormat.AUTO) is OutputFormat.TABLE


def test_resolve_respects_explicit_format() -> None:
    assert resolve_format(OutputFormat.CSV) is OutputFormat.CSV
    assert resolve_format("ndjson") is OutputFormat.NDJSON


def test_emit_dataframe_json_emits_valid_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    df = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    emit_dataframe(df, OutputFormat.JSON)
    payload = json.loads(capsys.readouterr().out)
    assert payload == [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]


def test_emit_dataframe_ndjson_emits_one_object_per_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    df = pl.DataFrame({"x": [1, 2]})
    emit_dataframe(df, OutputFormat.NDJSON)
    captured = capsys.readouterr().out
    lines = [line for line in captured.splitlines() if line]
    assert [json.loads(line) for line in lines] == [{"x": 1}, {"x": 2}]


def test_emit_dataframe_csv_has_header_and_rows(
    capsys: pytest.CaptureFixture[str],
) -> None:
    df = pl.DataFrame({"x": [1, 2]})
    emit_dataframe(df, OutputFormat.CSV)
    text = capsys.readouterr().out
    assert text.splitlines()[0] == "x"
    assert "1" in text and "2" in text


def test_classify_parse_error_maps_to_64() -> None:
    exc = duckdb.ParserException("oops")
    err = classify_duckdb_error(exc)
    assert err.kind == "parse_error"
    assert err.exit_code == EXIT_CODES["parse_error"] == 64
    assert err.hint is not None


def test_classify_catalog_error_maps_to_65() -> None:
    exc = duckdb.CatalogException("view not found")
    err = classify_duckdb_error(exc)
    assert err.kind == "catalog_error"
    assert err.exit_code == EXIT_CODES["catalog_error"] == 65


def test_classify_runtime_error_maps_to_70() -> None:
    exc = duckdb.ConversionException("cast failed")
    err = classify_duckdb_error(exc)
    assert err.kind == "runtime_error"
    assert err.exit_code == EXIT_CODES["runtime_error"] == 70


def test_emit_error_json_payload(
    capsys: pytest.CaptureFixture[str],
) -> None:
    err = ClassifiedError(kind="parse_error", exit_code=64, message="oops", hint="hint")
    emit_error(err, OutputFormat.JSON)
    payload = json.loads(capsys.readouterr().err)
    assert payload == {"error": {"kind": "parse_error", "message": "oops", "hint": "hint"}}


def test_run_or_die_returns_result_on_success() -> None:
    assert run_or_die(lambda: 42) == 42


def test_run_or_die_exits_with_classified_code_on_duckdb_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom() -> None:
        raise duckdb.CatalogException("nope")

    with pytest.raises(SystemExit) as ei:
        run_or_die(boom, fmt=OutputFormat.JSON)
    assert ei.value.code == 65
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"]["kind"] == "catalog_error"


def test_emit_dataframe_table_respects_tty_config(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Table format prints a polars-rendered box -- not JSON / not CSV."""
    monkeypatch.setattr(sys, "stdout", _FakeTTY(sys.stdout, tty=True))
    df = pl.DataFrame({"x": [1]})
    emit_dataframe(df, OutputFormat.AUTO)
    text = capsys.readouterr().out
    # polars' default table renderer emits "shape: (1, 1)" preamble
    assert "shape:" in text

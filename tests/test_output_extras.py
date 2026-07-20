"""Coverage top-up for :mod:`claude_sql.output`.

Targets:

* Lines 146-147 — :class:`InputValidationError` carries ``hint``.
* Lines 160-163 — :func:`validate_glob` accepts empty / single-``**``,
  rejects multi-``**`` with hint.
* Lines 213-216 — :func:`emit_error` TABLE branch writes ``[kind] msg``
  + ``hint:`` to stderr.
* Lines 237-244 — :func:`run_or_die` catches ``InputValidationError``,
  exits 64.

The closed-set guard at the end of :func:`emit_dataframe` is now defensive
code: ``OutputFormat`` covers AUTO/TABLE/JSON/NDJSON/CSV with no member
that falls through. Reaching it would require dynamic enum injection,
which doesn't represent any real call path — the guard stays as
``# pragma: no cover``.
"""

from __future__ import annotations

import json
import sys

import pytest

from claude_sql.domain.errors import (
    EXIT_CODES,
    ClassifiedError as _ClassifiedError,
    InputValidationError,
)
from claude_sql.interfaces.cli.output import (
    OutputFormat,
    emit_error,
    run_or_die,
    validate_glob,
)

# ---------------------------------------------------------------------------
# InputValidationError + validate_glob (lines 146-147, 160-163)
# ---------------------------------------------------------------------------


def test_input_validation_error_carries_hint() -> None:
    """The custom error wraps both message + hint (lines 146-147)."""
    exc = InputValidationError("bad", hint="try this")
    assert str(exc) == "bad"
    assert exc.hint == "try this"


def test_validate_glob_passes_through_none_and_empty() -> None:
    """``None`` / empty short-circuits — caller falls back to default glob."""
    validate_glob(None)  # no-op
    validate_glob("")  # no-op
    # Single ** is fine — use a benign workspace path so S108 stays quiet.
    validate_glob("/home/user/.claude/projects/**/*.jsonl")


def test_validate_glob_rejects_multiple_recursive_segments() -> None:
    """Two ``**`` segments raise InputValidationError (lines 162-163)."""
    with pytest.raises(InputValidationError) as ei:
        validate_glob("/a/**/b/**/c.jsonl", flag="--glob")
    assert ei.value.hint is not None
    assert "**" in str(ei.value)


# ---------------------------------------------------------------------------
# emit_error TABLE branch (lines 213-216)
# ---------------------------------------------------------------------------


class _FakeTTY:
    """File-like wrapper with a switchable ``isatty``."""

    def __init__(self, real: object, tty: bool) -> None:
        self._real = real
        self._tty = tty

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)

    def isatty(self) -> bool:
        return self._tty


def test_emit_error_table_format_writes_prefix_and_hint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """TABLE format emits ``[kind] msg`` + a ``hint:`` line on stderr."""
    err = _ClassifiedError(
        kind="parse_error",
        exit_code=64,
        message="boom",
        hint="check syntax",
    )
    emit_error(err, OutputFormat.TABLE)
    cap = capsys.readouterr()
    assert "[parse_error] boom" in cap.err
    assert "hint: check syntax" in cap.err


def test_emit_error_table_format_omits_hint_line_when_no_hint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No hint → no ``hint:`` line."""
    err = _ClassifiedError(kind="runtime_error", exit_code=70, message="nope", hint=None)
    emit_error(err, OutputFormat.TABLE)
    cap = capsys.readouterr()
    assert "[runtime_error] nope" in cap.err
    assert "hint:" not in cap.err


def test_emit_error_auto_off_tty_uses_json(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AUTO off-TTY → JSON payload on stderr."""
    monkeypatch.setattr(sys, "stdout", _FakeTTY(sys.stdout, tty=False))
    err = _ClassifiedError(kind="catalog_error", exit_code=65, message="missing", hint=None)
    emit_error(err, OutputFormat.AUTO)
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"]["kind"] == "catalog_error"


# ---------------------------------------------------------------------------
# run_or_die InputValidationError branch (lines 237-244)
# ---------------------------------------------------------------------------


def test_run_or_die_catches_input_validation_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``run_or_die`` catches ``InputValidationError`` and exits 64."""

    def boom() -> None:
        raise InputValidationError("bad glob", hint="single ** only")

    with pytest.raises(SystemExit) as ei:
        run_or_die(boom, fmt=OutputFormat.JSON)
    assert ei.value.code == EXIT_CODES["invalid_input"] == 64
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"]["kind"] == "invalid_input"
    assert payload["error"]["hint"] == "single ** only"

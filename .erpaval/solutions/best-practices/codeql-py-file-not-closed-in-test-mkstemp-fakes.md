---
title: CodeQL `py/file-not-closed` fires on `os.open` inside a `tempfile.mkstemp` fake — open per-call, let the consumer own the close
track: knowledge
category: best-practices
module: tests/**/*.py — `tempfile.mkstemp` monkeypatch shapes
component: GitHub Advanced Security CodeQL `py/file-not-closed`
severity: warning
tags: [codeql, github-advanced-security, file-not-closed, mkstemp, monkeypatch, fixtures, tempfile, os-open, claude-sql]
applies_when:
  - "Tests monkeypatch `tempfile.mkstemp` to return a (fd, path) tuple"
  - "Production code at the patch site already does `os.close(fd)` (the conventional shape — mkstemp returns an fd you're meant to close)"
  - "GitHub Advanced Security / CodeQL is enabled on PRs"
pattern: |
  ``tempfile.mkstemp`` returns ``(fd, path)``. Conventional consumers
  immediately close the fd because they only want the path:

  ```python
  fd, db_path = tempfile.mkstemp(suffix=".duckdb")
  os.close(fd)  # we only care about the path
  ```

  When tests patch ``mkstemp`` to redirect the temporary file to a
  fixture-managed path, the fake has to return *something* fd-shaped
  so the consumer's ``os.close(fd)`` doesn't blow up. The
  module-level approach — open one fd, hand it back from the closure —
  reads cleanly to a human reviewer but trips two distinct CodeQL
  warnings:

  ```python
  # SHAPE 1 — single fd reused across calls, never closed
  def _patch_mkstemp_for_duckdb(monkeypatch, tmp_path):
      db_path = tmp_path / "shell_test.duckdb"
      fd_holder = os.open(os.devnull, os.O_RDONLY)  # ← CodeQL: file not closed
      def _fake_mkstemp(*_a, **_kw):
          return fd_holder, str(db_path)
      monkeypatch.setattr(cli.tempfile, "mkstemp", _fake_mkstemp)
      return db_path
  ```

  CodeQL traces the ``os.open`` to see whether the resulting fd is
  passed to ``os.close``. With the closure above, CodeQL sees the
  open-site but not the close-site (which lives in production code at
  ``cli.py:674``); the static linkage is broken because the closure
  decouples them.

  Worse, the seemingly-equivalent **add-finalizer fix is also
  incorrect**: if the fake is invoked twice (or the production
  consumer calls ``os.close(fd)`` itself), you get ``OSError: [Errno
  9] Bad file descriptor`` from the double close.

  ```python
  # SHAPE 2 — also broken: production already closes, double-close
  fd_holder = os.open(os.devnull, os.O_RDONLY)
  request.addfinalizer(lambda: os.close(fd_holder))  # ← double close
  ```

  **Fix**: open inside the closure, **once per call**. Each invocation
  produces its own fd; the consumer's existing ``os.close(fd)`` owns
  it; producer and consumer are paired in one path so CodeQL's data
  flow links them statically:

  ```python
  # CORRECT — fresh fd per call, consumer closes
  def _patch_mkstemp_for_duckdb(monkeypatch, tmp_path):
      db_path = tmp_path / "shell_test.duckdb"
      def _fake_mkstemp(*_a, **_kw):
          # The CLI's `os.close(fd)` (cli.py:674) owns the close.
          # Opening inside the closure keeps each invocation self-
          # contained so CodeQL's py/file-not-closed sees the producer
          # and consumer paired.
          return os.open(os.devnull, os.O_RDONLY), str(db_path)
      monkeypatch.setattr(cli.tempfile, "mkstemp", _fake_mkstemp)
      return db_path
  ```

  This satisfies CodeQL AND survives multiple calls AND survives the
  production consumer's existing ``os.close(fd)`` because each fd is
  closed exactly once by exactly one party.

  **Why a comment helps here**: the closure body looks like a
  resource leak in isolation. Document the ownership invariant
  inline: "the CLI's ``os.close(fd)`` owns the close." Future you (or
  a reviewer) will not have to grep for the consumer.
example_files:
  - tests/test_cli_coverage.py — `_patch_mkstemp_for_duckdb` fixture
  - src/claude_sql/cli.py:673-674 — `fd, path = tempfile.mkstemp(...)`; `os.close(fd)`
counter_examples:
  - "Per-test ``request.addfinalizer(lambda: os.close(fd))`` — looks safer because the finalizer guarantees a close, but produces ``Bad file descriptor`` on a double close when the production consumer ALSO closes the fd. The finalizer adds work, not safety."
  - "Returning a sentinel like ``-1`` from the fake — works only if the production consumer never calls ``os.close``. Fragile to refactor; the consumer is allowed to close (mkstemp's contract says it owns the fd) and one day will."
  - "Wrapping the fake in ``contextlib.contextmanager`` and yielding the fd — overkill for a closure-shaped fake; pytest's ``monkeypatch`` already handles teardown for the patch itself, just not for ``os.open`` calls inside the closure."
  - "Adding ``# noqa: <CodeQL rule>`` — CodeQL doesn't read noqa comments. Same with ``# nosem`` (semgrep). Suppression must happen at the ``.codeql`` query-suite level, which is heavier than just opening the fd in the right place."
references:
  - "GitHub CodeQL py/file-not-closed: https://codeql.github.com/codeql-query-help/python/py-file-not-closed/"
  - "Python tempfile.mkstemp docs: https://docs.python.org/3/library/tempfile.html#tempfile.mkstemp"
  - "claude-sql PR #42 (2026-05-13) — `py/file-not-closed` fired on `tests/test_cli_coverage.py:450` after the v1.0 patch-coverage push added the `_patch_mkstemp_for_duckdb` helper. First fix attempt (`addFinalizer`) produced `Bad file descriptor` on a double close because cli.py:674 already calls `os.close(fd)`. Second fix (open per-call) satisfied CodeQL and the test suite simultaneously."
---

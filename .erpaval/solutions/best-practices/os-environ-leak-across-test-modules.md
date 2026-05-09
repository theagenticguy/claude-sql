---
title: Raw `os.environ[...] =` in a test helper leaks across modules — pair with an autouse purge fixture
track: knowledge
category: best-practices
module: tests/test_cli.py (and any test calling Settings()/BaseSettings inside)
component: pytest + pydantic-settings env-var coupling
severity: warning
tags: [pytest, monkeypatch, os.environ, leakage, pydantic-settings, test-isolation, claude-sql]
applies_when:
  - "A test helper sets a config env var via ``os.environ[\"FOO\"] = ...`` (not pytest's ``monkeypatch.setenv``)"
  - "The project uses pydantic-settings (or anything that re-reads env vars on every ``Settings()`` instantiation)"
  - "Other test modules construct a fresh ``Settings`` and depend on the env being clean"
pattern: |
  Pytest's ``monkeypatch.setenv("FOO", ...)`` is undone automatically at
  the end of the requesting test. Raw ``os.environ["FOO"] = ...`` is
  not — it persists for the rest of the pytest process. When the helper
  must run in a context where ``monkeypatch`` isn't available (e.g. a
  module-level helper function, a generator that constructs CLI ``Common``
  instances), it's tempting to fall back to ``os.environ``. Don't ship
  it bare — the leak surfaces as failures in *unrelated* test modules
  the next time the suite runs.

  Concrete failure mode (claude-sql, 2026-05-09):
  ``test_cli._common`` did
  ``os.environ["CLAUDE_SQL_SUBAGENT_META_GLOB"] = tmp_corpus["..."]``
  to wire fixture paths into ``Settings()``. Tests in ``test_cli.py``
  passed. Then ``test_team_corpus.py`` (a different module) constructed
  ``Settings(team_corpus_root=...)``, the leaked env var pinned the
  ``subagent_meta_glob``, the user-pinned-glob detector treated it as
  user input, and the team-corpus path-derivation logic short-circuited.
  6 tests failed in the *other* module without any change to its code.

  The fix is a module-level autouse fixture that snapshots and restores:

  ```python
  @pytest.fixture(autouse=True)
  def _purge_meta_glob_env() -> Iterator[None]:
      """Snapshot/restore CLAUDE_SQL_SUBAGENT_META_GLOB around every test."""
      import os
      prior = os.environ.get("CLAUDE_SQL_SUBAGENT_META_GLOB")
      yield
      if prior is None:
          os.environ.pop("CLAUDE_SQL_SUBAGENT_META_GLOB", None)
      else:
          os.environ["CLAUDE_SQL_SUBAGENT_META_GLOB"] = prior
  ```

  The cleaner alternative — making ``_common`` accept a ``monkeypatch``
  parameter and threading it through every caller — is invasive (50+
  call sites in ``test_cli.py``) and produces a worse signal-to-noise
  ratio. The autouse-purge pattern keeps the helper signature small
  while restoring isolation per test.

  **Detection**: when adding tests that exercise ``Settings()`` or any
  pydantic-settings subclass via env vars, run the *full* suite once
  (``mise run test``, not just the new file) before merging. Single-file
  test runs miss cross-module leakage by definition.
example_files:
  - tests/test_cli.py            # _common helper + _purge_meta_glob_env fixture
  - tests/test_team_corpus.py    # was the cross-module victim
  - src/claude_sql/config.py     # pydantic-settings re-reads env on each Settings()
counter_examples:
  - "``os.environ[\"FOO\"] = bar`` in a helper without an autouse purge — works locally, breaks in CI on test order interleaving."
  - "Threading ``monkeypatch`` through every helper signature — works but adds parameter cruft to every caller; only worth it when the helper is small."
  - "Relying on ``Settings.model_copy`` to override the env value — pydantic-settings doesn't read from a copy, so the fresh ``Settings()`` in another test still picks up the leaked env."
references:
  - "pytest monkeypatch docs: https://docs.pytest.org/en/stable/how-to/monkeypatch.html"
  - "pydantic-settings env priority: https://docs.pydantic.dev/latest/concepts/pydantic_settings/"
  - "claude-sql session-d80d08 (2026-05-09) — caught after running the full suite, fixed in tests/test_cli.py"
---

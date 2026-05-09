---
title: Importing helpers from `conftest.py` needs `pythonpath` (pytest) AND `root` (ty)
track: knowledge
category: best-practices
module: tests/conftest.py + pyproject.toml
component: pytest + ty static analysis
severity: warning
tags: [pytest, conftest, ty, static-analysis, pythonpath, test-infrastructure, claude-sql]
applies_when:
  - "Test files want to ``from conftest import FakeFoo, make_bar`` to share helper classes/functions, not just fixtures"
  - "The project uses ty (or pyright) in strict mode over both ``src/`` and ``tests/``"
pattern: |
  Pytest auto-loads ``conftest.py`` as a plugin and registers its
  ``@pytest.fixture`` definitions, but **does NOT** expose its module
  symbols (classes, helper functions, constants) as importable. Symbols
  like a ``class FakeBedrockClient`` that a test needs to instantiate
  directly aren't reachable via ``from conftest import …`` unless the
  ``tests/`` directory is on ``sys.path``.

  The ``from tests.conftest import …`` form works only when ``tests/``
  is a package (has ``__init__.py``) AND the project root is on
  ``sys.path``. Adding an ``__init__.py`` is a deeper change — pytest's
  rootdir discovery, tox-style isolation, and namespace packaging all
  start to disagree. Don't go there for a sharing concern.

  The fix is two-line, in ``pyproject.toml``:

  ```toml
  [tool.pytest.ini_options]
  testpaths = ["tests"]
  pythonpath = ["tests"]   # <-- enables `from conftest import …` in tests

  [tool.ty.environment]
  root = ["./src", "./tests"]   # <-- ty resolves the same import statically
  ```

  Without ``pythonpath``, pytest collects but ``ModuleNotFoundError`` at
  import time. Without ty's ``root`` extension, the runtime tests pass
  but ``mise run typecheck`` fails with ``unresolved-import``. Both must
  agree.

  **Why this is non-obvious**: when conftest only exposes fixtures
  (``def my_fixture(...) -> Foo``), tests get them via injection — no
  import needed, the conftest path is irrelevant. The sharing problem
  surfaces only once helpers grow to include reusable classes
  (``FakeBedrockClient`` for capturing ``invoke_model`` calls,
  ``write_session_jsonl`` for building JSONL fixtures inline) that tests
  want to instantiate directly rather than receive as fixtures. Don't
  reach for ``import sys; sys.path.insert(0, …)`` boilerplate at the
  top of every test file — fix it once at the project level.
example_files:
  - tests/conftest.py                       # the helper module
  - tests/test_embed_worker.py              # `from conftest import FakeBedrockClient`
  - tests/test_session_text.py              # `from conftest import _seed_subagent_stub`
  - pyproject.toml                          # the two settings above
counter_examples:
  - "``from tests.conftest import …`` — fails at runtime unless ``tests/__init__.py`` exists; if you add it, pytest's rootdir discovery and namespace-package semantics diverge."
  - "``import sys; sys.path.insert(0, str(Path(__file__).parent))`` at the top of every test — works but adds boilerplate to every new test, and violates ruff E402 unless silenced per file."
  - "Demoting helpers to fixtures only — works but loses the ability to use a class as a `dataclass`-style return type, and forces every helper to take ``request`` or have indirection. Fixtures are for setup/teardown around a test, not a substitute for a shared utility module."
references:
  - "pytest docs: https://docs.pytest.org/en/stable/reference/reference.html#confval-pythonpath"
  - "ty configuration: https://docs.astral.sh/ty/configuration/#environment"
  - "claude-sql PR for 90%+ coverage push (session-d80d08, 2026-05-09)"
---

---
title: `__version__` should read from `importlib.metadata`, not a hardcoded constant
track: knowledge
category: best-practices
module: src/<pkg>/__init__.py + cz bump version_provider
component: commitizen + uv version_provider + Python __version__
severity: warning
tags: [version, importlib-metadata, commitizen, uv, claude-sql]
applies_when:
  - "Project uses ``[tool.commitizen].version_provider = ""uv""`` (or any provider that updates only ``[project].version``)"
  - "Project's ``src/<pkg>/__init__.py`` defines a hardcoded ``__version__`` constant"
  - "CLI surfaces the version (e.g. ``mypkg --version``) by formatting that constant"
pattern: |
  ``cz bump`` with ``version_provider = "uv"`` updates ONE place:
  ``[project].version`` in pyproject.toml. It does not (and should
  not) edit any ``__version__ = "..."`` constant in your package
  source. If you have one, it drifts the moment you do your first
  bump after writing it.

  The drift is invisible during local dev — your editable install
  reads the source constant and shows whatever you typed there. The
  drift only manifests after a wheel is built and installed:

  ```
  $ pip install mypkg==0.4.0
  $ mypkg --version
  mypkg 0.1.0       ← stale, baked in at v0.1.0 release time
  ```

  And it manifests *only* under ``--version``-class flags; the wheel
  itself is correct (PyPI shows v0.4.0, ``importlib.metadata`` shows
  v0.4.0, the file system has the new code). The bug is purely in
  the self-report.

  **Fix**: derive ``__version__`` from the installed metadata at
  runtime. The stdlib ``importlib.metadata`` is the canonical path
  since Python 3.8:

  ```python
  # src/<pkg>/__init__.py
  from importlib.metadata import PackageNotFoundError, version

  try:
      __version__ = version("mypkg")
  except PackageNotFoundError:
      # Editable / source installs without metadata fall back to a
      # dev sentinel so ``mypkg --version`` doesn't crash before
      # the package is built. CI / pip / uv always have metadata.
      __version__ = "0.0.0+local"
  ```

  This single change makes ``cz bump`` (or any bumper that touches
  pyproject.toml) the **only** version-of-truth surface. The
  constant becomes a derivation, not a hand-maintained fact.

  **Why ``[tool.commitizen].version_files``-style multi-file rewrite
  is worse**: cz lets you list extra files to keep in sync (e.g.
  ``__init__.py:^__version__``). It works but adds:
  1. A list-of-paths config that drifts whenever someone moves a
     file or renames a constant.
  2. A regex per file that's silently brittle.
  3. A failure mode where the file isn't on the list and the bump
     half-completes (pyproject + .lock right, source wrong).

  The importlib-metadata path moves the source of truth to a single
  place (pyproject.toml) and lets the runtime resolve it. There is no
  list of files to keep in sync, by construction.

  **Detect the drift in CI**: a smoke-test step that runs
  ``mypkg --version`` against the freshly-built wheel and asserts
  the output contains the version cz just bumped to. The
  ``uv-native-pypi-publish-with-smoke-tests`` lesson covers the wheel
  smoke test; bolting on a regex assert against ``${{ github.ref_name
  }}`` (the tag, e.g. ``v0.4.0``) closes the loop.
example_files:
  - src/claude_sql/__init__.py             # had `__version__ = "0.1.0"` hardcoded since v0.1.0
  - src/claude_sql/install_source.py       # CLI version formatter that consumed the constant
  - pyproject.toml                         # `[tool.commitizen].version_provider = "uv"` updates only `[project].version`
counter_examples:
  - "``[tool.commitizen].version_files = [""src/<pkg>/__init__.py:^__version__""]`` — keeps the constant up to date but adds a brittle regex + extra fact to maintain. Solves the symptom, not the root cause."
  - "Hardcoding ``__version__`` and remembering to update it manually — works exactly until the first time you forget. Manual sync against an automation-driven bumper is a perpetual oversight."
  - "Reading ``importlib.metadata.version()`` at every call site — works but causes lookup churn on hot paths. Compute once at import time at ``__init__.py`` so the constant retains its O(1) read property."
references:
  - "PEP 396 (Module Version Numbers, withdrawn but still cited): https://peps.python.org/pep-0396/"
  - "importlib.metadata: https://docs.python.org/3/library/importlib.metadata.html"
  - "commitizen version_provider: https://commitizen-tools.github.io/commitizen/config/#version_provider"
  - "claude-sql diagnosis (2026-05-10) — caught when ``uvx claude-sql@0.4.0 --version`` printed ``claude-sql 0.1.0``; PyPI metadata was correct, the bug was a `__version__ = ""0.1.0""` constant in `__init__.py` that cz had no reason to touch."
---

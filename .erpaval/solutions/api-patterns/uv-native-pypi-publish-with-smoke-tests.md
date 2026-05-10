---
title: uv-native PyPI publish workflow — `uv build` + isolated smoke test + `uv publish` (skip the pypa action)
track: knowledge
category: api-patterns
module: .github/workflows/publish.yml + pyproject.toml
component: uv + GitHub Actions OIDC + PyPI Trusted Publishing
severity: warning
tags: [uv, pypi, github-actions, trusted-publishing, oidc, smoke-test, claude-sql]
applies_when:
  - "Project ships to PyPI and uses uv as the project / dependency manager"
  - "GitHub Actions Trusted Publishing (OIDC) is configured on PyPI / TestPyPI"
  - "You want to catch missing transitive dependencies BEFORE the upload, not in user space"
pattern: |
  Astral's recommended publish shape (https://docs.astral.sh/uv/guides/integration/github/)
  is meaningfully better than the pre-uv flow with
  ``pypa/gh-action-pypi-publish``. Three reasons to switch:

  1. **No token plumbing.** ``uv publish`` reads the GitHub Actions OIDC
     token automatically when the job has ``permissions.id-token:
     write`` and an ``environment:`` block. The pypa action does the
     same dance but adds a layer of indirection (action inputs vs. uv
     CLI flags). Drop the layer.

  2. **Idempotent retries via `--check-url`.** When ``uv publish`` is
     given ``--index <name>`` (or ``UV_PUBLISH_URL``), it computes
     sha256 hashes of the local artifacts against the registry's
     simple-index manifest. Files that already match are skipped;
     files with different hashes refuse to upload. This means a
     half-uploaded release is safely re-runnable, AND a content drift
     (e.g. you changed deps and the wheel METADATA shifted) gets
     caught before silently overwriting users' installed copies.

  3. **Smoke tests catch missing transitive deps before upload.**
     Insert this between build and publish:

     ```yaml
     - name: Smoke test (wheel)
       run: uv run --isolated --no-project --with dist/*.whl <pkg> --version
     - name: Smoke test (sdist)
       run: uv run --isolated --no-project --with dist/*.tar.gz <pkg> --version
     ```

     ``--isolated --no-project --with dist/*.whl`` builds an ephemeral
     venv from ONLY the freshly-built wheel + its declared deps —
     mirroring what an ``uv tool install <pkg>`` user gets. If a
     module imports something that's transitively satisfied in your
     dev venv but isn't declared in ``[project].dependencies``, the
     smoke test fails with ``ModuleNotFoundError`` and the publish
     job's ``needs: build`` dependency keeps the upload from firing.

     The ``--version`` probe is intentionally cheap: every CLI command
     in a typical Cyclopts/Click/Typer app eagerly imports the worker
     modules to register subcommands, so ``--version`` already
     exercises the full import graph. No fixtures needed.

  Configure ``[[tool.uv.index]]`` in pyproject.toml to enable
  ``uv publish --index testpypi`` for dry-runs:

  ```toml
  [[tool.uv.index]]
  name = "testpypi"
  url = "https://test.pypi.org/simple/"
  publish-url = "https://test.pypi.org/legacy/"
  explicit = true
  ```

  ``url`` is the PEP-503 simple endpoint used for ``--check-url``
  hashing. ``publish-url`` is the legacy upload endpoint. ``explicit
  = true`` keeps TestPyPI as a publish-only target — uv won't pull
  resolution candidates from it, even by accident.

  Pin the GitHub Action via ``astral-sh/setup-uv@<sha>  # v8.1.0``
  (Astral's first-party runner — faster than mise + correct caching
  defaults). Don't forget Python: ``uv python install 3.13`` after
  ``setup-uv`` because setup-uv doesn't ship Python by default.

  Full Astral-recommended workflow shape with TestPyPI dry-run +
  production paths is in ``.github/workflows/publish.yml``.
example_files:
  - .github/workflows/publish.yml          # full Astral-shape implementation
  - pyproject.toml                         # `[[tool.uv.index]]` block
  - https://docs.astral.sh/uv/guides/integration/github/  # canonical reference
counter_examples:
  - "Skipping the smoke-test step ""because mise run check passed"" — your dev venv has a richer dep graph than what the wheel declares; transitive-only deps slip through every time. The PyPI install fails silently in user space hours later."
  - "Using ``twine check dist/*`` instead of an isolated install — twine validates metadata syntax (long_description renders, classifiers are real, etc.), not import resolution. It would not have caught the missing ``packaging`` dep."
  - "Sharing one job for both build + publish — break them so ``needs: build`` can gate the upload on smoke-test success. A failed smoke step then blocks the upload, instead of uploading a broken wheel and ``needs: smoke`` only catching it on a re-run."
  - "Hardcoding ``UV_PUBLISH_URL`` env var instead of ``[[tool.uv.index]]`` — works but disables ``--check-url`` (see uv docs: ""bypassing the project configuration prevents uv from checking if the package is already published"")."
references:
  - "uv GitHub Actions guide: https://docs.astral.sh/uv/guides/integration/github/#publishing-to-pypi"
  - "uv publish docs: https://docs.astral.sh/uv/guides/package/#publishing-your-package"
  - "uv publish --check-url: https://docs.astral.sh/uv/guides/package/#retries-and-resumability"
  - "PyPI Trusted Publishers: https://docs.pypi.org/trusted-publishers/"
  - "claude-sql PR #25 (2026-05-09) — switched from pypa/gh-action-pypi-publish to uv-native; first run of new shape caught the missing `packaging` dep that v0.3.0 had silently shipped to TestPyPI."
---

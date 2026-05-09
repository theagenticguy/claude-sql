---
title: bandit `-f sarif` needs the `[sarif]` extra — pulls jschema-to-python + sarif-om
track: knowledge
category: api-patterns
module: pyproject.toml
component: bandit (PyCQA)
severity: warning
tags: [bandit, sarif, dev-deps, ci, sast]
applies_when:
  - "Adding bandit to a Python project as a SARIF-emitting CI gate"
  - "`uv add --group dev bandit` then `bandit -f sarif` raises a missing-dep error"
pattern: |
  Bandit ships SARIF formatting in core, but the formatter's runtime
  dependencies (`jschema-to-python`, `sarif-om`) install only via the
  `[sarif]` extra. Without it, `bandit -f sarif` fails at runtime with a
  missing-import error — not at install time. The CI surface is silent
  until the first run.

  **Always install with the extra:**

      uv add --group dev 'bandit[sarif]>=1.9.4'

  Or in a CI install step:

      pip install 'bandit[sarif]>=1.9.4'

  The extra resolves to:
    - `jschema-to-python>=1.2.3`
    - `jsonpickle>=4.1.1`
    - `pbr>=7.0.3`
    - `sarif-om>=1.0.4`
    - `setuptools>=82.0.1`
    - `stevedore>=5.7.0`

  Pin a floor (`>=1.9.4`) — bandit's SARIF emitter has rolled twice in
  the last two minor releases and the field shape is stable from 1.9
  onward.

  **Pair with a `[tool.bandit]` config block in `pyproject.toml`** so
  CI and local invocations share the same `exclude_dirs` / `skips`:

      [tool.bandit]
      exclude_dirs = ["tests", ".venv", ".erpaval", "docs"]
      skips = ["B101", "B404", "B603", "B607", "B608"]

  When running bandit alongside ruff's `flake8-bandit` (S) selectors,
  align bandit's `skips` 1:1 with the ruff S-ignores so the two
  scanners agree on what's a false positive — otherwise you get
  conflicting verdicts on the same issue and contributors lose trust
  in either signal.
example_files:
  - pyproject.toml                              # [tool.bandit] block
  - .github/workflows/bandit.yml                # CI install + SARIF emit
  - .erpaval/sessions/session-2293a5/research-security.yaml  # research that surfaced this
---

# Why this matters

The default `pip install bandit` looks complete — `bandit --help` lists
`-f sarif` in `--format` choices — so a CI step that runs
`bandit -r src -f sarif -o bandit.sarif` looks fine until the first run
in CI fails with `ModuleNotFoundError: No module named 'jschema_to_python'`.
Pinning `bandit[sarif]` makes the dependency explicit and surfaces the
break at install time, not at first run.

The 1:1 alignment with ruff S-ignores matters more than it looks.
Bandit and ruff overlap ~70% on the bandit B-rule set; if you add
bandit to CI without aligning the skips, you get conflicting CI
signals (ruff passes, bandit fails on the same line) and contributors
end up adding `# nosec` annotations that drift from `# noqa: S...`.
Keeping the two scanners' opinions identical on a small set of
principled false positives keeps the signal-to-noise high.

# Example

```toml
# pyproject.toml — bandit aligned with ruff S
[dependency-groups]
dev = [
  "bandit[sarif]>=1.9.4",  # the [sarif] extra is required for `-f sarif`
  # ...
]

[tool.bandit]
# Align with ruff's S-ignores so the two scanners agree on false positives.
exclude_dirs = ["tests", ".venv", ".erpaval", "docs"]
skips = [
  "B101",   # assert: matches per-file-ignore for tests + production type-narrowing
  "B404",   # import subprocess: matches the read-only pattern in binding.py
  "B603",   # subprocess.run without shell=True: matches ruff `S603` ignore
  "B607",   # partial executable path in subprocess: matches ruff `S607`
  "B608",   # SQL string concat: matches ruff `S608` (DuckDB f-string SQL)
]
```

```yaml
# .github/workflows/bandit.yml — CI side
- name: Install bandit[sarif]
  run: pip install 'bandit[sarif]>=1.9.4'
- name: Run bandit (SARIF)
  # --exit-zero keeps step green so SARIF uploads even on findings;
  # gating happens through GitHub code scanning, not the scan exit code.
  run: bandit -r src -f sarif -o bandit.sarif --exit-zero -c pyproject.toml
- uses: github/codeql-action/upload-sarif@v4
  if: always()
  with:
    sarif_file: bandit.sarif
    category: bandit
```

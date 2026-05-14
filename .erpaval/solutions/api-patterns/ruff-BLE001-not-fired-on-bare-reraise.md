---
title: Ruff `BLE001` does not fire on `except Exception: ...; raise` — `# noqa: BLE001` is rejected as unused
track: knowledge
category: api-patterns
module: pyproject.toml / ruff config
component: ruff (32-family selector incl. BLE)
severity: info
tags: [ruff, lint, BLE001, broad-except, noqa, codeql, claude-sql]
applies_when:
  - "Project enables `flake8-blind-except` (`BLE001`) under ruff's strict selector"
  - "Adding a one-line rationale comment to a register-or-fail-loud broad except"
  - "The except body bare-raises after logging — the immediate-reraise pattern"
pattern: |
  Ruff's `BLE001` rule (flake8-blind-except) flags `except BaseException`
  / `except Exception` *as a smell* — but it does **not** fire when the
  except body re-raises immediately. The reasoning: a broad except that
  bare-raises is just a try/finally with a logging hook; the exception
  isn't being swallowed, so the rule's stated risk (silent error
  consumption) doesn't apply.

  Example that ruff considers clean:

  ```python
  try:
      do_thing()
  except Exception:
      logger.exception("…")
      raise
  ```

  Adding `# noqa: BLE001 — <reason>` to that line trips `RUF100`
  ("unused noqa directive") because the suppressed rule never
  triggered in the first place.

  This bites the project's CodeQL hygiene practice: `CLAUDE.md` says
  every broad except gets a one-line rationale comment per the
  CodeQL `py/empty-except` rule. The instinct is to write the comment
  AS a `# noqa: BLE001` since that's how five other sites in the repo
  already do it (`sql_views.py:1751`, `checkpointer.py:159`,
  `judge_worker.py:339`, `trajectory_worker.py:647 / :862`). But all
  those sites have a body that does NOT bare-raise — they log,
  decide, and continue or swallow. So `BLE001` fires on those sites,
  and the `noqa` is load-bearing. On a register-or-fail-loud
  reraise, `BLE001` doesn't fire, so the `noqa` is rejected.

  **Fix:** put the rationale on the line ABOVE the except, as a
  plain comment. Same readability, no false `noqa`:

  ```python
  try:
      do_thing()
  except Exception:
      # register-or-fail-loud — any DuckDB error must surface.
      logger.exception("…")
      raise
  ```

  This satisfies CodeQL's `py/empty-except` (which ruff doesn't
  proxy) and avoids ruff's `RUF100`. The discipline is identical;
  only the comment placement differs.

  Verification: `mise run check`'s lint stage will surface
  `RUF100 [*] Unused noqa directive (unused: BLE001)` if you put a
  noqa on a bare-raise except. Treat it as the signal that the
  comment belongs above the line.
example_files:
  - src/claude_sql/sql_views.py:601 (register_raw — register-or-fail-loud, comment above)
  - src/claude_sql/sql_views.py:1075 (register_views — same shape)
  - src/claude_sql/checkpointer.py:209 (bulk-INSERT rollback — same shape)
  - src/claude_sql/sql_views.py:1751 (best-effort migration — `# noqa: BLE001` IS fired and load-bearing)
  - src/claude_sql/checkpointer.py:159 (same — best-effort migration, noqa fires)
counter_examples:
  - "`except Exception: pass` (no reraise) — BLE001 fires; noqa is required."
  - "`except Exception as exc: handle(exc); return None` — BLE001 fires; noqa is required."
references:
  - "Ruff docs: https://docs.astral.sh/ruff/rules/blind-except/"
  - "CLAUDE.md `CodeQL hygiene` — `py/empty-except` rule"
  - ".erpaval/solutions/best-practices/codeql-finds-narrow-except-and-unused-nested-classes.md"
---

The session that produced this lesson hit it during v1.0.1: three
new `# noqa: BLE001 — <reason>` comments tripped `RUF100`. Re-shaped
the comments to live above the except instead of beside it; ruff +
CodeQL both happy.

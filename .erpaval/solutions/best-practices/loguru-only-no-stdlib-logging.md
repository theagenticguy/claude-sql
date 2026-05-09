---
title: Single-logger discipline — ban stdlib `logging` when the package uses loguru
track: knowledge
category: best-practices
module: claude_sql/logging_setup.py + pyproject.toml ruff config
component: loguru + tenacity + ruff flake8-tidy-imports
severity: warning
tags: [loguru, logging, tenacity, before_sleep, ruff, banned-api, claude-sql]
applies_when:
  - "Project standardizes on loguru (`from loguru import logger`) for all observability"
  - "One or more modules use tenacity ``@retry(before_sleep=...)`` against transient errors"
  - "Anything else (third-party libraries, CLI startup probes) is tempted to pull in stdlib ``logging``"
pattern: |
  Mixing stdlib ``logging`` with loguru is silently broken: messages
  emitted via ``logging.getLogger(...)`` land in the stdlib root logger,
  which has no handler unless something downstream explicitly bridges
  the two. The user's loguru sink (the one ``claude-sql shell`` /
  every CLI subcommand installs in ``configure_logging``) never sees
  them. They just disappear.

  The classic foothold for stdlib ``logging`` in an otherwise
  loguru-only package is tenacity's ``before_sleep_log`` callback:

  ```python
  # bad — pulls stdlib logging into a loguru-only module
  import logging
  from tenacity import before_sleep_log, retry

  _retry_logger = logging.getLogger("claude_sql.embed_worker")

  @retry(before_sleep=before_sleep_log(_retry_logger, logging.WARNING))
  def call_bedrock(...): ...
  ```

  ``before_sleep_log`` accepts anything matching ``LoggerProtocol``, but
  loguru's ``logger`` doesn't (it's a different class with a different
  API). The historic shape pulls in stdlib ``logging`` for *just* this
  one hook and the ``logger.warning`` lines from those retries silently
  bypass the loguru sink.

  **Fix**: write a tiny loguru-backed callback once and ban the import:

  ```python
  # logging_setup.py
  def loguru_before_sleep(level: str = "WARNING") -> Callable[[RetryCallState], None]:
      def _before_sleep(retry_state: RetryCallState) -> None:
          if retry_state.outcome is None or retry_state.next_action is None:
              return
          fn_name = getattr(retry_state.fn, "__qualname__", repr(retry_state.fn))
          if retry_state.outcome.failed:
              exc = retry_state.outcome.exception()
              verb, value = "raised", f"{exc.__class__.__name__}: {exc}"
          else:
              verb, value = "returned", retry_state.outcome.result()
          logger.log(
              level,
              "Retrying {} in {:.3g} seconds as it {} {}.",
              fn_name,
              retry_state.next_action.sleep,
              verb,
              value,
          )
      return _before_sleep
  ```

  Then enforce the ban via ruff:

  ```toml
  [tool.ruff.lint.flake8-tidy-imports.banned-api]
  "logging".msg = "Use loguru via `from loguru import logger`. For tenacity retry callbacks use `claude_sql.logging_setup.loguru_before_sleep('LEVEL')`."
  ```

  Ruff emits ``TID251`` on any future ``import logging`` so the next
  contributor can't accidentally re-introduce the dual-logger split.

  **Detection**: run ``grep -rn '^import logging\|from logging import'
  src/`` after every PR. With the banned-api rule active the grep is
  belt + suspenders, but it's worth doing once when adopting the rule.

  **Enforcement coverage**: The ruff banned-api rule fires on any path
  ruff lints. Make sure the lint task includes ``src/`` and ``tests/``
  (in this repo: ``mise run lint`` runs ``ruff check .``). Don't scope
  the ban to ``src/`` only — test code that imports ``logging`` for
  caplog is the *one* legitimate exception, and it should be opted into
  via per-file ``# noqa: TID251`` rather than a blanket carve-out.
example_files:
  - src/claude_sql/logging_setup.py        # `loguru_before_sleep` definition
  - src/claude_sql/embed_worker.py         # consumer, no stdlib logging
  - src/claude_sql/llm_worker.py           # consumer
  - src/claude_sql/judge_worker.py         # consumer
  - pyproject.toml                          # `[tool.ruff.lint.flake8-tidy-imports.banned-api]`
  - CLAUDE.md                              # § "Logging: loguru only, no stdlib `logging`"
counter_examples:
  - "Adding a stdlib ``logging`` handler that forwards to loguru — works but adds startup cruft + a place where future-you forgets to install the bridge. Keep the topology one-way: nothing imports ``logging``."
  - "Per-file ``# noqa: TID251`` to silence the ban for one new caller — only legitimate when *receiving* a logger from a third-party API that genuinely demands stdlib (rare). For a hook callback, write a loguru adapter instead."
  - "Letting tenacity emit warnings via stdlib root logger because ""that's just how tenacity works"" — it isn't; tenacity accepts any ``before_sleep`` callable and the retry state is fully accessible to a custom function."
references:
  - "tenacity before_sleep API: https://tenacity.readthedocs.io/en/latest/#before-and-after-retry-and-logging"
  - "loguru migration from logging: https://loguru.readthedocs.io/en/stable/resources/migration.html"
  - "ruff TID251 banned-api: https://docs.astral.sh/ruff/rules/banned-api/"
  - "claude-sql PR #22 follow-up (2026-05-09) — caught after CodeQL flagged a narrow except + made me notice three workers still imported stdlib logging just for the tenacity hook."
---

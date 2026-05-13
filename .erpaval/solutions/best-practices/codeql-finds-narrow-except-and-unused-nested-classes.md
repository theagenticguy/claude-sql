---
title: CodeQL flags three patterns ruff misses — narrow `except: pass` without a comment, unused classes nested inside a function, and `except BaseException` swallowing CancelledError
track: knowledge
category: best-practices
module: tests/**/*.py + pyproject.toml ruff config
component: GitHub Advanced Security CodeQL vs ruff
severity: warning
tags: [codeql, github-advanced-security, ruff, S110, F841, BLE001, dead-code, base-exception, anyio, cancellation, claude-sql]
applies_when:
  - "Project has GitHub Advanced Security / CodeQL enabled on PRs"
  - "Project relies on ruff (any selector, even strict) as the local lint gate"
  - "Tests or implementation code uses ``try/except: pass`` blocks or defines classes/functions inside a test body"
pattern: |
  Two CodeQL rules consistently fire on ``claude-sql`` PRs after `mise
  run check` is fully green. Both correspond to ruff rules — but ruff
  only catches a strict subset of what CodeQL flags.

  **`py/empty-except`** — empty ``except: pass`` without an explanatory
  comment. Ruff's ``S110`` (try-except-pass) only fires when the
  exception is broad (``except Exception:``, ``except BaseException:``).
  A *narrow* multi-class exception is invisible to ruff. Example:

  ```python
  try:
      import tenacity.nap
      monkeypatch.setattr(tenacity.nap.time, "sleep", noop)
  except (ImportError, AttributeError):
      pass  # ← CodeQL: empty except without explanatory comment
  ```

  Ruff sees ``except (ImportError, AttributeError)`` as narrow and
  doesn't flag it. CodeQL flags ANY empty body without an inline
  comment. **Fix**: add a one-line comment explaining what the
  exception represents and why ignoring it is correct:

  ```python
  except (ImportError, AttributeError):
      # tenacity.nap is private API; ImportError covers a future tenacity
      # version dropping the submodule, AttributeError covers one that
      # stops re-exporting `time`. Either way the top-level time.sleep
      # patch above is enough — fall through silently.
      pass
  ```

  **`py/unused-local-variable`** — classes (or functions) defined inside
  another function but never referenced. Ruff's ``F841`` is the closest
  match, but only fires on ``name = expr`` assignments. ``class Foo:``
  declarations nested inside a function are NOT inspected:

  ```python
  def test_safe_preview_falls_back_to_str_on_unserializable() -> None:
      class NotJSON:               # ← CodeQL: unused local variable
          def __repr__(self): ...
      class Hostile:                # ← also unused
          ...
      # only this third one actually gets used
      class HardToSerialize: pass
      h = HardToSerialize()
      assert isinstance(_safe_preview(h), str)
  ```

  **Fix**: delete the unused scaffolding. If the test was originally
  trying to demonstrate three exotic shapes and only the third matters
  for coverage, just keep the third one — bricolage scaffolding adds
  noise and rots.

  **`py/catch-base-exception`** — `except BaseException` (not just
  `except Exception`). Ruff's ``BLE001`` is the closest match but it
  fires on `except Exception:` too — most projects ignore or
  per-project-disable it because of how often the broad form shows up
  in async dispatchers (see `pipeline-accumulator-explicit-key.md`).
  CodeQL is stricter: catching `BaseException` swallows
  `KeyboardInterrupt`, `SystemExit`, and crucially
  `asyncio.CancelledError` / `anyio.get_cancelled_exc_class()`. In an
  anyio task group, swallowing `CancelledError` *deadlocks* shutdown —
  the parent's `aclose()` waits forever for child tasks that already
  observed cancellation but never re-raised it. Example from
  `trajectory_worker.py` (caught by GHAS on PR #42):

  ```python
  # WRONG — task group can't tear down cleanly
  try:
      ... # bedrock + parquet write
  except BaseException as exc:  # noqa: BLE001 — never abort the task group
      logger.warning("session {} aborted ({})", sid, exc)
      retry_queue.enqueue(..., error=str(exc))
      return
  ```

  ```python
  # RIGHT — Exception covers all recoverables; CancelledError still propagates
  try:
      ...
  except Exception as exc:  # noqa: BLE001 — non-cancel exceptions go to retry; CancelledError still tears down the task group
      logger.warning("session {} aborted ({})", sid, exc)
      retry_queue.enqueue(..., error=str(exc))
      return
  ```

  **Heuristic for review**: when you write `except BaseException` in
  an async dispatcher, ask "if the user hits Ctrl-C, do I want this
  branch to run instead of the cancellation propagating?" The answer
  is almost always no. The narrow exception list this is meant to
  catch (network blip, parquet schema error) are all `Exception`
  subclasses already. The only legitimate uses of `except
  BaseException` are signal-handling shims and CLI top-level
  fall-throughs that re-raise after logging — neither shows up in
  worker code.

  **No ruff selector covers any of the three gaps today**, so
  prevention has to live in two places:
  1. ``CLAUDE.md`` instructions tell future Claude sessions to add an
     explanatory comment to every ``except: pass``, to delete unused
     test classes, and to never `except BaseException` in async
     dispatchers before committing.
  2. ``pyproject.toml`` keeps ``S``, ``BLE``, ``SIM``, ``F``, ``RUF``
     families maximally aggressive so the next ruff release that adds a
     real rule for either pattern picks up automatically.
example_files:
  - tests/test_llm_worker_pipelines.py        # narrow except: pass + comment fix
  - tests/test_review_sheet_worker_extras.py  # unused-class deletion
  - src/claude_sql/trajectory_worker.py       # except BaseException → except Exception in async dispatcher
  - CLAUDE.md                                 # § "CodeQL hygiene"
counter_examples:
  - "Adding a per-file ``# noqa: S110`` to silence the ruff variant — doesn't help because ruff didn't flag it; the comment satisfies CodeQL but is wasted noise on the ruff side."
  - "Wrapping the empty except in ``contextlib.suppress(ImportError, AttributeError):`` — equivalent semantics but converts a narrow opt-out into a more deliberate construct that CodeQL accepts. Slightly more verbose, but eliminates the comment-or-bust requirement. Use this for repeated-pattern cases (e.g. multiple optional-import probes)."
  - "Keeping unused test classes ""for documentation purposes"" — they get out of sync with the implementation and become misleading. If you need an example, put it in a docstring."
references:
  - "GitHub CodeQL py/empty-except: https://codeql.github.com/codeql-query-help/python/py-empty-except/"
  - "GitHub CodeQL py/unused-local-variable: https://codeql.github.com/codeql-query-help/python/py-unused-local-variable/"
  - "GitHub CodeQL py/catch-base-exception: https://codeql.github.com/codeql-query-help/python/py-catch-base-exception/"
  - "ruff S110 docs: https://docs.astral.sh/ruff/rules/try-except-pass/"
  - "ruff BLE001 docs: https://docs.astral.sh/ruff/rules/blind-except/"
  - "anyio cancellation guide: https://anyio.readthedocs.io/en/stable/cancellation.html"
  - "claude-sql PR #22 (2026-05-09) — three findings: 1× py/empty-except, 2× py/unused-local-variable."
  - "claude-sql PR #42 (2026-05-13) — two py/catch-base-exception findings on `trajectory_worker.py:647` and `:862`. Both narrowed to `except Exception`; tests still green. Caught after a green ``mise run check`` because CodeQL only runs in CI."
---

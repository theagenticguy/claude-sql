# CodeQL hygiene — the rules ruff misses

GitHub Advanced Security runs CodeQL on every PR. Ruff's 32-family selector
catches most issues, but four CodeQL rules fire on patterns ruff lets through.
Each has a one-line fix; each is easy to forget. These checks run automatically —
there is no `mise run codeql` task — so treat the discipline below as part of
writing the code (and the test) in the first place.

## `py/empty-except` — every `try/except: pass` needs a comment

Ruff's `S110` only fires on broad excepts (`except Exception:`). A *narrow*
except (`except (ImportError, AttributeError):`) with a bare `pass` body is
invisible to ruff, but CodeQL flags it without an inline explanation. Always pair
the `except: pass` with a one-line comment naming what the exception represents
and why ignoring it is correct:

```python
try:
    import optional_dep
except ImportError:
    # Optional dep — no-op when not installed; the caller's fallback handles it.
    pass
```

**Bare-reraise wrappers don't trip `BLE001`.** A `try / except Exception: log;
raise` block is treated as clean by ruff (the exception isn't swallowed). Putting
`# noqa: BLE001 — <reason>` on that line trips `RUF100` (unused noqa). Put the
rationale **on the line above** the except as a plain comment instead — same
intent, no false noqa. The `# noqa: BLE001 — <reason>` form is still load-bearing
on broad excepts whose body **doesn't** re-raise (best-effort migrations,
log-and-skip workers).

## `py/unused-local-variable` — nested classes/functions in tests must be used

Ruff's `F841` only flags assigned names, not nested `class Foo: …` declarations.
Don't define test scaffolding (mock classes, fake exceptions, helper functions)
you don't end up using. If a class was kept "just in case" or for a branch that
never materialized, delete it before committing.

## `py/catch-base-exception` — never `except BaseException` in async task bodies

`BaseException` swallows `KeyboardInterrupt`, `SystemExit`, and crucially
`asyncio.CancelledError`. In an anyio task group, swallowing `CancelledError`
deadlocks shutdown — the parent's `aclose()` waits forever for child tasks that
observed cancel but never re-raised it. The fix is one character: `BaseException`
→ `Exception`. Recoverable errors (network blip, parquet schema mismatch, refused
refusal) are all `Exception` subclasses; cancellation cleanly cascades. The only
legitimate `except BaseException` lives in CLI top-level `try/except` blocks that
re-raise after logging — never in worker code.

## `py/file-not-closed` — open the fd inside the closure when faking `mkstemp`

When monkeypatching `tempfile.mkstemp` to return a `(fd, path)` tuple, open the fd
inside the closure per-call, not at module level. Production code conventionally
calls `os.close(fd)` because mkstemp's contract gives the caller fd ownership. A
module-level `fd = os.open(...)` reused across calls trips CodeQL's data-flow
tracker (open-site and close-site are decoupled) AND a per-test
`addFinalizer(lambda: os.close(fd))` produces `Bad file descriptor` on a
double-close because the consumer ALSO closes. Open inside the closure each call
so producer and consumer pair statically. See
`.erpaval/solutions/best-practices/codeql-py-file-not-closed-in-test-mkstemp-fakes.md`
for the full pattern.

## Keep the ruff analogs aggressive

The ruff rules in `pyproject.toml` should be kept maximally aggressive so the
*next* class of CodeQL findings has a ruff-side analog wherever one exists.

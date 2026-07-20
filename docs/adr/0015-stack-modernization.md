# ADR 0015 — Dev toolchain modernization (Python 3.13, strict ruff + ty)

**Status:** accepted — 2026-05-08
**Context:** branch `chore/stack-modernization`

## Context

The dev toolchain was working but baseline. Python floor was `>=3.12`; ruff ran
with the minimal `E,F,I,N,UP,B,SIM` selector set; ty ran on defaults over
`src/` only; `[tool.uv]` wasn't configured; lefthook used v1.x-ish idioms in a
v2 file; commitizen lacked pre/post bump hooks.

The bar this ADR raises: the stack catches defects at commit time that used to
land in PRs (security smells, datetime-naive bugs, blind exception swallows,
pathlib misuse), without drowning contributors in false positives.

## Decision

### Python: default `3.14`, floor `>=3.13`

**Update 2026-07-20:** `hdbscan 0.8.44` shipped cp314 wheels, so the flip
below has been executed. `.python-version` and `mise.toml [tools].python`
now pin `3.14`; `requires-python` stays `>=3.13` so the wheel installs on
both interpreters. hdbscan floor bumped to `>=0.8.44`; the stale
`leidenalg` pin was widened to `>=0.11.0,<0.13` (resolves to 0.12.0).
Full suite (664 tests) green on 3.14.4, including the seed-determinism
tests. See Consequences below.

Original rationale (pre-flip): every native dep in the closure **except
hdbscan** shipped cp314 wheels as of 2026-05-08 (duckdb, numpy, scipy,
scikit-learn, pyarrow, pydantic-core, pyyaml, numba, llvmlite).
`hdbscan 0.8.42` stopped at cp313. On 3.14, pip/uv fell back to the sdist
— Cython + numpy + C toolchain at install time. Unacceptable for the
`uv tool install claude-sql` end-user path per `CLAUDE.md`. The floor was
held at `>=3.13` until `hdbscan 0.8.43+` published cp314 wheels.

### Ruff: 32-family selector superset

Expanded from `E,F,I,N,UP,B,SIM` to add: `ANN`, `ASYNC`, `BLE`, `C4`, `DTZ`,
`ERA`, `FBT`, `G`, `ICN`, `ISC`, `LOG`, `PERF`, `PIE`, `PL`, `PT`, `PTH`,
`RET`, `RSE`, `S`, `T20`, `TID`, `TRY`, `PGH`, `RUF` (plus `W`).

What this catches beyond the baseline: naive datetimes (DTZ), blind
`except Exception` swallows (BLE, TRY), eager logging formatting (G, LOG),
boolean-trap APIs (FBT), `os.path` → pathlib (PTH), pytest idioms (PT),
security smells (S — bandit), dead commented code (ERA), async anti-patterns
(ASYNC — directly relevant to `anyio` / `AsyncBedrockRuntime`), perf patterns
(PERF), import-convention enforcement (ICN), stray `print` outside the CLI
(T20), relative-import ban (TID252), and ruff-native catches (RUF) that ship
new bug-finders between releases.

Principled ignores — reasons documented inline in `pyproject.toml`:

- **Formatter overlap** — `E501`, `ISC001`, `COM812`.
- **Dogmatic noise** — `ANN401` (deliberate `Any`), `TRY003`, `TRY300`.
- **PL refactor metrics / magic-value** — `PLR0911/12/13/15`, `PLR2004`.
- **Project-specific patterns** — `PLC0415` (lazy imports for expensive/optional
  deps), `S608` (local DuckDB SQL f-strings, not injection), `RUF00{1,2,3}`
  (intentional non-ASCII in docstrings), `PTH111` (os.path.expanduser-heavy
  config layer; call sites immediately coerce back to str), `PERF203`
  (try/except in loop is the DuckDB IOException retry pattern).

Per-file ignores scope test-friendly relaxations (S101, PT018, PLR2004, ANN,
T20, FBT001/2) without polluting `src/`.

### ty: strict mode via `[tool.ty.rules] all = "error"`

ty has no `--strict` flag; `all = "error"` is the Astral-canonical knob. It
promotes the ~9 warn-default rules (deprecated, ineffective-final, invalid-
ignore-comment, mismatched-type-name, unused-ignore-comment, etc.) so new ty
rules surface as CI failures instead of drifting into acknowledged noise.

`division-by-zero` stays at `ignore` because Astral themselves disable it by
default over false-positive rate.

`src.include = ["src", "tests"]` closes the test-directory blind spot — the
old `ty check src/` invocation missed test-only regressions.

`error-on-warning = true` future-proofs the strict stance so any future
warn-default rule fails CI the moment ty ships it.

Tests carry a `[[tool.ty.overrides]]` block relaxing the DuckDB `Optional`-
subscript false-positive class (`not-subscriptable`, `not-iterable`,
`unsupported-operator`, `invalid-argument-type`). Each override line has a
one-line reason. Scope is `tests/**`; `src/` stays tight.

### uv: reproducibility-first `[tool.uv]` block

`required-version = ">=0.11.7"` gates parsing before any command runs — older
uv can't silently ignore newer keys. `python-preference = "only-managed"`
refuses to bind against whatever `python3` happens to be on PATH.
`compile-bytecode = true` moves `.pyc` generation to install time (matters
for a CLI cold-start). `link-mode = "clone"` is the Linux/macOS default;
pinning it makes intent explicit across machines.

### lefthook: v2.1 idioms

Floor bumped to `2.1.6`, `assert_lefthook_installed: true`, `glob_matcher:
doublestar`, a `templates.uv` macro that DRYs `uv run` and can be overridden
in `lefthook-local.yml`, `output: [meta, summary, failure, execution_info]`
(replacing the dropped-in-v2 `skip_output`), `priority: 1/2/3` on pre-commit
jobs for deterministic dispatch, `interactive: false` on commitizen,
`fail_text` on every job pointing contributors at the equivalent `mise run`
command. ty scope widened to match `[tool.ty.src.include]`.

### commitizen: pre/post bump hooks, merged-prerelease changelog

Added `changelog_merge_prerelease = true`, explicit `allow_abort = false` +
`allowed_prefixes`, and `pre_bump_hooks = ["uv lock"]` +
`post_bump_hooks = ["uv sync"]` so `uv.lock` is current at the release tag
and `.venv` is hydrated immediately after.

### mise: sources-cached quality gates + uv-aware venv

`min_version = "2025.1.0"`, `[settings] python.uv_venv_auto = "create|source"`
(replacing the legacy `_.python.venv` block), hygiene env
(`PYTHONDONTWRITEBYTECODE=1`, `UV_LINK_MODE=copy`), `sources = [...]` on lint/
fmt/typecheck/test for ~95% speedup on docs-only `mise run check` loops, new
`lock:check` task for CI gating, and `hide = true` on internal tasks.

## Consequences

- **Stricter CI.** Every commit now passes 32 ruff families, strict ty, and
  has its message format verified by commitizen. Rejected-because-noisy:
  contributors run `mise run fmt:write` + `ruff check --fix` locally.
- **Expected first-run surface.** Enumerated in session research notes —
  PTH/DTZ/BLE/G/S311/PERF401 were the top categories; all fixed or given
  principled in-file noqa with a one-line reason.
- **Test-dir blind spot closed.** ty now runs over `tests/`; 17 DuckDB-
  Optional false positives caught at config time and scoped to a tests/**
  override.
- **Reproducibility.** `[tool.uv]` + `.python-version = 3.14` +
  `python-preference = only-managed` pins the exact interpreter across
  machines. `uv lock --check` available as a CI gate.
- **Hdbscan cp314 watch — resolved 2026-07-20.** `hdbscan 0.8.44` shipped
  cp314 wheels, so the flip is done: `.python-version` and
  `mise.toml [tools].python` → `3.14`, `requires-python` held at `>=3.13`
  (wheel installs on both), hdbscan floor → `>=0.8.44`, and the stale
  `leidenalg` pin widened to `>=0.11.0,<0.13` (resolves 0.12.0). The full
  664-test suite is green on cp314 (Python 3.14.4), seed-determinism tests
  included; no other dep was blocking.

## Sources

Per-tool research packets live under `.erpaval/sessions/session-638df1/`:
`research-ruff.yaml`, `research-ty.yaml`, `research-uv.yaml`,
`research-lefthook.yaml`, `research-commitizen.yaml`, `research-mise.yaml`,
`research-python314.yaml`.

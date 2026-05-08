# ty strict mode = `rules.all = "error"` (no --strict flag)

**Category:** api-patterns
**Tags:** ty, astral, type-checker, strict-mode, python
**Applies to:** any project using Astral's `ty` (2026-05-08 alpha stream, 0.0.x)

## Situation

ty (Astral's Python type checker, alpha) does **not** expose a `--strict`
CLI flag or a `mode = "strict"` pyproject option like pyright does. The
canonical "make it strict" knob is instead:

```toml
[tool.ty.rules]
all = "error"
```

This promotes every warn-default diagnostic (~9 rules as of 0.0.34:
`ambiguous-protocol-member`, `deprecated`, `ineffective-final`,
`ignore-comment-unknown-rule`, `invalid-enum-member-annotation`,
`invalid-ignore-comment`, `invalid-named-tuple-override`,
`mismatched-type-name`, `unused-ignore-comment`) and the one
ignore-default rule (`division-by-zero`). Combined with
`[tool.ty.terminal] error-on-warning = true`, any future warn rule
ty adds also fails CI the moment it ships.

**The only rule to keep silenced is `division-by-zero`** — Astral's own
docs disable it by default because of false-positive rate. Promoting it
adds noise without catching bugs.

## Canonical strict config

```toml
[tool.ty.environment]
python-version = "3.13"
python-platform = "linux"
python = ".venv"
root = ["./src"]

[tool.ty.src]
include = ["src", "tests"]         # include tests; src-only is a blind spot
respect-ignore-files = true

[tool.ty.terminal]
error-on-warning = true
output-format = "concise"

[tool.ty.rules]
all = "error"
division-by-zero = "ignore"

[[tool.ty.overrides]]
include = ["tests/**"]

[tool.ty.overrides.rules]
# Relax only specific false-positive classes that come from dynamic
# library returns (DuckDB fetchone() typed as Optional, etc.).
not-subscriptable = "ignore"
not-iterable = "ignore"
unsupported-operator = "ignore"
invalid-argument-type = "ignore"
possibly-unresolved-reference = "warn"
possibly-missing-attribute = "warn"
unused-ignore-comment = "warn"
```

## Escape-hatch hierarchy (prefer first)

When ty flags something that's actually OK:

1. **Typed shim** — `cast(tuple[str, int], row)` or a `NamedTuple`/
   `TypedDict` wrapping the dynamic return. Makes the schema reviewable.
2. **Per-line `# ty: ignore[rule-name]`** — scoped to one call site. The
   `unused-ignore-comment` rule keeps these honest by flagging stale
   suppressions.
3. **Narrow `replace-imports-with-any`** in `[tool.ty.analysis]` for a
   specific misbehaving submodule.
4. **Per-file override** via `[[tool.ty.overrides]] include = [...]` —
   last resort, scoped to a directory.
5. **Blanket file-level ignores** — banned. They hide real bugs.

## Why non-obvious

- No `--strict` flag means searching "ty strict mode" in docs leads to
  dead ends; the `all = "error"` idiom is documented but easy to miss.
- The `--error`/`--warn`/`--ignore` CLI flags take `all` as a
  special-case target — mirrors the pyproject rules block but is
  config-path-independent.
- Widening `src.include` beyond `["src"]` is critical; the default
  misses regressions in `tests/`.

## Verification

```bash
uv run ty check                    # picks up pyproject config
# or with explicit flags:
uv run ty check --error all --error-on-warning --output-format concise src/ tests/
```

First-run expect: a burst of newly-promoted diagnostics from the ~9
warn-default rules. Each is either a real deprecated-API usage to fix,
a stale `# ty: ignore` pragma to delete, or a legitimate protocol
ambiguity. Do **not** revert `all = "error"` to silence the burst — fix
or narrowly suppress each.

## See also

- ADR `docs/adr/0015-stack-modernization.md` in claude-sql (session-638df1).
- Astral ty docs: https://docs.astral.sh/ty/ (rules, configuration).

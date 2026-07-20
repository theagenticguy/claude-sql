# `returns` Result[T,E] under ty strict: the plugin-free subset

**Category:** api-patterns
**Tags:** returns, ty, typecheck, result-type, hexagonal
**Session:** session-4fb0fd

## Lesson

`returns` (0.28.0) type-safety was designed around its **mypy plugin**; ty has
no plugin system and Astral won't add one. Empirically verified under ty strict
(`all = "error"`), same venv as the project:

| Pattern | ty verdict |
|---|---|
| `Success(x)` / `Failure(e)`, `Result[T, E]` annotations | clean |
| `is_successful(r)` + `.unwrap()` / `.failure()` | clean |
| `match r: case Success(v) / case Failure(e)` narrowing | clean |
| `.map(fn)` / `.alt(fn)` | clean (error type may degrade to Any, no error) |
| **`.bind(fn)`** | **hard error** (raw `KindN` HKT signature) |
| `@safe`, `flow`, `pipe`, do-notation | plugin-dependent — banned |

**Rule adopted:** `Result[T, DomainError]` construction/consumption only —
`Success`/`Failure`/`is_successful`/`match`/`.map`/`.alt`. Never `.bind` or the
functional stack. `is_successful` imports from `returns.pipeline`.

Import cost ~50ms — fine at module top in domain/application; keep it out of a
CLI's sub-second entry path only if the budget is razor-thin.

## Also decided (same research): no dishka for CLIs with lazy-import budgets

dishka costs ~120ms at import (vendored `_adaptix` type introspection) — it
fights a test-enforced sub-second CLI startup, and a facade like
`ClaudeSql(embedder=...)` is better served by plain constructor injection +
hand-rolled factory functions with heavy imports inside the factory bodies.

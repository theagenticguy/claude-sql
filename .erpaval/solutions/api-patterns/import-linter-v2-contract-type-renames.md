---
name: import-linter-v2-contract-type-renames
description: import-linter v2.x renamed `type = "layered"` to `type = "layers"`. The old name fails silently (exit 1, prints "layered", no visible error) until you run with `--debug`. Maps to `NoSuchContractType: layered`.
metadata:
  type: api-pattern
  tags: [import-linter, contract, layered, layers, monorepo, dag]
  ref: https://import-linter.readthedocs.io/en/v2.3/usage.html
---

# import-linter v2 contract types: `layered` → `layers`, plus silent-failure mode

In import-linter v2.x the `type` value for the layered architecture contract is **`layers`**, not `layered`. The old name throws `NoSuchContractType` internally but the CLI swallows it: you see only the contract name printed, exit code 1, no error output.

## Failure shape

```toml
[[tool.importlinter.contracts]]
name = "claude-sql layered architecture"
type = "layered"   # ← WRONG in v2.x
layers = [...]
```

```
$ uv run lint-imports
... (banner) ...
layered           # ← prints the contract NAME, looks OK
$ echo $?
1
```

Run with `--debug` to see the real error:

```
NoSuchContractType: layered
```

## Fix

```toml
[[tool.importlinter.contracts]]
name = "claude-sql layered architecture"
type = "layers"    # ← CORRECT in v2.x
layers = [
    "claude_sql.app",
    "claude_sql.analytics | claude_sql.evals | claude_sql.provenance",
    "claude_sql.core",
]
```

`independence` and `forbidden` keep their v1 names. Only `layered` was renamed.

## Discovering the available types

```bash
uv run python -c "
import pkgutil, importlinter.contracts
for _, name, _ in pkgutil.iter_modules(importlinter.contracts.__path__):
    print(name)
"
# _common, acyclic_siblings, forbidden, independence, layers, protected
```

`acyclic_siblings` and `protected` are also new in v2. Existing v1 configs that used `type = "layered"` need a one-character fix; everything else works as documented.

## When you'll hit this

Migrating any project that used import-linter v1 to v2.x — including any new monorepo split that copies a config example from older blog posts. The bonk synthesis from 2026-05-14 wrote `type = "layered"` because that's what the v1 docs showed; the `[[tool.importlinter.contracts]]` array shape worked but the type name was stale.

## See also

- [[uv-build-namespace-form-1a-explicit-module-name]] — the bonk synthesis pivot lessons
- [[parallel-agents-shared-file]] — orchestration discipline during a workspace split

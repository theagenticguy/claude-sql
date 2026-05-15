---
name: uv-build-namespace-form-1a-explicit-module-name
description: For PEP 420 namespace packages under uv workspaces, use explicit dotted `[tool.uv.build-backend] module-name = "ns.member"` per member — NOT `namespace = true`, NOT `module-root = ""`. Three alternative forms, only 1a is the recommended shape.
metadata:
  type: api-pattern
  tags: [uv, uv-build, pep-420, namespace, workspace, monorepo]
  ref: https://docs.astral.sh/uv/concepts/build-backend/
---

# uv_build form 1a: explicit dotted `module-name` per member

The uv_build backend offers **three** alternative knobs for namespace packages. The bonk research synthesis from 2026-05-14 conflated them into one TOML block. Per current uv docs (verified 2026-05-15), they are mutually-exclusive forms.

## Use this — form 1a

```toml
# packages/<member>/pyproject.toml
[build-system]
requires = ["uv_build>=0.11.14,<0.12"]
build-backend = "uv_build"

[tool.uv.build-backend]
module-name = "claude_sql.core"
```

Project structure:

```
packages/core/
├── pyproject.toml
└── src/
    └── claude_sql/                 # NO __init__.py — namespace root
        └── core/
            └── __init__.py         # leaf init OK (and required)
```

uv docs: *"The `__init__.py` file is not included in `foo`, since it's the shared namespace module."* This is the canonical 2026 form for one member shipping one portion under a shared namespace.

## Don't use these alternative forms

**Form 1b** — `module-name = ["foo", "bar"]` ships multiple roots in one wheel. uv docs explicitly say *"we do not recommend this structure (i.e., you should use a workspace with multiple packages instead)"*.

**Form 1c** — `namespace = true` *without* `module-name` is the legacy/complex fallback that disables uv's safety checks. uv docs warn: *"Using `namespace = true` disables safety checks. Using an explicit list of module names is strongly recommended outside of legacy projects."*

When 1c is paired with `module-name`, the `module-name` value is the *single-segment shared root* (`module-name = "foo"`), not a dotted leaf. It's for one wheel that ships multiple sibling portions — orthogonal to the workspace case.

## `module-root` is not the namespace flag

`module-root` controls where uv looks for the module:
- omitted (default): `src/<module-name>/`
- `module-root = ""`: flat layout, module at project root
- `module-root = "src"` is *implicit* — uv docs don't show it as a documented value

**Drop `module-root = ""` from any namespace config.** The bonk synthesis paired it with `namespace = true`; both are wrong for src-layout members.

## Verification fingerprint

```bash
uv sync --all-packages   # virtual root WON'T install members without --all-packages
uv run python -c "
import claude_sql
assert type(claude_sql.__path__).__name__ == '_NamespacePath'
print(list(claude_sql.__path__))
"
```

If you see `list` instead of `_NamespacePath`, an `__init__.py` snuck into one of the `src/<namespace>/` dirs and collapsed the namespace into a regular package. Per [[ci-guard-no-namespace-root-init-py]] for the CI guard.

## See also

- [[uv-workspace-virtual-root-needs-all-packages]] — `uv sync` vs `uv sync --all-packages` gotcha
- [[import-linter-v2-contract-type-renames]] — companion lesson on the import-linter v2 type-name change
- [[parallel-agents-shared-file]] — sed-rewrite phase must be single-agent

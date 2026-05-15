---
name: uv-workspace-virtual-root-needs-all-packages
description: When the workspace root pyproject sets `package = false` (virtual project), `uv sync` resolves dev-deps but does NOT install workspace members. Use `uv sync --all-packages` to include them. Common gotcha — symptom is `ModuleNotFoundError` for every member.
metadata:
  type: api-pattern
  tags: [uv, uv-sync, workspace, virtual-project, all-packages, monorepo]
  ref: https://docs.astral.sh/uv/concepts/projects/sync/
---

# `uv sync --all-packages` is required for virtual-root workspaces

Every uv workspace where the root is a "virtual project" (`[tool.uv] package = false`) needs `--all-packages` on `uv sync`, otherwise members under `packages/*` are NOT installed into the venv.

## Failure shape

Workspace root pyproject.toml:

```toml
[project]
name = "claude-sql-workspace"
version = "1.0.1"
dependencies = []

[tool.uv]
package = false       # virtual project — root itself is not a package

[tool.uv.workspace]
members = ["packages/*"]

[tool.uv.sources]
claude-sql-core = { workspace = true }
# ...one entry per member
```

After `uv sync` (no flags):

```bash
$ uv pip list | grep claude
# (empty — no members installed)

$ uv run python -c "import claude_sql"
ModuleNotFoundError: No module named 'claude_sql'
```

## Fix

```bash
uv sync --all-packages --all-extras
```

After:

```bash
$ uv pip list | grep claude
claude-sql                     1.0.1       /efs/.../packages/app
claude-sql-analytics           1.0.1       /efs/.../packages/analytics
claude-sql-core                1.0.1       /efs/.../packages/core
claude-sql-evals               1.0.1       /efs/.../packages/evals
claude-sql-provenance          1.0.1       /efs/.../packages/provenance
```

## Why

`uv sync` without `--all-packages` syncs just the *current project* — for a virtual root, that's the dev-deps in `[dependency-groups] dev`, nothing else. `--all-packages` unions every member's deps + dev-deps into the workspace venv and installs each member editable.

## When you'll hit this

- First `uv sync` after migrating to a workspace structure with a virtual root.
- CI flows that rely on `uv sync` to seed the test environment.
- `mise run install` tasks that wrap `uv sync` — must pass `--all-packages`.

## Codify in mise.toml

```toml
[tasks.install]
description = "Install every workspace member + dev deps + git hooks"
depends = ["hooks:install"]
run = "uv sync --all-packages --all-extras"
```

The bonk research synthesis didn't surface this — it's a 2026 uv-workspace operational detail that lives in the [uv sync docs](https://docs.astral.sh/uv/concepts/projects/sync/) but doesn't appear in any of the synthesis-cited example repos (postmodern-mono uses a non-virtual root; polylith uses a custom build hook).

## See also

- [[uv-build-namespace-form-1a-explicit-module-name]] — the per-member pyproject shape
- [[import-linter-v2-contract-type-renames]] — DAG enforcement for the new layout

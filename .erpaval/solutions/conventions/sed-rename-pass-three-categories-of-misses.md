---
name: sed-rename-pass-three-categories-of-misses
description: A scripted sed pass to rewrite imports during a module rename catches the dominant `^from claude_sql.<X> import` form. Three categories slip past. Plan for each.
metadata:
  type: convention
  tags: [sed, refactor, rename, imports, ripgrep, monorepo, workspace]
---

# Three categories a `sed` import-rewrite pass will miss

When renaming `claude_sql.X` → `claude_sql.<cluster>.X` across hundreds of import sites, a line-anchored sed pass (`^from claude_sql\.<X> import` and `^import claude_sql\.<X>`) catches **the dominant form**. Three categories slip past and produce silent ImportErrors at test/runtime.

## Category 1: Line-mid lazy imports inside function bodies

```python
def setup_lance(con):
    from claude_sql import lance_store   # ← indented; line-anchor `^from` skips it
    lance_store.attach(con)
```

These are common in Python: lazy import to defer expensive deps (boto3, lance, umap) until first use. They appear in:
- CLI subcommand bodies (deferred work)
- Test bodies that need a fresh import after monkeypatching
- `if TYPE_CHECKING:` blocks (separate hazard, see Category 3)

**Find them:**

```bash
rg -n " from claude_sql import [a-zA-Z_]+" packages/ tests/
rg -n " from claude_sql\.[a-z_]+ import" packages/ tests/
```

**Fix:** drop the line-start anchor in your sed pattern, OR run a follow-up `ruff check --fix --select I001` after the main pass (ruff isort doesn't help here, but it does normalize the rewritten lines so they're at least visible to the next pass).

## Category 2: Multi-name `from claude_sql import (a, b, c)` blocks

```python
from claude_sql import (
    binding as _binding,           # provenance
    blind_handover as _blind_hand, # evals
    checkpointer,                  # core
    freeze as _freeze,             # evals
    skills_catalog as _sk,         # analytics
)
```

When the names span clusters, no single `from claude_sql.<cluster> import (...)` rewrite works. You must **split the block** by cluster, manually:

```python
from claude_sql.analytics import skills_catalog as _sk
from claude_sql.core import checkpointer
from claude_sql.evals import blind_handover as _blind_hand, freeze as _freeze
from claude_sql.provenance import binding as _binding
```

**Find them:**

```bash
rg "^from claude_sql import \(" tests/ packages/
```

**Fix:** hand-edit each occurrence using a Python heredoc (sed multi-line is fragile; Edit tool races autoformatters):

```bash
python3 <<'PY'
path = "packages/app/src/claude_sql/app/cli.py"
src = open(path).read()
old = """from claude_sql import (
    binding as _binding,
    ...
)"""
new = """from claude_sql.analytics import skills_catalog as _sk
from claude_sql.core import checkpointer
..."""
assert old in src, "block not found — check whitespace"
open(path, "w").write(src.replace(old, new))
PY
```

## Category 3: String-form module references

These are NEVER caught by an `import` line scan. They fail at runtime, not parse time:

```python
# pytest test_binding.py — checks sys.modules
assert "claude_sql.cli" in sys.modules

# pytest test_ingest.py — subprocess
subprocess.run([sys.executable, "-m", "claude_sql.cli", "ingest"])

# pytest monkeypatch — string path
monkeypatch.setattr("claude_sql.llm_shared.boto3.client", _fake)
```

**Find them:**

```bash
rg -n '"claude_sql\.[a-z_]+"' tests/ packages/
rg -n '"-m", "claude_sql\.[a-z_]+"' tests/ packages/
```

**Fix:** sed each string literal individually. There aren't many, but every miss is a load-bearing test failure.

## Excluded false-positive: docstring `:mod:` / `:func:` references

Sphinx-flavored cross-references in docstrings:

```python
"""
The schema descriptions in :mod:`claude_sql.schemas` carry label semantics.
"""
```

These are documentation-only and don't break runtime. **Don't rewrite them** — they may turn into broken Sphinx links later but aren't import errors today. If your project uses Sphinx, a follow-up `chore(docs): update :mod: refs` PR is the right shape.

## The discovery loop

After each sed pass, run:

```bash
# 1. Static check: collect-only, see what fails to import
uv run pytest --collect-only --no-header -q 2>&1 | grep -i "error\|import"

# 2. Type-check: ty surfaces unresolved-import diagnostics
uv run ty check 2>&1 | grep "unresolved-import"

# 3. Runtime check: import every cluster
uv run python -c "
import claude_sql.core.config
import claude_sql.analytics.embed_worker
import claude_sql.evals.judges
import claude_sql.provenance.binding
import claude_sql.app.cli
"
```

Each surface catches a different miss. ty catches Category 1+2 (lazy + multi-name). Pytest collect catches Category 3 (string-form via test discovery). Runtime import catches everything left.

## Why this matters

In claude-sql's workspace migration (2026-05-15, PR #60), the first sed pass left:
- 9 line-mid lazy imports (Category 1)
- 2 multi-line bare-import blocks across clusters (Category 2)
- 3 string-form references (Category 3)

The first pass produced 79 test failures; after the three categories were swept, **3 failures remained** — all from Category 3 (the only category sed cannot reach via line-anchor patterns alone). One pass each through Categories 1 and 2 dropped failures from 79 → 3; one targeted sed for Category 3 dropped them to 0.

## See also

- [[parallel-agents-shared-file]] — the rewrite phase MUST be single-agent (last-write-wins git race)
- [[uv-build-namespace-form-1a-explicit-module-name]] — the build-backend shape that motivates the rename
- [[import-linter-v2-contract-type-renames]] — DAG enforcement post-rename

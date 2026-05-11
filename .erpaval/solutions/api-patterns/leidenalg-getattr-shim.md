---
title: Use getattr() to call CPMVertexPartition.quality() under ty/Pyright
track: knowledge
category: api-patterns
module: src/claude_sql/community_worker.py
component: leidenalg.find_partition / CPMVertexPartition / VertexClustering
severity: info
tags: [leidenalg, igraph, type-checking, ty, pyright, api-shim]
applies_when:
  - calling `leidenalg.find_partition(...)` and treating the return as a `CPMVertexPartition`
  - the project uses ty or Pyright (or both) as static type checkers
  - leidenalg version 0.11.x and python-igraph 1.0.x
pattern: |
  `leidenalg.CPMVertexPartition` (and the other `*VertexPartition` classes)
  inherit from `igraph.VertexClustering` and DO carry runtime
  `.quality()`, `.membership`, `.resolution_parameter`, `.sizes()` accessors.
  But the shipped type stubs for both `leidenalg` and `igraph` 1.0 do NOT
  expose those attributes on the partition class hierarchy — ty + Pyright
  both flag `partition.quality()` as `unresolved-attribute`.

  The cheapest workaround is to type the wrapper as `object` and route
  attribute access through `getattr` with the `# noqa: B009` linting
  exception (B009 forbids `getattr(x, "literal")` in favor of `x.literal`):

  ```python
  def _run_leiden_cpm(g, *, gamma, seed, n_iterations) -> object:
      import leidenalg as la
      return la.find_partition(
          g, la.CPMVertexPartition, weights="weight",
          resolution_parameter=gamma, seed=seed, n_iterations=n_iterations,
      )

  partition = _run_leiden_cpm(g, gamma=γ, seed=42, n_iterations=-1)
  quality = float(getattr(partition, "quality")())     # noqa: B009
  membership = list(getattr(partition, "membership")) # noqa: B009
  n_communities = len(set(membership))                 # not len(partition);
                                                       # the stubs reject Sized too
  ```

  Tradeoff: cleaner IDE assistance is lost on these calls. Acceptable because
  the alternative (`# type: ignore[attr-defined]` or vendoring stubs) is more
  intrusive and brittle across leidenalg version bumps.
example_files:
  - src/claude_sql/community_worker.py
---

# Why this matters

If you write `partition.quality()` directly, `mise run typecheck` (and the
pre-commit `ty` hook) fail with

```
error[unresolved-attribute]: Object of type `VertexClustering` has no attribute `quality`
```

even though the call works at runtime. Because the project's pre-commit hook
runs ty across `src/`, the failure blocks every commit until you either fix
the access pattern or downgrade ty's strictness — and downgrading ty is the
wrong direction for a project that already enforces strict mode.

The `getattr` shim is local, reversible (delete the noqa once leidenalg
ships proper stubs), and keeps `mise run check` green.

# Example

```python
# Bad: type-check fails
quality = float(partition.quality())
n = len(partition)

# Good: shim out to runtime attribute access
quality = float(getattr(partition, "quality")())   # noqa: B009
membership = list(getattr(partition, "membership"))  # noqa: B009
n = len(set(membership))
```

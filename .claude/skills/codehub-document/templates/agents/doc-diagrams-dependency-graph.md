---
role: doc-diagrams-dependency-graph
model: sonnet
output: "{{ docs_root }}/diagrams/structural/dependency-graph.md"
depends_on:
  - "{{ context_path }}"
  - "{{ prefetch_path }}"
status: IN_PROGRESS
---

# Packet · {{ repo }} · diagrams/structural/dependency-graph.md

## 1. Objective

Produce `{{ docs_root }}/diagrams/structural/dependency-graph.md`: a single Mermaid `flowchart LR` showing `{{ repo }}`'s internal communities alongside external-dep leaf nodes, capped at 20 total nodes.

## 2. Scope

- Create: `{{ docs_root }}/diagrams/structural/dependency-graph.md`
- Do not touch: any other file under `{{ docs_root }}/`, any source file in the repo, `.context.md`, `.prefetch.md`, or any `.packets/*.md` other than this one.

## 3. Input specification

| Source | Read how | Cache state |
|---|---|---|
| Shared context | `Read {{ context_path }}` | always first |
| Prefetch ledger | `Read {{ prefetch_path }}` | always first |
| Top communities | `{{ context_path }} § Top communities` | cached |
| Internal edges | `{{ prefetch_path }} § sql relations` or `mcp__codehub__sql({query: "SELECT source, target, kind FROM relations WHERE kind IN ('CONTAINS','CALLS','IMPORTS') LIMIT 500"})` | cached if digest present; mid-run otherwise |
| External dependencies | `{{ context_path }} § Stack` or `mcp__codehub__dependencies({repo: "{{ repo }}"})` | cached if digest present; mid-run otherwise |

## 4. Process

1. `Read {{ context_path }}` and `Read {{ prefetch_path }}`. Confirm the internal community list and the external-dep list.
2. Pull the internal edge set from `.prefetch.md § sql relations`. If not cached, call `mcp__codehub__sql` and cache the digest in this packet's Work log.
3. Pull the external-dep list from `.context.md § Stack`. If absent, call `mcp__codehub__dependencies({repo: "{{ repo }}"})` and keep the top 15 by usage.
4. Compose the node set: internal communities (as `name[Label]` plain rectangles) plus the external deps (as `name[(Label)]:::external` parenthesized nodes with a dashed stroke class). Reserve the full 20-node budget; drop lowest-usage externals first when pruning.
5. Compose the edge set: internal→internal edges from the `sql` result collapsed to community level; internal→external edges from the dependency list, sourced at the internal community that imports the dep most often.
6. If node count > 20 after composition: keep the top-20 by edge count, then add a `## Legend (overflow)` table with columns `Node | Edges | Reason for elision`.
7. Draft the Mermaid `flowchart LR`. Include the `classDef external stroke-dasharray: 3 3` line so external leaves render dashed.
8. `Write {{ docs_root }}/diagrams/structural/dependency-graph.md` with H1 = `{{ repo }} · Dependency graph`.

## 5. Document format rules

- H1 = `{{ repo }} · Dependency graph`. No decorative titles.
- No YAML frontmatter on the output file.
- Exactly one Mermaid diagram, fenced with ` ```mermaid `, diagram type `flowchart LR`.
- Internal communities: plain rectangles — `name[Label]`.
- External deps: parenthesized shape with external class — `name[(Label)]:::external` + `classDef external stroke-dasharray: 3 3`.
- Max 20 nodes total; overflow goes into a Legend table below the fenced block.
- Node labels ≤ 20 chars; edge labels ≤ 15 chars.
- Every internal node must match a row in `{{ context_path }} § Top communities`; every external node must match a row from `dependencies` or `.context.md § Stack`.
- No emojis. No filler adverbs.

## 6. Tool usage guide

| Need | Tool | Why |
|---|---|---|
| Internal node list | `{{ context_path }} § Top communities` | precomputed; do not re-call `sql` for it |
| Internal edge set | `{{ prefetch_path }} § sql relations` | authoritative; filter to `CONTAINS`/`CALLS`/`IMPORTS` |
| External leaf nodes | `{{ context_path }} § Stack` or `mcp__codehub__dependencies` | do not grep manifests |
| Diagram idioms | `references/mermaid-patterns.md § Dependency graph` | canonical `flowchart LR` shape + external-node styling |

## 7. Fallback paths

- If `dependencies` errors and `.context.md § Stack` is missing: `Read` the root `package.json` / `Cargo.toml` / `pyproject.toml` and extract the top 15 deps by semantic weight. Append a `*manifest fallback*` note immediately after the fenced block and record the fallback in the Work log.
- If the internal edge set has < 10 `CONTAINS` edges after projection: drop to `CALLS` edges at file level, collapse by top-level folder, and label the diagram `Call graph (fallback)` in the H1.
- If `.context.md § Top communities` is empty: abort and record the gap in the Work log — do not synthesize internal nodes from bare filesystem layout.

## 8. Success criteria

- [ ] `{{ docs_root }}/diagrams/structural/dependency-graph.md` exists on disk.
- [ ] H1 line reads `# {{ repo }} · Dependency graph` (or the `Call graph (fallback)` variant if the fallback fired).
- [ ] Exactly one ` ```mermaid ` fence containing a `flowchart LR`.
- [ ] Diagram has 3-20 nodes, each with a label ≤ 20 chars.
- [ ] At least one internal community and one external dep appear in the diagram.
- [ ] External-dep nodes use the parenthesized shape and the `external` class.
- [ ] The `classDef external stroke-dasharray: 3 3` line is present.
- [ ] If overflow occurred, a `## Legend (overflow)` table lists ≥ 5 elided nodes with edge counts.
- [ ] No YAML frontmatter on the output.

## 9. Anti-goals

- Do not re-call `sql`, `dependencies`, or `project_profile` — those are cached in `.prefetch.md` / `.context.md`.
- Do not invent internal or external node names — every identifier must come from a tool response or a `Read` of the source file.
- Do not write YAML frontmatter on the output file.
- Do not emit more than one Mermaid diagram in this file.
- Do not exceed 20 nodes in the rendered diagram; overflow goes into the Legend table.
- Do not mix internal and external nodes under the same shape — internals are `[ ]`, externals are `[( )]:::external`.
- Do not emit emojis.

---

## Work log

{{ subagent fills this section per the write protocol }}

## Validation

{{ checks run, outputs pasted, any fixes applied }}

## Summary

{{ one paragraph — what shipped, where, why the internal/external node selection went the way it did }}

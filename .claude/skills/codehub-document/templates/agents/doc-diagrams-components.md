---
role: doc-diagrams-components
model: sonnet
output: "{{ docs_root }}/diagrams/architecture/components.md"
depends_on:
  - "{{ context_path }}"
  - "{{ prefetch_path }}"
status: IN_PROGRESS
---

# Packet · {{ repo }} · diagrams/architecture/components.md

## 1. Objective

Produce `{{ docs_root }}/diagrams/architecture/components.md`: a single Mermaid `classDiagram` of the top 8 components of `{{ repo }}` with HAS-A / USES edges, capped at 20 total nodes.

## 2. Scope

- Create: `{{ docs_root }}/diagrams/architecture/components.md`
- Do not touch: any other file under `{{ docs_root }}/`, any source file in the repo, `.context.md`, `.prefetch.md`, or any `.packets/*.md` other than this one.

## 3. Input specification

| Source | Read how | Cache state |
|---|---|---|
| Shared context | `Read {{ context_path }}` | always first |
| Prefetch ledger | `Read {{ prefetch_path }}` | always first |
| Top communities | `{{ context_path }} § Top communities` | cached |
| Community relations | `{{ prefetch_path }} § sql relations` or `mcp__codehub__sql({query: "SELECT source, target, kind FROM relations WHERE kind IN ('CONTAINS','CALLS','IMPORTS') LIMIT 500"})` | cached if digest present; mid-run otherwise |
| Component method list | `mcp__codehub__context({symbol: <community-name>})` per top 8 | mid-run |

## 4. Process

1. `Read {{ context_path }}` and `Read {{ prefetch_path }}`. Confirm top-8 community names and their file-path roots.
2. Pull the raw edge set from `.prefetch.md § sql relations` (kinds `CONTAINS`, `CALLS`, `IMPORTS`). If not cached, call `mcp__codehub__sql` with the query above and cache the digest in this packet's Work log.
3. Project file-level edges down to the community level: collapse every `(source_community, target_community, kind)` triple into a single edge, labeled with a one-word verb (`contains`, `invokes`, `imports`, `depends`).
4. For each of the top 8 communities, call `mcp__codehub__context({symbol: <community-name>})` and select the top 3-5 outbound method names by call count. These populate the `classDiagram` method list for that class.
5. If the projected graph has > 20 nodes, keep the top-20 by edge count and move the overflow into a `## Legend (overflow)` table with columns `Node | Edges | Reason for elision`.
6. Draft the Mermaid `classDiagram`. Max 8 classes; each class has 3-5 methods; relationships labeled with one-word verbs.
7. `Write {{ docs_root }}/diagrams/architecture/components.md` with H1 = `{{ repo }} · Component view`.

## 5. Document format rules

- H1 = `{{ repo }} · Component view`. No decorative titles.
- No YAML frontmatter on the output file.
- Exactly one Mermaid diagram, fenced with ` ```mermaid `, diagram type `classDiagram`.
- Max 20 nodes in the diagram; overflow goes into a Legend table below the fenced block.
- Node labels ≤ 20 chars; edge labels ≤ 15 chars.
- Every community / class name must match a row in `{{ context_path }} § Top communities` or a symbol returned by `context` — no invented identifiers.
- No emojis. No filler adverbs.

## 6. Tool usage guide

| Need | Tool | Why |
|---|---|---|
| Community list | `{{ context_path }} § Top communities` | precomputed; do not re-call `sql` |
| Edge set | `{{ prefetch_path }} § sql relations` | authoritative; filter to `CONTAINS`/`CALLS`/`IMPORTS` |
| Class method list | `mcp__codehub__context` | picks methods by call count |
| Diagram idioms | `references/mermaid-patterns.md § Component view` | canonical `classDiagram` shape + rules |

## 7. Fallback paths

- If the cached edge set has < 10 `CONTAINS` edges: fall back to `CALLS` edges at file level, collapsing by top-level folder, and label the diagram `Call graph (fallback)` in the H1. Note the fallback in the Work log.
- If a community's `context` call errors: keep the class in the diagram with an empty method list and append `*methods unavailable*` immediately after the fenced block.
- If `.context.md § Top communities` is empty: abort and record the gap in the Work log — do not emit a synthetic class list.

## 8. Success criteria

- [ ] `{{ docs_root }}/diagrams/architecture/components.md` exists on disk.
- [ ] H1 line reads `# {{ repo }} · Component view` (or `# {{ repo }} · Call graph (fallback)` if fallback fired).
- [ ] Exactly one ` ```mermaid ` fence containing a `classDiagram`.
- [ ] Diagram has 3-20 nodes, each with a label ≤ 20 chars.
- [ ] Max 8 classes carry method lists; each class has 3-5 methods.
- [ ] Every edge carries a one-word verb label.
- [ ] If overflow occurred, a `## Legend (overflow)` table lists ≥ 5 elided nodes with edge counts.
- [ ] No YAML frontmatter on the output.

## 9. Anti-goals

- Do not re-call `sql` or `project_profile` — those are cached in `.prefetch.md` / `.context.md`.
- Do not invent class names, method names, or edges — every identifier must come from a tool response or a `Read` of the source file.
- Do not write YAML frontmatter on the output file.
- Do not emit more than one Mermaid diagram in this file.
- Do not exceed 20 nodes in the rendered diagram; overflow goes into the Legend table, never back into the diagram.
- Do not emit emojis.

---

## Work log

{{ subagent fills this section per the write protocol }}

## Validation

{{ checks run, outputs pasted, any fixes applied }}

## Summary

{{ one paragraph — what shipped, where, why the component selection went the way it did }}

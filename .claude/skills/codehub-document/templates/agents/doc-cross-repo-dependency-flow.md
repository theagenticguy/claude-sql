---
role: doc-cross-repo-dependency-flow
model: sonnet
output: "{{ group_docs_root }}/cross-repo/dependency-flow.md"
depends_on:
  - "{{ group_context_path }}"
  - "{{ group_prefetch_path }}"
status: IN_PROGRESS
---

# Packet · {{ group }} · cross-repo/dependency-flow.md

> **Group mode only.** This packet runs when the skill orchestrator is invoked with `--group {{ group }}`. In single-repo mode it is never seeded.

## 1. Objective

Produce `{{ group_docs_root }}/cross-repo/dependency-flow.md`: a single Mermaid `flowchart TB` showing inter-repo data flow across the `{{ group }}` group — nodes are repos (plus events/streams as needed), edges are contract groups labeled by HTTP verb + path or event type.

## 2. Scope

- Create: `{{ group_docs_root }}/cross-repo/dependency-flow.md`
- Do not touch: any other file under `{{ group_docs_root }}/`, any file under a member repo (including `{{ member_repos }}/`), `{{ group_context_path }}`, `{{ group_prefetch_path }}`, or any `.packets/*.md` other than this one.

## 3. Input specification

| Source | Read how | Cache state |
|---|---|---|
| Group shared context | `Read {{ group_context_path }}` | always first |
| Group prefetch ledger | `Read {{ group_prefetch_path }}` | always first |
| Member list + freshness | `{{ group_context_path }} § Members` / `{{ group_prefetch_path }} § group_list,group_status` | cached |
| Group contracts | `{{ group_prefetch_path }} § group_contracts` | cached |
| Per-member route inventory | `{{ group_prefetch_path }} § route_map(<repo>)` per `{{ member_repos }}` | cached |
| Async / event topics | `mcp__codehub__group_query({group: "{{ group }}", text: "message consumer"})` or `"publishes"` | mid-run, only if stream edges are suspected |

## 4. Process

1. `Read {{ group_context_path }}` and `Read {{ group_prefetch_path }}`. Confirm the member list and re-verify every member is tagged `fresh` in the cached `group_status` digest. Abort to Work log if any member is stale.
2. Pull the contract inventory from `{{ group_prefetch_path }} § group_contracts`. For each `(producer_repo, consumer_repo)` pair, group the contracts by `method + path-prefix` to form a single edge labeled with the representative HTTP verb + path (e.g., `POST /invoices`).
3. Scan the contract list for event/stream shapes (kind = `topic` / `queue` / `stream` in the cached digest). For each stream, add a parenthesized node `name[(stream.name)]` and draw dashed edges (`-.->`) from consumers and solid edges (`-->`) from publishers.
4. Compose the node set: one node per member repo (`repo["repo"]` plain rectangle) plus one parenthesized node per event/stream. Cap at 20 nodes — if the group plus streams exceed 20, keep the top-20 by edge count and move overflow into a `## Legend (overflow)` table.
5. Compose the edge set: solid arrows for synchronous HTTP calls, dashed arrows for async/pub-sub. Each edge label carries the HTTP verb + path (routes) or the event type (streams); ≤ 15 chars.
6. Draft the Mermaid `flowchart TB`. Every repo node's string identifier and every path/event on every edge must trace back to `group_contracts` or a `route_map` digest — no invented flows.
7. `Write {{ group_docs_root }}/cross-repo/dependency-flow.md` with H1 = `{{ group }} · Dependency flow`.

## 5. Document format rules

- **Every citation MUST use the group-qualified `repo:path:LOC` form.** Phase E's regex depends on this — bare `path:LOC` will not be rewritten into cross-repo links. Citations appear in any accompanying narrative text, in per-edge footnotes (if used), and in the Legend table's "source" column.
- H1 = `{{ group }} · Dependency flow`. No decorative titles.
- No YAML frontmatter on the output file.
- Exactly one Mermaid diagram, fenced with ` ```mermaid `, diagram type `flowchart TB`.
- Repo nodes: plain rectangles — `repo["repo"]`.
- Event/stream nodes: parenthesized shape — `name[(stream.name)]`.
- Solid arrows (`-->`) for synchronous calls; dashed arrows (`-.->`) for async/pub-sub.
- Edge labels = HTTP verb + path for routes, event type for streams; ≤ 15 chars.
- Max 20 nodes total; overflow goes into a `## Legend (overflow)` table below the fenced block.
- No emojis. No filler adverbs.

## 6. Tool usage guide

| Need | Tool | Why |
|---|---|---|
| Member list + freshness | `{{ group_prefetch_path }} § group_list,group_status` | precondition gate; precomputed |
| Contract inventory | `{{ group_prefetch_path }} § group_contracts` | authoritative spine; source for every edge |
| Producer file resolution | `{{ group_prefetch_path }} § route_map(<repo>)` | maps contract `path` → handler `file:LOC` for citations |
| Async/stream discovery | `mcp__codehub__group_query` with `"message consumer"` / `"publishes"` | only if stream edges are suspected and not cached |
| Diagram idioms | `references/mermaid-patterns.md § Cross-repo dependency flow` | canonical `flowchart TB` shape + solid/dashed edge rules |

## 7. Fallback paths

- If `group_contracts` returned zero contracts: emit the diagram with isolated (edge-less) repo nodes and add a `> No inter-repo data flow detected in the group graph` banner immediately beneath the H1. Record the fallback in the Work log.
- If a member repo is stale despite Phase 0 checks: abort — write `{{ group_docs_root }}/cross-repo/_stale.md` instead, explaining which repo blocked generation, and stop.
- If a `route_map` digest is missing for a producer: derive the edge label from the raw `method + path` in `group_contracts` and cite the bullet's source as `<producer-repo>:<path>:1`. Record the fallback in the Work log.
- If `group_query` for `"message consumer"` / `"publishes"` returns nothing: assume the group has no stream edges for this pass and emit only HTTP edges.

## 8. Success criteria

- [ ] `{{ group_docs_root }}/cross-repo/dependency-flow.md` exists on disk.
- [ ] H1 line reads `# {{ group }} · Dependency flow`.
- [ ] Exactly one ` ```mermaid ` fence containing a `flowchart TB`.
- [ ] Diagram has 1-20 nodes; every repo node matches an entry in `group_list`.
- [ ] Every event/stream node uses the parenthesized `[( )]` shape.
- [ ] Every solid edge corresponds to a synchronous contract in `group_contracts`; every dashed edge corresponds to a stream / async contract.
- [ ] Every edge label ≤ 15 chars and matches a real HTTP verb+path or event type from the cached digest.
- [ ] Every citation in the file uses the `repo:path:LOC` form — no bare `path:LOC` (grep the output to verify).
- [ ] If overflow occurred, a `## Legend (overflow)` table lists ≥ 5 elided nodes with edge counts.
- [ ] No YAML frontmatter on the output.

## 9. Anti-goals

- Do not re-call any MCP tool whose digest is already in `{{ group_prefetch_path }}` — read the cached summary.
- Do not emit a citation in the bare `path:LOC` form; every citation MUST be `repo:path:LOC`.
- Do not invent repos, streams, routes, HTTP methods, or event types — every identifier must come from a cached tool response.
- Do not write YAML frontmatter on the output file.
- Do not emit more than one Mermaid diagram.
- Do not exceed 20 nodes in the rendered diagram; overflow goes into the Legend table.
- Do not use solid arrows for async/stream edges, or dashed arrows for synchronous HTTP calls.
- Do not emit emojis.

---

## Work log

{{ subagent fills this section per the write protocol }}

## Validation

{{ checks run, outputs pasted, any fixes applied }}

## Summary

{{ one paragraph — what shipped, where, why the sync/async edge partitioning went the way it did }}

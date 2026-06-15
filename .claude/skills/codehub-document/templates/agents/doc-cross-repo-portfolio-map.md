---
role: doc-cross-repo-portfolio-map
model: sonnet
output: "{{ group_docs_root }}/cross-repo/portfolio-map.md"
depends_on:
  - "{{ group_context_path }}"
  - "{{ group_prefetch_path }}"
status: IN_PROGRESS
---

# Packet · {{ group }} · cross-repo/portfolio-map.md

> **Group mode only.** This packet runs when the skill orchestrator is invoked with `--group {{ group }}`. In single-repo mode it is never seeded.

## 1. Objective

Produce `{{ group_docs_root }}/cross-repo/portfolio-map.md`: a 2-paragraph narrative of the `{{ group }}` group's shape, a Mermaid `flowchart LR` of its member repos with contract-count edges, and a `## Repos` section with per-member H2 + relative link into each member's own `.codehub/docs/` tree.

## 2. Scope

- Create: `{{ group_docs_root }}/cross-repo/portfolio-map.md`
- Do not touch: any other file under `{{ group_docs_root }}/`, any file under a member repo (including `{{ member_repos }}/.codehub/docs/`), `{{ group_context_path }}`, `{{ group_prefetch_path }}`, or any `.packets/*.md` other than this one.

## 3. Input specification

| Source | Read how | Cache state |
|---|---|---|
| Group shared context | `Read {{ group_context_path }}` | always first |
| Group prefetch ledger | `Read {{ group_prefetch_path }}` | always first |
| Member list + freshness | `{{ group_context_path }} § Members` / `{{ group_prefetch_path }} § group_list,group_status` | cached |
| Group contracts | `{{ group_prefetch_path }} § group_contracts` | cached |
| Per-member one-line description | `{{ group_context_path }} § Member profiles` | cached |
| Member docs-root paths | derived: `../{{ member_repos[i] }}/.codehub/docs/README.md` | n/a |

## 4. Process

1. `Read {{ group_context_path }}` and `Read {{ group_prefetch_path }}`. Confirm the member list under `§ Members` and re-verify that every member is tagged `fresh` in the cached `group_status` digest. If any member is stale, abort and write the gap to Work log — the orchestrator's Phase 0 gate should have caught this.
2. Pull the contract inventory from `{{ group_prefetch_path }} § group_contracts`. Aggregate by `(producer_repo, consumer_repo)` into an edge-count table.
3. Pull one-line domain descriptions for each member from `{{ group_context_path }} § Member profiles`. These become the second line of each node label via `<br/>`.
4. Draft the 2-paragraph narrative: paragraph 1 = what the group does as a whole (domain, scope); paragraph 2 = how the members relate (who produces for whom, which member is the hub). Every factual claim carries a `repo:path:LOC` citation.
5. Draft the Mermaid `flowchart LR`: one node per member repo with the two-line label `"<repo><br/><one-line domain>"`; one edge per `(producer, consumer)` pair labeled `contracts: N`. Cap at 20 nodes — if the group has > 20 members, keep the top-20 by inbound+outbound contract count and move the overflow to a `## Legend (overflow)` table.
6. Draft the `## Repos` section: one H2 per member (`## <repo> — <one-line description>`), each with 1-2 `repo:path:LOC` citations and a `[See {{ member_repos[i] }} docs →](../{{ member_repos[i] }}/.codehub/docs/README.md)` link.
7. `Write {{ group_docs_root }}/cross-repo/portfolio-map.md` with H1 = `{{ group }} · Portfolio map`.

## 5. Document format rules

- **Every citation MUST use the group-qualified `repo:path:LOC` form.** Phase E's regex depends on this — bare `path:LOC` will not be rewritten into cross-repo links.
- H1 = `{{ group }} · Portfolio map`. No decorative titles.
- No YAML frontmatter on the output file.
- Exactly one Mermaid diagram, fenced with ` ```mermaid `, diagram type `flowchart LR`.
- Node labels use the `"<repo><br/><one-line domain>"` two-line form; labels ≤ 20 chars per line.
- Edge labels `contracts: N` where N comes directly from the aggregated `group_contracts` count; ≤ 15 chars.
- Member-repo links use a relative path rooted at the group directory: `../<repo>/.codehub/docs/README.md`.
- `## Repos` section has exactly one H2 per member, in the order returned by `group_list`.
- No emojis. No filler adverbs.

## 6. Tool usage guide

| Need | Tool | Why |
|---|---|---|
| Member list + freshness | `{{ group_prefetch_path }} § group_list,group_status` | precondition gate; precomputed |
| Contract inventory | `{{ group_prefetch_path }} § group_contracts` | authoritative spine; do not re-call |
| Per-member profile | `{{ group_context_path }} § Member profiles` | cached; source for node labels + H2 text |
| Diagram idioms | `references/mermaid-patterns.md § Cross-repo portfolio` | canonical `flowchart LR` shape + edge labeling |

## 7. Fallback paths

- If `group_contracts` returned zero contracts: keep the diagram with isolated (edge-less) nodes; narrative paragraph 2 becomes a 1-sentence note that the group graph does not currently encode cross-repo edges. Record the fallback in the Work log.
- If any member is stale despite Phase 0 checks: abort — do not write `portfolio-map.md`. Instead, write `{{ group_docs_root }}/cross-repo/_stale.md` explaining which repo blocked generation, and stop.
- If `{{ group_context_path }} § Member profiles` lacks a one-liner for a member: call `mcp__codehub__project_profile({repo: <member>})` once, extract the summary, cache the digest in Work log, and use it. Do not invent a description.

## 8. Success criteria

- [ ] `{{ group_docs_root }}/cross-repo/portfolio-map.md` exists on disk.
- [ ] H1 line reads `# {{ group }} · Portfolio map`.
- [ ] Narrative is 2 paragraphs, every factual claim cited with `repo:path:LOC`.
- [ ] Exactly one ` ```mermaid ` fence containing a `flowchart LR` with 1-20 nodes.
- [ ] Every node label uses the two-line `"<repo><br/><one-line domain>"` form.
- [ ] `## Repos` section has one H2 per member; each H2 has at least one `repo:path:LOC` citation and a relative `../<repo>/.codehub/docs/README.md` link.
- [ ] Every citation in the file uses the `repo:path:LOC` form — no bare `path:LOC` (grep the output to verify).
- [ ] No YAML frontmatter on the output.

## 9. Anti-goals

- Do not re-call any MCP tool whose digest is already in `{{ group_prefetch_path }}` — read the cached summary.
- Do not emit a citation in the bare `path:LOC` form; every citation MUST be `repo:path:LOC`.
- Do not invent member repos, contract counts, or domain descriptions — every identifier must come from a cached tool response or `Read` of the source file.
- Do not write YAML frontmatter on the output file.
- Do not emit more than one Mermaid diagram.
- Do not exceed 20 nodes in the rendered diagram; overflow goes into the Legend table.
- Do not link to a member's docs with an absolute path — relative `../<repo>/.codehub/docs/...` only.
- Do not emit emojis.

---

## Work log

{{ subagent fills this section per the write protocol }}

## Validation

{{ checks run, outputs pasted, any fixes applied }}

## Summary

{{ one paragraph — what shipped, where, why the member ordering and narrative shape went the way they did }}

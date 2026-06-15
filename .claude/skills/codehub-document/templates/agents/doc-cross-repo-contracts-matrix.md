---
role: doc-cross-repo-contracts-matrix
model: sonnet
output: "{{ group_docs_root }}/cross-repo/contracts-matrix.md"
depends_on:
  - "{{ group_context_path }}"
  - "{{ group_prefetch_path }}"
status: IN_PROGRESS
---

# Packet · {{ group }} · cross-repo/contracts-matrix.md

> **Group mode only.** This packet runs when the skill orchestrator is invoked with `--group {{ group }}`. In single-repo mode it is never seeded.

## 1. Objective

Produce `{{ group_docs_root }}/cross-repo/contracts-matrix.md`: the N×N producer/consumer matrix for the `{{ group }}` group (rows = producers, columns = consumers, cells = contract counts), followed by a `## Notable contracts` H2 listing the top 10 contracts with both-ends `repo:path:LOC` citations.

## 2. Scope

- Create: `{{ group_docs_root }}/cross-repo/contracts-matrix.md`
- Do not touch: any other file under `{{ group_docs_root }}/`, any file under a member repo (including `{{ member_repos }}/`), `{{ group_context_path }}`, `{{ group_prefetch_path }}`, or any `.packets/*.md` other than this one.

## 3. Input specification

| Source | Read how | Cache state |
|---|---|---|
| Group shared context | `Read {{ group_context_path }}` | always first |
| Group prefetch ledger | `Read {{ group_prefetch_path }}` | always first |
| Member list + freshness | `{{ group_context_path }} § Members` / `{{ group_prefetch_path }} § group_list,group_status` | cached |
| Group contracts (the spine) | `{{ group_prefetch_path }} § group_contracts` | cached |
| Per-member route inventory | `{{ group_prefetch_path }} § route_map(<repo>)` per `{{ member_repos }}` | cached |
| Concept → symbol disambiguation | `mcp__codehub__group_query({group: "{{ group }}", text: <concept>})` | mid-run, on demand |

## 4. Process

1. `Read {{ group_context_path }}` and `Read {{ group_prefetch_path }}`. Confirm the member list and re-verify every member is tagged `fresh` in the cached `group_status` digest. Abort to Work log if any member is stale.
2. Pull the contract list from `{{ group_prefetch_path }} § group_contracts`. Each row has `{producer_repo, consumer_repo, path, method, shape}`. This is the spine of the entire artifact.
3. Build the producer/consumer matrix: rows = producer repos (in `group_list` order), columns = consumer repos (same order). Each cell = the contract count for that `(producer, consumer)` pair. Diagonal cells show `—`. Render as a Markdown table.
4. Select the top 10 contracts by consumer count (ties broken by producer path lexicographically). For each, resolve both ends to real files via the cached `route_map` digests — producer file comes from `route_map(<producer>)`, consumer file from the `group_contracts` row's `consumer_repo` + `path`. Cite both with `repo:path:LOC`.
5. Draft the `## Notable contracts` H2: a bullet list of the 10 entries in the form `` `<producer-repo>:<producer-path>:<LOC>` ← consumed by `<consumer-repo>:<consumer-path>:<LOC>` (method + shape summary) ``.
6. If the matrix has zero non-diagonal contracts, still emit the full N×N matrix (all `0` / `—` cells), add a `> No inter-repo contracts detected — the group graph does not currently encode cross-repo edges` banner immediately beneath the H1, and replace `## Notable contracts` with `## Notable contracts\n\n_None detected._`.
7. `Write {{ group_docs_root }}/cross-repo/contracts-matrix.md` with H1 = `{{ group }} · Contracts matrix`.

## 5. Document format rules

- **Every citation MUST use the group-qualified `repo:path:LOC` form — on both ends of every Notable contracts bullet.** Phase E's regex depends on this — bare `path:LOC` will not be rewritten.
- H1 = `{{ group }} · Contracts matrix`. No decorative titles.
- No YAML frontmatter on the output file.
- Matrix is a full Markdown table, N×N, even when most cells are zero. Column and row order match `group_list`. Diagonal = `—`; empty cells = `0`.
- `## Notable contracts` section has at most 10 bullets, each with two `repo:path:LOC` citations (producer end and consumer end) plus a short `(method + shape)` summary.
- No emojis. No filler adverbs.

## 6. Tool usage guide

| Need | Tool | Why |
|---|---|---|
| Member list + freshness | `{{ group_prefetch_path }} § group_list,group_status` | precondition gate; precomputed |
| Contract inventory | `{{ group_prefetch_path }} § group_contracts` | authoritative spine; do not re-call |
| Producer file resolution | `{{ group_prefetch_path }} § route_map(<repo>)` | maps contract `path` → handler `file:LOC` |
| Concept disambiguation | `mcp__codehub__group_query` | when a contract's target symbol has multiple matches across the group |

## 7. Fallback paths

- If `group_contracts` returned zero contracts: follow the all-zero path in step 6. Do not omit the matrix.
- If a member repo is stale despite Phase 0 checks: abort — write `{{ group_docs_root }}/cross-repo/_stale.md` instead, explaining which repo blocked generation, and stop.
- If a `route_map` digest is missing for a producer: fall back to the raw `path` string from `group_contracts` and cite `<producer-repo>:<path>:1`. Record the fallback in the Work log.
- If `group_query` returns empty for a concept while disambiguating: try `"http route"`, `"mcp tool"`, `"message consumer"` in order before giving up and marking the bullet `*consumer unresolved*`.

## 8. Success criteria

- [ ] `{{ group_docs_root }}/cross-repo/contracts-matrix.md` exists on disk.
- [ ] H1 line reads `# {{ group }} · Contracts matrix`.
- [ ] Matrix is a full N×N Markdown table where N = number of members; diagonals are `—`; cells show integer contract counts.
- [ ] `## Notable contracts` H2 is present; bullet count = `min(10, total non-diagonal contract count)`.
- [ ] Every Notable contracts bullet has two `repo:path:LOC` citations (producer + consumer).
- [ ] Every citation in the file uses the `repo:path:LOC` form — no bare `path:LOC` (grep the output to verify).
- [ ] If zero contracts: the empty-state banner and `_None detected._` sub-section are present.
- [ ] No YAML frontmatter on the output.

## 9. Anti-goals

- Do not re-call any MCP tool whose digest is already in `{{ group_prefetch_path }}` — read the cached summary.
- Do not emit a citation in the bare `path:LOC` form; every citation MUST be `repo:path:LOC`.
- Do not invent contracts, producer/consumer pairs, file paths, or line numbers — every entry must come from a cached tool response.
- Do not write YAML frontmatter on the output file.
- Do not omit the matrix when contract count is zero — emit an empty N×N table with the banner.
- Do not exceed 10 bullets under `## Notable contracts`.
- Do not collapse the matrix to producer-only or consumer-only — it must be N×N.
- Do not emit emojis.

---

## Work log

{{ subagent fills this section per the write protocol }}

## Validation

{{ checks run, outputs pasted, any fixes applied }}

## Summary

{{ one paragraph — what shipped, where, why the top-10 contract selection went the way it did }}

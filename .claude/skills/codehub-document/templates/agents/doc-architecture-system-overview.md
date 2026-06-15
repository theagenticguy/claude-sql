---
role: doc-architecture-system-overview
model: sonnet
output: "{{ docs_root }}/architecture/system-overview.md"
depends_on:
  - "{{ context_path }}"
  - "{{ prefetch_path }}"
status: IN_PROGRESS
---

# Packet · {{ repo }} · architecture/system-overview.md

## 1. Objective

Produce `{{ docs_root }}/architecture/system-overview.md`: a 400–600-word narrative of what `{{ repo }}` does and how its top-level pieces fit, with one stack table and one Mermaid `flowchart LR` of the top 6 modules.

## 2. Scope

- Create: `{{ docs_root }}/architecture/system-overview.md`
- Do not touch: any other file under `{{ docs_root }}/`, any source file in the repo, `.context.md`, `.prefetch.md`, or any `.packets/*.md` other than this one.

## 3. Input specification

| Source | Read how | Cache state |
|---|---|---|
| Shared context | `Read {{ context_path }}` | always first |
| Prefetch ledger | `Read {{ prefetch_path }}` | always first |
| Project profile | `{{ context_path }} § Repo profile` | cached |
| Top communities | `{{ context_path }} § Top communities` | cached |
| Top processes | `{{ context_path }} § Top processes` | cached |
| External deps | `{{ context_path }} § Stack` or `mcp__codehub__dependencies({repo: "{{ repo }}"})` | cached if digest present |
| Module relations | `mcp__codehub__context({symbol: <community-name>})` per top 6 modules | mid-run |

## 4. Process

1. `Read {{ context_path }}` and `Read {{ prefetch_path }}`. Confirm `{{ repo }}` profile, top-6 community names, top processes.
2. For each of the top 6 communities (from `.context.md § Top communities`): call `mcp__codehub__context({symbol: <community-name>})` to pull inbound/outbound relation counts. Cache the summary in this packet's Work log.
3. Pull the external dependencies list from `.context.md § Stack` (or call `dependencies` if not cached). Keep the top 15 by usage.
4. Resolve the stack layers by inspecting `project_profile.entry_points` and cross-referencing the top communities with their `file_path` prefixes.
5. Draft the narrative (400–600 words). Structure: paragraph 1 = what the repo does, paragraph 2 = how the pieces fit.
6. Draft the Stack table with columns `Layer | Technology | Source`. Every row cites `path:LOC`.
7. Draft the Module-map Mermaid `flowchart LR` — ≤ 20 nodes, short labels (≤ 20 chars), edges from relation counts ≥ 1.
8. `Write {{ docs_root }}/architecture/system-overview.md` with H1 = `{{ repo }} · System overview`.

## 5. Document format rules

- H1 = `{{ repo }} · System overview`. No decorative titles.
- No YAML frontmatter on the output file.
- Every factual claim has a backtick `path:LOC` citation; file-level cites append ` (N LOC)`.
- Mermaid fenced with ` ```mermaid `. Exactly one diagram.
- No emojis. No filler adverbs.
- Stack table has 3 columns exactly: `Layer | Technology | Source`.

## 6. Tool usage guide

| Need | Tool | Why |
|---|---|---|
| Module list + cohesion | `{{ context_path }} § Top communities` | precomputed; do not re-call `sql` |
| Symbol neighborhood | `mcp__codehub__context` | inbound/outbound + cochange counts |
| External dependency list | `{{ context_path }} § Stack` or `dependencies` | authoritative over grepping manifests |
| File line count for `(N LOC)` | `Read` then line count | graph does not store LOC |

## 7. Fallback paths

- If `.context.md § Top communities` is empty: fall back to `sql({query: "SELECT name, file_path FROM nodes WHERE kind='File' ORDER BY file_path LIMIT 200"})` and group by top-level folder. Cite the fallback in the Work log.
- If a community's `context` call errors: list the module in the Mermaid diagram but omit its relation counts from the narrative; mark the narrative line `*context unavailable*`.
- If `dependencies` is unavailable and `.context.md` lacks a Stack section: `Read` the root `package.json` / `Cargo.toml` / `pyproject.toml`, extract the top 15 deps by semantic weight. Mark the Source column `*manifest fallback*`.

## 8. Success criteria

- [ ] `{{ docs_root }}/architecture/system-overview.md` exists on disk.
- [ ] H1 line reads `# {{ repo }} · System overview`.
- [ ] Narrative is 400–600 words (use `wc -w` to verify).
- [ ] Stack table has ≥ 5 rows, every row cites `path:LOC`.
- [ ] Exactly one Mermaid ` ```mermaid ` fence containing a `flowchart LR`.
- [ ] Mermaid diagram has 3–20 nodes with labels ≤ 20 chars.
- [ ] No YAML frontmatter on the output.
- [ ] No `path:LOC` citation references a file that does not exist (spot-check 3).

## 9. Anti-goals

- Do not re-call `project_profile` or `sql` over communities — those are cached in `.context.md` / `.prefetch.md`.
- Do not invent community or module names — every node in the Mermaid diagram must map to a row in `.context.md § Top communities`.
- Do not write YAML frontmatter on the output file.
- Do not emit more than one Mermaid diagram.
- Do not exceed 600 words in the narrative; if analysis is longer, move extra detail to `module-map.md` (which another packet produces).

---

## Work log

{{ subagent fills this section per the write protocol }}

## Validation

{{ checks run, outputs pasted, any fixes applied }}

## Summary

{{ one paragraph — what shipped, where, why the module selection went the way it did }}

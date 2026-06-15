---
role: doc-architecture-module-map
model: sonnet
output: "{{ docs_root }}/architecture/module-map.md"
depends_on:
  - "{{ context_path }}"
  - "{{ prefetch_path }}"
status: IN_PROGRESS
---

# Packet · {{ repo }} · architecture/module-map.md

## 1. Objective

Produce `{{ docs_root }}/architecture/module-map.md`: one H2 per top module, each with a one-paragraph description and a bullet list of the module's top 8 files cited as `` `path:LOC` `` with `(N LOC)` suffix.

## 2. Scope

- Create: `{{ docs_root }}/architecture/module-map.md`
- Do not touch: any other file under `{{ docs_root }}/`, any source file in the repo, `.context.md`, `.prefetch.md`, or any `.packets/*.md` other than this one.

## 3. Input specification

| Source | Read how | Cache state |
|---|---|---|
| Shared context | `Read {{ context_path }}` | always first |
| Prefetch ledger | `Read {{ prefetch_path }}` | always first |
| Top communities | `{{ context_path }} § Top communities` | cached |
| Community member files | `{{ prefetch_path }} § community members` or `mcp__codehub__sql({query: "SELECT name, file_path, start_line FROM nodes WHERE kind='File' AND community_id = <id> ORDER BY file_path"})` | cached if digest present |
| Community relations | `mcp__codehub__context({symbol: <community-name>})` per module | mid-run (only if cache miss) |
| File LOC | `Read <file>` then count lines | mid-run |

## 4. Process

1. `Read {{ context_path }}` and `Read {{ prefetch_path }}`. Confirm `{{ repo }}` profile and the ordered list of top communities.
2. For each of the top 8 communities, pull the member file list from `.prefetch.md § community members`. If absent, call `sql({query: "SELECT name, file_path, start_line FROM nodes WHERE kind='File' AND community_id = <id> ORDER BY file_path LIMIT 50"})` and cache the digest in this packet's Work log.
3. For each module, rank files by semantic weight (inbound relation count from cached `context` digest if present, else alphabetical fallback). Keep the top 8.
4. For every file in the shortlist, `Read` it to compute LOC. Record `path:LOC` + total line count per file in the packet Work log.
5. Draft the one-paragraph description per module. Anchor the paragraph on the module's `inferred_label` and its highest-weight file, each cited `` `path:LOC` ``.
6. Collapse any module with fewer than 3 files into a trailing `## Supporting code` H2 that lists those files as flat bullets.
7. `Write {{ docs_root }}/architecture/module-map.md` with H1 = `{{ repo }} · Module map` and one H2 per module in top-community order.

## 5. Document format rules

- H1 = `{{ repo }} · Module map`. No decorative titles.
- No YAML frontmatter on the output file.
- One H2 per module. The H2 text is the community `inferred_label` (or its canonical name if no label).
- Each module H2 is followed by: one paragraph (≤ 4 sentences) + a bullet list of the top 8 files.
- Every bullet is a `` `path` `` reference with a trailing `(N LOC)`; every paragraph claim has a `` `path:LOC` `` citation.
- A trailing `## Supporting code` H2 collects modules with < 3 files as flat bullets.
- No Mermaid. No emojis. No filler adverbs.

## 6. Tool usage guide

| Need | Tool | Why |
|---|---|---|
| Module list | `{{ context_path }} § Top communities` | precomputed; do not re-call `sql` |
| Community member files | `{{ prefetch_path }}` or `sql` over `nodes WHERE kind='File' AND community_id=...` | authoritative member set |
| Module relation counts | `mcp__codehub__context` | already cached for top-6 by system-overview packet; reuse digest |
| File LOC | `Read` then line count | graph does not store LOC |

## 7. Fallback paths

- If `.context.md § Top communities` is empty: fall back to `sql({query: "SELECT file_path FROM nodes WHERE kind='File' ORDER BY file_path LIMIT 500"})`, group by top-level folder, and treat each folder as a synthetic module. Note the fallback in the Work log.
- If a community's member list is empty: list the community under `## Supporting code` with an `*unresolved*` marker and cite the community node's `file_path` prefix only.
- If `Read` fails for a file (missing or binary): drop it from the bullet list and log the skip in the Work log. Do not invent a LOC count.
- If more than 12 communities are seeded: keep the top 8 as full H2 sections and append the remainder as one-line rows under `## Supporting code`.

## 8. Success criteria

- [ ] `{{ docs_root }}/architecture/module-map.md` exists on disk.
- [ ] H1 line reads `# {{ repo }} · Module map`.
- [ ] At least 3 H2 sections are present (count `^## ` matches, excluding `## Supporting code`).
- [ ] Every H2 is followed by a paragraph with at least one `` `path:LOC` `` citation.
- [ ] Every file bullet has the form `` `path` (N LOC) `` — verify with a grep for `(\d+ LOC)` on every bullet line.
- [ ] No file bullet references a path that does not exist on disk (spot-check 3).
- [ ] No module lists more than 8 bulleted files (supporting-code dump excluded).
- [ ] No YAML frontmatter on the output.
- [ ] No Mermaid fences in the output.

## 9. Anti-goals

- Do not re-call any MCP tool whose digest is in `.prefetch.md` — read the cached summary. If the cached slice is truncated, call with a narrower filter, not a blanket re-fetch.
- Do not invent module names, file paths, or LOC counts — every identifier must come from a tool response or a `Read` of the source file.
- Do not write YAML frontmatter on the output file.
- Do not emit Mermaid diagrams in this file — the flowchart belongs to `architecture/system-overview.md`.
- Do not emit emojis. Do not use filler adverbs.
- Do not exceed 8 file bullets per module; overflow belongs in `## Supporting code`.

---

## Work log

{{ subagent fills this section per the write protocol }}

## Validation

{{ checks run, outputs pasted, any fixes applied }}

## Summary

{{ one paragraph — what shipped, where, why the module ordering went the way it did }}

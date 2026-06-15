---
role: doc-analysis-ownership
model: sonnet
output: "{{ docs_root }}/analysis/ownership.md"
depends_on:
  - "{{ context_path }}"
  - "{{ prefetch_path }}"
status: IN_PROGRESS
---

# Packet · {{ repo }} · analysis/ownership.md

## 1. Objective

Produce `{{ docs_root }}/analysis/ownership.md`: a ranked table of folders in `{{ repo }}` by top-contributor share, followed by a `## Single points of failure` H2 listing paths where the top owner holds > 70% of commits — each SPOF row gets a one-sentence mitigation suggestion.

## 2. Scope

- Create: `{{ docs_root }}/analysis/ownership.md`
- Do not touch: `{{ docs_root }}/analysis/risk-hotspots.md`, `{{ docs_root }}/analysis/dead-code.md`, any other file under `{{ docs_root }}/`, any source file in the repo, `.context.md`, `.prefetch.md`, or any `.packets/*.md` other than this one.

## 3. Input specification

| Source | Read how | Cache state |
|---|---|---|
| Shared context | `Read {{ context_path }}` | always first |
| Prefetch ledger | `Read {{ prefetch_path }}` | always first |
| Owners summary | `{{ context_path }} § Owners summary` | cached |
| Owners per folder | `{{ prefetch_path }} § owners` or `mcp__codehub__owners({path: <folder>})` | cached when present |
| Folder roster | `mcp__codehub__sql({query: "SELECT DISTINCT file_path FROM nodes WHERE kind='File'"})` + folder-prefix grouping | mid-run, only if `.context.md` roster is truncated |
| Top-contributor share | `owners` response field `share` | cached or mid-run |

## 4. Process

1. `Read {{ context_path }}` and `Read {{ prefetch_path }}`. Confirm Owners summary and cached owners digest.
2. Build the folder roster: use `.context.md § Owners summary` when it enumerates folders; otherwise group `File` nodes by top-level folder prefix (`packages/<x>/src`, `src/<module>`, etc.).
3. For each folder (top 15 by file count): resolve top owner and share from `{{ prefetch_path }} § owners` when present, else call `mcp__codehub__owners({path: <folder>})`. Capture top owner, share %, total contributors.
4. Rank folders by top-owner share, descending. Draft the ranking table with columns `Folder | Top owner | Share | Total contributors`. Every Folder cell is a backtick path.
5. Identify Single Points of Failure: folders or files where the top owner's share is > 70%. For each SPOF: draft a bullet under the `## Single points of failure` H2 stating the path, the owner's share percentage, and a one-sentence mitigation (pair reviewer, knowledge-transfer session, cross-training target, etc.).
6. Draft the intro (1-2 paragraphs): what "share" means (commit share, not line share), the window, and what a > 70% share signals for bus factor.
7. `Write {{ docs_root }}/analysis/ownership.md` with H1 = `{{ repo }} · Ownership`.

## 5. Document format rules

- H1 = `{{ repo }} · Ownership`. No decorative titles.
- No YAML frontmatter on the output file.
- Ranking table has exactly these columns: `Folder | Top owner | Share | Total contributors`.
- Every folder cell is a backtick `path`. Share is a whole-percent integer (e.g., `72%`).
- `## Single points of failure` exists as an H2 (not H3, not subsection).
- Each SPOF bullet follows the shape `` - `path` — owner (N%). <one-sentence mitigation>. ``.
- No emojis. No filler adverbs.

## 6. Tool usage guide

| Need | Tool | Why |
|---|---|---|
| Contributor share per folder | `mcp__codehub__owners` | authoritative over git-blame |
| Cached owners digest | `{{ prefetch_path }} § owners` | precomputed; do not re-call for cached paths |
| Folder roster | `{{ context_path }} § Owners summary` | precomputed |
| Folder roster (fallback) | `mcp__codehub__sql` over `File` nodes | only when `.context.md` slice is truncated |

## 7. Fallback paths

- If `mcp__codehub__owners` fails for a path: cite the top-3 authors from a manual git-log walk and mark the row `*git-log fallback*` in the Top owner cell.
- If `.context.md § Owners summary` is absent: use the `sql` folder roster and call `owners` per folder. Cite the fallback in the Work log.
- If no folder crosses the > 70% threshold: still emit the `## Single points of failure` H2 with a single line `No folders exceed the 70% bus-factor threshold.` — the H2 must exist for Phase E cross-references.
- If owners data is completely missing (new repo, shallow clone): write the gap to the Work log, skip the Write step, and do not emit an empty file with a single-row table.

## 8. Success criteria

- [ ] `{{ docs_root }}/analysis/ownership.md` exists on disk.
- [ ] H1 line reads `# {{ repo }} · Ownership`.
- [ ] The ranking table has at least 5 rows.
- [ ] Every folder row has a backtick `path` in the Folder column.
- [ ] Every Share cell is a whole-percent integer.
- [ ] A single `## Single points of failure` H2 exists (always present, even when empty).
- [ ] Every SPOF bullet includes a one-sentence mitigation suggestion.
- [ ] No SPOF lists an owner whose share ≤ 70% (verify each bullet).
- [ ] No YAML frontmatter on the output.

## 9. Anti-goals

- Do not re-call `owners` for paths already cached in `.prefetch.md`. Read the cached slice first.
- Do not invent contributor names or percentage shares — every name/number must come from `owners` output or the git-log fallback.
- Do not drop the `## Single points of failure` H2 even when no folder crosses 70% — emit the empty-state line instead.
- Do not write YAML frontmatter on the output file.
- Do not emit emojis.

---

## Work log

{{ subagent fills this section per the write protocol }}

## Validation

{{ checks run, outputs pasted, any fixes applied }}

## Summary

{{ one paragraph — what shipped, how many SPOFs surfaced, and any folders that required a git-log fallback }}

---
role: doc-analysis-dead-code
model: sonnet
output: "{{ docs_root }}/analysis/dead-code.md"
depends_on:
  - "{{ context_path }}"
  - "{{ prefetch_path }}"
status: IN_PROGRESS
---

# Packet · {{ repo }} · analysis/dead-code.md

## 1. Objective

Produce `{{ docs_root }}/analysis/dead-code.md`: three tables enumerating unreferenced exports, unreferenced files, and dead imports in `{{ repo }}`. The file is always emitted — when the graph reports no dead code, emit a `No unreferenced symbols detected.` banner plus a timestamp so Phase E cross-references have a stable target.

## 2. Scope

- Create: `{{ docs_root }}/analysis/dead-code.md`
- Do not touch: `{{ docs_root }}/analysis/risk-hotspots.md`, `{{ docs_root }}/analysis/ownership.md`, any other file under `{{ docs_root }}/`, any source file in the repo, `.context.md`, `.prefetch.md`, or any `.packets/*.md` other than this one.

## 3. Input specification

| Source | Read how | Cache state |
|---|---|---|
| Shared context | `Read {{ context_path }}` | always first |
| Prefetch ledger | `Read {{ prefetch_path }}` | always first |
| Dead-code inventory | `mcp__codehub__list_dead_code({repo: "{{ repo }}"})` | mid-run (rarely cached; the tool is cheap) |
| Last-modified per path | from `list_dead_code` response fields when present, else `git log -1 --format=%cs -- <path>` via shell | mid-run |
| Graph hash | `{{ graph_hash }}` for the empty-state banner timestamp anchor | cached |

## 4. Process

1. `Read {{ context_path }}` and `Read {{ prefetch_path }}`. Confirm graph hash and any cached dead-code digest.
2. Call `mcp__codehub__list_dead_code({repo: "{{ repo }}"})`. Partition the response into three buckets: `Unreferenced exports`, `Unreferenced files`, `Dead imports`.
3. If all three buckets are empty: skip to step 7 (empty-state banner).
4. For `Unreferenced exports`: draft a table with columns `Symbol | Path | Last modified`. Path is a backtick `path:LOC`. Last modified is an ISO date.
5. For `Unreferenced files`: draft a table with columns `File | Lines | Last modified`. File is a backtick `path`. Lines is an integer LOC count.
6. For `Dead imports`: draft a table with columns `Path | Symbol | Imported from`. Path is a backtick `path:LOC` at the import site; Imported from is a backtick `path` (module/package).
7. Empty-state handling: if `list_dead_code` returns no rows in any bucket, emit the banner `No unreferenced symbols detected.` on its own line, followed by `Graph hash: {{ graph_hash }}` and a UTC timestamp line. Still emit the three H2 headings (`## Unreferenced exports`, `## Unreferenced files`, `## Dead imports`) each with a single `_none_` line under them, so Phase E cross-references resolve.
8. `Write {{ docs_root }}/analysis/dead-code.md` with H1 = `{{ repo }} · Dead code`.

## 5. Document format rules

- H1 = `{{ repo }} · Dead code`. No decorative titles.
- No YAML frontmatter on the output file.
- Three H2s in this fixed order: `## Unreferenced exports`, `## Unreferenced files`, `## Dead imports`.
- `Unreferenced exports` table columns: `Symbol | Path | Last modified`.
- `Unreferenced files` table columns: `File | Lines | Last modified`.
- `Dead imports` table columns: `Path | Symbol | Imported from`.
- Every Path / File / Imported from cell is a backtick `path` or `path:LOC`.
- Empty-state banner is a single line `No unreferenced symbols detected.` followed by `Graph hash: {{ graph_hash }}` and a UTC timestamp.
- Under the empty state, each of the three H2 sections contains a single `_none_` line.
- No emojis. No filler adverbs.

## 6. Tool usage guide

| Need | Tool | Why |
|---|---|---|
| Unreferenced symbols + files | `mcp__codehub__list_dead_code` | graph-aware; deletes are safe |
| Last-modified per path | `list_dead_code` fields, else `git log -1` shell | graph does not always carry mtime |
| LOC per unreferenced file | `Read` then line count | graph stores node spans, not file LOC |
| Cross-check at import site | `Read` at `path:LOC` | verify import survives before listing as dead |

## 7. Fallback paths

- If `list_dead_code` errors or times out: write the failure to the Work log, retry once, and on a second failure emit the empty-state banner with an added line `*list_dead_code unavailable — re-run after codehub analyze*`. The file must still exist.
- If `list_dead_code` returns an `Unreferenced exports` row whose symbol is actually re-exported from a barrel (false positive): `Read` the barrel and drop the row, citing the removal in the Work log.
- If a `Last modified` field is missing from the response: run `git log -1 --format=%cs -- <path>` via the shell; if that also fails, use `—` in the cell and note in the Work log.
- If all three buckets are empty, always follow the step-7 empty-state flow; do not skip the Write step.

## 8. Success criteria

- [ ] `{{ docs_root }}/analysis/dead-code.md` exists on disk (even when empty).
- [ ] H1 line reads `# {{ repo }} · Dead code`.
- [ ] All three H2 headings exist in the fixed order `Unreferenced exports`, `Unreferenced files`, `Dead imports`.
- [ ] Every populated table row has a backtick path citation in its Path/File/Imported-from column.
- [ ] When all three buckets are empty: the banner `No unreferenced symbols detected.` appears once, the graph hash is cited, and each H2 contains `_none_`.
- [ ] When tables have rows: column headers match the fixed schema in Section 5.
- [ ] No YAML frontmatter on the output.
- [ ] No row cites a symbol absent from `list_dead_code` output (spot-check 3).

## 9. Anti-goals

- Do not re-call `list_dead_code` inside the same packet run — one call, then cache in the Work log.
- Do not invent unreferenced symbols, files, or imports — every row must map to a `list_dead_code` response entry.
- Do not omit the three H2 headings when buckets are empty; the skeleton must stay consistent for Phase E.
- Do not delete the file or skip the Write step when buckets are empty — the banner is the product in that case.
- Do not write YAML frontmatter on the output file.
- Do not emit emojis.

---

## Work log

{{ subagent fills this section per the write protocol }}

## Validation

{{ checks run, outputs pasted, any fixes applied }}

## Summary

{{ one paragraph — what shipped, row counts per bucket, whether the empty-state path was taken }}

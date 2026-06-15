---
role: doc-behavior-processes
model: sonnet
output: "{{ docs_root }}/behavior/processes.md"
depends_on:
  - "{{ context_path }}"
  - "{{ prefetch_path }}"
status: IN_PROGRESS
---

# Packet · {{ repo }} · behavior/processes.md

## 1. Objective

Produce `{{ docs_root }}/behavior/processes.md`: one H2 per top process in `{{ repo }}`, each with a numbered step list citing `path:LOC` on every line, entry-point attribution (HTTP route / MCP tool / CLI command / scheduled job), and a `### Related` subsection with backtick citations to handler files.

## 2. Scope

- Create: `{{ docs_root }}/behavior/processes.md`
- Do not touch: `{{ docs_root }}/behavior/state-machines.md`, any other file under `{{ docs_root }}/`, any source file in the repo, `.context.md`, `.prefetch.md`, or any `.packets/*.md` other than this one.

## 3. Input specification

| Source | Read how | Cache state |
|---|---|---|
| Shared context | `Read {{ context_path }}` | always first |
| Prefetch ledger | `Read {{ prefetch_path }}` | always first |
| Top processes | `{{ context_path }} § Top processes` | cached |
| Route inventory | `{{ prefetch_path }} § route_map` | cached |
| MCP tool inventory | `{{ prefetch_path }} § tool_map` | cached |
| Step sequence per process | `mcp__codehub__context({symbol: <process-name>})` | mid-run |
| Disambiguation lookup | `mcp__codehub__query({text: "<process name>"})` | mid-run, only on collision |

## 4. Process

1. `Read {{ context_path }}` and `Read {{ prefetch_path }}`. Confirm the Top processes list is present; if empty, follow Fallback paths.
2. Select the top 8 processes from `.context.md § Top processes` (ranked by `step_count`).
3. For each of the 8: call `mcp__codehub__context({symbol: <process-name>})` to pull entry point, ordered outbound calls, and handler files. Cache the per-process digest in this packet's Work log.
4. Group each process by its initiator using the cached `route_map` / `tool_map` digest: HTTP route, MCP tool, CLI command, scheduled job, or internal.
5. Draft `processes.md` with H1 = `{{ repo }} · Processes`. One H2 per process (max 8). Under each H2: a single-line `Entry point: <path:LOC>`, a numbered step list where every step cites `path:LOC`, then a `### Related` subsection listing the top 3-6 handler/helper files as backtick citations.
6. Processes with fewer than 3 concrete steps collapse into a trailing `## Minor flows` H2 (one bullet per flow, not an H2 section of their own).
7. `Write {{ docs_root }}/behavior/processes.md`.

## 5. Document format rules

- H1 = `{{ repo }} · Processes`. No decorative titles.
- No YAML frontmatter on the output file.
- One H2 per process. Max 8 H2s for full processes; everything else goes under `## Minor flows`.
- Each H2 opens with `Entry point: <backtick path:LOC>` on its own line.
- Step list is numbered. Every numbered step ends with a backtick `path:LOC` citation.
- `### Related` subsection lives at the end of each H2; bullets are backtick citations only (no prose).
- No emojis. No filler adverbs.

## 6. Tool usage guide

| Need | Tool | Why |
|---|---|---|
| Process roster + step counts | `{{ context_path }} § Top processes` | precomputed; do not re-call `sql` |
| Ordered steps per process | `mcp__codehub__context` | outbound edges + ordering hints |
| Initiator attribution | `{{ prefetch_path }} § route_map` / `§ tool_map` | cached inventories |
| Disambiguate colliding names | `mcp__codehub__query` | only when two symbols share a name |
| Recover stale graph | `Grep` the repo | fallback when `context` returns nothing |

## 7. Fallback paths

- If a process has fewer than 3 steps in `context`: collapse it into the trailing `## Minor flows` H2 rather than giving it its own section. Cite the collapse in the Work log.
- If `.context.md § Top processes` is empty: fall back to `mcp__codehub__sql({query: "SELECT name, file_path, step_count FROM nodes WHERE kind='Process' ORDER BY step_count DESC LIMIT 8"})`. Cite the fallback in the Work log.
- If `mcp__codehub__query` for a process name returns nothing (graph out of sync): `Grep` the repo for the name and cite the Grep hits with an inline `*graph stale for this process*` note.
- If `context` errors on a named process: omit the process's step list, keep the H2 as a stub with `*context unavailable — see Grep fallback*`, and enumerate its handler files from `Grep` hits.

## 8. Success criteria

- [ ] `{{ docs_root }}/behavior/processes.md` exists on disk.
- [ ] H1 line reads `# {{ repo }} · Processes`.
- [ ] At least 3 H2 entries exist for full processes (not counting `## Minor flows`).
- [ ] At most 8 full-process H2s exist.
- [ ] Every numbered step in every process has a backtick `path:LOC` citation.
- [ ] Every H2 contains a `### Related` subsection with at least 1 backtick citation.
- [ ] No H2 corresponds to a process name absent from the graph (or from the `Grep` fallback).
- [ ] No YAML frontmatter on the output.

## 9. Anti-goals

- Do not re-call `sql`, `route_map`, or `tool_map` — those digests are cached in `.context.md` / `.prefetch.md`.
- Do not invent process names — every H2 must map to a graph node or a `Grep`-verified function.
- Do not document more than 8 full processes; overflow goes to `## Minor flows`.
- Do not duplicate steps from handler files in prose form — cite them on a numbered line.
- Do not write YAML frontmatter on the output file.
- Do not emit emojis.

---

## Work log

{{ subagent fills this section per the write protocol }}

## Validation

{{ checks run, outputs pasted, any fixes applied }}

## Summary

{{ one paragraph — what shipped, where, and which processes landed in Minor flows vs. their own H2 }}

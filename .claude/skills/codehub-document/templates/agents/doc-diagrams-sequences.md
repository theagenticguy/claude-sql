---
role: doc-diagrams-sequences
model: sonnet
output: "{{ docs_root }}/diagrams/behavioral/sequences.md"
depends_on:
  - "{{ context_path }}"
  - "{{ prefetch_path }}"
status: IN_PROGRESS
---

# Packet · {{ repo }} · diagrams/behavioral/sequences.md

> **Conditional packet.** The orchestrator only seeds this skeleton when `{{ context_path }} § Top processes` reports at least one process with ≥ 3 steps. If the condition is not met, the packet is skipped at seed time and no file is produced.

## 1. Objective

Produce `{{ docs_root }}/diagrams/behavioral/sequences.md`: up to three Mermaid `sequenceDiagram` blocks, one per top process, each showing the outbound call order across 4-8 participants.

## 2. Scope

- Create: `{{ docs_root }}/diagrams/behavioral/sequences.md`
- Do not touch: any other file under `{{ docs_root }}/`, any source file in the repo, `.context.md`, `.prefetch.md`, or any `.packets/*.md` other than this one.

## 3. Input specification

| Source | Read how | Cache state |
|---|---|---|
| Shared context | `Read {{ context_path }}` | always first |
| Prefetch ledger | `Read {{ prefetch_path }}` | always first |
| Top processes (with step counts) | `{{ context_path }} § Top processes` | cached |
| Process step order | `mcp__codehub__context({symbol: <process-name>})` per top 3 processes | mid-run |
| Participant labels | `mcp__codehub__query({text: <actor-name>})` when a step's symbol is ambiguous | mid-run, on demand |

## 4. Process

1. `Read {{ context_path }}` and `Read {{ prefetch_path }}`. Confirm which processes in `§ Top processes` have ≥ 3 steps — those are candidates.
2. Pick the top 3 candidates by step count (ties broken by entry-point centrality from `.context.md`). If fewer than 3 qualify, emit only the qualifying count (1 or 2 diagrams).
3. For each chosen process, call `mcp__codehub__context({symbol: <process-name>})` and extract the outbound call sequence in dispatch order. Cache the digest in this packet's Work log.
4. Derive 4-8 participant lifelines per process by grouping step targets into community / module bands. Lifelines are listed in dispatch order at the top of each `sequenceDiagram`.
5. Draft each `sequenceDiagram`: solid arrows (`->>`) for synchronous calls, dashed (`-->>`) for returns. Short labels (≤ 15 chars on edges, ≤ 20 chars on participant names).
6. If any single diagram exceeds 20 nodes (participants + step-labeled messages), keep the top-20 and move overflow into a `## Legend (overflow)` table below that block.
7. `Write {{ docs_root }}/diagrams/behavioral/sequences.md` with H1 = `{{ repo }} · Sequences`, one H2 per process, one fenced `sequenceDiagram` per H2.

## 5. Document format rules

- H1 = `{{ repo }} · Sequences`. One H2 per process — `## <process-name>` — with the `sequenceDiagram` immediately beneath.
- No YAML frontmatter on the output file.
- Up to three ` ```mermaid ` fences, each containing exactly one `sequenceDiagram`.
- 4-8 participants per diagram; solid arrows for calls, dashed for returns.
- Every participant and step target must correspond to a real symbol from `context` — no invented identifiers.
- Node labels ≤ 20 chars; edge labels ≤ 15 chars.
- No emojis. No filler adverbs.

## 6. Tool usage guide

| Need | Tool | Why |
|---|---|---|
| Process list + step counts | `{{ context_path }} § Top processes` | precomputed; gates the conditional |
| Outbound call order | `mcp__codehub__context` | dispatch sequence for lifelines |
| Actor label disambiguation | `mcp__codehub__query` | when a step target has multiple matches |
| Diagram idioms | `references/mermaid-patterns.md § Top process` | canonical `sequenceDiagram` shape + rules |

## 7. Fallback paths

- If no process has ≥ 3 steps at seed time, the orchestrator never seeds this packet — you should not be running. If you are and the cached list still shows none ≥ 3: write the gap to Work log and stop.
- If a process's `context` call errors: skip that process entirely and note the skip in the Work log; proceed with the remaining candidates.
- If fewer than 3 processes qualify: emit only the qualifying count. Do not pad with sub-3-step processes.
- If participant count would exceed 8 for a process: group adjacent step targets into a single band (e.g., collapse `Parser` + `Lexer` into `Parsing`) and note the grouping in the Work log.

## 8. Success criteria

- [ ] `{{ docs_root }}/diagrams/behavioral/sequences.md` exists on disk.
- [ ] H1 line reads `# {{ repo }} · Sequences`.
- [ ] Between 1 and 3 ` ```mermaid ` fences, each containing a `sequenceDiagram`.
- [ ] Every diagram has 4-8 participants with labels ≤ 20 chars.
- [ ] Every message edge has a label ≤ 15 chars.
- [ ] One H2 per diagram; H2 text matches the process name from `.context.md § Top processes`.
- [ ] No YAML frontmatter on the output.
- [ ] Every participant name maps to a symbol returned by `context` (spot-check 3).

## 9. Anti-goals

- Do not re-call `sql` or `project_profile` — those are cached in `.prefetch.md` / `.context.md`.
- Do not invent participants, step targets, or message labels — every identifier must come from a tool response.
- Do not write YAML frontmatter on the output file.
- Do not emit more than three `sequenceDiagram` blocks.
- Do not emit a diagram for any process with fewer than 3 steps, even if it lets you reach 3 total.
- Do not exceed 20 nodes in any single diagram; overflow goes into a per-diagram Legend table.
- Do not emit emojis.

---

## Work log

{{ subagent fills this section per the write protocol }}

## Validation

{{ checks run, outputs pasted, any fixes applied }}

## Summary

{{ one paragraph — what shipped, where, why the process selection and lifeline grouping went the way they did }}

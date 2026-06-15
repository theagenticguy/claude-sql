---
role: doc-behavior-state-machines
model: sonnet
output: "{{ docs_root }}/behavior/state-machines.md"
depends_on:
  - "{{ context_path }}"
  - "{{ prefetch_path }}"
status: IN_PROGRESS
---

# Packet · {{ repo }} · behavior/state-machines.md

## 1. Objective

Produce `{{ docs_root }}/behavior/state-machines.md`: one H2 per state machine in `{{ repo }}`, each containing exactly one Mermaid `stateDiagram-v2` block that reflects the states and transitions declared in source, followed by a `path:LOC` citation to the definition site.

## 2. Scope

- Create: `{{ docs_root }}/behavior/state-machines.md`
- Do not touch: `{{ docs_root }}/behavior/processes.md`, any other file under `{{ docs_root }}/`, any source file in the repo, `.context.md`, `.prefetch.md`, or any `.packets/*.md` other than this one.
- Conditional file — this packet is only seeded when `sql(WHERE kind='StateMachine')` returns ≥ 2 rows. The state-machine count is carried in `{{ context_path }} § State machines`; the orchestrator verified the count before spawning.

## 3. Input specification

| Source | Read how | Cache state |
|---|---|---|
| Shared context | `Read {{ context_path }}` | always first |
| Prefetch ledger | `Read {{ prefetch_path }}` | always first |
| State-machine inventory | `{{ context_path }} § State machines` (count + names + paths) | cached |
| State-machine nodes (detail) | `mcp__codehub__sql({query: "SELECT name, file_path, start_line FROM nodes WHERE kind='StateMachine'"})` | mid-run, only if `.context.md` slice is truncated |
| States + transitions per machine | `mcp__codehub__context({symbol: <machine-name>})` | mid-run |
| Verbatim state/transition text | `Read <file_path>:<start_line>` | mid-run |

## 4. Process

1. `Read {{ context_path }}` and `Read {{ prefetch_path }}`. Confirm `.context.md § State machines` lists ≥ 2 machines. If the slice is truncated, call the `sql` fallback in Section 3.
2. For each state machine: call `mcp__codehub__context({symbol: <machine-name>})` to pull states, entry state, transitions, and terminal states.
3. For each machine: `Read` the definition file at `start_line..start_line+60` to verify state names and transition labels before drawing. Do not invent transitions.
4. Draft `state-machines.md` with H1 = `{{ repo }} · State machines`. One H2 per machine, in alphabetical order by machine name. Under each H2: exactly one fenced Mermaid `stateDiagram-v2` block, then a single-line `Defined at: <backtick path:LOC>` citation.
5. Transitions use the source-level event name as the Mermaid edge label (e.g., `start()`, `complete()`). Do not paraphrase event names.
6. `Write {{ docs_root }}/behavior/state-machines.md`.

## 5. Document format rules

- H1 = `{{ repo }} · State machines`. No decorative titles.
- No YAML frontmatter on the output file.
- One H2 per state machine. H2s in alphabetical order.
- Each H2 contains exactly one Mermaid fence with `stateDiagram-v2`. Not `stateDiagram`, not `flowchart`.
- Each H2 ends with `Defined at: <backtick path:LOC>` on its own line.
- Mermaid node names match source state identifiers verbatim; transition labels match source event names verbatim.
- No emojis. No filler adverbs.

## 6. Tool usage guide

| Need | Tool | Why |
|---|---|---|
| State-machine roster | `{{ context_path }} § State machines` | precomputed; do not re-call `sql` |
| States + transitions | `mcp__codehub__context` | outbound edges encode transitions |
| Verbatim state / event names | `Read` at `file_path:start_line` | graph stores the structure, not the literal text |
| Disambiguate machine names | `mcp__codehub__query` | only when two machines share a name |

## 7. Fallback paths

- If `.context.md § State machines` is truncated or absent: call `mcp__codehub__sql({query: "SELECT name, file_path, start_line FROM nodes WHERE kind='StateMachine'"})`. Cite the fallback in the Work log.
- If the sql roster returns < 2 machines (the conditional precondition was wrong): write the gap to the Work log, stop, and do not emit an empty or 1-machine file — the orchestrator will prune this packet from the README.
- If `context` returns no transitions for a machine: `Read` the definition file at `start_line..start_line+60` and parse the states/transitions manually; mark the machine H2 with `*transitions derived by direct read*` in the Work log.
- If a machine has no terminal state in source: draw the diagram without a `--> [*]` edge and note the absence in the Work log (do not invent a terminal state).

## 8. Success criteria

- [ ] `{{ docs_root }}/behavior/state-machines.md` exists on disk.
- [ ] H1 line reads `# {{ repo }} · State machines`.
- [ ] At least 2 H2 entries exist (matches the conditional precondition).
- [ ] Every H2 contains exactly one `stateDiagram-v2` Mermaid fence.
- [ ] No H2 contains a second Mermaid block or a non-Mermaid diagram.
- [ ] Every H2 ends with a `Defined at: <backtick path:LOC>` line.
- [ ] Every state name and transition label in the Mermaid blocks appears in the source file at the cited path (spot-check 1 machine).
- [ ] No YAML frontmatter on the output.

## 9. Anti-goals

- Do not re-call `sql` over `StateMachine` nodes — the count and names are cached in `.context.md`.
- Do not invent state names or transition labels — every identifier must come from `context` output or a `Read` of the definition file.
- Do not emit more than one Mermaid block per H2.
- Do not use `stateDiagram` (v1); use `stateDiagram-v2` only.
- Do not write YAML frontmatter on the output file.
- Do not emit emojis.

---

## Work log

{{ subagent fills this section per the write protocol }}

## Validation

{{ checks run, outputs pasted, any fixes applied }}

## Summary

{{ one paragraph — what shipped, which machines were drawn, and any that required a direct-read fallback }}

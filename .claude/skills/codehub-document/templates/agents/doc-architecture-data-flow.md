---
role: doc-architecture-data-flow
model: sonnet
output: "{{ docs_root }}/architecture/data-flow.md"
depends_on:
  - "{{ context_path }}"
  - "{{ prefetch_path }}"
status: IN_PROGRESS
---

# Packet · {{ repo }} · architecture/data-flow.md

## 1. Objective

Produce `{{ docs_root }}/architecture/data-flow.md`: a walk of the top 3 processes in `{{ repo }}`, each rendered as numbered steps plus one Mermaid `sequenceDiagram`. Every step cites `` `path:LOC` `` for the function that advances the flow.

## 2. Scope

- Create: `{{ docs_root }}/architecture/data-flow.md`
- Do not touch: any other file under `{{ docs_root }}/`, any source file in the repo, `.context.md`, `.prefetch.md`, or any `.packets/*.md` other than this one.

## 3. Input specification

| Source | Read how | Cache state |
|---|---|---|
| Shared context | `Read {{ context_path }}` | always first |
| Prefetch ledger | `Read {{ prefetch_path }}` | always first |
| Top processes | `{{ context_path }} § Top processes` | cached |
| Process entry points | `{{ prefetch_path }} § entry points` or `mcp__codehub__sql({query: "SELECT p.name, n.name AS entry_name, n.file_path, n.start_line FROM nodes p JOIN nodes n ON p.entry_point_id = n.id WHERE p.kind='Process'"})` | cached if digest present |
| Symbol neighborhoods along each flow | `mcp__codehub__context({symbol: <id>})` | mid-run (only if cache miss) |
| Query grounding for ambiguous steps | `mcp__codehub__query({text: "<concept>", limit: 10})` | mid-run (only if cache miss) |
| Source spans for step citations | `Read <file>` over `start_line..start_line+20` | mid-run |

## 4. Process

1. `Read {{ context_path }}` and `Read {{ prefetch_path }}`. Lock the ordered list of top processes; pick the top 3.
2. For each selected process, pull the entry point from `.prefetch.md § entry points`. If absent, call the `sql` query in the input spec and cache the digest in this packet's Work log.
3. For each flow, walk from the entry point outward using `context({symbol: <entry>})` (reuse cached digest if present). Record the ordered call chain: caller → callee → downstream participant. Cap at 8 steps per flow.
4. Resolve each participant to a logical actor (CLI, MCP server, Analysis, Storage, etc.) by cross-referencing its file path against `.context.md § Top communities`. Use the community `inferred_label` as the `participant` name in the Mermaid diagram.
5. For every step, `Read` the source span at `path:start_line-start_line+20` to confirm the function exists and extract the one-line description. Do not paraphrase beyond that.
6. Draft the H2 block per flow: `## Flow N: <process-name>`, followed by numbered steps (each citing `` `path:LOC` ``), then a fenced ` ```mermaid ` block containing one `sequenceDiagram`.
7. `Write {{ docs_root }}/architecture/data-flow.md` with H1 = `{{ repo }} · Data flow`, at most 3 `## Flow N:` H2 sections.

## 5. Document format rules

- H1 = `{{ repo }} · Data flow`. No decorative titles.
- No YAML frontmatter on the output file.
- One H2 per flow, in the form `## Flow N: <process-name>`. Maximum 3 H2 flow sections.
- Each flow body = numbered step list (Markdown ordered list) + exactly one `sequenceDiagram` fenced with ` ```mermaid `.
- Every numbered step cites `` `path:LOC` ``; the entry-point step must cite the function, not the file.
- `sequenceDiagram` participants use short labels (≤ 20 chars); participant names must match the community `inferred_label` (or process actor names from `.context.md`).
- No emojis. No filler adverbs.

## 6. Tool usage guide

| Need | Tool | Why |
|---|---|---|
| Process list + entry points | `{{ context_path }} § Top processes` + `{{ prefetch_path }}` | precomputed; do not re-call `sql` |
| Symbol neighborhood for call chain | `mcp__codehub__context` | inbound/outbound relations grounded in the graph |
| Concept grounding when a step is ambiguous | `mcp__codehub__query` | hybrid BM25+vector, process-grouped |
| Verifying step text | `Read` at `path:start_line-start_line+20` | avoid paraphrase drift |

## 7. Fallback paths

- If `.context.md § Top processes` lists fewer than 3 processes: render only the processes present. Do not pad with synthetic flows.
- If a process has no `entry_point_id`: fall back to `query({text: "<process-name> entry", limit: 5})`, pick the highest-ranked symbol, and mark the step `*entry inferred*`.
- If a `context` call errors mid-chain: truncate the flow at the last verified step and append `> _flow truncated: downstream context unavailable_` under the numbered list. Still emit the `sequenceDiagram` using the verified subset.
- If a participant's community has no `inferred_label`: use the top-level folder name as the participant label and note the substitution in the Work log.

## 8. Success criteria

- [ ] `{{ docs_root }}/architecture/data-flow.md` exists on disk.
- [ ] H1 line reads `# {{ repo }} · Data flow`.
- [ ] Between 1 and 3 H2 sections match `^## Flow \d+:` (verify with grep).
- [ ] Exactly one ` ```mermaid ` fence per flow section; each fence contains `sequenceDiagram` as the first non-empty line.
- [ ] Every numbered step has a backtick `` `path:LOC` `` citation (grep each ordered-list line for `` ` ``).
- [ ] The number of `` `mermaid `` fences equals the number of `## Flow` H2 sections.
- [ ] No `sequenceDiagram` participant label exceeds 20 characters.
- [ ] No YAML frontmatter on the output.
- [ ] No `path:LOC` citation references a file that does not exist (spot-check 3).

## 9. Anti-goals

- Do not re-call any MCP tool whose digest is in `.prefetch.md` — read the cached summary. If the cached slice is truncated, call with a narrower filter, not a blanket re-fetch.
- Do not invent process names, participants, or call edges — every step must map to a cached relation or a verified `Read` span.
- Do not write YAML frontmatter on the output file.
- Do not emit more than 3 `sequenceDiagram` blocks — overflow is for a separate packet.
- Do not emit emojis. Do not use filler adverbs.
- Do not paraphrase step bodies beyond a one-line summary of the quoted source span.

---

## Work log

{{ subagent fills this section per the write protocol }}

## Validation

{{ checks run, outputs pasted, any fixes applied }}

## Summary

{{ one paragraph — what shipped, where, why the flow selection went the way it did }}

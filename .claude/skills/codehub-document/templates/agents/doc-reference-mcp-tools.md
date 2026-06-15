---
role: doc-reference-mcp-tools
model: sonnet
output: "{{ docs_root }}/reference/mcp-tools.md"
depends_on:
  - "{{ context_path }}"
  - "{{ prefetch_path }}"
status: IN_PROGRESS
---

# Packet · {{ repo }} · reference/mcp-tools.md

## 1. Objective

Produce `{{ docs_root }}/reference/mcp-tools.md`: the authoritative reference for every MCP tool `{{ repo }}` exposes, one H2 per tool, each with a verbatim signature, input/output shapes, a one-sentence purpose, and a `path:LOC` citation to the handler file.

## 2. Scope

- Create: `{{ docs_root }}/reference/mcp-tools.md`
- Do not touch: `{{ docs_root }}/reference/public-api.md`, `{{ docs_root }}/reference/cli.md`, or any other file under `{{ docs_root }}/`, any source file in the repo, `.context.md`, `.prefetch.md`, or any `.packets/*.md` other than this one.
- Conditional file — this packet is only seeded when `{{ repo }}` contains an MCP server package (the orchestrator verified `project_profile.stacks` includes `"MCP"` or `tool_map` is non-empty before spawning).

## 3. Input specification

| Source | Read how | Cache state |
|---|---|---|
| Shared context | `Read {{ context_path }}` | always first |
| Prefetch ledger | `Read {{ prefetch_path }}` | always first |
| MCP tool inventory | `{{ prefetch_path }} § tool_map` | cached |
| Tool handler nodes | `mcp__codehub__sql({query: "SELECT name, file_path, start_line FROM nodes WHERE kind='Function' AND file_path LIKE '%mcp%'"})` | mid-run, only if `tool_map` slice is truncated |
| Verbatim signatures | `Read <file_path>:<start_line>` (signatures are not stored in the graph) | mid-run |
| Usage count per tool | `mcp__codehub__context({symbol: <tool-handler>})` | mid-run |

## 4. Process

1. `Read {{ context_path }}` and `Read {{ prefetch_path }}`. Confirm the `tool_map` digest is present and non-empty; if empty, follow Fallback paths.
2. Build the tool roster from `{{ prefetch_path }} § tool_map`: tool name, handler `file_path:start_line`, input/output schema digest.
3. For each tool in the roster: `Read` the handler file at `start_line..start_line+30` and extract the exact signature (function or registration block). Do not paraphrase.
4. For each tool: pull a one-sentence purpose from the registration description (the MCP `describe` string); if missing, infer from the handler docstring and mark the inference in the Work log.
5. Draft `reference/mcp-tools.md`: H1 = `{{ repo }} · MCP tools`, one H2 per tool in alphabetical order. Each H2 contains the verbatim signature fenced as code, a one-sentence purpose, an input/output shape summary, and a `path:LOC` citation.
6. `Write {{ docs_root }}/reference/mcp-tools.md`.

## 5. Document format rules

- H1 = `{{ repo }} · MCP tools`. No decorative titles.
- No YAML frontmatter on the output file.
- One H2 per tool. Tool order = alphabetical by tool name.
- Signatures are fenced code blocks, quoted verbatim from the handler file — never paraphrased, never retyped from memory.
- Every H2 ends with a backtick `path:LOC` citation pointing at the handler `file_path:start_line`.
- No emojis. No filler adverbs.

## 6. Tool usage guide

| Need | Tool | Why |
|---|---|---|
| Full tool inventory | `{{ prefetch_path }} § tool_map` | precomputed; do not re-call `tool_map` |
| Verbatim signature text | `Read` at `file_path:start_line` | graph stores names/locs, not signature text |
| Handler usage count | `mcp__codehub__context` | inbound count; signals which tools are load-bearing |
| Tools not in `tool_map` | `mcp__codehub__sql` filtered to MCP files | fallback only when `tool_map` is stale |

## 7. Fallback paths

- If `{{ prefetch_path }} § tool_map` is empty but `project_profile.stacks` includes `"MCP"`: call `mcp__codehub__sql({query: "SELECT name, file_path, start_line FROM nodes WHERE kind='Function' AND (file_path LIKE '%mcp%' OR file_path LIKE '%tools%') ORDER BY file_path"})`, filter to registered handlers by grepping for tool-registration decorators, and cite the fallback in the Work log.
- If `tool_map` returns `[]` and `project_profile.stacks` does not contain `"MCP"`: do not emit an empty file. Write the gap to the Work log, mark `status: COMPLETE` with a note, and skip the Write step — the orchestrator will prune this packet from the README.
- If a handler `Read` fails (file moved since the last `codehub analyze`): flag the row with `*graph stale — verify with codehub analyze*` and cite the graph-recorded path.
- If a tool registration has no description string: infer from the handler's top-level docstring; mark the H2 body `*description inferred from docstring*` in-line.

## 8. Success criteria

- [ ] `{{ docs_root }}/reference/mcp-tools.md` exists on disk.
- [ ] H1 line reads `# {{ repo }} · MCP tools`.
- [ ] Every tool returned by `tool_map` has exactly one H2 in the output.
- [ ] No H2 exists for a tool name not in `tool_map` (or the sql fallback roster).
- [ ] Every H2 contains a fenced code block with a verbatim signature.
- [ ] Every H2 ends with a backtick `path:LOC` citation.
- [ ] No YAML frontmatter on the output.
- [ ] Tool H2s are in alphabetical order (spot-check first and last).

## 9. Anti-goals

- Do not re-call `tool_map` — its digest is cached in `.prefetch.md`.
- Do not invent tool names, input fields, or output fields — every identifier must come from `tool_map` or a `Read` of the handler file.
- Do not paraphrase signatures. Quote the source or use the `Read` fallback.
- Do not emit a `reference/mcp-tools.md` with zero H2s; follow the Fallback path that skips the file instead.
- Do not write YAML frontmatter on the output file.
- Do not emit emojis.

---

## Work log

{{ subagent fills this section per the write protocol }}

## Validation

{{ checks run, outputs pasted, any fixes applied }}

## Summary

{{ one paragraph — what shipped, where, and any tools that required a fallback }}

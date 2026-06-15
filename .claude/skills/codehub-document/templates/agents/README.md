# templates/agents — per-file packet skeletons

Each `.md` file here is a **packet skeleton** for one output document produced by `codehub-document`. The skill orchestrator seeds one packet per output file by copying a skeleton to `<docs-root>/.packets/<file-role>.md`, substituting placeholders, and spawning a `general-purpose` subagent with a short prompt that points at the packet path.

## Contract

Every skeleton follows the same shape, ported from `erpaval/templates/session/task-skeleton.md` and adapted for doc-writing:

```yaml
---
role: <file-role>                         # doc-architecture-system-overview, doc-reference-public-api, ...
model: sonnet                             # overridable by the orchestrator
output: "{{ docs_root }}/<path>.md"       # exactly one file per packet
depends_on:
  - "{{ context_path }}"                  # .codehub/docs/.context.md (or equivalent)
  - "{{ prefetch_path }}"                 # .codehub/docs/.prefetch.md
status: IN_PROGRESS
---
```

Then ten numbered sections, in order, using the ERPAVal `<write_protocol>` discipline:

1. **Objective** — one sentence: the single file this packet produces and why it exists.
2. **Scope** — the exact output path; any other paths the subagent is *not* allowed to touch.
3. **Input specification** — source MCP tools / files, each marked `cached` (read from `.prefetch.md`) or `mid-run` (subagent may call).
4. **Process** — numbered steps from read-shared-context → draft → write.
5. **Document format rules** — H1 shape, citation grammar, diagram rules, "no YAML frontmatter on outputs".
6. **Tool usage guide** — a short table mapping need → tool → why.
7. **Fallback paths** — what to do when a tool returns empty / errors / is stale.
8. **Success criteria** — mechanical checks the subagent validates before flipping `status: COMPLETE`.
9. **Anti-goals** — common failure modes to avoid (re-calling cached tools, inventing edges, hallucinating symbol names).
10. **Work log / Validation / Summary** — three trailing sections the subagent edits during execution per the write protocol.

## Placeholders

Filled by the orchestrator at seed time. Common to every skeleton:

| Placeholder | Example value (single-repo) | Example value (group mode) |
|---|---|---|
| `{{ docs_root }}` | `.codehub/docs` | `.codehub/groups/<group>/docs` |
| `{{ repo }}` | `opencodehub` | member repo name |
| `{{ context_path }}` | `.codehub/docs/.context.md` | `.codehub/groups/<group>/docs/.context.md` |
| `{{ prefetch_path }}` | `.codehub/docs/.prefetch.md` | `.codehub/groups/<group>/docs/.prefetch.md` |
| `{{ packet_path }}` | `.codehub/docs/.packets/doc-architecture-system-overview.md` | `.codehub/groups/<group>/docs/.packets/...` |
| `{{ graph_hash }}` | `sha256:a1b2c3…` | (same) |

Cross-repo skeletons add:

| Placeholder | Example |
|---|---|
| `{{ group }}` | `platform` |
| `{{ group_docs_root }}` | `.codehub/groups/platform/docs` |

## Spawn prompt

The orchestrator spawns each subagent with the prompt in `../orchestrator-prompt.md`. That prompt is deliberately short — all role-specific instructions live in the packet itself.

## Write protocol

Every skeleton embeds the doc-variant `<write_protocol>` block verbatim. The rhythm is "read tool output → Write output file → edit packet section with the outcome" (vs. ERPAVal's "read code → edit code → run check → edit packet").

## Why file-level, not role-level

One packet = one output file. Blast radius of a failing subagent is one file. `--refresh` and `--section` become trivial 1-subagent dispatches. `.docmeta.json.sections[i].agent` still records the role for auditability.

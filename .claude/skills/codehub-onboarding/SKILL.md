---
name: codehub-onboarding
description: "Use when the user asks for an ONBOARDING, getting-started, or new-engineer guide for the current repo or group. Examples: \"write ONBOARDING.md\", \"generate an onboarding doc for new hires\", \"what should a new engineer read first\". Produces a ranked reading order from `project_profile` + top processes + entry points + owners + centrality. DO NOT use for full architecture books (use `codehub-document`) or PR summaries (use `codehub-pr-description`)."
allowed-tools: "Read, Write, Glob, mcp__codehub__project_profile, mcp__codehub__query, mcp__codehub__context, mcp__codehub__route_map, mcp__codehub__tool_map, mcp__codehub__owners, mcp__codehub__sql, mcp__codehub__list_repos, Task"
argument-hint: "[output-path] [--committed]"
color: green
model: sonnet
---

# codehub-onboarding

Produces a single ONBOARDING.md with a ranked reading order drawn from graph centrality. The wedge is the ranked reading list — a generic README scaffold cannot produce this.

## Preconditions

1. `mcp__codehub__list_repos` returns the target. If not, emit `Run codehub analyze first — repo <name> is not indexed.` and stop.
2. `codehub status` is fresh. If stale, emit `Run 'codehub analyze' first — index is stale` and stop.

## Arguments

- `[output-path]` — where to write. Defaults:
  - without `--committed`: `.codehub/ONBOARDING.md` (gitignored)
  - with `--committed`: `docs/ONBOARDING.md` (gitignored default flipped)
- `--committed` — write to a committed path instead of `.codehub/`.

## Process

1. Run the preconditions.
2. `mcp__codehub__project_profile({repo})` — languages, stacks, entry points, 2-sentence summary.
3. `mcp__codehub__route_map({repo})` — HTTP surface (if present).
4. `mcp__codehub__tool_map({repo})` — MCP/CLI surface (if present).
5. `mcp__codehub__sql({query: "SELECT name, file_path, in_degree + out_degree AS centrality FROM nodes WHERE kind IN ('File','Module','Class') ORDER BY centrality DESC LIMIT 15"})` — top-centrality nodes, the "read these first" candidates.
6. For the top 8 of those: `mcp__codehub__context({symbol: <id>})` to pull a one-line summary + owners.
7. `mcp__codehub__owners({path})` on the top-3 folders by file count — gives the "ask these humans" list.
8. Dispatch a single `Task` with the `doc-onboarding` specialty role (inline in this skill — see `references/onboarding-template.md`).
9. Assemble the output using the template below.
10. `Write` to the resolved output path.

## Output template

```markdown
# <repo> · Onboarding

*Generated <ISO-8601>. Refresh via `/codehub-onboarding`.*

## TL;DR

2 sentences: what this repo does + the single most important mental model to hold.

## Stack

| Layer | Tech | Source |
|---|---|---|
| Runtime | Node 22 | `package.json:7` |
| Storage | DuckDB | `packages/storage/src/index.ts:12` |
| ... | ... | ... |

## Read these 10 files first (in order)

1. `packages/cli/src/bin.ts` — CLI entry point. `(45 LOC)`
2. `packages/mcp/src/server.ts` — MCP server bootstrap. `(320 LOC)`
3. `packages/ingestion/src/pipeline.ts` — the phase DAG. `(180 LOC)`
... (ranked by centrality; each with a one-sentence reason)

## Walk one process end-to-end

Pick the highest-step-count process and walk it:

1. **Enter** at `packages/cli/src/commands/analyze.ts:14`.
2. **Dispatches** to `packages/ingestion/src/phases/parse.ts:22`.
3. **Writes** via `packages/storage/src/write.ts:88`.
4. **Exits** at `packages/cli/src/commands/analyze.ts:102`.

## Ask these humans

| Area | Owner | Share |
|---|---|---|
| `packages/mcp/` | alice@ | 72% |
| `packages/ingestion/` | bob@ | 45% |
| `packages/storage/` | charlie@ | 68% |

## Next steps

- Run `codehub analyze .` in your checkout to build the local graph.
- Ask the `code-analyst` subagent (or any MCP-aware client) "how does X work" to get a graph-grounded brief.
- Read [architecture/system-overview.md](./docs/architecture/system-overview.md) if `codehub-document` has already produced it.
```

## Document format rules

- H1 = "{{repo}} · Onboarding".
- Top reading list is exactly 10 items, ranked.
- Every file in the reading list has a one-sentence reason + LOC count in backticks.
- No YAML frontmatter on the output.
- No emojis.

## Fallback paths

- If `sql` over centrality returns fewer than 10 rows: pad with highest-fan-out `Function` nodes and note `*graph-small fallback*` in the list header.
- If no `Process` nodes exist (tiny repo or graph out of sync): skip the "Walk one process" section entirely; emit "This repo's graph does not yet encode processes. Re-run `codehub analyze` if unexpected." as a one-line note.
- If `owners` returns `[]` for all folders: replace the Ask-these-humans table with a `git shortlog -sn --since=90.days` pointer.

## Quality checklist

- [ ] Preconditions enforced.
- [ ] TL;DR is ≤ 2 sentences.
- [ ] Stack table has ≥ 3 rows.
- [ ] Reading list has exactly 10 items, each with a backtick citation.
- [ ] One process walked end-to-end (or the fallback note).
- [ ] Ask-these-humans table has ≥ 2 rows.
- [ ] Next-steps section exists with ≥ 3 concrete actions.

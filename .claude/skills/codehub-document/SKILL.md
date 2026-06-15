---
name: codehub-document
description: "Use when the user asks to generate, regenerate, or refresh long-form codebase documentation, an architecture book, a module map, or a per-repo reference — especially after `codehub analyze` finishes or after a large merge. Examples: \"document this repo\", \"regenerate the architecture docs\", \"write a module map for the monorepo\", \"produce a group-wide portfolio doc\". DO NOT use if the repo is not indexed — run `codehub analyze` first and confirm `mcp__codehub__list_repos` returns the repo. DO NOT use for PR descriptions (use `codehub-pr-description`), onboarding docs (use `codehub-onboarding`), or cross-repo contract maps alone (use `codehub-contract-map`)."
allowed-tools: "Read, Write, Edit, Glob, Grep, Bash(codehub:*), mcp__codehub__list_repos, mcp__codehub__project_profile, mcp__codehub__query, mcp__codehub__context, mcp__codehub__impact, mcp__codehub__dependencies, mcp__codehub__owners, mcp__codehub__risk_trends, mcp__codehub__route_map, mcp__codehub__tool_map, mcp__codehub__list_dead_code, mcp__codehub__list_findings, mcp__codehub__verdict, mcp__codehub__group_list, mcp__codehub__group_query, mcp__codehub__group_status, mcp__codehub__group_contracts, mcp__codehub__group_cross_repo_links, mcp__codehub__sql, Task"
argument-hint: "[output-dir] [--group <name>] [--committed] [--refresh] [--section <name>]"
color: indigo
model: sonnet
---

# codehub-document

Primary artifact generator. Produces a tree of cross-linked Markdown under `.codehub/docs/` (single-repo) or `.codehub/groups/<name>/docs/` (group mode) using a three-phase orchestration: **Phase 0 parallel precompute waves** → **Phase 1 file-level subagent fan-out** (one packet = one output file) → **Phase 2 deterministic cross-reference assembly**.

**Model policy.** This skill runs on Sonnet by default. Bump to Opus in two cases:

1. `--refresh --group` combined — the pruning + partial fan-out across members needs the extra judgment.
2. Any individual packet may set `model: opus` in its frontmatter to opt into Opus for synthesis-heavy roles. The cross-repo skeletons (`doc-cross-repo-*`) and `doc-analysis-risk-hotspots` typically do this; full-scan single-repo packets stay on Sonnet.

## Preconditions (check before Phase 0)

1. `mcp__codehub__list_repos` returns the target. If not, emit `Run codehub analyze first — repo <name> is not indexed.` and stop.
2. `codehub status` reports fresh. If stale, emit `Run 'codehub analyze' first — index is stale` and stop.
3. Group mode only: `mcp__codehub__group_status({group})` must return `fresh: true` for every member. If any member is stale, abort and name each stale repo.

## Arguments

- `[output-dir]` (optional positional) — where to write. Default is `.codehub/docs/` (gitignored). With `--committed`, default flips to `docs/codehub/` and the skill does not add a `.gitignore` entry.
- `--group <name>` — enable group mode. Phase 0 calls `group_list` + `group_status` + `group_contracts` + `group_query`. Phase CD dispatches `doc-cross-repo`.
- `--committed` — write under `docs/codehub/` (or user-supplied path) instead of `.codehub/docs/`. Does not touch `.gitignore`.
- `--refresh` — consult `.docmeta.json`, identify stale sections by comparing `max(mtime(section.sources[]))` against `section.mtime`, and dispatch exactly one file-level subagent per stale section (re-seeding its packet from `templates/agents/<role>.md`). Phase 2 always re-runs.
- `--section <name>` — regenerate one named section (e.g., `architecture/system-overview`). Dispatches exactly one subagent with one skeleton and re-runs Phase 2. Useful for targeted updates.

## Four-phase orchestration

### Phase 0 — Precompute shared context (parallel waves, no subagent)

Phase 0 writes `<docs-root>/.context.md` and `<docs-root>/.prefetch.md` so Phase 1 subagents read cached data instead of re-calling tools. It runs as three waves — **two of them are single-message tool-call batches**, so the MCP fan-out parallelizes.

**Wave 0a — independent precompute (one message, parallel)**. Issue all of these in a single tool-use batch:

- `mcp__codehub__list_repos`
- `mcp__codehub__project_profile`
- `mcp__codehub__sql` — schema probe: `SELECT table_name, column_name FROM information_schema.columns WHERE table_name IN ('nodes','relations') ORDER BY table_name, column_name`
- `mcp__codehub__route_map`
- `mcp__codehub__tool_map`
- `mcp__codehub__dependencies`
- `mcp__codehub__risk_trends`
- `mcp__codehub__list_dead_code`
- `mcp__codehub__list_findings`
- Group mode only: `mcp__codehub__group_list`, `mcp__codehub__group_status`, `mcp__codehub__group_contracts`

**Wave 0b — depends on 0a (one message, parallel)**. Needs schema column names + profile entry points from 0a, so it is a second batch. Issue in one message:

- `mcp__codehub__sql` — top communities (`SELECT … FROM nodes WHERE kind='Community' ORDER BY cohesion DESC LIMIT 10`)
- `mcp__codehub__sql` — top processes (`SELECT … FROM nodes WHERE kind='Process' ORDER BY step_count DESC LIMIT 10`)
- `mcp__codehub__sql` — relations slice for diagrams (filtered per the schema probe)
- `mcp__codehub__owners` × top-5 folders (derived from `project_profile` entry points + file-count heuristic)
- Group mode only: `mcp__codehub__group_query` for any canonical cross-repo search terms

**Wave 0c — inline Write (no tool batch)**. Deterministic post-processing; no MCP calls:

1. Assemble `<docs-root>/.context.md` (hard 200-line cap; per-section `truncated: true` when the raw output exceeds the cap). Sections: repo profile, schema probe, top communities, top processes, routes, MCP tools, owners summary, dependencies summary, dead-code counts, findings summary, risk trends. Group mode appends: group manifest, contracts matrix, freshness table.
2. Write `<docs-root>/.prefetch.md` — newline-delimited JSON, one record per tool call with `{tool, args, sha256, keys, cached_at, truncated}`. Example:

    ```json
    {"tool":"project_profile","args":{"repo":"opencodehub"},"sha256":"…","keys":["languages","stacks","entryPoints"],"cached_at":"2026-04-27T18:04:11Z","truncated":false}
    ```

The full layout of both files, plus the schema-preflight rationale and the Phase 0 pseudocode, live in `references/data-source-map.md`.

### Phase 1 — File-level subagent fan-out

One packet = one output file. The orchestrator seeds packets by copying `templates/agents/<role>.md` to `<docs-root>/.packets/<role>.md`, substituting the placeholders listed in `templates/agents/README.md § Placeholders`, and spawning a `general-purpose` subagent per packet with the prompt from `templates/orchestrator-prompt.md`.

**Skeletons (single-repo)**. Ten are always seeded; four are conditional on triggers observed in Phase 0:

| Role | Always / Conditional | Trigger |
|------|----------------------|---------|
| `doc-architecture-system-overview` | always | — |
| `doc-architecture-module-map` | always | — |
| `doc-architecture-data-flow` | always | — |
| `doc-reference-public-api` | always | — |
| `doc-behavior-processes` | always | — |
| `doc-analysis-risk-hotspots` | always | — |
| `doc-analysis-ownership` | always | — |
| `doc-analysis-dead-code` | always | — |
| `doc-diagrams-components` | always | — |
| `doc-diagrams-dependency-graph` | always | — |
| `doc-reference-cli` | conditional | CLI detected in `project_profile` entry points |
| `doc-reference-mcp-tools` | conditional | `tool_map` returns ≥ 1 row |
| `doc-behavior-state-machines` | conditional | ≥ 2 `StateMachine` nodes in the graph |
| `doc-diagrams-sequences` | conditional | ≥ 1 process with ≥ 3 steps |

**Group mode** adds three cross-repo skeletons seeded from `{{ group_docs_root }}/.packets/`:

- `doc-cross-repo-portfolio-map`
- `doc-cross-repo-contracts-matrix`
- `doc-cross-repo-dependency-flow`

**Dispatch priority (citation magnetism)**. Single-repo, up to ~10 packets per message, two messages back-to-back (no gate — this is purely how Claude Code's concurrent-Agent ceiling is managed):

1. **Message 1 (high-magnetism first)**: `system-overview`, `public-api`, `processes`, `components`, `module-map`, `data-flow`, plus up to four of the remaining always/conditional skeletons.
2. **Message 2 (immediately after)**: every remaining seeded skeleton.

**Group mode dispatch**. Sort all seeded packets by `(priority_class, repo_index)` and greedily fill batches of ≤ 10. Example: 3 repos × ~12 per-repo skeletons + 3 cross-repo skeletons ≈ 39 packets → 4 messages of ≤ 10, dispatched back-to-back.

**Spawn parameters** (per `templates/orchestrator-prompt.md § Usage at the orchestrator`):

- `subagent_type`: `general-purpose`
- `name`: the role (e.g. `doc-architecture-system-overview`)
- `description`: 3–5 words
- `model`: read from the packet's `model:` frontmatter (default `sonnet`, individual packets may request `opus`)
- `run_in_background`: `true`
- `prompt`: the canonical text from `templates/orchestrator-prompt.md`, with `{{ packet_path }}` substituted

**Monitoring**. Orchestrator tails `.packets/*.md` with `wc -l` at 30 s, 2 m, 5 m, then every 5 m — identical to the erpaval Act monitoring rhythm. A packet whose line count stops growing but whose `status:` line is still `IN_PROGRESS` is the signal to `SendMessage` a nudge or mark it failed.

### Phase 2 — Cross-reference assembler (inline, deterministic)

No LLM call. Pure regex + join. See `references/cross-reference-spec.md` for the full algorithm. Summary:

1. Extract every backtick `<path>:<LOC>` (or `<repo>:<path>:<LOC>`) citation from every generated Markdown file.
2. Build a co-occurrence index: `source_file → [docs_citing_it]`.
3. For any two docs sharing ≥ 2 common sources, append `## See also` (3–5 links) to both.
4. **Group mode — sourced cross-repo links (v2)**: call `mcp__codehub__group_cross_repo_links` with the current `--group` value. The tool returns a deterministic, alpha-sorted `links[]` array (each entry: `source_repo_uri`, `target_repo_uri`, `source_doc_path`, `target_doc_path`, `relation`, optional `evidence`). Embed that array **verbatim** into `.docmeta.json.cross_repo_links[]` (schema v2). Then render the `## See also (other repos in group)` footer by grouping links by `source_doc_path`, emitting one bullet per target, labelled by `relation` (e.g. `depends_on → orders-api/architecture.md`). Do NOT re-compute links heuristically; the tool is the single source of truth.
5. Write `<docs-root>/README.md` (landing page with the structure-is-deterministic disclaimer) and `<docs-root>/.docmeta.json` with `schema_version: 2`. `.docmeta.json.sections[i].agent` records the file-role (e.g. `doc-architecture-system-overview`) for `--refresh` traceability. Pre-v2 `.docmeta.json` files on disk remain readable; the orchestrator lazily upgrades them on the next regeneration by writing v2.

## `--refresh` algorithm

See `references/cross-reference-spec.md § --refresh algorithm` for the full procedure. One-line summary: compare `max(mtime(section.sources[]))` against `section.mtime`, dispatch exactly one file-level subagent per stale section (re-seeding its packet from the skeleton), then always re-run Phase 2.

## Progressive disclosure — references/

| Reference                          | When to consult                                          |
| ---------------------------------- | -------------------------------------------------------- |
| `references/document-templates.md` | Per-file structural templates (what goes in each section)|
| `references/data-source-map.md`    | Which MCP tools feed which subagent                      |
| `references/cross-reference-spec.md` | Phase E algorithm + `.docmeta.json` schema + `--refresh` |
| `references/mermaid-patterns.md`   | Mermaid idioms for each diagram type                     |

## Quality checklist

- [ ] Phase 0 ran waves 0a and 0b as single-message tool batches; `.context.md` is ≤ 200 lines; `.prefetch.md` has one JSON line per tool call including the schema probe.
- [ ] Phase 1 seeded one packet per output file under `<docs-root>/.packets/`, with placeholders substituted and `status: IN_PROGRESS`.
- [ ] Phase 1 dispatched packets in batches of ≤ 10 per message, priority-first (system-overview, public-api, processes, components, module-map, data-flow before the rest).
- [ ] Every generated file has H1 = identifier, no YAML frontmatter (the frontmatter lives in the packet, not the output).
- [ ] Every factual claim in every output has a backtick citation (`path:LOC` or `repo:path:LOC`).
- [ ] Every packet has `status: COMPLETE` and a populated Work log / Validation / Summary section before Phase 2 starts.
- [ ] Phase 2 wrote `.docmeta.json` validating against the schema in `references/cross-reference-spec.md`, with `sections[i].agent` set to the file-role.
- [ ] `See also` footers appear on every doc with ≥ 2 shared citations.
- [ ] Group mode: outputs from `doc-cross-repo-*` packets use `repo:path:LOC` citations exclusively.
- [ ] `codehub status` is fresh before this skill starts; otherwise the preconditions caught the stale state.

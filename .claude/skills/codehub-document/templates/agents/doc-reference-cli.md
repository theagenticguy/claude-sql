---
role: doc-reference-cli
model: sonnet
output: "{{ docs_root }}/reference/cli.md"
depends_on:
  - "{{ context_path }}"
  - "{{ prefetch_path }}"
status: IN_PROGRESS
---

# Packet · {{ repo }} · reference/cli.md

> **Conditional packet.** The orchestrator only seeds this packet when `project_profile.entry_points` for `{{ repo }}` includes a CLI. If the packet is present on disk, the orchestrator has already confirmed the precondition; do not re-check it as a reason to abort.

## 1. Objective

Produce `{{ docs_root }}/reference/cli.md`: one H2 per CLI subcommand (derived from `route_map`), each with a fenced usage block, a one-sentence description, a `` `path:LOC` `` citation, and a bulleted flag list where each flag cites `` `path:LOC` ``.

## 2. Scope

- Create: `{{ docs_root }}/reference/cli.md`
- Do not touch: any other file under `{{ docs_root }}/`, any source file in the repo, `.context.md`, `.prefetch.md`, or any `.packets/*.md` other than this one. In particular, do not write to `{{ docs_root }}/reference/public-api.md` — the public-api packet owns that file.

## 3. Input specification

| Source | Read how | Cache state |
|---|---|---|
| Shared context | `Read {{ context_path }}` | always first |
| Prefetch ledger | `Read {{ prefetch_path }}` | always first |
| Project profile (entry points) | `{{ context_path }} § Repo profile` | cached |
| CLI subcommand inventory | `{{ prefetch_path }} § route_map` or `mcp__codehub__route_map({repo: "{{ repo }}"})` | cached if digest present |
| Per-command signatures / flags | `mcp__codehub__signature({symbol: <id>})` per handler | mid-run (only if cache miss) |
| Handler source for flag extraction | `Read <file>` at `start_line..start_line+40` | mid-run |

## 4. Process

1. `Read {{ context_path }}` and `Read {{ prefetch_path }}`. Confirm `project_profile.entry_points` includes a CLI and capture the CLI binary name (e.g., `codehub`).
2. Pull the `route_map` digest from `.prefetch.md § route_map`. If absent, call `route_map({repo: "{{ repo }}"})` once and cache the digest in this packet's Work log. Each entry maps a subcommand to its handler `path:start_line`.
3. For each subcommand, `Read` the handler file at `path:start_line-start_line+40` and extract (a) the usage line as exposed to the CLI library, (b) the full flag list with each flag's source line. Reuse `signature({symbol: <handler-id>})` where the flag parser's signature captures the flag set.
4. If the CLI has more than 40 subcommands: group by top-level verb (`analyze`, `query`, `verdict`, …). Emit one H2 per verb group and nest subcommands as H3 within.
5. Draft one H2 per subcommand (or verb group). Format: `## <subcommand>`, then a fenced usage block (```` ``` ```` no language), a one-sentence description, a `` `path:LOC` `` citation, then a `Flags:` bullet list with each flag cited `` `path:LOC` ``.
6. Order subcommands by the order they appear in `route_map`; within a verb group, order alphabetically.
7. `Write {{ docs_root }}/reference/cli.md` with H1 = `{{ repo }} · CLI` and a one-sentence intro naming the CLI binary before the first H2.

## 5. Document format rules

- H1 = `{{ repo }} · CLI`. No decorative titles.
- No YAML frontmatter on the output file.
- A single one-sentence intro after the H1 names the CLI binary (e.g., ``The `codehub` CLI has the following subcommands.``).
- Each subcommand is an H2 (`## <subcommand>`), followed by:
  1. A fenced usage block with no language tag (e.g., ```` ``` ```` ... ```` ``` ````), showing the canonical invocation with optional flags in brackets.
  2. A one-sentence description.
  3. A `` `path:LOC` `` citation on its own line.
  4. A `Flags:` label followed by a bullet list; each bullet is `` `--flag` — description. `path:LOC`. ``.
- If > 40 subcommands: one H2 per verb group, one H3 per subcommand beneath. Otherwise flat H2s.
- No Mermaid. No emojis. No filler adverbs.

## 6. Tool usage guide

| Need | Tool | Why |
|---|---|---|
| CLI presence signal | `{{ context_path }} § Repo profile` (`project_profile.entry_points`) | precondition for this packet |
| Subcommand inventory | `mcp__codehub__route_map` (cached in `.prefetch.md`) | pre-parsed CLI tree; authoritative over handler grepping |
| Handler signature / flag parser | `mcp__codehub__signature` | captures the parameter shape without paraphrase |
| Flag source lines | `Read` at `path:start_line-start_line+40` | graph does not record per-flag line numbers |

## 7. Fallback paths

- If `project_profile.entry_points` does not include a CLI: the orchestrator should not have seeded this packet. Write the mismatch to the Work log, flip `status` to `BLOCKED`, and stop. Do not emit the output file.
- If `route_map` returns `[]`: log the empty inventory in the Work log and do not emit the file. The orchestrator will prune the empty packet in Phase E.
- If a subcommand handler has no extractable flags: emit the H2 with the usage block, description, and citation only; omit the `Flags:` bullet list rather than emit an empty one.
- If a handler file cannot be `Read` (missing or binary): emit the H2 with the usage block and description, mark the citation `*handler unavailable*`, and log the skip.
- If `route_map` is malformed (schema mismatch): run the `sql` fallback `SELECT name, file_path, start_line FROM nodes WHERE kind='Function' AND file_path LIKE '%cli%commands%'` and treat each hit as a candidate subcommand. Note the fallback in the Work log.

## 8. Success criteria

- [ ] `{{ docs_root }}/reference/cli.md` exists on disk.
- [ ] H1 line reads `# {{ repo }} · CLI`.
- [ ] The H1 is followed by a single one-sentence intro containing the CLI binary name in backticks.
- [ ] At least one H2 subcommand (or verb group) is present; count matches `route_map` entries (or verb-group count when grouping).
- [ ] Every H2 has exactly one fenced usage block immediately after the heading.
- [ ] Every H2 has exactly one `` `path:LOC` `` citation for the handler.
- [ ] Every `Flags:` bullet has a `` `path:LOC` `` citation (grep each bullet under a `Flags:` label for a backtick span).
- [ ] When subcommand count > 40: the output uses verb-group H2s with nested H3 subcommands. Otherwise: flat H2 subcommands.
- [ ] No YAML frontmatter on the output.
- [ ] No Mermaid fences in the output.

## 9. Anti-goals

- Do not re-call any MCP tool whose digest is in `.prefetch.md` — read the cached summary. If the cached slice is truncated, call with a narrower filter, not a blanket re-fetch.
- Do not invent subcommand names, flag names, or `path:LOC` citations — every identifier must come from `route_map`, `signature`, or a verified `Read` of the handler source.
- Do not paraphrase usage blocks or flag declarations. Quote them verbatim from source.
- Do not write to `{{ docs_root }}/reference/public-api.md`; that file is owned by the public-api packet.
- Do not emit the output file when `route_map` is empty or when the CLI precondition fails — flip `status` to `BLOCKED` and stop.
- Do not write YAML frontmatter on the output file.
- Do not emit emojis. Do not use filler adverbs.

---

## Work log

{{ subagent fills this section per the write protocol }}

## Validation

{{ checks run, outputs pasted, any fixes applied }}

## Summary

{{ one paragraph — what shipped, where, whether verb grouping was applied, and any route_map gaps encountered }}

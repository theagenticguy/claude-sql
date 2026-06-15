---
role: doc-reference-public-api
model: sonnet
output: "{{ docs_root }}/reference/public-api.md"
depends_on:
  - "{{ context_path }}"
  - "{{ prefetch_path }}"
status: IN_PROGRESS
---

# Packet · {{ repo }} · reference/public-api.md

## 1. Objective

Produce `{{ docs_root }}/reference/public-api.md`: one H3 per exported symbol for the top 30 public exports of `{{ repo }}`, each with a fenced code block that quotes the symbol's signature verbatim, a one-sentence description, and a `` `path:LOC` `` citation. When `{{ repo }}` is not a CLI, append an `## HTTP` H2 rendered from `route_map` at the bottom of the file.

## 2. Scope

- Create: `{{ docs_root }}/reference/public-api.md`
- Do not touch: any other file under `{{ docs_root }}/`, any source file in the repo, `.context.md`, `.prefetch.md`, or any `.packets/*.md` other than this one. In particular, do not write to `{{ docs_root }}/reference/cli.md` — the CLI packet owns that file.

## 3. Input specification

| Source | Read how | Cache state |
|---|---|---|
| Shared context | `Read {{ context_path }}` | always first |
| Prefetch ledger | `Read {{ prefetch_path }}` | always first |
| Project profile | `{{ context_path }} § Repo profile` | cached |
| Exported symbols (public surface) | `{{ prefetch_path }} § exports` or `mcp__codehub__sql({query: "SELECT name, kind, file_path, start_line FROM nodes WHERE kind IN ('Function','Class','Method') AND name NOT LIKE '\\_%' ORDER BY file_path LIMIT 500"})` | cached if digest present |
| Per-symbol signatures | `mcp__codehub__signature({symbol: <id>})` | mid-run (only if cache miss) |
| Per-symbol usage count | `mcp__codehub__context({symbol: <id>})` | mid-run (only if cache miss) |
| HTTP route inventory | `{{ prefetch_path }} § route_map` or `mcp__codehub__route_map({repo: "{{ repo }}"})` | cached if digest present |
| Source fallback for missing signatures | `Read <file>` at `start_line..start_line+20` | mid-run |

## 4. Process

1. `Read {{ context_path }}` and `Read {{ prefetch_path }}`. Confirm `{{ repo }}` profile; check `project_profile.entry_points` for the CLI flag.
2. Pull the exported-symbol list from `.prefetch.md § exports`. If absent, run the `sql` query in the input spec and cache the digest in this packet's Work log.
3. Filter to symbols whose file path is a barrel (`packages/*/src/index.ts`, `src/index.ts`, `mod.rs`, `__init__.py`, or equivalent). Keep that subset as the real exports.
4. For the top 30 exports (ordered by inbound relation count if cached, else by file path): fetch `signature({symbol: <id>})` and `context({symbol: <id>})`. Record signature, inbound count, and `path:start_line` in the Work log. Reuse cached digests wherever `.prefetch.md` already recorded them.
5. Determine HTTP rendering: if `project_profile.entry_points` includes a CLI, skip the HTTP section — the CLI packet owns route rendering. Otherwise, read the `route_map` digest (or call `route_map({repo: "{{ repo }}"})` once) and prepare an `## HTTP` H2.
6. Draft one H3 per export. Format: `### <symbol-name>`, then a fenced code block quoting the signature verbatim, then a one-sentence description, then the `` `path:LOC` `` citation on its own line.
7. If rendering HTTP, append `## HTTP` as the final H2, with one H3 per route (`### METHOD /path`), a one-sentence description, and a `` `path:LOC` `` citation. Order routes by `path`, then `method`.
8. `Write {{ docs_root }}/reference/public-api.md` with H1 = `{{ repo }} · Public API`.

## 5. Document format rules

- H1 = `{{ repo }} · Public API`. No decorative titles.
- No YAML frontmatter on the output file.
- Each exported symbol is an H3 (`### <symbol-name>`), followed by:
  1. One fenced code block quoting the signature verbatim from `signature` (or `Read` fallback). Never paraphrase.
  2. A one-sentence description.
  3. A `` `path:LOC` `` citation on its own line.
- The fenced code block's language tag matches the repo language (`ts`, `py`, `rs`, `go`, etc.) based on file extension.
- HTTP section (non-CLI repos only): one H2 `## HTTP`, one H3 per route `### <METHOD> <path>`, one-sentence description, `` `path:LOC` `` citation.
- No Mermaid. No emojis. No filler adverbs.

## 6. Tool usage guide

| Need | Tool | Why |
|---|---|---|
| Public-ish symbol surface | `sql` (cached in `.prefetch.md`) | Filter to non-underscore names grouped by barrel |
| Verbatim signatures | `mcp__codehub__signature` | authoritative signature text; never paraphrase |
| Usage count / publicness signal | `mcp__codehub__context` | inbound count orders the top-30 shortlist |
| HTTP route inventory | `mcp__codehub__route_map` | pre-parsed routes; avoids handler-file grepping |
| Signature fallback | `Read` at `path:start_line-start_line+20` | paste declaration verbatim when `signature` is empty |

## 7. Fallback paths

- If `signature` returns nothing for a symbol: `Read` the file at `path:start_line-start_line+20`, paste the declaration verbatim into the fenced code block, and note the fallback in the Work log.
- If fewer than 30 exports pass the barrel filter: emit whatever is present; do not pad with private helpers. Note the shortfall in the Work log.
- If `route_map` returns `[]` and `{{ repo }}` is not a CLI: skip the `## HTTP` section entirely rather than emit an empty one.
- If `{{ repo }}` is a CLI per `project_profile.entry_points`: do not emit the `## HTTP` section here — the `doc-reference-cli` packet renders `reference/cli.md` from `route_map` instead. Record the coordination choice in the Work log.
- If a barrel file is absent (language without explicit barrels): fall back to symbols whose `start_line == 1` AND whose file has ≥ 3 inbound import edges. Note the heuristic in the Work log.

## 8. Success criteria

- [ ] `{{ docs_root }}/reference/public-api.md` exists on disk.
- [ ] H1 line reads `# {{ repo }} · Public API`.
- [ ] At least 5 H3 entries are present (count `^### ` matches, excluding HTTP routes).
- [ ] Every H3 symbol block has one fenced code block immediately after the heading.
- [ ] Every H3 block has exactly one `` `path:LOC` `` citation.
- [ ] Every fenced signature block's content appears verbatim in either `signature`'s output or the cited source span (spot-check 3 by re-reading).
- [ ] If `project_profile.entry_points` includes CLI: no `## HTTP` section exists in the output.
- [ ] If `project_profile.entry_points` excludes CLI and `route_map` returned at least one route: an `## HTTP` section exists with one H3 per route.
- [ ] No YAML frontmatter on the output.
- [ ] No Mermaid fences in the output.

## 9. Anti-goals

- Do not re-call any MCP tool whose digest is in `.prefetch.md` — read the cached summary. If the cached slice is truncated, call with a narrower filter, not a blanket re-fetch.
- Do not invent symbol names, signatures, route paths, or citations — every identifier must come from a tool response or a `Read` of the source file.
- Do not paraphrase signatures. Quote them verbatim.
- Do not write to `{{ docs_root }}/reference/cli.md`; that file is owned by the CLI packet.
- Do not write YAML frontmatter on the output file.
- Do not emit emojis. Do not use filler adverbs.
- Do not emit more than 30 symbol H3 entries — overflow belongs to a future paginated packet.

---

## Work log

{{ subagent fills this section per the write protocol }}

## Validation

{{ checks run, outputs pasted, any fixes applied }}

## Summary

{{ one paragraph — what shipped, where, how the export shortlist was filtered, whether HTTP was rendered here or deferred to the CLI packet }}

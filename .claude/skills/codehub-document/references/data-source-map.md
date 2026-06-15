# data-source-map — which tools feed which subagent

Phase 0 precomputes a shared context from these sources. Subagents read the precompute from disk; they do not re-call tools whose digest is in `.prefetch.md`.

## `.context.md` (200-line cap)

```markdown
# Codehub context — <repo-or-group-name>
generated_at: <ISO-8601>
graph_hash: <from list_repos>

## Repo profile                      # from project_profile
- languages: TypeScript 87%, Rust 11%, Python 2%
- stacks: Node 22, pnpm 10, DuckDB, Vitest
- entry points: packages/mcp/src/index.ts, packages/cli/src/bin.ts

## Top communities (≤ 10)            # from sql: SELECT name, inferred_label, cohesion, symbol_count
                                      # FROM nodes WHERE kind='Community' ORDER BY cohesion DESC LIMIT 10
| name | inferred_label | cohesion | symbols |

## Top processes (≤ 10)              # from sql: SELECT name, entry_point, step_count
                                      # FROM nodes WHERE kind='Process' ORDER BY step_count DESC LIMIT 10
| name | entry_point | step_count |

## Routes                            # from route_map — truncated to 25 rows
| method | path | handler |

## MCP tools                         # from tool_map — truncated to 25 rows
| tool | summary |

## Owners summary                    # from owners on top 5 folders
| path | top_owner | share |

## Staleness envelope                # from list_repos._meta.codehub/staleness
- graph_hash: …
- indexed_at: …
- staleness_level: fresh | stale
```

**Group mode adds:**

```markdown
## Group manifest                    # from group_list
- group: <name>
- repos: [<list>]

## Group contracts matrix            # from group_contracts
| producer | consumer | count |

## Group freshness                   # from group_status
| repo | fresh | last_indexed |
```

## `.prefetch.md` (no cap, ledger)

Newline-delimited JSON. One line per tool call. Example:

```json
{"tool":"project_profile","args":{"repo":"opencodehub"},"sha256":"8c5f…","keys":["languages","stacks","entryPoints"],"cached_at":"2026-04-27T18:04:11Z","truncated":false}
{"tool":"tool_map","args":{"repo":"opencodehub"},"sha256":"1b9e…","keys":["tools"],"cached_at":"2026-04-27T18:04:12Z","truncated":true}
```

Subagents use the ledger two ways:

1. **Skip re-call.** If an agent would call a tool whose digest is here, it reads `.prefetch.md` + `.context.md` instead.
2. **Know when data is truncated.** Sections with `truncated: true` signal that raw tool output is larger than what's in `.context.md`. The agent may re-call the tool for a targeted slice if needed.

## Per-role input table

File-level fan-out means one role may seed multiple packets (for example, `doc-architecture` seeds `system-overview`, `module-map`, `data-flow`; `doc-diagrams` seeds `components`, `sequences`, `dependency-graph`). This table is indexed by role — the `codehub-document` orchestrator reads it when deciding which cached digests to mention in each packet's Input specification.

| Role               | Primary tools (Phase 0 cached)                              | Mid-run tools (not cached; agent may call)                |
|--------------------|-------------------------------------------------------------|-----------------------------------------------------------|
| `doc-architecture` | `project_profile`, `sql` (communities, processes)           | `context`, `query`, `dependencies`, `sql` for deeper joins |
| `doc-reference`    | `tool_map`, `route_map`, `project_profile`                  | `signature`, `context`, `sql` for export filtering        |
| `doc-behavior`     | `sql` (processes), `route_map`, `tool_map`                  | `context` per process, `query` to disambiguate names      |
| `doc-analysis`     | `owners`, `risk_trends`, `list_findings`, `list_dead_code`  | `verdict` (optional), `sql` for drill-down                |
| `doc-diagrams`     | `sql` (relations), `dependencies`                           | `context` per process, `query` for actor labels           |
| `doc-cross-repo`   | `group_list`, `group_status`, `group_contracts`             | `group_query`, `route_map` per member                     |

## Schema preflight (non-optional)

**Before composing any SQL query over `nodes`, `relations`, or any other
graph table, Phase 0 MUST probe the schema once and cache the result in
`.prefetch.md`.** Subagents then consult the cached schema instead of
guessing column names, which would fail with `Binder Error: Referenced
column "X" not found in FROM clause`.

The probe is one SQL call:

```
sql("SELECT table_name, column_name FROM information_schema.columns
     WHERE table_name IN ('nodes','relations') ORDER BY table_name, column_name")
```

Write the result as a dedicated `.context.md § Schema` subsection (top 30
rows, no cap) and as a digest line in `.prefetch.md` with
`keys: ["table_name","column_name"]`.

Historical note: `nodes` does not have a `path` column — routes store their
endpoint under `name` (as `"METHOD /path"`), and the file path is
`file_path`. Observed during a 2026-04-27 dogfood when subagent prompts
blindly referenced `path` and hit a Binder Error on an otherwise fresh
graph. The preflight prevents this class of bug across every subagent.

## Phase 0 algorithm (pseudocode)

Steps marked `# wave 0a` and `# wave 0b` each run as a single parallel tool-use batch — every line inside a wave issues concurrently in one message.

```
# wave 0a — independent precompute (one parallel batch)
1.  staleness = list_repos → entry for this repo → _meta.codehub/staleness
2.  profile = project_profile({repo})
3.  schema = sql("SELECT table_name, column_name FROM information_schema.columns …")
4.  routes = route_map({repo})
5.  tools = tool_map({repo})
6.  deps = dependencies({repo})
7.  risk = risk_trends({repo})
8.  dead = list_dead_code({repo})
9.  findings = list_findings({repo})
10. if --group: group_manifest = group_list
                group_freshness = group_status({group})
                group_contracts_matrix = group_contracts({group})
                // precondition check: every member fresh; abort otherwise

# wave 0b — depends on schema + profile (one parallel batch)
11. communities = sql("SELECT … FROM nodes WHERE kind='Community' …")
12. processes   = sql("SELECT … FROM nodes WHERE kind='Process' …")
13. relations   = sql("SELECT … FROM relations …")   # for diagrams
14. top_folders = top-5 folders by file count (from profile.entryPoints + glob)
15. owners_summary = [owners({path}) for path in top_folders]
16. if --group: group_hits = group_query({group, canonical_terms})

# wave 0c — inline deterministic post-processing (no MCP calls)
17. write .context.md (enforce 200-line cap; truncate per-section, mark flags)
18. write .prefetch.md (one JSON line per tool call with sha256 of response)
```

The algorithm is **deterministic** given the same `graph_hash` — the file list and section structure are identical across runs; only the exact content varies.

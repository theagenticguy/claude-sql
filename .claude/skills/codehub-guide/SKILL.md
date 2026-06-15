---
name: codehub-guide
description: "Use when the user asks about OpenCodeHub itself — available MCP tools, resources, graph schema, or workflow reference. Examples: \"What OpenCodeHub tools are available?\", \"How do I query the code graph?\", \"Show me the schema\"."
---

# OpenCodeHub Guide

Quick reference for every OpenCodeHub MCP tool, MCP resource, and the graph + temporal store schema.

## Always Start Here

For any task that touches code understanding, debugging, impact analysis, refactoring, or PR review:

1. Call `mcp__codehub__list_repos` — confirm the repo is indexed and pick a `repo` name.
2. Read `codehub://repo/{name}/context` — codebase stats and a staleness envelope.
3. Match the task to a skill below and follow that skill's checklist.

> If the context envelope reports the index is stale, run `codehub analyze` in the terminal first. If it says weights are missing, run `codehub setup --embeddings` to fetch the 768d gte-modernbert-base ONNX weights.

## Skills · analysis

| Task                                          | Skill to read                 |
| --------------------------------------------- | ----------------------------- |
| Understand architecture / "How does X work?"  | `codehub-exploring`       |
| Blast radius / "What breaks if I change X?"   | `codehub-impact-analysis` |
| Trace bugs / "Why is X failing?"              | `codehub-debugging`       |
| Plan a rename / extract / move (analysis only) | `codehub-refactoring`     |
| Review a PR / "Is this safe to merge?"        | `codehub-pr-review`       |
| Tools, resources, schema reference            | `codehub-guide` (here)    |

## Skills · artifact factory (spec 001)

These skills produce committed Markdown artifacts — the durable output
Principal engineers ship. See [ADR 0007](../../../docs/adr/0007-artifact-factory.md)
for the scope rationale.

| Task                                          | Skill to invoke               | Trigger phrase                                  |
| --------------------------------------------- | ----------------------------- | ----------------------------------------------- |
| Generate the full doc tree (single or group)  | `codehub-document`            | "document this repo", "regenerate architecture docs" |
| Draft a PR description from the current diff  | `codehub-pr-description`      | "write the PR description", "summarize this branch" |
| Write an onboarding guide with reading order  | `codehub-onboarding`          | "write ONBOARDING.md", "what should a new hire read first" |
| Map inter-repo contracts for a group          | `codehub-contract-map`        | "map the contracts", "show the contract matrix for <group>" |
| Build a deterministic 9-item code-pack BOM    | `codehub-code-pack`           | "pack this repo for an LLM", "deterministic code pack", "pack the platform group" |
| Draft an ADR (P1 — not yet shipped)           | `codehub-adr` *(P1 backlog)*  | —                                               |

Fire these directly; do not nest them inside analysis skills. Each is a
standalone artifact producer with its own preconditions and output path.

## Tool Inventory (28 MCP tools)

> Every tool is **read-only with respect to your source**. No MCP tool edits the working tree; planning and verification tools surface what to change, and you (or your editor) apply the edit.

### Code intelligence (per-repo)

| Tool                          | What it gives you                                                                 |
| ----------------------------- | --------------------------------------------------------------------------------- |
| `mcp__codehub__list_repos`        | Enumerate indexed repos on this machine                                   |
| `mcp__codehub__query`             | Hybrid BM25 + vector search over symbols, grouped by process              |
| `mcp__codehub__context`           | 360-degree symbol view + `confidenceBreakdown` + `cochanges` side-section |
| `mcp__codehub__impact`            | Blast radius with risk tier + `confidenceBreakdown`                       |
| `mcp__codehub__detect_changes`    | Map an uncommitted or committed diff to affected symbols and flows        |
| `mcp__codehub__sql`               | Read-only query: `sql` arg → temporal DuckDB (cochanges/summaries); `cypher` arg → lbug graph (5 s timeout) |
| `mcp__codehub__signature`         | Symbol declaration + stubbed members (class/interface header + method/property signatures, bodies elided) |

### HTTP / RPC surface

| Tool                          | What it gives you                                                                 |
| ----------------------------- | --------------------------------------------------------------------------------- |
| `mcp__codehub__route_map`         | HTTP route inventory (method, path, handler, middleware)                  |
| `mcp__codehub__tool_map`          | MCP tool inventory exported by this repo                                  |
| `mcp__codehub__shape_check`       | Producer/consumer response-shape mismatches                               |
| `mcp__codehub__api_impact`        | HTTP consumer chain + middleware + affected processes for one route       |

### Cross-repo (groups)

| Tool                          | What it gives you                                                                 |
| ----------------------------- | --------------------------------------------------------------------------------- |
| `mcp__codehub__group_list`        | Discover named repo groups                                                |
| `mcp__codehub__group_query`       | BM25 fan-out across a group with reciprocal-rank fusion                   |
| `mcp__codehub__group_status`      | Per-repo staleness + contract freshness for a group                       |
| `mcp__codehub__group_contracts`   | HTTP contract cross-links (consumer FETCHES edge → producer Route)        |
| `mcp__codehub__group_cross_repo_links` | Audit trail of every typed cross-repo edge, both endpoints `repo_uri`-qualified |
| `mcp__codehub__group_sync`        | Rebuild the cross-repo contract registry + link table after a re-index    |

### Supply-chain / PR review (OpenCodeHub differentiators)

| Tool                             | What it gives you                                                              |
| -------------------------------- | ------------------------------------------------------------------------------ |
| `mcp__codehub__verdict`              | 5-tier PR decision (`auto_merge` → `block`) with top drivers           |
| `mcp__codehub__scan`                 | Run Priority-1 scanners (openWorld — spawns child processes)           |
| `mcp__codehub__list_findings`        | Browse SARIF findings produced by `scan` or `ingest-sarif`             |
| `mcp__codehub__list_findings_delta`  | Diff latest scan vs. frozen baseline (new / fixed / unchanged / updated) |
| `mcp__codehub__list_dead_code`       | Unreferenced exported symbols (read-only listing — you delete them)    |
| `mcp__codehub__license_audit`        | Copyleft / unknown / proprietary tier check over dependencies          |
| `mcp__codehub__dependencies`         | External package list (ecosystem + version + manifest path)            |
| `mcp__codehub__owners`               | File/symbol ownership from CODEOWNERS + git blame signal               |
| `mcp__codehub__risk_trends`          | Per-community trend lines and 30-day projections                       |
| `mcp__codehub__project_profile`      | High-level repo summary (languages, stacks, entry points)              |
| `mcp__codehub__pack_codebase`        | Deterministic LLM-ready code-pack snapshot of the repo                 |

## Differentiators to surface in responses

- **`confidenceBreakdown`** — every `context` and `impact` response carries a `{confirmed, heuristic, unknown}` tally. `confirmed` means a SCIP indexer (scip-typescript / scip-python / scip-go / rust-analyzer / scip-java) has confirmed the edge at confidence ≥ 0.95. `heuristic` is tree-sitter or tier-1/tier-2 inference the SCIP oracle has not confirmed. `unknown` is demoted (≤ 0.2) — a SCIP-confirmed edge exists at the same triple and the heuristic carries a `+scip-unconfirmed` reason suffix. Report this breakdown when an agent is about to take a destructive action based on edges.
- **`cochanges` side-section** — `context` includes files historically co-edited with the target (by lift, from the dedicated `cochanges` table). This is a **git-history signal, not call dependencies** — label it that way when you report it. It is excluded from the graph hash.
- **`verdict` + `list_findings` + `list_findings_delta`** — PR review grade is deterministic, not opinion.
- **`license_audit`** + **`risk_trends`** + **`owners`** — first-class audit workflows without writing SQL.

## Resource Inventory

Lightweight reads for navigation (every URI uses the `codehub://` scheme):

| Resource                                       | Content                                     |
| ---------------------------------------------- | ------------------------------------------- |
| `codehub://repos`                              | Registry list: names, roots, graph hashes   |
| `codehub://repo/{name}/context`                | Stats + staleness envelope                  |
| `codehub://repo/{name}/schema`                 | Live node kinds / relation types for `sql`  |

> Cluster and process navigation resources (`codehub://repo/{name}/clusters`, `codehub://repo/{name}/processes`, etc.) are slated for a later wave. Until then, use the typed tools or Cypher (below) filtered to `kind = 'Community'` / `kind = 'Process'`.

## Where the graph lives (ADR 0016)

There are **two stores**, and they are queried differently:

- **Graph tier — `graph.lbug`** (ladybug, Cypher dialect). Holds nodes, edges,
  and embeddings. Query it via the typed tools (`query` / `context` / `impact` /
  `route_map` / …) or, for bespoke questions, **Cypher** via the MCP `sql`
  tool's `cypher` argument. There is NO `nodes` or `relations` SQL table.
- **Temporal tier — `temporal.duckdb`** (DuckDB SQL). Holds only the
  `cochanges` and `symbol_summaries` tables. The `sql` argument of the MCP
  `sql` tool (and `codehub sql` on the CLI) targets THIS store.

Pass exactly one of `sql` (temporal DuckDB) or `cypher` (lbug graph) to the MCP
`sql` tool.

### Graph schema (lbug / Cypher)

One node label `CodeNode` carrying `kind` as a **property** (NOT a per-kind
label). One relationship table per relation type. Properties are **snake_case**
(`file_path`, `start_line`, `inferred_label`, `step_count`, `entry_point_id`);
a camelCase RETURN alias comes back as the alias you give it, but the stored
property names are snake_case.

**Node kinds** (`n.kind` values): File, Folder, Function, Class, Method,
Interface, Constructor, Struct, Enum, Macro, Typedef, Union, Namespace, Trait,
Impl, TypeAlias, Const, Static, Variable, Property, Record, Delegate,
Annotation, Template, Module, CodeElement, Community, Process, Route, Tool,
Finding, Dependency, Contributor, Repo, ProjectProfile, Section.

**Relationship types** (each is its own edge label): CONTAINS, DEFINES, IMPORTS,
CALLS, EXTENDS, IMPLEMENTS, HAS_METHOD, HAS_PROPERTY, ACCESSES, METHOD_OVERRIDES,
OVERRIDES, METHOD_IMPLEMENTS, MEMBER_OF, PROCESS_STEP, HANDLES_ROUTE, FETCHES,
HANDLES_TOOL, ENTRY_POINT_OF, WRAPS, QUERIES, REFERENCES, FOUND_IN, DEPENDS_ON,
OWNED_BY.

Cochanges live only in the **temporal** `cochanges` table (DuckDB SQL), never as
graph edges.

## Cypher cheat-sheet (MCP `sql` tool, `cypher` arg)

All inbound callers of a function by name:

```cypher
MATCH (caller:CodeNode)-[r:CALLS]->(callee:CodeNode)
WHERE callee.name = 'validateUser' AND callee.kind = 'Function'
RETURN caller.name AS name, caller.file_path AS file, caller.start_line AS line,
       r.confidence AS confidence, r.reason AS reason
ORDER BY r.confidence DESC
LIMIT 50
```

Top communities by cohesion:

```cypher
MATCH (n:CodeNode)
WHERE n.kind = 'Community'
RETURN n.name AS name, n.inferred_label AS label, n.cohesion AS cohesion,
       n.symbol_count AS symbols
ORDER BY n.cohesion DESC
LIMIT 20
```

Process entry points:

```cypher
MATCH (n:CodeNode)
WHERE n.kind = 'Process'
RETURN n.name AS name, n.inferred_label AS label, n.step_count AS steps,
       n.entry_point_id AS entry_point
ORDER BY n.step_count DESC
```

SCIP-confirmed CALLS edges only (strict impact):

```cypher
MATCH ()-[r:CALLS]->()
WHERE r.confidence >= 0.95 AND r.reason STARTS WITH 'scip:'
RETURN r
```

### Temporal SQL cheat-sheet (MCP `sql` tool, `sql` arg)

Tightest co-change pairs (DuckDB SQL — temporal store):

```sql
SELECT source_file, target_file, lift, cocommit_count
FROM cochanges
ORDER BY lift DESC
LIMIT 20;
```

## Invariants agents must respect

- Every per-repo tool accepts an optional `repo` argument. When exactly one repo is indexed, `repo` is optional. When two or more are indexed and `repo` is omitted, the tool returns `AMBIGUOUS_REPO` — pass `repo` explicitly.
- Every response may carry `_meta.codehub/staleness` when the index is behind HEAD. Surface that to the user when it is present.
- Every response includes a `next_steps` array under `structuredContent`. Use it to pick the next tool without guessing.
- No MCP tool edits user source. Tools like `impact`, `context`, and `detect_changes` tell you what a change touches; you (or your editor) make the edit, then re-run `detect_changes` to verify.
- `scan` has `openWorldHint: true` — it spawns child processes. Do not invoke it on every turn.

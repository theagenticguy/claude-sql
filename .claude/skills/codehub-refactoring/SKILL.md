---
name: codehub-refactoring
description: "Use when the user is planning a rename, extraction, split, move, or restructure and wants to know the blast radius and verify the result. Examples: \"What breaks if I rename this function?\", \"Map every caller before I extract this module.\", \"Did my refactor touch anything I didn't expect?\". OpenCodeHub does NOT edit source — it plans and verifies; you (or your editor) apply the edits."
---

# Refactoring support with OpenCodeHub

OpenCodeHub's MCP surface is **read-only with respect to your source** — no
tool edits the working tree. This skill uses the graph to **plan** a refactor
(map the blast radius and every reference before you touch anything) and to
**verify** it afterward (confirm the diff matches the plan and nothing
unexpected moved). The actual edits are made by you or your editor; OpenCodeHub
tells you exactly what to change and then checks your work.

## When to Use

- "What will break if I rename `validateUser`?"
- "Map every caller before I extract this into a module."
- "Where are all the references to this symbol — including dynamic ones?"
- "I just refactored — did the change touch only what I expected?"
- Any rename, extract, split, or move where you need the dependency picture
  before editing and a verification pass after.

## Workflow

```
PLAN
1. mcp__codehub__impact({ name: target, direction: "upstream" })   → All dependents (the edit list)
2. mcp__codehub__context({ name: target })                         → Incoming / outgoing refs + processes
3. mcp__codehub__shape_check (if the symbol is a route/payload)    → Producer/consumer drift to watch
4. Build the edit plan from impact's d=1 set + context's refs

APPLY (you / your editor — OpenCodeHub does not do this)
5. Make the edits in your editor or with the language server's rename

VERIFY
6. mcp__codehub__detect_changes({ scope: "unstaged" })             → What the diff actually touched
7. Cross-check changed_symbols against the plan — any surprise is a miss
8. Run tests for the affected processes
```

> If the context envelope warns the index is stale, run `codehub analyze` first — a stale graph produces an incomplete plan and a misleading verification.

## Checklists

### Plan and verify a symbol rename

```
PLAN
- [ ] mcp__codehub__impact({ name, direction: "upstream" }) — enumerate every dependent; this is your edit list
- [ ] mcp__codehub__context({ name }) — see inbound + outbound refs, including ACCESSES edges
      (pass `file_path` and/or `kind` to disambiguate when the name is ambiguous)
- [ ] Note the confidenceBreakdown: `unknown > 0` means a heuristic edge the SCIP
      oracle contradicted — inspect those by reading source before trusting them
- [ ] Watch for dynamic references the graph cannot see (string-keyed dispatch,
      reflection, config JSON, doc comments) — list them as manual-check items

APPLY (your editor / LSP rename — not an OpenCodeHub tool)
- [ ] Apply the rename across the files impact + context named

VERIFY
- [ ] mcp__codehub__detect_changes({ scope: "unstaged" }) — confirm the diff scope
- [ ] Cross-check detect_changes against impact's d=1 count — a gap means a missed reference
- [ ] Run tests for every affected process
```

### Extract a module

```
- [ ] mcp__codehub__context({ name: target }) — see every external ref
- [ ] mcp__codehub__impact({ name: target, direction: "upstream" }) — callers outside the new module
- [ ] Define the new public surface (export only what external callers use)
- [ ] Move code; update imports (in your editor)
- [ ] mcp__codehub__detect_changes — verify scope
- [ ] Run tests for the affected processes
- [ ] Re-run codehub analyze so the next agent sees the new module boundary
```

### Split a function or service

```
- [ ] mcp__codehub__context({ name: target }) — understand outgoing calls
- [ ] Group outgoing calls by responsibility (the seams for the split)
- [ ] mcp__codehub__impact({ name: target, direction: "upstream" }) — map callers to update
- [ ] Create the new functions / services and update callers (in your editor)
- [ ] mcp__codehub__detect_changes — verify scope
- [ ] Run tests
```

## Tools

### `mcp__codehub__impact` — enumerate dependents before you edit

```
mcp__codehub__impact({
  name: "validateUser",
  direction: "upstream",
  depth: 2,
  repo: "my-app"
})

→ byDepth.d1: direct callers — every one needs updating
→ affected_processes: which execution flows the change rides
→ confidenceBreakdown: {confirmed, heuristic, unknown}
→ risk: LOW | MEDIUM | HIGH | CRITICAL
```

The d=1 set IS your rename edit list. If `unknown > 0`, the demote phase
contradicted a heuristic edge — that edge may not be a real call, so inspect it
before counting it.

### `mcp__codehub__context` — every inbound and outbound reference

```
mcp__codehub__context({ name: "validateUser", repo: "my-app" })

→ callers:   inbound CALLS / REFERENCES edges
→ callees:   outbound edges
→ accesses:  ACCESSES edges (field / property reads and writes)
→ processes: execution flows the symbol participates in
```

Use `context` to catch references `impact` does not surface as direct callers —
re-exports, shadowed locals, and ACCESSES edges. The combination of `impact`
(d=1) and `context` (all refs) is the complete edit list the graph can see.

### `mcp__codehub__detect_changes` — verify the post-refactor diff

```
mcp__codehub__detect_changes({ scope: "unstaged", repo: "my-app" })

→ changed_symbols: [...]
→ affected_processes: [...]
→ risk_level: LOW | MEDIUM | HIGH | CRITICAL
```

Always run this **after** you apply the edits. Any symbol you did not expect to
change is a miss; any planned symbol that does not appear was not edited.

### `mcp__codehub__shape_check` — catch contract drift on routes/payloads

When the refactor touches an HTTP route or a response payload, `shape_check`
flags producer/consumer mismatches so a response-shape change does not silently
break a downstream consumer.

```
mcp__codehub__shape_check({ route: "GET /users/:id", repo: "my-app" })

→ mismatches: [{ consumer, expected, actual }]
```

### `mcp__codehub__sql` — custom reference query (temporal store)

The `sql` arg is read-only DuckDB over the temporal store (cochanges +
symbol_summaries). To enumerate every file referencing a symbol from the graph,
use the `cypher` arg of the same tool instead (the node/edge graph lives in
`graph.lbug`, not the SQL store):

```cypher
MATCH (caller:CodeNode)-[r:REFERENCES|CALLS|IMPORTS]->(target:CodeNode)
WHERE target.name = 'validateUser'
RETURN DISTINCT caller.file_path AS file
ORDER BY file
```

This catches references a textual rename might miss — useful as a manual-check
list before and after you edit.

## Risk Rules

| Risk factor                       | Mitigation                                                              |
| --------------------------------- | ----------------------------------------------------------------------- |
| Many callers (> 5)                | Use your editor's LSP rename for the mechanical work; `impact` is the checklist |
| Cross-module references           | Run `detect_changes` after editing; watch for missed imports            |
| String / dynamic references       | Use the `cypher` arg with `REFERENCES`; the graph cannot see string-keyed dispatch — read those by hand |
| Public / exported API             | Version and deprecate; mirror symbol names in a transition layer        |
| Heuristic edges (confirmed = 0)   | Cross-check by reading source; the SCIP oracle did not weigh in         |

## Example: plan and verify renaming `validateUser` to `authenticateUser`

```
PLAN
1. mcp__codehub__impact({ name: "validateUser", direction: "upstream", repo: "my-app" })
   → d=1: loginHandler, apiMiddleware, tests/auth.test.ts
   → affected_processes: [LoginFlow, TokenRefresh]
   → confidenceBreakdown: {confirmed: 3, heuristic: 0, unknown: 0}

2. mcp__codehub__context({ name: "validateUser", repo: "my-app" })
   → callers confirm the same three sites
   → also surfaces a string reference in config/routes.json and a mention in docs/auth.md
     (dynamic refs the graph flags but cannot edit) — add both to the manual-check list

APPLY (in your editor — OpenCodeHub does not write files)
3. Use the language server's rename across loginHandler, apiMiddleware, and the
   test, then hand-edit config/routes.json and docs/auth.md.

VERIFY
4. mcp__codehub__detect_changes({ scope: "unstaged", repo: "my-app" })
   → changed_symbols: [authenticateUser, loginHandler, apiMiddleware, ...]
   → affected_processes: [LoginFlow, TokenRefresh]
   → risk_level: MEDIUM

5. Cross-check: every d=1 caller from step 1 appears in changed_symbols. No
   surprises. Run LoginFlow + TokenRefresh integration tests, then re-run
   codehub analyze so the graph picks up the new name.
```

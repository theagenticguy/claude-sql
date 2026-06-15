---
name: codehub-impact-analysis
description: "Use when the user wants to know what will break if they change something, or needs safety analysis before editing or merging code. Examples: \"Is it safe to change X?\", \"What depends on this?\", \"What will break?\", \"Blast radius for this change\"."
---

# Impact Analysis with OpenCodeHub

## When to Use

- "Is it safe to change this function?"
- "What will break if I modify X?"
- "Show me the blast radius."
- "Who uses this code?"
- Before committing or merging a non-trivial change.

## Decision Tree

```
Is the target a symbol (function, class, method, property)?
  └─ yes → mcp__codehub__impact
Is the target an HTTP route or API endpoint?
  └─ yes → mcp__codehub__api_impact + mcp__codehub__route_map
Is the change a response-shape edit on a route?
  └─ yes → mcp__codehub__shape_check (find consumer key-access mismatches)
Is the target a dependency version bump?
  └─ yes → mcp__codehub__dependencies + mcp__codehub__license_audit
Want to see what the working tree currently touches?
  └─ yes → mcp__codehub__detect_changes
```

## Workflow

```
1. mcp__codehub__impact({ name, direction: "upstream", repo })  → Dependents of the target
2. Read confidenceBreakdown                                          → Trust the confirmed count
3. If HTTP-adjacent: mcp__codehub__api_impact + shape_check      → Route + shape mismatches
4. mcp__codehub__detect_changes                                  → Map the current diff to flows
5. Assess risk tier and write the summary
```

> If the context envelope warns the index is stale, run `codehub analyze` first — stale impact results are worse than no impact results.

## Checklist

```
- [ ] mcp__codehub__impact({ name, direction: "upstream", repo })
- [ ] Review byDepth.d1 first — these WILL BREAK
- [ ] Read confidenceBreakdown; demand confirmed >= heuristic for destructive calls
- [ ] Filter to confidence >= 0.9 if the target is load-bearing (auth, payments, data integrity)
- [ ] If target is a Route: mcp__codehub__api_impact + mcp__codehub__shape_check
- [ ] mcp__codehub__detect_changes to map the current diff to affected processes
- [ ] Produce a risk tier and a one-paragraph summary
```

## Understanding impact output

Risk levels map to blast-radius tiers:

| Depth | Risk Level       | Meaning                                |
| ----- | ---------------- | -------------------------------------- |
| d=1   | WILL BREAK       | Direct callers / importers / overrides |
| d=2   | LIKELY AFFECTED  | One hop through d=1                    |
| d=3   | MAY NEED TESTING | Two hops — transitive effects          |

`confidenceBreakdown` on the impact response categorises the edges the tool actually traversed:

- `confirmed` — a SCIP indexer (scip-typescript, scip-python, scip-go, rust-analyzer, scip-java) confirmed the edge at confidence ≥ 0.95. Trust these for refactor/impact decisions.
- `heuristic` — tree-sitter or tier-1/tier-2 inference; no SCIP indexer covers this triple. Treat as a signal, not a ground truth.
- `unknown` — confidence ≤ 0.2. The demote phase flagged the edge (`+scip-unconfirmed`). Do not act on these alone.

## Risk Tier Guide

| Signal                                         | Risk     |
| ---------------------------------------------- | -------- |
| < 5 symbols, ≤ 1 process, all confirmed        | LOW      |
| 5–15 symbols, 2–5 processes                    | MEDIUM   |
| > 15 symbols OR many processes OR many heuristic edges | HIGH |
| Critical path (auth, payments, data integrity) | CRITICAL |

## Tools

### `mcp__codehub__impact` — symbol blast radius

```
mcp__codehub__impact({
  name: "validateUser",
  direction: "upstream",
  depth: 3,
  repo: "my-app"
})

→ target: {uid, kind, filePath}
→ byDepth: {d1: [...], d2: [...], d3: [...]}
→ affected_processes: [CheckoutFlow, LoginFlow]
→ confidenceBreakdown: {confirmed, heuristic, unknown}
→ risk: LOW | MEDIUM | HIGH | CRITICAL
```

Disambiguation: if the name is ambiguous, `impact` returns a ranked candidate list; pass `uid` (preferred) or `{name, file_path, kind}` to pick one.

### `mcp__codehub__api_impact` — route blast radius

```
mcp__codehub__api_impact({ method: "POST", path: "/api/payments", repo })

→ consumers: FETCHES callers across this repo (and across repos when a group is defined)
→ middleware: applied handlers
→ mismatches: producer/consumer shape mismatches
→ affected_processes: flows that pass through this route
```

### `mcp__codehub__shape_check` — response-shape sanity

```
mcp__codehub__shape_check({ repo })

→ mismatches: [{route, producer_keys, consumer_access, consumer_file}]
```

Run it when a PR changes a response payload. Any new entry in `mismatches` is a bug surface.

### `mcp__codehub__detect_changes` — map the current diff to flows

```
mcp__codehub__detect_changes({ scope: "staged", repo })

→ changed_symbols: [{uid, name, kind, filePath, change}]
→ affected_processes: [...]
→ risk_level: LOW | MEDIUM | HIGH | CRITICAL
```

Scopes: `unstaged`, `staged`, `all`, `compare` (requires `base_ref`).

## Example: "What breaks if I change `validateUser`?"

```
1. mcp__codehub__impact({ name: "validateUser", direction: "upstream", depth: 3, repo: "my-app" })
   → byDepth.d1: loginHandler, apiMiddleware (WILL BREAK)
   → byDepth.d2: authRouter, sessionManager (LIKELY AFFECTED)
   → affected_processes: [LoginFlow, TokenRefresh]
   → confidenceBreakdown: {confirmed: 4, heuristic: 0, unknown: 0}
   → risk: MEDIUM

2. Every d=1 edge is LSP-confirmed — high trust. Two processes touch the target.

3. mcp__codehub__detect_changes({ scope: "unstaged", repo: "my-app" })
   → changed_symbols: [validateUser]
   → affected_processes: [LoginFlow, TokenRefresh]

4. Verdict: MEDIUM risk. LoginFlow and TokenRefresh need regression tests before merging.
```

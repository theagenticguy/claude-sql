---
name: codehub-exploring
description: "Use when the user asks how code works, wants to understand architecture, trace execution flows, or explore unfamiliar parts of a codebase. Examples: \"How does X work?\", \"What calls this function?\", \"Show me the auth flow\", \"Where does the request enter the system?\"."
---

# Exploring Codebases with OpenCodeHub

## When to Use

- "How does authentication work?"
- "What's the project structure?"
- "Where is the database logic?"
- "Show me the request lifecycle."
- Onboarding to a codebase you have not seen before.

## Workflow

```
1. mcp__codehub__list_repos                              → Pick the repo name
2. READ codehub://repo/{name}/context                        → Stats + staleness envelope
3. mcp__codehub__query({query: "<concept>", repo})       → Hybrid BM25 + vector hits, grouped by process
4. mcp__codehub__context({name: "<symbol>", repo})       → 360-degree view, categorised edges, cochange side-section
5. mcp__codehub__sql for process traces                  → Step ordering from the relations table
6. Read source files only after the graph has pointed you at the hot spots
```

> If the context envelope warns the index is stale, run `codehub analyze` in the terminal first.

## Checklist

```
- [ ] mcp__codehub__list_repos — confirm the repo
- [ ] READ codehub://repo/{name}/context — check freshness
- [ ] mcp__codehub__query for the concept
- [ ] Scan returned processes (execution flows) before scanning files
- [ ] mcp__codehub__context on the strongest hit
- [ ] Inspect confidenceBreakdown — prefer confirmed edges for architectural claims
- [ ] Review cochanges as a secondary signal (git history, NOT call deps)
- [ ] Follow PROCESS_STEP edges via mcp__codehub__sql to trace the flow end-to-end
- [ ] Read source files for implementation detail only
```

## Tools

### `mcp__codehub__query` — hybrid BM25 + vector search, grouped by process

```
mcp__codehub__query({ query: "payment processing", repo: "my-app", limit: 20 })

→ results: ranked symbols with file path, kind, score
→ processes: execution flows that contain the top hits (populated when PROCESS_STEP edges exist)
→ next_steps: pointer to the next tool
```

Use it as the first broad read after the context resource. Prefer it over Grep for concept-level questions.

### `mcp__codehub__context` — 360-degree view of a symbol

```
mcp__codehub__context({ name: "validateUser", repo: "my-app" })

→ target: {uid, kind, filePath, startLine}
→ incoming: categorised by edge type (CALLS, REFERENCES, IMPORTS, METHOD_OVERRIDES, ...)
→ outgoing: same, in the other direction
→ processes: [{process, step}]
→ cochanges: files historically edited with target's file (git history, NOT call deps)
→ confidenceBreakdown: {confirmed, heuristic, unknown}
```

Always read `confidenceBreakdown` before making an architectural claim. If `confirmed` is zero and `unknown` is non-zero, the edges are contradicted by the LSP oracle — report that caveat.

When a name is ambiguous, `context` returns a ranked candidate list instead of silently picking one. Disambiguate by passing `uid` (preferred) or `{name, file_path, kind}`.

### `mcp__codehub__sql` — trace a named process end-to-end

```sql
SELECT r.step, callee.name, callee.file_path, callee.start_line
FROM relations r
JOIN nodes proc   ON proc.id = r.from_id
JOIN nodes callee ON callee.id = r.to_id
WHERE r.type = 'PROCESS_STEP'
  AND proc.kind = 'Process'
  AND proc.name = 'CheckoutFlow'
ORDER BY r.step ASC;
```

## Cross-repo exploration

When the concept crosses repo boundaries (e.g. a microservice calls another), use the group tools:

- `mcp__codehub__group_list` — discover named groups.
- `mcp__codehub__group_query({ group, query, limit })` — BM25 fan-out across every repo in the group, results tagged with `_repo`.
- `mcp__codehub__group_contracts({ group })` — HTTP contract cross-links (consumer `FETCHES` edge → producer `Route`).

## Example: "How does payment processing work?"

```
1. mcp__codehub__list_repos
   → [{name: "my-app", root: "/Users/.../my-app", nodes: 918}]

2. READ codehub://repo/my-app/context
   → 918 symbols, 45 processes, index fresh.

3. mcp__codehub__query({ query: "payment processing", repo: "my-app" })
   → results: processPayment (Function, src/payments/processor.ts),
              chargeStripe, validateCard, handleRefund
   → processes: CheckoutFlow (step_count=7), RefundFlow (step_count=5)

4. mcp__codehub__context({ name: "processPayment", repo: "my-app" })
   → incoming.CALLS: [checkoutHandler, webhookHandler]
   → outgoing.CALLS: [validateCard, chargeStripe, saveTransaction]
   → processes: [{process: "CheckoutFlow", step: 3}]
   → confidenceBreakdown: {confirmed: 4, heuristic: 1, unknown: 0}
   → cochanges: [src/payments/refund.ts (lift 4.2), src/webhooks/stripe.ts (lift 3.1)]
                (git history signal — files often edited together, NOT call deps.)

5. mcp__codehub__sql to dump the ordered CheckoutFlow.

6. Read src/payments/processor.ts for the actual implementation.
```

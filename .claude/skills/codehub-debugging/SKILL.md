---
name: codehub-debugging
description: "Use when the user is debugging a bug, tracing an error, or asking why something fails. Examples: \"Why is X failing?\", \"Where does this error come from?\", \"Trace this bug\", \"This endpoint returns 500\"."
---

# Debugging with OpenCodeHub

## When to Use

- "Why is this function failing?"
- "Trace where this error comes from."
- "Who calls this method?"
- "This endpoint returns 500 intermittently."
- Investigating bugs, errors, test failures, or unexpected behaviour.

## Workflow

```
1. mcp__codehub__detect_changes                              → Is this a recent regression?
2. mcp__codehub__query({ query: "<error or symptom>" })      → Find code related to the symptom
3. mcp__codehub__context({ name: "<suspect>" })              → Callers, callees, processes, cochanges
4. mcp__codehub__sql for a process trace                     → Ordered PROCESS_STEP chain
5. mcp__codehub__list_findings                               → SARIF findings that already flagged the area
6. Read source files only after the graph has narrowed the suspect set
```

> If the context envelope warns the index is stale, run `codehub analyze` first — stale graphs point at the wrong suspect.

## Checklist

```
- [ ] Capture the symptom (error text, stack frame, failing assertion, wrong output)
- [ ] mcp__codehub__detect_changes({ scope: "unstaged" | "staged" })
      — if non-empty, recent edits are the primary suspect
- [ ] mcp__codehub__query for the error text or related concept
- [ ] mcp__codehub__context on the suspect symbol
- [ ] Read confidenceBreakdown — if unknown > 0, an LSP contradicted a heuristic edge;
      the suspect may be a stale or dead reference
- [ ] Review cochanges (git-history signal, NOT call deps) — files historically edited
      together often point at the hidden collaborator
- [ ] Trace the execution via mcp__codehub__sql on PROCESS_STEP edges
- [ ] mcp__codehub__list_findings to see if a scanner already flagged the area
- [ ] Read source to confirm root cause
```

## Debugging Patterns

| Symptom              | OpenCodeHub path                                                           |
| -------------------- | -------------------------------------------------------------------------- |
| Error message        | `query` for the error text → `context` on every throw site it returned     |
| Wrong return value   | `context` on the function → trace outgoing CALLS for the data flow         |
| Intermittent failure | `context` → look for outgoing FETCHES (external I/O) + cochanges           |
| Performance issue    | `impact` (downstream) → find symbols with many outbound CALLS (hot paths)  |
| Recent regression    | `detect_changes` → symbols touched in the working tree or staged diff      |
| Failing scanner gate | `list_findings({ severity: "error" })` → read the rule + the hit line      |

## Tools

### `mcp__codehub__query` — find code related to the error

```
mcp__codehub__query({ query: "payment validation error", repo })

→ results: validatePayment, handlePaymentError, PaymentException
→ processes: CheckoutFlow (populated when PROCESS_STEP edges exist)
```

### `mcp__codehub__context` — full context for a suspect

```
mcp__codehub__context({ name: "validatePayment", repo })

→ incoming.CALLS: [processCheckout, webhookHandler]
→ outgoing.CALLS: [verifyCard, fetchRates]          ← fetchRates is external I/O — likely culprit
→ outgoing.FETCHES: [POST https://api.stripe.com/...]
→ processes: [{process: "CheckoutFlow", step: 3}]
→ confidenceBreakdown: {confirmed: 3, heuristic: 1, unknown: 0}
→ cochanges: [src/payments/refund.ts (lift 4.2)]   (git history, NOT call deps)
```

The `cochanges` section frequently surfaces the hidden collaborator a heuristic call graph misses. Always label it as git-history signal so the user does not treat it as a dependency.

### `mcp__codehub__sql` — custom call-chain trace

Two-hop upstream trace for every caller of `validatePayment`:

```sql
WITH direct AS (
  SELECT from_id, to_id, 1 AS depth
  FROM relations
  WHERE type = 'CALLS'
    AND to_id IN (SELECT id FROM nodes WHERE name = 'validatePayment' AND kind = 'Function')
),
indirect AS (
  SELECT r.from_id, d.to_id, 2 AS depth
  FROM relations r
  JOIN direct d ON d.from_id = r.to_id
  WHERE r.type = 'CALLS'
)
SELECT caller.name, caller.file_path, caller.start_line, u.depth
FROM (SELECT * FROM direct UNION ALL SELECT * FROM indirect) u
JOIN nodes caller ON caller.id = u.from_id
ORDER BY u.depth ASC, caller.name;
```

### `mcp__codehub__list_findings` — scanner hits in the suspect area

```
mcp__codehub__list_findings({
  repo,
  severity: "error",
  path_prefix: "src/payments"
})

→ [{rule, severity, message, file, line, snippet}]
```

Use it early — a scanner often has the bug pre-labelled.

## Example: "Payment endpoint returns 500 intermittently"

```
1. mcp__codehub__detect_changes({ scope: "staged", repo: "my-app" })
   → empty. Not a recent regression.

2. mcp__codehub__query({ query: "payment error handling", repo: "my-app" })
   → results: validatePayment, handlePaymentError, PaymentException
   → processes: [CheckoutFlow (7 steps), ErrorHandling (4 steps)]

3. mcp__codehub__context({ name: "validatePayment", repo: "my-app" })
   → outgoing.CALLS: [verifyCard, fetchRates]
   → outgoing.FETCHES: [GET https://rates.example.com/v1/latest]   ← external I/O
   → confidenceBreakdown: {confirmed: 3, heuristic: 0, unknown: 0}
   → cochanges: [src/utils/http-client.ts (lift 5.1)]              ← investigate this file

4. mcp__codehub__list_findings({
     repo: "my-app",
     path_prefix: "src/utils/http-client"
   })
   → security-rule "missing-request-timeout" at src/utils/http-client.ts:42 (error)

5. Root cause: `fetchRates` uses a shared HTTP client with no timeout configured;
   Stripe rate endpoint occasionally hangs, caller returns 500.
```

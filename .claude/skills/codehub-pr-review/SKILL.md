---
name: codehub-pr-review
description: "Use when the user wants to review a pull request, understand what a PR changes, assess risk of merging, or check missing test coverage. Examples: \"Review this PR\", \"What does PR #42 change?\", \"Is this PR safe to merge?\", \"Audit the dependencies in this PR\"."
---

# PR Review with OpenCodeHub

## When to Use

- "Review this PR."
- "What does PR #42 change?"
- "Is this safe to merge?"
- "What's the blast radius of this PR?"
- "Are there missing tests for this PR?"
- "Did this PR introduce a copyleft / unknown license?"
- Reviewing someone else's code changes before merge.

## The Golden Workflow

```
1. mcp__codehub__verdict({ base, head })                      → 5-tier merge decision
2. mcp__codehub__list_findings_delta({ base })                → New / fixed / unchanged / updated findings
3. mcp__codehub__detect_changes({ scope: "compare", base_ref }) → Changed symbols + affected flows
4. For each non-trivial changed symbol:
   mcp__codehub__impact({ name, direction: "upstream" })     → Blast radius + confidenceBreakdown
5. mcp__codehub__license_audit                                → Copyleft / unknown / proprietary tiers
6. mcp__codehub__scan (opt-in)                                → Fresh scanner run — spawns processes
7. Write the review using the output template below
```

> If the context envelope warns the index is stale, run `codehub analyze` before starting — stale graphs produce stale verdicts.

## Checklist

```
- [ ] Fetch the PR diff (gh pr diff <n> or git diff <base>...<head>)
- [ ] mcp__codehub__verdict — start here; it aggregates the review signal
- [ ] Capture the verdict tier, top drivers, and blockers
- [ ] mcp__codehub__list_findings_delta — new findings since the baseline
- [ ] mcp__codehub__detect_changes — map the diff to affected processes
- [ ] mcp__codehub__impact on each non-trivial changed symbol
- [ ] Inspect confidenceBreakdown per impact — prefer confirmed edges for breakage claims
- [ ] mcp__codehub__license_audit — flag copyleft or unknown license changes
- [ ] (optional) mcp__codehub__scan to re-run scanners if the baseline is stale
- [ ] Write the review in the output template below
```

## Tools

### `mcp__codehub__verdict` — the starting point

```
mcp__codehub__verdict({ base: "main", head: "HEAD", repo: "my-app" })

→ tier: "auto_merge" | "single_review" | "dual_review" | "expert_review" | "block"
→ drivers: [{ signal, weight, evidence }]  // top reasons the tier was chosen
→ blockers: [...]                          // non-empty only for tier=block
→ next_action: "merge" | "request review from X" | "add tests for Y" | "fix finding Z"
→ exit_code: 0 | 1 | 2
```

Always lead your review with the tier. If it is `block`, do not recommend merge. If it is `auto_merge`, the rest of the review is confirmation, not discovery.

### `mcp__codehub__list_findings_delta` — what changed since baseline

```
mcp__codehub__list_findings_delta({
  repo: "my-app",
  base: "main"      // compare current scan output to the baseline frozen at base
})

→ new: [{rule, severity, file, line, message}]        // introduced by this PR — the scariest bucket
→ fixed: [...]                                        // removed by this PR — give credit
→ unchanged: [...]                                    // still present, not touched
→ updated: [...]                                      // same rule hit at a shifted location
```

The `new` bucket is the first thing to surface — it is the PR author's new debt.

### `mcp__codehub__detect_changes` — diff → flows

```
mcp__codehub__detect_changes({ scope: "compare", base_ref: "main", repo: "my-app" })

→ changed_symbols: [{uid, name, kind, filePath, change}]
→ affected_processes: [CheckoutFlow, RefundFlow]
→ risk_level: LOW | MEDIUM | HIGH | CRITICAL
```

### `mcp__codehub__impact` — blast radius per changed symbol

```
mcp__codehub__impact({
  name: "validatePayment",
  direction: "upstream",
  depth: 2,
  repo: "my-app"
})

→ byDepth.d1: processCheckout, webhookHandler        // WILL BREAK if signature changed
→ byDepth.d2: checkoutRouter                         // LIKELY AFFECTED
→ affected_processes: [CheckoutFlow]
→ confidenceBreakdown: {confirmed, heuristic, unknown}
→ risk: MEDIUM
```

If any d=1 caller is NOT in the PR diff, flag it as a potential breakage in your review.

### `mcp__codehub__license_audit` — dependency license tiers

```
mcp__codehub__license_audit({ repo: "my-app" })

→ by_tier: {
    copyleft: [{ name, ecosystem, version, license, manifest }],
    unknown:  [...],
    proprietary: [...],
    permissive: [...]
  }
→ warnings: [...]   // e.g. "package `foo` has no license field in manifest"
```

If the PR diff touches `package.json`, `pyproject.toml`, `go.mod`, or `Cargo.toml`, run this and compare tiers against the pre-PR baseline. A new `copyleft` or `unknown` entry is a review finding.

### `mcp__codehub__scan` — re-run scanners

Only run this when the baseline is obviously stale. `scan` has `openWorldHint: true` and spawns child processes, so use it deliberately.

```
mcp__codehub__scan({ repo: "my-app" })
```

### `mcp__codehub__risk_trends` — context on the area being changed

```
mcp__codehub__risk_trends({ repo: "my-app" })

→ communities: [{ name, risk_score, trend, projection_30d }]
```

Useful when a PR lands inside a community whose risk is already trending up — call that out in the review.

### `mcp__codehub__owners` — who should review?

```
mcp__codehub__owners({ repo: "my-app", path: "src/payments" })

→ [{ owner, source: "codeowners" | "git-blame", files, recent_edits }]
```

## Review Dimensions

| Dimension            | OpenCodeHub surface                                                       |
| -------------------- | ------------------------------------------------------------------------- |
| **Correctness**      | `context` shows callers — are they all compatible with the change?        |
| **Blast radius**     | `impact.byDepth` — anything at d=1 not in the diff is a potential miss    |
| **Completeness**     | `detect_changes.affected_processes` — are they all handled?               |
| **Confidence**       | `confidenceBreakdown.confirmed` vs `heuristic` — LSP-backed claims win    |
| **Net new bugs**     | `list_findings_delta.new` — introduced by this PR                         |
| **Tests**            | `impact` filtered to `kind = 'Function'` inside test files                |
| **License hygiene**  | `license_audit` before/after diff                                         |
| **Ownership**        | `owners` — right reviewers requested?                                     |
| **Trend**            | `risk_trends` — is this area already hot?                                 |

## Risk Tier Guide

| Signal                                                  | Risk     |
| ------------------------------------------------------- | -------- |
| < 3 symbols touched, 0–1 processes, no new findings     | LOW      |
| 3–10 symbols, 2–5 processes, ≤ 1 new finding            | MEDIUM   |
| > 10 symbols OR many processes OR several new findings  | HIGH     |
| Touches auth, payments, data integrity, or new copyleft | CRITICAL |
| d=1 callers exist outside the PR diff                   | Flag it  |

## Example: "Review PR #42"

```
1. gh pr diff 42 > /tmp/pr42.diff
   → 4 files changed: payments.ts, checkout.ts, types.ts, utils.ts

2. mcp__codehub__verdict({ base: "main", head: "HEAD", repo: "my-app" })
   → tier: "dual_review"
   → drivers: [
       {signal: "high-impact symbol changed", weight: 0.4, evidence: "validatePayment"},
       {signal: "new scanner finding", weight: 0.3, evidence: "security/no-eval"},
       {signal: "missing test coverage on CheckoutFlow", weight: 0.3}
     ]
   → next_action: "request review from @payments-team"

3. mcp__codehub__list_findings_delta({ repo: "my-app", base: "main" })
   → new: [{rule: "security/no-eval", severity: "error", file: "src/utils/format.ts", line: 44}]
   → fixed: []

4. mcp__codehub__detect_changes({ scope: "compare", base_ref: "main", repo: "my-app" })
   → changed_symbols: [validatePayment, PaymentInput, formatAmount]
   → affected_processes: [CheckoutFlow, RefundFlow]
   → risk_level: MEDIUM

5. mcp__codehub__impact({ name: "validatePayment", direction: "upstream", repo: "my-app" })
   → byDepth.d1: processCheckout, webhookHandler
   → webhookHandler is NOT in the PR diff — flag as potential breakage.
   → confidenceBreakdown: {confirmed: 2, heuristic: 0, unknown: 0}

6. mcp__codehub__impact({ name: "PaymentInput", direction: "upstream", repo: "my-app" })
   → byDepth.d1: validatePayment (in PR), createPayment (NOT in PR)
   → createPayment uses the old PaymentInput shape — breaking change.

7. mcp__codehub__license_audit({ repo: "my-app" })
   → No tier changes vs. main — clean.

8. Compose the review (template below).
```

## Review Output Template

```markdown
## PR Review: <title>

**Tier: dual_review**   **Risk: MEDIUM**

### Verdict drivers
- validatePayment blast radius crosses the PR boundary
- 1 new scanner finding: security/no-eval at src/utils/format.ts:44
- CheckoutFlow has no test coverage for the new branch

### Changes
- 3 symbols changed across 4 files
- 2 execution flows affected: CheckoutFlow, RefundFlow

### Findings
1. **[blocker]** `webhookHandler` (src/webhooks.ts:15) calls `validatePayment`
   but is NOT updated in this PR. New signature will throw at runtime.
2. **[blocker]** `createPayment` (src/payments/create.ts:22) uses the old
   `PaymentInput` shape. This change is breaking.
3. **[error]** New scanner finding: security/no-eval at src/utils/format.ts:44.
   `eval(userInput)` is unsafe.
4. **[ok]** `formatAmount` added optional param — backwards compatible.

### Missing coverage
- CheckoutFlow has no integration test for the new branch.
- No webhook test exercises validatePayment.

### Recommendation
REQUEST CHANGES — resolve the three blockers and add a CheckoutFlow
integration test before re-review.
```

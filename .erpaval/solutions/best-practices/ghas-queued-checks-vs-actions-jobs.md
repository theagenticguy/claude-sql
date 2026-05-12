---
title: GitHub Advanced Security check-runs duplicate Actions job names — they sit `queued` indefinitely
track: knowledge
category: best-practices
module: .github/workflows/
component: GitHub Actions + GitHub Advanced Security
severity: info
tags: [github-actions, ghas, code-scanning, sarif, branch-protection, merge-gates]
applies_when:
  - Branch-protected `main` with required status checks
  - You upload SARIF from CI scanners (Bandit, Semgrep, OSV, betterleaks, CodeQL)
  - `gh pr checks` shows duplicate names with one `pass` and one `pending`
pattern: |
  When a CI workflow uploads SARIF via `github/codeql-action/upload-sarif`,
  GitHub Advanced Security creates a check-run with the **same name as
  the scanner's GitHub Apps integration** — but distinct from the
  Actions job that ran the scan. So PR checks show two entries:

  ```
  bandit         pass     20s   .github/workflows/bandit.yml job
  Bandit         pending  0     GHAS app check-run (SARIF receipt)
  ```

  The capitalized `Bandit` / `Semgrep OSS` / `CodeQL` / `osv-scanner` /
  `betterleaks` rows are SARIF upload acknowledgements posted by the
  GHAS service. They sit `queued` indefinitely after a rebase or
  push because the GHAS service binds them to the *previous* commit's
  scanner run, not the new SHA. The corresponding GitHub Actions
  workflow on the new SHA passes cleanly.

  `gh pr view --json mergeStateStatus` reports `UNSTABLE` (mergeable +
  pending checks). `gh pr merge --squash` is rejected with "9 of 9
  required status checks are expected".

  Resolution: when the lowercase Actions job names all pass, the
  uppercase GHAS check-runs are safe to ignore. Use `--admin` to
  bypass:

  ```
  gh pr merge <N> --squash --delete-branch --admin
  ```
example_files:
  - .github/workflows/bandit.yml
  - .github/workflows/semgrep.yml
  - .github/workflows/leaks.yml
  - .github/workflows/osv.yml
  - .github/workflows/codeql.yml
---

# Why this matters

Across PRs #35, #36, #38, #39, #40 in this repo (one week), every
single PR ended with 4-5 GHAS check-runs in `queued` state forever.
The Actions jobs all passed. The merge gate would have blocked
indefinitely waiting on checks that were never going to complete.

Diagnostic test: query check-runs by app:

```python
gh api repos/<owner>/<repo>/commits/<SHA>/check-runs | jq \
  '.check_runs[] | "\(.app.name) | \(.name) | \(.status)"'
```

The `app.name == "GitHub Advanced Security"` rows are the duplicates;
their corresponding `app.name == "GitHub Actions"` rows are the real
test signal.

# When to NOT bypass

If the lowercase Actions row also fails or is missing entirely, the
SARIF upload didn't happen and there's a real CI bug. Don't `--admin`
through that — fix the workflow first.

If a *new* GHAS finding exists from the scanner run, it surfaces as a
PR comment from `github-advanced-security[bot]` (we saw three on PR
#37 for `BaseException` catches). Those are real signals — read and
address before merging, not after.

# Reproducibility

Live observed across PRs #35, #36, #38, #39, #40 on
`theagenticguy/claude-sql` between 2026-05-11 and 2026-05-12. The
pattern is consistent: rebase or push → 4-5 GHAS check-runs go
`queued` and never complete; their Actions counterparts pass.

# Pinned by

Process knowledge, not a regression test. The right tool for these
recurring "merge with admin when only GHAS-pending" decisions is a
slash command or `gh` extension that diffs job names by app, not a
test.

---
title: GitHub Actions `codeload.github.com` 429 throttling — rerun, don't diagnose
track: knowledge
category: best-practices
module: .github/workflows/
component: GitHub Actions runner action download
severity: info
tags: [github-actions, ci, throttling, transient, rerun]
applies_when:
  - A CI job fails with `Response status code does not indicate success: 429 (Too Many Requests)`
  - The 429 is fetching `codeload.github.com/<org>/<action>/tar.gz/<sha>`
  - The job hasn't run any project code yet — failure is in `Set up job` / step setup
pattern: |
  GitHub's `codeload.github.com` rate-limits action archive downloads.
  When many runners on the same repo concurrently start (e.g. a
  rebase fires CI on a fresh SHA), the codeload CDN returns
  `429 Too Many Requests` and the runner's 3-attempt retry exhausts.
  The job fails before any project code runs.

  Diagnostic signature in the failed log:

      ##[warning]Failed to download action 'https://codeload.github.com/...'.
      Error: Response status code does not indicate success: 429 (Too Many Requests).
      ##[warning]Back off N seconds before retry.
      ...
      ##[error]Failed to download archive '...' after 3 attempts.

  This is **transient infrastructure**, never a real failure.
  Rerun the failed jobs:

      gh run rerun <run-id> --failed

  No code change is necessary.

  Don't pin the action by SHA "to avoid this" — SHA pins use the same
  codeload endpoint. Don't switch to a different action. Just rerun.
example_files: []
---

# Why this matters

PRs #35 and #36 (dependabot bumps) each had 1-2 jobs fail on the
first push with 429s on `codeload.github.com/github/codeql-action/...`.
We diagnosed the failure (read the logs, identified the 429), reran,
got green. Total intervention: ~15 seconds per occurrence — but only
because we recognized the symptom immediately.

The trap on a first occurrence is to over-investigate: read the failed
log, look for project code paths the failure touched, file an issue
about action versions. The 429 is on archive *download*, not action
*execution* — there's nothing to fix locally.

# Auto-rerun candidate

Worth automating IF this happens often enough to be a recurring tax.
Shape would be:

- Trigger: `workflow_run: completed`, conclusion: `failure`
- Step: download the failed run's logs, grep for `codeload.github.com`
  and `429`
- If matched and step name is "Set up job": `gh run rerun <id> --failed`

Capped at one auto-rerun per failure (avoid retry storms on a
genuinely-broken codeload).

Not automated yet for this repo — frequency was 2 occurrences across
6 PRs in one week, which is borderline. Revisit if it crosses ~20%
of PR runs.

# Pinned by

Process knowledge. Test would require mocking the runner's action
download path — disproportionate effort for a transient issue.

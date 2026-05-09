---
title: SARIF uploads need a unique `category:` per scanner — GitHub stopped merging same-tool runs in July 2025
track: knowledge
category: best-practices
module: .github/workflows/*.yml
component: github/codeql-action/upload-sarif
severity: warning
tags: [sarif, github-actions, code-scanning, upload-sarif, ci]
applies_when:
  - "Adding a new SARIF-emitting scanner workflow alongside an existing one"
  - "Two scanners' SARIF uploads silently collide and only one set of findings appears in code scanning"
pattern: |
  Since **2025-07-22**, GitHub no longer merges multiple SARIF runs
  uploaded under the same `tool` + `category` combination on the same
  commit. The newer upload silently *replaces* the older one in the
  code-scanning UI, so findings disappear without an error.

  **Rule: one unique `category:` per scanner per matrix leg.** If your
  matrix has multiple legs (e.g. `language: [python, go]`), each leg
  needs its own category too — `category: bandit-${{ matrix.language }}`
  is the canonical shape.

  Categories we use today (as of 2026-05-09):

      .github/workflows/bandit.yml      → category: bandit
      .github/workflows/leaks.yml       → category: betterleaks
      .github/workflows/semgrep.yml     → category: semgrep
      .github/workflows/osv.yml         → category: osv-scanner
      .github/workflows/codeql.yml      → category: <implicit per language>
      .github/workflows/scorecard.yml   → category: <implicit>

  **Bumps required when adding a new scanner:**
    1. Pick a category string that doesn't collide with any existing
       workflow (search `.github/workflows/*.yml` for `category:`).
    2. Use it on `github/codeql-action/upload-sarif@v4` (v4 is current
       since 2025-10-07; v3 deprecates December 2026).
    3. Verify in code scanning UI after the first run that the
       findings appear under the new category — if they're missing,
       you collided.

  **Don't merge two SARIFs into one upload to "save a step."** The pre-
  2025-07 workaround was to write a small JSON merge script that ran
  multiple scanners and combined their SARIF runs into one upload.
  That breaks under the new rules — only one of the inner runs
  surfaces. Each scanner gets its own upload-sarif step now.
example_files:
  - .github/workflows/bandit.yml
  - .github/workflows/leaks.yml
  - .github/workflows/semgrep.yml
  - .github/workflows/osv.yml
  - .erpaval/sessions/session-2293a5/research-security.yaml  # research that surfaced this
---

# Why this matters

Easy to miss because there's no error — the upload step succeeds, but
the second scanner's findings silently overwrite the first's in the
code-scanning UI. A workflow that "looks fine" (green CI, SARIF
uploaded, no error message) is hiding half its security signal.

The change date is precise: 2025-07-22 per the GitHub changelog
post (https://github.blog/changelog/2025-07-21-code-scanning-will-stop-combining-multiple-sarif-runs-uploaded-in-the-same-sarif-file/).
Anything authored before that date may have used the old merge
behavior; anything after needs unique categories. claude-sql's
existing workflows (semgrep.yml, osv.yml, codeql.yml — all from the
2026-05-08 ci-hardening session) already use unique categories, so we
were on the right side of the change without knowing why.

The bandit + leaks additions in the 2026-05-09 security-hardening
session followed the same pattern by virtue of the existing
precedent — but if a future session adds (e.g.) a second Python SAST
under the same `category: bandit`, it will silently lose findings.
The lesson exists to prevent that.

# Example

```yaml
# .github/workflows/bandit.yml — uses unique category
- uses: github/codeql-action/upload-sarif@v4
  if: always()
  with:
    sarif_file: bandit.sarif
    category: bandit              # unique across all workflows in this repo

# .github/workflows/leaks.yml — same upload step, different category
- uses: github/codeql-action/upload-sarif@v4
  if: always()
  with:
    sarif_file: betterleaks.sarif
    category: betterleaks         # unique; never reuse "leaks" since semgrep
                                  # might emit a "leaks" rule someday
```

```yaml
# Anti-pattern (won't work post-2025-07-22):
# Two scanners, two upload-sarif steps, same category — second silently wins.
- uses: github/codeql-action/upload-sarif@v4
  with:
    sarif_file: scanner-a.sarif
    category: security             # same as below
- uses: github/codeql-action/upload-sarif@v4
  with:
    sarif_file: scanner-b.sarif
    category: security             # collides → scanner-a's findings disappear
```

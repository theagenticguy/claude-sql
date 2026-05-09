---
title: Branch protection + workflow filters — PR retarget alone does not trigger CI
track: knowledge
category: best-practices
module: .github/workflows/*.yml
component: GitHub Actions branch filters
severity: info
tags: [github-actions, branch-protection, ci-triggers, stacked-prs]
applies_when:
  - "Repository has branch protection requiring PR (no direct push to main)"
  - "CI workflows filter triggers via `pull_request: branches: [main]`"
  - "You stack PRs (PR-A targets main, PR-B targets PR-A's branch) and later retarget to main"
pattern: |
  Two compounding gotchas to remember:

  1. **Branch-protection on main rejects direct push.** The strategy commit (or any
     "scaffold" commit) cannot land on main directly even from the user with admin rights —
     it must go through a PR. Solution: move the commit to its own branch, push, open PR.
     Reset main to origin/main locally to clear the divergence.

  2. **Stacked PRs do not trigger CI when the base is a feature branch.** The repo's
     workflows filter on `pull_request: branches: [main]`. A PR with `base=feat/foo, head=feat/bar`
     fires no checks at all (`gh pr checks` returns "no checks reported"). The PR is "mergeable"
     because there's nothing to gate it.

     Two-step fix:
       a. `gh pr edit <num> --base main` — retarget the PR to main
       b. The retarget alone does NOT trigger workflows; you also need a `synchronize` event.
          Push an empty commit:
              git commit --allow-empty -m "ci: trigger CI on retarget to main"
              git push origin <branch>
          That fires the `pull_request.synchronize` event and CI runs.

  Stack disciplines:
    - Open stacked PRs with `--base <feature-branch>` for diff visibility during review,
      but retarget to main + trigger sync BEFORE merging
    - The dependency-order merge is unchanged: parent feature branch first, then dependent
example_files:
  - .github/workflows/ci.yml         # the `branches: [main]` filter
  - .erpaval/sessions/session-2293a5/validation.yaml  # session that hit this
---

# Why this matters

Stacked PRs are the cleanest pattern when one feature depends on another (we used it for
`feat/pr-review-sheet` stacked on `feat/transcript-binding` — A2 needed A1's `binding.py`).
But if your workflows filter on `branches: [main]`, the stacked PR shows zero checks —
which can read as "all green" in a casual PR list view but is actually "untested."

The session-2293a5 fix: open with `--base feat/transcript-binding` for diff cleanliness,
then `gh pr edit --base main` + an empty `ci: trigger CI on retarget to main` commit to
fire the synchronize event.

# Example

```bash
# Open stacked PR (diff stays clean — only A2's changes show)
gh pr create --base feat/transcript-binding --head feat/pr-review-sheet --title "..." --body "..."

# Later, before requesting review or merging, retarget + trigger CI:
gh pr edit <num> --base main
git checkout feat/pr-review-sheet
git commit --allow-empty -m "ci: trigger CI on retarget to main"
git push origin feat/pr-review-sheet
# Wait ~60s, then `gh pr checks <num>` shows the full CI suite
```

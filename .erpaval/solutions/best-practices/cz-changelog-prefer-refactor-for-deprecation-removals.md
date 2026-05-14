---
title: `cz bump` hides `chore` commits from CHANGELOG — use `refactor` for deprecation-removal commits
track: knowledge
category: best-practices
module: CLAUDE.md / commitizen
component: commitizen + conventional-commits
severity: warning
tags: [release, commitizen, cz-bump, conventional-commits, changelog, deprecation, claude-sql]
applies_when:
  - "Cutting a release driven by `cz bump` with the default commitizen changelog filter"
  - "The release closes out a v0.x deprecation window — code being deleted because the comment said 'removed in next minor / next release'"
  - "You want users to see the removal in the auto-generated CHANGELOG.md without hand-editing"
pattern: |
  `cz bump` generates the changelog by category from the commit-type
  prefix. The default filter surfaces `feat`, `fix`, and `refactor`,
  but **excludes `chore`, `style`, `docs`, `ci`, `build`, `test`,
  `perf`, `revert`** unless `[tool.commitizen]` is reconfigured.

  When deleting a deprecated API the natural commit type is `chore`
  (it's not a bug fix; it's not a feature). But `chore` is hidden,
  so users reading CHANGELOG.md never see that the deprecation was
  completed. The dry-run is the gate:

  ```
  mise run bump:dry-run
  # If the changelog preview is missing your deletion commits,
  # you have a visibility gap.
  ```

  **Pick `refactor:` for deprecation-removal commits.** It's accurate
  (deleting code IS a refactor of the API surface), it shows up in
  the auto changelog, and it stays at PATCH increment so a single
  cleanup-release stays at X.Y.Z+1.

  Example, this session (v1.0.1):
  - `chore(views): drop deprecated task_spawns view` ← invisible
  - `chore(config): drop Settings.concurrency` ← invisible
  - `chore(views): drop describe_all` ← invisible
  - All three were rebased to `refactor(views|config): ...` and
    surfaced in the v1.0.1 CHANGELOG entry.

  Recovery: `git rebase -i <base>` and reword. The rebase is safe on
  a release branch never pushed to main; squash-merge happens after
  the reword so origin/main only sees the merge commit.

  **Counter-pattern:** if the deprecated API was never user-visible
  (private helper, dev-only flag), `chore:` is fine — there's nothing
  to surface. The `refactor:` discipline applies only when the user
  needs to know the API surface changed.
example_files:
  - .erpaval/solutions/best-practices/cz-bump-via-pr-with-branch-protection.md
  - .erpaval/solutions/conventions/changelog-write-only-via-cz-bump.md
  - CHANGELOG.md (v1.0.1 section)
counter_examples:
  - "Internal-only refactor with no user-facing API change → `refactor:` is correct but the changelog entry is a no-op for users."
  - "Bug fix in deprecated code → `fix(deprecated-area): ...`; the deprecation status is irrelevant to the type."
references:
  - "commitizen change_type defaults: https://commitizen-tools.github.io/commitizen/changelog/#available-options"
  - "PR #55 → v1.0.1 — the session that surfaced this"
---

The session that produced this lesson hit it during v1.0.1: three
`chore(views|config): drop deprecated …` commits did not appear in
the cz-generated `## v1.0.1` section. Rebased to `refactor:` and the
changelog populated. Cost of the rebase: one minute. Cost of missing
it: a release where users can't see that `task_spawns`,
`Settings.concurrency`, and `describe_all` were removed.

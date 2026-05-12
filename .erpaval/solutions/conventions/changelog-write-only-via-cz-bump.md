---
title: CHANGELOG.md is write-only via `cz bump` — no drift gate, no `## Unreleased`, no manual edits
track: knowledge
category: conventions
module: CHANGELOG.md
component: commitizen + GitHub squash-merge
severity: info
tags: [changelog, commitizen, ci, drift-gate, conventional-commits, release, squash-merge]
applies_when:
  - You use `cz bump` for releases and `cz changelog` would otherwise re-render `CHANGELOG.md`
  - GitHub squash-merges multi-commit PRs (default for branch-protected `main`)
  - You're tempted to add a CI gate that asserts `CHANGELOG.md` matches `cz changelog`'s projection
supersedes:
  - .erpaval/solutions/conventions/changelog-drift-gate-cz-overwrites-handwritten.md
pattern: |
  **Invariant**: `CHANGELOG.md` mutates exactly twice per release:

  1. `cz bump` writes the new `## vX.Y.Z` block on the release branch.
  2. GitHub squash-merge appends `(#NNN)` to the merged commit's bullet
     line.

  Nothing else writes to it. No `## Unreleased` placeholder, no manual
  edits, no `cz changelog` between releases, no CI drift gate.

  Between releases, query commit history directly:
  - `mise run bump:dry-run` — preview the next version + the bullets
    `cz bump` would write.
  - `git log --oneline v0.7.0..HEAD` — raw conventional-commit list.
pattern_continued: |
  **Why no drift gate**: a structural byte-diff between `CHANGELOG.md`
  and `cz changelog --incremental`'s re-render conflates three things:

  1. **Squash-merge `(#NNN)` injection.** Multi-commit PR with
     `feat: A`, `fix: B`, `docs: C` collapses to one commit `feat: A
     (#NNN)` on main. cz regenerates one bullet; committed file has
     three. False-positive drift; blocks every subsequent PR.

  2. **Empty `## Unreleased` placeholder.** After `cz bump`, the next
     non-bump-eligible commit (a `ci:` or `docs:`) makes
     `cz --incremental` inject a hollow `## Unreleased` header. False-
     positive drift; blocks the very next PR.

  3. **Real content drift** — the only catch worth firing. Vanishingly
     rare; not worth (1) and (2)'s ongoing tax.
example_files:
  - .github/workflows/ci.yml
  - CLAUDE.md
---

# Why this matters

In a single week the drift gate produced four false-positive failures
that required manual intervention:

- PR #37 squash-merged → PR #35 + #36 blocked on (1) — fixed via manual
  follow-up PR #38 that ran `cz changelog` on main.
- PR #40 (this PR) blocked on (2) — fixed by committing an empty
  `## Unreleased` block.

Both classes of failure are mechanical, not content drift. The
proposed "post-merge fixup workflow" (auto-PR that runs `cz changelog`
on every push to main) was shipped on the same branch — but adding a
self-healing loop to a self-inflicting gate is more machinery than the
problem warrants when the gate itself can be deleted.

The simpler invariant: `cz bump` is the only thing that writes to
`CHANGELOG.md`. The same tool that *projects* commit history into
the file is the only thing that *writes* to the file, so by
construction they cannot diverge.

# What about between-release visibility?

Old workflow: contributors could read the `## Unreleased` section to
see what had landed since the last release. New workflow:

- `mise run bump:dry-run` produces the same projection cz would write,
  formatted identically.
- `git log v0.7.0..HEAD --oneline` is the raw view.
- The PR list itself (`gh pr list --state merged --base main`) is
  the canonical view of "what's landed".

# Supersedes

`.erpaval/solutions/conventions/changelog-drift-gate-cz-overwrites-handwritten.md`
documented the *previous* approach: live with the gate, write detailed
context in commit messages and PR descriptions instead of CHANGELOG.
That advice is still good for commit/PR hygiene, but the "live with
the gate" framing is wrong now. There is no gate.

# Operational notes

- `mise.toml` task `changelog` removed — actively encouraged the wrong
  thing.
- `.github/workflows/ci.yml` has no `changelog-check` job.
- `cz bump` config still has `update_changelog_on_bump = true`,
  `changelog_incremental = true`. Those control the release-time
  rewrite; we're not changing them.

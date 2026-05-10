---
title: Rebase onto origin/main before running cz bump
track: knowledge
category: best-practices
module: release / git
component: commitizen + cz bump
severity: warning
tags: [release, commitizen, rebase, semver, conventional-commits]
applies_when:
  - branching off main and the branch lives more than a few hours
  - someone else may have shipped a release on main since your branch was cut
  - cz bump is configured with `version_provider = "uv"` and `tag_format = "v$version"`
pattern: |
  Always run `git fetch origin && git log main..origin/main --oneline | head` BEFORE
  invoking `mise run bump` (or `cz bump`). cz reads ONLY the local commit history
  to compute the next version — it has no awareness of upstream releases. If
  origin/main has shipped a release while you were branched, your local bump
  will produce a version (and therefore a git tag) that is already taken on
  the remote, and the tag push will be rejected.

  Recovery is fiddly: you have to (a) `git tag -d <wrong>` locally, (b)
  `git reset --soft HEAD~1` and discard the bump's pyproject/CHANGELOG/lock
  changes, (c) `git fetch && git rebase origin/main`, (d) re-resolve any
  conflicts the rebase surfaces, (e) `cz bump` again so it picks the correct
  next-after-remote version.

  A 30-second `git fetch + git log` check before bumping prevents the entire mess.
example_files:
  - planning/leiden-cpm-rip-replace/plan.md
---

# Why this matters

`cz bump` walks local conventional-commit history and infers MAJOR / MINOR /
PATCH from the types it finds (`feat!:` → MINOR under `major_version_zero=true`,
`fix:` → PATCH, etc.). It does NOT compare against `git ls-remote --tags`. If
two engineers cut branches from the same `main` commit and both ship `feat!:`
work, both will compute the same next version locally and the second push wins
or, in our case, gets rejected because the tag was already created upstream.

The pattern is: **treat `cz bump` as a release-time decision, not a commit-time
decision**, and always re-base off the freshest `origin/main` immediately
before running it.

# Example

```bash
# WRONG: bump on a stale branch
git checkout feat/foo
mise run bump                       # creates v0.3.0 locally — but origin already has v0.3.0
git push origin v0.3.0              # rejected

# RIGHT: re-sync first
git fetch origin
git log main..origin/main --oneline | head    # sanity-check what's been shipped
git checkout main && git reset --hard origin/main
git checkout feat/foo
git rebase origin/main              # resolve any conflicts
mise run bump:dry-run               # confirm the version cz picks doesn't clash
mise run bump                       # safe: cz now sees the real next version
git push origin <new-tag>
```

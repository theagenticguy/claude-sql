---
title: cz changelog overwrites hand-written CHANGELOG entries (SUPERSEDED)
track: knowledge
category: conventions
module: CHANGELOG.md
component: commitizen
severity: info
tags: [changelog, commitizen, ci, drift-gate, conventional-commits, release, superseded]
superseded_by:
  - .erpaval/solutions/conventions/changelog-write-only-via-cz-bump.md
applies_when:
  - The repo has a CI gate that runs `cz changelog` and asserts the file matches the regenerated output
  - You're tempted to hand-write detailed `### BREAKING CHANGE` / `### Fix` body text in CHANGELOG.md
  - Branch protection requires the changelog gate to pass before merge
pattern: |
  When a repo enforces a "CHANGELOG matches `cz changelog` output" CI
  gate, hand-written body text in `## Unreleased` will be obliterated
  on the next regenerate. cz parses the commit log and emits a single
  bullet per commit under the type-derived heading (`### Feat` for
  feat:, `### Fix` for fix:, etc.) — no body text, no notes.

  Workflow that survives the gate:

  1. **Hand-craft details in the commit message body and the PR
     description** — both are durable, surfaced in `git log` and on the
     PR page, and reach release readers via `gh release create
     --notes-from-tag`. Conventional commits parse type + scope from
     the subject line; everything below is free text that stays.

  2. **Let cz regenerate CHANGELOG.md before pushing.** Run
     `mise run changelog` (or `uv run cz changelog`) after committing
     the substantive change, then commit the regenerated file as
     `chore(changelog): regenerate for PR N`. CI's drift gate then
     passes because local matches CI's output.

  3. **For BREAKING CHANGE notes**, add a `BREAKING CHANGE: <summary>`
     footer to the commit message — cz will surface it. But still
     keep the *details* in the PR description; cz only carries the
     footer summary into the changelog, not the full explanation.

  Don't use `git commit --amend` to restage a CHANGELOG fix on a
  pushed branch — open a follow-up commit so the audit trail stays
  honest.
example_files:
  - CHANGELOG.md
  - .github/workflows/ci.yml
---

# Why this matters

Multiple PRs in this session hit the same dance: write a substantive PR with carefully-crafted CHANGELOG body, push, watch `changelog-check` fail with a diff showing CI wants to overwrite my prose with a one-line auto-generated bullet. Push a follow-up `chore(changelog): regenerate` commit. The detailed prose I wrote lives on in the PR description and the parent commit's body — it doesn't disappear, it just doesn't live in CHANGELOG.md.

This is `commitizen`'s designed behavior, not a bug. CHANGELOG.md is the auto-derived index; the source of truth is the commit history. Trying to force CHANGELOG.md to be both an index AND a release-notes document fights the tool. Use the PR description for narrative, the commit body for technical detail, and CHANGELOG.md as the auto-summary.

For projects that want narrative changelog entries without fighting cz, the alternative is to vendor a different changelog tool (towncrier, changie) or remove the cz changelog drift gate. Neither is worth doing for the perceived value.

# Example

PR #29 (markdown cleanup) workflow:
1. Commit `feat(cli): drop --format markdown ...` with detailed body explaining the breaking change.
2. Run `mise run changelog` → cz regenerates CHANGELOG.md with `### Feat - **cli**: drop --format markdown from public OutputFormat`.
3. Commit `chore(changelog): regenerate to match cz output` (no `--amend` — separate commit).
4. PR description carries the full BREAKING CHANGE narrative for release notes.

The lesson is documented in `.erpaval/solutions/best-practices/cz-bump-via-pr-with-branch-protection.md` for the related release-flow case (the bump-and-tag dance under branch-protection); this lesson is the changelog-content corollary.

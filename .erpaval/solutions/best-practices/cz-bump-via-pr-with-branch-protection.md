---
title: `cz bump` ↔ branch-protected `main` — bump on a release branch, PR-merge, re-tag the merge SHA, then push the tag
track: knowledge
category: best-practices
module: CLAUDE.md / mise.toml release workflow
component: commitizen + GitHub Actions branch-protection
severity: warning
tags: [release, commitizen, cz-bump, branch-protection, github-actions, sbom, conventional-commits, claude-sql]
applies_when:
  - "Repository has a 'PR required for main' branch-protection rule (the standard `theagenticguy/claude-sql` shape)"
  - "Cutting a release driven by `cz bump` (commitizen)"
  - "A release-triggered workflow exists (e.g. `sbom.yml` on `release.published`)"
pattern: |
  `cz bump` produces ONE local commit (`chore(release): X → Y`) plus a
  local annotated tag `vY`. Branch-protection on `main` rejects direct
  pushes — so the naive `mise run bump && git push main && git push v0.3.0`
  flow fails with `GH013: Repository rule violations found`.

  Because squash-merge rewrites the commit (new SHA on main), the tag
  that `cz bump` already created locally points at a SHA that never lands
  upstream. The recovery is mechanical but easy to skip steps:

  1. **Hard-reset main** to origin so the local rejected push doesn't
     stay diverged: `git reset --hard origin/main`. Also delete the local
     tag `cz bump` left behind: `git tag -d vY`.
  2. **Branch the bump.** `git checkout -b chore/release-X.Y.Z`, then
     `mise run bump` again on the branch (re-creates the commit + tag).
  3. **Push branch + open PR.** `gh pr create --title 'chore(release): A → B'`
     with a body that lists the highlights from `CHANGELOG.md`.
  4. **Wait for CI green** (the bump touches only `pyproject.toml`,
     `uv.lock`, `CHANGELOG.md` — should pass cleanly).
  5. **Squash-merge.** `gh pr merge <num> --squash --delete-branch`.
     This rewrites the commit; the new SHA on `main` is what the tag
     must point at.
  6. **Re-tag the merge SHA.** `git checkout main && git fetch && git reset --hard origin/main`.
     Then `git tag -d vY` (the local tag still points at the
     pre-merge SHA), and `git tag -a vY -m 'vY' <new-merge-sha>`.
  7. **Push the tag.** `git push origin vY`. Branch-protection does not
     gate tags, so this works.
  8. **Cut the GitHub release.** `gh release create vY --title vY --notes-file <path>`.
     Use `awk '/^## vY/{flag=1; next} /^## v/{flag=0} flag' CHANGELOG.md`
     to extract the section. The release-published event then fires any
     `release: types: [published]` workflows (SBOM, Slack notify, etc.).

  **If a release-triggered workflow needs the post-merge ref but you
  cut the release before fixing the workflow** (the v0.3.0 SBOM case),
  re-running via `gh workflow run sbom.yml --ref vY` runs the workflow
  as it existed at the tag — which is the broken version. Instead,
  generate the artifact locally and `gh release upload vY <artifact>
  --clobber`. The next release fires the fixed workflow naturally.

  **CLAUDE.md should not say "push to main directly".** The terse
  rhythm `mise run bump && git push origin main && git push origin vY`
  hides the protection step. Spell out the branch-PR-tag dance so
  future sessions don't re-discover it.
example_files:
  - CLAUDE.md                                          # release section in this repo
  - .github/workflows/sbom.yml                         # release-published consumer
  - .erpaval/solutions/api-patterns/cyclonedx-python-uv-environment.md  # SBOM how-to
counter_examples:
  - "`mise run bump && git push origin main` — rejected by branch-protection (GH013)."
  - "Pushing the local `cz bump` tag without re-tagging the merge SHA — the tag points at a SHA that doesn't exist on main, so future `git describe` and release tooling silently disagree about which commit was tagged."
  - "Re-running a release-triggered workflow with `gh workflow run --ref vY` to retry an SBOM upload — the workflow body is the *tagged* version, so a broken workflow at the tag stays broken on retry. Local artifact + `gh release upload --clobber` is the recovery."
references:
  - "GitHub branch-protection rejection error: GH013"
  - "commitizen `cz bump` docs: https://commitizen-tools.github.io/commitizen/bump/"
  - "PR #18 → #19 → #20 chain (claude-sql v0.3.0 release, 2026-05-09)"
  - ".erpaval/solutions/best-practices/branch-protection-pr-targeting-ci-trigger.md (sibling lesson on stacked-PR retarget)"
---

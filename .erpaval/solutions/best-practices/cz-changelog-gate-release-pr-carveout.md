---
title: `cz changelog --incremental` chokes on release branches — carve `chore(release):` PRs out of the gate
track: knowledge
category: best-practices
module: .github/workflows/ci.yml + commitizen
severity: warning
tags: [commitizen, cz, changelog, github-actions, release, claude-sql]
applies_when:
  - "Project enforces a `cz changelog --incremental` drift gate in CI"
  - "Release flow uses `cz bump` (which writes a new `## vX.Y.Z` header to CHANGELOG.md before the matching tag exists)"
pattern: |
  ``cz changelog --incremental`` walks back to the most recent
  release tag and renders the delta as the "Unreleased" section. On
  a release branch produced by ``cz bump``, that walk hits a snag:
  CHANGELOG.md ALREADY has a ``## vX.Y.Z`` header for the
  about-to-be-tagged release, but no tag yet exists. cz exits with:

  ```
  No tag found to do an incremental changelog
  ```

  …and code 16. Your generic "did the PR author update CHANGELOG?"
  drift gate fails on every release PR.

  The release PR's CHANGELOG is canonical *by construction* — it's
  what cz wrote when you ran ``mise run bump``. The gate has nothing
  useful to verify on this PR specifically. Skip it via a title-based
  carve-out:

  ```yaml
  changelog-check:
    if: >-
      github.event_name == 'pull_request' &&
      !startsWith(github.event.pull_request.title, 'chore(release):')
    runs-on: ubuntu-latest
    # …
  ```

  Two reasons to gate on the **PR title** rather than the commit
  subject or branch name:

  1. **Title is stable across the PR's lifetime.** Branch names get
     renamed during reviews; commit subjects shift on amends. The PR
     title is set once by ``cz bump`` (via the bot-push helper) and
     reviewers rarely touch it.
  2. **Conventional-commits already standardizes the prefix.**
     ``chore(release):`` is what ``cz bump_message`` produces by
     default. No new convention to maintain.

  **Why not skip on push-to-main too**: GitHub's squash-merge
  appends ``(#NNN)`` to the merged commit subject, which cz pulls
  into the next changelog rendering. Running the gate on
  push-to-main fails every squash-merge until a follow-up PR
  re-rendered. The PR-only restriction was already the right
  scoping; the release-PR carve-out is a second narrowing.

  **Don't try to outsmart cz with a regex strip.** Earlier versions
  of this gate stripped ``(#NNN)`` from both sides of the diff to
  accommodate squash-merge. That works for the squash case but
  doesn't help the release-PR case (cz fails before it ever writes
  output). The simpler "skip release PRs entirely" is the right
  shape; cz's behavior on a release branch is "rendered the canonical
  changelog", which is what we wanted to verify anyway.
example_files:
  - .github/workflows/ci.yml               # `if:` clause on `changelog-check`
  - CLAUDE.md                              # "Version bumping & changelog" section
counter_examples:
  - "Forcing the release branch to ``git tag`` before opening the PR — the tag would have to be unpushed, fragile to amends, and races branch-protection (you can't push tags during PR review reliably)."
  - "Using ``cz changelog`` (not ``--incremental``) to render the whole file from scratch in CI — rewrites the v0.x.0 history sections, which is gate noise. Stick with ``--incremental``."
  - "Branch-name match (``startsWith(github.head_ref, 'chore/release-')``) — works but couples the gate to a naming convention. Title match is more robust because cz controls the title."
  - "Letting the release PR fail the gate and merging anyway with ""it's a known issue"" — sets a precedent that the gate is advisory, which negates the entire point of having it."
references:
  - "commitizen changelog docs: https://commitizen-tools.github.io/commitizen/changelog/"
  - "claude-sql PR #26 (2026-05-10) — caught when the first cz-bump release PR after introducing the changelog gate failed CI; bundled the carve-out fix into the same PR so the release wasn't blocked."
---

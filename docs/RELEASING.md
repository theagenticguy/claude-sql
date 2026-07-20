# Releasing claude-sql

The runbook for publishing to PyPI, bumping the version, and cutting a release
under branch protection. See `AGENTS.md` for the day-to-day contributor gates.

## Publishing to PyPI

Trusted Publishing (OIDC) wires `pypi.org` and `test.pypi.org` to this repo via
`.github/workflows/publish.yml`. There are no API tokens; the workflow assumes a
short-lived OIDC token at job runtime. Two paths:

- **PyPI on release published.** `gh release create vX.Y.Z` →
  `release: types: [published]` fires → the `publish-pypi` job builds + uploads
  to https://pypi.org/p/claude-sql. Same trigger as `sbom.yml`, so the release
  flow already produces both the SBOM and the wheel in one motion.
- **TestPyPI on workflow_dispatch.** `gh workflow run publish.yml -f
  target=testpypi` (or click "Run workflow" in the Actions UI) builds + uploads
  to https://test.pypi.org/p/claude-sql for dry-run verification of metadata +
  the upload shape. Use this first when a contributor PR changes anything
  load-bearing about the package metadata.

Trusted Publisher entries on PyPI / TestPyPI must list:

- Project name: `claude-sql`
- Owner / repo: `theagenticguy / claude-sql`
- Workflow filename: `publish.yml`
- Environment name: `pypi` (production) or `testpypi` (dry-run)

The workflow declares `environment: name: pypi` / `environment: name: testpypi`
per job — that's the load-bearing scope check. Don't drop the environment block;
without it any compromised PR would inherit the ability to publish.

If a wheel ever lands on PyPI and you need to redact it: `twine` can't unpublish,
but you can mark it `yanked` via the project page, then publish a corrected
version. Trusted Publishing also supports per-environment `pending` publishers
that auto-clear once the first release lands.

## Version bumping & changelog

**`CHANGELOG.md` is write-only via `cz bump`.** Never hand-edit it. Never run
`mise run changelog` between releases — its output drifts against `cz`'s
post-squash-merge projection (the `## Unreleased` placeholder + `(#NNN)` suffix
injection) and any drift gate built on top will fight you. CI no longer enforces
any drift check; the file mutates exactly twice per release: once when `cz bump`
writes the new `## vX.Y.Z` block on the release branch, and once when
squash-merge appends `(#NNN)` to the merged commit's bullet.

Between releases, query commit history directly:

- `mise run bump:dry-run` — preview next version + the bullets `cz bump` would
  write.
- `git log --oneline v2.0.0..HEAD` — raw conventional-commit list.

**Pick `refactor:` (not `chore:`) for deprecation-removal commits.** The default
cz changelog filter surfaces `feat`, `fix`, and `refactor`; `chore` is invisible.
When a commit deletes a deprecated public API (a view, a Settings field, an
introspection helper), label it `refactor(<area>): drop deprecated <name>` so the
auto changelog mentions it. `refactor:` keeps the bump at PATCH for routine
cleanup releases. Use `chore:` for private-only refactors that don't need to
surface to users (internal helpers, lefthook tweaks, gitignore changes).

Version management is driven by `cz bump`, which reads commit history, picks
MAJOR/MINOR/PATCH from the conventional-commits types, updates `pyproject.toml` +
`uv.lock` (via `version_provider = "uv"`), writes `CHANGELOG.md`, and creates an
annotated tag.

- `mise run bump:dry-run` — preview the next version + tag without writing.
- `mise run bump` — bump, update changelog, commit as `chore(release): X → Y`,
  create annotated `vY` tag.

Commitizen config lives in `[tool.commitizen]` in `pyproject.toml`:
`tag_format = "v$version"`, `version_provider = "uv"`,
`major_version_zero = true`, `update_changelog_on_bump = true`,
`annotated_tag = true`.

## Release flow under branch protection (the branch-PR-tag dance)

`origin/main` is protected by a repository ruleset (pull-request required,
required status checks, code scanning, linear history — no bypass actors). Direct
pushes are rejected with `GH013: Repository rule violations found`. The release
commit MUST go through a PR. Don't `mise run bump` on `main` and try to push —
the local commit + tag will land in limbo. Use this sequence:

1. **Branch the bump.**
   ```bash
   git checkout main && git fetch origin && git reset --hard origin/main
   git checkout -b chore/release-X.Y.Z
   mise run bump:dry-run    # confirm version + increment
   mise run bump            # creates the commit AND the local tag
   ```
2. **Push branch + open PR.**
   ```bash
   git push -u origin chore/release-X.Y.Z
   gh pr create --title "chore(release): A → B" --body "$(cat <<'EOF'
   ...notes from CHANGELOG.md...
   EOF
   )"
   ```
3. **Wait for CI green**, then squash-merge:
   ```bash
   gh pr checks <pr-num> --watch
   gh pr merge <pr-num> --squash --delete-branch
   ```
4. **Re-tag the merge SHA.** Squash-merge rewrites the commit, so the local tag
   from step 1 points at a SHA that doesn't exist on main.
   ```bash
   git checkout main && git fetch origin && git reset --hard origin/main
   git tag -d vY                                    # drop pre-merge tag
   git tag -a vY -m "vY" $(git rev-parse HEAD)      # tag the merge SHA
   git push origin vY                               # tags bypass branch-protection
   ```
5. **Cut the GitHub release** so `release: types: [published]` workflows
   (`sbom.yml` etc.) fire:
   ```bash
   awk '/^## vY/{flag=1; next} /^## v/{flag=0} flag' CHANGELOG.md > /tmp/notes.md
   gh release create vY --title vY --notes-file /tmp/notes.md --verify-tag
   ```

**If a release-triggered workflow at the tag is broken** (e.g. the v0.3.0 SBOM
run hit a bad cyclonedx-py flag shape), don't `gh workflow run --ref vY` to retry
— that re-runs the *tagged* (broken) workflow body. Fix the workflow on a
follow-up PR for the next release, and recover the missed artifact locally:

```bash
uvx --from cyclonedx-bom cyclonedx-py environment .venv \
  --output-format JSON --output-file /tmp/SBOM.cdx.json
gh release upload vY /tmp/SBOM.cdx.json --clobber
```

See `.erpaval/solutions/best-practices/cz-bump-via-pr-with-branch-protection.md`
for the lesson capturing this; the cyclonedx-py flag-shape fix is in
`.erpaval/solutions/api-patterns/cyclonedx-python-uv-environment.md`.

# Semgrep in CI: use the container, not the deprecated action, and `scan` not `ci`

**Category:** api-patterns
**Tags:** semgrep, github-actions, sast, sarif, ci-workflow
**Applies to:** any project adding Semgrep to GitHub Actions for registry-driven scans
**Date:** 2026-05-08

## Two traps to avoid

1. **`returntocorp/semgrep-action` is deprecated.** Semgrep's GitHub org
   was renamed to `semgrep/` in 2024; the old action under
   `returntocorp/` no longer receives updates. Guides that reference it
   are stale.
2. **`semgrep ci` rejects `--config` flags.** The `ci` subcommand pulls
   rulesets from the Semgrep AppSec Platform via `SEMGREP_APP_TOKEN`.
   If you're running a self-hosted scan against free Registry rulesets
   (`p/auto`, `p/owasp-top-ten`, etc.), `semgrep ci --config p/auto`
   silently fails or errors out depending on Semgrep version.

## Canonical pattern (2026-current)

Use `semgrep/semgrep` Docker image as the job container and run
`semgrep scan` directly:

```yaml
# .github/workflows/semgrep.yml
name: Semgrep
on:
  push: { branches: [main] }
  pull_request: { branches: [main] }
  schedule: [{ cron: "20 17 * * 1" }]
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
permissions:
  contents: read
  security-events: write
jobs:
  semgrep:
    runs-on: ubuntu-latest
    container: { image: semgrep/semgrep }
    steps:
      - uses: actions/checkout@v6
      - name: semgrep scan (p/auto + p/owasp-top-ten)
        run: |
          semgrep scan \
            --config p/auto \
            --config p/owasp-top-ten \
            --sarif --output=semgrep.sarif \
            --metrics=off || true
      - uses: github/codeql-action/upload-sarif@v4
        if: always()
        with:
          sarif_file: semgrep.sarif
          category: semgrep
```

## Why the `|| true`

`semgrep scan` exits non-zero when it finds issues. Without `|| true`,
the job fails before the next step runs, so SARIF never reaches GitHub
code scanning. The gate is code scanning (severity thresholds are set
there), not the scan's exit code. If you want CI to hard-fail on any
finding regardless of severity, drop `|| true` *and* remove the
`upload-sarif` step — they're mutually exclusive strategies.

## Multi-config vs single config

`--config` takes multiple values by passing the flag repeatedly (as
shown). Registry rulesets available: `p/auto` (language-detected
defaults), `p/owasp-top-ten`, `p/security-audit`, `p/cwe-top-25`,
`p/r2c-ci`, etc. See https://semgrep.dev/explore for the full catalog.

## Why the container over `uvx semgrep`

Running Semgrep inside its own Docker image guarantees the bundled
rulesets, pre-warmed caches, and matched core binary. `uvx semgrep`
works but reinstalls a fresh semgrep install on every run; slower and
no cache hit.

## See also

- Semgrep CLI reference: https://semgrep.dev/docs/cli-reference
- Sample CI configs: https://semgrep.dev/docs/semgrep-ci/sample-ci-configs
- claude-sql commit `ba4d1b3` on `chore/ci-hardening` (session-c4635d).

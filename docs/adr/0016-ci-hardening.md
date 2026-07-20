# ADR 0016 — GitHub Actions CI hardening (SAST, SBOM, coverage)

**Status:** accepted — 2026-05-08
**Branch:** `chore/ci-hardening`

## Context

ADR 0015 gave the repo a strong *local* quality gate (mise + lefthook +
strict ruff + strict ty + pytest), but there was no CI — every guarantee
relied on contributor-machine discipline. That worked while the team was
one person; it breaks the moment anyone pushes via `git commit
--no-verify` or the review queue expands.

The target: every check that runs locally also runs on pull requests and
`push: main`, plus a second layer of security posture (SAST / SBOM /
supply chain) that isn't reasonable to run on every local save.

Reference implementation: a sibling Python repo by the same contributor —
same convention set, translated for this project's layout.

## Decision

Seven workflow files, scoped by intent:

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci.yml` | push:main, pr:main | `lint + fmt + typecheck + test(+coverage) + lock-check` — the mise run check mirror. Codecov upload via tokenless OIDC. |
| `codeql.yml` | push, pr, weekly | CodeQL Python, `security-and-quality` queries. SARIF to code scanning. |
| `semgrep.yml` | push, pr, weekly | `p/auto` + `p/owasp-top-ten` via the `semgrep/semgrep` container. SARIF to code scanning. |
| `osv.yml` | push, pr, weekly | osv-scanner over `uv.lock`. SARIF to code scanning + hard fail on findings. |
| `scorecard.yml` | branch_protection_rule, weekly, push:main | OpenSSF Scorecard. SARIF + artifact. |
| `sbom.yml` | release.published, workflow_dispatch | CycloneDX SBOM via `cyclonedx-py environment .venv`, attached to release. |
| `commitlint.yml` | pr | `cz check` over every commit in the PR. |

Plus one lefthook pre-commit job (`uv-lock-sync`) that runs `uv lock
--check` when `pyproject.toml` or `uv.lock` is staged — catches the
"bumped a dep without regenerating the lockfile" drift locally before
CI.

## Decisions worth calling out

- **Semgrep action vs container.** The `returntocorp/semgrep-action` is
  deprecated (org renamed to `semgrep/` in 2024). `semgrep ci` explicitly
  rejects `--config` flags — that path pulls rulesets from the Semgrep
  AppSec Platform. For a self-contained, registry-driven scan we run
  `semgrep scan --config p/auto --config p/owasp-top-ten --sarif` inside
  the `semgrep/semgrep` container. `|| true` on the scan step lets the
  SARIF upload complete on findings; code scanning is the gate.

- **Codecov tokenless OIDC.** `codecov-action@v5` supports
  `use_oidc: true` for public repos. Requires `id-token: write` on the
  test job only. No `CODECOV_TOKEN` secret needed.

- **CycloneDX for Python.** `cdxgen` doesn't understand `uv.lock` (only
  `requirements.txt`, `setup.py`, `pyproject.toml`, `poetry.lock`). `uv
  export` has no `--format cyclonedx` flag. `cyclonedx-py environment
  .venv` walks the installed venv and emits the resolved dep graph —
  closest thing to first-class uv support.

- **OSV-scanner two-step.** Scan once with `|| true` + SARIF output →
  upload SARIF → re-run without `--format` to fail the job. Separates
  "report" from "gate" so findings still reach code scanning when CI
  blocks.

- **Action pinning.** Tag pins (`@v4`, `@v5`, `@v7`, `@v2.4.3`, `@v6`),
  not Dependabot SHAs. Scorecard will flag us on
  `pinning-dependencies`; acceptable trade-off for v1 — keeps the
  update cadence sane.

## Consequences

- Every PR now runs 6 CI workflows (ci, codeql, semgrep, osv, commitlint
  — plus scorecard on main-branch events). Cold PR is ~4 min; cached
  incremental runs much faster.
- Branch protection can now require `ci / lint`, `ci / fmt`, `ci /
  typecheck`, `ci / test`, `ci / lock-check`, `Commitlint /
  commitizen` before merge.
- Weekly security cadence: CodeQL (Weds), Scorecard (Mon), Semgrep
  (Mon), OSV (Tue). Spread the load, catch new CVEs even on quiet
  weeks.
- First publish triggers SBOM generation, attached to the release as
  `SBOM.cdx.json`.

## Follow-up

- Enable branch protection requiring the workflow names above (owner
  action, not in PR).
- Revisit pinning-dependencies score once Dependabot is tuned for the
  actions ecosystem.
- Consider adding `pip-audit` if osv-scanner surface proves thin on
  Python-specific advisories.

## Sources

- A sibling Python repo's GitHub Actions workflows by the same contributor.
- Session research `.erpaval/sessions/session-c4635d/research-sast.yaml`.

# SARIF-emitting CI scanners: two steps, not one

**Category:** best-practices
**Tags:** sarif, github-actions, security-scanner, osv-scanner, semgrep, code-scanning
**Applies to:** any GitHub Actions workflow that runs a SAST/SCA scanner and wants findings in code scanning
**Date:** 2026-05-08

## Rule

When a security scanner emits SARIF *and* has a non-zero exit on
findings, split "report" from "gate":

1. Run the scanner with `|| true` + `--format=sarif --output=…` so the
   step succeeds even when findings exist.
2. `github/codeql-action/upload-sarif@v4` the SARIF file.
3. Re-run the scanner *without* `--format` to let its exit code fail
   the job on findings — the gate.

## Why

If step 1 fails, subsequent steps (`if: always()` aside) don't run, and
findings never reach GitHub code scanning. Without step 3, the job
passes silently on findings that *should* block merge.

**Why:** severity gating belongs in code scanning settings (branch
protection alerts), not in the scanner exit code. The scanner's job is
to *report*; CI's job is to *block*. Decoupling the two lets you adjust
the gate without touching the workflow.

**How to apply:** use for OSV-scanner, Semgrep, trivy, grype, and any
other tool whose docs say "exits non-zero on findings."

## Canonical snippet (osv-scanner; Semgrep structure is similar)

```yaml
- name: Install osv-scanner
  run: |
    curl -sL -o /tmp/osv-scanner \
      https://github.com/google/osv-scanner/releases/download/v2.3.5/osv-scanner_linux_amd64
    chmod +x /tmp/osv-scanner
- name: Scan lockfile (SARIF output)
  run: |
    /tmp/osv-scanner scan source \
      --lockfile=uv.lock \
      --format=sarif \
      --output=osv.sarif || true          # don't fail; upload runs
- uses: github/codeql-action/upload-sarif@v4
  if: always()
  with:
    sarif_file: osv.sarif
    category: osv-scanner
- name: Fail on vulnerabilities
  run: /tmp/osv-scanner scan source --lockfile=uv.lock   # the gate
```

## Anti-patterns

- **Single step, no `|| true`.** Findings mean exit 1, which skips
  `upload-sarif`, which means no code-scanning visibility. Pass without
  seeing why → silent regression risk.
- **`|| true` *without* the gate step.** Every run passes; the SARIF
  upload adds alerts but nothing blocks merge. False sense of safety.
- **Relying on `if: always()` alone.** `always()` runs the step even
  after a failed predecessor, but the workflow result is still failed
  — upload happens but job fails for the wrong reason, obscuring
  signal.

## Semgrep nuance

Semgrep skips the gate step by design: we want code scanning to be the
arbiter since severity thresholds are configurable there. OSV is
different — any CVE match is usually worth blocking (modulo
severity-tier tuning via scanner flags or an advisory allowlist).

## See also

- claude-sql `.github/workflows/osv.yml` + `.github/workflows/semgrep.yml`
  on `chore/ci-hardening` (session-c4635d).
- A sibling Python repo's `ci.yml` osv job — the origin pattern.

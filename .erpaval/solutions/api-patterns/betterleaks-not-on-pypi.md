---
title: betterleaks is a Go binary, not on PyPI — install via mise (`aqua:`) locally + raw curl in CI
track: knowledge
category: api-patterns
module: mise.toml
component: betterleaks (gitleaks successor)
severity: warning
tags: [betterleaks, secrets-scanner, gitleaks, mise, ci, sast]
applies_when:
  - "Adding betterleaks to a Python project's local + CI security sweep"
  - "Searching `uv add betterleaks` or `pip install betterleaks` and getting nothing"
pattern: |
  **betterleaks is NOT a Python package.** It is the successor to
  gitleaks — same maintainer, backed by Aikido Security, distributed as
  a Go binary via brew, dnf, ghcr.io docker, and GitHub release
  binaries. There is no `pip install` / `uv add` path, and no
  first-party `betterleaks/betterleaks-action` repo (404 as of
  2026-05-09).

  **Local: use mise's `aqua:` backend.** mise's `aqua:` registry has
  betterleaks; the deprecated `ubi:` backend works too but mise
  upstream is removing it in 2027.1.0 (use `github:` going forward
  if `aqua:` ever stops resolving):

      [tools]
      "aqua:betterleaks/betterleaks" = "1.2.0"

  Pin the version — `latest` will silently roll under you. When you
  bump, bump the CI install URL (next section) in the same commit.

  **CI: raw curl + tar (no first-party action exists).** The release
  filename pattern is `betterleaks_<version>_linux_x64.tar.gz` — note
  the embedded version string. Easy gotcha: my first attempt used
  `betterleaks_linux_x64.tar.gz` (no version) and CI failed with
  `gzip: stdin: not in gzip format` because curl 404'd silently → 0-byte
  file → tar failure. Pattern that works:

  ```yaml
  - name: Install betterleaks
    env:
      BETTERLEAKS_VERSION: "1.2.0"
    run: |
      curl -sSL -o /tmp/betterleaks.tgz \
        "https://github.com/betterleaks/betterleaks/releases/download/v${BETTERLEAKS_VERSION}/betterleaks_${BETTERLEAKS_VERSION}_linux_x64.tar.gz"
      tar -xzf /tmp/betterleaks.tgz -C /tmp
      sudo mv /tmp/betterleaks /usr/local/bin/betterleaks
      betterleaks version
  ```

  Note the `tar -xzf <archive> -C /tmp` form — *don't* pass an inner
  filter argument like `-C /tmp betterleaks` (which asks tar to extract
  only an inner path named "betterleaks", a different operation).

  **Run with `--exit-code=0` in CI.** betterleaks defaults to exit 1 on
  findings, exit 0 on no findings. With `--exit-code=0` the step always
  succeeds; SARIF uploads via codeql-action; gating happens through
  code scanning UI per the report-vs-gate pattern:

      betterleaks git . -f sarif -r betterleaks.sarif --exit-code=0

  The `git` subcommand (default) scans full commit history — *the
  checkout step needs `fetch-depth: 0`* or you only see the merge-commit
  shadow.

  **Lock local + CI versions together.** Bumping mise's pin without
  bumping the CI URL (or vice versa) creates silent drift where local
  and CI scan with different rule sets. Always co-bump in one commit.
example_files:
  - mise.toml                                    # aqua: install
  - .github/workflows/leaks.yml                  # CI install + scan
  - .erpaval/sessions/session-2293a5/research-security.yaml  # research that surfaced this
---

# Why this matters

The natural `uv add betterleaks` / `pip install betterleaks` paths both
fail (PyPI has nothing under that name), and the obvious next move
("there must be a `betterleaks-action` GitHub Action") also 404s. A
session that doesn't pause to ground via Context7 / web search will
chase those dead ends or fall back to `gitleaks` (the predecessor),
missing betterleaks's expanded ruleset and Aikido-backed maintenance.

The release-URL gotcha is the kind of bug that passes local checks
(mise installs the binary fine) but fails CI on the very first run
with a message (`gzip: stdin: not in gzip format`) that doesn't
obviously point at the URL. Pinning the version + co-bumping mise.toml
+ leaks.yml in the same commit is the discipline that prevents this
class of bug from re-surfacing.

# Example

```toml
# mise.toml — local
[tools]
"aqua:betterleaks/betterleaks" = "1.2.0"  # pin; co-bump with leaks.yml

[tasks."security:leaks"]
description = "Secrets sweep over git history via betterleaks"
run = """
mkdir -p .sarif
betterleaks git . -f sarif -r .sarif/betterleaks.sarif --exit-code=0
"""
```

```yaml
# .github/workflows/leaks.yml — CI
- uses: actions/checkout@v6
  with:
    fetch-depth: 0  # full history needed for `betterleaks git .`
- name: Install betterleaks
  env:
    BETTERLEAKS_VERSION: "1.2.0"  # co-pinned with mise.toml
  run: |
    curl -sSL -o /tmp/betterleaks.tgz \
      "https://github.com/betterleaks/betterleaks/releases/download/v${BETTERLEAKS_VERSION}/betterleaks_${BETTERLEAKS_VERSION}_linux_x64.tar.gz"
    tar -xzf /tmp/betterleaks.tgz -C /tmp
    sudo mv /tmp/betterleaks /usr/local/bin/betterleaks
    betterleaks version
- name: Run betterleaks (SARIF)
  run: betterleaks git . -f sarif -r betterleaks.sarif --exit-code=0
- uses: github/codeql-action/upload-sarif@v4
  if: always()
  with:
    sarif_file: betterleaks.sarif
    category: betterleaks
```

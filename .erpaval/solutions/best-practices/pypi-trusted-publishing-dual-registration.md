---
title: PyPI Trusted Publishing requires separate registration on pypi.org AND test.pypi.org
track: knowledge
category: best-practices
module: PyPI Trusted Publisher web forms
component: PyPI + TestPyPI + GitHub Actions OIDC
severity: warning
tags: [pypi, testpypi, trusted-publishing, oidc, github-actions, claude-sql]
applies_when:
  - "Setting up Trusted Publishing for the first time on a new PyPI project"
  - "The publish workflow has both a TestPyPI dry-run path and a production PyPI path"
pattern: |
  PyPI and TestPyPI are **separate sites with separate accounts and
  separate Trusted Publisher databases**, even though their forms look
  identical. A successful registration on pypi.org grants nothing to
  test.pypi.org and vice-versa.

  The failure mode is a misleading OIDC error during the workflow run:

  ```
  Trusted publishing exchange failure:
  Token request failed: the server refused the request for the
  following reasons:
  * `invalid-publisher`: valid token, but no corresponding publisher
    (Publisher with matching claims was not found)
  ```

  The claims rendered for debugging look correct (repo + workflow +
  environment all match what you registered). The catch: the OIDC
  token is being presented to the wrong site. PyPI sees the production
  publisher, TestPyPI doesn't have one yet (or vice-versa).

  **Fix**: register on both, even if you only plan to publish to one
  long-term. The dry-run path during initial bring-up is genuinely
  useful, and the registration cost is two minutes per site.

  | Field                | pypi.org form       | test.pypi.org form  |
  |----------------------|---------------------|---------------------|
  | Project name         | `your-package`      | `your-package`      |
  | Owner                | `your-org`          | `your-org`          |
  | Repository name      | `your-repo`         | `your-repo`         |
  | Workflow name        | `publish.yml`       | `publish.yml`       |
  | Environment name     | `pypi`              | `testpypi`          |

  The **Environment name** is the load-bearing scope check. Your
  workflow declares ``environment: name: pypi`` on the production
  job and ``environment: name: testpypi`` on the dry-run job; PyPI
  verifies BOTH the workflow filename AND the environment name
  before issuing a publish credential. Two different envs → two
  separate publisher entries → no cross-contamination if a
  compromised PR somehow inherited workflow access.

  **Project name field** lets you register a publisher BEFORE the
  project exists ("pending publisher"). The first successful upload
  claims the name and converts the pending publisher into a regular
  one. Useful for first-release setup; no mental gymnastics needed
  for "do I create the project first or the publisher first?"

  TestPyPI accounts are entirely separate from pypi.org accounts —
  if you've never logged in to test.pypi.org before, you'll need to
  create a fresh account there even though the username/email might
  match your pypi.org account.

  **Workflow filename** in the form is the bare filename
  (``publish.yml``), NOT the path
  (``.github/workflows/publish.yml``). PyPI implicitly assumes the
  ``.github/workflows/`` prefix.
example_files:
  - .github/workflows/publish.yml          # `environment:` block per job
  - https://pypi.org/manage/account/publishing/
  - https://test.pypi.org/manage/account/publishing/
counter_examples:
  - "Registering only on pypi.org and assuming TestPyPI ""inherits"" — it doesn't. The first dry-run gets the misleading ``invalid-publisher`` error and you waste a CI run debugging."
  - "Pasting the full workflow path (``.github/workflows/publish.yml``) into the Workflow filename field — silently mismatches and rejects the OIDC token."
  - "Sharing the environment name across pypi + testpypi (e.g. ``release`` for both) — works, but loses the per-env scope check; a malicious actor with workflow_dispatch access could publish to the wrong target."
references:
  - "PyPI Trusted Publishers docs: https://docs.pypi.org/trusted-publishers/"
  - "Troubleshooting invalid-publisher: https://docs.pypi.org/trusted-publishers/troubleshooting/"
  - "claude-sql session-d80d08 (2026-05-10) — caught after the first TestPyPI workflow_dispatch returned ``invalid-publisher`` despite a working production publisher entry. Two-minute fix once you know."
---

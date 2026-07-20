# 07 — Test / CI / Build baseline + existing-docs inventory

**Perspective:** the v2 rewrite's *definition of done* — what green looks like today,
what tests get deleted when `evals/` (and `provenance/`) drop, and the full worklist
of docs the rewrite must touch.

**Repo facts as of this pass** (`git 7f9e5a2`, 2026-07-19):

- Single package `claude-sql` **v1.2.1** (`pyproject.toml:3`), one namespace root
  `src/claude_sql/` with five layer sub-packages: `core` (L0) < `{analytics, evals,
  provenance}` (L1 siblings) < `app` (L2).
- Python floor **`>=3.13`** (`pyproject.toml:8`), agreeing with `.python-version`
  (`3.13`) and `mise.toml:16` (`python = "3.13"`).
- Code-intelligence tool is **CodeGraph** (`.codegraph/codegraph.db`, 8.1 MB);
  `AGENTS.md` already rewritten for it (all OpenCodeHub tooling removed).

---

## 1. Test baseline

### Real collected count: **826 tests** (verified)

```
$ uv run pytest --collect-only -q  →  "826 tests collected in 2.43s"
```

The memory figure of ~826 is **confirmed**. (A raw `grep 'def test_'` count returns
760 — the 66-test gap is `@pytest.mark.parametrize` expansion, so trust the collector,
not grep.) 59 test files under `tests/`, plus `tests/conftest.py` and
`tests/__init__.py`.

### Which tests die when `evals/` drops (v2 DROP): **88 tests, 8 files**

All import `claude_sql.evals`. Collected counts:

| Test file | Tests | Covers |
|---|---|---|
| `tests/test_kappa_worker.py` | 17 | Fleiss/Cohen kappa agreement gate |
| `tests/test_judge_worker_extras.py` | 13 | judge panel edge cases |
| `tests/test_ungrounded_worker.py` | 12 | ungrounded-claim detection |
| `tests/test_judge_worker.py` | 12 | cross-provider Bedrock judge panel |
| `tests/test_blind_handover.py` | 11 | blind-handover context stripping |
| `tests/test_judges.py` | 10 | judge catalog |
| `tests/test_freeze.py` | 9 | study freeze/pre-registration |
| `tests/test_judges_extras.py` | 4 | judge catalog extras |

### Which tests die if `provenance/` also drops (v2 "likely DROP"): **80 tests, 5 files**

All import `claude_sql.provenance`. Collected counts:

| Test file | Tests | Covers |
|---|---|---|
| `tests/test_binding_extras.py` | 23 | transcript↔PR binding edge cases |
| `tests/test_review_sheet_worker_extras.py` | 22 | review-sheet worker edge cases |
| `tests/test_binding.py` | 15 | commit-trailer / git-notes binding (RFC 0001) |
| `tests/test_review_sheet_render.py` | 13 | review-sheet render |
| `tests/test_review_sheet_worker.py` | 7 | review-sheet worker |

### Mixed / surgical

- `tests/test_pr3_perf.py` (**11 tests**) imports **both** evals and provenance — a
  perf test; it dies with them.
- `tests/test_cli.py` (**83 tests**) imports `claude_sql.provenance` because the CLI
  registers `bind` / `judges` / `freeze` / `judge` / `kappa` subcommands. This file
  is **mostly core CLI** and must survive — it needs a **surgical edit** to remove the
  eval/provenance-command assertions while keeping the ~60+ core-command tests.

### The "must keep passing" core+analytics baseline

`826 − 88 (evals) − 80 (provenance) − 11 (pr3_perf) = 647` tests remain, **minus** the
eval/provenance-command subset inside `test_cli.py`. These cover `core/` (46 test files
import `claude_sql.core`) and `analytics/` (20 test files import `claude_sql.analytics`):
SQL views, embed/cluster/community/conflicts/friction/trajectory/classify workers,
ingest/simhash, lance store, parquet shards, checkpointer, retry queue, S3 source,
output formatting, session text, schemas, skills catalog, config, home. **This is the
v2 regression target: every one of these must stay green through the hexagonal rewrite.**

The new v2 work (pluggable embedding providers: Cohere/Bedrock + Ollama + ONNX bge)
will *add* provider-port test files that don't exist yet.

### Test infrastructure

- **`tests/conftest.py`** — the whole fixture spine. Exposes: `make_session_jsonl` /
  `tmp_corpus` (two-session on-disk JSONL corpus under `tmp_path/projects/`) /
  `registered_con` (in-memory DuckDB with `register_raw`+`register_views`), `tmp_settings`
  (a `Settings` with every cache path redirected under `tmp_path` — no touching the real
  `~/.claude/`), plus **Bedrock fakes**: `FakeBedrockClient` (captures `invoke_model`
  body, returns configurable payload, can queue exceptions — covers Cohere Embed v4 +
  Sonnet 4.6 shapes), `fake_bedrock_client` factory, and `mock_invoke_model` MagicMock
  fallback. **Tests never hit Bedrock** — this is the mock layer that guarantees it.
- **moto (S3 mock)** — used by exactly **one** file, `tests/test_s3_source.py`
  (dev dep `moto[s3,server]>=5.2.0`, `pyproject.toml:99`). The S3 transcript source
  (`core/s3_source.py`) is the only AWS-network surface tested against a mock.
- **testcontainers** — **not used anywhere** (no dependency, no import). Don't assume it.
- **Config:** `[tool.pytest.ini_options]` `testpaths = ["tests"]`, `pythonpath = ["tests"]`
  (`pyproject.toml:231-233`). CI adds coverage: `--cov=src --cov-report=xml`.

> **v2 note:** the fixture spine (`tmp_corpus`, `registered_con`, `FakeBedrockClient`)
> is provider-agnostic and survives the rewrite intact. Only the eval/provenance-specific
> fixtures inside the deleted test files go away. The Bedrock fake will want a sibling for
> the Ollama / ONNX provider ports.

---

## 2. CI surface — `.github/workflows/`

Ten workflows. **Green CI on a PR** requires the five gating jobs in `ci.yml` plus the
two PR-scoped extras (CodeQL, commitlint) and the SAST/SCA jobs to complete (most SAST
jobs are report-only, gating via code-scanning UI, not exit code).

| Workflow | Trigger | Gates / does | Blocking? |
|---|---|---|---|
| **ci.yml** | push+PR to `main` | 5 jobs: `lint` (ruff check), `fmt` (ruff format --check), `typecheck` (`ty check`), `lock-check` (`uv lock --check`), `test` (pytest + `--cov` → Codecov OIDC upload) | **YES — hard gate** |
| **codeql.yml** | push+PR + weekly cron | CodeQL `security-and-quality` on Python | Findings gate via branch protection |
| **semgrep.yml** | push+PR + cron | `semgrep scan p/auto + p/owasp-top-ten` → SARIF | report-only (`|| true`) |
| **bandit.yml** | push+PR + cron | Python SAST → SARIF | report-only (`--exit-zero`). **BUG: scans `-r packages` — that dir no longer exists** (repo is `src/`); the composite/mise path scans `src`. Fix to `-r src` in v2. |
| **osv.yml** | push+PR + cron | osv-scanner over `uv.lock` → SARIF **and** a final `scan source` step that **fails on vulnerabilities** | **YES — the second osv step is a hard gate** |
| **leaks.yml** | push+PR + cron | betterleaks (gitleaks successor) secrets sweep over full history → SARIF | report-only (`--exit-code=0`) |
| **scorecard.yml** | branch_protection_rule + push + cron | OpenSSF Scorecard → SARIF + published badge | report-only |
| **commitlint.yml** | PR only | `cz check --rev-range base..head` — conventional-commits on every PR commit | **YES — hard gate on PRs** |
| **sbom.yml** | release published + dispatch | CycloneDX SBOM from `.venv` via `cyclonedx-py environment`, attached to the GitHub release | release-time only |
| **publish.yml** | release published + dispatch | build wheel+sdist, smoke-test both isolated, `uv publish` via OIDC to PyPI/TestPyPI | release-time only |

### Composite action: `.github/actions/setup-claude-sql/action.yml`

Used by `ci.yml`, `sbom.yml`, `commitlint.yml`. Steps: `jdx/mise-action@v4` (installs
mise-pinned Python 3.13 + uv), restore `~/.cache/uv` keyed on `hashFiles('uv.lock')`,
then `uv sync --locked --all-extras --all-groups`. **Deliberately does not use
`astral-sh/setup-uv`** — mise is the single source of truth for the uv+Python versions;
a second uv install would risk drift. (bandit/osv/leaks/semgrep/codeql/scorecard install
their own tooling and skip this composite.)

**"Green CI" = ** ruff clean + ruff-format clean + `ty` strict clean + `uv.lock` fresh +
826 pytest green + CodeQL no new alerts + osv no vulns + conventional commits on the PR.

---

## 3. Local quality gate — the definition of done

### `mise run check` (`mise.toml:126-128`) — **THE definition of done**

```toml
[tasks.check]
description = "All quality gates: lint + fmt + typecheck + import-linter + test"
depends = ["lint", "fmt", "typecheck", "lint:imports", "test"]
```

It fans out to five tasks (each `sources`-cached, so docs-only reruns skip):

- **`lint`** → `uv run ruff check .`
- **`fmt`** → `uv run ruff format --check .`
- **`typecheck`** → `uv run ty check`
- **`lint:imports`** → `uv run lint-imports` (enforces the layered DAG:
  `core < {analytics,evals,provenance} < app`, `pyproject.toml:261-280`)
- **`test`** → `uv run pytest --no-header -q` (all 826)

> Note: the CLAUDE.md prose says check is "lint + fmt + typecheck + test"; the real
> `mise.toml` `depends` list **also includes `lint:imports`** — five gates, not four.
> v2's import-linter contract must be rewritten when evals/provenance drop (see §5).

### Full mise task inventory (`mise.toml`)

`install` (→ `hooks:install` then `uv sync --all-extras`), `upgrade`, `lock:check`,
`hooks:install` / `hooks:uninstall` / `hooks:run`, `commit`, `bump`, `bump:dry-run`,
`lint`, `fmt`, `fmt:write`, `typecheck`, `lint:imports`, `test`, **`check`**,
`security:bandit` / `security:semgrep` / `security:osv` / `security:leaks` /
**`security`** (all four in parallel → SARIF under `.sarif/`), `build` (`uv build`),
`tool:install` / `tool:upgrade` / `tool:uninstall`, `cli`.
(There is **no `migrate` or `proofs` task** — the CLAUDE.md-adjacent "migrate" is the
CLI `claude-sql cache migrate` subcommand, not a mise task; "proofs" does not exist here.)

### `lefthook.yml` git hooks

- **pre-commit** (parallel, staged-file scoped): `ruff check --fix` (stage_fixed),
  `ruff format` (stage_fixed), `ty check` (whole tree), `lint-imports`, `uv lock --check`
  (only when `pyproject.toml`/`uv.lock` staged).
- **commit-msg**: `cz check --allow-abort` (conventional commits).
- **pre-push**: full `pytest --no-header -q` (belt-and-suspenders before push).

### ruff / ty strictness

- **ruff** (`pyproject.toml:109-186`): `target-version = py313`, line-length 100,
  **32-family strict selector** (E, W, F, I, N, UP, B, SIM, ANN, ASYNC, BLE, C4, DTZ,
  ERA, FBT, G, ICN, ISC, LOG, PERF, PIE, PL, PT, PTH, RET, RSE, S, T20, TID, TRY, PGH,
  RUF) with principled inline-documented ignores. Bans stdlib `logging`
  (`flake8-tidy-imports.banned-api` — loguru-only), bans relative imports.
- **ty** (`pyproject.toml:191-219`): strict — `[tool.ty.rules] all = "error"`,
  `error-on-warning = true`, over `src/ + tests/`. `tests/**` carries a narrow override
  for the DuckDB-`Optional`-subscript false-positive class.

---

## 4. Build & release

- **Build backend:** `uv_build>=0.11.14,<0.12` (`pyproject.toml:61-63`). Single
  namespace package (`[tool.uv.build-backend]` `module-name = "claude_sql"`,
  `namespace = true`, `pyproject.toml:70-72`) → **one self-contained wheel**;
  `uv build` "just works", no bundle step. `mise run build` = `uv build`.
- **Version:** **1.2.1** (`pyproject.toml:3`), driven by commitizen
  (`version_provider = "uv"`, `tag_format = "v$version"`, `pyproject.toml:239-253`).
  `pre_bump_hooks = ["uv lock"]`, `post_bump_hooks = ["uv sync"]`. **`CHANGELOG.md` is
  write-only via `cz bump`** — never hand-edit.
- **`mise run bump`** = `uv run cz bump`: reads conventional-commit history, picks
  MAJOR/MINOR/PATCH, updates `pyproject.toml` + `uv.lock` + `CHANGELOG.md`, cuts an
  annotated tag. `bump:dry-run` previews.
- **Release entails** (per CLAUDE.md, branch-protected `main`): branch → `mise run bump`
  → push branch + open PR → wait CI green → squash-merge → re-tag the merge SHA (squash
  rewrites the SHA) → push tag → `gh release create`. The release trigger fires both
  `sbom.yml` and `publish.yml`.
- **Publish** (`publish.yml`): OIDC Trusted Publishing (no tokens). `build` job builds +
  smoke-tests wheel AND sdist in isolated envs; `publish-pypi` (on release or
  `dispatch target=pypi`, `environment: pypi`) → https://pypi.org/p/claude-sql;
  `publish-testpypi` (on `dispatch target=testpypi`, `environment: testpypi`) →
  TestPyPI. Named index `testpypi` in `pyproject.toml:87-91`; PyPI is the implicit default.

> **v2 note:** dropping evals/provenance changes the wheel's contents but not the
> single-wheel build shape. Version will bump (likely a MINOR/MAJOR given the rewrite).
> Nothing in the build/publish machinery needs structural change — but the import-linter
> contract in `pyproject.toml:264-280` **will fail to lint** the moment the
> `evals`/`provenance` modules are deleted, so both contracts must be rewritten in the
> same PR (this is a build-gate blocker, not optional).

---

## 5. EXISTING DOCS INVENTORY — the rewrite worklist

Change-type legend: **rebrand** (naming/badges/links), **drop-evals**, **drop-prov**,
**provider** (pluggable-embedding switch), **3.13** (Python-floor note),
**codegraph** (codehub/codehub→codegraph), **hexagon** (architecture reshape),
**done** (already updated), **keep** (survives ~unchanged).

### Root docs

| Doc | Size | Currently covers | v2 change-type |
|---|---|---|---|
| `README.md` | 34 KB | Full user guide: pitch, mermaid data-flow, install, AWS creds, S3, quick tour, **31-subcommand CLI surface** (incl. `bind`/`judges`/`freeze`/`judge`/`kappa`/`review`), views/macros, env-var table, dev/quality/security sections, design notes | **drop-evals · drop-prov · provider · hexagon** — biggest single rewrite. Remove eval-gym + provenance CLI rows and design notes; rework "Cohere Embed v4 on Bedrock"-only framing into pluggable providers (Cohere/Bedrock + Ollama + ONNX bge); prune subcommand count |
| `CLAUDE.md` | 37 KB | Project instructions: architecture, quality/security gates, CodeQL hygiene, loguru ban, hooks, publishing, release dance, Bedrock, DuckDB, structured output, per-worker notes (trajectory/conflicts/friction), determinism, Leiden/CPM, resilience patterns, CLI surface, env vars, rollback, deferred decisions, **CodeGraph section** | **drop-evals · drop-prov · provider · hexagon** — second-biggest. CodeGraph section already added; but "four analytics layers"/five-sub-package framing, the embedding-provider assumptions, and any eval/provenance guidance all need rewriting. Correct the `mise run check` description (it's 5 gates incl. `lint:imports`, not 4) |
| `AGENTS.md` | 1.3 KB | CodeGraph code-intelligence pointer only | **done** — already rewritten for codegraph; verify it still points at codegraph after other tooling churn |
| `CHANGELOG.md` | 6.4 KB | cz-generated release history through v1.2.1 | **keep** — write-only via `cz bump`; the v2 release bump appends the next block. Do not hand-edit |
| `SECURITY.md` | 619 B | **Stock GitHub template** — placeholder "5.1.x / 4.0.x" version table and "tell them where to go" boilerplate; never customized | **rebrand** — trivially stale; fill in real supported-version + reporting policy (low effort, high embarrassment-avoidance) |

### `docs/` tree

**ADRs / RFCs** (`docs/adr/`, `docs/rfc/`):

| Doc | Covers | v2 change-type |
|---|---|---|
| `docs/adr/0015-stack-modernization.md` | Python 3.13 floor + strict ruff/ty rationale; **the canonical 3.14-deferral record** (hdbscan cp314 wheels) | **3.13** — this is the doc to cite for the floor; update the hdbscan-watch line if v2 revisits it |
| `docs/adr/0016-ci-hardening.md` | SAST/SBOM/coverage CI hardening rationale | **keep / rebrand** — fix the `bandit -r packages` path bug reference if noted here |
| `docs/adr/0017-claude-code-tool-taxonomy-transition.md` | Claude Code tool-taxonomy transition (v2.1.16/63) | **keep** |
| `docs/rfc/0001-transcript-pr-binding.md` | The provenance/binding design (706 lines) | **drop-prov** — becomes historical/archived if provenance is dropped |
| `docs/rfc/0002-vision-and-roadmap.md` | v3+ vision, data model, roadmap; **hard constraints incl. "No model substitution"** (Sonnet 4.6 + Cohere Embed v4) | **provider · hexagon** — the pluggable-provider direction directly contradicts constraint #3 ("No model substitution"); this RFC needs a v2 successor or an amendment |

**Cookbooks / notes / reference** (`docs/`):

| Doc | Covers | v2 change-type |
|---|---|---|
| `docs/README.md` | Doc-tree index, "one package, five layer sub-packages", built against a stale commit | **hexagon · drop-evals · drop-prov** — regenerate; layer list changes |
| `docs/BACKLOG.md` | Forward-looking un-promoted ideas | **keep** |
| `docs/cookbook.md` | v1 SQL recipes (sessions/messages/tool_calls/semantic_search) | **keep / provider** — semantic_search recipes may note provider-agnostic embeddings |
| `docs/analytics_cookbook.md` | v2 analytics-surface recipes (clusters/communities/classifications/trajectory/conflicts/friction) | **keep** — core analytics survive |
| `docs/research_notes.md` | Engine + analytics research narrative | **provider · drop-evals** |
| `docs/reference/cli.md` | Full CLI reference, "31 subcommands" | **drop-evals · drop-prov** — remove eval/provenance commands, re-count |
| `docs/reference/public-api.md` | Public-symbol surface, "five layer sub-packages (core, analytics, evals, provenance, app)" | **drop-evals · drop-prov · hexagon** — layer list + symbol ranking change |

**Generated architecture/analysis docs** (all say "five layer sub-packages" / cite
`evals`+`provenance`; all regenerate under the new layout):

| Doc | v2 change-type |
|---|---|
| `docs/architecture/system-overview.md` | **drop-evals · drop-prov · hexagon** |
| `docs/architecture/module-map.md` | **drop-evals · drop-prov · hexagon** |
| `docs/architecture/data-flow.md` | **provider · hexagon** |
| `docs/behavior/processes.md` | **drop-evals · drop-prov** (drops eval-gym + provenance-review processes) |
| `docs/behavior/state-machines.md` | **keep / hexagon** |
| `docs/analysis/dead-code.md` | **regen** |
| `docs/analysis/ownership.md` | **regen** (git-history stats) |
| `docs/analysis/risk-hotspots.md` | **regen** |
| `docs/insights/business-logic.md` | **drop-evals · drop-prov** (kappa/blind-handover rules go) |
| `docs/insights/contract-map.md` | **drop-evals · drop-prov · hexagon** (import-DAG contract changes) |
| `docs/insights/debugging-guide.md` | **hexagon** |
| `docs/insights/impact-analysis.md` | **hexagon · provider** |
| `docs/insights/tech-debt.md` | **regen** |
| `docs/diagrams/architecture/components.md` | **drop-evals · drop-prov · provider** |
| `docs/diagrams/behavioral/sequences.md` | **drop-evals · drop-prov** |
| `docs/diagrams/structural/dependency-graph.md` | **hexagon** |

**Not prose (leave / regenerate mechanically):** `docs/jsonl_schema_v1.sql`,
`docs/queries/thread_walk.sql` (**keep** — JSONL schema is immutable per RFC 0002),
`docs/.packets/*` (18 generated doc-gen input packets — **regen or delete**),
`docs/.repomix/*` (**stale — from removed repomix/codehub tooling; delete**).

> **`docs/v2/`** is currently **empty** — this file is the first artifact in it
> (`docs/v2/understanding/07-tests-ci-build-docs.md`).

---

## 6. The 3.13-vs-3.14 note

- **Floor confirmed:** `pyproject.toml:8` — `requires-python = ">=3.13"`. Agrees with
  `.python-version` (`3.13`) and `mise.toml:16` (`python = "3.13"`).
- **The actual 3.14 blocker is `hdbscan`, not numba.** Per `docs/adr/0015` and CLAUDE.md:
  every native dep ships cp314 wheels *except* `hdbscan` (stops at cp313); on 3.14 uv
  falls back to the sdist (Cython + C toolchain at install), which breaks the
  `uv tool install claude-sql` end-user path. Flip to 3.14 in a one-line PR once
  `hdbscan 0.8.43+` publishes cp314 wheels.
- **Separate numba/numpy ceiling** (the task's "ceiling comment") is `pyproject.toml:38`:
  ```
  "numpy>=2.4.4,<2.5",  # ceiling: numba (0.66.0, requires_dist numpy<2.5) caps numpy<2.5; without this uv --upgrade backsolves numpy 2.5 by downgrading numba to 0.53.1 (breaks umap/hdbscan)
  ```
  This pins numpy (numba caps it) — it is a *resolver* ceiling, distinct from the 3.14
  *interpreter* deferral. Both matter for v2, and both keep the umap/hdbscan clustering
  stack (which v2 KEEPS) intact.

> **v2 stays on 3.13.** Since v2 keeps retrieval+clustering (umap/hdbscan), the hdbscan
> cp314 constraint still binds. If v2 ever drops hdbscan for a different clusterer, the
> 3.14 door reopens — but that is not the stated direction.

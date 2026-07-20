# 05 — The Drop Analysis

**Perspective:** exactly what leaves `claude-sql` when the `evals/` plane is
removed, and whether `provenance/` should go with it.

**Verdict up front:**

- **Dropping `evals/` is a CLEAN drop.** The only cross-boundary edge into
  `evals/` is `app/cli.py`. Nothing in `core/` or `analytics/` imports it. The
  sibling-independence contract holds on the call graph.
- **Drop `provenance/` too.** It is a PR-provenance harness (transcript→commit
  binding + a Sonnet review-sheet), not retrieval/clustering. Its only external
  consumer is `app/cli.py` (three commands). Nothing in `core/` or `analytics/`
  depends on it. It does not serve "finding & reading transcripts" — it serves
  "attribute a merged PR back to the agent transcript that wrote it," which is
  eval/audit-adjacent baggage.
- **Zero shared runtime deps are lost.** Every third-party library `evals/` and
  `provenance/` import is also imported by KEEP code (`core`/`analytics`).
  Removable-only weight is `tiktoken`? No — see §5: the removable set is the two
  *already-dead* declared deps (`anthropic`, `scipy` as a direct pin), not
  anything evals/provenance-exclusive.

Method: every claim below is grounded in `codegraph impact` / `codegraph
callers` run inside the repo, cross-checked against `grep` of the import graph.

---

## 1. Blast radius of dropping `evals/`

### 1.1 Every cross-boundary edge INTO `evals/`

Grep of the whole `src/` tree for importers of `claude_sql.evals`, excluding
`evals/` itself:

```
src/claude_sql/app/cli.py:91:  from claude_sql.evals import (
                                   blind_handover as _blind_handover,
                                   freeze as _freeze,
                                   judges as _judge_catalog,
                               )
src/claude_sql/app/cli.py:2638: from claude_sql.evals import judge_worker as _judge_worker      # deferred, inside judge_cmd
src/claude_sql/app/cli.py:2691: from claude_sql.evals import ungrounded_worker as _ungrounded_worker  # deferred, inside ungrounded_cmd
src/claude_sql/app/cli.py:2732: from claude_sql.evals import kappa_worker as _kappa_worker      # deferred, inside kappa_cmd
```

**That is the complete list.** `app/cli.py` is the *only* module outside
`evals/` that names `claude_sql.evals`. There is **no** `core/*` or
`analytics/*` importer. There are **no** dynamic imports (`importlib` /
`__import__` over eval strings return nothing).

CodeGraph confirms nothing structural downstream. `codegraph impact freeze`
(the study entry point) touches only `evals/freeze.py` itself plus test files —
no `app`/`core`/`analytics` symbol:

```
$ codegraph impact freeze
Impact of changing "freeze" — affected symbols:
  src/claude_sql/evals/freeze.py  (self)
  tests/test_freeze.py  (8 test functions)
```

`codegraph impact run` (the `judge_worker.run` dispatch entry) shows **no**
`app/core/analytics` symbol affected (the CLI reaches it only through a
deferred, function-body import that the graph does not treat as a structural
edge). `codegraph callers run_async` / `callers plan` resolve entirely inside
`evals/judge_worker.py`. `callers compute_pairwise` / `compute_fleiss` /
`detect` (kappa + ungrounded entry points) return **no callers** in `src/` —
only the CLI command bodies reach them at runtime via deferred import.

### 1.2 CLI commands that get REMOVED

`app/cli.py` registers these eval commands (all deleted with the plane):

| Command | cli.py line | Wires to |
|---|---|---|
| `judges` (`judges_cmd`) | `cli.py:2519` | `evals.judges.catalog()` |
| `freeze` (`freeze_cmd`) | `cli.py:2539` | `evals.freeze.freeze` / `SessionScope` |
| `replay` (`replay_cmd`) | `cli.py:2581` | `evals.freeze.replay` |
| `blind-handover` (`blind_handover_cmd`) | `cli.py:2590` | `evals.blind_handover.strip_text` / `original_hash` |
| `judge` (`judge_cmd`) | `cli.py:2621` | `evals.judge_worker.run` (deferred) |
| `ungrounded-claim` (`ungrounded_cmd`) | `cli.py:2677` | `evals.ungrounded_worker.detect` (deferred) |
| `kappa` (`kappa_cmd`) | `cli.py:2716` | `evals.kappa_worker` (deferred) |

Seven commands. All also referenced in the `_default` help banner at
`cli.py:3159` (`"  judges | freeze | replay | judge | ungrounded-claim | kappa | blind-handover"`).

### 1.3 Sibling-independence contract — verified on the graph

`pyproject.toml` `[tool.importlinter]` declares two contracts:

1. Layers: `app` > `{analytics | evals | provenance}` > `core`.
2. Independence: `analytics`, `evals`, `provenance` are mutually independent.

**Both hold in reality.** The only claude_sql imports *inside* `evals/` are:

```
evals/judge_worker.py:48  from claude_sql.core.logging_setup import loguru_before_sleep
evals/judge_worker.py:49  from claude_sql.evals import judges as judge_catalog   # intra-evals
evals/judge_worker.py:50  from claude_sql.evals.judges import Judge              # intra-evals
evals/freeze.py:25        from claude_sql.evals import judges as judge_catalog   # intra-evals
```

So `evals/` reaches **down** only to `core.logging_setup` (a KEEP module) and
sideways only within itself. It reaches **zero** into `analytics/` and **zero**
into `provenance/`. The independence contract is not aspirational here — it is
factual. **Dropping `evals/` cannot break `core` or `analytics`** because no
edge runs in that direction.

---

## 2. What `evals/` actually does (so the rewrite can say what was removed)

`evals/` is a **pre-registered eval gym** for grading transcripts with a
cross-provider Bedrock judge panel. Per `evals/__init__.py`: *"judges panel,
freeze/replay, blind-handover, judge, ungrounded-claim, kappa."* The six files:

- **`judges.py` (239 LOC)** — a static catalog of cross-provider Bedrock judge
  models (`Judge` dataclass: shortname, model_id, provider, family, role).
  `resolve()`, `panel()`, `catalog()`, family selectors. Pure config, no I/O.
- **`freeze.py` (189 LOC)** — pre-registration. `freeze()` writes an immutable
  study manifest (`~/.claude/studies/<sha>/manifest.json` + rubric copy) hashing
  rubric + panel + git commit SHA; `replay()` loads it; `list_studies()`. This is
  the "freeze/replay manifest" gym primitive.
- **`judge_worker.py` (462 LOC)** — the dispatch engine. Parses a YAML rubric
  into `Axis`es, renders judge prompts, calls Bedrock Converse concurrently
  (`run_async`), scores sessions, writes a scores parquet. `plan()` produces a
  dry-run cost estimate (the cost-guard convention).
- **`kappa_worker.py` (336 LOC)** — inter-rater agreement. Cohen's + Fleiss'
  kappa with bootstrapped 95% CIs, plus a `delta_gate_excludes_zero()`
  pre-registered stopping rule. The `kappa` command exits `66` below the floor.
- **`ungrounded_worker.py` (191 LOC)** — hallucination detector. Extracts claims
  from assistant text and checks each is grounded in the tool-output text of the
  same turn.
- **`blind_handover.py` (155 LOC)** — grader-safety. `strip_text()` scrubs
  identity markers (Slack user/channel IDs, personas, session UUIDs) so a human
  grader can't be biased; `original_hash()` stamps a stable re-link key.

**Why it leaves:** this is the "gym" — freeze/replay, judge panels, Fleiss
kappa, blind handover, ungrounded-claim gating. It belongs to a dedicated eval
project. None of it participates in *finding or reading* transcripts.

---

## 3. `provenance/` — keep-or-drop decision: **DROP**

### 3.1 What the three files do

- **`binding.py` (743 LOC)** — implements RFC 0001. Pure-stdlib helpers that
  write and read a "three commit-trailer + JSON git-note" convention pointing a
  merged commit at the agent transcript that produced it. `build_binding`,
  `write_trailer`, `write_note`, `read_trailer`, `read_note`,
  `resolve_all_sources`, `resolve_commit_to_transcript` (precedence: trailer
  first, note fallback, loud on digest disagreement). Plus transcript-discovery
  helpers: `find_active_transcript`, `projectify`, `compute_digest`,
  `detect_agent_runtime`.
- **`review_sheet_worker.py` (465 LOC)** — resolves a commit's bound transcript,
  flattens the JSONL to text, and asks **Sonnet 4.6** (structured output via
  `PR_REVIEW_SHEET_SCHEMA`) to compress it into a ~1K-token PR review sheet.
- **`review_sheet_render.py` (167 LOC)** — pure Markdown renderers for the sheet
  and for refusals.

### 3.2 Who calls them (codegraph callers/impact)

Only `app/cli.py`, from three commands:

```
src/claude_sql/app/cli.py:96:  from claude_sql.provenance import binding as _binding
src/claude_sql/app/cli.py:97:  from claude_sql.provenance.review_sheet_render import render_markdown, render_refusal_markdown
src/claude_sql/app/cli.py:3073: from claude_sql.provenance.review_sheet_worker import generate_review_sheet  # deferred
```

| Command | cli.py line | Wires to |
|---|---|---|
| `bind` (`bind_cmd`) | `cli.py:2793` | `binding.build_binding` / `write_trailer` / `write_note` / `find_active_transcript` |
| `resolve` (`resolve_cmd`) | `cli.py:2926` | `binding.resolve_commit_to_transcript` / `resolve_all_sources` |
| `review-sheet` (`review_sheet_cmd`) | `cli.py:3032` | `binding.resolve_commit_to_transcript` + `review_sheet_worker.generate_review_sheet` + the render fns |

`codegraph impact generate_review_sheet` — the review-sheet entry — reaches only
`review_sheet_worker` itself, `cli.py:review_sheet_cmd`, and test files. No
`core`/`analytics` symbol:

```
$ codegraph impact generate_review_sheet
Impact of changing "generate_review_sheet":
  src/claude_sql/provenance/review_sheet_worker.py  (self)
  src/claude_sql/app/cli.py:3033  review_sheet_cmd
  tests/... (review-sheet tests)
```

**Crucially — the reverse-dependency check.** No retrieval/reading code depends
on `binding`. `codegraph callers` for every discovery helper returns callers
*only* inside `provenance/` (and tests), never `core`/`analytics`:

```
$ codegraph callers find_active_transcript   → (no src caller outside provenance)
$ codegraph callers projectify               → (no src caller outside provenance)
$ codegraph callers detect_agent_runtime     → (no src caller outside provenance)
$ codegraph callers compute_digest           → (no src caller outside provenance)
$ codegraph callers resolve_commit_to_transcript → _resolve_uri_for_commit (provenance), + tests
```

So even the tempting "but `find_active_transcript` / `projectify` sound like
retrieval" helpers are used **only** by `binding`/`review_sheet_worker` and the
CLI's `bind`/`resolve` commands. They are not load-bearing for `search`,
`embed`, `cluster`, `ingest`, or any reading path.

### 3.3 Does evals depend on provenance, or vice versa?

**Neither.** Grep and CodeGraph agree:

- `evals/*` claude_sql imports: only `core.logging_setup` + intra-`evals`
  (§1.3). **No `provenance` edge.**
- `provenance/*` claude_sql imports:
  ```
  review_sheet_worker.py:45  from claude_sql.core.llm_shared import (BedrockRefusalError, _build_bedrock_client, _invoke_classifier_sync)
  review_sheet_worker.py:50  from claude_sql.core.schemas import PR_REVIEW_SHEET_SCHEMA
  review_sheet_worker.py:51  from claude_sql.provenance.binding import resolve_commit_to_transcript  # intra-provenance
  review_sheet_worker.py:56  from claude_sql.core.config import Settings  # TYPE_CHECKING only
  ```
  Reaches **down** to `core` only; sideways only within itself. **No `evals`
  edge.**

The two planes are fully disjoint. They can be dropped independently or
together with no cross-repair.

### 3.4 The decision

The owner flagged provenance as "confirm drop," to lean drop unless load-bearing
for retrieval. **It is not load-bearing for retrieval.** The evidence:

1. Zero `core`/`analytics` importer (§3.2).
2. Zero reverse dependency from any reading/retrieval symbol (§3.2).
3. Its purpose is PR-audit provenance (git trailers/notes) + an LLM review-sheet
   — a "how do I attribute this merged PR to a transcript" harness. That is
   eval/audit-adjacent, the same gym-family concern as `evals/`, not the
   "find & read Claude Code transcripts" core mission.
4. `binding.py` is a git-trailer/git-note protocol (RFC 0001). The `bind`
   command is documented as a `prepare-commit-msg` hook entry point, but
   **`lefthook.yml` wires no such hook** — the binding path is not even active
   in the repo's own git flow. It is speculative infrastructure.

**Recommendation: DROP `provenance/` in the same cut as `evals/`.** It sheds 1375
LOC of source and ~1771 LOC of tests (§6) with zero impact on retrieval,
clustering, or reading.

*(If the team later wants transcript↔commit linking back, `binding.py` is a
clean, dependency-free stdlib module (743 LOC) that can be resurrected verbatim
into the eval/provenance project — it imports nothing but `core.logging`-free
stdlib. Nothing about dropping it now forecloses that.)*

---

## 4. Shared surface that MUST survive (do not over-remove)

These `core` symbols are imported by evals/provenance **and** by KEEP code, so
they stay:

| Shared symbol | Used by dropped code | Also used by KEEP code (stays) |
|---|---|---|
| `core.logging_setup.loguru_before_sleep` | `evals/judge_worker.py:48` | `analytics/embed_worker.py` retry path; `core.logging_setup` is core infra |
| `core.llm_shared` (`BedrockRefusalError`, `_build_bedrock_client`, `_invoke_classifier_sync`) | `provenance/review_sheet_worker.py:45` | **All analytics LLM workers** (`classify_worker`, `trajectory_worker`, `conflicts_worker`, `friction_worker`). `grep`: `llm_shared` has 19 KEEP importers vs 2 dropped. |
| `core.config.Settings` | `provenance/review_sheet_worker.py:56` (TYPE_CHECKING) | Every worker + `app/cli.py`. Core config. |
| `core.schemas.PR_REVIEW_SHEET_SCHEMA` / `PRReviewSheet` | `provenance/review_sheet_worker.py:50` | **Nobody else.** See note below. |

**Action on `core.schemas`:** `PRReviewSheet` (schemas.py:509) and
`PR_REVIEW_SHEET_SCHEMA` (schemas.py:580, in `__all__` at :584) are used **only**
by `provenance/review_sheet_worker.py` + `tests/test_schemas.py` +
`tests/test_review_sheet_worker.py`. They are *provenance-only schema surface
living in a core module.* When provenance drops:

- **Remove** `class Correction` (schemas.py:477), `class PRReviewSheet`
  (schemas.py:509), and `PR_REVIEW_SHEET_SCHEMA = _bedrock_schema(PRReviewSheet)`
  (schemas.py:580), plus their `__all__` entries (:584, :592).
- **Keep** everything else in `schemas.py`: `SessionClassification` /
  `SESSION_CLASSIFICATION_SCHEMA`, `TrajectoryArrayResult` /
  `TRAJECTORY_ARRAY_SCHEMA`, `ConflictsResult` / `SESSION_CONFLICTS_SCHEMA`,
  `UserFrictionSignal` / `USER_FRICTION_SCHEMA`, and the `_bedrock_schema` /
  `_flatten` helpers — all analytics-owned and KEEP.
- Prune the two `PR_REVIEW_SHEET_SCHEMA`/`PRReviewSheet` test blocks in
  `tests/test_schemas.py` (imports at :9/:16; asserts at :117, :127, :133,
  :156–157).

Everything else in `core/` (`llm_shared`, `config`, `logging_setup`,
`session_text`, `sql_views`, `lance_store`, `s3_source`, `parquet_shards`,
`checkpointer`, `home`, `output`, `retry_queue`) is untouched by the drop.

---

## 5. Dependency / extra cleanup

I checked every `[project.dependencies]` entry against the import graph (usage
count inside `evals/`+`provenance/` vs. outside). **No top-level dependency is
exclusive to evals/provenance** — every library the dropped code imports is also
imported by KEEP code:

| Dep | evals/prov importer | KEEP importer (why it stays) |
|---|---|---|
| `numpy` | `kappa_worker.py` | `analytics/{community,ingest,terms,cluster}_worker.py` |
| `boto3` / `botocore` | `judge_worker.py`, (`review_sheet_worker` via llm_shared) | `core/llm_shared.py`, `analytics/embed_worker.py` |
| `tenacity` | `judge_worker.py` | `core/llm_shared.py`, `core/logging_setup.py`, `analytics/embed_worker.py` |
| `polars` | `judge_worker`, `kappa_worker`, `ungrounded_worker` | 14 KEEP importers |
| `loguru` | `judge_worker`, `review_sheet_worker` | 19 KEEP importers (whole package logs via loguru) |
| `duckdb` | `review_sheet_worker` (TYPE_CHECKING) | 15 KEEP importers |
| `pyyaml` | `judge_worker.py:136` (lazy) | `analytics/skills_catalog.py:40` |

So the drop **does not** let you remove any currently-used runtime dep. Install
weight from the drop comes from the ~2949 LOC of source, not the dep tree.

**However — the drop is the right moment to remove two ALREADY-DEAD declared
deps** (fix-what-you-touch):

- **`anthropic>=0.40`** — declared in `[project.dependencies]` but
  `grep -rE '^\s*(import|from)\s+anthropic'` over `src/` and `tests/` returns
  **zero** direct imports (only string literals like model-id prefixes and
  `JudgeFamily = Literal["anthropic", ...]`). The package talks to Claude via
  **Bedrock/boto3**, not the `anthropic` SDK. This dep is dead today and should
  be dropped: `uv remove anthropic`.
- **`scipy>=1.13`** — declared as a *direct* dep but
  `grep -rE '^\s*(import|from)\s+scipy'` returns **zero** direct imports. `scipy`
  is a transitive dependency of `scikit-learn` / `umap-learn` / `hdbscan` (it
  appears in `uv.lock` as their child), so it will still be installed — but the
  *direct* pin in `pyproject.toml` is unnecessary and can be dropped:
  `uv remove scipy`. (Verify the lock still resolves scipy transitively after
  removal; it will, via sklearn/umap.)

Neither of these is caused by the evals/provenance drop, but both are cheap,
correct cleanups to fold into the same PR.

*(Do `uv remove` — never hand-edit `[dependencies]` — so `pyproject.toml` and
`uv.lock` stay in sync.)*

---

## 6. The clean-drop checklist

### 6.1 Source files to delete (7 + 3 = 10 files, ~2949 LOC)

```
src/claude_sql/evals/__init__.py
src/claude_sql/evals/judges.py
src/claude_sql/evals/freeze.py
src/claude_sql/evals/judge_worker.py
src/claude_sql/evals/kappa_worker.py
src/claude_sql/evals/ungrounded_worker.py
src/claude_sql/evals/blind_handover.py
src/claude_sql/provenance/__init__.py
src/claude_sql/provenance/binding.py
src/claude_sql/provenance/review_sheet_worker.py
src/claude_sql/provenance/review_sheet_render.py
```
(Delete the two directories entirely.)

### 6.2 `app/cli.py` edits

- Delete the module-top import block `cli.py:91–97` (the `from claude_sql.evals
  import (...)` tuple + the two `provenance` imports).
- Delete the deferred worker imports at `cli.py:2638`, `2691`, `2732`, `3073`.
- Delete these command functions and their `@app.command` decorators:
  - `judges_cmd` (2519), `freeze_cmd` (2539), `replay_cmd` (2581),
    `blind_handover_cmd` (2590), `judge_cmd` (2621), `ungrounded_cmd` (2677),
    `kappa_cmd` (2716) — the **evals** commands.
  - `bind_cmd` (2793), `resolve_cmd` (2926), `review_sheet_cmd` (3032),
    plus the `RenderFormat` StrEnum + `_review_sheet_format` helper (defined
    just above review_sheet_cmd) — the **provenance** commands.
- Fix the `_default` help banner (`cli.py:3158–3160`): remove the
  `judges | freeze | replay | judge | ungrounded-claim | kappa | blind-handover`
  and `bind | resolve | review-sheet` lines.
- Update the module-top NOTE comment (`cli.py:45–56`) that lists deferred
  eval/provenance workers.

### 6.3 `core/schemas.py` edits (over-removal guard, §4)

- Remove `Correction` (477), `PRReviewSheet` (509),
  `PR_REVIEW_SHEET_SCHEMA` (580), and their `__all__` entries (584, 592).
- Keep every other schema (classification/trajectory/conflicts/friction).

### 6.4 import-linter contract edits (`pyproject.toml`)

- **Layers contract** (`[[tool.importlinter.contracts]]` "claude-sql layered
  architecture"): change the L1 layer from
  `"claude_sql.analytics | claude_sql.evals | claude_sql.provenance"` to
  just `"claude_sql.analytics"`.
- **Independence contract** ("analytics / evals / provenance are independent
  siblings"): with only one L1 sibling left, the independence contract is moot —
  **delete the entire contract block**, or (if a placeholder is wanted) reduce
  `modules` to `["claude_sql.analytics"]` (a one-module independence contract is
  vacuous; deletion is cleaner).
- Update the explanatory comment block above `[tool.importlinter]`
  (currently "core (L0) < {analytics, evals, provenance} (L1 siblings) < app").
- The package-layout comment under `[tool.uv.build-backend]` ("five layer
  sub-packages (core / analytics / evals / provenance / app)") should read
  "three layer sub-packages (core / analytics / app)".

### 6.5 lefthook.yml edit

- Update the `lint-imports` job comment (`lefthook.yml`) that says
  "core (L0) < {analytics, evals, provenance} (L1 siblings) < app (L2)".
- No hook logic changes — `bind` was never wired as a `prepare-commit-msg` hook
  (confirmed: `lefthook.yml` has no `prepare-commit-msg` section).

### 6.6 Dependency removals

```
uv remove anthropic   # dead: zero direct imports, uses Bedrock/boto3
uv remove scipy       # dead direct pin: transitive via sklearn/umap/hdbscan
```
(No evals/provenance-exclusive dep exists to remove — see §5.)

### 6.7 Test files to delete (13 files, ~2886 LOC)

```
tests/test_judges.py                    (10)
tests/test_judges_extras.py             (4)
tests/test_freeze.py                    (9)
tests/test_judge_worker.py              (12)
tests/test_judge_worker_extras.py       (13)
tests/test_kappa_worker.py              (17)
tests/test_ungrounded_worker.py         (12)
tests/test_blind_handover.py            (11)
tests/test_binding.py                   (15)   ← also delete test_binding_module_imports_via_cli
tests/test_binding_extras.py            (23)
tests/test_review_sheet_render.py       (13)
tests/test_review_sheet_worker.py       (7)
tests/test_review_sheet_worker_extras.py (22)
```

### 6.8 Test edits (surgical — shared files that KEEP code also uses)

- **`tests/test_cli.py`** — delete the eval/provenance command tests:
  `test_review_sheet_format_*` (231, 235, 242), and everything in the
  `# judges / freeze / replay / blind-handover / kappa` block onward:
  `test_judges_cmd_lists_catalog` (703), `test_freeze_then_replay_roundtrip`
  (715), `test_freeze_empty_panel_raises` (744), `test_blind_handover_*`
  (753, 770), `test_judge_cmd_*` (780, 816), `test_ungrounded_cmd_*` (839, 868),
  `test_kappa_cmd_*` (889, 923), `test_bind_cmd_*` (963),
  `test_resolve_cmd_*` (987, 1005, 1022), `test_review_sheet_cmd_*` (1039, 1057,
  1069, 1100, 1159). Keep the `_resolve_settings` / `_resolve_memory_limit`
  tests (core CLI helpers).
- **`tests/test_schemas.py`** — remove the `PR_REVIEW_SHEET_SCHEMA` /
  `PRReviewSheet` imports (9, 16) and their assertions (117, 127, 133, 156).
- **`tests/test_pr3_perf.py`** — in `_CLI_FORBIDDEN_EAGER_IMPORTS`
  (the `test_cli_import_is_lean` guard), remove the four now-deleted entries:
  `claude_sql.evals.judge_worker`, `claude_sql.evals.kappa_worker`,
  `claude_sql.evals.ungrounded_worker`,
  `claude_sql.provenance.review_sheet_worker`.

### 6.9 Docs sections to cut

- **`docs/reference/cli.md`** — delete sections `## judges` (307), `## freeze`
  (316), `## replay` (334), `## blind-handover` (343), `## judge` (352),
  `## ungrounded-claim` (370), `## kappa` (384), `## bind` (399), `## resolve`
  (414), `## review-sheet` (428). (Everything from line 307 to EOF is the
  eval/provenance block; the KEEP CLI surface ends at `## analyze`, 276.)
- **`README.md`** — cut the "Provenance:" + "Eval gym:" lines in the Quick tour
  (236, 240–245); the **"Eval gym"** table (305) and **"Transcript ↔ PR
  provenance (RFC 0001)"** table (317) in the CLI-surface section; the design
  notes "Pre-registered eval gym" (606) and "Transcript ↔ PR provenance" (617)
  bullets; and the exit-code note about `kappa`/`review-sheet` (334).
- **`docs/rfc/0001-transcript-pr-binding.md`** — this RFC *is* the provenance
  design. Delete it (or move it to the eval/provenance project).
- The generated `docs/` analysis packets (`docs/.packets/*`, `docs/analysis/*`,
  `docs/architecture/*`, `docs/insights/*`, `docs/diagrams/*`) reference
  evals/provenance throughout — they are regenerated artifacts, so regenerate
  after the drop rather than hand-editing.

### 6.10 Post-drop validation gates

Run the full gate after the cut (every one is a blocker):

```
uv run lint-imports     # layered DAG must still pass with the reduced contract
uv run ty check         # no dangling references to removed schemas/modules
uv run ruff check .
uv run pytest -q        # test_cli_import_is_lean must stay green with the trimmed forbidden list
uv lock --check         # after uv remove anthropic scipy
```

---

## Appendix — CodeGraph evidence log

- `codegraph callers freeze` → 8 callers, **all in `tests/test_freeze.py`**.
- `codegraph callers replay` → 2, both `tests/test_freeze.py`.
- `codegraph callers run_async` / `callers plan` → 1 each, both
  `src/claude_sql/evals/judge_worker.py:430` (intra-evals).
- `codegraph callers compute_pairwise` / `compute_fleiss` / `detect` → **no
  callers** in `src/` (reached only via deferred CLI import).
- `codegraph callers strip_text` → 10, all `tests/test_blind_handover.py`.
- `codegraph callers resolve_commit_to_transcript` → `_resolve_uri_for_commit`
  (provenance) + tests; `codegraph impact` reaches only provenance + cli
  `review_sheet_cmd` + tests.
- `codegraph callers generate_review_sheet` → `cli.py:3033 review_sheet_cmd` +
  tests.
- `codegraph callers find_active_transcript` / `projectify` /
  `detect_agent_runtime` / `compute_digest` → **no `src` caller outside
  `provenance/`** (proves no retrieval-path dependency).
- `codegraph impact freeze` / `impact run` → **no `app`/`core`/`analytics`
  symbol** affected.
- Grep: `anthropic` and `scipy` have **zero** direct `import`/`from` statements
  in `src/` or `tests/`.

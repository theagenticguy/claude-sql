# Rumelt Packet — transcripts-as-compiler

**Status:** COMPLETE
**Authored by:** rumelt-architect
**Framing:** `framing.md`

---

## Challenge

Should we spend the next 90 days lifting `claude-sql` from a single-developer
transcript query tool into the substrate for **transcripts-as-PR-review** —
PR↔transcript binding, reviewer-facing compression, and a minimum team-corpus
ingestion path — or defer that lift and keep `claude-sql` a personal analytics
tool? The visible symptom driving the question is that AI-authored diffs are
outrunning the review loop (`framing.md` §Audience, lines 13–21). The root
cause candidate — which this packet will test — is that code review and
software archaeology assume the diff *is* the source, but for AI-authored code
the diff is compiled output and the real source (the transcript) is being
thrown away at `commit` time. "Solving this" means (1) naming the single
pivotal obstacle that decides whether transcript-as-source can become a
category primitive, (2) committing a 90-day capability budget to that
obstacle, and (3) saying out loud what this policy rules out.

---

## Diagnosis

### Symptoms

- **AI-authored PRs get skimmed, not reviewed.** GitHub Copilot Code Review
  ships LLM-generated line comments but emits no transcript provenance and no
  audit trail; reviews are advisory, not blocking (`framing.md` §Known-facts
  line 73, https://docs.github.com/en/copilot/using-github-copilot/code-review/using-copilot-code-review).
  Reviewers read the diff without the prompt→decision→tool-call path that
  produced it.
- **Incident retros can't answer "why did the agent do this?"** The
  transcripts that carry the reasoning trail (`~/.claude/projects/**/*.jsonl`
  plus subagent sidecars — `CLAUDE.md` "What this project is" + "Resilience
  patterns to preserve") are unbound to the commits they produced, so the
  blast-radius analysis hits a dead end at the squash-merge boundary.
- **Knowledge leaks when agents (or agent operators) leave.** Transcripts
  live under a user's `~/.claude/` — there is no multi-user ingestion path
  today (`framing.md` §Known-facts line 61; `CLAUDE.md` "Deferred decisions"
  Snapshot-tier entry). Pattern discovery across a team is blocked not by
  analytics — `claude-sql` already has clusters, communities, friction
  classification, stance-conflict detection — but by the absence of a team
  corpus to run them over.
- **Third-party session-replay tools (SpecStory, cursor-replay, vibe-replay)
  are already racing in but are not PR-bound.** (`framing.md` §Known-facts
  line 74, https://forum.cursor.com/t/built-a-cursor-extension-to-save-and-share-chat-and-composer-history/35314,
  https://vibe-replay.com/). They prove demand for the artifact; they do not
  close the binding gap.

### Root cause (the diagnosis)

**Code review, blame, and software archaeology all assume the source of truth
is what got checked in, but for AI-authored code the diff is compiled output
— the real source is the transcript — and every agent runtime today throws
the source away at `git commit` time.** This is not a "we need better
tooling" problem; it is a category error about *what the source of truth is*.
The file `main.py` after a Claude Code session is analogous to a `.o` file
shipped without the `.c` file: the artifact is technically buildable and
runnable, but every downstream activity that depends on causal reasoning —
review, rollback-justification, bisect-what-intent, new-hire onboarding,
governance of AI-authored change — degrades to "inspect the object code and
guess." `claude-sql` already proves the transcript is queryable at rest
(`CLAUDE.md` "What this project is" — 18 DuckDB views + Cohere Embed v4 on
global CRIS + Sonnet 4.6 structured output + UMAP/HDBSCAN/Louvain/c-TF-IDF +
friction classifier with regex fast-path + persistent HNSW at
`~/.claude/hnsw.duckdb` + sharded parquet caches). The intelligence layer
exists; the missing primitive is the pointer from a merged PR back to the
transcript(s) that produced it. Without that pointer, every downstream
capability — reviewer compression, incident replay, team forensics, fleet
analytics — operates on a disconnected corpus and degrades to "logs we happen
to have."

**Tradeoff named:** This diagnosis is a stronger claim than "transcripts are
useful context for reviewers." The stronger claim commits us to a substrate
position (define the source-of-truth binding); the weaker claim would let us
settle for a reviewer-sidebar feature (help review the object code better).
Chose the stronger claim because the weaker one has no defensible position
against GitHub Copilot Review's distribution (`framing.md` §Stakes line 38).

---

## The Crux

**Endorsed with sharpening:** The crux is defining and shipping a stable
PR↔transcript binding — a small, boring, commit-trailer-plus-git-notes
convention — because the primitives already exist as commodity git
infrastructure (RFC-documented since git 1.6, adopted by Gerrit, Linux
kernel, `Signed-off-by:`, `Co-authored-by:`), the pivot is adopting a
convention rather than inventing one, and every downstream capability
(reviewer compression, replay, forensics, team analytics) collapses to "logs
we happen to have" without it.

**Why this overrules noise, governance, and substrate-lift as candidate
cruxes:**

- *Noise* (transcripts are 10–100× the token volume of the diff) is a
  compression problem, and `claude-sql` already has the compressor — the
  Sonnet 4.6 structured-output classification pipeline with
  `output_config.format` (`CLAUDE.md` "Structured output (Sonnet
  classification)"), friction detection (`friction_worker.py`), stance
  conflict and trajectory scoring. Pointing existing analytics at a bound
  corpus is a product decision, not a research one.
- *Governance / IP / PII* is a redaction + policy surface — hard, but the
  same work is required whether the transcript is bound or unbound. Binding
  does not make governance harder; an unbound transcript carries the same
  risks with none of the accountability.
- *Substrate-lift* (multi-user ingestion, ACLs, retention) is a build cost.
  It decides *scale*, not *whether the thing works*. Binding decides whether
  the thing works.

**Surmountability evidence:** Commit trailers already carry arbitrary
structured key-value pairs and survive cherry-pick. `git notes` is
designed for exactly this use case; its known weakness (non-default push,
near-invisible in GitHub UI) is a UX problem a CLI + GitHub App can fix in
weeks, not a missing primitive. The 90-day bet is shaped exactly right to
ship a convention + a reference implementation; it is not shaped to
convince GitHub to ship a native feature.

---

## Guiding Policy

**Compete on being the neutral binding-plus-analytics layer that sits
between the agent runtime and the code host, by concentrating the 90-day
capability budget on (a) the PR↔transcript binding convention and (b) the
analytics that specifically require a bound corpus, because the agent-runtime
vendors (Anthropic, Cursor, Amp, Windsurf) and the code-host vendors (GitHub,
GitLab, Gerrit) will not agree on a cross-vendor binding on their own — each
has private incentives to lock transcripts inside their surface — and the
Wardley packet maps the binding component at Genesis, which is the only
evolution stage where neutrality is defensible (`wardley-packet.md` §Value
Chain line 44, §Gameplay move 1 line 73).**

Unpacked:

- **Compete on** — the binding convention + the thin analytics slice that
  *requires* a bound corpus to deliver value (PR-scoped review compression,
  repo-scoped forensics, team-scoped pattern discovery). Not the reviewer
  sidebar UI; not the IDE plugin; not the multi-tenant SaaS.
- **By concentrating** — capability budget on (a) spec + reference impl for
  the binding, (b) PR-sized compression of an existing transcript (reuse the
  Sonnet 4.6 pipeline in `llm_worker.py`), (c) a team-corpus ingestion path
  that is just past the single-`~/.claude/` assumption but stops far short
  of enterprise ACL/SOC2/retention-policy.
- **Because** — the binding primitive has no natural owner (agent vendors
  win by locking transcripts in; code-host vendors win by locking reviews
  in), and the category is Genesis today (`wardley-packet.md` Evolution table
  lines 42–44 — reviewer surface, analytics, and binding all flagged
  Genesis). The window is the 12–18 months before one of the incumbents
  notices the category and builds it vertically inside their surface (the
  `framing.md` §Stakes line 38 Red Queen observation).

**Defensible alternative ruled out by this policy:** "Race GitHub Copilot
Review / Cursor session-replay to the reviewer-sidebar feature." Rejected
because distribution is the incumbent's game and the sidebar is where they
will meet us; if the fight is "whose sidebar is prettier on github.com,"
GitHub wins on `MAUs × default-on × zero-install`. Policy forbids any
90-day action that prioritizes reviewer UX polish over the binding spec or
the team-corpus path.

**Second alternative ruled out:** "Boil the ocean — team ingestion +
identity + ACLs + retention engine + redaction pipeline all in Q1."
Rejected because it spends the capability budget on build cost that
doesn't answer the Crux. Policy forbids any action inside the 90-day
window whose value is conditional on enterprise readiness that we have
not yet proven anyone wants.

---

## Coherent Actions

Temporal frame: 90 days. Owner is Laith (solo) unless otherwise noted. Dates
are weeks-from-commit. Every action must reinforce the policy *and* at least
one other action; the coherence check at the bottom names the dependency
arrows.

1. **Binding spec v0 — RFC + reference impl.** By **week 2**. Author a
   `docs/rfc/0001-transcript-pr-binding.md` in the `claude-sql` repo
   defining a commit-trailer + git-notes convention that carries (a)
   `Claude-Transcript-Digest:` (sha256 over canonicalized JSONL), (b)
   `Claude-Transcript-URI:` (local path or object-store URI), (c)
   `Claude-Agent-Runtime:` (`claude-code/0.x`, `cursor/…`, `amp/…` —
   extension point), (d) standard `Author:` / `Co-authored-by:` carrying
   the *human* in the loop. Ship a reference `claude-sql bind` subcommand
   that attaches the trailer pre-commit and writes a `git notes --ref
   transcripts` entry. Reinforces: policy §"binding convention" — this
   *is* the policy's centerpiece.
2. **Reviewer compression — PR-sized review sheet.** By **week 6**.
   Extend the existing Sonnet 4.6 classification pipeline
   (`llm_worker.py`, `schemas.py`) with a new structured-output schema
   `PRReviewSheet` that emits, per bound transcript: (i) what the human
   asked for, (ii) what the agent explored, (iii) where it got corrected
   (lean on the existing stance-conflict detector in
   `ungrounded_worker.py` / `judge_worker.py`), (iv) what tools it used +
   refused, (v) a compressed diff-rationale. Expose as `claude-sql
   review-sheet --commit <sha>` → markdown. Reinforces: action 1 (needs
   the binding to resolve `--commit` to a transcript) and the policy §
   "analytics that require a bound corpus."
3. **Team-scope ingestion path v0.** By **week 10**. Lift the corpus
   assumption from `~/.claude/projects/**/*.jsonl` (one user) to
   `{object-store prefix}/<team>/<user>/projects/**/*.jsonl` (N users per
   team). Minimum viable: (a) a `Settings.team_corpus_root` config
   knob; (b) the DuckDB `read_json` glob parameterized over it (the
   existing resilience flags in `CLAUDE.md` "Resilience patterns" —
   `filename=true, union_by_name=true, sample_size=-1,
   ignore_errors=true, maximum_object_size=67108864` — already tolerate
   heterogeneous schemas); (c) an HNSW-per-team store under
   `{object-store}/<team>/hnsw.duckdb`; (d) a minimal ACL shim that just
   reads `<team>/members.json`. **Explicit non-goals:** no Okta/SSO, no
   SOC2 evidence pipeline, no retention policy engine, no redaction
   service. Reinforces: action 2 (team pattern discovery only makes
   sense on a team corpus) and the policy §"stops far short of
   enterprise."
4. **Reference integration — wire into one real repo's PR workflow.** By
   **week 12**. Install the `claude-sql bind` pre-commit hook in the
   `claude-sql` repo itself (dogfood) and at least one other repo Laith
   owns. Add a GitHub App (or Action, if App gate slips — cheaper
   fallback) that on PR-open reads the trailer, resolves the transcript,
   runs `review-sheet`, and posts the compressed review as a PR comment
   tagged `claude-sql/review-sheet`. Reinforces: all three prior
   actions — this is the acceptance test that the spec, the
   compressor, and the ingestion path compose end-to-end. Also the
   artifact that makes the RFC credible to external adopters.

### Explicit deletions (out of scope for the 90-day window)

Named so they do not re-enter the roadmap through the back door:

- **Enterprise ACL / Okta / SSO** — policy §"stops far short of enterprise."
- **SOC2 evidence pipeline.** Same reason. Revisit when a design-partner org
  asks for it in writing.
- **Retention policy engine.** Same reason.
- **Full redaction service.** A regex-based secrets scrub inside the binding
  step is in scope (half a day of work); a policy-driven redaction engine
  that understands IP, PII, third-party content, and per-org taxonomies is
  not.
- **IDE plugin (VS Code, JetBrains).** Distribution play we cannot win
  against Cursor / Copilot; the CLI + GitHub App covers the 90-day use
  case.
- **Web UI beyond the CLI.** The CLI's agent-friendly surface
  (`--format {auto,table,json,ndjson,csv}`, structured error exit codes
  64/65/70, `list-cache` parquet introspection — `CLAUDE.md`
  "Agent-friendly CLI surface (load-bearing — do not regress)") is the
  product for the 90-day window. A reviewer-facing web UI is a post-90d
  call, contingent on the GitHub App dogfood surfacing demand.
- **Multi-tenant SaaS.** Object-store + per-team prefix is the
  substrate; a hosted multi-tenant plane is out of scope.

### Coherence check

Dependency arrows, drawn explicitly:

- Action 1 (binding) **is the foundation for** actions 2, 3, 4.
- Action 2 (compression) **consumes** action 1's binding to resolve
  `--commit → transcript`.
- Action 3 (team corpus) **widens the substrate** that actions 1 and 2
  operate over. Without it, the binding ships but only one user
  benefits; analytics that need inter-author patterns degrade to
  "patterns within Laith's own work."
- Action 4 (reference integration) **is the acceptance test**:
  binding + compression + team corpus must compose into a PR comment
  that is *actually useful*. If week 12 ships and reviewers ignore the
  comment, the diagnosis is wrong and the strategy must reset.

**Tensions checked:**

- Action 3 (team corpus) vs. policy §"stops far short of enterprise" —
  tension resolved by the explicit-deletions list. v0 is object-store
  glob + `members.json` ACL shim; anything beyond that is out of scope
  for the quarter.
- Action 4 (GitHub App) vs. policy §"neutral binding layer" —
  implementing on GitHub first is a distribution choice, not a
  vendor-lock choice. The binding spec (action 1) is host-agnostic; the
  App is the *first* integration, chosen for reach, not the *only*
  integration. Flagged here so the RFC doesn't drift GitHub-specific.
- Action 2 (compression using existing analytics) vs. policy §
  "concentrate capability budget" — resolved by reusing the existing
  Sonnet 4.6 pipeline rather than building a new compressor. Marginal
  cost of one additional schema in `schemas.py` + one worker entry
  point; not a new subsystem.

---

## Tactical sub-bets (from CLAUDE.md deferred decisions)

These live *under* the strategic thesis. Each is scored against the guiding
policy — accelerate, defer, or kill — with reason tied to the policy.

| Item | Call | Reason |
| ---- | ---- | ------ |
| **Python 3.14 flip** (blocked on `hdbscan 0.8.43+` cp314 wheels — ADR 0015 §"Consequences — Hdbscan cp314 watch" line 128) | **Defer** | Unblocks `uv tool install` UX on 3.14-default hosts. Does not move the binding or compression needles. Flip when the wheel lands; do not prioritize ahead of week-2 RFC or week-6 compression. |
| **Snapshot tier / `CacheNode` DAG** (`CLAUDE.md` "Deferred decisions" line "Snapshot tier") | **Defer — but expect action 3 to force it.** | CLAUDE.md gates on "~10 GB or `EXPLAIN ANALYZE` showing JSONL re-scan dominating." A team corpus (action 3) will cross one or both thresholds inside 90 days. Do *not* pre-build the DAG; let week-10 ingestion surface the actual bottleneck, then spend a week on the DAG if needed. Keeps the capability budget on the Crux, not on speculative plumbing. |
| **`CreateModelInvocationJob` batch embeddings + vector quantization** (`CLAUDE.md` "Deferred decisions" line "CreateModelInvocationJob") | **Defer.** | Global CRIS on `global.cohere.embed-v4:0` at `CLAUDE_SQL_EMBED_CONCURRENCY=8` already saturates without throttling (`CLAUDE.md` "Bedrock" + "Environment variables"). A team corpus adds N×, not 100×, embeddings volume in 90 days — tenacity + concurrency bump covers it. Reopen if embeddings parquet crosses ~10 GB (same trigger CLAUDE.md already names). Kills nothing. |
| **`CLAUDE_SQL_CONCURRENCY` env alias removal** (`CLAUDE.md` "Environment variables" — "DEPRECATED: aliases onto both pipelines with a `DeprecationWarning`. Removed in the next release.") | **Accelerate.** | One-line delete. Ship with the next point release (before the RFC so that external adopters don't encounter the deprecated knob in documentation). Zero risk, removes a reader-confusing surface that would muddy the substrate story. |
| **Scorecard `pinning-dependencies`** (ADR 0016 §"Decisions worth calling out — Action pinning" line 65, §"Follow-up" line 88) | **Accelerate — contingent on action 4.** | The moment we ask external repos to trust a `claude-sql bind` pre-commit hook + GitHub App (actions 1 and 4), the supply-chain posture of `claude-sql` itself becomes part of the sales story. Tag pins (`@v4`, `@v5`) were "acceptable for v1" when the audience was Laith; they are not acceptable when the audience is "a repo owner installing our App." Pin to Dependabot-managed SHAs across `ci.yml`, `codeql.yml`, `semgrep.yml`, `osv.yml`, `scorecard.yml`, `sbom.yml`, `commitlint.yml` before week 12. |

**One re-prioritization the strategy forces that the list above does not
capture:** the team-corpus lift (action 3) changes the release-hygiene
profile of `claude-sql`. Today every parquet cache and HNSW store is
per-user under `~/.claude/`; tomorrow they are per-team under an object
store. The existing `claude-sql cache compact` / `claude-sql cache
migrate` commands (`CLAUDE.md` "Resilience patterns to preserve") must
grow a team-scope variant by week 10. Not a new tactical item — an
expansion of an existing one.

---

## Bad-Strategy Checks

Explicit pass/fail on Rumelt's four hallmarks [1]:

| Hallmark | Pass/Fail | Evidence |
| -------- | --------- | -------- |
| **Fluff** — buzzwords masking substance | **Pass, with two candidates flagged and defended** | The phrase "transcripts-as-compiler" is load-bearing metaphor, not fluff: the diagnosis cashes it out concretely ("the diff is compiled output; the real source is the transcript; we throw the source away at commit") and the actions follow from that claim. The phrase "first-class artifact" is fluff *unless* paired with the specific binding primitive — so the packet uses it only where action 1's commit-trailer + git-notes convention cashes out what "first-class" means operationally. Banned words from `framing.md` §Voice-and-audience ("revolutionize," "next-generation," "unlock value," "leverage synergies") do not appear. |
| **Failure to face the challenge** | **Pass** | The real obstacle — "there is no stable pointer from a merged PR to the transcript that produced it, and the agent-vendor and code-host incumbents each have incentives not to build one" — is named in the Diagnosis and is the subject of the Crux and action 1. The policy explicitly *rules out* the escape route (the reviewer-sidebar race) most strategies would default to. |
| **Mistaking goals for strategy** | **Pass** | Every action has a specific artifact and a date. "Binding spec v0" = RFC markdown in `docs/rfc/0001-*.md` by week 2. "Reviewer compression" = a new `PRReviewSheet` pydantic schema in `schemas.py` consumed by `llm_worker.py` by week 6. "Team-scope ingestion" = a `Settings.team_corpus_root` knob + parameterized DuckDB glob + per-team HNSW by week 10. "Reference integration" = a GitHub App posting `claude-sql/review-sheet` comments on a real repo by week 12. None of these are aspirations ("be the category leader") dressed up as strategy. |
| **Bad strategic objectives (scattered / blue-sky)** | **Pass** | Four actions, each reinforcing at least one other (coherence check above). The explicit-deletions list (enterprise ACL, SOC2, retention, IDE plugin, web UI, multi-tenant SaaS) is longer than the action list — the kitchen-sink risk is actively rejected in writing. Tactical sub-bets score 2 accelerate / 3 defer / 0 kill, which matches Rumelt's prescription that a good strategy concentrates resources rather than hedging across a long list. |

---

## Alternatives Considered

Two alternative guiding policies, each rejected with a specific reason tied
to the diagnosis.

1. **"Build the reviewer-facing product first, figure out binding later."**
   Policy shape: compete on reviewer UX by concentrating the 90-day
   capability budget on a polished PR sidebar that summarizes the latest
   in-progress session, because reviewers are the pain-point and the binding
   can be backfilled. **Rejected** because the diagnosis says the diff is
   *compiled output whose source is the transcript* — an unbound
   reviewer-sidebar product is, by the diagnosis's own logic, a better
   inspector for `.o` files, not a compiler-source linker. It also has no
   defensible position against GitHub Copilot Review's distribution
   (`framing.md` §Stakes line 38, `wardley-packet.md` §Climatic-patterns
   line 63 — Red Queen on developer-productivity products). The reviewer UI
   is a feature that fits inside the incumbent's surface; the binding is a
   primitive that does not.
2. **"Open-source the whole thing as a standard and let the ecosystem win."**
   Policy shape: concentrate the capability budget on a polished
   specification + a set of vendor-neutral Apache-2 libraries, because
   the category will commoditize in 12–18 months and first-mover on the
   spec is worth more than first-mover on a product. **Rejected** because
   a spec without a reference integration has no adoption vector — the
   `wardley-packet.md` §Gameplay-move 1 "open the binding primitive"
   proposal only makes sense if there is already a working implementation
   that proves the primitive's shape. Specifications without running code
   get rewritten by the first vendor with distribution (Anthropic,
   GitHub). Action 4 — wire it into one real repo — is how the spec
   survives contact with reality. This alternative skips that step and
   cedes the reference-implementation slot to whoever ships first.

**Why these two and not "stay personal-scoped, publish substrate as OSS
only":** that third alternative was considered in the framing (`framing.md`
§Voice + the open-questions list) and is implicitly ruled out by the
Challenge section's premise that we are deciding *whether to invest the
quarter*. "Stay personal" is the null hypothesis, not a competing guiding
policy, so it is tested by the Stakes analysis (`framing.md` §Stakes —
"option value of not betting"), not by a second-best policy.

---

## Attribution Note

The Rumelt packet contributed the diagnosis that source-of-truth has
inverted for AI-authored code — the diff is compiled output and the
transcript is the source, which today is discarded at `git commit` time —
and sharpened the candidate crux from the framing into a one-sentence,
surmountable pivot: ship a PR↔transcript binding as a commit-trailer +
git-notes convention, because the primitives already exist and the pivot
is adopting a convention rather than inventing one. The guiding policy
commits to neutral binding-plus-analytics between the agent runtime and
the code host, explicitly ruling out the reviewer-sidebar race that a
non-diagnostic strategy would default to. The coherent-actions set is
four artifacts with dates (binding RFC by week 2, PR review-sheet by
week 6, team-corpus ingestion by week 10, reference GitHub App by week
12) and an explicit-deletions list (enterprise ACL, SOC2, retention,
IDE plugin, web UI, multi-tenant SaaS) that is longer than the
action list on purpose. The tactical sub-bets surface the CLAUDE.md
deferred items as accelerate / defer calls, with the Scorecard pinning
posture flipped to *accelerate* the moment we ask external repos to
trust our binding hook + GitHub App.

---

## Citations

- [1] Rumelt, R. *Good Strategy / Bad Strategy.* 2011.
- [3] Rumelt, R. *The Crux.* 2022.
- [Additional sources cited inline above.]

---

When every section has real content, flip `Status:` to `COMPLETE`.

# Strategy Memo — transcripts-as-compiler

**Status:** COMPLETE (revision round 1 applied)
**Authored by:** strategy-synthesizer
**Last updated:** 2026-05-08
**Framing:** `framing.md`
**Packets composed:** rumelt-packet.md, wardley-packet.md, minto-outline.md

---

## Executive Summary

**Recommendation:** Bet the next 90 days on turning `claude-sql` into the **transcript-bound review substrate for AI-authored PRs** — a stable PR↔transcript binding (commit-trailer + `git notes` convention), a reviewer-sized compression of the bound transcript, and a minimum team-corpus ingestion path — and keep consuming everything below the LLM API without modification, even when team-corpus scale tempts a home-grown vector index or embedding pipeline.

**Situation + complication.** Agents now author real production code and the transcripts that produced the diff — `~/.claude/projects/**/*.jsonl` plus subagent sidecars (`CLAUDE.md §What this project is`; `framing.md §Known-facts #2`) — exist on disk and are cheap to capture. But PR review, incident forensics, and software archaeology still treat the diff as the source of truth, and for AI-authored code the diff is compiled output whose real source is the transcript — the prompt→tool-call→correction→diff path that produced the merged commit (`rumelt-packet §Diagnosis — Root cause`). Meanwhile GitHub Copilot Review, Graphite Diamond, Greptile, and Cursor's third-party session-replay extensions (SpecStory, cursor-replay, vibe-replay) are racing to ship LLM-over-diff review without transcript provenance (`framing.md §Known-facts #8`, 2026-05-08; `wardley-packet §Evolution Axis` — reviewer surface at Genesis visibility 0.05).

**Question + answer.** The question is whether to lift `claude-sql`'s already-shipping analytics substrate (18 DuckDB views, Cohere Embed v4 on global CRIS, Sonnet 4.6 structured output, UMAP+HDBSCAN, Louvain, friction classifier, persistent HNSW, sharded parquet caches — `CLAUDE.md §What this project is` + `§Resilience patterns to preserve`) into the transcript-bound review layer, or stay personal-scoped. **The answer is lift it, and the binding is the crux.** The pivotal primitive is the PR↔transcript binding; the primitives to build it (commit trailers, `git notes`) are commodity git infrastructure shipped since git 1.6; and the window closes the moment an incumbent agrees with themselves on a vendor-locked convention.

Supporting arguments:

1. **Diagnosis + substrate are coherent.** The review failure mode for AI-authored code is a category error about what the source of truth is, not a volume problem; `claude-sql` already ships the analytics layer that makes transcripts reviewable, so the missing 10% is binding + multi-user ingestion, not a re-platform (`minto-outline §Supporting argument 1`; `rumelt-packet §Diagnosis`; `CLAUDE.md §What this project is`).
2. **The map says invest above the LLM API, and the window is short.** Reviewer-facing surface, PR↔transcript binding, and bound-transcript analytics sit in the Genesis/Custom band; every component from the agent runtime down is Product or Commodity with ≥7 providers per row (`wardley-packet §Map-implied Diagnosis`; `wardley-packet §Evolution Axis`). The Red Queen on AI PR review is active and the window is the 12–18 months before an incumbent ships a vendor-locked binding (`wardley-packet §Climatic Patterns #4`).
3. **The tactical backlog ranks under the thesis.** Every item in `CLAUDE.md §Deferred decisions` either gets forced by the team-corpus lift (snapshot tier, batch embeddings) and sequenced under action 3, defers cleanly (3.14 flip), or ships as hygiene with a specific unlock condition (Scorecard SHA-pinning becomes table stakes the moment external repos install our GitHub App) (`rumelt-packet §Tactical sub-bets`; `minto-outline §Supporting argument 3`).

---

## Diagnosis

**Symptoms** (`rumelt-packet §Diagnosis — Symptoms`):

- **AI-authored PRs get skimmed, not reviewed.** GitHub Copilot Code Review ships LLM-generated line comments but emits no transcript provenance and no audit trail; reviews are advisory, not blocking (https://docs.github.com/en/copilot/using-github-copilot/code-review/using-copilot-code-review, 2026-05-08). Reviewers read the diff without the prompt→decision→tool-call path that produced it.
- **Incident retros can't answer "why did the agent do this?"** Transcripts that carry the reasoning trail (`~/.claude/projects/**/*.jsonl` plus subagent sidecars — `CLAUDE.md §What this project is` + `§Resilience patterns to preserve`) are unbound to the commits they produced; blast-radius analysis dead-ends at the squash-merge boundary.
- **Team knowledge leaks when agent operators leave.** Transcripts live under a user's `~/.claude/` — no multi-user ingestion path exists (`CLAUDE.md §Deferred decisions` — Snapshot tier entry). Pattern discovery across a team is blocked not by analytics (`claude-sql` already ships clusters, communities, friction classification, stance-conflict detection) but by the absence of a team corpus to run them over.
- **Third-party session-replay tools are racing in but are not PR-bound.** SpecStory, cursor-replay, vibe-replay export chat/composer history to HTML or markdown (https://forum.cursor.com/t/built-a-cursor-extension-to-save-and-share-chat-and-composer-history/35314, https://vibe-replay.com/, 2026-05-08). They prove the artifact has demand; they do not close the binding gap.

**Root cause** (`rumelt-packet §Diagnosis — Root cause`, reinforced by `wardley-packet §Map-implied Diagnosis`): Code review, blame, and software archaeology all assume the source of truth is what got checked in. For AI-authored code the diff is **compiled output** — the real source is the transcript — and every agent runtime today throws the source away at `git commit` time. This is a category error about *what the source of truth is*, not a tooling gap. The file after a Claude Code session is analogous to a `.o` file shipped without the `.c`: technically runnable, but every downstream activity that depends on causal reasoning (review, rollback-justification, bisect-what-intent, onboarding, governance of AI-authored change) degrades to "inspect the object code and guess." Operationally, the metaphor is cashed out by the commit-trailer + `git notes` binding — the pointer from `.o` back to `.c` that every agent runtime currently discards at `git commit` time. The Wardley map makes the same diagnosis from a different angle — the top three components (reviewer surface, PR↔transcript binding, bound-transcript analytics) sit at Genesis/Custom with no shipping product, no standard, and no category, while everything from the agent runtime down is Product or Commodity with ≥7 providers per row (`wardley-packet §Evolution Axis`, 2026-05-08). The split is the central fact: the intelligence layer exists, the primitive that points from a merged PR to the transcript that produced it does not.

**The Crux:** The pivotal, surmountable obstacle is defining and shipping a stable PR↔transcript binding as a boring, commodity convention — a commit trailer (`Claude-Transcript-Digest:`, `Claude-Transcript-URI:`, `Claude-Agent-Runtime:`) plus a `git notes --ref transcripts` entry — because the primitives already exist (RFC-documented since git 1.6, adopted by Gerrit, Linux kernel, `Signed-off-by:`, `Co-authored-by:`), the pivot is adopting a convention rather than inventing one, and every downstream capability (reviewer compression, replay, team forensics, fleet analytics) collapses to "logs we happen to have" without it (`rumelt-packet §The Crux`).

---

## Guiding Policy

**Compete on being the neutral binding-plus-analytics layer that sits between the agent runtime and the code host, by concentrating the 90-day capability budget on (a) the PR↔transcript binding convention and (b) the analytics that specifically require a bound corpus, because no incumbent has an incentive to ship a cross-vendor binding and the map places that binding layer at Genesis — the only evolution stage where neutrality is defensible** (`rumelt-packet §Guiding Policy`; reinforced by `wardley-packet §Map-implied Diagnosis` — binding sits at the boundary between Genesis-we-shape and Product-we-consume).

Unpacked:

- **Compete on** — the binding convention + the thin analytics slice that *requires* a bound corpus to deliver value (PR-scoped review compression, repo-scoped forensics, team-scoped pattern discovery). Not the reviewer-sidebar UI, not the IDE plugin, not the multi-tenant SaaS.
- **By concentrating** — capability budget on (a) spec + reference implementation for the binding, (b) PR-sized compression using the existing Sonnet 4.6 pipeline in `llm_worker.py` + `schemas.py`, (c) a team-corpus ingestion path that is just past the single-`~/.claude/` assumption but stops far short of enterprise ACL / SOC2 / retention-policy.
- **Because** — the binding primitive has no natural owner (agent vendors win by locking transcripts in; code-host vendors win by locking reviews in), the category is Genesis today across reviewer surface, binding, and bound-transcript analytics (`wardley-packet §Evolution Axis` visibility rows 0.05–0.25), and the Red Queen on AI PR review is active (`wardley-packet §Climatic Patterns #4` — Graphite Diamond, Greptile, Copilot Review, Cursor Bugbot all reading the diff, none reading the transcript). The window is the 12–18 months before one incumbent ships a vendor-locked binding inside their surface.

**Defensible alternative ruled out #1 — "Race GitHub Copilot Review / Cursor session-replay to the reviewer-sidebar feature."** Rejected because distribution is the incumbent's game; if the fight is "whose sidebar is prettier on github.com," GitHub wins on `MAUs × default-on × zero-install` (`rumelt-packet §Guiding Policy` — defensible alternative; `framing.md §Stakes`). The policy forbids any 90-day action that prioritizes reviewer UX polish over the binding spec or the team-corpus path.

**Defensible alternative ruled out #2 — "Boil the ocean — team ingestion + identity + ACLs + retention + redaction all in Q1."** Rejected because it spends the capability budget on build cost that doesn't answer the Crux (`rumelt-packet §Guiding Policy` — second alternative). The policy forbids any action inside the 90-day window whose value is conditional on enterprise readiness that nobody has asked for in writing.

---

## Coherent Actions

Temporal frame: 90 days. Owner is Laith (solo) unless otherwise noted. Dates are weeks-from-commit (`rumelt-packet §Coherent Actions`). Each action reinforces at least one other; Wardley's gameplay moves are folded in where they add concrete shape.

1. **Binding spec v0 — RFC + reference implementation.** By **week 2**. **Who:** Laith. **What:** Author `docs/rfc/0001-transcript-pr-binding.md` in the `claude-sql` repo defining a commit-trailer + `git notes` convention carrying (a) `Claude-Transcript-Digest:` (sha256 over canonicalized JSONL), (b) `Claude-Transcript-URI:` (local path or object-store URI), (c) `Claude-Agent-Runtime:` (`claude-code/0.x`, `cursor/…`, `amp/…` — extension point), (d) standard `Author:` / `Co-authored-by:` carrying the human in the loop. Ship a reference `claude-sql bind` subcommand that attaches the trailer pre-commit and writes a `git notes --ref transcripts` entry. Publish the spec under Apache-2 / CC-BY — Wardley's Gameplay Move 1 "open the binding primitive" (`wardley-packet §Gameplay Moves #1`) says commoditize the layer below us on purpose, so the analytics above it is where the moat compounds. Reinforces: policy §"binding convention" — this *is* the policy's centerpiece (`rumelt-packet §Coherent Actions #1`).

2. **Reviewer compression — PR-sized review sheet.** By **week 6**. **Who:** Laith. **What:** Extend the existing Sonnet 4.6 classification pipeline (`llm_worker.py`, `schemas.py`) with a new structured-output schema `PRReviewSheet` that emits, per bound transcript: (i) what the human asked for, (ii) what the agent explored, (iii) where it got corrected (reuse the existing stance-conflict detector in `ungrounded_worker.py` / `judge_worker.py`), (iv) which tools it used and refused, (v) a compressed diff-rationale. Expose as `claude-sql review-sheet --commit <sha>` → markdown. Wardley's Gameplay Move 2 "double-down on analytics while the category is Genesis" (`wardley-packet §Gameplay Moves #2`) makes this the amplification move — we already have 80% of the primitives, this is the one schema + one worker entry point that turns them into a reviewer-facing artifact. Reinforces: action 1 (needs binding to resolve `--commit → transcript`) and the policy §"analytics that require a bound corpus" (`rumelt-packet §Coherent Actions #2`).

3. **Team-scope ingestion path v0 — minimum viable corpus lift.** By **week 10**. **Who:** Laith. **What:** Lift the corpus assumption from `~/.claude/projects/**/*.jsonl` (one user) to a multi-user root via (a) a `Settings.team_corpus_root` config knob, (b) the DuckDB `read_json` glob parameterized over it (the existing resilience flags — `filename=true, union_by_name=true, sample_size=-1, ignore_errors=true, maximum_object_size=67108864` — already tolerate heterogeneous schemas, `CLAUDE.md §Resilience patterns to preserve`), and (c) a smoke test on a synthetic 2-user fixture directory proving views + macros resolve across authors. **Explicit non-goals inside the 90 days:** no Okta/SSO, no SOC2 evidence pipeline, no retention policy engine, no redaction service, **no HNSW-per-team store and no `cache compact` / `cache migrate` team variants** — those defer to post-90-day, triggered by team-corpus production use surfacing the actual bottleneck (per `CLAUDE.md §Deferred decisions`, embeddings parquet crossing ~10 GB is the named reopen-trigger). **Explicit non-goals from `rumelt-packet §Coherent Actions — Explicit deletions`** still apply. The scope cut is deliberate: items (a)+(b)+(c) are what a solo Principal can ship in 4 weeks on top of actions 1 and 2, which are also solo-Laith. Reinforces: action 2 (team pattern discovery only works on a team corpus, even a 2-user one) and the policy §"stops far short of enterprise" (`rumelt-packet §Coherent Actions #3`).

4. **Reference integration — wire it into one real repo's PR workflow.** By **week 12**. **Who:** Laith, with dogfood on the `claude-sql` repo itself + one other repo Laith owns. **What:** Install the `claude-sql bind` pre-commit hook in both repos. Ship a GitHub App (or Action if App-gate slips — cheaper fallback) that on PR-open reads the trailer, resolves the transcript, runs `review-sheet`, and posts the compressed review as a PR comment tagged `claude-sql/review-sheet`. This is Wardley's Gameplay Move 4 "eat the code host's lunch on provenance — be the Codecov-shaped layer" (`wardley-packet §Gameplay Moves #4`) — neutral across agent vendors because the binding is, and meeting the reviewer in the code host rather than the IDE where the incumbents live. Reinforces: all three prior actions — it's the acceptance test that binding + compression + team corpus compose end-to-end. If week 12 ships and reviewers ignore the comment, the diagnosis is wrong and the strategy resets (`rumelt-packet §Coherent Actions #4` + Coherence check).

   **Policy coherence check (action 4 vs. policy §"reviewer-sidebar race" rule-out).** The binding spec (action 1) is host-agnostic — it is a commit-trailer + `git notes` convention that any code host or review tool can consume, not a GitHub feature. The GitHub App is the *first* integration we build for reach, not the only one; it exists because GitHub is where the reviewers are, not because the strategy is GitHub-specific (`rumelt-packet §Coherence check` — "implementing on GitHub first is a distribution choice, not a vendor-lock choice"). And the App competes on *transcript provenance* — the bound `.c` for the merged `.o` — which GitHub Copilot Review and every other LLM-over-diff incumbent structurally cannot emit, because none of them have the transcript. It does not compete on comment quality or sidebar polish (which is the race the policy forbids on line 50). Flagged here so the RFC and App implementation stay host-neutral and provenance-first.

**Explicit deletions (out of scope for the 90-day window)** (`rumelt-packet §Coherent Actions — Explicit deletions`): enterprise ACL / Okta / SSO; SOC2 evidence pipeline; retention policy engine; full redaction service (a regex secrets scrub inside the binding step is in scope, a policy-driven redaction engine is not); IDE plugin (VS Code, JetBrains); web UI beyond the CLI; multi-tenant SaaS. The list is deliberately longer than the action list — the kitchen-sink risk is actively rejected in writing.

---

## Build / Buy / Partner Read

Per-component call tied to evolution stage, lifted from `wardley-packet §Build-vs-Buy Read`.

| Component | Stage | Call | Rationale |
| --- | --- | --- | --- |
| Reviewer-facing surface | Genesis | **Build** | No product exists. The differentiated layer — it's the whole point of the bet. |
| Transcript analytics | Custom | **Build + extend** | `claude-sql` already ships the engine (`CLAUDE.md §What this project is`). Extend with PR-scoped views + compression artifact; do not rewrite. |
| PR ↔ transcript binding | Genesis | **Build + open** | Build the reference implementation; publish the spec under CC-BY / Apache-2 so the binding commoditizes below us. |
| Transcript artifact schema | Custom → early Product | **Consume + extend** | Claude Code JSONL exists for our corpus; SpecStory for Cursor; OTel GenAI for everyone else once it stabilizes. Do not invent a rival schema. |
| Agent runtime | Product | **Consume** | Claude Code, Cursor, Windsurf, Copilot via OTel GenAI — ≥7 good options; building our own is motion without differentiation. |
| LLM API | Product → Commodity | **Consume** | Bedrock global CRIS stays the wiring. |
| Embeddings | Product / Commodity | **Consume** | Cohere Embed v4 on `global.cohere.embed-v4:0` CRIS saturates without throttle (`CLAUDE.md §Bedrock`); no reason to switch. |
| Vector index | Product | **Consume (with sidecar watch)** | DuckDB VSS works for single-user; at team-corpus scale (~10 GB embeddings parquet threshold per `CLAUDE.md §Deferred decisions`), revisit pgvector or Turbopuffer. Still a consume either way. |
| Identity / ACL | Commodity | **Consume** | GitHub org membership + AWS IAM. |
| Storage / retention | Commodity | **Consume** | S3 with lifecycle policies. |
| Foundation models | Product | **Consume** | Opus 4.7 / Sonnet 4.6 via Bedrock. |
| Compute | Commodity | **Consume** | EC2 / Lambda / Bedrock-managed. |

**Contradiction check** (`wardley-packet §Contradictions flagged`): one borderline case — the persistent HNSW wrapper at `~/.claude/hnsw.duckdb` (mtime-checked, self-healing) is custom glue over a Product-stage primitive. Justified as a 2-month bridge; delete when DuckDB VSS productizes persistence or team-corpus forces pgvector (which ships persistent HNSW as a first-class feature).

---

## Tactical sub-bets

These rank *under* the strategic thesis. Each is the existing CLAUDE.md-deferred item scored against the guiding policy (`rumelt-packet §Tactical sub-bets`).

| Item | Call | Reason |
| --- | --- | --- |
| **Python 3.14 flip** — blocked on `hdbscan 0.8.43+` cp314 wheels (ADR 0015 §"Consequences — Hdbscan cp314 watch"; `.erpaval/INDEX.md` lesson `hdbscan cp314 wheel gap`) | **Defer** | Unblocks `uv tool install` UX on 3.14-default hosts. Does not move binding, compression, or ingestion. Flip when wheels land; do not prioritize ahead of week-2 RFC or week-6 compression. |
| **Snapshot tier / `CacheNode` DAG** (`CLAUDE.md §Deferred decisions`) | **Defer — but expect action 3 to force it** | CLAUDE.md gates on "~10 GB or `EXPLAIN ANALYZE` showing JSONL re-scan dominating." A team corpus (action 3) crosses one or both inside 90 days. Do *not* pre-build — let week-10 ingestion surface the actual bottleneck, then spend a week on the DAG. Keeps the capability budget on the Crux. |
| **`CreateModelInvocationJob` batch embeddings + vector quantization** (`CLAUDE.md §Deferred decisions`) | **Defer** | Global CRIS on `global.cohere.embed-v4:0` at `CLAUDE_SQL_EMBED_CONCURRENCY=8` already saturates without throttling (`CLAUDE.md §Bedrock` + `§Environment variables`). A team corpus adds N×, not 100×, volume in 90 days; tenacity + concurrency bump covers it. Reopen at ~10 GB embeddings parquet — same trigger CLAUDE.md already names. |
| **`CLAUDE_SQL_CONCURRENCY` env alias removal** (`CLAUDE.md §Environment variables` — DEPRECATED) | **Accelerate — ship as hygiene** | One-line delete. Ship with the next point release before the RFC so external adopters don't encounter the deprecated knob in documentation. Zero risk, removes a reader-confusing surface that muddies the substrate story. |
| **Scorecard `pinning-dependencies` posture** (ADR 0016 §"Decisions worth calling out — Action pinning" + §"Follow-up") | **Accelerate — unconditional** | Ship SHA pins by week 12 regardless of App adoption; it's cheap, one-shot work and removes a future blocker. Tag pins (`@v4`, `@v5`) were acceptable when the audience was Laith; they become a liability the first time any external repo or reviewer audits the supply chain — and with the action 1 RFC published and action 4 dogfooded, that first audit is a matter of when, not if. Pin to Dependabot-managed SHAs across `ci.yml`, `codeql.yml`, `semgrep.yml`, `osv.yml`, `scorecard.yml`, `sbom.yml`, `commitlint.yml` before week 12. |

**Strategy-forced re-prioritization not captured in the table above** (`rumelt-packet §Tactical sub-bets` — closing paragraph): the team-corpus lift eventually changes the release-hygiene profile. Today parquet caches + HNSW stores are per-user under `~/.claude/`; under a production team corpus they become per-team under an object store. Per the action 3 scope cut, `claude-sql cache compact` and `claude-sql cache migrate` team-scope variants (`CLAUDE.md §Resilience patterns to preserve`) **defer to post-90-day, triggered by team-corpus production use** — not a week-10 deliverable. Not a new tactical item either — an expansion of an existing one, sequenced after the v0 corpus glob lands.

---

## Risks

Consolidated across framing + packets. Honest mitigation where it exists; flagged unmitigated otherwise.

1. **Incumbent adoption risk — GitHub, Cursor, or Anthropic ships a native binding first.** `wardley-packet §Climatic Patterns #4` — Red Queen on AI PR review is active; Graphite Diamond, Greptile, Copilot Review, Cursor Bugbot all shipped in the last 12 months. **Mitigation (two-part).** (i) The spec is open under Apache-2 / CC-BY with a reference implementation (Wardley's Gameplay Move 1 — "commoditize the binding layer below us before an incumbent locks it up"), so if an incumbent ships a compatible primitive, the analytics above it ports rather than stranding. (ii) Action 4 is the adoption-vector bet — by week 12 we have a working binding operating on a real repo's PR workflow, which is what survives a 6-month GitHub native-binding ship: a spec document alone gets rewritten by whoever has distribution; a spec + running integration competes on transcript provenance that the incumbent's LLM-over-diff feature structurally cannot emit. **Residual risk:** if an incumbent ships a binding *and* bundles the analytics (GitHub Code Review adding transcript provenance + LLM summary in one feature), the moat compresses. The 12-week cadence on actions is calibrated against this.

2. **Governance / IP / PII surface.** Transcripts carry secrets pasted into prompts, rejected paths, third-party IP, inter-agent communications (`framing.md §Audience — pushback #2`; `framing.md §Open questions — Governance and redaction surface`). **Partial mitigation:** a regex-based secrets scrub inside the `claude-sql bind` step is in scope for the 90-day window (`rumelt-packet §Coherent Actions — Explicit deletions` names a regex scrub as in-scope; a policy-driven redaction engine is not). **Unmitigated beyond that:** IP / third-party content / per-org taxonomies need a redaction service we explicitly ruled out for the quarter. This is a deliberate deferral, not a gap — named in the memo so a design-partner conversation reopens it. Flagging for readers: the 90-day bet ships *safe-by-default on secrets*, not *audit-grade on IP*.

3. **Adoption-vector risk — spec without integration gets rewritten by whoever has distribution.** `rumelt-packet §Alternatives Considered #2` — the "open-source the whole thing as a standard" alternative was rejected precisely because specs without running code get rewritten by the first vendor with distribution. **Mitigation:** action 4 wires the spec into one real repo's PR workflow inside the 90 days. This is the acceptance test — if week 12 ships and reviewers ignore the comment, the diagnosis is wrong and the strategy resets (`rumelt-packet §Coherent Actions — Coherence check`). The GitHub App carries the distribution vector; a spec document alone does not.

4. **Team-corpus lift slips — action 3 v0 blows past week 10.** `CLAUDE.md §Deferred decisions` flags snapshot-tier and batch-embeddings as likely-forced once the corpus crosses ~10 GB. **Mitigation:** action 3's v0 scope is deliberately trimmed to what a solo Principal can ship in 4 weeks on top of actions 1+2 — `Settings.team_corpus_root` + parameterized `read_json` glob + 2-user synthetic-fixture smoke test. HNSW-per-team store, `cache compact` / `cache migrate` team variants, ACL shims, Okta/SSO, SOC2, retention, and redaction are all explicit non-goals for the 90-day window. If even the trimmed lift slips, actions 1+2+4 still compose on the single-user corpus — team pattern discovery degrades to "patterns within Laith's own work," but the binding + compression + GitHub App still ship as the acceptance test.

5. **Reviewer ignores the review-sheet.** Named as the acceptance test in `rumelt-packet §Coherent Actions — Coherence check` — if week 12 ships and reviewers ignore the posted comment, the diagnosis is wrong. **Mitigation:** compression target is ~1K tokens of structured output (per action 2), uses the existing Sonnet 4.6 + stance-conflict detector that `claude-sql` has already validated on the personal corpus. **Residual risk:** reviewer cognition is the hardest thing to measure and the bet accepts this. Instrument PR-comment reactions + read-through time once the GitHub App is live; plan to iterate the schema if the week-12 dogfood shows low engagement.

> *Delivery-risk footnote, not a thesis-level risk.* `ty` is alpha and strict (`[tool.ty.rules] all = "error"` per ADR 0015); a `ty` regression can block CI and any release supporting the RFC or App. Redundant CI gates (`semgrep.yml`, `codeql.yml`, `osv.yml`, pytest via the `pre-push` lefthook) catch regressions without blocking semantic checks, and ADR 0015 named the rollback paths when the tradeoff was accepted. Called out here so it doesn't surprise if Python 3.14 + `ty` 0.x interact poorly mid-quarter.

---

## Evidence

Deduplicated across framing + packets. Newest-first; external URLs grounded via nova_web_grounding on 2026-05-08 unless otherwise noted.

| # | Source | Date | Used by |
| --- | --- | --- | --- |
| 1 | GitHub Copilot Code Review — https://docs.github.com/en/copilot/using-github-copilot/code-review/using-copilot-code-review | 2026-05-08 | Diagnosis, Risks #1, Market gap |
| 2 | Cursor Forum — session-history extension thread — https://forum.cursor.com/t/built-a-cursor-extension-to-save-and-share-chat-and-composer-history/35314 | 2026-05-08 | Diagnosis, Market gap |
| 3 | vibe-replay — https://vibe-replay.com/ | 2026-05-08 | Diagnosis, Market gap |
| 4 | Graphite Diamond — https://diamond.graphite.dev/ and https://graphite.com/blog/introducing-graphite-agent-and-pricing | 2026-05-08 | Climatic patterns #4, Risks #1 |
| 5 | Greptile PR reviewer benchmarks — https://www.greptile.com/benchmarks | 2026-05-08 | Climatic patterns #4 |
| 6 | Graphite 2025 PR reviewer landscape — https://graphite.com/guides/best-ai-pull-request-reviewers-2025 | 2026-05-08 | Climatic patterns #4 |
| 7 | AI coding tools 2026 — tldl — https://www.tldl.io/resources/ai-coding-tools-2026 | 2026-05-08 | Climatic patterns #1, agent-runtime commoditization |
| 8 | AI coding tools 2026 — verdent — https://www.verdent.app/guides/ai-coding-tools-comparison-2026 | 2026-05-08 | Climatic patterns #1 |
| 9 | AI coding agents 2026 — codersera — https://codersera.com/blog/ai-coding-agents-complete-guide-2026/ | 2026-05-08 | Climatic patterns #1 |
| 10 | Cursor / Windsurf / Replit / Claude Code comparison — marsdevs — https://www.marsdevs.com/compare/cursor-vs-windsurf-vs-replit-claude-code | 2026-05-08 | Climatic patterns #1 |
| 11 | OpenTelemetry GenAI semantic conventions — https://opentelemetry.io/docs/specs/semconv/gen-ai/ and https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans | 2026-05-08 | Transcript schema evolution, Build/buy — transcript artifact schema |
| 12 | OTel GenAI tracing + PII — https://maketocreate.com/opentelemetry-genai-tracing-ai-agents-without-leaking-pii/ | 2026-05-08 | Governance / redaction context |
| 13 | Vector DB provider landscape — pecollective — https://pecollective.com/tools/best-vector-databases/ | 2026-05-08 | Build/buy — vector index |
| 14 | Vector DB landscape — mixpeek — https://mixpeek.com/curated-lists/best-vector-databases | 2026-05-08 | Build/buy — vector index |
| 15 | Vector DB landscape — encore — https://www.encore.dev/articles/best-vector-databases | 2026-05-08 | Build/buy — vector index |
| 16 | AI code review tools — smartremotegigs — https://smartremotegigs.com/ai-code-review-tools/ | 2026-05-08 | Climatic patterns #4 |
| 17 | Rumelt, R. *The Crux.* Public Affairs | 2022 | Crux framing |
| 18 | Minto, B. *The Pyramid Principle.* Pearson (current edition) | 2021 | Executive Summary structure |
| 19 | Rumelt, R. *Good Strategy / Bad Strategy.* Crown Business | 2011 | Diagnosis / Guiding Policy / Coherent Actions kernel |
| 20 | Wardley, S. *Learn Wardley Mapping* — https://learnwardleymapping.com/book/ and https://learnwardleymapping.com/climate/ | CC-BY-SA, ongoing | Evolution axis + climatic patterns |
| 21 | Wardley map reference — Wikipedia — https://en.wikipedia.org/wiki/Wardley_map | ongoing | Evolution axis |
| 22 | Expert Program Management, "Barbara Minto Pyramid Principle" — https://expertprogrammanagement.com/2022/11/barbara-minto-pyramid-principle/ | 2022 | Pyramid reference |
| 23 | `CLAUDE.md` — project root — §What this project is, §Bedrock, §Structured output (Sonnet classification), §Friction classifier, §Resilience patterns to preserve, §Analytics pipeline determinism, §Environment variables, §Deferred decisions, §Agent-friendly CLI surface (load-bearing — do not regress) | repo, current | Substrate evidence throughout |
| 24 | ADR 0015 — Python floor and hdbscan cp314 watch | repo, current | Tactical sub-bet #1, Risks #6 |
| 25 | ADR 0016 — GitHub Actions CI hardening + action-pinning posture | repo, current | Tactical sub-bet #5 |
| 26 | `framing.md` — §Challenge, §Crux read, §Stakes, §Known facts, §Open questions | repo, current | Framing throughout |

---

## Convergence Notes

**Divergences the memo made a call on (read this first — it's the answer to "did you hide anything?").** Rumelt scoped 4 coherent actions tight; Wardley's Gameplay Moves implied 5, including a fifth "eat the code host's lunch on provenance" move that Rumelt's policy §"concentrate the capability budget" explicitly rules out as a standalone action. The memo takes Rumelt's discipline — four actions — and folds Wardley's move 4 (the GitHub App carrying the "be the Codecov-shaped layer" distribution vector) into action 4 rather than spawning a fifth. Separately, Minto collapsed the brief's five candidate argument groups to three, which landed the adoption-vector question inside Group 2 ("propose the binding primitive openly") rather than as its own group. The memo accepts Minto's collapse and treats adoption-vector as the *rationale* for action 1's "publish under Apache-2 / CC-BY" shape, not as a separate fifth action. Net: the memo is prescriptive — Rumelt's action count, Wardley's gameplay moves folded in as refinement, Minto's MECE groups as the SCQA skeleton — rather than a comparative read of three packets.

**Convergence:** all three packets landed independently on the PR↔transcript binding as the pivotal layer. Rumelt endorsed the framing's candidate crux with a sharpening (commit trailer + `git notes` because the primitives ship with git 1.6+) and named surmountability as the reason it's the crux and not the other three candidates (`rumelt-packet §The Crux`). Wardley arrived at the same claim from the map — binding sits exactly at the Genesis-Product boundary, which is "the component where climatic patterns intersect most sharply" (`wardley-packet §Map-implied Diagnosis`). Minto's three-group MECE structure organized *around* binding — Group 1 "why is this bet coherent?" treats binding as the missing 10%, Group 2 "where and when to spend?" names binding as the unclaimed Genesis component (`minto-outline §Supporting argument 1` + `§Supporting argument 2`). Three different lenses, one answer.

---

## Revision Notes

**Round 1 — applied 2026-05-08 against `review-strategy.md` (Round 1, Needs revision: 2 critical / 5 warnings / 4 suggestions).**

1. **C1 — Action 4 vs. policy reconciliation (applied).** Added a "Policy coherence check" sub-paragraph inside action 4 reproducing `rumelt-packet §Coherence check`: binding spec is host-agnostic, GitHub App is the *first* integration chosen for reach not the only one, App competes on transcript provenance (which Copilot Review structurally cannot emit) not on comment quality. Reader no longer sees contradiction against the line-50 sidebar-race rule-out.

2. **C2 — Action 3 scope cut (applied).** Rewrote action 3 to what a solo Principal can ship in 4 weeks atop actions 1+2: `Settings.team_corpus_root` + parameterized `read_json` glob + smoke test on a synthetic 2-user fixture. Explicitly deferred HNSW-per-team store and `cache compact` / `cache migrate` team variants to post-90-day, triggered by team-corpus production use. Updated tactical sub-bets closing paragraph to match (compact/migrate team variants now "post-90-day, triggered by team-corpus production use"). Softened Risk 4 language to reflect the trimmed v0 scope.

3. **W1 — "Consume below the LLM API" sharpened (applied).** Executive Summary recommendation now reads "keep consuming everything below the LLM API without modification, even when team-corpus scale tempts a home-grown vector index or embedding pipeline" — names the temptation being resisted so it's a choice, not a restatement of status quo.

4. **W2 — Metaphor operationalization in Diagnosis (applied).** Added the sentence "Operationally, the metaphor is cashed out by the commit-trailer + `git notes` binding — the pointer from `.o` back to `.c` that every agent runtime currently discards at `git commit` time." immediately after the `.o`/`.c` analogy. Metaphor is now anchored to the primitive, not to a vibe.

5. **W3 — Risk 1 mitigation rewritten (applied).** Replaced the mitigation block with the two-part structure: (i) spec is open under Apache-2 / CC-BY so analytics port if an incumbent ships a compatible primitive; (ii) action 4 is the adoption-vector bet — a working binding on a real repo by week 12 is what survives a 6-month GitHub native-binding ship. Answers the specific "GitHub ships in 6 months" pushback instead of the generic open-spec hand-wave.

6. **W4 — Scorecard row made unconditional (applied).** Tactical sub-bets row 5 now reads "Accelerate — unconditional. Ship SHA pins by week 12 regardless of App adoption; cheap and removes a future blocker." (Option (a) in the critic's fix — tighter than conditional language.)

7. **W5 — Executive Summary density (applied).** Broke the SCQA paragraph into three paragraphs: (i) one-sentence recommendation, (ii) "Situation + complication" with 2–3 sentences, (iii) "Question + answer" with 1–2 sentences ending in the bolded "The answer is lift it, and the binding is the crux" call.

8. **S3 — Convergence Notes divergence promoted (applied).** Moved the divergence paragraph to the top with a bolded lead ("Divergences the memo made a call on (read this first — it's the answer to 'did you hide anything?')."). Convergence now follows.

9. **S4 — Risk 6 demoted (applied).** Cut Risk 6 from a numbered risk entry to a blockquoted footnote under the risks section, flagged explicitly as "Delivery-risk footnote, not a thesis-level risk." Memo stays shorter; strategic risks keep the numbered list.

None declined. C1 and C2 (the blocking critical findings) are both resolved; all 5 warnings and both applicable suggestions (S3, S4) are applied.

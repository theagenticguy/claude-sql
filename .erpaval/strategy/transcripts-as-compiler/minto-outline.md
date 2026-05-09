# Minto Outline — transcripts-as-compiler

**Status:** COMPLETE
**Authored by:** minto-pyramid-builder
**Framing:** `framing.md`
**Packets consulted:** `rumelt-packet.md` (IN PROGRESS — scaffold only; cited by §-heading), `wardley-packet.md` (IN PROGRESS — scaffold only; cited by §-heading)

> **Note on in-flight siblings.** At time of writing, both sibling packets are skeleton-only with bracketed placeholders under every decision section. Citations below reference the *scaffold sub-headings* the architect/cartographer are working under (e.g. `rumelt §Diagnosis`, `wardley §Build-vs-Buy Read`) so the synthesizer can follow the trail once those sections land. Where the concrete claim comes from the framing itself, the framing is cited directly.

---

## Answer (top of pyramid)

Spend the next quarter turning `claude-sql` into the transcript-bound review substrate for AI-authored PRs — committing to (a) a stable PR↔transcript binding, (b) a reviewer-facing compression of the transcript into a review-sized artifact, and (c) a minimum multi-user / team-corpus ingestion path — while consuming everything below the LLM API (Bedrock, embeddings, vector index, agent runtime).

---

## SCQA framing

- **Situation** — agents now author real production code. In every team that has adopted Claude Code, Cursor, Copilot Workspace, or Amp, PRs are landing whose diffs were written by the agent and whose human author pressed "merge." This is already table stakes for the audience (`framing.md` §Audience — "Agents write production code now; that's table stakes"). The transcripts that produced those diffs exist on disk and are cheap to capture — `~/.claude/projects/**/*.jsonl` plus subagent sidecars today, analogous artifacts from every other agent tomorrow (`CLAUDE.md` §What this project is; `framing.md` §Known facts #2).

- **Complication** — PR review, incident forensics, and team knowledge management all assume **code is the source**. For AI-authored code the real source is the *transcript* — the prompt→tool-call→correction→diff path that produced the merged commit — and today every team throws that source away once the PR merges. Reviewers read LLM output without the context that produced it (`framing.md` §Audience, pushback item 1 "Noise"). Meanwhile GitHub, Cursor, Graphite, Sourcegraph and the agent vendors are racing to ship "LLM-over-diff" code review in 2025–2026 — GitHub Copilot Code Review is GA; Cursor's Bugbot and third-party session-replay extensions (SpecStory, cursor-replay, vibe-replay) are early-stage — **but none of them bind the review to the transcript that produced the diff** (`framing.md` §Known facts #8, cited 2026-05-08 URLs). The commoditization wave on LLM APIs (`wardley-packet.md §Climatic Patterns` item 1 "Agent runtime commoditization") is pushing value up into the artifact and analytics layers, and the artifact layer is still Genesis (`wardley-packet.md §Evolution Axis`, reviewer-surface + binding at visibility 0.1–0.25).

- **Question** — do we invest the next quarter lifting `claude-sql`'s already-shipping analytics substrate (18 DuckDB views, Cohere Embed v4 on global CRIS, Sonnet 4.6 structured-output classification, UMAP+HDBSCAN clustering, Louvain communities, friction detection, persistent HNSW, sharded parquet caches — `CLAUDE.md` §What this project is + §Resilience patterns to preserve) into the transcript-bound review layer, or do we not? (`framing.md` §Challenge, verbatim 90-day bet.)

- **Answer** — yes, bet the quarter. The bet is concentrated on **binding + compression + minimal team ingestion**, with everything below the LLM API (Bedrock, embeddings, vector index, agent runtime) consumed not built (`wardley-packet.md §Build-vs-Buy Read` — Consume calls for LLM API / embeddings / vector index / agent runtime). Binding is the unclaimed Genesis-stage component — no shipping product owns it — and the incumbents are racing toward compression-without-provenance, which means the window for a neutral, cross-vendor binding is real and short.

---

## Supporting argument 1 — Diagnosis + substrate: transcripts are a first-class source, and we already have 90% of what's needed to treat them that way

**Claim:** The PR-review failure mode for AI-authored code is not "reviewers are overloaded" — it is that reviewers are handed the *compiled output* (the diff) and asked to reason about intent (the prompts, corrections, rejected paths) that was never attached. `claude-sql` already ships the analytics layer that makes transcripts reviewable; the missing 10% is binding + multi-user ingestion, not re-platforming.

**Evidence:**

- **Transcripts are causal, not log-like.** A transcript contains the prompt→tool-call→correction→diff causal chain; `git log` and CI logs capture neither the prompt nor the rejected paths. The diagnosis is about *source*, not volume — which is why "reviewers will drown in transcript tokens" (`framing.md` §Audience pushback #1) is a compression problem, not an intrinsic problem. Source: `rumelt-packet.md §Diagnosis` (root cause slot) once the architect fills it; the crux-read in `framing.md` §Crux read ("diagnosis is about source") is the grounded version today.

- **Transcripts are addressable.** Each message, tool call, todo, and subagent invocation has a UUID; `claude-sql`'s 18 DuckDB views already normalize these into queryable shape with zero-copy `read_json(...)` over the JSONL corpus (`CLAUDE.md` §What this project is #1–2; §Resilience patterns to preserve — `read_json(..., filename=true, union_by_name=true, sample_size=-1, ignore_errors=true, maximum_object_size=67108864)`). Addressable means bindable — each transcript has a handle a commit can point at.

- **Transcripts are compressible to a review-sized artifact.** Sonnet 4.6 with Bedrock GA `output_config.format` structured output already drives the classification, trajectory, stance-conflict, and friction pipelines (`CLAUDE.md` §Structured output (Sonnet classification); §Friction classifier (`friction_worker.py`)). Cohere Embed v4 via `global.cohere.embed-v4:0` CRIS embeds every message; DuckDB VSS HNSW serves semantic search in milliseconds; UMAP+HDBSCAN + c-TF-IDF name clusters; Louvain groups sessions into communities — all shipping, all deterministic under `CLAUDE_SQL_SEED=42` (`CLAUDE.md` §Analytics pipeline determinism). A reviewer-facing compression is a product decision on which of these to surface, not a research problem.

- **The substrate handles corpus reality.** Persistent HNSW at `~/.claude/hnsw.duckdb` (lazy-built, mtime-checked, self-heals on corruption), sharded parquet caches under `<cache>/part-<ts_ns>.parquet`, session-level checkpointers + per-uuid anti-join making reruns free, tenacity retries absorbing Bedrock throttling, and DuckDB tuned via `CLAUDE_SQL_DUCKDB_*` env vars — these are already load-bearing for a 2 GB / 15K-JSONL personal corpus (`CLAUDE.md` §Resilience patterns to preserve; §Environment variables). A team-corpus lift inherits this, not replaces it.

- **Tradeoff named:** this group collapses two candidate groups from the brief (A "diagnosis is about source" and D "substrate already exists") because both answer the same question for the reader — *why is this bet coherent?* Keeping them separate read as padding; collapsing sharpens the argument that diagnosis and substrate are two halves of one answer.

*MECE hint: this group answers "why does this bet make sense at all?" — the diagnostic and asset case. It does not touch where-to-spend (Group 2) or which-deferred-items-move (Group 3).*

---

## Supporting argument 2 — The map says invest above the LLM API, and the window is real and short

**Claim:** On the Wardley map of this value chain, LLM API, embeddings, vector index, and agent runtime are Product or Commodity — consume them. Reviewer surface, transcript analytics, and especially PR↔transcript binding are Genesis — build them. The competitive window is real: GitHub, Cursor, and the agent vendors have the distribution but none of them has shipped the bound artifact yet. Budget follows stage, and the stage says spend above the API *now*.

**Evidence:**

- **The map's stage-to-budget mapping is unambiguous.** Reviewer-facing surface at Genesis (visibility 0.1), transcript-to-PR binding at Genesis (visibility 0.25), transcript analytics at Genesis/Custom (visibility 0.2) — all three are Build calls in `wardley-packet.md §Build-vs-Buy Read`. LLM API at Product→Commodity, embeddings at Product/Commodity, vector index at Product, agent runtime at Product, identity at Commodity, storage at Commodity — all Consume (`wardley-packet.md §Evolution Axis` and §Build-vs-Buy Read). Running our own Bedrock replacement or our own vector DB would be "building a commodity" — the contradiction the build-vs-buy check is explicitly meant to catch (`wardley-packet.md §Contradictions flagged` scaffold).

- **The climatic pattern is commoditization pulling value up-stack.** Agent-runtime commoditization (2024 ~3 credible coding agents; 2026 a dozen-plus — Claude Code, Cursor, Amp, Copilot Workspace, Windsurf, Cline, Aider, Graphite Diamond) is pulling the transcript artifact toward Product stage with it (`wardley-packet.md §Climatic Patterns` item 1). The LLM API is already a line item, not a differentiator. Every dollar spent on LLM-API-layer engineering today is a dollar spent on a commodity. The only layer still open to a differentiated play is above the API.

- **Binding is unclaimed territory on a verified 2026-05-08 scan.** GitHub Copilot Code Review ships LLM-generated PR comments tied to specific lines/commits but **does not emit transcript provenance as a public artifact** (`framing.md` §Known facts #8; https://docs.github.com/en/copilot/using-github-copilot/code-review/using-copilot-code-review). Cursor has no first-party session replay bound to PRs; SpecStory / cursor-replay / vibe-replay are early-stage third-party extensions that emit HTML or markdown exports, none PR-bound (`framing.md` §Known facts #8; https://forum.cursor.com/t/built-a-cursor-extension-to-save-and-share-chat-and-composer-history/35314, https://vibe-replay.com/). Claude Code has no native `/session replay` inside PRs — the corpus lives on-disk and `claude-sql` is the only thing querying it. Net: **transcript-as-first-class-PR-artifact with stable binding is unclaimed** as of the research date.

- **The window is not permanent.** The Red Queen pattern on developer-productivity products — GitHub, Cursor, Sourcegraph, Graphite all racing to ship LLM-over-diff review (`wardley-packet.md §Climatic Patterns` item 4) — means an incumbent can eat this if they agree on a convention or if one of them ships a binding primitive before we do. The gameplay move is to **propose the binding primitive openly** — a standard commit trailer / git-notes schema (`wardley-packet.md §Gameplay Moves` candidate 1) — so the binding becomes commodity infrastructure and the fight shifts up to analytics where `claude-sql` already has the moat. This also closes the adoption-vector gap a reader will ask about: a neutral, open primitive is how a cross-vendor binding gets adopted without waiting for any one incumbent to bless it.

- **Tradeoff named:** this group collapses candidate groups B (spend above the API) and C (window is short) from the brief. Keeping them separate created an overlap flagged in the task ("both touch competitive dynamics"); collapsing resolves it by framing them as *one posture*: the map tells you where to spend, the climatic patterns tell you when. The adoption-vector sub-point folds in gap F inline rather than spawning a fourth group.

*MECE hint: this group answers "where and when do we spend?" — the strategic-posture case. It does not touch whether the bet is coherent (Group 1) or how existing tactical items re-order (Group 3).*

---

## Supporting argument 3 — The tactical sub-bets are legible under the thesis — each accelerates, defers cleanly, or ships as hygiene

**Claim:** The five items already flagged in `CLAUDE.md §Deferred decisions` and adjacent ADRs do not compete with the thesis — they rank under it. Every one of them is either forced by the team-corpus lift (and sequenced), deferred cleanly (and named), or ships as hygiene (and dispatched). Naming this explicitly prevents the tactical backlog from being re-derived every time the strategy surfaces.

**Evidence:**

- **Python 3.14 flip — defers cleanly.** One-line PR, blocked on `hdbscan 0.8.43+` cp314 wheels (ADR 0015 "Consequences — Hdbscan cp314 watch"; `.erpaval/INDEX.md` lesson `hdbscan cp314 wheel gap`). Not on the critical path for binding, compression, or team ingestion. Ships whenever the wheels land; no strategy change required. Reference: `rumelt-packet.md §Tactical sub-bets`.

- **Snapshot tier / `CacheNode` DAG — probably forced by the team-corpus lift, sequenced under it.** Today's 2 GB / 15K-JSONL corpus doesn't justify the abstraction (`CLAUDE.md §Deferred decisions` — Snapshot tier entry). A team-scope corpus crosses the ~10 GB threshold quickly, and `EXPLAIN ANALYZE` will show JSONL re-scan dominating wall clock once multi-user ingestion lands. This becomes the first tactical item inside the team-ingestion action, not a separate bet. Reference: `rumelt-packet.md §Tactical sub-bets` + `CLAUDE.md §Deferred decisions`.

- **Bedrock `CreateModelInvocationJob` batch embeddings + vector quantization — probably forced at team scale; deferred until then.** Async-at-concurrency-8 saturates global CRIS today without throttling; VSS doesn't natively support FLOAT16/INT8 (`CLAUDE.md §Deferred decisions` — Batch/quantization entry). Reopens when embeddings parquet crosses ~10 GB. Same ordering pattern as the snapshot tier — inside the team-scope action, not ahead of it.

- **`CLAUDE_SQL_CONCURRENCY` env alias cleanup — ships as hygiene.** Already deprecated with a `DeprecationWarning`, scheduled for removal in the next release (`CLAUDE.md §Environment variables` — DEPRECATED line). Cut in a routine refactor commit, not a strategy item.

- **Scorecard `pinning-dependencies` posture — hygiene with a specific unlock condition.** Tag pins (`@v4/@v5/…`) over SHA pins accepted as v1 tradeoff; revisit once Dependabot is tuned for Actions (ADR 0016 §Decisions worth calling out — Action pinning + Follow-up). **Accelerator**: if the bet lands and others adopt `claude-sql` as a dependency, tightening to SHA pins becomes table stakes for their supply-chain story. Move it up the queue when that happens; leave it alone until then.

- **Tradeoff named:** this group is the "coherence check" in pyramid shape — it proves every known tactical item either supports the thesis or is explicitly parked. Without this group, a skeptical reader asks "what about 3.14? snapshot tier? batch embeddings?" and the memo looks like a strategy that ignored half the backlog. With it, the tactical items are legible as the thesis's implementation detail, which is exactly what `rumelt-packet.md §Tactical sub-bets` is scaffolded to say.

*MECE hint: this group answers "what happens to the tactical backlog under this thesis?" — the coherence-with-existing-plans case. It does not touch diagnosis/substrate (Group 1) or map/window (Group 2).*

---

## MECE check

**Mutually exclusive — named and resolved.**

- **Candidate overlap A/D (diagnosis vs substrate):** collapsed into Group 1. Both originally answered "why is this bet coherent?" — one diagnostic, one asset-based. Kept separate they read as two half-arguments for the same claim. Merged, they read as one argument with two halves (the problem shape + the asset shape).
- **Candidate overlap B/C (where to spend vs window is short):** collapsed into Group 2. The brief explicitly flagged these "both touch competitive dynamics" and asked them to be distinguished. The distinction held only if B was narrowly "map says invest above the API" and C was narrowly "incumbents can eat this." In the filled-out form they are the same strategic posture read two ways — map says where, climate says when — and merging them sharpens rather than loses the argument.
- **Groups 1 / 2 / 3 after collapse:** answer three distinct questions — *why bet at all* (Group 1), *where/when to spend* (Group 2), *what happens to the existing backlog* (Group 3). No overlap at the level of the question each group answers. Pass.

**Collectively exhaustive — named and closed.**

- **Gap named in the brief — adoption vector: how does a cross-vendor binding actually get adopted?** Closed inside Group 2 as the "propose the binding primitive openly" sub-point (standard commit trailer / git-notes schema per `wardley-packet.md §Gameplay Moves` candidate 1). Promoting this to Group F would create a fourth group for one sub-point and violate MECE at the top level; folding it into Group 2 as the explicit adoption-vector answer preserves the 3-group structure and still addresses the reader's question.
- **Gap a skeptical technical reader would raise — "you skipped governance/redaction":** explicitly deferred in `framing.md` §Open questions — "Out of scope for the 90-day bet on whether to bet — but in scope for coherent actions if Rumelt commits." This is the right deferral, not a gap. The Rumelt coherent-actions section is where redaction sequencing gets written; the Minto pyramid's job is to name the deferral, not to pretend governance is a strategy group.
- **Gap a skeptical technical reader would raise — "you skipped artifact locality":** same shape — `framing.md` §Open questions names this and the rumelt-architect owns it in §Coherent Actions. Pyramid flags it, doesn't own it.

**Pass.** Three MECE groups; two named overlaps collapsed with explicit rationale; the adoption-vector gap closed inside Group 2; the governance and artifact-locality gaps explicitly passed to the rumelt-packet's Coherent Actions section with their framing citations.

---

## Evidence ↔ Packet Cross-Reference

| Supporting argument | Primary packet / source | Specific section |
| --- | --- | --- |
| **1 — Diagnosis + substrate** | `framing.md` (grounded today) + `rumelt-packet.md` (scaffold) + `CLAUDE.md` (shipping truth) | `framing.md §Crux read` ("binding over the three alternates"); `rumelt-packet.md §Diagnosis` — root-cause slot + §Symptoms; `CLAUDE.md §What this project is` items 1–4; `CLAUDE.md §Structured output (Sonnet classification)`; `CLAUDE.md §Friction classifier (friction_worker.py)`; `CLAUDE.md §Resilience patterns to preserve`; `CLAUDE.md §Analytics pipeline determinism` |
| **2 — Map says invest above the API, window is short** | `wardley-packet.md` (scaffold) + `framing.md §Known facts` #8 | `wardley-packet.md §Value Chain`; §Evolution Axis (stage + visibility table); §Climatic Patterns items 1 + 4 (runtime commoditization + Red Queen); §Build-vs-Buy Read (Consume vs Build calls); §Gameplay Moves candidate 1 (open the binding primitive) + candidate 3 (consume below the API); `framing.md §Known facts` #8 with 2026-05-08 URLs (GitHub Copilot Review; Cursor third-party extensions); `framing.md §Stakes` (ceded-category downside) |
| **3 — Tactical sub-bets legible under the thesis** | `rumelt-packet.md §Tactical sub-bets` (scaffold) + `CLAUDE.md §Deferred decisions` + ADRs | `rumelt-packet.md §Tactical sub-bets`; `CLAUDE.md §Deferred decisions` (Snapshot tier + Batch embeddings entries); `CLAUDE.md §Environment variables` (`CLAUDE_SQL_CONCURRENCY` deprecation); ADR 0015 §Consequences — Hdbscan cp314 watch; ADR 0016 §Decisions worth calling out — Action pinning + §Follow-up; `.erpaval/INDEX.md` lesson `hdbscan cp314 wheel gap` |

---

## Attribution Note

The Minto pyramid contributed the three-group MECE argument structure, the SCQA opener, and the explicit closure of the adoption-vector gap that the framing's Open Questions flagged. Specifically: it collapsed the brief's five-candidate list (A diagnosis, B map, C window, D substrate, E tactical) into three groups by merging A+D (both answer *why coherent?*) and B+C (both answer *where and when to spend?*), leaving E as the tactical-backlog coherence check; and it folded the would-be sixth adoption-vector group into Group 2 as the "propose the binding primitive openly" sub-point rather than spawning a stub group. The synthesizer can use Answer + SCQA for the memo's Executive Summary, Groups 1–3 with their claims as the Guiding Policy body paragraphs, and the cross-reference table as the footnote trail. The MECE-check section is the answer to "did we drop anything?" — two governance/locality gaps are explicitly passed to the rumelt-packet's Coherent Actions section with their framing citations, so the synthesizer should not re-derive them here.

---

## Citations

- [8] Minto, B. *The Pyramid Principle: Logic in Writing and Thinking.* Pearson, 1973 (current edition 2021).
- [9] Expert Program Management, "Barbara Minto Pyramid Principle." (2022). https://expertprogrammanagement.com/2022/11/barbara-minto-pyramid-principle/
- GitHub Copilot Code Review documentation (2026-05-08). https://docs.github.com/en/copilot/using-github-copilot/code-review/using-copilot-code-review
- Cursor Forum — session-history extension thread (2026-05-08). https://forum.cursor.com/t/built-a-cursor-extension-to-save-and-share-chat-and-composer-history/35314
- vibe-replay (2026-05-08). https://vibe-replay.com/
- `framing.md` §Crux read, §Stakes, §Known facts, §Open questions (inline throughout).
- `rumelt-packet.md` §Diagnosis, §Tactical sub-bets (scaffold; cited by heading).
- `wardley-packet.md` §Value Chain, §Evolution Axis, §Climatic Patterns, §Build-vs-Buy Read, §Gameplay Moves (scaffold; cited by heading).
- `CLAUDE.md` §What this project is, §Bedrock, §Structured output (Sonnet classification), §Friction classifier, §Resilience patterns to preserve, §Analytics pipeline determinism, §Environment variables, §Deferred decisions.
- ADR 0015 — Python floor and hdbscan cp314 watch.
- ADR 0016 — GitHub Actions CI hardening and action-pinning posture.

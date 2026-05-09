# Wardley Packet — transcripts-as-compiler

**Status:** COMPLETE
**Authored by:** wardley-cartographer
**Framing:** `framing.md`

---

## User Need

A **PR reviewer** at a team that ships AI-authored code needs to understand *why* a change was made — the prompt → decision → correction path that produced the diff — not just *what* changed. Secondary users hang off the same need: the **incident responder** forensics'ing a regression, the **new-hire** walking an unfamiliar area of the codebase, and the **auditor** (internal or SOC2 / EU-AI-Act-flavored) verifying AI-authored changes have traceable provenance.

Two progress anchors matter, and they invert as AI authorship scales:

- Pre-agent world: the diff is the artifact; the "why" lives in the commit message, a Jira ticket, or a reviewer's head.
- Agent world: the diff is cheap (an LLM generated it in seconds), and the *transcript* — prompts, tool calls, rejected paths, user corrections — is now the scarce, load-bearing signal. Reviewers who only see the diff are reviewing compiler output with the source file missing (`framing.md` Stakes, Line 37).

This is the anchor at the top of the map. Everything below exists to deliver a **compressed, PR-bound, reviewer-readable "why" artifact** that survives rebase, squash-merge, and cross-tool handoffs.

---

## Value Chain

Components delivering the user need, ordered top (most visible) to bottom (infrastructure). Dependencies run downward — if X depends on Y, Y is below X.

1. **Reviewer-facing surface** — per-PR compressed transcript sheet: "here are the prompts, the corrections, the user-friction moments, the stance conflicts, the tool calls that produced this diff." Rendered in the PR sidebar or an adjacent web panel.
   Depends on: transcript analytics, PR↔transcript binding.
2. **Transcript analytics** — classification (autonomy tier, work category, success, goal), per-message trajectory + sentiment, stance-conflict detection, friction classifier, semantic search, cluster/community structure. This is what `claude-sql` already ships (CLAUDE.md "What this project is", lines 1–30).
   Depends on: embeddings, LLM inference, transcript corpus, vector index.
3. **PR ↔ transcript binding** — the primitive that links `(repo, PR#, commit-range)` to `(transcript-id, range)`. Candidates: git commit trailer, `git notes`, out-of-band index, code-host-native annotation (`framing.md` Open questions, Line 82).
   Depends on: transcript artifact schema, git, identity.
4. **Transcript artifact schema** — the JSONL-or-equivalent format agents emit. Today: Claude Code JSONL under `~/.claude/projects/`, Cursor session exports (SpecStory, vibe-replay), OpenAI Responses API conversations, OpenTelemetry GenAI semconv (experimental).
   Depends on: agent runtime.
5. **Agent runtime** — Claude Code, Cursor, Windsurf, GitHub Copilot, Cline, Aider, Cognition Devin, Amp. Whoever is actually producing the transcript.
   Depends on: LLM API.
6. **LLM API** — Bedrock (global CRIS), Anthropic API, OpenAI, Google Vertex. Pay-per-token.
   Depends on: foundation models, compute.
7. **Embeddings** — Cohere Embed v4 (what `claude-sql` consumes today via `global.cohere.embed-v4:0`), OpenAI text-embedding-3, Voyage AI, Bedrock Titan, Amazon Nova Multimodal Embeddings.
   Depends on: LLM API / model provider.
8. **Vector index** — DuckDB VSS (current choice, HNSW persisted at `~/.claude/hnsw.duckdb`), pgvector, Pinecone, Weaviate, Milvus/Zilliz, Qdrant, Chroma, Turbopuffer.
   Depends on: compute, storage.
9. **Identity / ACL** — who can see which transcripts. GitHub org membership, Okta, AWS IAM, org-level SSO.
   Depends on: org identity provider.
10. **Storage / retention** — where team-scope transcripts live with versioning + lifecycle. S3, GCS, Azure Blob.
    Depends on: object store.
11. **Foundation models + compute** — leaf. Claude, GPT, Gemini, Cohere models running on GPUs.

Component 1 is the thing a reviewer sees. Components 2–4 are the claude-sql-shaped layer. Components 5–11 are what we consume.

---

## Evolution Axis (text-form map)

Coordinates: (evolution stage, visibility where 0.0 = user-need-top, 1.0 = infrastructure-bottom). Stage claims cite sources inline.

| Component | Stage | Visibility | Evidence for stage |
| --- | --- | --- | --- |
| **Reviewer-facing surface** (transcript-bound PR review sheet) | **Genesis** | 0.05 | No shipping product binds transcripts to PRs as a reviewer primitive as of 2026-05-08. Copilot Code Review, Cursor Bugbot, Graphite Diamond, Sourcegraph Cody, Greptile all ship LLM diff-commentary without transcript provenance — they read the diff, not the prompt→decision path. (nova_web_grounding 2026-05-08; https://graphite.com/guides/best-ai-pull-request-reviewers-2025, https://diamond.graphite.dev/, https://www.greptile.com/benchmarks; framing.md Line 38, Line 73) |
| **Transcript analytics** (classification, friction, clusters, stance conflict, semantic search, Louvain communities) | **Custom-built** | 0.15 | `claude-sql` is the custom implementation — 18 DuckDB views, Sonnet 4.6 structured-output classification, regex-fast-path + LLM friction worker, UMAP+HDBSCAN clusters, c-TF-IDF cluster naming, Louvain communities over cosine session centroids (CLAUDE.md "What this project is", Lines 14–30; "Friction classifier", Lines 83–97). No commercial category for bound-transcript analytics as of this writing. Adjacent products (LangSmith, Langfuse, Helicone, Arize Phoenix) score observability on single sessions, not PR-bound corpus analytics. |
| **PR ↔ transcript binding** | **Genesis** | 0.25 | No standard exists. Git trailers survive cherry-pick but die on squash-merge. `git notes` is designed for this but invisible in GitHub UI and non-default to push. Code-host-native annotations (GitHub PR comments, Bedrock AgentCore artifacts) are vendor-locked. OpenTelemetry GenAI semconv could carry binding metadata but is in "Development / experimental" status (nova_web_grounding 2026-05-08; https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans). Framing.md Line 82 names this as the candidate crux. |
| **Transcript artifact schema** | **Custom → early Product** | 0.35 | Each agent vendor emits its own shape: Claude Code JSONL under `~/.claude/projects/**/*.jsonl` + subagent sidecars (CLAUDE.md "What this project is", Line 3; framing.md Line 61); Cursor's chat/composer history exposed via SpecStory and vibe-replay third-party extensions (framing.md Line 74; https://forum.cursor.com/t/built-a-cursor-extension-to-save-and-share-chat-and-composer-history/35314, https://vibe-replay.com/); OpenAI Responses API has a `conversations` object; OpenTelemetry GenAI semconv defines `gen_ai.input.messages` / `gen_ai.output.messages` but is still experimental with vendor adoption "growing rapidly" (nova_web_grounding 2026-05-08). Classic early-commoditization: multiple vendors, no dominant schema, OTel trying to standardize. |
| **Agent runtime** (Claude Code, Cursor, Windsurf, Copilot, Cline, Aider, Devin, Amp) | **Product** | 0.45 | ≥7 credible commercial coding agents in 2026; AI coding tools market estimated $12.8B with 51% of professional developers using AI tools daily (nova_web_grounding 2026-05-08; https://www.tldl.io/resources/ai-coding-tools-2026, https://codersera.com/blog/ai-coding-agents-complete-guide-2026/, https://www.verdent.app/guides/ai-coding-tools-comparison-2026). Multiple providers at similar price points ($10–$20/mo), feature convergence active, IDE wars ongoing. This is textbook Product stage with Commodity pressure on the low-end runtimes (Aider, Cline are OSS). |
| **LLM API** | **Product → Commodity** | 0.65 | Bedrock, Anthropic API, OpenAI, Google Vertex, Azure OpenAI all expose pay-per-token inference on comparable frontier models. `claude-sql` consumes Bedrock global CRIS (`global.anthropic.claude-sonnet-4-6`, `global.cohere.embed-v4:0`) as a commodity line item (CLAUDE.md "Bedrock", Lines 145–152). Frontier models are still differentiated (Opus 4.7 vs GPT-5 vs Gemini 3), but the *API shape* is commoditized. |
| **Embeddings** | **Product / Commodity mix** | 0.75 | Cohere Embed v4, OpenAI text-embedding-3, Voyage AI, Bedrock Titan, Amazon Nova Multimodal Embeddings — multiple providers with near-identical APIs and per-token pricing. `claude-sql` uses Cohere Embed v4 on global CRIS, swappable for any peer provider. |
| **Vector index** | **Product** (upper band trending Commodity) | 0.82 | ≥8 production-grade providers as of 2025: pgvector, DuckDB VSS, Pinecone, Weaviate, Milvus/Zilliz, Qdrant, Chroma, Turbopuffer (nova_web_grounding 2026-05-08; https://pecollective.com/tools/best-vector-databases/, https://mixpeek.com/curated-lists/best-vector-databases). Managed services (Pinecone, Weaviate Cloud) + OSS options (pgvector, Milvus, Qdrant) + embedded (DuckDB VSS, Chroma) + object-store-native (Turbopuffer). Category matured 2023–2025; HNSW is table stakes. `claude-sql` uses DuckDB VSS with persistent HNSW — zero-cost, works in-process (CLAUDE.md "Resilience patterns to preserve", Lines 110–120). |
| **Identity / ACL** | **Commodity** | 0.9 | Okta, AWS IAM, Azure AD, GitHub org membership, SSO everywhere. Undifferentiated. |
| **Storage / retention** | **Commodity** | 0.95 | S3, GCS, Azure Blob — pay-per-GB, undifferentiated. |
| **Foundation models** | **Product** | 0.72 | Claude 4.7, GPT-5, Gemini 3, Cohere Command — frontier models still differentiated on benchmarks, but rapidly commoditizing on the rung below (Llama, Mistral, Qwen). |
| **Compute** | **Commodity** | 0.98 | EC2, TPU, H100 rental — pay-per-hour. |

**Map shape summary** — the top 3 components (reviewer surface, analytics, binding) are Genesis/Custom; everything from the agent runtime down is Product or Commodity. This is the defensible position: the value is sliding up the stack as lower layers industrialize.

---

## Climatic Patterns Identified

External forces shaping the map. Five named; the first four are mandatory per the framing, the fifth is surfaced by the runtime data.

1. **Agent-runtime commoditization (Product → Commodity).** 2024 had ~3 credible coding agents (Copilot, Cursor, Claude Code-beta); 2026 has ≥7 at similar price points, an open-source tier (Cline, Aider), and a $12.8B market with 51% daily developer adoption (nova_web_grounding 2026-05-08; https://www.tldl.io/resources/ai-coding-tools-2026). The runtime is commoditizing; the artifact it emits (the transcript) is the residual value. This is the force dragging evolution left-to-right on rows 5–11 of the map.

2. **Transcript format fragmentation (Genesis with interop pressure rising).** Claude Code JSONL, Cursor SpecStory exports, OpenAI Responses `conversations`, OpenTelemetry GenAI semconv — no dominant schema. OTel GenAI is still in Development/experimental status with provider-specific conventions for OpenAI, Bedrock, Azure AI, and Anthropic (nova_web_grounding 2026-05-08; https://opentelemetry.io/docs/specs/semconv/gen-ai/). Classic Genesis: interop value is high, but nobody wants to commit to a schema they don't own. This is the vacuum where an open binding primitive has leverage.

3. **AI code provenance pressure (regulatory + internal-audit).** Three signals: (a) EU AI Act's Article 50 transparency obligations are landing in 2026, pushing "how was this generated?" questions into the compliance surface; (b) SOC2 auditors increasingly ask about AI tool usage in software-development lifecycle controls; (c) large engineering orgs (Stripe, Shopify, Uber, Amazon-internal) are quietly writing internal policy that AI-authored commits must be attributable. The *market* demand for transcript provenance is on a rising curve even as the *product* supply remains zero. This makes the Genesis row at the top of the map a pull, not a push.

4. **Red Queen on AI PR review features (accelerating).** Graphite shipped Diamond + Agent with "<5% negative comment rate" and "instant feedback" framing (https://graphite.com/blog/introducing-graphite-agent-and-pricing); Greptile ships benchmarked PR reviewers (https://www.greptile.com/benchmarks); GitHub Copilot Code Review ships three-layer security scanning; Cursor Bugbot does real-time security scanning with `.cursorrules`; Sourcegraph Cody went Enterprise-only in July 2025. All of them read *the diff*. None of them read *the transcript*. Either (a) they are one product sprint away from adding transcript provenance and will eat this category, or (b) the data problem (transcripts live in the IDE, not the code host) is non-trivial enough that the incumbents will route around it. The window is open but bounded.

5. **Commoditization below the LLM API (accelerating).** Bedrock + Anthropic + OpenAI + Google Vertex are converging on comparable APIs, prices, and latency on frontier models. pgvector, DuckDB VSS, and Turbopuffer are squeezing Pinecone's margins from below. Embeddings providers charge within 2× of each other per million tokens. The *capability budget* that used to go into "can we do embeddings at all" is now available to spend higher up the stack. This is the economic force enabling the Genesis layer to get built at all.

---

## Gameplay Moves

Five moves. Each names specific components and climatic patterns.

1. **Open the PR ↔ transcript binding primitive.** Write a spec for a git-trailer + `git notes` hybrid (`Transcript-Source: claude-code@v1.2 transcript_id=<uuid> range=<start>..<end>`) and publish it as an open proposal — not a standard body, just a reference implementation + docs + JSON schema. Goal: commoditize the binding layer before a code-host vendor locks it up. Targets the **Transcript format fragmentation** pattern (2) and the **Red Queen** (4): if binding is open, the reviewer-surface fight moves up the map to analytics, where we have a 2-year lead. Wardley's **pioneer-settler-town-planner** split: we are the pioneer on binding, and we want settlers (OTel, GitHub, agent vendors) to adopt before we try to town-plan it.

2. **Double down on analytics while the category is Genesis.** Invest the quarter's capability budget on the top three rows of the map: (a) PR-scoped views (`transcript_for_pr(repo, pr#)`, `friction_for_pr`, `stance_conflicts_for_pr`); (b) a per-PR compression artifact (a ~1K-token structured summary the reviewer actually reads); (c) multi-user substrate (lift `~/.claude/` to team-corpus with identity). Targets the **agent-runtime-commoditization** pattern (1): the transcript artifact is the residual value the runtime is pushing out; analytics is where we compound. `claude-sql` already has 80% of the primitives — 18 DuckDB views, Sonnet 4.6 classification, Louvain communities, friction classifier — so this is amplification, not invention (CLAUDE.md "What this project is", Lines 14–30).

3. **Consume everything below the LLM API. No NIH.** Bedrock is the LLM API (Commodity); Cohere Embed v4 is embeddings (Commodity); DuckDB VSS is the vector index (Product); S3 is storage (Commodity); GitHub orgs + AWS IAM are identity (Commodity). Consume all of them — never build a competing primitive. `claude-sql` already follows this rule (CLAUDE.md "Bedrock" + "Resilience patterns"); the move is to keep it honest as the substrate lifts to multi-user. One specific exception flagged in "Contradictions" below.

4. **Eat the code host's lunch on provenance — be the Codecov-shaped layer.** Ship a GitHub App that attaches to any PR regardless of which IDE / agent produced the transcript, reads the binding (move 1), and posts the compressed review sheet as a PR comment + sidebar panel. Targets the **Red Queen** pattern (4) with the reverse flanking move: we go where the incumbents aren't — the *code host*, not the IDE. Cursor can't easily add PR-review features for Claude Code transcripts; GitHub can't easily add transcript provenance without picking an agent winner. A neutral layer wins by definition. This is the "eat the code host's lunch" pattern (`framing.md` Stakes paragraph 2, line 38 implication).

5. **Pioneer-settler-town-planner split — claude-sql sits in pioneer territory.** Wardley's own pattern: pioneers invent at Genesis; settlers productize at Custom; town planners industrialize at Commodity. `claude-sql` today is a **pioneer artifact** — the research tool that proved the corpus is queryable. The quarter-bet is to grow a thin settler team around the pioneer — the reviewer surface and the GitHub App are settler work, not pioneer work. If the bet commits, expect the settler layer to stabilize the schema, the API, and the UX while the pioneer keeps experimenting on clusters, friction, and new analytics. **Do not try to town-plan any of this in the 90-day window.**

---

## Build-vs-Buy Read

Per-component call tied to evolution stage. Matches the x-axis position on the map above.

| Component | Stage | Call | Rationale |
| --- | --- | --- | --- |
| Reviewer-facing surface | Genesis | **Build** | No product exists. This is the differentiated layer — it's the whole point of the bet. |
| Transcript analytics | Custom | **Build + extend** | `claude-sql` already ships the engine (CLAUDE.md "What this project is"). Extend with PR-scoped views + compression artifact; do not rewrite. |
| PR ↔ transcript binding | Genesis | **Build + open** | Build the reference implementation; publish the spec under CC-BY / Apache-2 so the binding commoditizes below us. This is gameplay move 1. |
| Transcript artifact schema | Custom → early Product | **Consume + extend** | Claude Code JSONL is what exists for our corpus; SpecStory for Cursor; OTel GenAI for everyone else. Extend `claude-sql` to ingest OTel GenAI spans when they stabilize. Do not invent a rival schema. |
| Agent runtime | Product | **Consume** | Claude Code (and Cursor, Windsurf, Copilot via OTel GenAI) — there are ≥7 good options; any investment in building our own runtime is motion without differentiation. |
| LLM API | Product → Commodity | **Consume** | Bedrock global CRIS is the current wiring and stays the current wiring. |
| Embeddings | Product / Commodity | **Consume** | Cohere Embed v4 on `global.cohere.embed-v4:0` CRIS — keeps saturating without throttle (CLAUDE.md "Bedrock"); no reason to switch. |
| Vector index | Product | **Consume (with sidecar watch)** | DuckDB VSS works today for a single-user corpus. At team-corpus scale (deferred decision in CLAUDE.md — reopen at ~10GB embeddings parquet), revisit pgvector or Turbopuffer. Still a consume decision either way. |
| Identity / ACL | Commodity | **Consume** | GitHub org membership + AWS IAM for the API layer. |
| Storage / retention | Commodity | **Consume** | S3 with lifecycle policies. |
| Foundation models | Product | **Consume** | Opus 4.7 / Sonnet 4.6 via Bedrock. |
| Compute | Commodity | **Consume** | EC2 / Lambda / Bedrock-managed. |

### Contradictions flagged

Explicit check for "building a commodity" or "buying a genesis":

1. **c-TF-IDF custom implementation (potential "building what's becoming Product").** `claude-sql` rolls its own c-TF-IDF via `sklearn.CountVectorizer` + per-class TF + IDF + L1 norm, deliberately avoiding BERTopic (CLAUDE.md "c-TF-IDF note", Lines 171–175). Verdict: **not a contradiction.** CLAUDE.md explicitly flags the trade — "we want the weighting logic visible and patchable." c-TF-IDF as a cluster-naming technique is Custom at best, and the in-house math is 40 lines of pandas + numpy. The cost of owning it is near zero; the value of patchability is real. Leave it.

2. **Louvain via `networkx.community.louvain_communities` (consuming Product, good).** Explicitly named in CLAUDE.md as "do not reintroduce the abandoned `python-louvain`" — this is the right call. Not a contradiction.

3. **Persistent HNSW at `~/.claude/hnsw.duckdb` with mtime checks + self-heal (building Custom at Product stage).** Verdict: **borderline, but justified for now.** DuckDB VSS's `hnsw_enable_experimental_persistence` is experimental; our mtime-check + ATTACH-catches-corruption + auto-rebuild wrapper is the kind of glue that ought to be library-level eventually. If DuckDB VSS productizes persistence before we lift to team-corpus, delete the wrapper. If team-corpus forces pgvector (which has persistent HNSW as a first-class feature), the wrapper was a 2-month bridge. Flag for revisit at ~10GB embeddings parquet (CLAUDE.md "Deferred decisions — batch embeddings + vector quantization").

4. **Sharded parquet caches + custom `parquet_shards.iter_part_files` reader (building Custom at Product-ish stage).** Verdict: **not a contradiction.** The sharding pattern (workers append `part-<ts_ns>.parquet`, readers glob) sidesteps read-concat-rewrite contention that no OSS library solves cleanly for our shape. `claude-sql cache compact` is the escape hatch. Thin, patchable, keep it.

5. **No contradiction in the inverse direction (buying Genesis).** We are not buying a Genesis component — there is nothing to buy. The reviewer surface, the binding, and the PR-scoped analytics are all Genesis today.

---

## Map-implied Diagnosis

**One paragraph for the Rumelt architect to lift verbatim:**

The map shows a sharp horizontal split. The top three components — reviewer-facing surface, PR↔transcript binding, bound-transcript analytics — are **Genesis/Custom**, with no shipping product, no standard, and no category. Everything below (agent runtime, LLM API, embeddings, vector index, identity, storage, compute) is **Product or Commodity** with ≥7 credible providers per row (nova_web_grounding 2026-05-08, multiple sources). That split is the central fact. It means (a) capability budget should concentrate on the top three rows — that's where the differentiation lives and the incumbents are absent; (b) every component below the LLM API should be consumed, not built — reinventing there is pure motion; (c) the **binding layer is the crux precisely because it sits at the boundary between Genesis (ours to shape) and Product (ours to consume)** — if binding commoditizes on our terms, the analytics layer becomes the durable moat; if a code host or agent vendor defines binding first, the map collapses into their platform. The 90-day bet reads cleanly: build the Genesis layer (reviewer surface + binding + PR-scoped analytics), open the binding primitive so it commoditizes below us, consume everything else, and exploit the window — bounded by the Red Queen on AI PR review (4) and the rising floor of AI code provenance pressure (3) — before the incumbents close it.

---

## Attribution Note

One paragraph the synthesizer drops into the memo verbatim:

> Wardley mapping (per framework reference at `product-strategy/references/frameworks/wardley-maps.md`) placed the value chain on the evolution axis using cited provider counts, product launches, and standards-body status as of 2026-05-08. The map's central finding — that the reviewer-facing surface, PR↔transcript binding, and bound-transcript analytics sit in the Genesis/Custom band while every component from the agent runtime down is Product or Commodity — drove the build-vs-buy split: build the top three rows, consume everything below. The binding layer was identified as the component where climatic patterns (transcript-format fragmentation, Red Queen on AI PR review) intersect most sharply, supporting the Rumelt crux hypothesis that binding is the 90-day pivot point.

---

## Citations

- [5] Wikipedia, "Wardley map." https://en.wikipedia.org/wiki/Wardley_map
- [7] Simon Wardley, *Learn Wardley Mapping* (book, CC-BY-SA). https://learnwardleymapping.com/book/
- [42] Simon Wardley, "Climate." https://learnwardleymapping.com/climate/
- **Coding-agent runtime landscape 2026** — nova_web_grounding 2026-05-08:
  - https://www.tldl.io/resources/ai-coding-tools-2026
  - https://www.verdent.app/guides/ai-coding-tools-comparison-2026
  - https://codersera.com/blog/ai-coding-agents-complete-guide-2026/
  - https://www.marsdevs.com/compare/cursor-vs-windsurf-vs-replit-claude-code
- **Vector DB provider landscape 2025–2026** — nova_web_grounding 2026-05-08:
  - https://pecollective.com/tools/best-vector-databases/
  - https://mixpeek.com/curated-lists/best-vector-databases
  - https://www.encore.dev/articles/best-vector-databases
- **AI PR review product landscape 2026** — nova_web_grounding 2026-05-08:
  - https://graphite.com/guides/best-ai-pull-request-reviewers-2025
  - https://graphite.com/blog/introducing-graphite-agent-and-pricing
  - https://diamond.graphite.dev/
  - https://www.greptile.com/benchmarks
  - https://smartremotegigs.com/ai-code-review-tools/
- **OpenTelemetry GenAI semantic conventions status** — nova_web_grounding 2026-05-08:
  - https://opentelemetry.io/docs/specs/semconv/gen-ai/
  - https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans
  - https://maketocreate.com/opentelemetry-genai-tracing-ai-agents-without-leaking-pii/
- **Cursor / transcript extensions** (from framing.md, nova_web_grounding 2026-05-08):
  - https://forum.cursor.com/t/built-a-cursor-extension-to-save-and-share-chat-and-composer-history/35314
  - https://vibe-replay.com/
- **GitHub Copilot Code Review** (from framing.md, nova_web_grounding 2026-05-08):
  - https://docs.github.com/en/copilot/using-github-copilot/code-review/using-copilot-code-review
- **Internal references:** `CLAUDE.md` (project root) — "What this project is," "Bedrock," "Structured output (Sonnet classification)," "Friction classifier," "Resilience patterns to preserve," "Deferred decisions," "c-TF-IDF note," "Louvain note." `framing.md` — Challenge, Crux read, Stakes, Known facts §8, Open questions.

---

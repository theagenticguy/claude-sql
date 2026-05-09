# Product strategy work log — cpo

**Status:** COMPLETE
**Role:** cpo
**Slug:** transcripts-as-compiler
**Working directory:** `.erpaval/strategy/transcripts-as-compiler/`
**Your output file:** `.erpaval/strategy/transcripts-as-compiler/framing.md`

## 1. Objective

Frame the "transcripts + intelligence = a new compiler" strategic question for Phase 2 framework agents, naming the audience, candidate crux, stakes, and fan-out plan.

## 2. Scope

- **Input**: user's strategic thesis (in task prompt), existing codebase (`/efs/lalsaado/workplace/claude-sql`), CLAUDE.md, ADRs, `.erpaval/INDEX.md`.
- **Output**: `.erpaval/strategy/transcripts-as-compiler/framing.md`
- **Role reference**: `${CLAUDE_PLUGIN_ROOT}/skills/product-strategy/references/roles/cpo.md`

Sections: Challenge, Audience, Crux read, Stakes, Framework fan-out plan, Known facts, Open questions, Voice guidelines.

Out of scope: running any framework, composing the memo, making the build-vs-buy call.

## 3. Inputs

Files to read:
- Existing `.erpaval/INDEX.md` and session artifacts under `.erpaval/sessions/`
- `CLAUDE.md` (project)
- `docs/adr/0015-stack-modernization.md` + `docs/adr/0016-ci-hardening.md`
- Source tree at `/efs/lalsaado/workplace/claude-sql/src/claude_sql/` to ground what already ships

## 4. Success criteria

Every required section has real content, cited inline, with a committed fan-out plan (Rumelt + Wardley + Minto; PR-FAQ skipped per user). Crux is named as a candidate, not hedged.

## 5. Anti-goals

- Don't guess at what claude-sql does — read the code
- Don't let the fan-out drift to "all four frameworks" — user explicitly skipped PR-FAQ
- Don't hedge the crux; name a candidate

---

## Work log

1. **Read ground truth in parallel.** `CLAUDE.md` (project), `src/claude_sql/` listing, ADR 0015 (stack modernization), ADR 0016 (CI hardening), `.erpaval/INDEX.md` (accumulated lessons), `.erpaval/sessions/` (3 merged sessions, all internal to claude-sql), `README.md` (badges + intro + CLI surface). Confirmed current shipped surface: 18 views, 14 macros, Cohere Embed v4 via global CRIS, Sonnet 4.6 structured output, UMAP+HDBSCAN, Louvain, friction classifier, persistent HNSW, sharded parquet caches, cost-guarded dry-runs, agent-friendly CLI.
2. **Confirmed framework routing.** Read `skills/product-strategy/references/frameworks/INDEX.md` decision table. Rumelt + Wardley + Minto is the right subset for an internal strategic bet aimed at a technical decision-maker; PR-FAQ is for customer-facing launches and correctly skipped per user scope.
3. **External grounding, bounded.** Two parallel Nova web searches: (a) GitHub Copilot AI code review transcript provenance, (b) Cursor agent session replay. Net finding: incumbents (GitHub, Cursor, Claude Code) have not shipped first-class transcript-bound-to-PR artifacts; early third-party signal exists (SpecStory, cursor-replay, vibe-replay) but nothing PR-bound. Window is real and not permanent.
4. **Drafted framing in one write.** All eight sections populated with inline citations to file paths, `CLAUDE.md` headings, ADR numbers, and the two URLs from step 3. Named binding as candidate crux with explicit reasoning vs. the three alternates (signal-to-noise / trust-governance / substrate-lift). Committed fan-out: Rumelt + Wardley + Minto RUN, PR-FAQ SKIP with justification. Surfaced the five deferred-item sub-bets (3.14 flip, snapshot tier, batch embeddings, CONCURRENCY alias cleanup, Scorecard pinning-dependencies) in "Known facts" item 7 so they don't get re-derived in Phase 2.
5. **Flipped Status to COMPLETE on both `framing.md` and this log.**

---

## Validation

- [x] Every required section populated
- [x] Citations inline (file paths, CLAUDE.md headings, ADR 0015/0016, two external URLs)
- [x] Crux named as candidate with reasoning, not hedged
- [x] Fan-out committed (Rumelt + Wardley + Minto RUN; PR-FAQ SKIP with reason)
- [x] Status flipped to COMPLETE

---

## Summary

Framing memo for the "transcripts-as-compiler" strategic bet is ready for Phase 2. The challenge is a concrete 90-day decision (lift `claude-sql` from personal tool to transcripts-as-PR-review substrate). The candidate crux is **PR↔transcript binding** — named over signal-to-noise, governance, and substrate-lift with explicit reasoning for each. The fan-out plan commits Rumelt + Wardley + Minto in parallel; PR-FAQ is skipped because this is an internal strategic memo, not a customer launch. Competitive window validated via one round of web grounding: no incumbent has shipped transcript-bound-to-PR as a first-class artifact as of 2026-05-08. Deferred tactical items (3.14 flip, snapshot tier, batch embeddings, CONCURRENCY alias, Scorecard pinning) surfaced as sub-bets, not headline. Rumelt-architect can now sharpen or overrule the binding-crux hypothesis; Wardley can map the value chain; Minto composes the final memo.

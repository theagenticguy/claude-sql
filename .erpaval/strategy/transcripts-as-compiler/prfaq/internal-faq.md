# Internal FAQ — claude-sql transcript-bound PR review

**Status:** COMPLETE — discovery artifact, not for publication.
**Audience:** AGS Tech AI Engineering leadership + peers, peer Principals, partner teams, security / legal reviewers once a design-partner path opens.

Ten adversarial questions, answered honestly, including explicit deferrals.

---

## 1. How is this different from GitHub Copilot Code Review?

Copilot Review emits LLM-generated line comments over the diff with no transcript provenance, no binding to the session that produced the diff, and no audit trail; the reviews are advisory (https://docs.github.com/en/copilot/using-github-copilot/code-review/using-copilot-code-review, 2026-05-08; memo §Evidence row 1). `claude-sql` does not compete on comment quality — it competes on **provenance**: the review sheet shows the prompt, the explored-and-rejected paths, the corrections, and the tools, all resolvable from a commit trailer. Copilot Review structurally cannot emit that surface because it does not have the transcript. The memo's §Coherent Actions #4 policy-coherence check makes this the explicit positioning.

## 2. Why is binding the crux, and not compression or analytics?

Compression and analytics are already shipping in `claude-sql` — 18 DuckDB views, Sonnet 4.6 structured output, Cohere Embed v4, UMAP+HDBSCAN, Louvain, friction classifier (CLAUDE.md §What this project is). The missing 10% is the pointer from a merged PR back to the transcript that produced it. Without that pointer, every downstream capability (review sheet, retro, onboarding, governance) collapses to "logs we happen to have." The primitives to build the pointer — commit trailers and `git notes` — have shipped with git since 1.6 and are adopted by Gerrit, Linux kernel, `Signed-off-by:`, `Co-authored-by:` (memo §The Crux). The pivot is adopting a convention, not inventing one. That is why it is the crux: pivotal + surmountable.

## 3. What about secrets, PII, and third-party IP in transcripts?

Honest answer: the 90-day scope ships a **regex-based secrets scrub** inside the `claude-sql bind` step (memo §Risks #2; §Coherent Actions — Explicit deletions). A policy-driven redaction engine, per-org taxonomies, and IP / third-party content handling are **explicitly deferred** — they need a design-partner conversation to define the taxonomy, and building them inside the 90 days spends the capability budget on a non-crux problem. The memo flags this as "safe-by-default on secrets, not audit-grade on IP." Any design-partner conversation reopens it — and any compliance-regulated workload (HIPAA, PCI) has to wait on it. Surfacing the deferral in the memo and here is the mitigation, not the fix.

## 4. Why a GitHub App, not a native GitHub feature?

A GitHub App is the **first** integration, chosen because github.com is where the reviewers sit and the reach is highest per week of build (memo §Coherent Actions — Policy coherence check). The binding spec is host-neutral — it is a commit trailer + `git notes` convention that any code host or review tool can consume. If GitLab, Bitbucket, or an internal code host matters to a design partner later, the spec already covers it and only a second integration is incremental. We do not ship a native GitHub feature because we do not own that surface and distribution is GitHub's game; we ship a provenance-carrying App on top of it, where GitHub structurally cannot compete (their reviewer has no transcript).

## 5. Will this work with Cursor, Amp, Windsurf, Copilot Workspace?

Yes — the binding spec is **agent-agnostic**. The `Claude-Agent-Runtime:` trailer is an extension point (`claude-code/0.x`, `cursor/…`, `amp/…`, etc.). The 90-day bet ships a reference implementation against Claude Code transcripts because that is where our corpus is and where we have the analytics substrate already (memo §Coherent Actions #1). The convergent cross-vendor path is OpenTelemetry GenAI semconv once it stabilizes (memo §Evidence row 11; §Build/Buy — transcript artifact schema). Any runtime that emits OTel GenAI spans will be consumable through the same binding.

## 6. What if GitHub or Anthropic ships native transcript binding first?

Two-part mitigation (memo §Risks #1). First, the spec is published under Apache-2 / CC-BY with a reference implementation (Wardley gameplay move 1 — "commoditize the binding layer below us"); if an incumbent ships a **compatible** primitive, the analytics above it ports rather than stranding. Second, action 4 is the adoption-vector bet — by week 12 we have a working binding on a real repo's PR workflow, which is what survives a 6-month native-binding ship: a spec alone gets rewritten by whoever has distribution; a spec + running integration competes on transcript provenance that an LLM-over-diff feature structurally cannot emit. Residual risk: if an incumbent ships a binding **and bundles the analytics** in one feature, the moat compresses. The 12-week cadence is calibrated against this window.

## 7. Who is the DRI and what is the team size for the 90-day bet?

**DRI: Laith Al-Saadoon (solo).** Scope for the 90 days is trimmed to what a solo Principal can ship on top of the already-shipping `claude-sql` substrate: binding spec v0 + reference implementation (week 2), review-sheet compression on Sonnet 4.6 (week 6), team-corpus ingestion v0 — `Settings.team_corpus_root` + parameterized `read_json` glob + 2-user synthetic fixture (week 10), GitHub App dogfooded on the `claude-sql` repo + one other Laith-owned repo (week 12). The action 3 scope cut in memo revision round 1 explicitly defers HNSW-per-team store and `cache compact` / `cache migrate` team variants to post-90-day. All four actions compose on single-user if the team-corpus lift slips (memo §Risks #4).

## 8. What is the resourcing ask beyond Laith?

**Allocation ask, no dollar estimate.** Laith's time is protected at ~75% of weekly capacity against the four coherent actions for the 90-day window; the remaining ~25% covers hygiene and dependencies (action 5 in the tactical sub-bets — Scorecard SHA pins, `CLAUDE_SQL_CONCURRENCY` deprecation removal, ADR 0015 3.14 watch). **Beyond Laith:** at week 6 (review-sheet schema stable) add one design-partner conversation with a platform team that is already ≥20% AI-authored to validate the review-sheet schema against a non-Laith corpus; at week 10 add a security review on the regex secrets scrub before any external repo installs the App. Neither is a headcount ask — both are time-boxed asks against existing partner teams. If the bet survives week 12 on its acceptance test, the post-90-day allocation reopens as a portfolio question (see Q10).

## 9. How do we measure success at week 12?

Three gates, in order of priority.

1. **Reviewers read the sheet.** Instrument PR-comment reactions and read-through time on the `claude-sql/review-sheet` comment on both dogfood repos. If reviewers ignore the comment, the diagnosis is wrong and the strategy resets (memo §Risks #5; §Coherent Actions — Coherence check).
2. **The spec attracts one external consumer.** At least one external repo (adjacent team, open-source project, or design partner) installs the pre-commit hook or reads the RFC enough to file a spec issue.
3. **The bound corpus enables a review action that the unbound corpus did not.** At least one PR review, incident retro, or onboarding session on the dogfood repos cites a specific transcript element (rejected path, correction, refused tool) as load-bearing in the decision. This is the narrow operational proof that the transcript-as-source framing is cashing out — not vibes, a specific artifact-in-comment citation.

## 10. When does this become a hiring bet or a portfolio move?

**Triggers, named.** (a) Two or more AWS internal teams adopt the GitHub App and the review-sheet comment on their repos without Laith pushing it — adoption from below is the pull signal. (b) A design-partner conversation with a regulated-workload team (financial services, healthcare) surfaces a concrete redaction-taxonomy requirement, forcing the governance deferral from Q3 off the deferred list. (c) The OTel GenAI semconv stabilizes enough that "bind trailer ↔ OTel span" becomes a one-week bridge and opens Cursor / Amp / Copilot Workspace as consumption paths. Any one of those flips the question from "solo bet" to "does this deserve a team and a PR/FAQ that goes through AEM / Legal for real." The three are the named reopen conditions — without one of them, the bet stays solo Principal scope.

---

**Status:** COMPLETE

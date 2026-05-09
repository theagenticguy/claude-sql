# Product strategy work log — strategy-synthesizer

**Status:** COMPLETE
**Role:** strategy-synthesizer
**Slug:** transcripts-as-compiler
**Working directory:** `.erpaval/strategy/transcripts-as-compiler/`
**Your output file:** `.erpaval/strategy/transcripts-as-compiler/strategy-memo.md`

## 1. Objective

Compose the final top-down strategy memo from framing.md + rumelt-packet.md + wardley-packet.md + minto-outline.md.

## 2. Scope

- **Input**: all 4 files above (all COMPLETE)
- **Output**: `.erpaval/strategy/transcripts-as-compiler/strategy-memo.md`
- **Role reference**: `${CLAUDE_PLUGIN_ROOT}/skills/product-strategy/references/roles/strategy-synthesizer.md`

Sections: Executive Summary (Minto), Diagnosis + Crux, Guiding Policy, Coherent Actions, Build/Buy/Partner, Tactical sub-bets, Risks, Evidence, Convergence Notes.

## 3. Inputs

- `.erpaval/strategy/transcripts-as-compiler/framing.md` (COMPLETE)
- `.erpaval/strategy/transcripts-as-compiler/rumelt-packet.md` (COMPLETE, 399 lines)
- `.erpaval/strategy/transcripts-as-compiler/wardley-packet.md` (COMPLETE)
- `.erpaval/strategy/transcripts-as-compiler/minto-outline.md` (COMPLETE, 3-group structure)

## 4. Success criteria

- Every non-obvious claim has inline packet attribution
- Executive Summary reads top-down (Minto-shaped)
- Convergence Notes name at least one convergence AND one divergence
- No framework body reproduced — use packet outputs
- Evidence bibliography deduplicated, newest-first

## 5. Anti-goals

- Don't re-run a framework
- Don't let "everything agreed" stand as convergence — find a divergence to name
- Don't reproduce the Wardley evolution table in the memo — use the map-read and build-vs-buy callouts
- Don't drop the tactical sub-bets — they need to appear but below the strategic thesis

---

## Work log

- 2026-05-08 — Read framing + all 3 packets + skeleton. All three packets COMPLETE. Rumelt endorsed binding crux with commit-trailer + git-notes sharpening; Wardley placed reviewer/binding/analytics at Genesis with everything below LLM API at Product/Commodity; Minto collapsed 5 candidate groups → 3 (A+D → Group 1 diagnosis+substrate, B+C → Group 2 map+window, E → Group 3 tactical coherence), folded adoption-vector gap into Group 2.
- 2026-05-08 — Decision: Executive Summary will open with a one-sentence recommendation, then one SCQA paragraph (lifted compactly from minto §SCQA), then the 3 supporting arguments as a numbered list. Diagnosis section will lift from rumelt §Diagnosis and sharpen with wardley §Map-implied Diagnosis; Crux gets its own bolded sentence with the surmountability argument. Guiding Policy will use rumelt's "Compete on X by concentrating Y, because Z" with Wardley's Genesis/Commodity split as reinforcement. Coherent Actions = rumelt's 4 with Wardley's Gameplay Move 4 (GitHub App) folded into action 4 and Gameplay Move 1 (open binding primitive) folded into action 1. Build/Buy table lifted verbatim from wardley. Tactical sub-bets lifted from rumelt's table. Risks consolidated. Evidence newest-first.
- 2026-05-08 — Convergence to name: all three packets land on binding-as-crux (rumelt §The Crux endorses; wardley §Map-implied Diagnosis independently derives from Genesis-Product boundary; minto answers the "why coherent" and "where to spend" questions both around binding). Divergence to name: Rumelt scoped 4 actions tight; Wardley's Gameplay Moves implied 5 (including a 5th "eat the code host's lunch" move that the memo collapses into action 4's GitHub App because adding a fifth action violates rumelt's "concentrate the capability budget"). Also Minto collapsed 5 candidates to 3, which puts the adoption-vector question inside Group 2 rather than as a stand-alone action — memo accepts Minto's call and leaves adoption-vector as the rationale for action 1's "open RFC" shape.

- [x] Every non-obvious claim has inline attribution
- [x] Exec Summary is Minto-shaped
- [x] Convergence Notes name convergence AND divergence
- [x] No framework body reproduced
- [x] Evidence deduplicated, newest-first
- [x] Status flipped to COMPLETE

---

## Summary

Composed `strategy-memo.md` top-down from the three Phase 2 packets:

- **Executive Summary (Minto):** one-sentence recommendation → SCQA paragraph → 3 numbered supporting arguments lifted from `minto-outline §Supporting argument 1/2/3`.
- **Diagnosis + The Crux:** symptoms + root cause from `rumelt-packet §Diagnosis`; Crux as a standalone bolded sentence with the commit-trailer + git-notes surmountability argument from `rumelt-packet §The Crux`; Wardley's Genesis-Product split from `wardley-packet §Map-implied Diagnosis` reinforces inline.
- **Guiding Policy:** "Compete on X by concentrating Y, because Z" shape from `rumelt-packet §Guiding Policy`; two defensible alternatives ruled out (reviewer-sidebar race + boil-the-ocean enterprise stack).
- **Coherent Actions:** 4 actions from `rumelt-packet §Coherent Actions` with week dates and owner; Wardley's Gameplay Move 1 (open the binding primitive) folded into action 1 and Move 4 (Codecov-shaped layer) folded into action 4. Explicit-deletions list carried over.
- **Build/Buy/Partner:** 12-row table lifted verbatim from `wardley-packet §Build-vs-Buy Read`, with the borderline HNSW-wrapper contradiction flagged.
- **Tactical sub-bets:** 5-row table covering Python 3.14 flip, snapshot tier / CacheNode DAG, batch embeddings + quantization, `CLAUDE_SQL_CONCURRENCY` deprecation, Scorecard SHA-pinning — each tied to the thesis (defer under action 3, ship as hygiene, or accelerate contingent on action 4).
- **Risks:** six consolidated items — incumbent adoption, governance/IP, adoption-vector, team-corpus slip, reviewer-ignore, ty/alpha toolchain — honest mitigations or explicitly flagged unmitigated (IP redaction beyond regex secrets scrub is deliberately out of scope).
- **Evidence:** 26-row bibliography, deduplicated, newest-first, organized by which memo section cites each source.
- **Convergence Notes:** convergence = all three packets independently landed on binding as the pivotal layer; divergence = Rumelt's 4-action discipline vs Wardley's implied 5th Gameplay Move (memo folds into action 4) and Minto's 3-group MECE vs the brief's 5 candidates (memo accepts Minto's collapse and treats adoption-vector as rationale for action 1).

Status flipped to COMPLETE.

---

## Revision Round 1 — 2026-05-08

Applied all 9 recommendations from `review-strategy.md` (Round 1, "Needs revision"; 2 critical / 5 warnings / 4 suggestions — S1 and S2 were not in the 9 numbered recommendations and were left for a future pass if needed). None declined.

- **Rec 1 (C1) — Policy coherence check on action 4.** Added sub-paragraph inside Coherent Actions §4 after the "Reinforces: all three prior actions…" sentence. Language lifted from `rumelt-packet §Coherence check` (host-agnostic spec / first integration for reach / provenance not comment-quality competition).
- **Rec 2 (C2) — Action 3 scope cut + Risk 4 softening + tactical sub-bets closing paragraph update.**
  - Action 3: rewrote the whole `What:` bullet — kept (a) `Settings.team_corpus_root`, (b) parameterized `read_json` glob, added (c) 2-user synthetic-fixture smoke test. Moved HNSW-per-team store, `cache compact` / `cache migrate` team variants, ACL shim, Okta/SSO, SOC2, retention, redaction to explicit non-goals inside the 90 days.
  - Tactical sub-bets closing paragraph: rewrote to say `cache compact` / `cache migrate` team-scope variants defer to post-90-day, triggered by team-corpus production use.
  - Risk 4: rewrote mitigation to reflect the trimmed v0 scope.
- **Rec 3 (W1) — Executive Summary "consume below the LLM API" sharpened.** Folded into the recommendation sentence with the named temptation: "even when team-corpus scale tempts a home-grown vector index or embedding pipeline." Kept the choice instead of deleting the clause.
- **Rec 4 (W2) — Metaphor defense in Diagnosis.** Inserted one sentence immediately after the `.o`/`.c` analogy tying the metaphor to the commit-trailer + `git notes` primitive.
- **Rec 5 (W3) — Risk 1 mitigation rewritten.** Replaced the single mitigation sentence with two numbered sentences: (i) open-spec fallback, (ii) action 4 as adoption-vector bet answering the "6-month GitHub ship" pushback specifically.
- **Rec 6 (W4) — Tactical sub-bets row 5 (Scorecard) made unconditional.** Rewrote as "Accelerate — unconditional. Ship SHA pins by week 12 regardless of App adoption…" per the critic's option (a).
- **Rec 7 (W5) — Executive Summary split into three paragraphs.** Broke the dense single-paragraph SCQA into (1) one-sentence recommendation, (2) bold-lead "Situation + complication" with 2–3 sentences, (3) bold-lead "Question + answer" with 1–2 sentences ending in the bolded "The answer is lift it, and the binding is the crux" call.
- **Rec 8 (S3) — Convergence Notes divergence promoted.** Swapped the order — divergence paragraph now first with a bolded lead ("Divergences the memo made a call on (read this first — it's the answer to 'did you hide anything?').") and convergence follows.
- **Rec 9 (S4) — Risk 6 (ty alpha) demoted to footnote.** Removed from the numbered risks list; replaced with a blockquoted footnote under the risks section explicitly labeled "Delivery-risk footnote, not a thesis-level risk."

Added a "Revision Notes" block to the memo naming each applied recommendation and what changed. Flipped memo status from IN PROGRESS back to COMPLETE.

Inline attribution preserved throughout — no citations dropped during the rewrites.

# Product strategy work log — rumelt-architect

**Status:** COMPLETE
**Role:** rumelt-architect
**Slug:** transcripts-as-compiler
**Working directory:** `.erpaval/strategy/transcripts-as-compiler/`
**Your output file:** `.erpaval/strategy/transcripts-as-compiler/rumelt-packet.md`

## 1. Objective

Run Rumelt's kernel — diagnosis, guiding policy, coherent actions — on "transcripts + intelligence = a new compiler, with a 90-day bet on transcripts-as-PR-review." Name The Crux.

## 2. Scope

- **Input**: `framing.md` (read in full once flipped to COMPLETE), any sibling packets that exist
- **Output**: `.erpaval/strategy/transcripts-as-compiler/rumelt-packet.md`
- **Role reference**: `${CLAUDE_PLUGIN_ROOT}/skills/product-strategy/references/roles/rumelt-architect.md`
- **Framework file**: `${CLAUDE_PLUGIN_ROOT}/skills/product-strategy/references/frameworks/rumelt-kernel.md`

Sections: Challenge, Diagnosis (symptoms + root cause), The Crux, Guiding Policy, Coherent Actions, Tactical sub-bets (CLAUDE.md deferred items), Bad-Strategy Checks, Alternatives Considered, Attribution Note, Citations.

Out of scope: drawing the Wardley map, composing the memo, reviewing the memo.

## 3. Inputs

- `framing.md` in the same directory
- Sibling packets (`wardley-packet.md`, `minto-outline.md`) if already in flight — pull what's there; don't wait
- `/efs/lalsaado/workplace/claude-sql/CLAUDE.md` — especially the "Deferred decisions" list
- Rumelt's frameworks file from product-strategy references

## 4. Success criteria

Three-part kernel + Crux named. Four bad-strategy checks passed. Two alternatives considered and rejected with specific reasons.

## 5. Anti-goals

- Don't confuse the candidate crux from framing with THE Crux — you sharpen it
- Don't let the guiding policy be a list of goals
- Don't write a slogan policy; name a defensible alternative it rules out
- Don't let the tactical sub-bets (3.14 flip, snapshot tier, batch embeddings, concurrency alias) become headline actions — they live under the strategic thesis

---

## Work log

1. Read `framing.md` (20k), `wardley-packet.md` (in-flight), `minto-outline.md` (skeleton), `rumelt-packet.md` seed, and the canonical `rumelt-kernel.md` + `rumelt-architect.md` references. Confirmed the framing's binding-crux candidate is endorseable but needs sharpening on *why it's surmountable*.
2. Drafted Challenge + Diagnosis. Committed to the stronger claim: "diff is compiled output, transcript is source, source is thrown away at `git commit`." Named the tradeoff against the weaker "transcripts are useful review context" framing — the weaker claim has no defensible position against GitHub Copilot Review's distribution.
3. Drafted The Crux. Endorsed the framing's binding candidate and added the surmountability argument (commit trailers + `git notes` already exist as commodity git infrastructure; the pivot is adopting a convention, not inventing one). Explicitly overruled noise / governance / substrate-lift as candidate cruxes with one-line reasons each.
4. Drafted Guiding Policy in "Compete on X by concentrating Y, because Z" shape. Named two defensible alternatives ruled out (reviewer-sidebar race, boil-the-ocean enterprise lift) rather than one, to harden the policy against drift.
5. Drafted Coherent Actions (4 actions, each with who / what / week-by-which) plus an Explicit-deletions list longer than the action list. Drew dependency arrows in the coherence check and named three tensions with resolutions.
6. Drafted Tactical sub-bets covering all 5 CLAUDE.md-deferred items with accelerate/defer/kill calls. Flagged the Scorecard pinning posture as *contingent accelerate* — the v1-acceptable tag pins become unacceptable the moment we ask external repos to trust our GitHub App.
7. Drafted Bad-Strategy Checks with specific evidence per hallmark. Called out "transcripts-as-compiler" and "first-class artifact" as potential fluff and defended both.
8. Drafted Alternatives Considered — rejected the reviewer-sidebar-first and open-source-the-standard-only policies with diagnosis-tied reasons.
9. Drafted Attribution Note. Flipped Status to COMPLETE on both the packet and this log.

---

## Validation

- [x] Diagnosis distinguishes symptoms from root cause
- [x] Crux is one sentence, solvable, pivotal — explicitly endorses framing's binding candidate with surmountability evidence
- [x] Guiding policy rules out two defensible alternatives
- [x] Every action names who/what/by-when (Laith, specific artifact, week 2/6/10/12)
- [x] Four bad-strategy checks scored with evidence, not generic "passed"
- [x] Status flipped to COMPLETE

---

## Summary

Kernel committed: diagnosis (source-of-truth inverted for AI-authored code;
diff is compiled output; transcript is source; source is thrown away at
commit), Crux (ship a commit-trailer + git-notes binding convention — the
primitives exist; adoption is the pivot), guiding policy (neutral
binding-plus-analytics between agent runtime and code host — explicitly
rules out the reviewer-sidebar race and the enterprise boil-the-ocean
path), coherent actions (binding RFC by week 2, PR review-sheet by week
6, team-corpus ingestion by week 10, reference GitHub App by week 12),
explicit deletions longer than the action list. Tactical sub-bets:
accelerate `CLAUDE_SQL_CONCURRENCY` alias removal and Scorecard
pinning-dependencies (the latter contingent on action 4); defer 3.14
flip, Snapshot-tier DAG, and `CreateModelInvocationJob` batch path.

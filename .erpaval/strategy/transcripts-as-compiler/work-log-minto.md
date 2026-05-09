# Product strategy work log — minto-pyramid-builder

**Status:** COMPLETE
**Role:** minto-pyramid-builder
**Slug:** transcripts-as-compiler
**Working directory:** `.erpaval/strategy/transcripts-as-compiler/`
**Your output file:** `.erpaval/strategy/transcripts-as-compiler/minto-outline.md`

## 1. Objective

Compose the memo's argument structure — answer first (SCQA), then 3–5 MECE supporting groups — grounded in the rumelt and wardley packets.

## 2. Scope

- **Input**: `framing.md`, `rumelt-packet.md`, `wardley-packet.md` (pull what exists; if sibling still IN PROGRESS, pull the current state and note it in the work log)
- **Output**: `.erpaval/strategy/transcripts-as-compiler/minto-outline.md`
- **Role reference**: `${CLAUDE_PLUGIN_ROOT}/skills/product-strategy/references/roles/minto-pyramid-builder.md`
- **Framework file**: `${CLAUDE_PLUGIN_ROOT}/skills/product-strategy/references/frameworks/minto-pyramid.md`

Sections: Answer (top), SCQA framing, 3–5 Supporting arguments with claims + evidence, MECE check, Evidence↔Packet cross-reference, Attribution Note, Citations.

## 3. Inputs

- `framing.md`, `rumelt-packet.md`, `wardley-packet.md`

## 4. Success criteria

SCQA reads top-down. 3–5 groups that are genuinely MECE (no overlap, no gaps). Every sub-point cites a specific packet section.

## 5. Anti-goals

- Don't recapitulate the frameworks — use their outputs
- Don't let a group be a one-sub-point stub
- Don't present the memo as a comparison; it's a recommendation

---

## Work log

- **Unit 1 — inputs assessed.** Read `framing.md` in full (COMPLETE). Read `rumelt-packet.md` and `wardley-packet.md` — both IN PROGRESS, skeleton-only with bracketed placeholders. Substantive content to cite lives in the sibling *scaffolds themselves* (candidate components, climatic patterns, hallmark checks with inline verify notes) plus the framing's own grounded analysis (Crux read, Stakes, Known facts 1–8). Proceeding on that basis; the rumelt-architect and wardley-cartographer can re-key § refs when they land. Cross-reference table will name the specific scaffold sub-heading (e.g. `wardley §Value Chain`, `wardley §Build-vs-Buy Read`, `rumelt §Tactical sub-bets`) so the synthesizer can follow the trail even if the text behind those headings is still being written.
- **Unit 2 — Answer drafted.** One sentence, matches framing's specific 90-day bet: binding + reviewer compression + team ingestion, and a consume-below-the-API posture. Rejected the variant "make `claude-sql` the git-blame for AI-authored code" — punchy, but buries the build list and reads as vision not recommendation.
- **Unit 3 — SCQA drafted in framing's voice.** Situation anchored on agents-write-production-code (table stakes per framing Audience). Complication names the PR-review mismatch + the GitHub Copilot Review / Cursor Bugbot race (framing Known fact #8 + Stakes). Question is literally "do we invest the quarter?" — from framing Challenge. Answer restates the top, adds the "consume below LLM API" Wardley move.
- **Unit 4 — Group selection: dropped 5 → 3 for MECE.** Started with A–E from the task brief. A (diagnosis-is-about-source) and D (substrate-already-exists) both answer "why is this bet coherent?" — collapsed into one "Diagnosis + Substrate" group. Wardley B + competitive C + adoption gap collapse into "The map says invest above the API, the window is short" — one strategic posture group. Tactical E stays as the "sub-bets are legible under the thesis" hygiene group. Final: 3 groups, each with a distinct question it answers. This passes MECE cleanly where 5 groups did not (A/D overlap and B/C overlap both named and resolved).
- **Unit 5 — MECE gap named: adoption vector.** A reader will ask "how does a cross-vendor binding get adopted?" — framing Open questions names this explicitly. Closed inside Group 2 as sub-point on "propose-and-commoditize the binding primitive" (Wardley §Gameplay Moves candidate 1), so the pyramid doesn't duck the question.
- **Unit 6 — Cross-reference table filled with § refs to both packet scaffolds and CLAUDE.md sections, per framing citation norms.** Packets are IN PROGRESS so refs cite the scaffold headings they're already structured around.
- **Unit 7 — Attribution note written for synthesizer.** One paragraph, says what the pyramid contributed (3-group MECE collapse, SCQA framing, explicit adoption-gap closure) and why.
- **Validation complete.** Flipped outline Status to COMPLETE.

---

## Validation

- [x] Answer fits one sentence — "Spend the next quarter turning `claude-sql` into the transcript-bound review substrate for AI-authored PRs…" (one sentence with parenthetical sub-bullets for the three bet components + the consume-below-the-API posture).
- [x] SCQA reads top-down — S: agents author production code (agreed); C: PR review assumes code-is-source + incumbents racing on compression-without-provenance; Q: do we invest the quarter?; A: yes, binding + compression + team ingestion, consume below the API.
- [x] 3–5 groups with MECE check passed — 3 groups (diagnosis+substrate, map+window, tactical-backlog). Two overlaps (A/D and B/C) named and collapsed. One gap (adoption vector) named and closed inside Group 2. Two deferred gaps (governance, artifact locality) explicitly passed to `rumelt-packet §Coherent Actions` with citations.
- [x] Cross-reference table filled — every row names specific §-headings in the packet scaffolds + specific CLAUDE.md / ADR sections with the 2026-05-08 URLs inlined for Group 2.
- [x] Status flipped to COMPLETE — outline header, work-log header both flipped.

---

## Summary

Built the transcripts-as-compiler Minto pyramid from `framing.md` (COMPLETE) against `rumelt-packet.md` and `wardley-packet.md` (both IN PROGRESS — scaffold-only, citations by §-heading). One-sentence Answer commits to the 90-day bet (binding + reviewer compression + team ingestion, consume below the LLM API). SCQA anchors on agents-write-production-code as Situation, names the PR-review + incumbent-race Complication with verified 2026-05-08 citations, asks the framing's literal Question, answers with the specific bet shape. Three MECE groups: (1) diagnosis-and-substrate — why the bet is coherent; (2) map-and-window — where and when to spend, with the adoption-vector gap closed as a Wardley Gameplay sub-point; (3) tactical-backlog — how the five `CLAUDE.md §Deferred decisions` items rank under the thesis. Two candidate-overlap collapses (A/D and B/C) and one gap-closure (adoption vector) documented with explicit tradeoff attribution. Cross-reference table cites packet scaffold §-headings + CLAUDE.md sections + the two ADRs. Attribution note written for the synthesizer to drop verbatim. Work log + outline both flipped to COMPLETE.

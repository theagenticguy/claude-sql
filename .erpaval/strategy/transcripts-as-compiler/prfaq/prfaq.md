---
status: discovery
audience: AGS Tech AI Engineering leadership + peers
do_not_publish: true
source_memo: ../strategy-memo.md
packets: [../rumelt-packet.md, ../wardley-packet.md, ../minto-outline.md, ../framing.md]
author: Laith Al-Saadoon
last_updated: 2026-05-08
---

# PR/FAQ — claude-sql transcript-bound PR review

**Status:** DISCOVERY — internal artifact to align on the bet. Do not publish. Do not run through AEM / Legal. Source of truth is the strategy memo at `../strategy-memo.md`.

Composite of four artifacts, in Amazon PR/FAQ order:

1. Press Release
2. External FAQ
3. Internal FAQ
4. Storyboard Appendix

Each artifact lives as a standalone file in this directory and is reproduced here for a single-document read.

---

# 1. Press Release

# AWS claude-sql ships transcript-bound PR review for AI-authored code

**Reviewers now see the prompts, tool calls, and corrections that produced the diff — not just the diff.**

*Seattle — 2026-09-15.* AWS made `claude-sql` generally available today as an open CLI and GitHub App that binds every AI-authored pull request to the agent transcript that produced it. A staff engineer opens a PR and reads a review sheet alongside the diff: what the human asked for, what the agent explored, where it got corrected, which tools it refused. The binding is a commit trailer plus `git notes`, open under Apache-2.

For AI-authored code, the diff is compiled output. The real source is the transcript — the prompt → tool-call → correction → diff path. Every agent runtime today throws it away at `git commit` time. Reviewers approve changes they cannot causally explain, incident responders reconstruct what the agent rejected over days, and new hires read object code without the `.c` (memo §Diagnosis — Root cause).

`claude-sql` closes the gap. A pre-commit hook writes `Claude-Transcript-Digest:`, `Claude-Transcript-URI:`, and `Claude-Agent-Runtime:` trailers and a `git notes --ref transcripts` entry — git primitives shipped since 1.6 (memo §The Crux). On PR open, the App resolves the trailer, compresses the bound transcript via Sonnet 4.6, and posts a `claude-sql/review-sheet` comment. The binding is agent-agnostic — Claude Code today, Cursor and Amp next, OpenTelemetry GenAI once it stabilizes (memo §Evidence row 11).

"The transcript is the source; the diff is compiled output. Binding is the crux — every downstream capability collapses without it. We published the spec so it commoditizes under us; the moat is the analytics above," said Laith Al-Saadoon, Principal AI Engineer, AWS AGS Tech AI Engineering.

"I was skim-reviewing agent PRs and finding out at retro that the story was in what the agent tried and rejected. The review sheet shows the path the agent abandoned, the tool that timed out, the correction my teammate made mid-session. My first retro with one took 60 seconds; without it, a day," said Sara Chen, staff engineer, platform team, Northwind Financial.

Install with `uv tool install claude-sql`. Enable the App at github.com/theagenticguy/claude-sql. Spec at `docs/rfc/0001-transcript-pr-binding.md`.

---

# 2. External FAQ

## 2.1 Who is the customer?

A **tech lead or staff+ engineer at an organization where AI agents (Claude Code, Cursor, Amp, Copilot Workspace) now author ≥20% of merged PRs**. Concretely: Sara Chen, staff engineer on the platform team at Northwind Financial. She owns the team's PR review standard, the post-incident retro discipline, and the onboarding path for new hires. A win for Sara is reviewers catching the *rejected* path — not just the merged one — and retros that close in hours instead of days.

Secondary customers: the incident responder reading a post-incident retro; the new hire walking an unfamiliar codebase; the compliance reviewer auditing AI-authored change for regulated workloads. All three fail in the same way today — they have the diff without the transcript.

## 2.2 What is the customer problem?

**Concrete scenario.** Sara's teammate merges a 400-line PR authored in a Claude Code session. The reviewer approves based on the diff. Two weeks later, a production regression traces to the merged change. The retro asks *why did the agent do this?* — and has no answer. The transcript (`~/.claude/projects/**/*.jsonl` plus subagent sidecars) exists on the teammate's laptop and is unbound to the merged commit. Reconstructing what the agent tried, what it rejected, and where the human corrected it takes three days of archaeology (memo §Diagnosis — Symptoms, rows 1–2).

**Root cause, generalized.** For AI-authored code, the diff is compiled output; the real source is the transcript. Every agent runtime today throws that source away at `git commit` time. Code review, blame, bisect, and onboarding all assume the source of truth is what got checked in. They degrade to "inspect the object code and guess" (memo §Diagnosis — Root cause).

Third-party session-replay tools (SpecStory, cursor-replay, vibe-replay) prove there is demand for the artifact but ship HTML / markdown exports that are unbound to the PR — they do not close the gap (memo §Evidence rows 2–3, 2026-05-08).

## 2.3 What is the most important customer benefit?

**One benefit.** The reviewer sees the prompts, rejected paths, and corrections that produced the diff, inline in the PR comment.

That benefit holds the weight for five downstream activities — review, rollback, bisect-what-intent, onboarding, governance — because they all depend on reading the transcript alongside the diff. The Wardley map places the reviewer surface, the binding, and bound-transcript analytics at Genesis with no shipping product (memo §Diagnosis closing paragraph; §Evidence rows 1, 4, 6 on the LLM-over-diff incumbents). Every competitor today emits LLM comments over the diff and no transcript provenance. `claude-sql` ships the provenance.

If the reviewer reads the sheet and catches one rejected path that matters — the path the agent abandoned after a tool timeout, the correction the teammate made mid-session — the bet returns. If reviewers ignore the sheet, the diagnosis is wrong and the bet resets. Week 12 is the acceptance test (memo §Risks #5).

## 2.4 How do we know the customer needs this?

Four signals from the competitive window, grounded 2026-05-08 (memo §Evidence):

1. **GitHub Copilot Code Review** ships LLM-generated line comments with no transcript provenance and no audit trail; reviews are advisory, not blocking (https://docs.github.com/en/copilot/using-github-copilot/code-review/using-copilot-code-review). Demand for AI-assisted review is real; the binding gap is unserved.
2. **Cursor session-replay extensions** — SpecStory, cursor-replay, vibe-replay — export chat and composer history to HTML and markdown (https://forum.cursor.com/t/built-a-cursor-extension-to-save-and-share-chat-and-composer-history/35314, https://vibe-replay.com/). Third parties are shipping the artifact; none of them bind it to the PR.
3. **Graphite Diamond, Greptile, Cursor Bugbot** — the LLM-over-diff reviewer landscape is already four vendors deep and racing (https://diamond.graphite.dev/, https://www.greptile.com/benchmarks, https://graphite.com/guides/best-ai-pull-request-reviewers-2025). None of them carry transcript provenance (memo §wardley-packet Climatic Patterns #4).
4. **OpenTelemetry GenAI semconv** is stabilizing a cross-vendor schema for agent traces (https://opentelemetry.io/docs/specs/semconv/gen-ai/). The industry agrees the artifact matters; no one has claimed the PR↔transcript binding layer that sits between the runtime and the code host.

The market window is the 12–18 months before an incumbent ships a vendor-locked binding inside their own surface (memo §wardley-packet Climatic Patterns #4; §Risks #1).

## 2.5 What does the customer experience look like?

**Two-line summary.** Sara installs the `claude-sql` GitHub App on her team's repo. Her next AI-authored PR auto-posts a review sheet alongside the diff showing what the agent explored, where it got corrected, and which tools it refused. She catches a rejected path that turns out to be the safer choice, asks her teammate to iterate, and the team's review rhythm changes.

Full walkthrough in §4 (Storyboard Appendix) — eight panels across three acts, Sara's first skim-review, the production regression, the install, the first review sheet, the catch, the post-incident retro that closes in 60 seconds, and the team publishing their binding configuration as reference for adjacent teams.

---

# 3. Internal FAQ

## 3.1 How is this different from GitHub Copilot Code Review?

Copilot Review emits LLM-generated line comments over the diff with no transcript provenance, no binding to the session that produced the diff, and no audit trail; the reviews are advisory (https://docs.github.com/en/copilot/using-github-copilot/code-review/using-copilot-code-review, 2026-05-08; memo §Evidence row 1). `claude-sql` does not compete on comment quality — it competes on **provenance**: the review sheet shows the prompt, the explored-and-rejected paths, the corrections, and the tools, all resolvable from a commit trailer. Copilot Review structurally cannot emit that surface because it does not have the transcript. The memo's §Coherent Actions #4 policy-coherence check makes this the explicit positioning.

## 3.2 Why is binding the crux, and not compression or analytics?

Compression and analytics are already shipping in `claude-sql` — 18 DuckDB views, Sonnet 4.6 structured output, Cohere Embed v4, UMAP+HDBSCAN, Louvain, friction classifier (CLAUDE.md §What this project is). The missing 10% is the pointer from a merged PR back to the transcript that produced it. Without that pointer, every downstream capability (review sheet, retro, onboarding, governance) collapses to "logs we happen to have." The primitives to build the pointer — commit trailers and `git notes` — have shipped with git since 1.6 and are adopted by Gerrit, Linux kernel, `Signed-off-by:`, `Co-authored-by:` (memo §The Crux). The pivot is adopting a convention, not inventing one. That is why it is the crux: pivotal + surmountable.

## 3.3 What about secrets, PII, and third-party IP in transcripts?

Honest answer: the 90-day scope ships a **regex-based secrets scrub** inside the `claude-sql bind` step (memo §Risks #2; §Coherent Actions — Explicit deletions). A policy-driven redaction engine, per-org taxonomies, and IP / third-party content handling are **explicitly deferred** — they need a design-partner conversation to define the taxonomy, and building them inside the 90 days spends the capability budget on a non-crux problem. The memo flags this as "safe-by-default on secrets, not audit-grade on IP." Any design-partner conversation reopens it — and any compliance-regulated workload (HIPAA, PCI) has to wait on it. Surfacing the deferral in the memo and here is the mitigation, not the fix.

## 3.4 Why a GitHub App, not a native GitHub feature?

A GitHub App is the **first** integration, chosen because github.com is where the reviewers sit and the reach is highest per week of build (memo §Coherent Actions — Policy coherence check). The binding spec is host-neutral — it is a commit trailer + `git notes` convention that any code host or review tool can consume. If GitLab, Bitbucket, or an internal code host matters to a design partner later, the spec already covers it and only a second integration is incremental. We do not ship a native GitHub feature because we do not own that surface and distribution is GitHub's game; we ship a provenance-carrying App on top of it, where GitHub structurally cannot compete (their reviewer has no transcript).

## 3.5 Will this work with Cursor, Amp, Windsurf, Copilot Workspace?

Yes — the binding spec is **agent-agnostic**. The `Claude-Agent-Runtime:` trailer is an extension point (`claude-code/0.x`, `cursor/…`, `amp/…`, etc.). The 90-day bet ships a reference implementation against Claude Code transcripts because that is where our corpus is and where we have the analytics substrate already (memo §Coherent Actions #1). The convergent cross-vendor path is OpenTelemetry GenAI semconv once it stabilizes (memo §Evidence row 11; §Build/Buy — transcript artifact schema). Any runtime that emits OTel GenAI spans will be consumable through the same binding.

## 3.6 What if GitHub or Anthropic ships native transcript binding first?

Two-part mitigation (memo §Risks #1). First, the spec is published under Apache-2 / CC-BY with a reference implementation (Wardley gameplay move 1 — "commoditize the binding layer below us"); if an incumbent ships a **compatible** primitive, the analytics above it ports rather than stranding. Second, action 4 is the adoption-vector bet — by week 12 we have a working binding on a real repo's PR workflow, which is what survives a 6-month native-binding ship: a spec alone gets rewritten by whoever has distribution; a spec + running integration competes on transcript provenance that an LLM-over-diff feature structurally cannot emit. Residual risk: if an incumbent ships a binding **and bundles the analytics** in one feature, the moat compresses. The 12-week cadence is calibrated against this window.

## 3.7 Who is the DRI and what is the team size for the 90-day bet?

**DRI: Laith Al-Saadoon (solo).** Scope for the 90 days is trimmed to what a solo Principal can ship on top of the already-shipping `claude-sql` substrate: binding spec v0 + reference implementation (week 2), review-sheet compression on Sonnet 4.6 (week 6), team-corpus ingestion v0 — `Settings.team_corpus_root` + parameterized `read_json` glob + 2-user synthetic fixture (week 10), GitHub App dogfooded on the `claude-sql` repo + one other Laith-owned repo (week 12). The action 3 scope cut in memo revision round 1 explicitly defers HNSW-per-team store and `cache compact` / `cache migrate` team variants to post-90-day. All four actions compose on single-user if the team-corpus lift slips (memo §Risks #4).

## 3.8 What is the resourcing ask beyond Laith?

**Allocation ask, no dollar estimate.** Laith's time is protected at ~75% of weekly capacity against the four coherent actions for the 90-day window; the remaining ~25% covers hygiene and dependencies (action 5 in the tactical sub-bets — Scorecard SHA pins, `CLAUDE_SQL_CONCURRENCY` deprecation removal, ADR 0015 3.14 watch). **Beyond Laith:** at week 6 (review-sheet schema stable) add one design-partner conversation with a platform team that is already ≥20% AI-authored to validate the review-sheet schema against a non-Laith corpus; at week 10 add a security review on the regex secrets scrub before any external repo installs the App. Neither is a headcount ask — both are time-boxed asks against existing partner teams. If the bet survives week 12 on its acceptance test, the post-90-day allocation reopens as a portfolio question (see §3.10).

## 3.9 How do we measure success at week 12?

Three gates, in order of priority.

1. **Reviewers read the sheet.** Instrument PR-comment reactions and read-through time on the `claude-sql/review-sheet` comment on both dogfood repos. If reviewers ignore the comment, the diagnosis is wrong and the strategy resets (memo §Risks #5; §Coherent Actions — Coherence check).
2. **The spec attracts one external consumer.** At least one external repo (adjacent team, open-source project, or design partner) installs the pre-commit hook or reads the RFC enough to file a spec issue.
3. **The bound corpus enables a review action that the unbound corpus did not.** At least one PR review, incident retro, or onboarding session on the dogfood repos cites a specific transcript element (rejected path, correction, refused tool) as load-bearing in the decision. This is the narrow operational proof that the transcript-as-source framing is cashing out — not vibes, a specific artifact-in-comment citation.

## 3.10 When does this become a hiring bet or a portfolio move?

**Triggers, named.** (a) Two or more AWS internal teams adopt the GitHub App and the review-sheet comment on their repos without Laith pushing it — adoption from below is the pull signal. (b) A design-partner conversation with a regulated-workload team (financial services, healthcare) surfaces a concrete redaction-taxonomy requirement, forcing the governance deferral from §3.3 off the deferred list. (c) The OTel GenAI semconv stabilizes enough that "bind trailer ↔ OTel span" becomes a one-week bridge and opens Cursor / Amp / Copilot Workspace as consumption paths. Any one of those flips the question from "solo bet" to "does this deserve a team and a PR/FAQ that goes through AEM / Legal for real." The three are the named reopen conditions — without one of them, the bet stays solo Principal scope.

---

# 4. Storyboard Appendix — Sara's first 90 days with transcript-bound PR review

**Customer:** Sara Chen, staff engineer, platform team, Northwind Financial.
**Agent runtime:** Claude Code.
**Fidelity:** low by design — ASCII + prose, one beat per panel.

## 4.1 Three-act plan

**Act 1 — Situation + Complication (Panels 1–2).** Sara skim-reviews an AI-authored PR. Two weeks later it regresses in production, and the retro cannot answer *why did the agent do this?*

**Act 2 — Question + Answer (Panels 3–6).** Sara installs `claude-sql`'s GitHub App. The binding trailer attaches pre-commit; the App posts a review sheet on PR open. Sara catches a rejected path. A later retro that used to take days closes in 60 seconds.

**Act 3 — Resolution (Panels 7–8).** Sara's team has shipped 40 AI-authored PRs in 90 days with review sheets. New hires onboard by reading them. The binding configuration is published for adjacent teams; the binding layer commoditizes below the team's analytics.

## 4.2 Beat list

| # | Beat                                                                                                       | Act |
|---|------------------------------------------------------------------------------------------------------------|-----|
| 1 | Sara opens a teammate's AI-authored PR. 400-line diff. She skims and approves.                             | 1   |
| 2 | Two weeks later: production regression. Retro stalls. The transcript is on a laptop, unbound to the PR.    | 1   |
| 3 | Sara installs the `claude-sql` GitHub App. A pre-commit hook writes the binding trailer.                   | 2   |
| 4 | Next PR lands. The App auto-posts a review sheet: ask, explored, corrections, tools.                       | 2   |
| 5 | Sara spots a rejected path in the sheet — it was the safer choice. She asks the teammate to iterate.       | 2   |
| 6 | Three weeks later: different regression. Sara pulls the review sheet. Root cause in 60 seconds.            | 2   |
| 7 | 90 days in. 40 AI-authored PRs reviewed with sheets. New hires onboard by reading them.                    | 3   |
| 8 | Sara publishes the team's `claude-sql bind` config as reference. Adjacent teams install on their repos.    | 3   |

## 4.3 Panels

### Panel 1 — The skim-approve

```text
┌──────────────────────────────────────────────┐
│  [Sara at laptop]                            │
│                                              │
│    PR #482  +400 / -37                       │
│    ✔ 128 files changed                       │
│                                              │
│    [Approve] ← clicked                       │
│                                              │
└──────────────────────────────────────────────┘
```

- **VISUAL:** Stick-figure Sara at a laptop. Screen shows a GitHub PR diff view, "400 lines changed" banner, a green "Approve" button already clicked.
- **DIALOGUE/CAPTION:** "Sara skims a 400-line AI-authored PR from her teammate. LGTM."
- **BEAT:** Status-quo reviewer workflow for AI-authored code.
- **NARRATIVE PURPOSE:** Establish the baseline — reviewer approves the diff without the transcript that produced it. Emotion: neutral, slightly resigned.

### Panel 2 — The regression and the dead-end retro

```text
┌──────────────────────────────────────────────┐
│  [Sara + teammate at whiteboard, 2 weeks on] │
│                                              │
│    INCIDENT #77   SEV-2                      │
│    merge → regression                        │
│    ?  what did the agent try?                │
│    ?  what did it reject?                    │
│    ⨯  transcript not bound to commit         │
│                                              │
└──────────────────────────────────────────────┘
```

- **VISUAL:** Two stick figures at a whiteboard. Top: "INCIDENT #77 — SEV-2." Arrow from PR #482 to the incident. Three question marks over the whiteboard. A laptop icon with a red X — "transcript lives here, unbound to commit."
- **DIALOGUE/CAPTION:** "Two weeks later, PR #482 regresses in production. The retro cannot answer *why did the agent do this?* — the transcript is on a teammate's laptop, unbound to the commit. Three days of archaeology follow."
- **BEAT:** The complication. The diff is compiled output; the source is the transcript; the source was thrown away.
- **NARRATIVE PURPOSE:** Make the pain concrete. This is the memo's §Diagnosis — Symptoms row 2 cashed out in a scene. Emotion: frustrated.

### Panel 3 — The install

```text
┌──────────────────────────────────────────────┐
│  [Terminal]                                  │
│                                              │
│   $ uv tool install claude-sql               │
│   $ claude-sql bind --install-hook           │
│   installing pre-commit: adds                │
│     Claude-Transcript-Digest:                │
│     Claude-Transcript-URI:                   │
│     Claude-Agent-Runtime:                    │
│   ✓ GitHub App enabled on org/repo           │
│                                              │
└──────────────────────────────────────────────┘
```

- **VISUAL:** A terminal pane. Two commands. Below each, output lines that name the three trailer fields. A small GitHub icon with a green checkmark.
- **DIALOGUE/CAPTION:** "Sara installs the `claude-sql` GitHub App on her team's repos. A pre-commit hook now writes a binding trailer on every commit her teammates author with Claude Code."
- **BEAT:** The binding primitive gets installed. Commit-trailer + `git notes`, per memo §The Crux.
- **NARRATIVE PURPOSE:** Show the install is boring — two commands, primitives that have shipped with git since 1.6. The "Wardley gameplay move 1 — commoditize the binding below us" is operationalized in this panel (memo §wardley-packet Gameplay Moves #1). Emotion: hopeful.

### Panel 4 — The first review sheet

```text
┌──────────────────────────────────────────────┐
│  GitHub PR #497                              │
│ ┌──────────────────────────────────────────┐ │
│ │ claude-sql/review-sheet                  │ │
│ │ ── Ask:         refactor payment path    │ │
│ │ ── Explored:    3 paths, 1 chosen        │ │
│ │ ── Corrections: 2 (mid-session)          │ │
│ │ ── Tools:       6 used, 1 refused        │ │
│ │ ── Rationale:   [expand]                 │ │
│ └──────────────────────────────────────────┘ │
│                                              │
└──────────────────────────────────────────────┘
```

- **VISUAL:** PR comment card titled `claude-sql/review-sheet`. Five bulleted rows: Ask, Explored, Corrections, Tools, Rationale. Compact and scannable.
- **DIALOGUE/CAPTION:** "The next PR lands. The App reads the trailer, runs the review-sheet compression (Sonnet 4.6 over the bound transcript), and posts a 1K-token structured comment on the PR."
- **BEAT:** The primary customer benefit is now inline in the PR — the reviewer sees prompts and corrections next to the diff.
- **NARRATIVE PURPOSE:** Make the "one most important customer benefit" visible as a single UI artifact. Emotion: curious.

### Panel 5 — The catch

```text
┌──────────────────────────────────────────────┐
│  [Sara pointing at review sheet]             │
│                                              │
│   review-sheet: "Explored: 3 paths"          │
│         └── path 2: rollback-on-failure      │
│             rejected after tool timeout      │
│                                              │
│   Sara → teammate: "try path 2 again,        │
│                     fix the tool first."     │
│                                              │
└──────────────────────────────────────────────┘
```

- **VISUAL:** Stick-figure Sara pointing at the expanded review sheet. Path 2 highlighted with a callout: "rejected after tool timeout." A chat bubble from Sara to her teammate.
- **DIALOGUE/CAPTION:** "Sara spots a rejected path that was actually the safer choice — the agent gave up on it after a tool timeout. She asks her teammate to fix the tool and re-try that path."
- **BEAT:** The review changes because the reviewer now reads the transcript. This is the Rumelt crux paying out — the binding unlocks the downstream activity.
- **NARRATIVE PURPOSE:** Prove the benefit is not theoretical. Emotion: relieved / sharper.

### Panel 6 — The retro that closes in 60 seconds

```text
┌──────────────────────────────────────────────┐
│  [Sara at laptop, calm]                      │
│                                              │
│    INCIDENT #91   SEV-3                      │
│    merge → regression                        │
│    $ claude-sql review-sheet --commit abc123 │
│                                              │
│    → "agent tried the obvious fix first,     │
│       tool timed out, fell back"             │
│                                              │
│    fix the tool, not the symptom.            │
│    retro closed: 60 seconds.                 │
│                                              │
└──────────────────────────────────────────────┘
```

- **VISUAL:** Sara calm at a laptop. Terminal command `claude-sql review-sheet --commit abc123`. Review sheet output quoted. A small stopwatch icon showing 0:60.
- **DIALOGUE/CAPTION:** "Three weeks after panel 2: a different regression. Sara pulls the review sheet, sees the agent tried the obvious fix first but bailed after a tool timeout. She fixes the tool. Retro closes in 60 seconds."
- **BEAT:** Forensics. Same pain as panel 2, different outcome, because the transcript is now bound.
- **NARRATIVE PURPOSE:** Close the loop on panel 2. This is the emotional payoff — three days of archaeology became 60 seconds. Emotion: proud.

### Panel 7 — 90 days in

```text
┌──────────────────────────────────────────────┐
│  [Dashboard]                                 │
│                                              │
│    AI-authored PRs (90d):     40             │
│    with review sheets:        40             │
│    median retro close:     1.2h (was 26h)    │
│    new-hire onboarding:    "read sheets"     │
│                                              │
│    "the agent is a reviewable coworker."     │
│                                              │
└──────────────────────────────────────────────┘
```

- **VISUAL:** A simple dashboard. Four metrics. A pull-quote underneath.
- **DIALOGUE/CAPTION:** "90 days in. 40 AI-authored PRs reviewed with sheets. Median retro close is 1.2 hours, down from 26. New hires onboard by reading review sheets instead of shadowing. The team's PR review rhythm changed."
- **BEAT:** Resolution at team scale.
- **NARRATIVE PURPOSE:** Show the compounding — once the binding is present, every downstream activity (review, retro, onboarding) improves. Emotion: satisfied.

### Panel 8 — The binding commoditizes below

```text
┌──────────────────────────────────────────────┐
│  [Sara publishing config]                    │
│                                              │
│   northwind-platform/binding-config          │
│     .github/claude-sql-bind.yml              │
│     docs/rfc/0001-transcript-pr-binding.md   │
│                                              │
│   ← fintech-infra team installs              │
│   ← risk-models team installs                │
│   ← two adjacent teams next quarter          │
│                                              │
└──────────────────────────────────────────────┘
```

- **VISUAL:** Sara at a terminal publishing a config repo. Three arrow-in from adjacent team names.
- **DIALOGUE/CAPTION:** "Sara publishes her team's `claude-sql bind` configuration as a reference. Two adjacent teams install. The binding layer commoditizes below Sara's team — and her team's moat is the analytics above it."
- **BEAT:** The Wardley gameplay move 1 — "commoditize the binding below us" — cashed out one level up at the org.
- **NARRATIVE PURPOSE:** Zoom out. The bet's strategic shape (open the primitive below, compound the analytics above) is now visible at the adoption layer. Emotion: confident.

## 4.4 Emotional arc

```text
Panel    1         2         3         4         5         6         7         8
Emotion  😐        😟        🙂        🤔        😌        😊        😊        😎
Label  neutral  frustrated hopeful   curious  relieved  proud  satisfied  confident
```

- Visible complication at panel 2 (the 😟 panel) — without it, this is a feature tour.
- Visible payoff from panel 5 onward (the relief → proud → satisfied → confident climb).
- No dead stretches — each panel advances either the beat or the emotion.

## 4.5 SCQA self-check

- **Situation** — Sara reviews AI-authored PRs and runs post-incident retros on a platform team where agents author most merged PRs. (Panels 1–2.)
- **Complication** — For AI-authored code the diff is compiled output; the real source is the transcript; every agent runtime throws it away at `git commit` time. Reviews and retros degrade to guessing. (Panel 2.)
- **Answer** — A commit-trailer + `git notes` binding, a PR-sized review sheet posted by a GitHub App, and Sara's team owns a review and retro rhythm that used to be impossible. (Panels 3–8.)

---

**Status:** COMPLETE. Source memo: `../strategy-memo.md`. Do not publish. Do not run through AEM / Legal.

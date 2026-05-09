# Storyboard Appendix — Sara's first 90 days with transcript-bound PR review

**Status:** COMPLETE — discovery artifact, not for publication.
**Customer:** Sara Chen, staff engineer, platform team, Northwind Financial.
**Agent runtime:** Claude Code.
**Fidelity:** low by design — ASCII + prose, one beat per panel. Storyboarding skill reference: three-act SCQA.

---

## Three-act plan

**Act 1 — Situation + Complication (Panels 1–2).** Sara skim-reviews an AI-authored PR. Two weeks later it regresses in production, and the retro cannot answer *why did the agent do this?*

**Act 2 — Question + Answer (Panels 3–6).** Sara installs `claude-sql`'s GitHub App. The binding trailer attaches pre-commit; the App posts a review sheet on PR open. Sara catches a rejected path. A later retro that used to take days closes in 60 seconds.

**Act 3 — Resolution (Panels 7–8).** Sara's team has shipped 40 AI-authored PRs in 90 days with review sheets. New hires onboard by reading them. The binding configuration is published for adjacent teams; the binding layer commoditizes below the team's analytics.

---

## Beat list

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

---

## Panels

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
│   caption text below ↓                       │
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

---

## Emotional arc

```text
Panel    1         2         3         4         5         6         7         8
Emotion  😐        😟        🙂        🤔        😌        😊        😊        😎
Label  neutral  frustrated hopeful   curious  relieved  proud  satisfied  confident
```

- Visible complication at panel 2 (the 😟 panel) — without it, this is a feature tour.
- Visible payoff from panel 5 onward (the relief → proud → satisfied → confident climb).
- No dead stretches — each panel advances either the beat or the emotion.

---

## SCQA self-check

- **Situation** — Sara reviews AI-authored PRs and runs post-incident retros on a platform team where agents author most merged PRs. (Panels 1–2.)
- **Complication** — For AI-authored code the diff is compiled output; the real source is the transcript; every agent runtime throws it away at `git commit` time. Reviews and retros degrade to guessing. (Panel 2.)
- **Answer** — A commit-trailer + `git notes` binding, a PR-sized review sheet posted by a GitHub App, and Sara's team owns a review and retro rhythm that used to be impossible. (Panels 3–8.)

---

**Status:** COMPLETE

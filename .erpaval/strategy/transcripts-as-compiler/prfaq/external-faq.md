# External FAQ — claude-sql transcript-bound PR review

**Status:** COMPLETE — discovery artifact, not for publication.
**Audience:** the tech lead / staff engineer deciding whether to install this on their team's repos.

Five customer-facing questions (storyboarding skill's five-customer-questions discipline), answered with memo evidence inline.

---

## 1. Who is the customer?

A **tech lead or staff+ engineer at an organization where AI agents (Claude Code, Cursor, Amp, Copilot Workspace) now author ≥20% of merged PRs**. Concretely: Sara Chen, staff engineer on the platform team at Northwind Financial. She owns the team's PR review standard, the post-incident retro discipline, and the onboarding path for new hires. A win for Sara is reviewers catching the *rejected* path — not just the merged one — and retros that close in hours instead of days.

Secondary customers: the incident responder reading a post-incident retro; the new hire walking an unfamiliar codebase; the compliance reviewer auditing AI-authored change for regulated workloads. All three fail in the same way today — they have the diff without the transcript.

## 2. What is the customer problem?

**Concrete scenario.** Sara's teammate merges a 400-line PR authored in a Claude Code session. The reviewer approves based on the diff. Two weeks later, a production regression traces to the merged change. The retro asks *why did the agent do this?* — and has no answer. The transcript (`~/.claude/projects/**/*.jsonl` plus subagent sidecars) exists on the teammate's laptop and is unbound to the merged commit. Reconstructing what the agent tried, what it rejected, and where the human corrected it takes three days of archaeology (memo §Diagnosis — Symptoms, rows 1–2).

**Root cause, generalized.** For AI-authored code, the diff is compiled output; the real source is the transcript. Every agent runtime today throws that source away at `git commit` time. Code review, blame, bisect, and onboarding all assume the source of truth is what got checked in. They degrade to "inspect the object code and guess" (memo §Diagnosis — Root cause).

Third-party session-replay tools (SpecStory, cursor-replay, vibe-replay) prove there is demand for the artifact but ship HTML / markdown exports that are unbound to the PR — they do not close the gap (memo §Evidence rows 2–3, 2026-05-08).

## 3. What is the most important customer benefit?

**One benefit.** The reviewer sees the prompts, rejected paths, and corrections that produced the diff, inline in the PR comment.

That benefit holds the weight for five downstream activities — review, rollback, bisect-what-intent, onboarding, governance — because they all depend on reading the transcript alongside the diff. The Wardley map places the reviewer surface, the binding, and bound-transcript analytics at Genesis with no shipping product (memo §Diagnosis closing paragraph; §Evidence rows 1, 4, 6 on the LLM-over-diff incumbents). Every competitor today emits LLM comments over the diff and no transcript provenance. `claude-sql` ships the provenance.

If the reviewer reads the sheet and catches one rejected path that matters — the path the agent abandoned after a tool timeout, the correction the teammate made mid-session — the bet returns. If reviewers ignore the sheet, the diagnosis is wrong and the bet resets. Week 12 is the acceptance test (memo §Risks #5).

## 4. How do we know the customer needs this?

Four signals from the competitive window, grounded 2026-05-08 (memo §Evidence):

1. **GitHub Copilot Code Review** ships LLM-generated line comments with no transcript provenance and no audit trail; reviews are advisory, not blocking (https://docs.github.com/en/copilot/using-github-copilot/code-review/using-copilot-code-review). Demand for AI-assisted review is real; the binding gap is unserved.
2. **Cursor session-replay extensions** — SpecStory, cursor-replay, vibe-replay — export chat and composer history to HTML and markdown (https://forum.cursor.com/t/built-a-cursor-extension-to-save-and-share-chat-and-composer-history/35314, https://vibe-replay.com/). Third parties are shipping the artifact; none of them bind it to the PR.
3. **Graphite Diamond, Greptile, Cursor Bugbot** — the LLM-over-diff reviewer landscape is already four vendors deep and racing (https://diamond.graphite.dev/, https://www.greptile.com/benchmarks, https://graphite.com/guides/best-ai-pull-request-reviewers-2025). None of them carry transcript provenance (memo §wardley-packet Climatic Patterns #4).
4. **OpenTelemetry GenAI semconv** is stabilizing a cross-vendor schema for agent traces (https://opentelemetry.io/docs/specs/semconv/gen-ai/). The industry agrees the artifact matters; no one has claimed the PR↔transcript binding layer that sits between the runtime and the code host.

The market window is the 12–18 months before an incumbent ships a vendor-locked binding inside their own surface (memo §wardley-packet Climatic Patterns #4; §Risks #1).

## 5. What does the customer experience look like?

**Two-line summary.** Sara installs the `claude-sql` GitHub App on her team's repo. Her next AI-authored PR auto-posts a review sheet alongside the diff showing what the agent explored, where it got corrected, and which tools it refused. She catches a rejected path that turns out to be the safer choice, asks her teammate to iterate, and the team's review rhythm changes.

Full walkthrough in `storyboard-appendix.md` — eight panels across three acts, Sara's first skim-review, the production regression, the install, the first review sheet, the catch, the post-incident retro that closes in 60 seconds, and the team publishing their binding configuration as reference for adjacent teams.

---

**Status:** COMPLETE

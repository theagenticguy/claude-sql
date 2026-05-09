# Press Release — claude-sql transcript-bound PR review

**Status:** COMPLETE — discovery artifact, not for publication. Do not run through AEM / Legal.
**Audience:** AGS Tech AI Engineering leadership + peers.

---

# AWS claude-sql ships transcript-bound PR review for AI-authored code

**Reviewers now see the prompts, tool calls, and corrections that produced the diff — not just the diff.**

*Seattle — 2026-09-15.* AWS made `claude-sql` generally available today as an open CLI and GitHub App that binds every AI-authored pull request to the agent transcript that produced it. A staff engineer opens a PR and reads a review sheet alongside the diff: what the human asked for, what the agent explored, where it got corrected, which tools it refused. The binding is a commit trailer plus `git notes`, open under Apache-2.

For AI-authored code, the diff is compiled output. The real source is the transcript — the prompt → tool-call → correction → diff path. Every agent runtime today throws it away at `git commit` time. Reviewers approve changes they cannot causally explain, incident responders reconstruct what the agent rejected over days, and new hires read object code without the `.c` (memo §Diagnosis — Root cause).

`claude-sql` closes the gap. A pre-commit hook writes `Claude-Transcript-Digest:`, `Claude-Transcript-URI:`, and `Claude-Agent-Runtime:` trailers and a `git notes --ref transcripts` entry — git primitives shipped since 1.6 (memo §The Crux). On PR open, the App resolves the trailer, compresses the bound transcript via Sonnet 4.6, and posts a `claude-sql/review-sheet` comment. The binding is agent-agnostic — Claude Code today, Cursor and Amp next, OpenTelemetry GenAI once it stabilizes (memo §Evidence row 11).

"The transcript is the source; the diff is compiled output. Binding is the crux — every downstream capability collapses without it. We published the spec so it commoditizes under us; the moat is the analytics above," said Laith Al-Saadoon, Principal AI Engineer, AWS AGS Tech AI Engineering.

"I was skim-reviewing agent PRs and finding out at retro that the story was in what the agent tried and rejected. The review sheet shows the path the agent abandoned, the tool that timed out, the correction my teammate made mid-session. My first retro with one took 60 seconds; without it, a day," said Sara Chen, staff engineer, platform team, Northwind Financial.

Install with `uv tool install claude-sql`. Enable the App at github.com/theagenticguy/claude-sql. Spec at `docs/rfc/0001-transcript-pr-binding.md`.

---

**Status:** COMPLETE

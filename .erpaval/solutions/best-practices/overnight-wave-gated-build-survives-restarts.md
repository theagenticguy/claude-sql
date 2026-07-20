# Overnight autonomous builds: gate + commit + push per wave to survive restarts

**Category:** best-practices
**Tags:** erpaval, autonomous, background-agents, git, session-restart, verify-from-disk
**Applies to:** long autonomous multi-phase builds where the orchestrator session may restart mid-run
**Date:** 2026-07-19

## Rule

For an overnight/autonomous build split into waves, make each wave land as its
own **fully-gated, committed, and pushed** unit. Never leave a wave's work only
in the working tree or only in a background agent's memory.

1. One feature = one wave = one commit that passes the full `mise run check`
   (lint/fmt/typecheck/lint:imports/test) BEFORE committing.
2. `git push` after every green wave, so progress is durable off-machine even if
   the devbox/session dies.
3. Background subagents that "self-gate then return" frequently get a false
   `stopped` / "no completion record" notification when the parent process
   restarts. Their work is still on disk. ALWAYS verify from disk
   (`git status`, `find src/...`, re-run the gate yourself) before deciding to
   resume vs re-run. Do NOT trust the agent's own green claim, and do NOT trust
   the stopped signal — verify the tree and re-run the gate in the orchestrator.

## Why

This session built claude-sql v2 across ~4 process restarts. Waves A/B/D each
survived because they were committed+pushed the moment they went green. Two
background build agents (B and D) both got false "stopped, no completion record"
notifications — but all their files were on disk and passed the gate when the
orchestrator re-ran it. A wave held only in working memory or an uncommitted
tree would have been lost or left in a half-applied broken state.

**Why:** the failure mode of long autonomous runs is not bad code, it is lost
progress + unverified "done" claims. Commit-per-green-wave converts a fragile
multi-hour run into a sequence of durable checkpoints, and orchestrator-side
re-gating catches a subagent that died mid-gate.

**How to apply:** classify waves by risk. Additive/mock-testable waves (drop,
new adapter behind a Protocol) run autonomously and commit on green. A wave
whose correctness cannot be validated by the hermetic suite (a live-auth
transport, a stateful-connection refactor flagged as able to reintroduce a
fixed bug) is deferred to a human-in-the-loop PR rather than committed blind.

## Anti-patterns

- Reporting "run complete" from a skill/task wrapper return without checking
  `git log` — the wrapper returning is not the build finishing.
- Standing by "waiting for notifications" as the terminal state of a turn when
  the next phase is deterministic and ready to run — drive it.
- Blanket-trusting a subagent's "all gates green" — re-run the gate yourself
  before committing its tree.

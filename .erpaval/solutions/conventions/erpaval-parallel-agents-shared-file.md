---
title: ERPAVal parallel agents on a shared file — partition by symbol or serialize, never both at once
category: conventions
tags: [erpaval, orchestration, parallel-agents, merge-conflict, cli-py]
modules: [src/claude_sql/cli.py, .erpaval/]
added: 2026-05-13
---

## What happened

v1.0 windowed-pipelines ran two waves of parallel ERPAVal Act agents
that each touched `src/claude_sql/cli.py`:

- **Wave 1**: Act-6 (turn_window view registration) and Act-7
  (`_rebind_vss` helper + analyze-chain wiring) both edited `cli.py`
  concurrently. Act-6 landed first; Act-7's edits were temporarily
  *reverted* (overwritten by Act-6's larger commit) before
  self-recovering on the next iteration.
- **Wave 2**: Act-2 (CLI subcommand surface) and Act-3b (analyze
  flag plumbing) both edited `cli.py` concurrently — same hazard,
  same recovery path.

Net cost: ~30 minutes of agent re-work and one false-positive "this
broke" alarm in the orchestrator log. Net benefit: parallelism still
won — the alternative (full serialization) would have cost more
wall-clock.

## Why it happened

ERPAVal's Act phase fans out subagents that each get a *task packet*
(a Markdown file describing what to edit). When two packets name the
same source file without naming distinct symbols/regions, the
agents race. The merge model is "last write wins via git", which
sometimes produces clean superset commits and sometimes produces
silent reverts (when agent A reads the file, agent B commits a
larger change, agent A then commits its smaller change against the
older base — losing B's work in the textual region A touched).

The orchestrator's `wc -l` monitor catches the *symptom* (file size
went down unexpectedly) but not the *cause*. By then the revert has
already landed.

## Fix

Two options, used in priority order:

1. **Partition by symbol.** Name the functions or line-range each
   agent owns, in the task packet. Example:

   - Act-6 owns `register_views()` and `register_macros()` in
     `cli.py`.
   - Act-7 owns `_rebind_vss()` and `analyze()` in `cli.py`.

   Agents commit only their named symbols; if a stranger symbol
   appears in the diff, the agent's PR-prep step rebases and
   recovers (`git checkout origin/main -- <stranger-symbol-file>`
   is the escape hatch — but better to have not touched it).

2. **Serialize the wave.** When the file is small (<300 LOC) or
   the symbols overlap textually (one symbol's edit moves another's
   line numbers), don't fan out — run the agents sequentially.
   The orchestrator can dispatch waves of mixed-mode (parallel
   for non-conflicting agents, serial for the cli.py-touchers).

Either way, the agent-side discipline is: **commit minimal changes,
rebase if your tree was disturbed, do NOT trust that your initial
read of the file is still the HEAD when you commit.** A `git fetch
origin && git rebase origin/<branch>` before the final commit
detects the disturbance and triggers recovery.

## How to recall

- Symptom: an orchestrator's `wc -l` monitor reports a file shrinking
  unexpectedly between agent iterations.
- Symptom: an agent's commit message describes work that's no longer
  present in the file at HEAD.
- Symptom: a follow-up "self-recovery" iteration shows up in the
  agent log a few minutes after the initial commit.
- Trigger: when dispatching N parallel Act agents, scan the task
  packets for shared `modules:` entries. Any file appearing in 2+
  packets needs explicit symbol partitioning OR a serialize directive.
- Trigger: when reviewing the orchestrator's wave plan, ask "which
  files appear in more than one packet?" — that's the conflict
  surface.
- Search keywords: `cli.py shared edit`, "agent revert", "task
  packet partition", "ERPAVal Act wave", "wc -l monitor".

## References

- `.erpaval/INDEX.md` (this index — meta-lesson about how the
  orchestrator dispatches Act agents).
- claude-sql v1.0 windowed-pipelines session, 2026-05-13: Wave 1
  Act-6/Act-7 cli.py race + Wave 2 Act-2/Act-3b cli.py race, both
  on PR #42.
- Cross-ref: `conventions/erpaval-spiral-recovery.md` — the
  individual-agent discipline of pausing, grounding, and verifying
  before editing. This lesson is the orchestrator-layer counterpart.

---
title: Lift codecov patch coverage by spawning parallel agents — one per file, with the exact uncovered line list as input
track: process
category: conventions
module: ERPAVal Compound + Validate phases
component: Codecov gate recovery via subagent fan-out
severity: low
tags: [erpaval, codecov, patch-coverage, subagents, parallel-agents, claude-sql]
applies_when:
  - "A PR fails the codecov patch gate (e.g. <90% on the diff)"
  - "Multiple files contribute to the gap, each with its own concentration of uncovered lines"
  - "You have working test infrastructure (mocks, fixtures, fake bedrock clients, fake parquet caches) the agents can imitate"
pattern: |
  Codecov patch failures don't lie about which lines are uncovered.
  Don't ask an agent "improve coverage on file X" — give it the exact
  set of uncovered line ranges. The shape that worked on PR #42:

  1. **Run ``pytest --cov=<module> --cov-report=term-missing``** for
     each module with a coverage gap. The "Missing" column is the
     ground truth — every entry there is one branch the agent needs
     to exercise.

  2. **Spawn one ``general-purpose`` Agent per file**, in parallel.
     Each prompt carries:
     - The exact list of uncovered line ranges (verbatim from
       ``--cov-report``).
     - A pointer to the existing test file that already covers
       neighboring code, so the agent reuses fixtures (`_build_corpus`,
       fake `classify_one`, `_redirect_caches`) instead of reinventing.
     - The CLAUDE.md sections that explain the production code's
       invariants (so the test fakes match real semantics).
     - A single-file write target (`tests/test_<module>_coverage.py`)
       so commits don't textually overlap and you avoid the
       parallel-agents-shared-file gotcha
       (``conventions/erpaval-parallel-agents-shared-file.md``).
     - A concrete coverage target (≥90% on the file).
     - A check command the agent runs at the end:
       ``uv run pytest tests/<module>* --cov=<module> --cov-report=term-missing -q``.

  3. **Run them in the background** (``run_in_background: true``).
     Three coverage agents took 8/13/20 minutes; serial would have
     been ~40. The orchestrator does the BaseException narrowing or
     other tightly-coupled fixes while they work.

  4. **Verify ``mise run check`` after they all return**, before
     pushing. Agent-written tests sometimes have a stale fixture
     parameter or an unused import; the lefthook pre-commit catches
     these locally. Codecov re-runs on push and the gate flips green
     on its own — don't manually rebuild a coverage report.

  Concrete numbers from PR #42:
  - `trajectory_worker.py`: 70% → 100% (25 new tests, 8 min agent)
  - `conflicts_worker.py`: 77% → 100% (15 new tests, 13 min agent)
  - `cli.py`: 85% → 92% (14 new tests, 20 min agent)
  - Aggregate patch coverage: 82.78% → 91%+; codecov gate flipped green.

  **Why this is the lazy-but-correct path**: writing 50+ targeted
  tests by hand requires re-deriving the fixture imports, the
  monkeypatch shapes, and the assertion grain for each branch.
  Agents that already saw the existing test file as a template
  produce idiomatic-looking tests on the first try. The orchestrator
  spends three Agent calls and gets parallel coverage, instead of
  spending an afternoon writing tests serially.

  **What to be skeptical of**: the agents will sometimes report
  100% coverage on lines that are actually pre-existing uncovered
  code (not in the PR diff). Codecov gates patch coverage on the
  diff only, so cross-reference the agent's report with
  ``git diff main..HEAD -- <file>`` before assuming the gate is
  green. (PR #42 cli.py agent caught this and stopped at PR-diff
  lines — left the historic uncovered remainder alone.)
example_files:
  - tests/test_trajectory_coverage.py — 25 tests, 100% on `trajectory_worker.py`
  - tests/test_conflicts_coverage.py — 15 tests, 100% on `conflicts_worker.py`
  - tests/test_cli_coverage.py — 14 tests, 92% on `cli.py` (PR-diff scope)
counter_examples:
  - "Spawning ONE general-purpose agent for ""improve coverage across the PR"" — the agent picks the easiest file, declares victory, and the gate stays red. Per-file agents force per-file work."
  - "Letting the agent design the test cases without the uncovered line list — they'll over-test the happy paths and miss the actual gaps. The line list IS the work spec."
  - "Spawning agents serially because ""I want to review each one before the next"" — the agent reviews are fixture/idiom-grain, not architecture-grain. Run them in parallel and review the diffs at the end."
  - "Hand-writing the patch-coverage tests after the agents fail: agents fail on missing context (no CLAUDE.md pointer, no existing-test-file pointer), not on capability. Reprompt with the missing context, don't fall back to manual."
references:
  - "Codecov patch coverage docs: https://docs.codecov.com/docs/commit-status#patch-status"
  - "claude-sql PR #42 (2026-05-13) — codecov/patch failed at 82.78%; three parallel general-purpose agents lifted it to 91%+ in ~20 minutes wall-clock. Each agent's prompt carried the exact uncovered line ranges from `pytest --cov-report=term-missing` and a pointer to an existing test file. Conflicts agent additionally fixed two pre-existing ERA001 lint findings in an untracked test scaffold per the ""fix problems you encounter"" tenet."
  - "Pairs with `conventions/erpaval-parallel-agents-shared-file.md` (partition-by-symbol when files DO overlap) and `conventions/test-fakes-absorb-future-kwargs.md` (the fakes the agents copy from)."
---

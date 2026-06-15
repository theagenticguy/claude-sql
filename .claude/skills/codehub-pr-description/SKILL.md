---
name: codehub-pr-description
description: "Use when the user asks for a PR description, a pull request summary, a merge write-up, or a release note for a branch or diff. Examples: \"write the PR description\", \"summarize this branch for review\", \"draft release notes for HEAD\". Calls `detect_changes` + `verdict` + `owners` + `list_findings_delta` and writes Markdown. DO NOT use for open-ended architecture docs (use `codehub-document`) or onboarding guides (use `codehub-onboarding`). DO NOT use when no diff exists — the skill refuses on a clean tree."
allowed-tools: "Read, Write, Bash(git diff:*), Bash(git log:*), Bash(git rev-parse:*), mcp__codehub__detect_changes, mcp__codehub__verdict, mcp__codehub__owners, mcp__codehub__impact, mcp__codehub__signature, mcp__codehub__list_findings_delta, mcp__codehub__api_impact"
argument-hint: "[--base <rev>] [--head <rev>] [--out <path>]"
color: teal
model: sonnet
---

# codehub-pr-description

Generates a Markdown PR body from graph primitives. Linear (no subagents). Sonnet. Refuses on a clean tree.

## Preconditions

1. Resolve `--base` (default `main`) and `--head` (default `HEAD`) via `git rev-parse`.
2. `git diff --name-only <base>..<head>` must return ≥ 1 path. If empty, emit `No diff detected — resolve base/head or stage changes.` and stop.

## Arguments

- `--base <rev>` — base revision. Default: `main`.
- `--head <rev>` — head revision. Default: `HEAD`.
- `--out <path>` — output path. Default: `.codehub/pr/PR-<branch>.md`.

## Process

1. Run the preconditions. On clean tree, refuse and stop.
2. `mcp__codehub__detect_changes({base, head})` — map the diff to affected symbols + processes.
3. `mcp__codehub__verdict({base, head})` — 5-tier merge recommendation with reasons.
4. `mcp__codehub__owners({paths: <changed-files>})` — required reviewers per path.
5. `mcp__codehub__list_findings_delta({base, head})` — new/resolved scanner findings in the diff range.
6. For any symbol flagged as tier ≥ 3 by verdict: `mcp__codehub__impact({symbol, direction: "downstream", depth: 2})` — spell out who breaks.
7. For public API changes: `mcp__codehub__api_impact({route})` when the diff touches a handler.
8. Assemble the Markdown body using the template below.
9. `Write` to `<out>`.

## Output template

```markdown
# <branch-name or commit subject>

## Summary

2–3 sentences describing what this PR changes and why. Grounded in the
commit messages + the highest-impact change detected.

## Verdict

**Tier <N> — <label>** per `mcp__codehub__verdict`.

Reasons:
- ... (from verdict.reasons[])

## Affected surface

| Category | Count | Details |
|---|---|---|
| Files changed | N | `git diff --stat` summary |
| Symbols added | N | from `detect_changes` |
| Symbols removed | N | from `detect_changes` |
| Processes touched | N | from `detect_changes.processes[]` |

### Top touched files

| File | Change | Top owner |
|---|---|---|
| `packages/foo/src/bar.ts` | +40 / -12 | alice@ |
| ... | ... | ... |

## Blast radius

(Only when verdict tier ≥ 3.)

- Downstream consumers of `<symbol>`: <count>. See `impact` output.
- Affected routes: (if `api_impact` returned non-empty)

## Findings delta

| Change | Severity | File |
|---|---|---|
| new | error | `packages/foo/src/bar.ts:42` |
| resolved | warn | `packages/other.ts:88` |

(Or: "No new findings. 2 findings resolved." as a terse summary.)

## Required reviewers

- `packages/foo/` — alice@, bob@
- `packages/other/` — charlie@

## Test plan

- [ ] ...
- [ ] ...

(Extract TODO-shaped lines from `git log <base>..<head>` or leave a blank checklist for the author to fill in.)
```

## Document format rules

- H1 = the PR title. If the branch name is descriptive, use it; otherwise fall back to the first commit subject.
- Every file citation uses backtick `path:LOC` form where line information is meaningful.
- The `Verdict` tier line is always present, even on clean-verdict passes ("Tier 1 — safe to merge").
- No YAML frontmatter on the output.
- No emojis.

## Fallback paths

- If `verdict` errors: emit "Verdict unavailable — running in degraded mode" in the Verdict section and proceed with the rest.
- If `list_findings_delta` returns `status: "no_baseline"`: use `list_findings` for the current head and note "No baseline for findings delta; showing current findings only."
- If `owners` returns `[]` for all paths: omit the Required reviewers section and record `*owners unavailable*` inline.

## Quality checklist

- [ ] Preconditions enforced — refused on clean tree.
- [ ] Verdict tier appears.
- [ ] Affected surface table present with non-empty rows.
- [ ] Top touched files table has at least one row with owner.
- [ ] Blast radius section appears iff verdict tier ≥ 3.
- [ ] Findings delta has a row for every new/resolved finding, or a "No new findings" summary.
- [ ] Test plan section exists (may be empty checklist).
- [ ] Output written to `<out>`; default is `.codehub/pr/PR-<branch>.md`.

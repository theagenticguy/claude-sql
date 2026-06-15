---
role: doc-analysis-risk-hotspots
model: sonnet
output: "{{ docs_root }}/analysis/risk-hotspots.md"
depends_on:
  - "{{ context_path }}"
  - "{{ prefetch_path }}"
status: IN_PROGRESS
---

# Packet · {{ repo }} · analysis/risk-hotspots.md

## 1. Objective

Produce `{{ docs_root }}/analysis/risk-hotspots.md`: a ranked table of the top 12 files in `{{ repo }}` by combined risk score (30-day trend slope + current finding severity), followed by a `## Per-file drill-down` section with one H3 per top-5 hotspot covering what's there, recent activity, owners, and open findings.

## 2. Scope

- Create: `{{ docs_root }}/analysis/risk-hotspots.md`
- Do not touch: `{{ docs_root }}/analysis/ownership.md`, `{{ docs_root }}/analysis/dead-code.md`, any other file under `{{ docs_root }}/`, any source file in the repo, `.context.md`, `.prefetch.md`, or any `.packets/*.md` other than this one.

## 3. Input specification

| Source | Read how | Cache state |
|---|---|---|
| Shared context | `Read {{ context_path }}` | always first |
| Prefetch ledger | `Read {{ prefetch_path }}` | always first |
| Owners summary | `{{ context_path }} § Owners summary` | cached |
| Risk trends | `mcp__codehub__risk_trends({repo: "{{ repo }}", window_days: 30})` | mid-run |
| Current findings | `mcp__codehub__list_findings({repo: "{{ repo }}", severity: ">=warn"})` | mid-run |
| Owners per hotspot | `mcp__codehub__owners({path: <file>})` or `{{ prefetch_path }} § owners` | cached when present |
| Verdict context (optional) | `mcp__codehub__verdict({repo: "{{ repo }}"})` | mid-run, optional |

## 4. Process

1. `Read {{ context_path }}` and `Read {{ prefetch_path }}`. Confirm the Owners summary and any cached findings/trend digests.
2. Call `mcp__codehub__risk_trends({repo: "{{ repo }}", window_days: 30})`. Record the per-community / per-file trend slope (`↑ rising`, `→ flat`, `↓ falling`).
3. Call `mcp__codehub__list_findings({repo: "{{ repo }}", severity: ">=warn"})`. Group findings by `file_path`; compute `N warn, M error` counts per file.
4. Combine trend slope + severity into a risk score and rank files. Pick the top 12 for the ranking table; the top 5 feed the drill-down.
5. For each top-12 file: resolve the top owner from `{{ prefetch_path }} § owners` when present, else `mcp__codehub__owners({path: <file>})`. Capture percentage share.
6. Draft the ranked table with columns `File | Trend | Open findings | Top owner | Citation`. Every row cites the file as a backtick `path` with an optional `(N LOC)` suffix; every `Open findings` cell derives from `list_findings` output only.
7. Draft the intro (2 paragraphs): what "risk" means here, how the scoring is composed, the window. If `risk_trends` returned `status: "insufficient_history"`, note the limitation in the intro and drop the Trend column.
8. Draft the `## Per-file drill-down` section. One H3 per top-5 hotspot. Each H3 covers: What's there (2-sentence summary from `context`/`Read`), Recent activity (from `risk_trends`), Owners (top 1-2 with percentage share), Findings (counts by severity, each cited to `list_findings`).
9. `Write {{ docs_root }}/analysis/risk-hotspots.md` with H1 = `{{ repo }} · Risk hotspots`.

## 5. Document format rules

- H1 = `{{ repo }} · Risk hotspots`. No decorative titles.
- No YAML frontmatter on the output file.
- Ranking table has exactly these columns: `File | Trend | Open findings | Top owner | Citation`.
- Every ranking row cites the file via a backtick `path` in the Citation column.
- Trend column uses `↑ rising` / `→ flat` / `↓ falling` arrows — do not invent other symbols.
- Drill-down uses H3s (not H2s) under the single `## Per-file drill-down` H2.
- Owner shares use whole-percent integers (e.g., `68%`), not decimals.
- No emojis. No filler adverbs.

## 6. Tool usage guide

| Need | Tool | Why |
|---|---|---|
| Risk over time | `mcp__codehub__risk_trends` | slope + severity, pre-computed |
| Open findings per file | `mcp__codehub__list_findings` | grouped by path; merges scanners |
| Contributor share per path | `mcp__codehub__owners` | authoritative over git-blame |
| Hotspot context / summary | `mcp__codehub__context` | inbound/outbound for the 2-sentence summary |
| Severity framing (optional) | `mcp__codehub__verdict` | optional — frames the intro severity |

## 7. Fallback paths

- If `risk_trends` returns `status: "insufficient_history"`: drop the Trend column from the ranking table, rank by `list_findings` severity alone, and state the limitation in the 2-paragraph intro. Cite the fallback in the Work log.
- If `owners` fails for a path: fall back to the cached owners digest in `.prefetch.md` if present; otherwise do a manual git-log walk and mark the row `*git-log fallback*` in the Top owner cell.
- If `list_findings` returns `[]` for the repo: keep the ranking table but drop the Open findings column, rank by trend slope alone, and state the absence in the intro. Cite the fallback in the Work log.
- If fewer than 10 files qualify after ranking: lower the threshold to include the full eligible set, note the shortfall in the intro, and still satisfy the ≥ 10-row success criterion by including lower-risk rows.

## 8. Success criteria

- [ ] `{{ docs_root }}/analysis/risk-hotspots.md` exists on disk.
- [ ] H1 line reads `# {{ repo }} · Risk hotspots`.
- [ ] The ranking table has at least 10 rows.
- [ ] Every ranking row has a backtick path citation in the Citation column.
- [ ] A single `## Per-file drill-down` H2 exists, containing 5 H3s.
- [ ] Every H3 under drill-down cites at least 2 `path:LOC` references (What's there + Findings).
- [ ] Every `Open findings` cell count matches a `list_findings` grouping (spot-check 3).
- [ ] No YAML frontmatter on the output.

## 9. Anti-goals

- Do not re-call any tool whose digest is already cached in `.prefetch.md` (owners, findings grouping, process list). Read the cached slice first.
- Do not invent finding counts, owner names, or severity numbers — every number must map to a `list_findings` row or an `owners` response.
- Do not reorder the ranking columns; the downstream README expects the fixed schema.
- Do not write YAML frontmatter on the output file.
- Do not emit emojis.

---

## Work log

{{ subagent fills this section per the write protocol }}

## Validation

{{ checks run, outputs pasted, any fixes applied }}

## Summary

{{ one paragraph — what shipped, any fallbacks invoked, and which files made the top-5 drill-down }}

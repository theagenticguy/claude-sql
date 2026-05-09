# ADR 0017 — Claude Code tool taxonomy transition (v2.1.16 / v2.1.63)

**Status:** accepted — 2026-05-09
**Branch:** `feat/tool-taxonomy-2026`

## Context

`claude-sql` ingests Claude Code transcripts (`~/.claude/projects/**/*.jsonl`)
and projects them through DuckDB views. The two views that key off
specific tool names — `todo_events` (filter `TodoWrite`) and `task_spawns`
(filter `Task` / `Agent` / `TaskCreate` / `mcp__tasks__task_create`) —
predate two breaking shifts in the upstream Claude Code tool taxonomy:

1. **v2.1.63 — `Task` → `Agent` rename.** The subagent launcher tool was
   renamed from `Task` to `Agent`. Both names coexist in this user's
   corpus: pre-v2.1.63 transcripts still emit `Task`; v2.1.63+ transcripts
   emit `Agent`. The Agent SDK's subagents docs explicitly carry the
   note "the tool name was renamed from 'Task' to 'Agent' in Claude Code
   v2.1.63". Source: <https://code.claude.com/docs/en/sub-agents>.

2. **v2.1.16 (Jan 2026) — `TodoWrite` split for interactive sessions.**
   The interactive todo tracker was split into a four-tool family:
   `TaskCreate` (replaces `TodoWrite` snapshots with persistent rows),
   `TaskGet`, `TaskList`, `TaskUpdate` (in-place status / blocker
   mutation). `TodoWrite` was **not** removed — it remains the surface
   for `--print` mode and the Agent SDK. Interactive sessions starting
   v2.1.16 emit `TaskCreate` / `TaskUpdate` instead.
   Source: <https://code.claude.com/docs/en/tools-reference>.

3. **MCP mirror.** SDK-py runs route through MCP and emit
   `mcp__tasks__task_create` / `mcp__tasks__task_update` with the same
   semantic shape as the native interactive `TaskCreate` / `TaskUpdate`
   pair. This user's corpus shows both shapes side-by-side.

The pre-2026 `task_spawns` view UNION-ed all four names (`Task`, `Agent`,
`TaskCreate`, `mcp__tasks__task_create`) into one row shape and projected
`subagent_type` / `prompt`. That conflation is semantically broken: the
launcher input shape is `{subagent_type, description, prompt,
run_in_background}`, while the task-tracker input shape is `{subject,
description, activeForm, metadata?}`. Across this user's recent corpus,
164 of 200 sampled `TaskCreate` rows match the latter shape — meaning
every `TaskCreate` row in `task_spawns` had `subagent_type` and `prompt`
NULL, and the `description` column carried task-tracker semantics rather
than subagent-brief semantics. There was also no view at all for
`TaskUpdate`, so post-Jan-2026 interactive todo lifecycles were
invisible to claude-sql.

## Decision

Split the views by **semantic**, not by **tool name**:

| View | Filters on | Projects |
|---|---|---|
| `subagent_spawns` | `Task`, `Agent` | `subagent_type`, `description`, `prompt`, `run_in_background` |
| `task_creations` | `TaskCreate`, `mcp__tasks__task_create` | `subject`, `description`, `active_form`, `metadata` |
| `task_updates` | `TaskUpdate`, `mcp__tasks__task_update` | `task_id`, `status`, `add_blocked_by`, `owner` |
| `tasks_state_current` | `task_creations` ⟕ latest `task_updates` | `task_id`, `subject`, `status`, `last_updated_at` |
| `task_spawns` *(deprecated)* | UNION ALL alias over the two creation views | `spawn_tool`, `subagent_type`, `description`, `prompt` (NULL for TaskCreate rows) |

`todo_events` and `todo_state_current` are unchanged — they remain
`TodoWrite`-only. Pre-2026 transcripts and any `--print` / Agent-SDK runs
still flow through them.

`tasks_state_current` mirrors `todo_state_current` but for the v2.1.16+
family. The `task_id` is assigned by the runtime and not present on
`TaskCreate.input` — it appears in the matching `tool_result` text
(`"Task #N created..."`) or as a JSON `{taskId}`. The view recovers it
via `regexp_extract` first, then `json_extract_string`, falling back to
per-session creation order so the view is robust to any third shape.

## Decisions worth calling out

- **Split, don't rename.** Keeping `task_spawns` as a deprecated alias
  for one minor release lets external dashboards / notebooks keep
  rendering without a hard break. The alias UNIONs the two new views
  with NULL-padding so the column shape is unchanged. Removing the
  alias is queued for the next minor.

- **`taskId` (camel) vs `id` (mcp).** Native `TaskUpdate.input.taskId`
  is camelCase; `mcp__tasks__task_update.input.id` is `id`. The view
  COALESCEs both into `task_id`. Don't normalize at ingestion — keep
  the raw column names visible in the source view so downstream
  consumers can still pivot per-tool if needed.

- **No classifier prompt changes.** The four Sonnet 4.6 system prompts
  in `llm_worker.py` reference "tool calls" / "tool results" generically
  — none of them name specific tools. Same for the friction classifier.
  No prompt edit was needed.

- **Hook side-effects (informational).** The new `Task*` family bypassed
  `PreToolUse`/`PostToolUse` hooks in v2.1.16; the bug
  ([anthropics/claude-code#20243](https://github.com/anthropics/claude-code/issues/20243))
  was closed 2026-01-23 and fixed in v2.1.19. SessionStart
  `additionalContext` and Stop hook outputs ride as `<system-reminder>`
  text blocks on real user/assistant messages — they aren't a separate
  message role. `claude-sql` doesn't currently filter these out of
  embeddings; that's a separate (non-blocking) follow-up.

## Consequences

- Pre-2026 transcripts: unchanged — `Task` rows route through
  `subagent_spawns` (the rename made `Task` and `Agent` semantically
  identical, both go in the same view).
- Post-Jan-2026 interactive transcripts: `tasks_state_current` is the
  primary lifecycle view; `task_updates` exposes the per-event grain.
- SDK-py runs: same shape via the MCP mirrors.
- Existing analytics that read `task_spawns`: still work for one
  release, but `subagent_type`/`prompt` are NULL for `TaskCreate`
  rows. Migrate to the two underlying views before the next minor.

## References

- <https://code.claude.com/docs/en/tools-reference>
- <https://code.claude.com/docs/en/sub-agents>
- <https://code.claude.com/docs/en/agent-sdk/subagents>
- <https://code.claude.com/docs/en/hooks.md>
- [anthropics/claude-code#20243](https://github.com/anthropics/claude-code/issues/20243)
  — Task* hook bypass, fixed v2.1.19

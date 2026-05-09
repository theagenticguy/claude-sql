---
title: Claude Code tool taxonomy migrated in 2026 — split `Task`/`Agent` (launcher) from `TaskCreate`/`TaskUpdate` (tracker), keep `TodoWrite` for `--print`/SDK
track: knowledge
category: api-patterns
module: src/claude_sql/sql_views.py
component: Claude Code CLI / Claude Agent SDK
severity: warning
tags: [claude-code, claude-agent-sdk, tool-taxonomy, transcript-analytics, duckdb-views, todowrite, taskcreate, agent-tool, mcp-tools]
applies_when:
  - "Building a transcript-analytics tool that filters DuckDB views on `tool_name = '<X>'`"
  - "Writing a classifier that quotes specific Claude Code tool names in its prompt"
  - "Documenting Claude Code tool naming in cookbooks, ADRs, or schemas"
  - "Migrating analytics over a corpus that spans pre-2026 and 2026+ transcripts"
pattern: |
  Two breaking renames in the upstream tool taxonomy (verified against
  https://code.claude.com/docs/en/tools-reference + this user's corpus 2026-05-09):

  1. **v2.1.63 — `Task` → `Agent`** (subagent launcher).
     - Same input shape: `{subagent_type, description, prompt, run_in_background?}`.
     - Both names appear in any user's corpus that spans the rename — `Task`
       in pre-v2.1.63 transcripts, `Agent` in v2.1.63+. Filter on `IN ('Task', 'Agent')`.

  2. **v2.1.16 (Jan 2026) — `TodoWrite` SPLIT for interactive sessions** into
     `TaskCreate` / `TaskGet` / `TaskList` / `TaskUpdate`.
     - `TodoWrite` is **not deleted** — it remains the surface for `--print`
       mode and the Agent SDK (Python + TypeScript). Don't drop the
       `TodoWrite` filter.
     - Interactive sessions starting v2.1.16 emit `TaskCreate` (creation,
       shape `{subject, description, activeForm, metadata?}`) and
       `TaskUpdate` (mutation, shape `{taskId, status, addBlockedBy?, owner?}`).
     - SDK-py runs route through MCP and emit
       `mcp__tasks__task_create` / `mcp__tasks__task_update` with the same
       semantics. Note: native uses `taskId` (camelCase); mcp variant uses
       `id`. COALESCE both into one column.

  **The anti-pattern:** UNIONing the launcher (`Task`/`Agent`) with the
  tracker (`TaskCreate`) into one view that projects `subagent_type` /
  `prompt`. The two input shapes diverge — `TaskCreate` rows end up with
  NULL `subagent_type` / NULL `prompt`, and the `description` column
  carries different semantics (task title vs subagent brief). Across this
  user's recent corpus, 164 of 200 sampled `TaskCreate` rows had the
  tracker shape, not the launcher shape. The "old `task_spawns`" view was
  semantically broken for every TaskCreate row.

  **The pattern:** split by **semantic**, not by **tool name**:
    - `subagent_spawns` filters on `('Task', 'Agent')` — projects
      `subagent_type`, `description`, `prompt`, `run_in_background`.
    - `task_creations` filters on `('TaskCreate', 'mcp__tasks__task_create')` —
      projects `subject`, `description`, `active_form`, `metadata`.
    - `task_updates` filters on `('TaskUpdate', 'mcp__tasks__task_update')` —
      projects `task_id` (COALESCE of `taskId` / `id`), `status`, `add_blocked_by`.
    - `tasks_state_current` joins creations to latest updates per
      `(session_id, task_id)`. The runtime-assigned `task_id` isn't on
      `TaskCreate.input` — recover it from the matching `tool_result`
      via `regexp_extract('Task #(\d+) created', 1)` first, then
      `json_extract_string($.taskId)`, then per-session creation order
      as a final fallback.
    - Keep the legacy `task_spawns` view one release as a deprecated
      UNION ALL alias so external dashboards keep rendering.

  **Hook side-effects to watch for** (from
  https://code.claude.com/docs/en/hooks.md):
    - `Task*` PreToolUse/PostToolUse bypass was filed as
      anthropics/claude-code#20243 and **fixed in v2.1.19**
      (closed 2026-01-23). Hook-pipeline observers can rely on
      Task* tool calls firing both events.
    - `SessionStart` `additionalContext` is rendered as a
      `<system-reminder>` text block on the first user message — NOT
      a `system`-role message. Same envelope as CLAUDE.md, MCP server
      instructions, and skill listings.
    - `Stop` hook `decision: "block"` injects another `<system-reminder>`
      block that continues the assistant turn.
    - `PreToolUse` deny surfaces as a `tool_result` with
      `is_error: true`. Don't count these as model errors when
      classifying success.
example_files:
  - src/claude_sql/sql_views.py            # canonical view split (subagent_spawns / task_creations / task_updates / tasks_state_current)
  - tests/test_sql_views.py                # fixture exercises Agent + TaskCreate + TaskUpdate together
  - docs/adr/0017-claude-code-tool-taxonomy-transition.md
counter_examples:
  - "Filtering tool_calls on `IN ('Task', 'Agent', 'TaskCreate', 'mcp__tasks__task_create')` and projecting `subagent_type, prompt` — TaskCreate rows lose all data."
  - "Dropping the `TodoWrite` filter because v2.1.16 added TaskCreate — TodoWrite still emits in `--print` mode and every Agent SDK run."
  - "Treating `TaskUpdate.input.id` as missing on the mcp variant — it's named `id` there, not `taskId`."
references:
  - https://code.claude.com/docs/en/tools-reference
  - https://code.claude.com/docs/en/sub-agents
  - https://code.claude.com/docs/en/agent-sdk/subagents
  - https://code.claude.com/docs/en/hooks.md
  - https://github.com/anthropics/claude-code/issues/20243
  - "ADR 0017 — docs/adr/0017-claude-code-tool-taxonomy-transition.md"
---

-- jsonl_schema_v1.sql
-- Snapshot of the v1 views registered by `claude-sql` over
-- ~/.claude/projects/**/*.jsonl.
--
-- See docs/research_notes.md for why these views are shaped the way they are,
-- and docs/cookbook.md for usable recipes built on top of them.
--
-- File extension is .sql for editor syntax highlighting, but the payload is
-- DESCRIBE output (column_name + column_type), not executable DDL. The real
-- DDL lives in src/claude_sql/sql_views.py (register_views). The v2
-- analytics views (session_classifications, message_trajectory,
-- session_conflicts, message_clusters, cluster_terms, session_communities,
-- user_friction) are parquet-backed; see docs/analytics_cookbook.md.
--
-- Regenerate this file with: `claude-sql schema`.

-- =====================================================================
-- sessions (8 cols)
--   One row per top-level session transcript, rolled up from v_raw_events
--   grouped by session_id_file (the UUID extracted from the .jsonl name).
--   Source: ~/.claude/projects/*/*.jsonl (NOT **; ** would pull subagent
--   side-files in under subagents/ and break the GROUP BY).
-- =====================================================================
CREATE VIEW sessions (
    session_id           VARCHAR,
    cwd                  VARCHAR,
    git_branch           VARCHAR,
    started_at           TIMESTAMP,
    ended_at             TIMESTAMP,
    assistant_messages   BIGINT,
    record_count         BIGINT,
    transcript_path      VARCHAR
);

-- =====================================================================
-- messages (15 cols)
--   User + assistant events with usage counters. Nested message.content is
--   kept as JSON (content_json) and flattened at query time via content_blocks
--   — resilient to new block types (text / tool_use / tool_result / thinking).
-- =====================================================================
CREATE VIEW messages (
    uuid                 UUID,
    parent_uuid          UUID,
    session_id           UUID,
    ts                   TIMESTAMP,
    type                 VARCHAR,
    is_sidechain         BOOLEAN,
    role                 VARCHAR,
    model                VARCHAR,
    stop_reason          VARCHAR,
    input_tokens         BIGINT,
    output_tokens        BIGINT,
    cache_read           BIGINT,
    cache_write          BIGINT,
    content_json         JSON,
    source_file          VARCHAR
);

-- =====================================================================
-- content_blocks (12 cols)
--   One row per element in message.content[]. Produced by UNNEST over
--   json_extract(content_json, '$[*]'); switch on block_type to get
--   text / tool_use / tool_result / thinking columns.
-- =====================================================================
CREATE VIEW content_blocks (
    session_id           UUID,
    message_uuid         UUID,
    ts                   TIMESTAMP,
    role                 VARCHAR,
    block_type           VARCHAR,
    text                 VARCHAR,
    tool_use_id_field    VARCHAR,     -- id of the tool_use block itself
    tool_name            VARCHAR,
    tool_input           JSON,
    tool_use_id          VARCHAR,     -- id the tool_result references back to
    tool_result_content  JSON,
    thinking             VARCHAR
);

-- =====================================================================
-- messages_text (5 cols)
--   Flat projection of text blocks only — the substrate semantic search
--   embeds. Filters out empty / NULL text blocks.
-- =====================================================================
CREATE VIEW messages_text (
    uuid                 UUID,
    session_id           UUID,
    ts                   TIMESTAMP,
    role                 VARCHAR,
    text_content         VARCHAR
);

-- =====================================================================
-- tool_calls (6 cols)
--   content_blocks WHERE block_type='tool_use'. One row per tool invocation.
-- =====================================================================
CREATE VIEW tool_calls (
    message_uuid         UUID,
    session_id           UUID,
    ts                   TIMESTAMP,
    tool_name            VARCHAR,
    tool_use_id          VARCHAR,
    tool_input           JSON
);

-- =====================================================================
-- tool_results (5 cols)
--   content_blocks WHERE block_type='tool_result'. Join to tool_calls via
--   tool_use_id to reconstruct request/response pairs.
-- =====================================================================
CREATE VIEW tool_results (
    message_uuid         UUID,
    session_id           UUID,
    ts                   TIMESTAMP,
    tool_use_id          VARCHAR,
    content              JSON
);

-- =====================================================================
-- todo_events (7 cols)
--   One row per todo entry per TodoWrite snapshot. snapshot_ix is a per-
--   session sequence that todo_state_current uses to pick the latest state.
-- =====================================================================
CREATE VIEW todo_events (
    session_id           UUID,
    written_at           TIMESTAMP,
    message_uuid         UUID,
    subject              VARCHAR,
    status               VARCHAR,
    active_form          VARCHAR,
    snapshot_ix          BIGINT
);

-- =====================================================================
-- todo_state_current (5 cols)
--   Latest status per (session_id, subject). Window function picks
--   snapshot_ix DESC and keeps rn=1.
-- =====================================================================
CREATE VIEW todo_state_current (
    session_id           UUID,
    subject              VARCHAR,
    status               VARCHAR,
    active_form          VARCHAR,
    written_at           TIMESTAMP
);

-- =====================================================================
-- task_spawns (8 cols)
--   tool_calls filtered to Task / Agent / TaskCreate / mcp__tasks__task_create
--   — the launch site for subagents and managed tasks. Join to
--   subagent_sessions by matching parent_session_id + timing.
-- =====================================================================
CREATE VIEW task_spawns (
    session_id           UUID,
    spawned_at           TIMESTAMP,
    message_uuid         UUID,
    tool_use_id          VARCHAR,
    spawn_tool           VARCHAR,
    subagent_type        VARCHAR,
    description          VARCHAR,
    prompt               VARCHAR
);

-- =====================================================================
-- subagent_sessions (8 cols)
--   One row per subagent run. Rolls up v_raw_subagents (from
--   projects/*/*/subagents/agent-*.jsonl) joined to v_raw_subagent_meta
--   (sibling agent-*.meta.json files) for agent_type + description.
-- =====================================================================
CREATE VIEW subagent_sessions (
    parent_session_id    VARCHAR,
    agent_hex            VARCHAR,
    agent_type           VARCHAR,
    description          VARCHAR,
    started_at           TIMESTAMP,
    ended_at             TIMESTAMP,
    message_count        BIGINT,
    transcript_path      VARCHAR
);

-- =====================================================================
-- subagent_messages (13 cols)
--   user+assistant events from subagent transcripts. parent_session_id
--   connects to sessions.session_id; agent_hex uniquely identifies the run
--   within that parent.
-- =====================================================================
CREATE VIEW subagent_messages (
    uuid                 UUID,
    parent_uuid          UUID,
    session_id           UUID,
    parent_session_id    VARCHAR,
    agent_hex            VARCHAR,
    ts                   TIMESTAMP,
    type                 VARCHAR,
    role                 VARCHAR,
    model                VARCHAR,
    input_tokens         BIGINT,
    output_tokens        BIGINT,
    content_json         JSON,
    source_file          VARCHAR
);

-- =====================================================================
-- skill_invocations (7 cols)
--   Every Skill invocation observable in the transcripts, unioned across
--   the two shapes: the assistant's ``Skill`` tool call (source='tool')
--   and the user's <command-name>/<name></command-name> slash command
--   (source='slash_command').  skill_id stays raw so bare ('erpaval')
--   and namespaced ('personal-plugins:erpaval') live side by side.
-- =====================================================================
CREATE VIEW skill_invocations (
    session_id           UUID,
    ts                   TIMESTAMP,
    message_uuid         UUID,
    source               VARCHAR,      -- 'tool' | 'slash_command'
    skill_id             VARCHAR,
    args                 VARCHAR,
    tool_use_id          VARCHAR
);

-- =====================================================================
-- skills_catalog (9 cols) -- parquet-backed, populated by
--   ``claude-sql skills sync``.  Walks ~/.claude/skills/ and
--   ~/.claude/plugins/cache/** for SKILL.md + commands/*.md files.
-- =====================================================================
CREATE VIEW skills_catalog (
    skill_id             VARCHAR,      -- bare ('erpaval') or 'plugin:name'
    name                 VARCHAR,      -- bare name, never namespaced
    plugin               VARCHAR,      -- owning plugin or NULL for user skills
    plugin_version       VARCHAR,
    source_kind          VARCHAR,      -- user-skill | plugin-skill | plugin-command | builtin
    description          VARCHAR,
    argument_hint        VARCHAR,
    source_path          VARCHAR,      -- absolute path, for audit
    synced_at            TIMESTAMP
);

-- =====================================================================
-- skill_usage (13 cols) -- LEFT JOIN of skill_invocations against
--   skills_catalog on skill_id.  Falls back to skill_id pass-through
--   when the catalog parquet is missing.
-- =====================================================================
CREATE VIEW skill_usage (
    session_id           UUID,
    ts                   TIMESTAMP,
    message_uuid         UUID,
    source               VARCHAR,
    skill_id             VARCHAR,
    args                 VARCHAR,
    tool_use_id          VARCHAR,
    skill_name           VARCHAR,
    plugin               VARCHAR,
    plugin_version       VARCHAR,
    description          VARCHAR,
    source_kind          VARCHAR,
    is_builtin           BOOLEAN
);

-- =====================================================================
-- Macros (claude-sql + DuckDB built-ins)
--   model_used(sid)             -> VARCHAR    latest model used in a session
--   cost_estimate(sid)          -> DOUBLE     USD via config.DEFAULT_PRICING
--   tool_rank(last_n_days)      -> TABLE      tool_name, n (call count)
--   todo_velocity(sid)          -> DOUBLE     completed / distinct todos
--   subagent_fanout(sid)        -> BIGINT     count of subagent runs for session
--   semantic_search(qv, k)      -> TABLE      uuid, sim, distance via HNSW
--   skill_rank(last_n_days)     -> TABLE      skill leaderboard (tool + slash counts)
--   skill_source_mix(last_n_days) -> TABLE    per skill: n_tool vs n_slash
--   unused_skills(last_n_days)  -> TABLE      catalog entries never invoked in window
-- =====================================================================

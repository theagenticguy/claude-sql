-- thread_walk.sql
-- Reconstruct conversation trees by walking parent_uuid -> uuid on the
-- `messages` view.
--
-- Anchor: messages with no parent (parent_uuid IS NULL) are thread roots.
-- Recursion: join each newly-added row back to `messages` on
--   messages.parent_uuid = thread.descendant_uuid to pull in direct children.
-- Output columns: (session_id, thread_root_uuid, descendant_uuid, depth).
--
-- Usage:
--   uv run claude-sql query "$(cat docs/queries/thread_walk.sql)"
--
-- Notes:
-- * `messages` is view-backed by read_json over ~/.claude/projects/*/*.jsonl
--   so each recursion step re-scans the corpus. For per-session walks, wrap
--   the base case with `WHERE session_id = ?` and DuckDB will push the
--   predicate into the JSON scan.
-- * A single message can only have one parent_uuid, so this walk produces a
--   forest, not a DAG — no DISTINCT needed.

WITH RECURSIVE thread AS (
    -- Anchor: roots of every conversation tree.
    SELECT
        uuid       AS thread_root_uuid,
        uuid       AS descendant_uuid,
        session_id,
        0          AS depth
    FROM messages
    WHERE parent_uuid IS NULL

    UNION ALL

    -- Step: attach direct children to the in-flight thread.
    SELECT
        t.thread_root_uuid,
        m.uuid       AS descendant_uuid,
        m.session_id,
        t.depth + 1  AS depth
    FROM thread t
    JOIN messages m
      ON m.parent_uuid = t.descendant_uuid
)
SELECT
    session_id,
    thread_root_uuid,
    descendant_uuid,
    depth
FROM thread
ORDER BY session_id, thread_root_uuid, depth;

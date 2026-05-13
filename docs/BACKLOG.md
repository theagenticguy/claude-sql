# claude-sql backlog

Forward-looking ideas not yet promoted to an ADR/RFC or a tracked issue.
Each entry: one-line title, dated, with enough context that a future
session can decide whether to promote it to a real spec.

## Move derived analytics caches under `~/.claude-sql/` (not `~/.claude/`)

**Filed:** 2026-05-12
**Status:** proposed; not started

### Problem

Today every derived cache lives under `~/.claude/` next to Claude Code's
own state:

```
~/.claude/
  projects/                  ← source-of-truth JSONL transcripts (Claude Code owns this)
  embeddings_lance/          ← claude-sql writes
  embeddings/                ← claude-sql writes (legacy shards)
  message_trajectory/        ← claude-sql writes
  session_classifications/   ← claude-sql writes
  session_conflicts/         ← claude-sql writes
  user_friction/             ← claude-sql writes
  clusters.parquet           ← claude-sql writes
  cluster_terms.parquet      ← claude-sql writes
  session_communities.parquet← claude-sql writes
  community_profile.parquet  ← claude-sql writes
  hnsw.duckdb                ← claude-sql writes (legacy)
  claude_sql.duckdb          ← claude-sql writes
  state.db                   ← claude-sql writes (sqlite WAL checkpointer)
  duckdb_tmp/                ← claude-sql writes (spill dir)
  profiling/                 ← claude-sql writes
  …
```

Mixing third-party caches into `~/.claude/` has three real costs:

1. **Hard rule conflict.** The user-wide policy is "never delete raw
   `~/.claude/projects/` JSONLs." Today the cleanup blast radius for
   claude-sql output is ambiguously close to that path — one
   `rm -rf ~/.claude/embeddings*` typo and we're staring at a recovery
   plan. A dedicated parent dir (`~/.claude-sql/`) lets cleanup ops
   target a clearly-not-projects path.
2. **Pollution.** Anyone (`claude config`, third-party tools, support
   bundles) inspecting `~/.claude/` sees claude-sql's working set
   mixed with Claude Code's own state. Hard to tell what is yours vs.
   what's the agent's.
3. **Backup / rsync surfaces.** Users who rsync `~/.claude/` to keep
   their transcripts safe end up shipping multi-GB of regenerable
   caches. A separate root makes "back up the source, ignore the
   derivations" a one-line decision.

### Proposed shape

```
~/.claude/                   ← Claude Code owns; we read, never write
  projects/**.jsonl          ← source

~/.claude-sql/               ← claude-sql owns; we read+write freely
  embeddings_lance/
  message_trajectory/
  session_classifications/
  session_conflicts/
  user_friction/
  clusters.parquet (+ .embeddings_mtime)
  cluster_terms.parquet
  session_communities.parquet (+ community_profile.parquet)
  state.db                   ← sqlite WAL checkpointer
  duckdb_tmp/                ← spill
  profiling/
  backups/                   ← (move-not-delete safety net)
```

### Migration

- Add new env knob `CLAUDE_SQL_HOME` (default `~/.claude-sql/`).
- Default every existing path setting (`*_PARQUET_PATH`, `LANCE_URI`,
  `DUCKDB_TEMP_DIR`, checkpointer DB) to resolve under `CLAUDE_SQL_HOME`
  if the user hasn't pinned it.
- One-time auto-migration on first connect: if `~/.claude-sql/` is
  empty AND we find recognized caches under `~/.claude/`, log a one-line
  notice and `mv` the set across. Idempotent; no-op on subsequent runs.
- Keep the legacy paths supported via the same per-setting overrides
  that already exist, so a user with custom `CLAUDE_SQL_*_PATH` env vars
  is not affected.
- Update `list-cache` + `cache compact` + `cache migrate` to walk the
  new root by default.

### Open questions

- Should `~/.claude-sql/projects/` be a symlink-or-glob to
  `~/.claude/projects/`? (No — keeps the read-side glob explicit.)
- Should the LanceDB store use a sub-tier (`embeddings/lance/`) so a
  future "secondary" store can coexist? (Probably yes; cheap.)
- Does `XDG_DATA_HOME` deserve respect? (Yes —
  `${XDG_DATA_HOME:-~/.local/share}/claude-sql/` on Linux,
  `~/Library/Application Support/claude-sql/` on macOS, with
  `~/.claude-sql/` as the legacy/override.)

### Why "backlog" not "ADR"

The decision hinges on whether the migration is worth a minor version
bump's worth of churn. ADR makes sense once we want to commit; today
this is a reminder that the boundary is wrong, not a finished
proposal.
